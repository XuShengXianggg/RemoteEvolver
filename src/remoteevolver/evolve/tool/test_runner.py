"""
remoteevolver.evolve.tool.test_runner —— 冒烟测试 LLM 生成的 wrapper

策略:
  - 把 `from ..mcpx import mcp` 替换成一个**本地 stub**(装饰器透传),让 wrapper
    脱离 remoteevolver 包结构也能跑
  - 写到临时目录,起 subprocess,导入 + 调用目标函数
  - 用 subprocess 而不是 in-process import 是为了:
      (a) 干净的 sys.modules,避免上次失败的导入污染
      (b) 子进程崩溃不会拖垮主程序
      (c) 输出的 traceback 完整可解析,直接喂给 fixer

测试参数从哪来?
  - 第一版:由调用方显式传入(pipeline / CLI 决定)
  - 后续可让 LLM 看 README 自动出一份测试 args
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


# 替换 wrapper 里 MCP 引入语句的 stub。装饰器直接透传函数,不做任何注册。
_MCP_STUB = """\
# === injected by test_runner: replaces 'from ..mcpx import mcp' for testing ===
class _StubMcp:
    @staticmethod
    def tool(*a, **kw):
        def _deco(f):
            return f
        return _deco
mcp = _StubMcp()
# === end stub ===
"""

# 子进程执行的 harness 模板。{module}/{func}/{args_json} 由 Python 字符串注入。
_HARNESS_TEMPLATE = """\
import json, sys, traceback, importlib.util

try:
    spec = importlib.util.spec_from_file_location("__evolve_wrapper__", r{wrapper_path!r})
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
except Exception:
    print("__EVOLVE_FAIL__::IMPORT_FAILED", file=sys.stderr)
    traceback.print_exc()
    sys.exit(2)

func = getattr(mod, {func!r}, None)
if func is None:
    print("__EVOLVE_FAIL__::FUNCTION_NOT_FOUND::" + {func!r}, file=sys.stderr)
    print("module dir =", dir(mod), file=sys.stderr)
    sys.exit(3)

args = json.loads({args_json!r})
try:
    result = func(**args)
except Exception:
    print("__EVOLVE_FAIL__::CALL_RAISED", file=sys.stderr)
    traceback.print_exc()
    sys.exit(4)

# 协议:成功时 stdout 第一行是 __EVOLVE_OK__,之后是返回值字符串
print("__EVOLVE_OK__")
print(repr(result))
"""


@dataclass
class TestResult:
    """冒烟测试的结果。失败时 traceback 给 fixer 看,output 是返回值字符串(成功时)。"""
    success: bool
    function_name: str
    stage: str             # "ok" | "import" | "function_not_found" | "call_raised" | "timeout"
    output: str = ""       # 成功时 = 函数返回值的 repr;失败时空
    traceback: str = ""    # 失败时 = 完整 traceback / 错误信息

    @property
    def summary(self) -> str:
        if self.success:
            return f"OK | func={self.function_name} | returned: {self.output[:200]}"
        return f"FAIL[{self.stage}] | func={self.function_name}\n{self.traceback}"


# ---- 公开 API -----------------------------------------------------------

def run_smoke_test(
    wrapper_code: str,
    args: dict,
    *,
    timeout: int = 60,
    work_dir: Path | str | None = None,
) -> TestResult:
    """对一段 wrapper 代码做冒烟测试。

    Args:
        wrapper_code: LLM 生成的完整 wrapper.py 文本(包含 mcp 引入)。
        args:         传给目标函数的参数 dict(键 = 形参名)。
        timeout:      子进程超时(秒)。
        work_dir:     临时目录;None 则用系统 tempdir。

    Returns:
        TestResult。success=True 即可注册;False 时把 traceback 喂给 fixer。
    """
    timeout = float(os.getenv('REMOTEEVOLVER_SMOKE_TIMEOUT', str(timeout)))
    func_name = _extract_function_name(wrapper_code)
    if not func_name:
        return TestResult(
            success=False, function_name="?", stage="parse",
            traceback="未在 wrapper 里找到 @mcp.tool() 装饰的函数",
        )

    # 用本地 stub 替换掉 mcp 引入(否则脱离 remoteevolver 包跑不起来)
    transformed = _inject_mcp_stub(wrapper_code)

    # 落地临时文件
    work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="evolve_test_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    wrapper_path = work_dir / "wrapper_under_test.py"
    wrapper_path.write_text(transformed, encoding="utf-8")

    harness_path = work_dir / "harness.py"
    harness_path.write_text(
        _HARNESS_TEMPLATE.format(
            wrapper_path=str(wrapper_path),
            func=func_name,
            args_json=json.dumps(args, ensure_ascii=False),
        ),
        encoding="utf-8",
    )

    # 跑 subprocess
    try:
        out = subprocess.run(
            [sys.executable, str(harness_path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return TestResult(
            success=False, function_name=func_name, stage="timeout",
            traceback=f"smoke test timed out after {timeout}s",
        )

    stdout = out.stdout or ""
    stderr = out.stderr or ""

    if out.returncode == 0 and stdout.startswith("__EVOLVE_OK__"):
        # 第二行起是 repr(result)
        ret = stdout.split("\n", 1)[1].strip() if "\n" in stdout else ""

        # 假阳性检测:LLM 偶尔会把所有异常 try/except 然后 return 错误字符串,
        # 这种"工具其实没工作"的情况外表看跟成功一样。判定:返回值以
        # "[<word> error]" 开头(我们 prompt 让 LLM 这么标错误)→ 视为失败,
        # 让 fixer 看到信息后重新写一个真正能用的版本。
        if _looks_like_error_return(ret):
            return TestResult(
                success=False, function_name=func_name, stage="returned_error_string",
                traceback=(
                    "函数没抛异常,但返回了错误字符串而不是真实结果:\n"
                    f"  return value (repr): {ret[:400]}\n"
                    "请修正 wrapper:不要用 try/except 把所有异常吞掉,要让工具真正完成任务。"
                    "如果是依赖缺失(如 'No module named PIL'),请改用不依赖该模块的 API,"
                    "或在 wrapper 文档里说明,不要静默 return error。"
                ),
            )

        return TestResult(
            success=True, function_name=func_name, stage="ok", output=ret,
        )

    # 失败路径:从 stderr 解析阶段
    stage = "unknown"
    for marker, name in [
        ("__EVOLVE_FAIL__::IMPORT_FAILED", "import"),
        ("__EVOLVE_FAIL__::FUNCTION_NOT_FOUND", "function_not_found"),
        ("__EVOLVE_FAIL__::CALL_RAISED", "call_raised"),
    ]:
        if marker in stderr:
            stage = name
            break

    return TestResult(
        success=False,
        function_name=func_name,
        stage=stage,
        traceback=stderr.strip()[-3000:],  # 保留尾巴,LLM 看 traceback 就够了
    )


# ---- 内部辅助 -----------------------------------------------------------

# 找 @mcp.tool() ... def <name>(...)  ── 跨行匹配
_FUNC_PATTERN = re.compile(
    r"@mcp\.tool\s*\(.*?\)\s*\n\s*def\s+(\w+)\s*\(",
    re.DOTALL,
)

# 匹配 repr 后形如 "'[tool_name error] xxx'" / '"[tool_name error] xxx"' 的返回值
_ERROR_RETURN_PATTERN = re.compile(r"""^['"]\[\w+\s+error\]""")


def _looks_like_error_return(ret_repr: str) -> bool:
    """ret_repr 是 repr(result) 的形态;判断它是不是个"假装成功"的错误字符串。"""
    return bool(_ERROR_RETURN_PATTERN.match(ret_repr.strip()))


def _extract_function_name(code: str) -> str:
    m = _FUNC_PATTERN.search(code)
    return m.group(1) if m else ""


def _inject_mcp_stub(code: str) -> str:
    """删掉 wrapper 里 `from ..mcpx import mcp` 行,塞入测试用 stub。

    注意:`from __future__ import ...` 必须出现在文件最前。stub 是普通语句,
    若无脑前置会把 wrapper(按仓库惯例)开头的 `from __future__ import annotations`
    挤到非顶部 → SyntaxError(每个 wrapper 必炸 attempt1)。因此先把所有
    __future__ 行抽到最前,再插 stub,最后接其余代码。
    """
    # 删除任何形式的 mcp 引入
    code = re.sub(r"^\s*from\s+\.+mcpx\s+import\s+mcp\s*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"^\s*from\s+remoteevolver\.mcpx\s+import\s+mcp\s*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"^\s*from\s+mcpx\s+import\s+mcp\s*$", "", code, flags=re.MULTILINE)

    future_lines = re.findall(
        r"^[ \t]*from\s+__future__\s+import[^\n]*$", code, flags=re.MULTILINE
    )
    if future_lines:
        code = re.sub(
            r"^[ \t]*from\s+__future__\s+import[^\n]*$\n?", "", code, flags=re.MULTILINE
        )
        future_block = "\n".join(line.strip() for line in future_lines)
        return future_block + "\n" + _MCP_STUB + "\n" + code
    return _MCP_STUB + "\n" + code
