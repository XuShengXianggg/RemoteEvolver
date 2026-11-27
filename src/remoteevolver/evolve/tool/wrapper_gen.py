"""
remoteevolver.evolve.tool.wrapper_gen —— LLM 读仓库元数据 → 生成 wrapper.py 代码

职责:
  - 收集仓库的"够 LLM 看懂"的信号(README 摘要 + 包名 + 顶层导出)
  - 给 LLM 一个强约束的 prompt,产出一个**单文件、单工具函数**的 wrapper
  - **不**负责保存到 remoteevolver/tools/(那是 registrar.py 的事)
  - **不**负责执行/测试(那是 test_runner.py 的事)

输出格式约束(避免 LLM 自由发挥):
  - 必须是合法 Python 文件
  - 必须 `from ..mcpx import mcp` + 一个 `@mcp.tool()` 装饰的函数
  - 函数有完整类型注解 + docstring(FastMCP 据此生成 schema)
  - 返回 str(LLM 拿到的就是字符串,不需要 JSON)
  - 不要捏造仓库里不存在的函数 —— 拿不准就用 README 里出现过的 API
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ...llm.base import BaseLLM, Message
from .code_audit import PackageAPI, scan_package_api, write_code_report


# README 给 LLM 看的截断长度(过长会塞爆 token + 噪声多)
_README_MAX_CHARS = 4000

_UNSUPPORTED_PREFIX = "__EVOAGENT_REPOSITORY_UNSUPPORTED__"
_UNSUPPORTED_PATTERN = re.compile(
    rf"^{re.escape(_UNSUPPORTED_PREFIX)}\s*:\s*(\S[^\r\n]*)$"
)


class RepositoryCapabilityUnsupported(RuntimeError):
    """The selected repository cannot supply the requested core capability."""

    def __init__(self, reason: str) -> None:
        self.reason = reason.strip()
        super().__init__(self.reason)


_UNSUPPORTED_PROTOCOL = f"""\
【仓库能力不足时的唯一拒绝协议】
在写代码前，先根据公开 API 清单和 README 判断仓库是否能提供目标能力的核心领域操作。
- 参数名、标量/列表、字符串、路径和返回格式差异由 wrapper 胶水负责，不得因此拒绝。
- 可以用少量胶水组合仓库中有文档依据的 API，也不得因此拒绝。
- 文件遍历、参数解析、路径规范化和结果序列化只能作为适配胶水，不能替代仓库缺失的核心领域能力。
- 如果核心能力只能靠你用 Python 标准库、手写算法、文件名猜测或数据集专用规则重新实现，说明该仓库不合适。此时不要输出 Python，不要勉强包装，也不要解释；只输出一行：
  {_UNSUPPORTED_PREFIX}: <简短说明仓库缺少哪个必要 API 或能力的证据>
"""


@dataclass
class RepoSignals:
    """汇总给 LLM 看的"我是一个什么样的仓库"信号。

    除了 README 这种"人话文档",还有 code_api 这种"AST 扫出来的真实函数签名"
    (C-1 Code Report)。LLM 拿到 code_api 后,wrapper 里调的 API **可以真去
    校验存在性**(C-2 用),且能基本杜绝"编一个不存在的函数名"。
    """
    full_name: str           # owner/name
    package_name: str        # pip install 后 import 的包名;不确定时用 slug
    readme_excerpt: str      # README 前 ~4000 字符
    pyproject_present: bool
    setup_py_present: bool
    # C-1 新增:AST 扫描出来的真实公开 API。可能 found=False(包根没定位到)。
    code_api: PackageAPI | None = None


# ---- 收集仓库信号 ------------------------------------------------------

def collect_repo_signals(
    repo_dir: Path,
    full_name: str,
    *,
    scan_code: bool = True,
    write_report_to: Path | None = None,
) -> RepoSignals:
    """从 clone 下来的目录里抽 LLM 写 wrapper 需要的信息。

    Args:
        repo_dir:        仓库根目录(已 clone)。
        full_name:       owner/repo,只用来填回信号。
        scan_code:       True 时执行 C-1 AST 扫描公开 API,挂到 RepoSignals.code_api。
                         单测 / 不需要 wrapper 的场合可传 False 省时。
        write_report_to: 给一个目录就把 code_report.json 写到这里(一般是
                         `data/external_tools/<slug>/`,刚好是 repo_dir 的父目录)。
    """
    readme = _find_readme(repo_dir)
    pkg = _guess_package_name(repo_dir, full_name)

    code_api: PackageAPI | None = None
    if scan_code:
        code_api = scan_package_api(pkg, repo_dir)
        if write_report_to is not None:
            try:
                write_code_report(code_api, write_report_to)
            except Exception:  # noqa: BLE001
                # 报告写盘失败不阻塞主流程,wrapper 仍能拿到 code_api in-memory
                pass

    return RepoSignals(
        full_name=full_name,
        package_name=pkg,
        readme_excerpt=readme[:_README_MAX_CHARS] if readme else "(no README found)",
        pyproject_present=(repo_dir / "pyproject.toml").exists(),
        setup_py_present=(repo_dir / "setup.py").exists(),
        code_api=code_api,
    )


def _find_readme(repo_dir: Path) -> str:
    for name in ("README.md", "README.rst", "README.txt", "readme.md", "README"):
        p = repo_dir / name
        if p.exists() and p.is_file():
            try:
                return p.read_text(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                continue
    return ""


def _guess_package_name(repo_dir: Path, full_name: str) -> str:
    """猜 pip 安装后的 import 名。pyproject.toml > setup.py > slug。"""
    # pyproject.toml: [project] name = "xxx"
    pp = repo_dir / "pyproject.toml"
    if pp.exists():
        try:
            text = pp.read_text(encoding="utf-8", errors="replace")
            m = re.search(r'name\s*=\s*"([^"]+)"', text)
            if m:
                # PyPI 名常带 - 或 _,import 时 - 都要换成 _
                return m.group(1).strip().replace("-", "_")
        except Exception:  # noqa: BLE001
            pass
    # setup.py: setup(name="xxx", ...)
    sp = repo_dir / "setup.py"
    if sp.exists():
        try:
            text = sp.read_text(encoding="utf-8", errors="replace")
            m = re.search(r'name\s*=\s*["\']([^"\']+)["\']', text)
            if m:
                return m.group(1).strip().replace("-", "_")
        except Exception:  # noqa: BLE001
            pass
    # 兜底:仓库 slug
    return full_name.split("/")[-1].replace("-", "_")


# ---- 生成 wrapper 代码 -------------------------------------------------

_WRAPPER_PROMPT = """\
你正在为 remoteevolver 框架生成一个工具包装文件(wrapper)。remoteevolver 通过 MCP 协议
暴露工具给 LLM 调用,本文件必须严格遵守它的约定。

【任务】
用户希望获得能力:{capability}
我们已经 pip install 了下面这个 GitHub 仓库,请基于它的 API 写**一个**
@mcp.tool() 函数,实现上述能力。

【仓库信息】
  仓库:           {full_name}
  Python 包名:    {package_name}
  pyproject 存在:  {pyproject_present}
  setup.py 存在:  {setup_py_present}

【公开 API(AST 扫描,**这是真实存在的导出符号清单 — 你只能调用这里出现的名字**)】
{code_api_summary}

【README 节选】
```
{readme_excerpt}
```

{unsupported_protocol}

【仓库能够提供核心能力时的硬性输出格式 —— 任何违反都会被判 fail】
  1. 只输出 Python 源代码,不要任何 Markdown 围栏、不要解释、不要前缀。
  2. 文件结构必须严格按以下骨架,不能加额外顶层逻辑:

     \"\"\"<一行描述这个工具做什么>\"\"\"

     from __future__ import annotations

     import {package_name}
     from ..mcpx import mcp


     @mcp.tool()
     def <函数名>(<参数: 类型注解>, ...) -> str:
         \"\"\"<给 LLM 看的工具描述,第一行最重要>

         Args:
             <参数>: <说明>

         Returns:
             <返回值含义>
         \"\"\"
         # 业务代码:调 {package_name} 的 API 完成任务
         ...
         return "<结果字符串>"

  3. 函数必须只有一个,函数名用 snake_case,**不要带 tool 前缀**。
  4. 所有参数必须有类型注解。标量类型只用 str / int / float / bool
     (需要传多个值的参数,拼成一个 str)。**多值串接契约(关键,避免路径含空格被切碎)**:
     统一用**英文逗号 `,`** 分隔;函数内部**只能 `.split(",")` 再 `.strip()` 每项**,
     **绝对不要按空白 `.split()` 切**——单个值本身可能含空格，
     按空白切会把单个路径截断成多个、导致 FileNotFound。docstring 里注明该参数是
     "comma-separated"(逗号分隔),好让调用方按逗号拼。
     **可选参数(可以不传 / 允许为空)必须写成 `名: 类型 | None = None`,并给默认值**;
     函数内部要正确处理 None(例如 None 表示"不设该项",不要直接拿去运算)。
     **非必填参数一律给默认值**。这条很关键:若把可空参数标成裸 `float`/`int`
     而不带 `| None`/默认值,上层 LLM 传 None 时会被 MCP schema 当场拒掉(高频 bug)。
  5. 函数返回值类型必须是 str(MCP 会把 str 透传给上层 LLM)。
  6. **不要捏造 API**。优先调用上面"公开 API"清单里出现的符号。
     如果清单里没有合适的,fallback 到 README 中实际出现过的函数或类名。
     **绝对不要**写训练记忆里的"我以为这个库有"的函数。
  7. 异常处理:**用 try/except 包住核心调用**,失败时 return 一个以
     "[<工具名> error]" 开头的字符串,不要让异常冒出来 —— LLM 拿到错误
     字符串可以自己重试。
  8. 依赖纪律:**只 import 该仓库及其已随仓库安装的依赖**,不要引入仓库依赖
     之外的第三方库。写输出文件时优先使用仓库原生支持、无需额外可选依赖的参数；
     不要为优化或压缩功能临时引入新依赖。
  9. **互操作格式(关键:这些工具会被串成工具链,你的输出常是别的工具的输入)**:
     凡是**写到文件、可能被下一个工具读取**的中间产物,**必须用通用、自描述的
     标准格式**,让别的工具无需了解本 wrapper 就能读取。格式必须与能力语义、显式
     output_path 扩展名、输入元数据和仓库实际支持一致；不得为某一领域强制指定默认格式，
     也不得发明只有本 wrapper 能解码的私有表示。若任务没有要求落盘，不要擅自创建文件。
 10. **路径返回契约(关键:下游工具/agent 会直接拿你返回的路径去用)**:
     凡是返回"写出去的文件路径",**必须返回规范化的绝对路径、且用正斜杠 `/`**——
     即 `os.path.abspath(p).replace("\\\\", "/")` 之后再返回。返回多个就用逗号
     拼成一个 str。**禁止返回相对路径或带反斜杠 `\\` 的路径**:相对路径会被上层
     再拼一次目录(变成 `a/b/a/b/...` 重复路径而找不到文件),反斜杠会和正斜杠
     混用导致跨调用错乱。返回的路径必须是"下游可以一字不改直接打开"的。
 11. **结果可消费性**:返回值应直接表达本次调用得到的主要结果；若生成文件，同时返回
     可供下游直接使用的规范化路径。不要返回参考答案、任务专用常量或只能由本
     wrapper 解码的私有表示。
 12. **语义保真**:必须实现结构化契约的每项非空约束。不要自行加入领域假设、
     替代阈值、特定数据集规则或为某个评测样例设计的分支。只有契约未规定的边界情况，
     才可采用仓库文档中不冲突的通用行为，并在 docstring 中如实说明。

现在开始输出代码(只输出代码本身):
"""


def generate_wrapper(
    capability: str,
    signals: RepoSignals,
    llm: BaseLLM,
) -> str:
    """让 LLM 写一个 wrapper.py 的内容。

    返回纯代码字符串(不含 Markdown 围栏)。仓库明确不具备核心能力时抛出
    RepositoryCapabilityUnsupported；其余运行失败留给后续 test_runner。
    """
    code_api_summary = (
        signals.code_api.to_prompt_summary()
        if signals.code_api is not None
        else "(AST 扫描未运行,本次只能基于 README 推测 API)"
    )
    prompt = _WRAPPER_PROMPT.format(
        capability=capability.strip(),
        full_name=signals.full_name,
        package_name=signals.package_name,
        pyproject_present=signals.pyproject_present,
        setup_py_present=signals.setup_py_present,
        code_api_summary=code_api_summary,
        readme_excerpt=signals.readme_excerpt,
        unsupported_protocol=_UNSUPPORTED_PROTOCOL,
    )
    resp = llm.generate([Message(role="user", content=prompt)])
    return parse_wrapper_response(resp.text or "")


def parse_wrapper_response(text: str) -> str:
    """Return normalized wrapper code or raise an explicit repository rejection."""
    cleaned = _strip_markdown_fences(text).strip()
    unsupported = _UNSUPPORTED_PATTERN.fullmatch(cleaned)
    if unsupported:
        raise RepositoryCapabilityUnsupported(unsupported.group(1))
    return cleaned + "\n"


def _strip_markdown_fences(text: str) -> str:
    """LLM 偶尔不听话套个 ```python ... ```,这里强行剥掉。"""
    text = text.strip()
    # 整体被 ``` 包裹
    fence = re.match(r"^```(?:python|py)?\s*\n(.*?)\n```\s*$", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    # 单边围栏(只有开头或结尾)
    if text.startswith("```"):
        text = re.sub(r"^```(?:python|py)?\s*\n", "", text)
    if text.endswith("```"):
        text = re.sub(r"\n```\s*$", "", text)
    return text
