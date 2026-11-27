"""
remoteevolver.evolve.tool.fixer —— LLM 看到测试 traceback,生成修正版 wrapper

由 pipeline.py 调用:
  cur_code -> test fail -> fixer.run(cur_code, test_result) -> new_code -> test 再来

设计要点:
  - prompt 把"原代码 + 原 capability + 仓库信号 + 失败 traceback"一并塞进去
  - 强调"只输出完整新文件代码,不要 diff"
  - 输出走跟 wrapper_gen 同款的清洗(去 markdown 围栏)
"""

from __future__ import annotations

from ...llm.base import BaseLLM, Message
from .code_audit import CallIssue, format_issues_for_llm
from .test_runner import TestResult
from .wrapper_gen import (
    RepoSignals,
    _UNSUPPORTED_PROTOCOL,
    parse_wrapper_response,
)


_FIX_PROMPT = """\
你刚才写了一个 remoteevolver 工具 wrapper,**冒烟测试失败了**。请基于错误信息修正,
重新输出完整的 wrapper.py 文件内容。

【用户原始需求】
{capability}

【仓库信息】
  仓库:           {full_name}
  Python 包名:    {package_name}

【你上一版生成的 wrapper 代码】
```python
{prev_code}
```

【测试失败信息】
失败阶段: {stage}
  - import           : 文件根本 import 不进来,可能 import 错包名 / 写错语法
  - function_not_found: 没找到 @mcp.tool() 装饰的函数 / 函数名拼错
  - call_raised      : 函数能 import,但调用时抛了异常 —— 看 traceback 改
  - timeout          : 超时,通常是网络挂了或调了死循环
  - semantic_* / threshold_direction_failure / boundary_failure / unit_failure /
    nodata_failure / spatial_metadata_failure / output_contract_failure:
    基础冒烟已通过，但确定性合成数据断言失败。traceback 中包含能力契约、测试摘要、
    预期值、实际值、失败断言和已通过断言；不得修改预期值来迁就错误实现。

完整 traceback (尾部):
```
{traceback}
```

{unsupported_protocol}

【仓库能够提供核心能力时的硬性输出规则】
  1. 只输出**完整的、可直接保存为 .py 文件**的代码,不要 diff、不要解释、不要 Markdown 围栏。
  2. 文件结构跟你上次一样:docstring + imports + 一个 @mcp.tool() 函数。
  3. 沿用 `from ..mcpx import mcp`。
  4. 只用下面"公开 API"清单中真实存在的符号。**不要发明函数名**。
  5. 异常用 try/except 包住核心调用,失败时 return "[<name> error] ..." 字符串。
  6. 类型注解只用 str / int / float / bool。
  7. 语义失败时优先局部修公式、比较方向、边界、单位、NoData、返回结构或空间元数据；
     保持 traceback 中已通过的断言继续成立。

【公开 API(AST 扫描,真实导出符号)】
{code_api_summary}

【README 节选(原样给你方便对照)】
```
{readme_excerpt}
```

现在输出修正版完整代码:
"""


_CONSISTENCY_FIX_PROMPT = """\
你刚才写了一个 remoteevolver 工具 wrapper,**还没跑测试就被静态校验拦下**:
你引用了仓库里**不存在**的符号(幻觉 API)。请仅修这部分,然后重新输出完整文件。

【用户原始需求】
{capability}

【仓库信息】
  仓库:           {full_name}
  Python 包名:    {package_name}

【你上一版生成的 wrapper 代码】
```python
{prev_code}
```

【一致性校验报告(必须全部修掉)】
{issues_block}

【公开 API 清单(再贴一次,只能从这里挑)】
{code_api_summary}

{unsupported_protocol}

【仓库能够提供核心能力时的硬性输出规则】
  1. 只输出**完整的、可直接保存为 .py 文件**的代码。
  2. 不要 diff、不要解释、不要 Markdown 围栏。
  3. 文件结构跟原来一样:docstring + imports + 一个 @mcp.tool() 函数。
  4. **不要保留任何"幻觉 API"** —— 上面报告里列的所有 symbol 都必须替换或删除。
  5. 替换时优先用"最接近的真实符号"建议；可以组合真实 API。若没有任何真实 API
     能承担核心领域操作，必须使用上面的仓库能力不足协议，不得手写领域实现。
  6. 异常处理 / 类型注解 / 返回值约定 全部不变。

现在输出修正版完整代码:
"""


def fix_wrapper_for_consistency(
    prev_code: str,
    issues: list[CallIssue],
    capability: str,
    signals: RepoSignals,
    llm: BaseLLM,
) -> str:
    """C-2 用:当 audit_wrapper_calls 报出幻觉 API 时,直接让 LLM 改,**不进冒烟测试**。

    省一次 subprocess 启动 + 一次 AttributeError 回路,直接定向修复。
    """
    code_api_summary = (
        signals.code_api.to_prompt_summary()
        if signals.code_api is not None
        else "(无)"
    )
    prompt = _CONSISTENCY_FIX_PROMPT.format(
        capability=capability.strip(),
        full_name=signals.full_name,
        package_name=signals.package_name,
        prev_code=prev_code,
        issues_block=format_issues_for_llm(issues),
        code_api_summary=code_api_summary,
        unsupported_protocol=_UNSUPPORTED_PROTOCOL,
    )
    resp = llm.generate([Message(role="user", content=prompt)])
    return parse_wrapper_response(resp.text or "")


def fix_wrapper(
    prev_code: str,
    test_result: TestResult,
    capability: str,
    signals: RepoSignals,
    llm: BaseLLM,
) -> str:
    """根据 test_result 的 traceback 让 LLM 重写 wrapper。"""
    code_api_summary = (
        signals.code_api.to_prompt_summary()
        if signals.code_api is not None
        else "(AST 扫描未运行,本次只能基于 README 推测 API)"
    )
    prompt = _FIX_PROMPT.format(
        capability=capability.strip(),
        full_name=signals.full_name,
        package_name=signals.package_name,
        prev_code=prev_code,
        stage=test_result.stage,
        traceback=test_result.traceback or "(no traceback)",
        code_api_summary=code_api_summary,
        readme_excerpt=signals.readme_excerpt,
        unsupported_protocol=_UNSUPPORTED_PROTOCOL,
    )
    resp = llm.generate([Message(role="user", content=prompt)])
    return parse_wrapper_response(resp.text or "")
