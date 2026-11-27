"""
remoteevolver.evolve.tool.test_args_gen —— LLM 看 wrapper 代码 + README → 产出冒烟测试参数

为什么需要这个:
  capability-only 模式(用户只给一句话需求)下,我们没有人给 --test-arg。
  必须自动猜出一组**真实可用**的参数让 wrapper 跑一次。

设计:
  - 单独一次 LLM 调用(温度低,只要一个 JSON 对象)
  - prompt 要求 LLM 用真实值而不是占位符:股票用 AAPL,URL 用 https://example.com
  - 解析时容错:LLM 可能套 markdown fence、加前缀,清洗一下
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

from ...llm.base import BaseLLM, Message
from .wrapper_gen import RepoSignals


_PROMPT = """\
你刚才为 remoteevolver 写了下面这个工具 wrapper。现在请提出一组**真实可用**的测试参数,
我们会用这组参数跑一次冒烟测试(直接调用函数,看会不会抛异常)。

【硬性要求】
  1. 输出**仅** JSON 对象,例如 {{"symbol": "AAPL"}}。不要 Markdown、不要解释、不要前缀。
  2. 键名必须严格匹配 wrapper 函数的形参名(看下面代码)。
  3. 值必须是**真实可用的具体值**,不要占位符:
        - 股票代码用 "AAPL" / "MSFT",**不要** "TICKER"
        - URL 用 "https://example.com",**不要** "<url>"
        - 文件路径用临时生成的 / 已知存在的,**不要** "/path/to/file.png"
     如果工具需要文件路径输入而又没有现成文件,选一个"输出文件"型参数
     (如 path="out.png"),这样工具会自己创建文件而不是读取不存在的文件。
  4. 数值参数用合理值:width=100、height=100、limit=10 之类。
  5. 不要传 null;LLM 形参允许的就给一个具体值。
{sample_block}
【用户能力描述】
{capability}

【wrapper 代码】
```python
{wrapper_code}
```

【README 节选(参考真实 API 用法)】
```
{readme_excerpt}
```

现在输出 JSON:
"""


_SAMPLE_BLOCK = """\

【任务真实输入文件(强制使用)】
下面是本任务**真实存在**的输入文件路径。凡是需要"读取已有文件"的形参,
**必须从这个清单里选真实路径**(按扩展名/文件名语义匹配用途;需要多个文件的
形参就从清单里挑出对应的一组,组成 list 或按 wrapper 期望的格式)。
**严禁编造**清单以外的输入路径。
{sample_files}

【输出文件目录(强制)】
所有"输出文件"型形参(写出结果的路径)**必须**放到下面这个专用临时目录里,
文件名自拟(如 .../result.tif):
  {out_dir}
**绝对禁止**把输出写进上面输入文件所在的目录——那会污染任务输入数据。
"""


_CHAIN_BLOCK = """\

【上游工具刚产出的中间文件(本工具在链中要消费它们 —— 优先使用!)】
本工具处于一条工具链中,下面这些是**上游工具刚刚真实产出**的中间结果。如果本工具
的职责就是"读取上一步的产物再加工"(绝大多数链式工具都是),**请优先用这些路径**
作为"读取已有文件"型形参的取值——这样冒烟测试就在验证"本工具读得了上游的输出"
(集成正确性),而不仅是孤立能跑。
{chain_files}
"""


def generate_test_args(
    capability: str,
    wrapper_code: str,
    signals: RepoSignals,
    llm: BaseLLM,
    *,
    sample_inputs: list[str] | None = None,
    prior_outputs: list[str] | None = None,
) -> dict:
    """让 LLM 看着 wrapper 出一组冒烟测试参数。

    Args:
        sample_inputs: 任务真实存在的输入文件路径清单。非空时强制 LLM 对
            "读取已有文件"型形参从中选真实路径,而不是编造路径——这样 IO
            工具的冒烟测试能拿到真数据,真正验证 wrapper 跑得通。
        prior_outputs: 同一工具链中上游工具刚产出的中间文件。非空时提示 LLM
            **优先**用它们做输入——把冒烟测试变成"读得了上游输出"的集成校验
            (方向 B:让"串不起来"在进化时就暴露,而非拖到 ReAct 编排才炸)。

    Returns:
        dict;键=形参名,值=真实可用的值。

    Raises:
        ValueError: LLM 输出无法解析成 JSON 对象。
    """
    if sample_inputs or prior_outputs:
        # 专用临时输出目录:冒烟测试的产物写这里,绝不污染任务输入目录,测完即弃。
        out_dir = tempfile.mkdtemp(prefix="evolve_smoke_out_")
        listing = "\n".join(f"  - {p}" for p in (sample_inputs or [])) or "  (无)"
        sample_block = _SAMPLE_BLOCK.format(
            sample_files=listing, out_dir=str(Path(out_dir)),
        )
        if prior_outputs:
            chain_listing = "\n".join(f"  - {p}" for p in prior_outputs)
            sample_block += _CHAIN_BLOCK.format(chain_files=chain_listing)
    else:
        sample_block = ""
    prompt = _PROMPT.format(
        capability=capability.strip(),
        wrapper_code=wrapper_code,
        readme_excerpt=signals.readme_excerpt,
        sample_block=sample_block,
    )
    resp = llm.generate([Message(role="user", content=prompt)])
    text = (resp.text or "").strip()
    text = _strip_fences(text)
    data = _extract_json_object(text)
    if data is None:
        raise ValueError(
            f"test_args LLM 输出不包含合法 JSON 对象:\n  raw={text[:300]!r}"
        )

    if not isinstance(data, dict):
        raise ValueError(f"test_args 必须是 JSON 对象,拿到:{type(data).__name__}")
    if sample_inputs or prior_outputs:
        _reject_invented_input_paths(
            data,
            known_inputs=[*(sample_inputs or []), *(prior_outputs or [])],
        )
    return data


_PATH_SUFFIXES = {
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".json", ".csv",
    ".txt", ".nc", ".hdf", ".h5", ".shp", ".geojson",
}


def _reject_invented_input_paths(
    arguments: dict, *, known_inputs: list[str],
) -> None:
    """Reject fabricated read paths when the task supplied real file evidence."""
    known = {str(Path(value).resolve()).casefold() for value in known_inputs}

    def visit(key: str, value) -> None:
        lowered_key = key.casefold()
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(str(child_key), child)
            return
        if isinstance(value, list):
            for child in value:
                visit(key, child)
            return
        if not isinstance(value, str):
            return
        if "output" in lowered_key or lowered_key.startswith("out_"):
            return
        key_is_file_input = any(marker in lowered_key for marker in (
            "file", "path", "raster", "image", "band", "input",
        ))
        if not key_is_file_input:
            return
        for candidate in (part.strip() for part in value.split(",")):
            if not candidate:
                continue
            path = Path(candidate)
            if path.suffix.casefold() not in _PATH_SUFFIXES:
                continue
            resolved = str(path.resolve()).casefold()
            if resolved not in known:
                raise ValueError(
                    "test_args invented an input path outside the verified task "
                    f"samples: {key}={candidate!r}"
                )

    for name, value in arguments.items():
        visit(str(name), value)


def _strip_fences(text: str) -> str:
    """LLM 偶尔套 ```json ... ```,剥掉。"""
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*\n(.*?)\n```\s*$", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n", "", text)
    if text.endswith("```"):
        text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


def _extract_json_object(text: str) -> dict | None:
    """Return the first decodable JSON object, tolerating model prose.

    ``JSONDecoder.raw_decode`` keeps nested objects and braces inside strings
    correct, unlike a greedy regular-expression slice.  Trying each opening
    brace also skips examples or malformed brace fragments before the answer.
    """
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None
