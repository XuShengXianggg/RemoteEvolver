"""
remoteevolver.evolve.workflow._util —— 工作流模块内部小工具

只放两样跨文件复用的纯函数:
  - tokenize       : 抽词集合(检索打分用),和 memory 文件后端一致的口径
  - extract_json   : 从 LLM 输出里抠出第一个 JSON 对象(容忍前后说明文字)
"""

from __future__ import annotations

import json
import re
from typing import Any

_WORD = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_ROUTING_CONTEXT = re.compile(
    r"(?im)^\s*(?:"
    r"数据目录|input\s+data\s+directory|data\s+directory|"
    r"任务产物目录|task\s+artifact\s+directory|artifact\s+directory"
    r")\s*[:：]"
)


def routing_text(text: str) -> str:
    """Return task semantics without controlled execution-context suffixes.

    The complete task still reaches ReAct.  Only Skill retrieval drops the
    data/output-directory blocks appended by our runners, so shared harness
    wording cannot inflate semantic similarity.
    """
    value = str(text or "").strip()
    match = _ROUTING_CONTEXT.search(value)
    return value[:match.start()].strip() if match else value


def tokenize(text: str) -> set[str]:
    """Tokenize English identifiers and Chinese text for lightweight retrieval.

    Python's generic word regex treats an entire Chinese sentence as one token, so related
    Chinese task descriptions rarely overlapped.  Character bigrams retain a
    dependency-free matcher while allowing task-family memories to be found.
    """
    value = text or ""
    tokens = {m.group(0).lower() for m in _WORD.finditer(value)}
    for match in _CJK.finditer(value):
        sequence = match.group(0)
        if len(sequence) == 1:
            tokens.add(sequence)
        else:
            tokens.update(sequence[i:i + 2] for i in range(len(sequence) - 1))
    return tokens


def extract_json(text: str) -> dict[str, Any] | None:
    """整段是 JSON,或从中抠出第一个 {...} 块;失败返回 None。"""
    if not text:
        return None
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        pass
    # Decode each possible object start independently.  A greedy ``{.*}``
    # slice breaks as soon as the model emits two examples or braces in prose.
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None
