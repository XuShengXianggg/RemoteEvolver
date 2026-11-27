"""
remoteevolver.reflection.reflector —— 反思器

输入本轮执行轨迹，调用 LLM 提炼一条结构化经验，写入长期记忆。
重点是失败路径：把「为什么失败 / 怎么修」固化下来，下次同类任务能复用。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..llm.base import BaseLLM, Message
from ..memory.long_term import LongTermMemory
from ..prompt.manager import PromptManager


@dataclass
class TraceStep:
    """一次工具调用的轨迹片段。"""
    tool: str
    arguments: dict[str, Any]
    result: str
    error: bool = False
    # ReAct 模型在发出本次工具调用时附带的文本。多数模型这里为空，但若模型
    # 显式说明了选择、约束或预期产物，它是后续判断步骤因果作用的重要证据。
    assistant_text: str = ""
    # 运行时契约可能会规范化/适配参数。arguments 保存真正发给工具的值；仅当
    # 原始请求不同于实际调用时，额外保留 requested_arguments 供审计。
    requested_arguments: dict[str, Any] | None = None
    tool_call_id: str = ""
    validation_evidence: dict[str, Any] = field(default_factory=dict)
    # Compact declared meaning of this tool result, captured from the live MCP
    # schema.  Keep it beside the value so ReAct, Diagnose, and Repair do not
    # have to infer result semantics from a tool name or a bare scalar.
    result_semantics: str = ""
    # Runtime-observed argument provenance and the tool's abstract output shape.
    # This is operational metadata, not chain-of-thought.
    transition_event: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    """整个任务的执行轨迹摘要。"""
    task: str
    steps: list[TraceStep]
    final: str
    success: bool


class Reflector:
    def __init__(
        self,
        llm: BaseLLM,
        prompt_manager: PromptManager,
        long_term_memory: LongTermMemory,
    ) -> None:
        self._llm = llm
        self._prompt = prompt_manager
        self._mem = long_term_memory

    def run(self, trace: Trace) -> dict[str, Any] | None:
        """对一次任务执行做反思；写入一条 experience 到长期记忆。

        返回提炼出的经验 dict（含 task_signature/failure_mode/fix_strategy/evidence），
        失败则返回 None。
        """
        prompt_text = self._prompt.render(
            "reflection",
            task=trace.task,
            success="success" if trace.success else "failure",
            trace=_format_trace(trace),
            final=trace.final or "(no final output)",
        )

        resp = self._llm.generate(
            messages=[Message(role="user", content=prompt_text)],
            tools=None,
        )

        parsed = _try_parse_json(resp.text)
        if parsed is None:
            return None

        self._mem.add_experience(
            task_signature=str(parsed.get("task_signature", "")),
            failure_mode=str(parsed.get("failure_mode", "")),
            fix_strategy=str(parsed.get("fix_strategy", "")),
            evidence=str(parsed.get("evidence", "")),
        )
        return parsed


# ---- helpers ------------------------------------------------------------

def _format_trace(trace: Trace) -> str:
    if not trace.steps:
        return "(no tool calls)"
    lines: list[str] = []
    for i, s in enumerate(trace.steps, 1):
        flag = " [ERROR]" if s.error else ""
        args = json.dumps(s.arguments, ensure_ascii=False)
        requested = (
            "\n   requested=" + json.dumps(s.requested_arguments, ensure_ascii=False)
            if s.requested_arguments is not None else ""
        )
        rationale = f"\n   assistant={s.assistant_text[:500]}" if s.assistant_text else ""
        validation = (
            "\n   validation=" + json.dumps(s.validation_evidence, ensure_ascii=False)
            if s.validation_evidence else ""
        )
        semantics = (
            f"\n   result_semantics={s.result_semantics[:600]}"
            if s.result_semantics else ""
        )
        lines.append(
            f"{i}. {s.tool}({args}){flag}{requested}{rationale}{validation}{semantics}\n"
            f"   → {_clip_trace_result(s.result)}"
        )
    return "\n".join(lines)


def _clip_trace_result(result: str, limit: int = 1200) -> str:
    """保留长结果的首尾和原始长度，避免只截头部丢失汇总/输出路径。"""
    text = str(result)
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = limit - head
    return f"{text[:head]}\n... <truncated; chars={len(text)}> ...\n{text[-tail:]}"


_JSON_BLOCK = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def _try_parse_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    # 1) 整段直接是 JSON
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        pass
    # 2) 抽出第一个 { ... } 块（处理模型加了说明文字的情况）
    m = _JSON_BLOCK.search(text)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        return None
