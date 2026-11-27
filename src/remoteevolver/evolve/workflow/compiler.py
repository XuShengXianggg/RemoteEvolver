"""
remoteevolver.evolve.workflow.compiler —— workflow-to-skill 编译(W2S)

把一条**成功**的 ReAct 轨迹(一串工具调用)交给 LLM,蒸馏成参数化技能:
  - 识别跨实例会变化的具体参数 → 挖成 {slot} 槽位
  - 把每步 args 里的具体值替换成槽位,工具名原样保留
  - 产出调用接口说明 / 一句抽象 task_pattern / 文本 guideline
  - 记录源任务里这些槽位的具体取值(example_bindings)+ 一个期望子串(replay_expect),
    供 admission 阶段「用原始取值重放能否复现」做功能正确性闸

输入 steps 形如 execution_record.payload["steps"]:
  [{"tool": "...", "arguments": {...}, "result": "...", "error": bool}, ...]

任何不可编译(LLM 判 reusable=false / JSON 不合法 / 槽位对不上)都返回 None,
**不抛异常**——和 evolve/tool 流水线同一风格,便于上层决定下一步。
"""

from __future__ import annotations
import os

import json
import math
from copy import deepcopy
from typing import Any

from ...llm.base import BaseLLM, Message, Response, generate_required_tool
from .dataflow import matching_argument, matching_result_field
from .executor import _DERIVED_OPERATIONS, _finite_number, _resolve_value
from .record import (
    CapabilityInterface,
    CapabilityPort,
    ReuseConstraints,
    SkillRecord,
    USAGE_CONDITIONED_ONLY,
    USAGE_DIRECT,
)

# 编译 skill 的 LLM 调用解析失败时的重试次数(兜网关截断/坏 JSON)
_COMPILE_MAX_ATTEMPTS = int(os.getenv("REMOTEEVOLVER_SKILL_COMPILE_ATTEMPTS", "2"))

_COMPILER_TOOL_NAME = "submit_workflow_skill"
_COMPILER_TOOL = {
    "type": "function",
    "function": {
        "name": _COMPILER_TOOL_NAME,
        "description": "Submit a strictly structured workflow Skill classification and abstraction.",
        "parameters": {
            "type": "object",
            "properties": {
                "reusable": {"type": "boolean"},
                "usage_mode": {
                    "type": "string",
                    "enum": ["executable", "experience"],
                },
                "reason": {"type": "string"},
                "task_pattern": {
                    "type": "string",
                    "description": (
                        "Abstract task pattern enabled by the complete tool chain; "
                        "exclude source-task-specific answer wording and instance semantics."
                    ),
                },
                "capability_interface": {
                    "type": "object",
                    "properties": {
                        "inputs": {"type": "array", "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "required": {"type": "boolean"},
                            },
                            "required": ["name", "description", "required"],
                            "additionalProperties": False,
                        }},
                        "outputs": {"type": "array", "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "required": {"type": "boolean"},
                            },
                            "required": ["name", "description", "required"],
                            "additionalProperties": False,
                        }},
                    },
                    "required": ["inputs", "outputs"],
                    "additionalProperties": False,
                },
                "reuse_constraints": {
                    "type": "object",
                    "properties": {
                        "semantic_scope": {
                            "type": "string",
                            "description": (
                                "Brief domain or applicability qualifier; do not restate "
                                "the task_pattern."
                            ),
                        },
                        "soft_constraints": {"type": "array", "items": {"type": "string"}},
                        "fallback_mode": {"type": "string", "enum": ["conditioned", "dynamic"]},
                    },
                    "required": [
                        "semantic_scope", "soft_constraints", "fallback_mode",
                    ],
                    "additionalProperties": False,
                },
                "guideline": {"type": "string"},
                "abstract_stages": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "reusable", "usage_mode", "reason", "task_pattern",
                "capability_interface", "reuse_constraints", "guideline", "abstract_stages",
            ],
            "additionalProperties": False,
        },
    },
}


# LLM 只负责语义差异；完整工具骨架由成功轨迹在本地确定性生成。保留上面的
# 旧说明作为设计文档，实际请求使用这个紧凑协议。
# Final compiler contract: structure is built locally from runtime provenance.
# The LLM is deliberately limited to semantic abstraction and may only
# downgrade the locally supportable usage mode.
_COMPILE_PROMPT = """\
Semantically abstract one successful, verified tool trajectory. All generated
free-text fields and semantic identifiers must be in English.

Candidate kind requested by the caller: {target_kind}

Task:
{task}

Verified final-answer evidence:
{final}

Real tool trajectory (for semantic understanding only; do not select, remove,
add, or reorder tools, and do not generate argument references):
{steps}

Call submit_workflow_skill. Set usage_mode exactly to {target_kind}. You may
decide only whether this requested candidate kind has enough observed evidence;
its task_pattern, whole-Skill semantic inputs and outputs, scope, constraints,
high-level stages, and guidance. There are only two reusable Skill kinds:
executable and experience. Never substitute one kind for the requested kind.
Both are injected into the ordinary Agent ReAct loop, which chooses current
tools and arguments. "direct" means no Skill is injected and is not a Skill kind.

For an experience candidate, produce compact semantic guidance rather than an
approximate recipe. Its guideline must accurately summarize the
transferable lesson in 2-4 concise sentences: what generally helped, the narrow
conditions under which it applies, what must be checked on the current task,
and when it must be ignored. Describe a reasoning method, never a stored answer.
Where the successful trajectory supports them, include the useful task-analysis
decision, why an actually used tool or tool family was appropriate, and a
concrete mistake to avoid. Tie each claim to the supplied task, assistant_text,
actual_args, or tool result; do not invent a tool-selection rule or a failure
that the evidence does not show. Do not reproduce the complete tool sequence.
If the trace supports no non-obvious, transferable guidance, set reusable=false
instead of filling the Skill with generic advice.
Do not carry over source-specific conclusions, values, paths, entities, dates,
regions, or causal claims that were not established by the trajectory. If a
conclusion can change with current inputs, instruct the agent to recompute or
verify it instead of asserting it. Keep abstract_stages to 1-4 short strategy
steps and put uncertainty or applicability limits in soft_constraints.

For an experience candidate, put only 1-3 evidence-backed applicability limits
in soft_constraints and omit generic cautions such as "check the inputs". If the
evidence shows a specific incompatible input layout, method, scope, or output
reduction, state that narrow boundary. Never turn a source-task sample count,
incidental parameter, or missing log detail into a general applicability limit
unless the task or tool evidence explicitly establishes it. Treat archive markers
saying that arguments or results are unavailable as unknown evidence, never as
proof that arguments were empty or results were identical. Missing evidence for
one limit means omitting that limit, not rejecting an otherwise supported Skill.

For an executable candidate, make capability_interface.outputs and guideline
an answer-recovery contract rather than generic prose. Enumerate every result
component that a correct final answer must recover from current tool evidence,
including its semantic meaning, comparison direction, aggregation, and unit when
these are invariant for the task family. Explain in guideline how to map the
current tool results to those outputs and which checks prevent misinterpretation.
Anchor that mapping to the producing tool's role and its current resolved input
arguments (such as the entity/group files), never merely to a result's list
position or numeric magnitude. For derived comparison outputs, state that the
operand meaning and direction must be recovered from the current resolved
arguments and task wording.
Never include the source answer or any source-specific value in this contract.
Explain the reusable decision behind the observed tool order only when the
actual trajectory supports it. State the input/result check that matters before
moving to the next stage, and the condition under which the Agent should adapt
or abandon the reference sequence. Do not claim that an unobserved branch,
failure, or alternative tool was verified by this one successful trace.

For both candidate kinds, preserve the smallest stable structure that determines
reuse: distinct semantic input roles or groups, the order of essential
transformations and reductions, spatial-versus-temporal scope, and the requested
final statistic or relation. Generalize places, dates, paths, and literal values,
but do not generalize away input multiplicity when it changes the operation,
operation order, scope, reduction level, or the final reduction.

capability_interface describes the whole Skill, not individual tool arguments.
Use stable concise English names and descriptions; exclude source paths, regions,
years, filenames, and instance values. task_pattern must describe only the reusable
task pattern enabled by the complete tool chain: the input kind, stable processing,
and broad result type. Do not copy source-task-specific answer wording, comparison
direction, threshold formula, units, or requested result combination unless it is
an invariant of the tool chain itself. semantic_scope must be only a brief domain
or applicability qualifier and must not restate task_pattern. Do not generate hard
preconditions or postconditions; applicability enforcement remains outside this
descriptive Skill metadata. Every remaining constraint must be abstract English text.

Do not output tools, step numbers, argument names, slots, formulas, paths, edits,
transitions, or call templates; local code derives them from runtime provenance.
"""


def compile_trace(
    task: str,
    steps: list[dict[str, Any]],
    final: str,
    llm: BaseLLM,
    *,
    target_kind: str = "executable",
    verbose: bool = False,
    max_attempts: int | None = None,
) -> SkillRecord | None:
    """把成功轨迹编译成 SkillRecord;不可编译返回 None。"""

    def log(msg: str) -> None:
        if verbose:
            print(f"[workflow.compile] {msg}", flush=True)

    # 只保留有意义的工具步骤(去掉报错步;成功轨迹里报错步通常是中途试错)
    clean = [s for s in steps if s.get("tool") and not s.get("error")]
    if not clean:
        log("无可用工具步骤,跳过")
        return None
    if target_kind not in {"executable", "experience"}:
        raise ValueError(f"invalid target_kind: {target_kind!r}")

    prompt_steps = steps if target_kind == "experience" else clean
    prompt = _COMPILE_PROMPT.format(
        target_kind=target_kind,
        task=task.strip(),
        final=(final or "").strip()[:500] or "(无)",
        steps=_render_steps(prompt_steps),
    )
    # 一次性 LLM 调用易被网关截断成坏 JSON / 掉线 → 单次失败就丢技能。重试几次,
    # 让"本该成功的确定步骤"别被一次抽风的响应杀掉(连接错误由 client 内部重试,
    # 这里再兜住"返回了但解析不出 JSON"以及偶发异常)。
    attempts = _COMPILE_MAX_ATTEMPTS if max_attempts is None else max_attempts
    if attempts <= 0:
        raise ValueError("max_attempts must be > 0")
    data = None
    for attempt in range(1, attempts + 1):
        try:
            resp = generate_required_tool(
                llm, [Message(role="user", content=prompt)], _COMPILER_TOOL,
            )
            if verbose:
                preview = resp.text.strip()
                if len(preview) > 6000:
                    preview = preview[:6000] + f"... <truncated; chars={len(resp.text)}>"
                log(f"LLM 原始编译响应: {preview or '(empty)'}")
                if resp.tool_calls:
                    tool_preview = json.dumps(
                        [
                            {"name": call.name, "arguments": call.arguments}
                            for call in resp.tool_calls
                        ],
                        ensure_ascii=False,
                    )
                    if len(tool_preview) > 12000:
                        tool_preview = tool_preview[:12000] + (
                            f"... <truncated; chars={len(tool_preview)}>"
                        )
                    log(f"LLM 编译工具调用: {tool_preview}")
            data = _extract_compiler_response(resp)
        except Exception as e:  # noqa: BLE001
            log(f"编译 LLM 调用异常(第 {attempt}/{attempts} 次): {type(e).__name__}: {e}")
            data = None
        if data is not None:
            break
        if attempt < attempts:
            log(f"LLM 未返回合法 JSON,重试({attempt}/{attempts})")
    if data is None:
        log(f"LLM {attempts} 次仍未返回合法 JSON,跳过")
        return None
    if not data.get("reusable", False):
        reason = str(data.get("reason", "")).strip() or "模型未提供原因"
        log(f"LLM 判定不可复用,跳过: {reason[:500]}")
        return None

    if data.get("usage_mode") != target_kind:
        log(
            "compiler returned the wrong candidate kind: "
            f"requested={target_kind} returned={data.get('usage_mode')!r}"
        )
        return None

    # SkillRecord still accepts the historical storage values in old checkouts.
    # New compiler decisions are canonical and are converted only at this narrow
    # compatibility boundary; persisted SKILL.md records use the canonical kind.
    data = dict(data)
    data["candidate_kind"] = target_kind
    data["usage_mode"] = (
        USAGE_CONDITIONED_ONLY if target_kind == "experience" else USAGE_DIRECT
    )

    data = _apply_runtime_provenance(data, clean, log)

    if not data.get("call_template"):
        data = _expand_semantic_edits(data, clean, log)
        if data is None:
            return None

    skill = _validate_and_build(task, data, clean, log)
    if skill is not None:
        # A successful task may yield one candidate of each kind. Include the
        # kind in the id so equal task patterns cannot overwrite one another.
        skill.skill_id = f"{skill.skill_id}_{target_kind}"
    return skill


# ---- 内部 ---------------------------------------------------------------

def _extract_compiler_json(text: str) -> dict[str, Any] | None:
    """Parse only the complete compiler root object; never salvage inner JSON."""
    if not text or not text.strip():
        return None
    candidate = text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        if len(lines) < 3 or lines[0].strip().lower() not in {"```", "```json"}:
            return None
        candidate = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return _validate_compiler_payload(value)


def _extract_compiler_response(response: Response) -> dict[str, Any] | None:
    calls = [call for call in response.tool_calls if call.name == _COMPILER_TOOL_NAME]
    if response.tool_calls:
        if len(calls) != 1 or len(response.tool_calls) != 1:
            return None
        return _validate_compiler_payload(calls[0].arguments)
    # 兼容不支持 tool calling 的 provider；仍只接受完整根 JSON。
    return _extract_compiler_json(response.text)


def _validate_compiler_payload(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not isinstance(value.get("reusable"), bool):
        return None
    if value["reusable"]:
        # Read-only compatibility for historical provider/test payloads. The
        # live tool schema no longer exposes any of these structural fields.
        if "capability_interface" not in value:
            legacy_types = {
                "task_pattern": str, "parameters": list,
                "interface": dict, "guideline": str,
            }
            if any(not isinstance(value.get(key), kind) for key, kind in legacy_types.items()):
                return None
            if not isinstance(value.get("edits", value.get("call_template")), list):
                return None
            value = dict(value)
            value["capability_interface"] = {
                "inputs": [],
                "outputs": [{
                    "name": "task_result",
                    "description": str(value.get("task_pattern", "task result")),
                    "required": True,
                }],
            }
        required_types = {
            "task_pattern": str,
            "capability_interface": dict,
            "guideline": str,
        }
        if any(not isinstance(value.get(key), expected) for key, expected in required_types.items()):
            return None
        mode = value.get("usage_mode")
        if mode not in {"executable", "experience"}:
            return None
        if "reuse_constraints" in value and not isinstance(value["reuse_constraints"], dict):
            return None
        stages = value.get("abstract_stages", [])
        if not isinstance(stages, list) or any(not isinstance(item, str) for item in stages):
            return None
        if mode == "experience":
            if not any(item.strip() for item in stages):
                return None
        interface = value.get("capability_interface")
        if set(interface or {}) != {"inputs", "outputs"}:
            return None
        for direction in ("inputs", "outputs"):
            ports = interface.get(direction)
            if not isinstance(ports, list):
                return None
            for port in ports:
                if (not isinstance(port, dict)
                        or set(port) != {"name", "description", "required"}
                        or not isinstance(port.get("name"), str)
                        or not isinstance(port.get("description"), str)
                        or not isinstance(port.get("required"), bool)):
                    return None
    return value

def _expand_semantic_edits(
    data: dict[str, Any], clean_steps: list[dict[str, Any]], log,
) -> dict[str, Any] | None:
    """Apply the LLM's small semantic patch to an exact local trace skeleton."""
    if data.get("usage_mode") == USAGE_CONDITIONED_ONLY:
        expanded = dict(data)
        expanded["call_template"] = []
        expanded["example_bindings"] = {}
        expanded["compiler_corrections"] = [
            str(item) for item in (data.get("compiler_corrections") or [])
        ]
        return expanded
    template = [
        {"tool": str(step["tool"]), "args": deepcopy(step.get("arguments", {}) or {})}
        for step in clean_steps
    ]
    seen: set[tuple[int, str]] = set()
    corrections = [str(item) for item in (data.get("compiler_corrections") or [])]
    edits = data.get("edits") or []
    if not isinstance(edits, list):
        log("edits 不是 JSON 数组,跳过")
        return None
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {"step", "arg", "value"}:
            log(f"非法语义 edit {edit!r},跳过")
            return None
        index, arg = edit.get("step"), edit.get("arg")
        if (not isinstance(index, int) or index < 0 or index >= len(template)
                or not isinstance(arg, str) or arg not in template[index]["args"]):
            log(f"edit 指向不存在的参数 step={index!r} arg={arg!r},跳过")
            return None
        location = (index, arg)
        if location in seen:
            log(f"同一参数出现重复 edit {location!r},跳过")
            return None
        seen.add(location)
        template[index]["args"][arg] = edit["value"]
    expanded = dict(data)
    expanded["call_template"] = template
    expanded["compiler_corrections"] = corrections
    # 完整源值由 _capture_bindings 从 clean_steps 确定性捕获。
    expanded["example_bindings"] = {}
    return expanded

def _apply_runtime_provenance(
    data: dict[str, Any], clean_steps: list[dict[str, Any]], log,
) -> dict[str, Any]:
    """Build safe executable bindings from runtime evidence when available.

    Missing provenance is handled per argument rather than rejecting the whole
    trajectory.  An unclassified argument becomes a current-task slot, so its
    source value cannot leak into the reusable template.  Proven upstream data
    flow remains a structured reference.
    """
    if data.get("usage_mode") == USAGE_CONDITIONED_ONLY:
        return data
    fixed = deepcopy(data)
    fixed["parameters"] = []
    fixed["interface"] = {}
    fixed["edits"] = []
    fixed["hybrid_transitions"] = []
    edits = [deepcopy(item) for item in (fixed.get("edits") or [])]
    direct_edits: dict[tuple[int, str], dict[str, Any]] = {}
    task_inputs: list[tuple[int, str]] = []

    # Executable is a ReAct reference workflow, not a fixed replay. Runtime
    # provenance parameterizes both task-owned and ReAct-derived values. When
    # an argument lacks usable provenance, fail closed at the argument level by
    # turning it into a current-task slot instead of retaining the source value.
    fixed["usage_mode"] = USAGE_DIRECT

    for step_index, step in enumerate(clean_steps):
        event = step.get("transition_event")
        sources = event.get("argument_sources") if isinstance(event, dict) else None
        if not isinstance(sources, dict):
            sources = {}
        actual_args = step.get("arguments", {}) or {}
        for arg in actual_args:
            source = sources.get(arg)
            reference = (
                _reference_from_runtime_source(source, step_index)
                if isinstance(source, dict) else None
            )
            if reference is not None:
                direct_edits[(step_index, str(arg))] = {
                    "step": step_index, "arg": str(arg), "value": reference,
                }
            else:
                # task_input/react are expected runtime inputs. Unknown,
                # incomplete, and legacy provenance receives the same safe
                # treatment: parameterize it rather than copying its value.
                task_inputs.append((step_index, str(arg)))

    used_slot_names: set[str] = set()
    for step_index, arg in task_inputs:
        slot = arg
        if slot in used_slot_names:
            slot = f"step_{step_index}_{arg}"
        used_slot_names.add(slot)
        edits.append({"step": step_index, "arg": arg, "value": "{" + slot + "}"})
        fixed["parameters"].append(slot)

    edits = [
        item for item in edits
        if (item.get("step"), item.get("arg")) not in direct_edits
    ]
    edits.extend(direct_edits.values())

    # The live compiler protocol returns semantic edits, but keep strict-JSON
    # legacy/provider responses with a full template correct as well.
    template = fixed.get("call_template")
    if isinstance(template, list):
        for edit in edits:
            step_index, arg = edit.get("step"), edit.get("arg")
            if (isinstance(step_index, int) and isinstance(arg, str)
                    and 0 <= step_index < len(template)
                    and isinstance(template[step_index].get("args"), dict)
                    and arg in template[step_index]["args"]):
                template[step_index]["args"][arg] = deepcopy(edit["value"])

    # The replacement chain has no Hybrid candidate. Runtime facts are either
    # expressed as ordinary executable slots/references or rejected later by
    # structural validation.
    normalized: list[dict[str, Any]] = []

    fixed["edits"] = edits
    fixed["hybrid_transitions"] = normalized
    fixed["hybrid_steps"] = []
    if not isinstance(template, list):
        used_slots: set[str] = set()
        for item in edits:
            used_slots.update(_slots_in(item.get("value")))
        fixed["parameters"] = [
            str(name) for name in (fixed.get("parameters") or [])
            if str(name) in used_slots
        ]
        if isinstance(fixed.get("interface"), dict):
            fixed["interface"] = {
                str(name): description
                for name, description in fixed["interface"].items()
                if str(name) in used_slots
            }
    corrections = [str(item) for item in (fixed.get("compiler_corrections") or [])]
    corrections.append(
        "runtime transition events supplied proven dataflow; unclassified "
        "arguments were converted to current-task slots"
    )
    fixed["compiler_corrections"] = corrections
    log(
        "runtime transition events applied: "
        f"references={len(direct_edits)}, current-value-slots={len(task_inputs)}"
    )
    return fixed


def _reference_from_runtime_source(
    source: dict[str, Any], before_step: int,
) -> dict[str, Any] | None:
    source_step = source.get("source_step")
    selector = source.get("selector")
    if (source.get("mode") != "direct" or not isinstance(source_step, int)
            or source_step < 0 or source_step >= before_step
            or not isinstance(selector, dict)):
        return None
    if isinstance(selector.get("field"), str):
        return {"$from_step": source_step, "$field": selector["field"]}
    if isinstance(selector.get("arg"), str):
        return {"$from_step": source_step, "$arg": selector["arg"]}
    if selector.get("raw") is True:
        return {"$from_step": source_step, "$raw": True}
    return None


def _render_steps(steps: list[dict[str, Any]]) -> str:
    """Render enough evidence for abstraction without repeating long containers.

    The unabridged ``clean_steps`` remain available to deterministic validation,
    binding capture and source replay after the LLM returns.
    """
    lines: list[str] = []
    for i, s in enumerate(steps):
        args = json.dumps(
            _compact_prompt_value(s.get("arguments", {})), ensure_ascii=False,
        )
        result = str(s.get("result", ""))
        try:
            parsed_result = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            parsed_result = None
        if isinstance(parsed_result, (list, dict)):
            result = json.dumps(
                _compact_prompt_value(parsed_result), ensure_ascii=False,
            )
        elif len(result) > 1200:
            result = (
                f"{result[:800]}\n... <truncated; chars={len(result)}> ...\n"
                f"{result[-400:]}"
            )
        evidence = str(s.get("assistant_text", ""))[:400]
        transition = s.get("transition_event")
        transition_text = (
            json.dumps(_compact_prompt_value(transition), ensure_ascii=False)
            if isinstance(transition, dict) else "(legacy trace; unavailable)"
        )
        lines.append(
            f"{i}. status={'error' if s.get('error') else 'success'} "
            f"tool={s['tool']} actual_args={args}\n"
            f"   runtime_transition={transition_text}\n"
            f"   assistant_text={evidence or '(none)'}\n   → {result}"
        )
    return "\n".join(lines)


def _compact_prompt_value(value: Any, *, max_items: int = 4) -> Any:
    """Summarize long lists as non-addressable prompt-only text."""
    if isinstance(value, list):
        if len(value) > max_items:
            samples = json.dumps(
                [value[0], value[1], value[-1]], ensure_ascii=False,
            )
            return (
                f"[PROMPT_ONLY_LIST_SUMMARY actual_type=array length={len(value)} "
                f"samples={samples}; full_value_retained_locally]"
            )
        return [_compact_prompt_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _compact_prompt_value(item) for key, item in value.items()}
    return value


def _slots_in(value: Any) -> set[str]:
    """抽出字符串里的 {slot} 占位名(只认形如 {ident} 的整体或内嵌)。"""
    import re
    if isinstance(value, str):
        return set(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", value))
    if isinstance(value, list):
        return set().union(*(_slots_in(v) for v in value), set())
    if isinstance(value, dict):
        return set().union(*(_slots_in(v) for v in value.values()), set())
    return set()


def _template_slots(call_template: list[dict[str, Any]]) -> set[str]:
    used: set[str] = set()
    for step in call_template:
        for v in (step.get("args", {}) or {}).values():
            used |= _slots_in(v)
    return used


def is_from_step(value: Any) -> bool:
    """是不是「引用上一步输出」的结构化值 {"$from_step": N}。"""
    return isinstance(value, dict) and "$from_step" in value


def _upgrade_dataflow_refs(
    call_template: list[dict[str, Any]], clean_steps: list[dict[str, Any]], log,
) -> list[dict[str, Any]]:
    """Infer selectors from successful trace values instead of LLM guesses."""
    def upgrade(value: Any, real: Any, step_index: int) -> Any:
        if is_from_step(value):
            idx = value.get("$from_step")
            if isinstance(idx, int) and 0 <= idx < step_index and idx < len(clean_steps):
                upstream = str(clean_steps[idx].get("result", ""))
                field = matching_result_field(upstream, real)
                if field:
                    inferred = {"$from_step": idx, "$field": field}
                    if value != inferred:
                        log(
                            f"数据流 step {idx} → field {field!r}（由成功轨迹的真实参数反推，"
                            f"纠正原引用 {value!r}）"
                        )
                    return inferred
                argument = matching_argument(
                    (clean_steps[idx].get("arguments", {}) or {}), real
                )
                if argument:
                    inferred = {"$from_step": idx, "$arg": argument}
                    if value != inferred:
                        log(
                            f"dataflow step {idx} -> argument {argument!r} "
                            f"(inferred from the successful trace; corrected {value!r})"
                        )
                    return inferred
                if upstream == real:
                    return {"$from_step": idx, "$raw": True}
            return value
        if isinstance(value, list) and isinstance(real, list):
            return [
                upgrade(item, real[pos] if pos < len(real) else None, step_index)
                for pos, item in enumerate(value)
            ]
        if isinstance(value, dict) and isinstance(real, dict):
            return {k: upgrade(v, real.get(k), step_index) for k, v in value.items()}
        return value

    upgraded: list[dict[str, Any]] = []
    for i, step in enumerate(call_template):
        fixed = dict(step)
        aligned = i < len(clean_steps) and step.get("tool") == clean_steps[i].get("tool")
        real_args = (clean_steps[i].get("arguments", {}) if aligned else {}) or {}
        fixed["args"] = {
            key: upgrade(value, real_args.get(key), i)
            for key, value in (step.get("args", {}) or {}).items()
        }
        upgraded.append(fixed)
    return upgraded


def _capture_from(tmpl_v: Any, real_v: Any, pset: set[str], out: dict[str, Any]) -> None:
    """递归:模板值里凡是"整体 {slot}",就把对位的**真实值**记进 out[slot]。
    list 按下标对齐、dict 按键对齐;$from_step 不是槽位,跳过。
    """
    import re
    if isinstance(tmpl_v, str):
        m = re.fullmatch(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", tmpl_v.strip())
        if m and m.group(1) in pset and real_v is not None and m.group(1) not in out:
            out[m.group(1)] = real_v
        return
    if isinstance(tmpl_v, list) and isinstance(real_v, list):
        for tv, rv in zip(tmpl_v, real_v):
            _capture_from(tv, rv, pset, out)
        return
    if isinstance(tmpl_v, dict) and "$from_step" not in tmpl_v and isinstance(real_v, dict):
        for k, tv in tmpl_v.items():
            _capture_from(tv, real_v.get(k), pset, out)


def _capture_bindings(
    call_template: list[dict[str, Any]],
    clean_steps: list[dict[str, Any]],
    parameters: list[str],
) -> dict[str, Any]:
    """从真实轨迹按槽位**程序化提取** example_bindings(确定性,避免 LLM 复述长
    列表/长路径时失真——这是 file-IO 工作流 admission 重放失败的主因)。

    只在"模板某参数值是整体 `{slot}` 且该步工具名与轨迹同位对齐"时捕获真实值;
    对不齐 / 内嵌槽位 / LLM 也没标的,留给 LLM 复述兜底。
    """
    captured: dict[str, Any] = {}
    pset = set(parameters)
    for i, step in enumerate(call_template):
        if i >= len(clean_steps):
            break
        if str(step.get("tool", "")) != str(clean_steps[i].get("tool", "")):
            continue  # 顺序/工具对不齐,不敢按位置取真实值
        real_args = clean_steps[i].get("arguments", {}) or {}
        for key, tmpl_v in (step.get("args", {}) or {}).items():
            _capture_from(tmpl_v, real_args.get(key), pset, captured)
    return captured


def _bad_from_step_refs(call_template: list[dict[str, Any]]) -> list[Any]:
    """返回非法的 $from_step 引用(N 不是 int / 越界 / 引用了当前或之后的步)。"""
    bad: list[Any] = []

    def visit(value: Any, step_index: int) -> None:
        if is_from_step(value):
            idx = value.get("$from_step")
            has_field = isinstance(value.get("$field"), str) and bool(value.get("$field"))
            has_raw = value.get("$raw") is True
            has_arg = isinstance(value.get("$arg"), str) and bool(value.get("$arg"))
            selector_count = sum((has_field, has_raw, has_arg))
            allowed = (
                {"$from_step", "$field"} if has_field
                else {"$from_step", "$arg"} if has_arg
                else {"$from_step", "$raw"}
            )
            if (not isinstance(idx, int) or idx < 0 or idx >= step_index
                    or selector_count != 1 or set(value) - allowed):
                bad.append(value)
            return
        if isinstance(value, list):
            for child in value:
                visit(child, step_index)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child, step_index)

    for i, step in enumerate(call_template):
        visit(step.get("args", {}) or {}, i)
    return bad


def _bad_derived_expressions(call_template: list[dict[str, Any]]) -> list[Any]:
    """Return malformed or non-whitelisted declarative arithmetic expressions."""
    bad: list[Any] = []

    def visit(value: Any, depth: int = 0) -> None:
        if isinstance(value, dict) and "$derive" in value:
            expression = value.get("$derive")
            valid = (
                set(value) == {"$derive"}
                and isinstance(expression, dict)
                and set(expression) == {"op", "args"}
                and expression.get("op") in _DERIVED_OPERATIONS
                and isinstance(expression.get("args"), list)
                and len(expression["args"]) == 2
                and depth < 8
            )
            if not valid:
                bad.append(value)
                return
            for child in expression["args"]:
                visit(child, depth + 1)
            return
        if isinstance(value, dict):
            for child in value.values():
                visit(child, depth)
        elif isinstance(value, list):
            for child in value:
                visit(child, depth)

    for step in call_template:
        visit(step.get("args", {}) or {})
    return bad


def _derived_source_mismatches(
    call_template: list[dict[str, Any]],
    clean_steps: list[dict[str, Any]],
    bindings: dict[str, Any],
) -> list[str]:
    """Verify every derived value against the concrete successful source trace."""
    mismatches: list[str] = []
    prior_results = [str(step.get("result", "")) for step in clean_steps]
    prior_arguments = [(step.get("arguments", {}) or {}) for step in clean_steps]

    def visit(template: Any, actual: Any, step_index: int, path: str) -> None:
        if isinstance(template, dict) and "$derive" in template:
            try:
                derived = _resolve_value(
                    template,
                    bindings,
                    prior_results[:step_index],
                    prior_arguments[:step_index],
                )
                actual_number = _finite_number(actual)
            except (TypeError, ValueError) as exc:
                mismatches.append(f"{path}: {exc}")
                return
            if not math.isclose(derived, actual_number, rel_tol=1e-4, abs_tol=1e-6):
                mismatches.append(
                    f"{path}: derived={derived!r}, source_argument={actual_number!r}"
                )
            return
        if isinstance(template, dict) and isinstance(actual, dict):
            for key, child in template.items():
                visit(child, actual.get(key), step_index, f"{path}.{key}")
        elif isinstance(template, list) and isinstance(actual, list):
            for position, child in enumerate(template):
                real = actual[position] if position < len(actual) else None
                visit(child, real, step_index, f"{path}[{position}]")

    for index, template_step in enumerate(call_template):
        if index >= len(clean_steps) or template_step.get("tool") != clean_steps[index].get("tool"):
            if _bad_derived_expressions([template_step]) == [] and "$derive" in json.dumps(template_step):
                mismatches.append(f"step {index}: cannot align derived expression to source trace")
            continue
        actual_args = clean_steps[index].get("arguments", {}) or {}
        for name, template_value in (template_step.get("args", {}) or {}).items():
            visit(template_value, actual_args.get(name), index, f"step {index}.{name}")
    return mismatches


def _validate_and_build(
    task: str,
    data: dict[str, Any],
    clean_steps: list[dict[str, Any]],
    log,
) -> SkillRecord | None:
    task_pattern = str(data.get("task_pattern", "")).strip()
    call_template = data.get("call_template") or []
    parameters = [str(p) for p in (data.get("parameters") or [])]
    usage_mode = str(data.get("usage_mode", USAGE_DIRECT))
    abstract_stages = [
        str(item).strip() for item in (data.get("abstract_stages") or [])
        if str(item).strip()
    ]

    if not task_pattern:
        log("缺 task_pattern,跳过")
        return None

    reuse_constraints = _normalize_reuse_constraints(
        data.get("reuse_constraints"), task_pattern, log,
    )
    if reuse_constraints is None:
        return None
    capability_interface = _normalize_capability_interface(
        data.get("capability_interface"), log,
    )
    if capability_interface is None:
        return None

    if usage_mode == USAGE_CONDITIONED_ONLY:
        if not abstract_stages:
            log("Experience lacks abstract_stages; skip")
            return None
        return SkillRecord.new(
            task_pattern=task_pattern,
            source_task=task,
            call_template=[],
            parameters=[],
            interface={},
            example_bindings={},
            replay_expect="",
            cluster_key=make_cluster_key(task_pattern, [], []),
            guideline=str(data.get("guideline", "")),
            usage_mode=usage_mode,
            reuse_constraints=reuse_constraints,
            capability_interface=capability_interface,
            abstract_stages=abstract_stages,
        )

    if not call_template:
        log("Executable lacks call_template; skip")
        return None

    call_template = _upgrade_dataflow_refs(call_template, clean_steps, log)

    # 工具名必须是轨迹里出现过的(不允许 LLM 幻觉出新工具)
    trace_tools = {s["tool"] for s in clean_steps}
    tmpl_tools = [str(st.get("tool", "")) for st in call_template]
    bad = [t for t in tmpl_tools if t not in trace_tools]
    if bad:
        log(f"call_template 含轨迹中不存在的工具 {bad},跳过")
        return None

    # 占位符必须 ⊆ parameters
    used = _template_slots(call_template)
    missing = used - set(parameters)
    if missing:
        log(f"模板用到未声明的槽位 {missing},跳过")
        return None
    unused = set(parameters) - used
    if unused:
        log(f"声明槽位未在可执行模板中使用 {unused},跳过")
        return None

    # $from_step 索引必须合法:N < 当前步,且非负
    bad_ref = _bad_from_step_refs(call_template)
    if bad_ref:
        log(f"call_template 含非法 $from_step 引用 {bad_ref},跳过")
        return None

    hybrid_steps = []

    bad_derive = _bad_derived_expressions(call_template)
    if bad_derive:
        log(f"call_template 含非法 $derive 表达式 {bad_derive},跳过")
        return None

    # example_bindings:程序化从真实轨迹捕获(确定性、对长值保真)优先,
    # LLM 复述只作兜底——避免 LLM 复述长路径列表时数量/内容失真导致重放炸。
    llm_bindings = dict(data.get("example_bindings") or {})
    captured = _capture_bindings(call_template, clean_steps, parameters)
    example_bindings = {**llm_bindings, **captured}
    if captured:
        log(f"example_bindings 程序化捕获 {len(captured)}/{len(parameters)} 个槽位: "
            f"{sorted(captured)}")
    still_missing = [p for p in parameters if p not in example_bindings]
    if still_missing:
        log(f"example_bindings 仍缺槽位(程序化+LLM 都没拿到){still_missing}")

    derived_mismatches = _derived_source_mismatches(
        call_template, clean_steps, example_bindings,
    )
    if derived_mismatches:
        log(f"$derive 无法复现源轨迹参数 {derived_mismatches},跳过")
        return None

    return SkillRecord.new(
        task_pattern=task_pattern,
        source_task=task,
        call_template=call_template,
        parameters=parameters,
        interface={str(k): str(v) for k, v in (data.get("interface") or {}).items()},
        example_bindings=example_bindings,
        replay_expect=str(data.get("replay_expect", "")),
        cluster_key=make_cluster_key(task_pattern, call_template, parameters),
        guideline=str(data.get("guideline", "")),
        # 兼容旧编译响应：过去只有“可直接执行”一种候选语义。
        usage_mode=usage_mode,
        reuse_constraints=reuse_constraints,
        capability_interface=capability_interface,
        hybrid_steps=hybrid_steps,
        abstract_stages=abstract_stages,
    )


def _normalize_capability_interface(
    value: Any, log,
) -> CapabilityInterface | None:
    if not isinstance(value, dict) or set(value) != {"inputs", "outputs"}:
        log("capability_interface 缺失或字段不完整,跳过")
        return None

    def normalize(direction: str) -> list[CapabilityPort] | None:
        raw = value.get(direction)
        if not isinstance(raw, list):
            return None
        ports: list[CapabilityPort] = []
        names: set[str] = set()
        for item in raw:
            if not isinstance(item, dict):
                return None
            name = str(item.get("name", "")).strip()
            description = str(item.get("description", "")).strip()
            required = item.get("required")
            if not name or not description or not isinstance(required, bool) or name in names:
                return None
            names.add(name)
            ports.append(CapabilityPort(name, description, required))
        return ports

    inputs = normalize("inputs")
    outputs = normalize("outputs")
    if inputs is None or outputs is None or not outputs:
        log("capability_interface 输入/输出端口非法或没有可用输出,跳过")
        return None
    return CapabilityInterface(inputs=inputs, outputs=outputs)


def _normalize_reuse_constraints(
    value: Any, task_pattern: str, log,
) -> ReuseConstraints | None:
    """Validate declarative reuse boundaries while keeping legacy records loadable."""
    if value is None:
        return ReuseConstraints(semantic_scope=task_pattern, fallback_mode="dynamic")
    if not isinstance(value, dict):
        log("reuse_constraints 不是对象,跳过")
        return None
    allowed = {
        "semantic_scope", "hard_preconditions", "soft_constraints",
        "postconditions", "fallback_mode",
    }
    if set(value) - allowed:
        log(f"reuse_constraints 含未知字段 {sorted(set(value) - allowed)},跳过")
        return None
    scope = value.get("semantic_scope", "")
    fallback = value.get("fallback_mode", "dynamic")
    if not isinstance(scope, str) or not scope.strip():
        log("reuse_constraints.semantic_scope 为空,跳过")
        return None
    if fallback not in {"conditioned", "dynamic"}:
        log(f"非法 fallback_mode {fallback!r},跳过")
        return None
    lists: dict[str, list[str]] = {}
    for name in ("hard_preconditions", "soft_constraints", "postconditions"):
        raw = value.get(name, [])
        if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
            log(f"reuse_constraints.{name} 必须是字符串数组,跳过")
            return None
        lists[name] = [item.strip() for item in raw if item.strip()]
    return ReuseConstraints(
        semantic_scope=scope.strip(),
        hard_preconditions=lists["hard_preconditions"],
        soft_constraints=lists["soft_constraints"],
        postconditions=lists["postconditions"],
        fallback_mode=str(fallback),
    )


def make_cluster_key(
    task_pattern: str,
    call_template: list[dict[str, Any]],
    parameters: list[str],
) -> str:
    """同簇/去重键(借鉴 FlowEvo 的 transfer_cluster_key,按我们的域简化):

        <task_pattern 词集>::<工具调用序列>::<槽位数>

    同一类任务、同一串工具、同样槽位数 → 同簇,准入时据此去重,避免库里堆重复技能。
    """
    from ._util import tokenize
    pat = "_".join(sorted(tokenize(task_pattern)))
    tools = ">".join(str(st.get("tool", "")) for st in call_template)
    return f"{pat}::{tools}::{len(parameters)}"
