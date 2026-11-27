"""LLM-guided extraction of the effective workflow from a noisy ReAct trace.

The raw trajectory remains the audit source of truth.  This module creates a
smaller analysis/compilation view while deterministic checks only enforce that
the model's claims refer to real, chronologically valid trace steps.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from ...llm.base import BaseLLM, Message
from ._util import extract_json
from .dataflow import DataflowError, is_tool_error_result, parse_tool_result


@dataclass
class EffectiveTrace:
    steps: list[dict[str, Any]] = field(default_factory=list)
    original_indices: list[int] = field(default_factory=list)
    dropped_steps: list[dict[str, Any]] = field(default_factory=list)
    dependencies: list[tuple[int, int]] = field(default_factory=list)
    final_output_sources: list[int] = field(default_factory=list)
    retention_evidence: dict[int, str] = field(default_factory=dict)
    # 原始轨迹中曾失败、但已被后续成功重试替代的步骤。它们保留在审计/修复
    # 经验中，绝不能进入最终有效工作流或 Skill。
    resolved_error_indices: list[int] = field(default_factory=list)
    task_family: str = ""
    workflow_summary: str = ""
    confidence: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.steps) and not self.error


_EXTRACT_PROMPT = """\
你是工作流轨迹分析器。下面是一次任务的完整 ReAct 工具调用轨迹，其中可能包含探索、
失败重试、重复调用、占位输入以及最终答案未使用的分支。请提取真正参与本次任务求解的
有效执行工作流；不要重新设计一条理想工作流，也不要发明未发生的步骤。
All generated free-text fields, including task_family, workflow_summary, evidence,
and reasons, must be written in English.

原始轨迹是审计事实，压缩结果只是面向编译的因果保真视图。目标不是得到最短轨迹，
而是在不保留失败噪声的前提下，完整保留复现最终答案所需的动作、产物、验证和依赖。

任务结果状态: {outcome}
任务:
{task}

最终回答:
{final}

原始轨迹（步骤编号为 0-based）:
{steps}

要求：
1. kept_steps 按原始执行顺序给出，只能引用真实步骤。
2. 成功任务只保留对最终答案或最终产物有贡献的成功步骤，删除失败尝试、重复探索和
   能够确认未使用的分支。判断贡献时同时检查显式返回值、文件/目录产物、运行环境副作用、
   质量或约束验证，以及下游通过具体路径形成的隐式依赖；不能只看 `$from_step` 引用。
   不得仅因工具名和参数相同就判定重复调用无用；只有证据能确认调用只读、结果等价、
   无副作用且第一次结果未被单独消费时才可删除。证据不足的成功重复调用必须保留。
3. 失败任务要保留实际求解主链和最可能导致失败的关键步骤；症状性重试可放入 dropped_steps。
4. final_output_sources 标出最终回答直接依赖的步骤；无法确定可以为空，但必须相应降低
   confidence，不能默认把最后一步当作答案来源。
5. dependencies 只记录有证据的上游到下游关系，不凭空猜测。轨迹末尾给出的
   observed_dependency_evidence 是代码从显式引用或实际值流中提取的候选证据，应结合
   任务语义核对；它不是要求你机械保留所有步骤。
6. task_family 和 workflow_summary 都使用稳定、通用的任务语义，不包含地区、年份、路径、
   文件名和具体数值；summary 描述动作与产物类型，不复述本次产物名称。
7. 证据不足时降低 confidence，不要为了得到短工作流而强行删除步骤。
8. 先核对任务要求的最终输出目标，而不只看中间产物。阈值判断、聚合、筛选、统计或
   结果表达等确实参与最终答案的步骤必须保留；不能把只产生中间结果的子流程描述成
   完整任务工作流。若轨迹没有执行完成某个目标，不得发明步骤补齐。
9. 对每个保留步骤在 kept_step_evidence 中写明它对最终结果的具体贡献或被哪一步消费；
   “看起来有用”“可能相关”不算证据。删除步骤则说明未使用、失败、被成功重试替代或
   纯探索等可核查理由。
   如果任务未要求某项检查，其结果未被下游或最终回答使用，也没有必要文件/环境副作用，
   必须删除；不得把这种探索包装成“可选验证”留在工作流中。
10. actual_args 是真正传给工具的参数；requested_args 仅用于解释运行时适配，不得用后者
    覆盖前者编译 Skill。assistant_text 只是模型当时的意图说明，不能替代工具结果证据。

只输出一个 JSON：
{{
  "task_family": "多时相遥感指数分析",
  "workflow_summary": "文件发现→指数计算→时相聚合→统计",
  "kept_steps": [0, 2, 4],
  "kept_step_evidence": [
    {{"step": 0, "contribution": "产出被 step 2 消费的输入文件集合"}},
    {{"step": 2, "contribution": "生成 step 4 统计所需的指数产物"}},
    {{"step": 4, "contribution": "直接提供最终回答中的统计值"}}
  ],
  "dropped_steps": [
    {{"step": 1, "reason": "失败尝试"}},
    {{"step": 3, "reason": "输出未被后续步骤或最终答案使用"}}
  ],
  "dependencies": [{{"from": 0, "to": 2}}, {{"from": 2, "to": 4}}],
  "final_output_sources": [4],
  "confidence": 0.8
}}
"""


def extract_effective_trace(
    task: str,
    steps: list[dict[str, Any]],
    final: str,
    llm: BaseLLM,
    *,
    success: bool,
    execution_completed: bool | None = None,
) -> EffectiveTrace:
    """Extract a causally useful sub-trace. Failure is closed for compilation."""
    if not steps:
        return EffectiveTrace(error="原始轨迹为空")
    if execution_completed is None:
        execution_completed = success
    resolved_errors = _resolved_error_indices(steps)
    if success:
        # A verified successful workflow is the authoritative tool skeleton.
        # Remove only explicit failures here.  Whether a successful call was
        # semantically redundant cannot be decided safely from one trace; that
        # can be audited later, but must not make the compiled skeleton incomplete.
        kept = [
            index for index, step in enumerate(steps)
            if not step.get("error") and not is_tool_error_result(step.get("result"))
        ]
        kept, proven_redundant = _prune_locally_proven_redundant_successes(
            steps, kept,
        )
        if not kept:
            return EffectiveTrace(
                error="成功轨迹中没有非报错工具调用",
                resolved_error_indices=sorted(resolved_errors),
            )
        local = {
            "task_family": "verified successful tool workflow",
            "workflow_summary": " → ".join(
                str(steps[index].get("tool", "")) for index in kept
            ),
            "kept_steps": kept,
            "kept_step_evidence": [
                {"step": index, "contribution": "verified successful execution step"}
                for index in kept
            ],
            "dropped_steps": [
                {
                    "step": index,
                    "reason": (
                        "locally proven redundant side-effect-free success"
                        if index in proven_redundant else "explicit tool error"
                    ),
                }
                for index in range(len(steps)) if index not in set(kept)
            ],
            "dependencies": [],
            "final_output_sources": [kept[-1]],
            "confidence": 1.0,
        }
        trace = _validate_extraction(
            local,
            steps,
            success=True,
            execution_completed=True,
            resolved_error_indices=resolved_errors,
        )
        trace.resolved_error_indices = sorted(resolved_errors)
        return trace
    outcome = (
        "任务成功"
        if success else
        "执行已完成但任务验证失败"
        if execution_completed else
        "执行未完成且任务失败"
    )
    prompt = _EXTRACT_PROMPT.format(
        outcome=outcome,
        task=task.strip()[:2000],
        final=(final or "(无)").strip()[:1500],
        steps=_render_trace(steps),
    )
    try:
        response = llm.generate([Message(role="user", content=prompt)], tools=None)
    except Exception as exc:  # noqa: BLE001
        return EffectiveTrace(
            error=f"轨迹提取模型调用失败: {type(exc).__name__}: {exc}",
            resolved_error_indices=sorted(resolved_errors),
        )
    data = extract_json(response.text)
    if data is None:
        repair_prompt = (
            "下面是轨迹提取器刚才的输出，但它不是可解析的单个 JSON object。"
            "不要重新分析任务，不要添加解释或 Markdown；仅把原内容修复为一个合法 JSON，"
            "并保留字段 task_family、workflow_summary、kept_steps、kept_step_evidence、dropped_steps、"
            "dependencies、final_output_sources、confidence。\n\n原输出:\n"
            + response.text[:6000]
        )
        try:
            repaired = llm.generate(
                [Message(role="user", content=repair_prompt)], tools=None,
            )
            data = extract_json(repaired.text)
        except Exception as exc:  # noqa: BLE001
            return EffectiveTrace(
                error=("轨迹提取 JSON 修复调用失败: "
                       f"{type(exc).__name__}: {exc}"),
                resolved_error_indices=sorted(resolved_errors),
            )
        if data is None:
            raw = " ".join(response.text.strip().split())[:500]
            return EffectiveTrace(
                error=f"轨迹提取模型两次均未返回合法 JSON; first_output={raw!r}",
                resolved_error_indices=sorted(resolved_errors),
            )
    trace = _validate_extraction(
        data, steps, success=success,
        execution_completed=execution_completed,
        resolved_error_indices=resolved_errors,
    )
    # 即使模型提炼结果未通过校验，也把确定性识别出的恢复关系交给审计/归因层。
    trace.resolved_error_indices = sorted(resolved_errors)
    return trace


def _prune_locally_proven_redundant_successes(
    steps: list[dict[str, Any]], kept: list[int],
) -> tuple[list[int], set[int]]:
    """Drop only duplicate successes carrying explicit purity evidence.

    Equality alone is never proof: a repeated call may intentionally perform
    the same side effect twice.  The producer must explicitly record
    ``side_effect_free=true`` (directly or under ``execution_evidence``), and
    no later runtime provenance may refer to the earlier occurrence.
    """
    referenced: set[int] = set()
    for step in steps:
        event = step.get("transition_event")
        sources = event.get("argument_sources") if isinstance(event, dict) else None
        for source in (sources or {}).values():
            if isinstance(source, dict) and isinstance(source.get("source_step"), int):
                referenced.add(source["source_step"])

    dropped: set[int] = set()
    prior_by_signature: dict[str, int] = {}
    for index in kept:
        step = steps[index]
        evidence = step.get("execution_evidence") or {}
        pure = step.get("side_effect_free") is True or (
            isinstance(evidence, dict) and evidence.get("side_effect_free") is True
        )
        if not pure:
            continue
        try:
            signature = json.dumps(
                [step.get("tool"), step.get("arguments", step.get("args", {})), step.get("result")],
                ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            )
        except (TypeError, ValueError):
            continue
        prior = prior_by_signature.get(signature)
        if prior is not None and prior not in referenced:
            dropped.add(prior)
        prior_by_signature[signature] = index
    return [index for index in kept if index not in dropped], dropped


def _validate_extraction(
    data: dict[str, Any], raw_steps: list[dict[str, Any]], *, success: bool,
    execution_completed: bool | None = None,
    resolved_error_indices: set[int] | None = None,
) -> EffectiveTrace:
    if execution_completed is None:
        execution_completed = success
    resolved_error_indices = resolved_error_indices or set()
    raw_kept = data.get("kept_steps")
    if not isinstance(raw_kept, list):
        return EffectiveTrace(error="kept_steps 必须为数组")
    kept = sorted({i for i in raw_kept
                   if isinstance(i, int) and not isinstance(i, bool)})
    if not kept or len(kept) != len(raw_kept):
        return EffectiveTrace(error="kept_steps 为空、重复或含非整数编号")
    if kept[0] < 0 or kept[-1] >= len(raw_steps):
        return EffectiveTrace(error="kept_steps 含越界步骤")

    error_indices = {
        i for i, step in enumerate(raw_steps)
        if step.get("error") or is_tool_error_result(step.get("result"))
    }
    if success and error_indices.intersection(kept):
        return EffectiveTrace(error="成功工作流不能保留明确报错步骤")
    # 旧逻辑要求“任务失败就必须保留一个报错步骤”，会把已恢复的重试错误
    # 强塞回有效工作流。现在只禁止已恢复错误进入有效工作流；完整原轨迹仍
    # 交给归因模块并可沉淀为失败/修复经验。
    kept_resolved = resolved_error_indices.intersection(kept)
    if kept_resolved:
        return EffectiveTrace(
            error=f"有效工作流保留了已被成功重试替代的步骤 {sorted(kept_resolved)}"
        )

    kept_set = set(kept)
    ref_error = _validate_explicit_refs(raw_steps, kept_set)
    if ref_error:
        return EffectiveTrace(error=ref_error)

    dependencies: list[tuple[int, int]] = []
    raw_deps = data.get("dependencies") or []
    if isinstance(raw_deps, list):
        for edge in raw_deps:
            if not isinstance(edge, dict):
                continue
            src, dst = edge.get("from"), edge.get("to")
            if (isinstance(src, int) and not isinstance(src, bool)
                    and isinstance(dst, int) and not isinstance(dst, bool)
                    and src in kept_set and dst in kept_set and src < dst):
                dependencies.append((src, dst))
    dependencies = sorted(set(dependencies))

    raw_sources = data.get("final_output_sources") or []
    final_sources = sorted({i for i in raw_sources
                            if isinstance(i, int) and not isinstance(i, bool)
                            and i in kept_set}) if isinstance(raw_sources, list) else []

    retention_evidence: dict[int, str] = {}
    for item in data.get("kept_step_evidence") or []:
        if not isinstance(item, dict):
            continue
        index = item.get("step")
        contribution = str(item.get("contribution", "")).strip()
        if isinstance(index, int) and index in kept_set and contribution:
            retention_evidence[index] = contribution[:300]

    reason_by_step: dict[int, str] = {}
    for item in data.get("dropped_steps") or []:
        if isinstance(item, dict) and isinstance(item.get("step"), int):
            reason_by_step[item["step"]] = str(item.get("reason", "未说明"))[:200]
    dropped = [
        {"step": i, "reason": reason_by_step.get(i, "模型判定不属于有效工作流")}
        for i in range(len(raw_steps)) if i not in kept_set
    ]
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5

    selected: list[dict[str, Any]] = []
    index_map = {old: new for new, old in enumerate(kept)}
    for index in kept:
        step = deepcopy(raw_steps[index])
        step["trace_index"] = index
        for args_name in ("arguments", "args"):
            if args_name in step:
                step[args_name] = _remap_refs(step[args_name], index_map)
        event = step.get("transition_event")
        if isinstance(event, dict):
            remapped_event = deepcopy(event)
            remapped_event["before_step"] = index_map[index]
            raw_after = remapped_event.get("after_step")
            remapped_event["after_step"] = (
                index_map.get(raw_after, -1) if isinstance(raw_after, int) else -1
            )
            for source in (remapped_event.get("argument_sources") or {}).values():
                if not isinstance(source, dict):
                    continue
                raw_source = source.get("source_step")
                if isinstance(raw_source, int):
                    if raw_source in index_map:
                        source["source_step"] = index_map[raw_source]
                    else:
                        source.clear()
                        source.update({"source": "react", "mode": "semantic"})
            step["transition_event"] = remapped_event
        selected.append(step)
    return EffectiveTrace(
        steps=selected,
        original_indices=kept,
        dropped_steps=dropped,
        dependencies=dependencies,
        final_output_sources=final_sources,
        retention_evidence=retention_evidence,
        resolved_error_indices=sorted(resolved_error_indices),
        task_family=str(data.get("task_family", "")).strip()[:160],
        workflow_summary=str(data.get("workflow_summary", "")).strip()[:400],
        confidence=confidence,
    )


_IDENTITY_FIELD_PARTS = (
    "path", "file", "input", "output", "image", "raster", "dir",
    "source", "target", "uri", "url",
)


def _resolved_error_indices(steps: list[dict[str, Any]]) -> set[int]:
    """保守识别“失败后对同一资源成功重试”的步骤。"""
    resolved: set[int] = set()
    for index, step in enumerate(steps):
        if not (step.get("error") or is_tool_error_result(step.get("result"))):
            continue
        for later in steps[index + 1:]:
            if later.get("tool") != step.get("tool"):
                continue
            if later.get("error") or is_tool_error_result(later.get("result")):
                continue
            if _same_retry_target(step, later):
                resolved.add(index)
                break
    return resolved


def _same_retry_target(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """同工具且共享相同文件/路径等资源标识，才视为同一操作的重试。"""
    a = first.get("arguments", first.get("args", {})) or {}
    b = second.get("arguments", second.get("args", {})) or {}
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    shared = set(a).intersection(b)
    identity = [
        key for key in shared
        if any(part in str(key).lower() for part in _IDENTITY_FIELD_PARTS)
    ]
    return any(
        a[key] == b[key] and a[key] not in (None, "", [], {})
        for key in identity
    )


def _validate_explicit_refs(
    raw_steps: list[dict[str, Any]], kept: set[int],
) -> str:
    def walk(value: Any):
        if isinstance(value, dict):
            if "$from_step" in value:
                yield value.get("$from_step")
            for child in value.values():
                yield from walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from walk(child)

    for dst in kept:
        args = raw_steps[dst].get("arguments", raw_steps[dst].get("args", {}))
        for src in walk(args):
            if isinstance(src, int) and src not in kept:
                return f"保留的 step {dst} 显式引用了被删除的 step {src}"
            if isinstance(src, int) and src >= dst:
                return f"step {dst} 含非前向依赖 $from_step={src}"
    return ""


def _remap_refs(value: Any, index_map: dict[int, int]) -> Any:
    """Map original trace indices to compact effective-workflow indices."""
    if isinstance(value, dict):
        remapped = {key: _remap_refs(child, index_map) for key, child in value.items()}
        source = remapped.get("$from_step")
        if isinstance(source, int) and source in index_map:
            remapped["$from_step"] = index_map[source]
        return remapped
    if isinstance(value, list):
        return [_remap_refs(child, index_map) for child in value]
    return value


def _render_trace(steps: list[dict[str, Any]]) -> str:
    observed = _observed_dependency_evidence(steps)
    lines: list[str] = []
    for i, step in enumerate(steps):
        args = json.dumps(
            step.get("arguments", step.get("args", {})) or {},
            ensure_ascii=False,
        )
        requested = step.get("requested_arguments")
        requested_text = (
            json.dumps(requested, ensure_ascii=False)
            if requested is not None else "(same as actual)"
        )
        result = str(step.get("result", ""))
        error = bool(step.get("error") or is_tool_error_result(step.get("result")))
        lines.append(
            f"{i}. tool={step.get('tool', '')} error={error}\n"
            f"   actual_args={args}\n"
            f"   requested_args={requested_text}\n"
            f"   assistant_text={str(step.get('assistant_text', ''))[:600] or '(none)'}\n"
            f"   validation_evidence={json.dumps(step.get('validation_evidence', {}), ensure_ascii=False)[:1600]}\n"
            f"   result_chars={len(result)} result_schema={_result_schema(result)}\n"
            f"   result={_clip_result(result)}"
        )
    if observed:
        lines.append("observed_dependency_evidence:\n  - " + "\n  - ".join(observed))
    return "\n".join(lines)


def _clip_result(result: str, limit: int = 2400) -> str:
    """给压缩模型保留更多结果证据；超长结果同时保留开头、结尾和长度。"""
    if len(result) <= limit:
        return result
    head = int(limit * 0.7)
    tail = limit - head
    return f"{result[:head]}\n... <truncated; chars={len(result)}> ...\n{result[-tail:]}"


def _result_schema(result: str) -> str:
    try:
        fields = parse_tool_result(result)
    except DataflowError:
        return "error-result"
    summary: dict[str, str] = {}
    for key, value in fields.items():
        if isinstance(value, (list, dict)):
            summary[key] = f"{type(value).__name__}[{len(value)}]"
        else:
            summary[key] = type(value).__name__
    return json.dumps(summary, ensure_ascii=False)


def _observed_dependency_evidence(steps: list[dict[str, Any]]) -> list[str]:
    """提取显式引用和真实值相等形成的候选数据流，供 LLM 核对而非替它决策。"""
    evidence: list[str] = []
    parsed: list[dict[str, Any]] = []
    for step in steps:
        try:
            parsed.append(parse_tool_result(str(step.get("result", ""))))
        except DataflowError:
            parsed.append({})

    def visit(value: Any, path: str, dst: int) -> None:
        if isinstance(value, dict):
            source = value.get("$from_step")
            if isinstance(source, int) and 0 <= source < dst:
                selector = value.get("$field") or ("$raw" if value.get("$raw") else "?")
                evidence.append(f"step {source} -> step {dst}: explicit {path} uses {selector}")
                return
            for key, child in value.items():
                visit(child, f"{path}.{key}", dst)
        elif isinstance(value, list):
            for pos, child in enumerate(value):
                visit(child, f"{path}[{pos}]", dst)
        else:
            for source in range(dst):
                matches = [key for key, candidate in parsed[source].items() if candidate == value]
                if len(matches) == 1 and value not in (None, "", False, 0):
                    evidence.append(
                        f"step {source} -> step {dst}: actual {path} equals result field {matches[0]!r}"
                    )

    for dst, step in enumerate(steps):
        visit(step.get("arguments", step.get("args", {})) or {}, "args", dst)
    return list(dict.fromkeys(evidence))[:100]
