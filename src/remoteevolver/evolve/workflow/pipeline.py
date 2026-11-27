"""
remoteevolver.evolve.workflow.pipeline —— 工作流自进化门面(WorkflowEvolver)

把最小闭环的两个钩子封好,供 ReAct / demo 调用:

  plan(task)                      任务开头:检索 + 路由,注入 Skill 或普通 ReAct
  compile_and_admit(task, steps,  成功结尾:编译 → 静态检查 → 注入原题的
                    final)         完整 Agent ReAct 复测 → 入库
  record_outcome(skill_id, mode,  回写统计(use/success/failure + contrastive),供 curation
                 success)

admission(准入)= 一道效果闸:候选 Skill 注入原题后，独立 Agent 必须用
本次新工具证据重新得出正确答案。固定模板重放不是准入闸。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from copy import deepcopy
from pathlib import Path
from ...infra.paths import runtime_path
from typing import Any, Callable

from ...console import safe_print
from ...llm.base import BaseLLM, Message
from . import curation, verifier
from .bank import ScoredSkill, SkillBank
from .compiler import compile_trace
from .config import WorkflowConfig, DEFAULT
from .executor import (
    ToolCaller,
    synthesize,
)
from .record import (
    SkillRecord, STATUS_ACTIVE, STATUS_PRUNED,
    USAGE_CONDITIONED_ONLY, USAGE_DIRECT, USAGE_HYBRID,
)
from .router import choose_route
from .trace_extractor import EffectiveTrace, extract_effective_trace
from .verifier import VerifyContext


def _validate_hybrid_paths(
    arguments: dict[str, Any],
    prior_results: list[str],
    prior_arguments: list[dict[str, Any]],
) -> None:
    """Apply only cheap, task-independent path checks before a real tool call."""
    del prior_results, prior_arguments
    output_root_raw = os.environ.get("EVOAGENT_OUTPUT_DIR", "").strip()
    output_root = Path(output_root_raw).resolve() if output_root_raw else None

    def strings(value: Any):
        if isinstance(value, str):
            yield value
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from strings(item)

    for name, value in arguments.items():
        lowered = name.casefold()
        if not any(marker in lowered for marker in ("path", "file", "dir")):
            continue
        is_output = any(marker in lowered for marker in ("output", "save", "destination"))
        for raw in strings(value):
            candidate = Path(raw).resolve()
            if is_output:
                if output_root is not None:
                    try:
                        candidate.relative_to(output_root)
                    except ValueError as exc:
                        raise ValueError(
                            f"Hybrid 输出参数 {name} 必须位于当前任务产物目录 {output_root}"
                        ) from exc
            elif not candidate.exists():
                raise ValueError(f"Hybrid 输入参数 {name} 使用了不存在的路径: {raw}")




def _compact_replay_value(value: Any, *, depth: int = 0) -> Any:
    """Keep scalar evidence while bounding verbose arrays for answer synthesis."""
    if depth >= 4:
        return "<nested value omitted>"
    if isinstance(value, dict):
        return {
            str(key): _compact_replay_value(item, depth=depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, list):
        if len(value) <= 8:
            return [_compact_replay_value(item, depth=depth + 1) for item in value]
        return {
            "item_count": len(value),
            "head": [
                _compact_replay_value(item, depth=depth + 1)
                for item in value[:3]
            ],
            "tail": [
                _compact_replay_value(item, depth=depth + 1)
                for item in value[-3:]
            ],
            "omitted_middle_items": len(value) - 6,
        }
    return value


def _compact_replay_result(result: str, *, limit: int = 2200) -> str:
    """Prefer structured list compaction to arbitrary character truncation."""
    text = str(result)
    if len(text) <= limit:
        return text
    try:
        compact = json.dumps(
            _compact_replay_value(json.loads(text)),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        compact = text
    if len(compact) <= limit:
        return compact
    head = max(1, (limit * 2) // 3)
    tail = max(1, limit - head - 80)
    return (
        compact[:head]
        + f"\n... <bounded replay evidence; chars={len(compact)}> ...\n"
        + compact[-tail:]
    )


def _skill_answer_contract(skill: SkillRecord) -> str:
    """Render existing semantic fields as an answer contract, not a new format field."""
    contract = {
        "task_pattern": skill.task_pattern,
        "required_outputs": [asdict(port) for port in skill.capability_interface.outputs],
        "postconditions": list(skill.reuse_constraints.postconditions),
        "guideline": skill.guideline,
    }
    return json.dumps(contract, ensure_ascii=False, separators=(",", ":"))


def _labeled_replay_evidence(
    results: list[str], executed_steps: list[dict[str, Any]] | None,
) -> list[str]:
    """Keep each replay result attached to its concrete call semantics."""
    if not executed_steps or len(executed_steps) != len(results):
        return [_compact_replay_result(result) for result in results]
    evidence: list[str] = []
    for index, (step, result) in enumerate(zip(executed_steps, results), 1):
        item = {
            "step": index,
            "tool": str(step.get("tool", "")),
            "resolved_arguments": _compact_replay_value(
                step.get("arguments", {}) or {}
            ),
            "result": _compact_replay_result(result),
        }
        evidence.append(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
    return evidence


def _synthesize_skill_answer(
    task: str, results: list[str], skill: SkillRecord, llm: BaseLLM,
    *, executed_steps: list[dict[str, Any]] | None = None,
) -> str:
    bounded_results = _labeled_replay_evidence(results, executed_steps)
    contracted_task = (
        task.strip()
        + "\n\nExecutable Skill answer contract (semantic requirements, not evidence):\n"
        + _skill_answer_contract(skill)
        + "\nUse current replay evidence to satisfy every applicable required output and "
          "postcondition. Treat each evidence item's tool, resolved_arguments, and "
          "result as one inseparable observation. Infer entity/group identity and "
          "comparison direction from the resolved arguments and task, never from "
          "result position or numeric magnitude. Never copy source-task values from "
          "the contract."
    )
    return synthesize(contracted_task, bounded_results, llm)


def _skill_replay_expected(expected: Any) -> Any:
    """Give Skill admission an isolated judge scope when the oracle supports it."""
    return _scoped_expected(expected, "skill_replay")


def _scoped_expected(expected: Any, scope: str) -> Any:
    """Keep test-only comparisons separate from the task outcome verdict."""
    scoped = getattr(expected, "scoped", None)
    return scoped(scope) if callable(scoped) else expected




def _failure_audit_trace(steps: list[dict[str, Any]]) -> EffectiveTrace:
    """Build a non-destructive failure view without asking an LLM to prune evidence."""
    indexed: list[dict[str, Any]] = []
    original_indices: list[int] = []
    tools: list[str] = []
    for index, raw in enumerate(steps):
        step = deepcopy(raw)
        raw_index = step.get("trace_index", index)
        if not isinstance(raw_index, int) or isinstance(raw_index, bool):
            raw_index = index
        step["trace_index"] = raw_index
        indexed.append(step)
        original_indices.append(raw_index)
        tool = str(step.get("tool", ""))
        if tool:
            tools.append(tool)
    return EffectiveTrace(
        steps=indexed,
        original_indices=original_indices,
        task_family="failed task execution",
        workflow_summary=" → ".join(tools),
        confidence=1.0 if indexed else 0.0,
        error="" if indexed else "原始失败轨迹为空",
    )


@dataclass
class RoutePlan:
    mode: str                              # compatibility adapter for upstream ReAct
    route_kind: str = "direct"             # executable | experience | direct
    skill: SkillRecord | None = None       # 实际要用的技能(dynamic 时为 None)
    relevant_skill: SkillRecord | None = None  # 同簇 top 候选(即便 dynamic 也给,基线用)
    score: float = 0.0
    context_text: str = ""                 # 注入 prompt 的上下文(技能 + 修复提示 + 负面警告)








def _unpack_admission_run(
    value: tuple[str, list[dict[str, Any]]] | tuple[
        str, list[dict[str, Any]], dict[str, Any]
    ],
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    if len(value) == 2:
        answer, steps = value
        return str(answer), list(steps), {}
    answer, steps, metadata = value
    return str(answer), list(steps), dict(metadata or {})


def _assess_tool_errors(
    skill: SkillRecord, steps: list[dict[str, Any]],
) -> dict[str, Any]:
    """Separate occurrence, recovery, and unresolved critical failures.

    Executable required roles come from its verified tool structure. Experience
    has no required tool sequence, so it is not judged by Executable trajectory
    cleanliness; only a trace with errors and no successful evidence at all is
    considered an unrecovered critical failure.
    """
    errors: list[dict[str, Any]] = []
    successful_indices = [
        index for index, step in enumerate(steps) if not bool(step.get("error"))
    ]
    required_tools = {
        str(step.get("tool", "")) for step in skill.call_template
        if str(step.get("tool", ""))
    }
    experience = skill.usage_mode == USAGE_CONDITIONED_ONLY
    unresolved_critical = False
    any_recovered = False
    for index, step in enumerate(steps):
        if not bool(step.get("error")):
            continue
        tool = str(step.get("tool", ""))
        same_tool_later = any(
            later > index
            and not bool(steps[later].get("error"))
            and str(steps[later].get("tool", "")) == tool
            for later in range(index + 1, len(steps))
        )
        alternative_later = any(later > index for later in successful_indices)
        recovered = same_tool_later or alternative_later
        critical = (
            not successful_indices if experience
            else tool in required_tools
        )
        if critical and not recovered:
            unresolved_critical = True
        any_recovered = any_recovered or recovered
        errors.append({
            "step": index,
            "tool": tool,
            "critical": critical,
            "recovered": recovered,
            "same_tool_retry_succeeded": same_tool_later,
            "later_alternative_succeeded": alternative_later and not same_tool_later,
            "error": str(step.get("result", step.get("error", "")))[:1000],
        })
    return {
        "tool_error_occurred": bool(errors),
        "recovered_from_tool_error": any_recovered,
        "unrecovered_critical_tool_error": unresolved_critical,
        "tool_errors": errors,
    }


_INFRASTRUCTURE_ERROR_MARKERS = (
    "api timeout", "request timeout", "request timed out",
    "connection timeout", "connection timed out", "connection reset",
    "connection refused", "rate limit", "too many requests",
    "bad gateway", "gateway timeout", "service unavailable",
    "temporarily unavailable", "network is unreachable",
    "remote end closed", "empty completion",
)


def _is_infrastructure_tool_error(step: dict[str, Any]) -> bool:
    """Exclude transport/service incidents from reusable task experience."""
    kinds = " ".join(
        str(step.get(key, ""))
        for key in ("failure_kind", "error_type", "error_code")
    ).casefold()
    if "infrastructure" in kinds:
        return True
    detail = " ".join(
        str(step.get(key, "")) for key in ("error", "result", "message")
    ).casefold()
    return any(marker in detail for marker in _INFRASTRUCTURE_ERROR_MARKERS)


def _experience_evidence_steps(
    raw_steps: list[dict[str, Any]], effective: EffectiveTrace,
) -> list[dict[str, Any]]:
    """Keep effective successes plus conservatively proven recovery evidence.

    ``resolved_error_indices`` is computed deterministically by the trace
    extractor: a failed call must later succeed with the same tool and resource
    identity. Unresolved detours and infrastructure incidents remain available
    in the audit trace but are not shown to the Experience compiler.
    """
    kept = {
        index for index in effective.original_indices
        if isinstance(index, int) and 0 <= index < len(raw_steps)
    }
    for index in effective.resolved_error_indices:
        if (isinstance(index, int) and 0 <= index < len(raw_steps)
                and not _is_infrastructure_tool_error(raw_steps[index])):
            kept.add(index)
    return [deepcopy(raw_steps[index]) for index in sorted(kept)]


class WorkflowEvolver:
    def __init__(
        self,
        llm: BaseLLM,
        mcp: ToolCaller,
        *,
        bank_dir: Path | str = runtime_path("memory/skills"),
        cfg: WorkflowConfig = DEFAULT,
        judge_llm: BaseLLM | None = None,
        verbose: bool = False,
    ) -> None:
        self._llm = llm
        self._mcp = mcp
        # 裁判 LLM:不传就复用主 LLM。强烈建议传一个**独立模型**,
        # 否则同一模型自判可能产生系统性盲点重叠。
        self._judge_llm = judge_llm  # No implicit main-model self-judging.
        bank_dir = Path(bank_dir)
        self._bank = SkillBank(bank_dir, llm=llm)
        # 失败原因、避坑建议和已验证修复统一保存在同一条 experience 中。
        # 只保存路径，不在主体 Agent 初始化时创建诊断模块或诊断 Memory。
        self._cfg = cfg
        self._verbose = verbose
        self.last_admission_result: dict[str, Any] = {}
        self.last_admission_results: list[dict[str, Any]] = []
        # Installed by the owning Agent. Without a real, isolated ReAct runner,
        # admission must fail closed rather than silently revert to template replay.
        self._admission_runner: Callable[
            [str, SkillRecord | None],
            tuple[str, list[dict[str, Any]]] | tuple[
                str, list[dict[str, Any]], dict[str, Any]
            ],
        ] | None = None

    def set_admission_runner(
        self,
        runner: Callable[
            [str, SkillRecord | None],
            tuple[str, list[dict[str, Any]]] | tuple[
                str, list[dict[str, Any]], dict[str, Any]
            ],
        ],
    ) -> None:
        self._admission_runner = runner

    def admission_context_for(self, skill: SkillRecord) -> str:
        """Model-facing ReAct guidance without source bindings or answers."""
        constraints = skill.reuse_constraints
        lines = [
            f"[candidate Skill] {skill.task_pattern}",
            f"[kind] {'Experience' if skill.usage_mode == USAGE_CONDITIONED_ONLY else 'Executable'}",
            "[rule] Use this Skill only if it fits the current task. Choose and adapt "
            "tools freely using current observations; never copy an old answer.",
        ]
        if skill.guideline:
            lines.append(f"[guidance] {skill.guideline}")
        if skill.usage_mode == USAGE_CONDITIONED_ONLY:
            lines.extend(f"[reasoning] {stage}" for stage in skill.abstract_stages)
        else:
            lines.append(
                "[parameter rule] Choose every argument from this task, the live "
                "tool schema, and current tool observations. Check file/group "
                "identity, units, thresholds, and output paths before each call. "
                "Never reuse source-task argument values or inferred list positions."
            )
            for index, step in enumerate(skill.call_template, 1):
                tool = str(step.get("tool", ""))
                arguments = step.get("args") or {}
                arg_roles = ", ".join(
                    f"{name} <- {_argument_origin_hint(value)}"
                    for name, value in sorted(arguments.items())
                )
                lines.append(
                    f"[verified step {index}] tool={tool}; "
                    f"argument sources={arg_roles or '(none)'}"
                )
            for name, description in skill.interface.items():
                lines.append(f"[semantic input] {name}: {description}")
            for port in skill.capability_interface.outputs:
                lines.append(f"[required output] {port.name}: {port.description}")
        for condition in constraints.hard_preconditions:
            lines.append(f"[precondition] {condition}")
        for condition in constraints.soft_constraints:
            lines.append(f"[applicability] {condition}")
        for condition in constraints.postconditions:
            lines.append(f"[check] {condition}")
        return "\n".join(lines)

    def _judge(self) -> BaseLLM | None:
        return self._judge_llm if self._cfg.judge_enabled else None





    @property
    def bank(self) -> SkillBank:
        return self._bank

    def set_retrieval_context(
        self, question: str, input_names: list[str] | tuple[str, ...],
    ) -> None:
        """Provide exactly the information visible before the ReAct loop."""
        self._bank.set_retrieval_context(question, input_names)

    def _log(self, msg: str) -> None:
        if self._verbose:
            safe_print(f"[workflow] {msg}", flush=True)

    # ---- 钩子 1:任务开头检索 + 路由 ------------------------------------

    def plan(self, task: str) -> RoutePlan:
        route_kind, scored = choose_route(task, self._bank, self._cfg)

        if scored is None:
            self._log(f"route=direct(no matching Skill; bank={self._bank.count()})")
            return RoutePlan(
                mode="dynamic", route_kind="direct",
                context_text="",
            )

        s = scored.skill
        kind = "Experience" if s.usage_mode == USAGE_CONDITIONED_ONLY else "Executable"
        self._log(
            f"route={route_kind} kind={kind} sim={scored.score:.2f} "
            f"skill={s.skill_id} util={s.utility:.2f}"
        )
        used = s if route_kind in {"executable", "experience"} else None
        if route_kind == "executable":
            ctx = self.admission_context_for(s)
        elif route_kind == "experience":
            ctx = _render_context(s)
        else:
            ctx = ""
        # The upstream ReAct loop currently consumes conditioned/dynamic. This
        # is only a transport adapter; decision logic and telemetry use the
        # canonical executable/experience/direct vocabulary.
        execution_mode = "conditioned" if used is not None else "dynamic"
        return RoutePlan(mode=execution_mode, route_kind=route_kind,
                         skill=used, relevant_skill=s,
                         score=scored.score, context_text=ctx)







    def context_for(self, task: str, skill: SkillRecord | None) -> str:
        """On execution fallback, use only a separately admitted Experience Skill."""
        if skill is None or skill.usage_mode != USAGE_CONDITIONED_ONLY:
            skill = self._experience_skill(task)
        skill_context = _render_context(skill) if skill is not None else ""
        return skill_context

    def _experience_skill(self, task: str) -> SkillRecord | None:
        """Reuse only the Experience attached to the already matched source task."""
        candidate = self._bank.experience_for_last_match(
            task, self._cfg.experience_threshold,
        )
        return candidate.skill if candidate is not None else None

    def context_without_skill(self, task: str) -> str:
        """Return empty guidance when no Skill is selected."""
        return ""




    # ---- 钩子 2:成功结尾编译 + 准入 ------------------------------------

    def compile_and_admit(
        self, task: str, steps: list[dict[str, Any]], final: str,
        *, expected: Any = None, externally_verified: bool = False,
        external_verification_reason: str = "",
    ) -> SkillRecord | None:
        """Compile a verified trace, then admit only after candidate-guided ReAct.

        Structural/source checks precede one effect gate: solve the original
        task again with the candidate injected and verify the new answer.

        expected:数据集给的预期输出(现在通常为 None;接训练数据后传进来 → 触发 ORACLE 级)。
        """
        self.last_admission_result = {"stage": "extract", "passed": False}
        self.last_admission_results = []
        effective = extract_effective_trace(task, steps, final, self._llm, success=True)
        if not effective.ok:
            self.last_admission_result["reason"] = "effective_trace_extraction_failed"
            self.last_admission_results = [dict(self.last_admission_result)]
            self._log(f"有效工作流提取失败，拒绝编译污染 Skill: {effective.error}")
            return None
        self._log(
            f"有效工作流保留原步骤 {effective.original_indices}，"
            f"删除 {len(effective.dropped_steps)} 步；{effective.workflow_summary}"
        )
        experience_steps = _experience_evidence_steps(steps, effective)
        self._log(
            "Experience evidence retained "
            f"{len(experience_steps)}/{len(steps)} original steps "
            "(effective successes plus resolved domain errors)"
        )
        return self.compile_effective_and_admit(
            task, effective.steps, final, expected=expected,
            experience_steps=experience_steps,
            externally_verified=externally_verified,
            external_verification_reason=external_verification_reason,
        )


    def compile_effective_and_admit(
        self, task: str, effective_steps: list[dict[str, Any]], final: str,
        *, expected: Any = None, compile_max_attempts: int | None = None,
        experience_steps: list[dict[str, Any]] | None = None,
        externally_verified: bool = False,
        external_verification_reason: str = "",
    ) -> SkillRecord | None:
        """Generate and independently admit Executable and Experience candidates."""
        del externally_verified, external_verification_reason
        self.last_admission_results = []
        admitted: list[SkillRecord] = []
        for kind in ("executable", "experience"):
            candidate_steps = (
                list(experience_steps)
                if kind == "experience" and experience_steps is not None
                else effective_steps
            )
            usage_cursor = (
                self._llm.usage_cursor()
                if callable(getattr(self._llm, "usage_cursor", None)) else 0
            )
            skill = compile_trace(
                task,
                candidate_steps,
                final,
                self._llm,
                target_kind=kind,
                verbose=self._verbose,
                max_attempts=compile_max_attempts,
            )
            compile_usage = (
                self._llm.usage_since(usage_cursor)
                if callable(getattr(self._llm, "usage_since", None)) else {}
            )
            if skill is None:
                result = {
                    "stage": "compile",
                    "passed": False,
                    "kind": kind,
                    "reason": "insufficient_evidence_for_candidate_kind",
                    "compile_usage": compile_usage,
                }
                self.last_admission_results.append(result)
                continue
            accepted = self._admit_candidate(task, skill, expected=expected)
            self.last_admission_result["compile_usage"] = compile_usage
            self.last_admission_results.append(dict(self.last_admission_result))
            if accepted is not None:
                admitted.append(accepted)

        self.last_admission_result = (
            dict(self.last_admission_results[-1])
            if self.last_admission_results else {
                "stage": "compile", "passed": False,
                "reason": "no_candidate_attempted",
            }
        )
        # Upstream ignores the return value during ordinary learning. Preserve
        # both independently.
        return admitted[0] if admitted else None

    def _admit_candidate(
        self, task: str, skill: SkillRecord, *, expected: Any,
    ) -> SkillRecord | None:
        kind = (
            "experience" if skill.usage_mode == USAGE_CONDITIONED_ONLY
            else "executable"
        )
        injected_context = self.admission_context_for(skill)
        candidate_body = skill.to_dict()
        candidate_body["usage_mode"] = kind
        candidate_body["source_task"] = "[source task omitted from Skill body]"
        candidate_body.pop("example_bindings", None)
        candidate_body.pop("replay_expect", None)
        candidate_body.pop("source_replay_answer", None)
        candidate_body.pop("source_replay_reason", None)
        candidate_body.pop("source_replay_steps", None)
        self.last_admission_result = {
            "stage": "admission",
            "passed": False,
            "skill_id": skill.skill_id,
            "kind": kind,
            "candidate": candidate_body,
            "injected_context": injected_context,
            "final_answer": "",
            "tool_steps": [],
            "tool_sequence": [],
            "answer_correct": False,
            "tool_error_occurred": False,
            "recovered_from_tool_error": False,
            "call_count": 0,
            "cost": {
                "amount": None,
                "currency": "USD",
                "reason": "model pricing is not configured; token usage is authoritative",
            },
            "decision_influenced": None,
        }

        def reject(reason: str, **details: Any) -> None:
            self.last_admission_result.update(reason=reason, **details)

        duplicate = self._bank.find_by_cluster(skill.cluster_key)
        if duplicate is not None:
            reject("duplicate_active_cluster", duplicate_skill_id=duplicate.skill_id)
            return None

        if kind == "experience":
            if (skill.call_template or skill.parameters or skill.interface
                    or skill.example_bindings or not skill.abstract_stages):
                reject("experience_contains_executable_structure")
                return None
        elif len(skill.call_template) < self._cfg.min_tool_calls:
            reject(
                "executable_has_insufficient_tool_structure",
                observed_tool_count=len(skill.call_template),
                minimum_tool_count=self._cfg.min_tool_calls,
            )
            return None

        banned = _banned_tool(skill, self._cfg.banned_tool_substrings)
        if banned:
            reject("banned_tool", banned_tool=banned)
            return None
        if expected is None:
            reject("independent_oracle_unavailable")
            return None
        if self._admission_runner is None:
            reject("runner_unavailable")
            return None

        try:
            injected_run = self._admission_runner(task, skill)
            react_answer, react_steps, run_meta = _unpack_admission_run(injected_run)
        except Exception as exc:  # noqa: BLE001
            reject("runner_error", error_type=type(exc).__name__, error_message=str(exc))
            return None

        self.last_admission_result.update(
            final_answer=react_answer,
            tool_steps=react_steps,
            tool_sequence=[str(step.get("tool", "")) for step in react_steps],
            call_count=len(react_steps),
            usage=run_meta.get("usage", {}),
            cost=run_meta.get("cost", self.last_admission_result["cost"]),
        )
        # Gate order is intentional: correctness is decided before inspecting
        # trace cleanliness. A recovered or non-critical error must not hide a
        # correct source-task answer, especially for Experience candidates.
        vr = verifier.verify(
            VerifyContext(
                task=task,
                results=[],
                final=react_answer,
                expected=_skill_replay_expected(expected),
                oracle_final_only=True,
            ),
            judge_llm=self._judge(),
        )
        self.last_admission_result.update(
            answer_correct=bool(
                vr.passed and vr.level >= self._cfg.direct_eligible_min_level
            ),
            level=vr.level_name,
            verifier_reason=vr.reason,
            judge_confidence=vr.confidence,
        )
        if not self.last_admission_result["answer_correct"]:
            reject("answer_verification_failed")
            return None

        error_assessment = _assess_tool_errors(skill, react_steps)
        self.last_admission_result.update(error_assessment)
        if error_assessment["unrecovered_critical_tool_error"]:
            reject("unrecovered_critical_tool_error")
            return None
        if not react_steps:
            reject("no_new_tool_evidence")
            return None

        skill.direct_eligible = kind == "executable"
        skill.execution_enabled = kind == "executable"
        skill.status = STATUS_ACTIVE
        path = self._bank.add(skill)
        self.last_admission_result.update(
            passed=True,
            reason="admitted",
            artifact_path=str(path),
        )
        self._log(
            f"✓ Skill admitted [active·{kind}] {skill.skill_id} "
            f"verification={vr.level_name} -> {path}"
        )
        return skill

    def record_outcome(self, skill_id: str, mode: str, success: bool) -> str | None:
        """回写一次结果并跑 curation。

        mode:
          direct / conditioned —— 实际用了技能(guided)
          unguided             —— 同簇任务这次没用它(基线;用 plan().relevant_skill 的 id)

        返回 curation 触发的 suppress 动作或 None。
        """
        s = self._bank.get(skill_id)
        if s is None:
            return None
        action = curation.record_outcome(s, mode, success, self._cfg)
        self._bank.save(s)
        if action:
            self._log(f"curation: {s.skill_id} → {action}  (status={s.status})")
        return action

    def record_audit_result(
        self, skill_id: str, *, passed: bool, reason: str = "", hard: bool = False,
    ) -> str | None:
        """Apply an explicit, non-LLM artifact audit result.

        Ordinary task outcomes never call this method. A recoverable first audit
        failure disables executable reuse but retains conditioned guidance; a
        second independent failure prunes the artifact. Corruption or safety
        violations may set ``hard=True`` and prune immediately.
        """
        skill = self._bank.get(skill_id)
        if skill is None:
            return None
        if passed:
            skill.audit_failure_count = 0
            if skill.status == STATUS_ACTIVE:
                skill.execution_enabled = (
                    skill.usage_mode in (USAGE_DIRECT, USAGE_HYBRID)
                    and (skill.usage_mode != USAGE_DIRECT or skill.direct_eligible)
                )
            action = "audit:passed"
        else:
            skill.audit_failure_count += 1
            skill.execution_enabled = False
            if hard or skill.audit_failure_count >= self._cfg.prune_audit_fail_threshold:
                skill.status = STATUS_PRUNED
                action = "prune:audit"
            else:
                action = "audit:disable_execution"
        self._bank.save(skill)
        self._log(f"artifact audit: {skill.skill_id} → {action}: {reason[:120]}")
        return action

    # ---- 钩子 3:失败时归因 + 记忆 + 自动修复(步骤8-10) -----------------







# ---- 内部 ---------------------------------------------------------------

def _banned_tool(skill: SkillRecord, banned_substrings) -> str | None:
    """技能里任一工具名命中禁用子串就返回该工具名,否则 None。"""
    for step in skill.call_template:
        tool = str(step.get("tool", "")).lower()
        for b in banned_substrings:
            if b and b in tool:
                return step.get("tool", "")
    return None


def _argument_origin_hint(value: Any) -> str:
    """Describe provenance without exposing recorded source-task values."""
    if isinstance(value, dict):
        prior = value.get("$from_step", value.get("from_step"))
        if isinstance(prior, int) and prior >= 0:
            return f"current output of step {prior + 1}"
        if "$derive" in value:
            return "recompute from current observations"
        nested = {_argument_origin_hint(item) for item in value.values()}
        if len(nested) == 1:
            return nested.pop()
        if nested:
            return "current task and prior tool observations"
    if isinstance(value, list):
        nested = {_argument_origin_hint(item) for item in value}
        if len(nested) == 1:
            return nested.pop()
        if nested:
            return "current task and prior tool observations"
    if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
        slot = value[1:-1]
        if slot and slot.replace("_", "").isalnum():
            return f"current-task slot {slot}"
    return "current task and live tool schema"


def _render_context(skill: SkillRecord) -> str:
    """Render only semantic Experience guidance, never a source tool template."""
    if skill.usage_mode != USAGE_CONDITIONED_ONLY:
        raise ValueError("Executable Skills use admission_context_for for ReAct injection")
    import json
    constraints = skill.reuse_constraints
    lines = [
        (
            f"[relevant Experience Skill] {skill.task_pattern}"
            + (
                f" (scope: {constraints.semantic_scope})"
                if constraints.semantic_scope
                and constraints.semantic_scope.strip() != skill.task_pattern.strip()
                else ""
            )
        ),
        "[kind] Experience",
    ]
    lines.append(
        "[evidence boundary] This is a conditional semantic hint, not a stored "
        "answer or executable recipe. Verify it against the current request, "
        "inputs, schemas, and tool observations; ignore it on any material mismatch."
    )
    if skill.abstract_stages:
        lines.append("[suggested reasoning] " + " -> ".join(skill.abstract_stages))
    if skill.capability_interface.inputs or skill.capability_interface.outputs:
        lines.append(
            "[whole-Skill capability interface] "
            + json.dumps(asdict(skill.capability_interface), ensure_ascii=False)
        )
    if constraints.hard_preconditions:
        lines.append("[hard preconditions] " + "; ".join(constraints.hard_preconditions))
    if constraints.soft_constraints:
        lines.append("[soft constraints] " + "; ".join(constraints.soft_constraints))
    if constraints.postconditions:
        lines.append("[postconditions] " + "; ".join(constraints.postconditions))
    lines.append(f"[failure fallback] {constraints.fallback_mode}")
    if skill.guideline:
        lines.append(f"[guideline] {skill.guideline}")
    return "\n".join(lines)
