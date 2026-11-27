"""
remoteevolver.react —— ReAct 主控

主循环：
  1) 检索历史经验（特别是失败修复策略）→ 渲染 system prompt
  2) 调 LLM；若返回 tool_calls，则逐一通过 MCPClient 调用、把结果回灌到上下文
  3) 没有 tool_calls 时视为完成
  4) 完成（或超步数）后：写一条 execution_record；按 reflection.trigger 触发反思
"""

from __future__ import annotations
from itertools import count

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from jsonschema import validators

from .infra.config import Config
from .llm.base import BaseLLM, Message
from .memory.long_term import LongTermMemory
from .memory.short_term import ShortTermMemory
from .prompt.manager import PromptManager
from .reflection.reflector import Reflector, Trace, TraceStep
from .mcpx import MCPClient
from .base import BaseAgent
from .console import safe_print as print

if TYPE_CHECKING:
    from .evolve.workflow import WorkflowEvolver, RoutePlan
    from .evolve.tool.capability_contract import CapabilityContract
    from .evolve.tool.reliability import ToolReliabilityManager


TOOL_EVOLUTION_REQUEST_NAME = "system__request_tool_evolution"


def _tool_evolution_request_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": TOOL_EVOLUTION_REQUEST_NAME,
            "description": (
                "Request evolution of one minimal reusable domain capability only "
                "after the currently exposed tools have proved insufficient."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "input_description": {"type": "string"},
                    "expected_output": {"type": "string"},
                    "why_existing_tools_are_insufficient": {"type": "string"},
                    "single_operation_reason": {
                        "type": "string",
                        "description": (
                            "Explain why this request is one atomic domain operation, "
                            "not an end-to-end workflow."
                        ),
                    },
                    "excluded_downstream_operations": {
                        "type": "string",
                        "description": (
                            "Name downstream operations deliberately excluded from this "
                            "request because ReAct will handle them in later steps."
                        ),
                    },
                },
                "required": [
                    "operation",
                    "input_description",
                    "expected_output",
                    "why_existing_tools_are_insufficient",
                    "single_operation_reason",
                    "excluded_downstream_operations",
                ],
                "additionalProperties": False,
            },
        },
    }


@dataclass
class _LoopResult:
    """一遍 ReAct 循环的产物（供 run() 编排重试 / 归因 / 编译使用）。"""
    final_text: str
    success: bool
    # ReAct 是否已经正常产出最终回答。它与任务是否通过 oracle 是两个状态：
    # 允许“执行完成，但答案未通过验证”。
    execution_completed: bool = False
    steps: list[TraceStep] = field(default_factory=list)
    steps_dicts: list[dict[str, Any]] = field(default_factory=list)
    last_error: str | None = None
    verify_reason: str = ""
    status: str = "failed"
    requested_capability: dict[str, str] | None = None

    @property
    def steps_count(self) -> int:
        return len(self.steps)


@dataclass
class _HybridReActFrame:
    """Fixed Skill framework enforced inside the main ReAct loop."""

    skill: Any
    targets_by_step: dict[int, set[str]]
    step_index: int = 0
    results: list[str] = field(default_factory=list)
    resolved_arguments: list[dict[str, Any]] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.step_index >= len(self.skill.call_template)

    @property
    def expected_tool(self) -> str:
        if self.done:
            return ""
        return str(self.skill.call_template[self.step_index].get("tool", ""))

    def tool_schema(self, all_schemas: list[dict[str, Any]]) -> dict[str, Any]:
        expected = self.expected_tool
        source: dict[str, Any] | None = None
        for schema in all_schemas:
            if _schema_tool_name(schema) == expected:
                source = schema
                break
        if source is None:
            raise ValueError(f"Hybrid fixed tool is unavailable: {expected}")
        function = source.get("function", source)
        parameters = function.get("parameters") or {}
        properties = parameters.get("properties") or {}
        targets = self.targets_by_step.get(self.step_index, set())
        missing = sorted(targets - set(properties))
        if missing:
            raise ValueError(
                f"Hybrid runtime arguments missing from {expected} schema: {missing}"
            )
        description = str(function.get("description", "")).strip()
        bound_summary = self.runtime_bound_summary()
        if bound_summary:
            description = (
                description
                + "\nRuntime-filled inputs (locked; do not supply them): "
                + bound_summary
            ).strip()
        description = (
            description + " This is the required next tool in the fixed Hybrid Skill."
        ).strip()
        return {
            "type": "function",
            "function": {
                "name": expected,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        name: properties[name] for name in sorted(targets)
                    },
                    "required": sorted(targets),
                    "additionalProperties": False,
                },
            },
        }

    def runtime_bound_summary(self) -> str:
        """Describe locked inputs needed to understand the current call, compactly."""
        if self.done:
            return ""
        from .evolve.workflow.executor import _resolve_value

        targets = self.targets_by_step.get(self.step_index, set())
        template = self.skill.call_template[self.step_index].get("args") or {}
        items: list[str] = []
        for name, expression in template.items():
            if name in targets:
                continue
            try:
                value = _resolve_value(
                    expression, {}, self.results, self.resolved_arguments,
                )
            except Exception:  # noqa: BLE001
                value = expression
            source = _hybrid_binding_source(expression)
            rendered = _compact_argument_value(value)
            items.append(
                f"{name} <- {source} ({rendered})" if source else f"{name}={rendered}"
            )
        return "; ".join(items)

    def normalize_requested_arguments(
        self, tool: str, requested: dict[str, Any],
    ) -> dict[str, Any]:
        """Drop only duplicated runtime-bound arguments from a Hybrid request.

        The model owns ``targets_by_step``.  Arguments explicitly declared as
        automatic in the current Hybrid step are locked and will be resolved
        by the runtime, so a model-supplied copy must never override them.
        Truly unknown arguments remain an error rather than being discarded.
        """
        if self.done:
            raise ValueError("Hybrid fixed framework has already completed")
        if tool != self.expected_tool:
            raise ValueError(
                f"Hybrid requires next tool {self.expected_tool}, got {tool}"
            )
        targets = self.targets_by_step.get(self.step_index, set())
        hybrid_step = next(
            (
                item for item in self.skill.hybrid_steps
                if item.step == self.step_index
            ),
            None,
        )
        locked = (
            set(hybrid_step.automatic_args) - targets
            if hybrid_step is not None else set()
        )
        unknown = set(requested) - targets - locked
        if unknown:
            raise ValueError(
                f"Hybrid {tool} received unknown arguments {sorted(unknown)}; "
                f"runtime arguments are {sorted(targets)} and locked arguments "
                f"are {sorted(locked)}"
            )
        return {
            name: value for name, value in requested.items() if name in targets
        }

    def prepare_call(self, tool: str, requested: dict[str, Any]) -> dict[str, Any]:
        if self.done:
            raise ValueError("Hybrid fixed framework has already completed")
        if tool != self.expected_tool:
            raise ValueError(
                f"Hybrid requires next tool {self.expected_tool}, got {tool}"
            )
        targets = self.targets_by_step.get(self.step_index, set())
        if set(requested) != targets:
            raise ValueError(
                f"Hybrid {tool} requires exactly runtime arguments "
                f"{sorted(targets)}, got {sorted(requested)}"
            )
        from .evolve.workflow.executor import _resolve_value
        from .evolve.workflow.pipeline import _validate_hybrid_paths

        template = self.skill.call_template[self.step_index].get("args") or {}
        fixed = {
            name: _resolve_value(
                value, {}, self.results, self.resolved_arguments,
            )
            for name, value in template.items() if name not in targets
        }
        merged = {**fixed, **requested}
        _validate_hybrid_paths(merged, self.results, self.resolved_arguments)
        return merged

    def record_success(self, arguments: dict[str, Any], result: str) -> None:
        self.resolved_arguments.append(dict(arguments))
        self.results.append(str(result))
        self.step_index += 1


@dataclass
class OnDemandEvolutionState:
    requested_capability_fingerprints: set[str] = field(default_factory=set)
    requested_capability_keys: list[str] = field(default_factory=list)
    evolution_attempts: int = 0
    evolution_results: list[dict[str, Any]] = field(default_factory=list)
    registered_tools: list[str] = field(default_factory=list)
    trigger_sources: list[str] = field(default_factory=list)
    react_rerun_count: int = 0
    resume_checkpoints: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_capability_fingerprints": sorted(
                self.requested_capability_fingerprints
            ),
            "requested_capability_keys": list(self.requested_capability_keys),
            "evolution_attempts": self.evolution_attempts,
            "evolution_results": list(self.evolution_results),
            "registered_tools": list(self.registered_tools),
            "trigger_sources": list(self.trigger_sources),
            "react_rerun_count": self.react_rerun_count,
            "resume_checkpoints": list(self.resume_checkpoints),
            "stop_reason": self.stop_reason,
        }


@dataclass
class ReActAgent(BaseAgent):
    llm: BaseLLM
    mcp: MCPClient
    short_term: ShortTermMemory
    long_term: LongTermMemory
    prompt_manager: PromptManager
    reflector: Reflector
    config: Config
    verbose: bool = False    # True 时每步往 stdout 打 LLM 输出 / 工具调用 / 结果
    workflow: "WorkflowEvolver | None" = None   # 工作流自进化(None=关闭,行为同旧版)
    # 桥接工具进化轴:工作流归因为"缺工具"时回调它(capability)->bool 成功与否。
    # None=不桥接(行为同旧版,只打印建议)。成功后调用方会带新工具重试任务。
    evolve_tool_fn: "Callable[[str], bool] | None" = None
    evolve_tool_for_task_fn: "Callable[[str, dict[str, str] | str], bool] | None" = None
    tool_reliability: "ToolReliabilityManager | None" = None
    on_demand_states: dict[str, OnDemandEvolutionState] = field(default_factory=dict)

    def _attempt_on_demand_evolution(
        self,
        task: str,
        capability: dict[str, str] | str,
        *,
        source: str,
    ) -> tuple[bool, str]:
        state = self.on_demand_states.setdefault(task, OnDemandEvolutionState())
        key = _normalized_capability_key(capability)
        fingerprint = hashlib.sha256(key.encode("utf-8")).hexdigest()
        state.trigger_sources.append(source)
        duplicate = fingerprint in state.requested_capability_fingerprints or any(
            key == previous
            or (len(key) >= 4 and key in previous)
            or (len(previous) >= 4 and previous in key)
            for previous in state.requested_capability_keys
        )
        if duplicate:
            state.stop_reason = "duplicate_capability_request"
            return False, "同一能力已经请求或尝试过，拒绝重复进化"
        state.requested_capability_fingerprints.add(fingerprint)
        state.requested_capability_keys.append(key)

        budget = self.config.tools.max_on_demand_evolutions
        if state.evolution_attempts >= budget:
            state.stop_reason = "on_demand_evolution_budget_exhausted"
            return False, f"达到按需工具进化安全预算 {budget}"
        if self.evolve_tool_fn is None and self.evolve_tool_for_task_fn is None:
            state.stop_reason = "tool_evolution_unavailable"
            return False, "未配置工具进化桥接"

        before_names = set(self.mcp.list_names())
        state.evolution_attempts += 1
        description = _capability_for_evolution(capability)
        try:
            if self.evolve_tool_for_task_fn is not None:
                ok = bool(self.evolve_tool_for_task_fn(task, capability))
            else:
                assert self.evolve_tool_fn is not None
                ok = bool(self.evolve_tool_fn(description))
        except Exception as exc:  # infrastructure failures remain explicit
            state.evolution_results.append({
                "source": source,
                "fingerprint": fingerprint,
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
            state.stop_reason = "infrastructure_failure"
            raise
        state.evolution_results.append({
            "source": source,
            "fingerprint": fingerprint,
            "success": ok,
        })
        if not ok:
            state.stop_reason = "tool_evolution_failed"
            return False, "工具进化或注册验证未成功"

        after_names = set(self.mcp.list_names())
        state.registered_tools.extend(sorted(after_names - before_names))
        state.stop_reason = ""
        return True, "工具进化成功"

    def _run_react_with_explicit_gap_handling(
        self,
        task: str,
        expected: Any,
        wf_plan: "RoutePlan | None",
        *,
        resume_checkpoint: dict[str, Any] | None = None,
    ) -> _LoopResult:
        checkpoint = resume_checkpoint
        res = self._react_loop(
            task, expected, wf_plan, resume_checkpoint=resume_checkpoint,
        )
        while res.status == "tool_gap_requested" and res.requested_capability:
            state = self.on_demand_states[task]
            registered_before = len(state.registered_tools)
            ok, reason = self._attempt_on_demand_evolution(
                task, res.requested_capability, source="react_explicit_request",
            )
            if not ok:
                res.success = False
                res.execution_completed = False
                res.status = "failed"
                res.last_error = reason
                res.final_text = reason
                return res
            new_tools = state.registered_tools[registered_before:]
            checkpoint = _build_resume_checkpoint(
                task=task,
                completed_steps=res.steps_dicts,
                gap=res.requested_capability,
                new_tools=new_tools,
            )
            state.resume_checkpoints.append(checkpoint)
            state.react_rerun_count += 1
            self._vprint(
                "[react-on-demand] 新工具已注册，清空自然语言上下文并从缺口检查点续跑"
            )
            res = self._react_loop(
                task, expected, wf_plan, resume_checkpoint=checkpoint,
            )
        return res

    def _vprint(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def run(self, task: str, expected: Any = None) -> str:
        cfg = self.config
        self.on_demand_states[task] = OnDemandEvolutionState()

        # 0) 工作流自进化：检索 + 路由（workflow=None 时整段跳过，行为同旧版）
        #    - direct      命中且够稳 → 直接重放技能，成功即短路返回
        #    - hybrid      固定步骤自动执行，在声明的衔接点由受限 ReAct 生成参数后继续
        #    - conditioned 命中但不够稳 → 把技能作上下文喂进下面的正常循环
        #    - dynamic     不用技能；若有同簇 top 候选，末尾记一条 unguided 基线
        wf = self.workflow
        wf_plan = None
        initial_skill_mode = ""
        initial_skill = None
        initial_relevant_skill = None
        initial_skill_not_applicable = False
        if wf is not None:
            wf_plan = wf.plan(task)
            initial_skill_mode = wf_plan.mode
            initial_skill = wf_plan.skill
            initial_relevant_skill = wf_plan.relevant_skill

        # 1) ReAct 主循环。Hybrid 不再走 WorkflowEvolver 内部的旁路；
        # 固定 Skill 框架由同一个主 ReAct 循环逐步约束和执行。
        res = self._run_react_with_explicit_gap_handling(task, expected, wf_plan)
        initial_hybrid_attempt_failed = False
        if (wf is not None and wf_plan is not None
                and initial_skill_mode == "hybrid"
                and (
                    not res.execution_completed
                    or (expected is not None and not res.success)
                )):
            initial_hybrid_attempt_failed = True
            fallback = initial_skill.reuse_constraints.fallback_mode
            self._vprint(
                f"[workflow] hybrid main-ReAct failed({_clip(res.last_error or '')})"
                f"→ 回退 {fallback}"
            )
            if fallback == "conditioned":
                wf_plan.mode = "conditioned"
                wf_plan.skill = initial_skill
                wf_plan.context_text = wf.context_for(task, initial_skill)
            else:
                wf_plan.mode = "dynamic"
                wf_plan.skill = None
                wf_plan.context_text = wf.context_without_skill(task)
            res = self._run_react_with_explicit_gap_handling(
                task, expected, wf_plan,
            )
        loop_success = res.success   # 本回合 ReAct 自身的成败（技能统计用，不含 repair 功劳）
        final_text = res.final_text

        evolved = bool(self.on_demand_states[task].registered_tools)
        capability_gap_external = False

        execution_completed = res.execution_completed
        # success 的唯一正式含义是“任务成功”。没有 oracle/领域真值时即使执行
        # 完成也只能标 unverified,不能用于成功率、Skill 准入或正向强化。
        task_success = res.success and expected is not None
        task_status = (
            "success" if task_success
            else "unverified" if execution_completed
            else "failure"
        )
        if task in self.on_demand_states and not self.on_demand_states[task].stop_reason:
            self.on_demand_states[task].stop_reason = (
                "completed" if res.success else "react_failed"
            )

        # 3) 长期记忆：success 只记录任务成功；执行完成状态单列到 payload。
        summary = final_text if execution_completed else (res.last_error or "(unknown failure)")
        self.long_term.add_execution_record(
            task=task, success=task_success, summary=summary,
            payload={
                "steps": res.steps_dicts, "evolved": evolved,
                "execution_completed": execution_completed,
                "task_status": task_status,
                "on_demand_evolution": (
                    self.on_demand_states[task].to_dict()
                    if task in self.on_demand_states else None
                ),
            },
        )

        # 4) 反思：已知失败或有真值验证时才写经验。未验证的“跑完了”不作为
        # 成功/失败经验，避免把未知正确性的答案污染长期记忆。
        outcome_known = expected is not None or not execution_completed
        # 工作流失败已由“有效轨迹提取→根因归因→统一经验”处理时，不再让旧反思器
        # 对原始噪声轨迹重复生成一条可能冲突的经验。
        if outcome_known and _should_reflect(
            cfg.reflection.trigger, task_success, steps_count=res.steps_count
        ):
            trace = Trace(task=task, steps=res.steps, final=final_text, success=task_success)
            self.reflector.run(trace)

        #    - 编译在「首见/动态/刚进化新工具」成功时做：conditioned 复用已有技能，再编译只会近重复（库膨胀）
        #    - 进化重试成功属于"靠新工具走出的新链路",值得编译入库
        #    - 回写本回合技能效果：conditioned=guided / dynamic=unguided 基线
        if wf is not None:
            novel = wf_plan is None or wf_plan.mode == "dynamic" or evolved
            if loop_success and expected is not None and res.steps and novel:
                try:
                    wf.compile_and_admit(task, res.steps_dicts, final_text, expected=expected)
                except Exception as e:  # noqa: BLE001
                    self._vprint(f"[workflow] compile_and_admit 失败(忽略): {type(e).__name__}: {e}")
            # 一次任务最多写入一条 Skill 结果。统计的是初始路由尝试：Direct/
            # Hybrid 失败后即使 fallback 成功仍记失败；fallback 不另记一次。
            # 缺工具进化属于外部能力变化，不归责于 Skill。
            if (wf_plan is not None
                    and not (evolved or capability_gap_external
                             or initial_skill_not_applicable)):
                try:
                    if (initial_skill_mode == "direct"
                            and initial_skill is not None):
                        wf.record_outcome(
                            initial_skill.skill_id, "direct", False,
                        )
                    elif (initial_skill_mode == "hybrid"
                          and initial_skill is not None
                          and (initial_hybrid_attempt_failed
                               or expected is not None)):
                        wf.record_outcome(
                            initial_skill.skill_id, "hybrid",
                            not initial_hybrid_attempt_failed and loop_success,
                        )
                    elif (initial_skill_mode == "conditioned"
                          and initial_skill is not None
                          and (expected is not None or not loop_success)):
                        wf.record_outcome(
                            initial_skill.skill_id, "conditioned", loop_success,
                        )
                    elif (initial_skill_mode == "dynamic"
                          and initial_relevant_skill is not None
                          and (expected is not None or not loop_success)):
                        wf.record_outcome(
                            initial_relevant_skill.skill_id, "unguided", loop_success,
                        )
                except Exception as e:  # noqa: BLE001
                    self._vprint(f"[workflow] record_outcome 失败(忽略): {type(e).__name__}: {e}")

        return final_text

    def _react_loop(
        self,
        task: str,
        expected: Any,
        wf_plan: "RoutePlan | None",
        resume_checkpoint: dict[str, Any] | None = None,
    ) -> "_LoopResult":
        """跑一遍完整 ReAct：检索经验 → 渲染 prompt → 主循环 → oracle 强判据。

        抽成独立方法,是为了「缺工具→进化→带新工具重试」时能原样再跑一次。
        每次调用都会重置 short_term 并重新拉一次工具 schema(进化后新增的工具
        通过 mcp.restart() 已经就位,这里能拿到)。
        """
        cfg = self.config

        hybrid_frame: _HybridReActFrame | None = None
        if (wf_plan is not None and wf_plan.mode == "hybrid"
                and wf_plan.skill is not None):
            from .evolve.workflow.executor import slots_in_value

            targets_by_step = {
                item.step: set(item.react_args) for item in wf_plan.skill.hybrid_steps
            }
            # Slot-valued arguments are task-specific too.  Let this same main
            # ReAct conversation fill them instead of starting a separate binding
            # LLM call before the loop.
            for index, template_step in enumerate(wf_plan.skill.call_template):
                for name, value in (template_step.get("args") or {}).items():
                    if slots_in_value(value):
                        targets_by_step.setdefault(index, set()).add(name)
            hybrid_frame = _HybridReActFrame(
                skill=wf_plan.skill,
                targets_by_step=targets_by_step,
            )

        # 检索历史经验（侧重 experience，回退到全量）
        experiences = self.long_term.search_experiences(task, top_k=cfg.memory.long_term.top_k)
        experiences_text = _format_experiences(experiences)

        if (wf_plan is not None and wf_plan.mode in ("hybrid", "conditioned", "dynamic")
                and wf_plan.context_text):
            experiences_text = wf_plan.context_text + "\n\n" + experiences_text

        # 渲染 system + user
        system_text = self.prompt_manager.render("system", experiences=experiences_text)
        self.short_term.clear()
        self.short_term.append(Message(role="system", content=system_text))
        user_text = task
        if resume_checkpoint is not None:
            user_text += "\n\n" + _format_resume_checkpoint(resume_checkpoint)
        self.short_term.append(Message(role="user", content=user_text))

        schemas = self.mcp.schemas(
            whitelist=cfg.tools.whitelist or None,
            blacklist=cfg.tools.blacklist or None,
        )
        trusted_servers = set(cfg.tools.trusted_servers)
        trusted_schemas = [
            schema for schema in schemas
            if _schema_server_name(schema) in trusted_servers
        ]
        candidate_schemas = [
            schema for schema in schemas
            if _schema_server_name(schema) not in trusted_servers
        ]
        always_exposed_names = set(cfg.tools.always_exposed)
        always_exposed_schemas = [
            schema for schema in candidate_schemas
            if _schema_matches_configured_name(schema, always_exposed_names)
        ]
        candidate_schemas = [
            schema for schema in candidate_schemas
            if not _schema_matches_configured_name(schema, always_exposed_names)
        ]
        if self.tool_reliability is None:
            candidate_schemas = []
        else:
            candidate_schemas = self.tool_reliability.eligible_schemas_for_react(
                candidate_schemas, task=task,
            )
        self._vprint(
            f"[tool-selection] ReAct-first exposes {len(candidate_schemas)} "
            "engineering-eligible registered tools"
        )
        schemas = trusted_schemas + always_exposed_schemas + candidate_schemas
        # Do not advertise an action that this Agent is not configured to
        # execute.  In non-tool-evolution arms all existing tools remain
        # available, but requesting a new capability must not be presented as
        # a valid ReAct choice.
        if self.evolve_tool_fn is not None or self.evolve_tool_for_task_fn is not None:
            schemas.append(_tool_evolution_request_schema())
        trusted_tool_names = {
            _schema_tool_name(schema)
            for schema in trusted_schemas + always_exposed_schemas
        }
        if trusted_schemas:
            self._vprint(
                f"[tool-validation] exposing {len(trusted_schemas)} pre-approved trusted tools"
            )
        if always_exposed_schemas:
            self._vprint(
                f"[tool-selection] exposing {len(always_exposed_schemas)} "
                "configured ReAct base tools outside capability selection"
            )
        steps: list[TraceStep] = _checkpoint_trace_steps(resume_checkpoint)
        final_text = ""
        success = False
        last_error: str | None = None
        verify_reason = ""
        hybrid_requested_by_call_id: dict[str, dict[str, Any]] = {}
        hybrid_normalized_call_ids: set[str] = set()

        step_label = "无步数上限" if cfg.agent.max_steps is None else f"上限 {cfg.agent.max_steps} 步"
        print(f"\n┏━━ 🔧 编排阶段 ━ ReAct 推理+工具调用({step_label})"
              + "━" * 16, flush=True)
        step_indices = count() if cfg.agent.max_steps is None else range(cfg.agent.max_steps)
        for step_idx in step_indices:
            if hybrid_frame is not None:
                try:
                    turn_schemas = (
                        [] if hybrid_frame.done
                        else [hybrid_frame.tool_schema(schemas)]
                    )
                except Exception as exc:  # noqa: BLE001
                    return _LoopResult(
                        final_text="", success=False, execution_completed=False,
                        steps=steps,
                        steps_dicts=[_step_to_dict(item) for item in steps],
                        last_error=f"{type(exc).__name__}: {exc}",
                        status="hybrid_failed",
                    )
            else:
                turn_schemas = schemas
            resp = self.llm.generate(
                self.short_term.as_messages(), tools=turn_schemas,
            )
            # Some OpenAI-compatible gateways occasionally return an empty
            # assistant turn after a tool result. It is neither a final answer
            # nor a valid action; consume one step and ask again instead of
            # terminating the workflow with an empty answer.
            if not resp.tool_calls and not (resp.text or "").strip():
                self._vprint(
                    f"[step {step_idx + 1}] empty assistant response; retrying"
                )
                continue

            # 记录 assistant 这一轮发言（含 tool_calls）
            self.short_term.append(Message(
                role="assistant",
                content=resp.text,
                tool_calls=resp.tool_calls,
            ))

            # verbose: LLM 这一轮在想啥
            self._vprint(f"[step {step_idx + 1}] llm: {_clip(resp.text or '(no text)')}")

            if not resp.tool_calls and hybrid_frame is not None and not hybrid_frame.done:
                last_error = (
                    "Hybrid fixed framework is incomplete; next required tool is "
                    + hybrid_frame.expected_tool
                )
                self.short_term.append(Message(
                    role="user",
                    content=(
                        last_error
                        + ". Continue the same ReAct task and call that tool; "
                        "do not provide the final answer yet."
                    ),
                ))
                continue

            if not resp.tool_calls:
                final_text = resp.text or ""
                success = True
                self._vprint(f"  → final answer: {_clip(final_text)}")
                break

            if hybrid_frame is not None:
                if len(resp.tool_calls) != 1:
                    result = (
                        "[tool error] Hybrid fixed framework permits exactly one "
                        f"next tool call: {hybrid_frame.expected_tool}"
                    )
                    last_error = result
                    for tc in resp.tool_calls:
                        self.short_term.append(Message(
                            role="tool", content=result, tool_call_id=tc.id,
                        ))
                        steps.append(TraceStep(
                            tool=tc.name, arguments=tc.arguments, result=result,
                            error=True, assistant_text=resp.text or "",
                            tool_call_id=tc.id,
                        ))
                    continue
                tc = resp.tool_calls[0]
                requested = dict(tc.arguments)
                try:
                    normalized_requested = hybrid_frame.normalize_requested_arguments(
                        tc.name, requested,
                    )
                    action_schema = hybrid_frame.tool_schema(schemas)["function"]["parameters"]
                    validator_cls = validators.validator_for(action_schema)
                    validator_cls.check_schema(action_schema)
                    schema_errors = list(
                        validator_cls(action_schema).iter_errors(normalized_requested)
                    )
                    if schema_errors:
                        raise ValueError(schema_errors[0].message)
                    actual_arguments = hybrid_frame.prepare_call(
                        tc.name, normalized_requested,
                    )
                except Exception as exc:  # noqa: BLE001
                    result = f"[tool error] {type(exc).__name__}: {exc}"
                    last_error = result
                    self.short_term.append(Message(
                        role="tool", content=result, tool_call_id=tc.id,
                    ))
                    steps.append(TraceStep(
                        tool=tc.name, arguments=requested, result=result,
                        error=True, assistant_text=resp.text or "",
                        tool_call_id=tc.id,
                    ))
                    continue
                hybrid_requested_by_call_id[tc.id] = requested
                if normalized_requested != requested:
                    hybrid_normalized_call_ids.add(tc.id)
                # The stored assistant Action must show the real arguments sent
                # after deterministic Skill bindings are merged.
                tc.arguments = actual_arguments

            # 工具调用 —— 逐一执行
            for tc in resp.tool_calls:
                self._vprint(f"  → tool {tc.name}  args={tc.arguments}")
                if tc.name == TOOL_EVOLUTION_REQUEST_NAME:
                    gap, validation_error = _validate_tool_gap_request(tc.arguments)
                    if validation_error:
                        result = f"[tool error] invalid_tool_gap_request: {validation_error}"
                        last_error = result
                        steps.append(TraceStep(
                            tool=tc.name, arguments=tc.arguments, result=result,
                            error=True, assistant_text=resp.text or "",
                            tool_call_id=tc.id,
                        ))
                        self.short_term.append(Message(
                            role="tool", content=result, tool_call_id=tc.id,
                        ))
                        continue
                    steps_dicts = [_step_to_dict(s) for s in steps]
                    return _LoopResult(
                        final_text="",
                        success=False,
                        execution_completed=False,
                        steps=steps,
                        steps_dicts=steps_dicts,
                        last_error="ReAct requested a missing domain capability",
                        status="tool_gap_requested",
                        requested_capability=gap,
                    )
                original_requested_arguments = hybrid_requested_by_call_id.get(
                    tc.id, tc.arguments,
                )
                call_arguments = tc.arguments
                validation_evidence: dict[str, Any] = {}
                reused = _find_reusable_checkpoint_step(
                    resume_checkpoint, tc.name, call_arguments,
                )
                if reused is not None:
                    result = str(reused.get("result", ""))
                    self._vprint(
                        f"     checkpoint reuse: {tc.name} (tool not re-executed)"
                    )
                    result_semantics = str(
                        reused.get("result_semantics", "")
                    ) or _tool_result_semantics(tc.name, schemas)
                    observation = _hybrid_tool_observation(
                        tc.name, call_arguments, result, schemas,
                    )
                    if hybrid_frame is not None:
                        hybrid_frame.record_success(call_arguments, result)
                    if result_semantics:
                        observation = (
                            "[executed tool semantics]\n" + result_semantics
                            + "\n[actual tool result]\n" + result
                        )
                    self.short_term.append(Message(
                        role="tool", content=observation, tool_call_id=tc.id,
                    ))
                    if hybrid_frame is not None:
                        _reset_hybrid_prompt_history(
                            self.short_term, system_text, user_text,
                            hybrid_frame, schemas,
                        )
                    continue
                if self.tool_reliability is not None and tc.name not in trusted_tool_names:
                    pre = self.tool_reliability.before_call(tc.name, call_arguments)
                    validation_evidence["pre_call"] = pre.to_dict()
                    if not pre.passed:
                        result = (
                            f"[tool error] {pre.category}: " + "; ".join(pre.violations)
                        )
                        error_flag = True
                        self._vprint(f"  ← [ERR] {_clip(result)}")
                        last_error = result
                        steps.append(TraceStep(
                            tool=tc.name, arguments=tc.arguments, result=result, error=True,
                            assistant_text=resp.text or "",
                            tool_call_id=tc.id,
                            validation_evidence=validation_evidence,
                        ))
                        self.short_term.append(Message(
                            role="tool", content=result, tool_call_id=tc.id,
                        ))
                        continue
                    call_arguments = pre.adapted_arguments
                from .evolve.workflow.provenance import capture_transition_event
                transition_event = capture_transition_event(task, call_arguments, steps)
                try:
                    result = self.mcp.call(tc.name, call_arguments)
                    from .evolve.workflow.dataflow import is_tool_error_result
                    error_flag = is_tool_error_result(result)
                except Exception as e:  # noqa: BLE001
                    result = f"[tool error] {type(e).__name__}: {e}"
                    error_flag = True

                # 运行时只拦截可确定复现的工程故障：非有限数值、空文件或损坏产物。
                # 公式、领域范围和任务语义交给真实任务执行与最终评价判断。
                if not error_flag:
                    if self.tool_reliability is not None and tc.name not in trusted_tool_names:
                        post = self.tool_reliability.after_call(
                            tc.name, result, arguments=call_arguments,
                        )
                        validation_evidence["post_call"] = post.to_dict()
                        if not post.passed:
                            result = (
                                f"[tool error] {post.category}: " + "; ".join(post.violations)
                            )
                            error_flag = True
                            print(
                                f"     🛡️  契约栏杆拦截 {tc.name} → {_clip(post.violations[0], 110)}",
                                flush=True,
                            )

                if not error_flag:
                    try:
                        from .evolve.tool.invariants import check_tool_result
                        viol = check_tool_result(result)
                    except Exception:  # noqa: BLE001
                        viol = []
                    if viol:
                        validation_evidence["runtime_invariants"] = {
                            "passed": False,
                            "violations": list(viol),
                        }
                        result = ("[tool error] 输出未通过确定性工程健康检查(请勿采用此结果,"
                                  "请修复产物生成或如实说明无法完成):\n  - "
                                  + "\n  - ".join(viol))
                        error_flag = True

                        # 防瞎算关键事件:始终醒目打印(不受 verbose 限制)
                        print(f"     🛡️  运行时栏杆拦截 {tc.name} → {_clip(viol[0], 110)}",
                              flush=True)

                from .evolve.workflow.provenance import attach_output_shape
                transition_event = attach_output_shape(transition_event, result)
                result_semantics = (
                    _tool_result_semantics(tc.name, schemas)
                    if not error_flag else ""
                )

                self._vprint(f"  ← {'[ERR] ' if error_flag else ''}{_clip(result)}")

                if error_flag:
                    last_error = result
                elif hybrid_frame is not None:
                    hybrid_frame.record_success(call_arguments, result)

                steps.append(TraceStep(
                    tool=tc.name,
                    # 编译与重放必须基于真正发给工具的参数，而不是适配前请求。
                    arguments=call_arguments,
                    result=result,
                    error=error_flag,
                    assistant_text=resp.text or "",
                    requested_arguments=(
                        original_requested_arguments
                        if (call_arguments != original_requested_arguments
                            or tc.id in hybrid_normalized_call_ids) else None
                    ),
                    tool_call_id=tc.id,
                    validation_evidence=validation_evidence,
                    result_semantics=result_semantics,
                    transition_event=transition_event,
                ))
                observation = result
                if not error_flag:
                    observation = _hybrid_tool_observation(
                        tc.name, call_arguments, result, schemas,
                    )
                self.short_term.append(Message(
                    role="tool",
                    content=observation,
                    tool_call_id=tc.id,
                ))
                if hybrid_frame is not None and not error_flag:
                    _reset_hybrid_prompt_history(
                        self.short_term, system_text, user_text,
                        hybrid_frame, schemas,
                    )
        else:
            # for-else: 没 break 即超步数
            last_error = last_error or f"reached max_steps={cfg.agent.max_steps} without final answer"
            success = False
            final_text = final_text or last_error

        steps_dicts = [_step_to_dict(s) for s in steps]
        # 必须在 oracle 判分之前保存。否则“执行完成但答案不匹配”会被错误地
        # 当成“执行链未完成”，继而把历史上已经恢复的重试错误视为根因。
        execution_completed = success

        # oracle 强成功判据：给了 expected(数据集预期输出)时,"跑通"还要"答对"。
        # 不匹配则翻成失败 → 上层失败分支会归因(语义错/缺工具)并尝试修复/进化。
        if expected is not None and success:
            from .evolve.workflow.verifier import oracle_match
            if not oracle_match(final_text, expected):
                success = False
                last_error = f"answer mismatch (oracle): got {final_text!r}, expected {expected!r}"
                verify_reason = last_error
                self._vprint(f"[oracle] 答案不匹配 → 判失败: {_clip(last_error)}")

        # 编排轨迹摘要(始终打印)。严格区分“执行完成”和“任务成功”。
        n_block = sum(1 for s in steps
                      if s.error and "运行时健全性校验" in (s.result or ""))
        n_err = sum(1 for s in steps if s.error)
        chain = " → ".join(s.tool for s in steps) if steps else "(无工具调用)"
        if success:
            status = "✓任务成功" if expected is not None else "◇执行完成·任务未验证"
        else:
            status = "✗任务失败" if expected is not None else "✗执行失败"
        print(f"  ▣ 编排摘要: {len(steps)} 步 | 工具链: {chain} | "
              f"错误 {n_err}(栏杆拦截 {n_block}) | {status}", flush=True)

        return _LoopResult(
            final_text=final_text, success=success,
            execution_completed=execution_completed, steps=steps,
            steps_dicts=steps_dicts, last_error=last_error,
            verify_reason=verify_reason,
            status="completed" if execution_completed else "failed",
        )


# ---- helpers ------------------------------------------------------------

def _build_resume_checkpoint(
    *,
    task: str,
    completed_steps: list[dict[str, Any]],
    gap: dict[str, str],
    new_tools: list[str],
) -> dict[str, Any]:
    """Capture only successful execution evidence needed to resume a task."""
    successful = [
        dict(step) for step in completed_steps
        if not step.get("error")
        and not dict(step.get("validation_evidence", {}) or {}).get(
            "checkpoint_reuse", {}
        ).get("reused")
    ]
    return {
        "version": 1,
        "task": task,
        "completed_steps": successful,
        "gap": dict(gap),
        "new_tools": list(new_tools),
    }


def _diagnosed_capability_gap(capability: str) -> dict[str, str]:
    """Represent a diagnosis as a planning hint, not an evolution decision."""
    return {
        "operation": capability,
        "input_description": "reuse the original task inputs and validated prefix outputs",
        "expected_output": "the output required by the unfinished task step",
        "why_existing_tools_are_insufficient": (
            "Diagnose found the previously selected tool semantically insufficient; "
            "ReAct must independently evaluate the currently exposed tools"
        ),
        "single_operation_reason": "one failed workflow capability is being reconsidered",
        "excluded_downstream_operations": "all downstream task steps",
    }


def _format_resume_checkpoint(checkpoint: dict[str, Any]) -> str:
    prompt_checkpoint = dict(checkpoint)
    prompt_steps: list[dict[str, Any]] = []
    for step in checkpoint.get("completed_steps", []):
        if not isinstance(step, dict):
            continue
        item = dict(step)
        result = str(item.get("result", ""))
        if len(result) > 2000:
            item["result"] = result[:2000] + f"...(+{len(result) - 2000} chars)"
        prompt_steps.append(item)
    prompt_checkpoint["completed_steps"] = prompt_steps
    payload = json.dumps(prompt_checkpoint, ensure_ascii=False, sort_keys=True)
    if checkpoint.get("new_tools"):
        guidance = (
            "A tool was registered for the recorded gap. First evaluate the newly "
            "registered tool, but do not assume registration proves semantic correctness."
        )
    else:
        guidance = (
            "Diagnosis identified the recorded capability gap. Use the normal ReAct "
            "tool-selection policy: choose a suitable exposed tool, or call "
            "request_tool_evolution when no exposed tool satisfies the capability."
        )
    return (
        "[REACT_RESUME_CHECKPOINT]\n"
        f"{guidance} Continue from the gap using this structured evidence. "
        "Treat successful completed steps and their outputs as reusable; do not "
        "repeat them unless an artifact is missing or validation fails. First "
        "state a concrete capability, interface, precondition, or certification "
        "reason for rejecting a candidate. Replan only the unfinished step and "
        "downstream steps.\n"
        f"{payload}"
    )


def _checkpoint_trace_steps(
    checkpoint: dict[str, Any] | None,
) -> list[TraceStep]:
    if not checkpoint:
        return []
    restored: list[TraceStep] = []
    for step in checkpoint.get("completed_steps", []):
        if not isinstance(step, dict) or step.get("error"):
            continue
        restored.append(TraceStep(
            tool=str(step.get("tool", "")),
            arguments=dict(step.get("arguments", {}) or {}),
            result=str(step.get("result", "")),
            error=False,
            assistant_text=str(step.get("assistant_text", "")),
            requested_arguments=step.get("requested_arguments"),
            tool_call_id=str(step.get("tool_call_id", "")),
            validation_evidence=dict(step.get("validation_evidence", {}) or {}),
            result_semantics=str(step.get("result_semantics", "")),
            transition_event=dict(step.get("transition_event", {}) or {}),
        ))
    return restored


def _find_reusable_checkpoint_step(
    checkpoint: dict[str, Any] | None,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any] | None:
    """Reuse only an exact, previously successful tool call from this task."""
    if not checkpoint:
        return None
    for step in checkpoint.get("completed_steps", []):
        if (
            isinstance(step, dict)
            and not step.get("error")
            and step.get("tool") == tool_name
            and dict(step.get("arguments", {}) or {}) == dict(arguments)
        ):
            return step
    return None

def _clip(s: str, limit: int = 200) -> str:
    """单行显示 + 截断，verbose 打印用。避免多行 LLM 输出把终端撑爆。

    注意：用纯 ASCII 标记，否则 Windows 默认 GBK 控制台 print 会 UnicodeEncodeError。
    """
    s = (s or "").replace("\n", " \\n ").strip()
    suffix = f"...(+{len(s) - limit} chars)"
    return s if len(s) <= limit else s[:limit] + suffix


def _schema_tool_name(schema: dict[str, Any]) -> str:
    return str(schema.get("function", {}).get("name", schema.get("name", "")))


def _hybrid_tool_observation(
    tool_name: str,
    arguments: dict[str, Any],
    result: str,
    schemas: list[dict[str, Any]],
) -> str:
    """Keep compact call meaning, relevant inputs, and the value in one result card."""
    semantics = _tool_result_semantics(tool_name, schemas)
    inputs = "; ".join(
        f"{name}={_compact_argument_value(value)}"
        for name, value in arguments.items()
    ) or "none"
    return (
        "[tool result card]\n"
        + (semantics or "not separately declared")
        + "\ninputs: " + inputs
        + "\nactual result:\n"
        + str(result)
    )


def _hybrid_binding_source(expression: Any) -> str:
    if not isinstance(expression, dict):
        return ""
    step = expression.get("$from_step")
    if not isinstance(step, int):
        return ""
    if isinstance(expression.get("$arg"), str):
        return f"step {step + 1} argument {expression['$arg']}"
    if isinstance(expression.get("$field"), str):
        return f"step {step + 1} result field {expression['$field']}"
    if expression.get("$raw") is True:
        return f"step {step + 1} raw result"
    return ""


def _compact_argument_value(value: Any, limit: int = 240) -> str:
    """Compact model-facing values while keeping exact values in TraceStep."""
    if isinstance(value, list):
        samples = ", ".join(_compact_argument_value(item, 70) for item in value[:2])
        suffix = ", ..." if len(value) > 2 else ""
        return f"list[{len(value)}]({samples}{suffix})"
    if isinstance(value, dict):
        rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return rendered if len(rendered) <= limit else f"object[{len(value)} keys]"
    if isinstance(value, str):
        text = " ".join(value.split())
        path_hits = re.findall(r"[^,;\s]+\.(?:tif|tiff|img|vrt|nc|json|csv)", text, re.I)
        if len(path_hits) > 2:
            return f"{len(path_hits)} file references"
        if ("/" in text or "\\" in text) and len(text) > 90:
            leaf = re.split(r"[/\\]", text)[-1]
            return f".../{leaf}" if leaf else f"path ({len(text)} chars)"
        return text if len(text) <= limit else text[:limit] + f"... ({len(text)} chars)"
    return json.dumps(value, ensure_ascii=False)


def _compact_tool_result(result: str, limit: int = 700) -> str:
    text = str(result).strip()
    if len(text) <= limit:
        return text
    try:
        parsed = json.loads(text)
    except Exception:  # noqa: BLE001
        parsed = None
    if parsed is not None:
        return _compact_argument_value(parsed, limit)
    path_hits = re.findall(r"[^,;\s]+\.(?:tif|tiff|img|vrt|nc|json|csv)", text, re.I)
    if path_hits:
        return f"long tool output ({len(text)} chars; {len(path_hits)} file references)"
    return f"long tool output ({len(text)} chars)"


def _hybrid_execution_state(
    frame: _HybridReActFrame, schemas: list[dict[str, Any]],
) -> str:
    lines = [
        "[Hybrid execution state]",
        f"Completed {frame.step_index}/{len(frame.skill.call_template)} fixed tools.",
    ]
    for index, (arguments, result) in enumerate(
        zip(frame.resolved_arguments, frame.results), start=1,
    ):
        tool = str(frame.skill.call_template[index - 1].get("tool", ""))
        semantics = _tool_result_semantics(tool, schemas)
        meaning = next(
            (
                line.removeprefix("declared result meaning: ")
                for line in semantics.splitlines()
                if line.startswith("declared result meaning: ")
            ),
            tool,
        )
        arg_text = "; ".join(
            f"{name}={_compact_argument_value(value)}"
            for name, value in arguments.items()
        ) or "none"
        lines.extend([
            f"{index}. {tool}",
            f"   meaning: {meaning[:300]}",
            f"   inputs: {arg_text}",
            f"   result: {_compact_tool_result(result)}",
        ])
    if frame.results and len(frame.results[-1]) > 700 and not frame.done:
        lines.extend([
            "[latest raw result retained for the next semantic step]",
            frame.results[-1],
        ])
    if frame.done:
        lines.append(
            "The fixed framework is complete. Answer the current task from these "
            "actual results and their declared meanings."
        )
    else:
        lines.append(f"Next required tool: {frame.expected_tool}")
    return "\n".join(lines)


def _reset_hybrid_prompt_history(
    short_term: ShortTermMemory,
    system_text: str,
    user_text: str,
    frame: _HybridReActFrame,
    schemas: list[dict[str, Any]],
) -> None:
    """Keep a compact model view; the full audit trace remains in TraceStep."""
    short_term.replace([
        Message(role="system", content=system_text),
        Message(role="user", content=user_text),
        Message(role="user", content=_hybrid_execution_state(frame, schemas)),
    ])


def _tool_result_semantics(
    tool_name: str, schemas: list[dict[str, Any]],
) -> str:
    """Return only declared output meaning; never guess units or predicates."""
    schema = next(
        (item for item in schemas if _schema_tool_name(item) == tool_name),
        {},
    )
    function = schema.get("function", schema) if isinstance(schema, dict) else {}
    description = " ".join(str(function.get("description", "")).split())
    output_schema = (
        function.get("outputSchema")
        or function.get("output_schema")
        or function.get("returns")
    )
    lines = [f"tool: {tool_name}"]
    if description:
        lines.append("declared result meaning: " + description[:500])
    if output_schema:
        lines.append(
            "declared output schema: "
            + json.dumps(
                output_schema, ensure_ascii=False, separators=(",", ":"),
            )[:500]
        )
    return "\n".join(lines)


def _schema_server_name(schema: dict[str, Any]) -> str:
    return _schema_tool_name(schema).split("__", 1)[0]


def _schema_matches_configured_name(
    schema: dict[str, Any], configured_names: set[str],
) -> bool:
    qualified_name = _schema_tool_name(schema)
    local_name = qualified_name.split("__", 1)[-1]
    return qualified_name in configured_names or local_name in configured_names


def _validate_tool_gap_request(
    arguments: dict[str, Any],
) -> tuple[dict[str, str] | None, str]:
    required = (
        "operation",
        "input_description",
        "expected_output",
        "why_existing_tools_are_insufficient",
        "single_operation_reason",
        "excluded_downstream_operations",
    )
    if not isinstance(arguments, dict):
        return None, "arguments must be an object"
    missing = [name for name in required if not str(arguments.get(name, "")).strip()]
    if missing:
        return None, f"missing non-empty fields: {', '.join(missing)}"
    return {name: str(arguments[name]).strip() for name in required}, ""


def _normalized_capability_key(capability: dict[str, str] | str) -> str:
    raw = capability.get("operation", "") if isinstance(capability, dict) else capability
    normalized = str(raw).casefold().replace("_", " ").replace("-", " ")
    return " ".join(re.findall(r"[\w]+", normalized, flags=re.UNICODE))


def _capability_for_evolution(capability: dict[str, str] | str) -> str:
    if isinstance(capability, str):
        return capability.strip()
    return json.dumps(capability, ensure_ascii=False, sort_keys=True)


def _format_experiences(entries) -> str:
    if not entries:
        return "(暂无历史经验)"
    lines: list[str] = []
    for i, e in enumerate(entries, 1):
        p = e.payload
        if e.type == "experience":
            lines.append(
                f"{i}. [signature] {p.get('task_signature','')}\n"
                f"   [failure]   {p.get('failure_mode','')}\n"
                f"   [fix]       {p.get('fix_strategy','')}"
            )
        else:
            lines.append(f"{i}. {e.content}")
    return "\n".join(lines)


def _step_to_dict(s: TraceStep) -> dict[str, Any]:
    value = {
        "tool": s.tool,
        "arguments": s.arguments,
        "result": s.result,
        "error": s.error,
    }
    if s.assistant_text:
        value["assistant_text"] = s.assistant_text
    if s.requested_arguments is not None:
        value["requested_arguments"] = s.requested_arguments
    if s.tool_call_id:
        value["tool_call_id"] = s.tool_call_id
    if s.validation_evidence:
        value["validation_evidence"] = s.validation_evidence
    if s.result_semantics:
        value["result_semantics"] = s.result_semantics
    if s.transition_event:
        value["transition_event"] = s.transition_event
    return value


def _should_reflect(trigger: str, success: bool, steps_count: int) -> bool:
    if trigger == "on_finish":
        return True
    if trigger == "on_failure":
        return not success
    # every_n_steps 在单任务结束时退化为「步数达标就反思」
    return steps_count > 0
