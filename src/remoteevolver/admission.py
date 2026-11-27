"""Isolated original-task ReAct admission for a candidate Skill."""

from __future__ import annotations

from dataclasses import replace
from typing import Any


class _CaptureMemory:
    """Keep source-task answers and replay writeback out of admission."""

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []

    def search_experiences(self, *_args, **_kwargs):
        return []

    def add_execution_record(self, *, payload=None, **_kwargs) -> None:
        self.steps = list((payload or {}).get("steps") or [])


class _NoopReflector:
    def run(self, _trace) -> None:
        return None


class _ForcedSkillWorkflow:
    """One-candidate view: no bank retrieval, fallback, repair, or learning."""

    def __init__(self, inner, skill) -> None:
        self.inner = inner
        self.skill = skill

    def plan(self, _task):
        from remoteevolver.evolve.workflow import RoutePlan

        return RoutePlan(
            mode="conditioned",
            route_kind=(
                "experience"
                if getattr(self.skill, "usage_mode", "") == "conditioned_only"
                else "executable"
            ),
            skill=self.skill,
            relevant_skill=self.skill,
            context_text=self.inner.admission_context_for(self.skill),
        )

    def record_outcome(self, *_args, **_kwargs) -> None:
        return None



class _DirectWorkflow:
    """No-Skill baseline using the same ordinary ReAct entry point."""

    def plan(self, _task):
        from remoteevolver.evolve.workflow import RoutePlan

        return RoutePlan(mode="dynamic", route_kind="direct", context_text="")

    def record_outcome(self, *_args, **_kwargs) -> None:
        return None



def make_admission_runner(owner, workflow):
    """Return a fresh Agent-entry runner sharing only live LLM/tool capabilities."""
    from remoteevolver.memory.short_term import ShortTermMemory

    def run(
        task: str, skill,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        memory = _CaptureMemory()
        impl = owner._impl
        llm = getattr(owner, "_llm", None)
        cursor = (
            llm.usage_cursor()
            if callable(getattr(llm, "usage_cursor", None)) else 0
        )
        trial = replace(
            impl,
            short_term=ShortTermMemory(
                max_messages=impl.config.memory.short_term.max_messages,
            ),
            long_term=memory,
            reflector=_NoopReflector(),
            workflow=(
                _ForcedSkillWorkflow(workflow, skill)
                if skill is not None else _DirectWorkflow()
            ),
            evolve_tool_fn=None,
            evolve_tool_for_task_fn=None,
            on_demand_states={},
        )
        answer = trial.run(task, expected=None)
        usage = (
            llm.usage_since(cursor)
            if callable(getattr(llm, "usage_since", None)) else {}
        )
        return answer, memory.steps, {
            "usage": usage,
            "call_count": len(memory.steps),
            "cost": {
                "amount": None,
                "currency": "USD",
                "reason": "model pricing is not configured; token usage is authoritative",
            },
        }

    return run
