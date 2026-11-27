"""Workflow evolution configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkflowConfig:
    retrieve_top_k: int = 3
    # An Executable is injected only for a close same-family match.
    # Candidate recall floor. An Executable still requires the pre-ReAct LLM
    # applicability gate; this score alone never injects it.
    executable_threshold: float = 0.75
    # Unified four-field score: lower matches conservatively remain Direct.
    experience_threshold: float = 0.65

    require_replay_expect: bool = True
    min_tool_calls: int = 1
    banned_tool_substrings: tuple[str, ...] = (
        "delete", "rm_", "shutdown", "drop_", "format_",
    )
    judge_enabled: bool = True
    direct_eligible_min_level: int = 2
    smoothing_alpha: float = 0.5
    contrastive_harm_threshold: float = -0.1
    contrastive_min_guided: int = 5
    contrastive_min_unguided: int = 3
    low_success_min_guided: int = 3
    low_success_rate_threshold: float = 0.3
    suppress_consecutive_failures: int = 3
    prune_audit_fail_threshold: int = 2



DEFAULT = WorkflowConfig()
