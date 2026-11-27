"""Task-first routing for Executable, Experience and plain ReAct."""

from __future__ import annotations

from .bank import ScoredSkill, SkillBank
from .config import WorkflowConfig, DEFAULT
def choose_route(
    task: str,
    bank: SkillBank,
    cfg: WorkflowConfig = DEFAULT,
) -> tuple[str, ScoredSkill | None]:
    """Check up to three ranked sources and independently gate each Skill kind.

    Source score controls candidate eligibility.  One joint LLM call per source
    returns separate Executable and Experience verdicts.  A rejected or missing
    Top-1 kind falls through to the next source while its score remains above the
    Experience recall floor.
    """
    matches = bank.source_matches(task, top_k=cfg.retrieve_top_k)
    if not matches:
        return "direct", None
    top_relevant = matches[0].relevant
    for match in matches:
        if match.score < cfg.experience_threshold:
            break
        check_executable = (
            match.score >= cfg.executable_threshold
            and match.executable is not None
        )
        check_experience = match.experience is not None
        if not check_executable and not check_experience:
            continue
        executable_ok, experience_ok = bank.applicability_gate(
            task,
            match,
            check_executable=check_executable,
            check_experience=check_experience,
        )
        if executable_ok:
            return "executable", match.executable
        if experience_ok:
            return "experience", match.experience
    return "direct", top_relevant
