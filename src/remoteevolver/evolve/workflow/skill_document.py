"""Render and parse EvoAgent workflow artifacts as real ``SKILL.md`` files.

The Markdown is the persisted artifact. Its prose makes the learned workflow
inspectable, while the fenced record preserves only state used by routing,
reuse execution, or lifecycle governance. Admission-only evidence is omitted.
"""

from __future__ import annotations
import os

import json
import re
from pathlib import Path
from typing import Any

from .record import (
    SkillRecord,
    USAGE_CONDITIONED_ONLY,
    USAGE_DIRECT,
    USAGE_HYBRID,
)


SCHEMA_VERSION = 1
MAX_SKILL_DOCUMENT_BYTES = int(os.getenv("REMOTEEVOLVER_SKILL_MAX_BYTES", "8192"))
MAX_SOURCE_TASK_SUMMARY_CHARS = 512
_RECORD_START = "<!-- EVOAGENT_SKILL_RECORD_V1 -->"
_RECORD_END = "<!-- /EVOAGENT_SKILL_RECORD_V1 -->"
_RECORD_RE = re.compile(
    re.escape(_RECORD_START)
    + r"\s*```json\s*(\{.*?\})\s*```\s*"
    + re.escape(_RECORD_END),
    re.DOTALL,
)
_NON_NAME = re.compile(r"[^a-z0-9-]+")


_NON_RUNTIME_RECORD_FIELDS = frozenset({
    "example_bindings",
    "replay_expect",
    "source_replay_answer",
    "source_replay_reason",
    "source_replay_steps",
    "compiler_corrections",
    "verification_level",
    "failure_count",
    "last_audit_reason",
    "last_audit_at",
    "supersedes_skill_id",
    "created_at",
})


def skill_name(skill: SkillRecord) -> str:
    """Return a valid, stable Skill name derived from the internal id."""
    raw = skill.skill_id.casefold().replace("_", "-")
    name = _NON_NAME.sub("-", raw).strip("-") or "remoteevolver-workflow"
    if len(name) <= 64:
        return name
    # Skill ids end in an eight-character uniqueness suffix. Preserve it.
    suffix = name.rsplit("-", 1)[-1]
    return f"{name[:55].rstrip('-')}-{suffix}"[:64].rstrip("-")


def render_skill_document(skill: SkillRecord) -> str:
    """Serialize runtime-relevant *skill* state into a portable SKILL.md."""
    if skill.usage_mode == USAGE_CONDITIONED_ONLY:
        return _render_experience_document(skill)
    name = skill_name(skill)
    task_pattern = _one_line(skill.task_pattern) or "a verified learned workflow"
    description = (
        f"Use this verified EvoAgent workflow for same-family tasks requiring "
        f"{task_pattern}. Check current inputs and outputs before adapting its tools."
    )
    workflow = _workflow_lines(skill.call_template)
    constraints = _constraint_lines(skill)
    interface = _interface_lines(skill)
    payload = _record_payload(skill, summarize_source_task=True)
    document = (
        "---\n"
        f"name: {json.dumps(name, ensure_ascii=False)}\n"
        f"description: {json.dumps(description, ensure_ascii=False)}\n"
        "metadata:\n"
        "  remoteevolver-schema-version: 1\n"
        f"  remoteevolver-skill-id: {json.dumps(skill.skill_id, ensure_ascii=False)}\n"
        "  skill-kind: executable\n"
        "---\n\n"
        f"# {task_pattern}\n\n"
        "## Purpose\n\n"
        f"{skill.guideline.strip() or task_pattern}\n\n"
        "## Capability interface\n\n"
        f"{interface}\n\n"
        "## Verified workflow\n\n"
        f"{workflow}\n\n"
        "## Reuse constraints\n\n"
        f"{constraints}\n\n"
        "## Execution contract\n\n"
        "The tool sequence is verified source evidence, not a compulsory replay. "
        "Use its tool roles or adapt the workflow through current ReAct observations. "
        "Resolve task-specific inputs from the current task and check each needed "
        "result before continuing. If a precondition or required output is "
        "unavailable, stop using this reference rather than inventing a result.\n\n"
        f"{_RECORD_START}\n"
        "```json\n"
        f"{payload}\n"
        "```\n"
        f"{_RECORD_END}\n"
    )
    if len(document.encode("utf-8")) <= MAX_SKILL_DOCUMENT_BYTES:
        return document

    # Keep every required section and the complete machine record, but remove
    # duplicated prose when a model supplied unusually verbose descriptions.
    compact_document = _compact_document(
        name=name,
        description=description,
        skill=skill,
        task_pattern=task_pattern,
        payload=payload,
    )
    size = len(compact_document.encode("utf-8"))
    if size <= MAX_SKILL_DOCUMENT_BYTES:
        return compact_document

    # Keep the original source task for retrieval-profile recovery unless the
    # lean runtime record still exceeds the artifact budget. New records also
    # carry a sidecar profile, so this fallback mainly protects legacy banks.
    payload = _record_payload(skill, summarize_source_task=True)
    compact_document = _compact_document(
        name=name,
        description=description,
        skill=skill,
        task_pattern=task_pattern,
        payload=payload,
    )
    size = len(compact_document.encode("utf-8"))
    if size <= MAX_SKILL_DOCUMENT_BYTES:
        return compact_document
    raise ValueError(
        f"Skill machine record exceeds the {MAX_SKILL_DOCUMENT_BYTES}-byte "
        f"artifact limit after runtime-field and source-task compaction: {size} bytes"
    )


def _render_experience_document(skill: SkillRecord) -> str:
    """Render semantic guidance without pretending it is an executable recipe."""
    name = skill_name(skill)
    pattern = _one_line(skill.task_pattern) or "a reusable problem-solving experience"
    description = (
        f"Conditional experience for {pattern}. Use only when the current task "
        "matches its applicability; verify all conclusions from current evidence."
    )
    stages = "\n".join(
        f"{index}. {_one_line(stage)}"
        for index, stage in enumerate(skill.abstract_stages, 1)
        if _one_line(stage)
    ) or "1. Apply the summarized lesson only after checking the current task."

    def build(*, summarize_source_task: bool) -> str:
        payload = _record_payload(skill, summarize_source_task=True)
        return (
            "---\n"
            f"name: {json.dumps(name, ensure_ascii=False)}\n"
            f"description: {json.dumps(description[:300], ensure_ascii=False)}\n"
            "metadata:\n"
            "  remoteevolver-schema-version: 1\n"
            f"  remoteevolver-skill-id: {json.dumps(skill.skill_id, ensure_ascii=False)}\n"
            "  skill-kind: experience\n"
            "---\n\n"
            f"# {pattern[:300]}\n\n"
            "## Experience\n\n"
            f"{skill.guideline.strip() or pattern}\n\n"
            "## Applicability\n\n"
            f"{_constraint_lines(skill)}\n\n"
            "## Suggested reasoning\n\n"
            f"{stages}\n\n"
            "## Guardrail\n\n"
            "This is semantic guidance, not an executable recipe or a stored answer. "
            "Re-check applicability, inputs, tool observations, and the requested "
            "output in the current task. Ignore this experience when any material "
            "condition differs or current evidence contradicts it.\n\n"
            f"{_RECORD_START}\n```json\n{payload}\n```\n{_RECORD_END}\n"
        )

    document = build(summarize_source_task=False)
    if len(document.encode("utf-8")) <= MAX_SKILL_DOCUMENT_BYTES:
        return document
    document = build(summarize_source_task=True)
    size = len(document.encode("utf-8"))
    if size <= MAX_SKILL_DOCUMENT_BYTES:
        return document
    raise ValueError(
        f"Experience Skill exceeds the {MAX_SKILL_DOCUMENT_BYTES}-byte artifact "
        f"limit after compaction: {size} bytes"
    )


def _record_payload(skill: SkillRecord, *, summarize_source_task: bool = False) -> str:
    record = skill.to_dict()
    # Canonical persisted vocabulary. Historical in-memory constants are kept
    # only as a compatibility seam for the upstream checkout.
    record["usage_mode"] = (
        "experience" if skill.usage_mode == USAGE_CONDITIONED_ONLY
        else "executable"
    )
    for name in _NON_RUNTIME_RECORD_FIELDS:
        record.pop(name, None)

    if skill.usage_mode == USAGE_CONDITIONED_ONLY:
        # These values are fixed empty/false by the semantic-only invariant and
        # have no routing or governance role. The loader restores safe defaults.
        for name in (
            "call_template", "parameters", "interface", "hybrid_steps",
            "direct_eligible", "execution_enabled",
        ):
            record.pop(name, None)
    else:
        # SkillRecord.to_dict() currently drops this legacy slot interface,
        # Preserve declared slots for parameter validation after reload.
        record["interface"] = {
            name: str(skill.interface.get(name, ""))
            for name in skill.parameters
            if str(skill.interface.get(name, "")).strip()
        }
    if skill.usage_mode == USAGE_HYBRID:
        # Hybrid reuse is ordinary ReAct constrained by tool roles. Arguments
        # declared as react_args must be inferred from the current task, so the
        # persisted artifact must not retain their source-task values.
        template = record.get("call_template") or []
        for hybrid_step in record.get("hybrid_steps") or []:
            index = hybrid_step.get("step")
            if not isinstance(index, int) or not 0 <= index < len(template):
                continue
            args = template[index].get("args")
            if not isinstance(args, dict):
                continue
            for name in hybrid_step.get("react_args") or []:
                if str(name) in args:
                    args[str(name)] = None
    if summarize_source_task:
        record["source_task"] = _source_task_summary(skill)
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


def _compact_document(
    *, name: str, description: str, skill: SkillRecord,
    task_pattern: str, payload: str,
) -> str:
    return (
        "---\n"
        f"name: {json.dumps(name, ensure_ascii=False)}\n"
        f"description: {json.dumps(description[:240], ensure_ascii=False)}\n"
        "metadata:\n"
        "  remoteevolver-schema-version: 1\n"
        f"  remoteevolver-skill-id: {json.dumps(skill.skill_id, ensure_ascii=False)}\n"
        "  skill-kind: executable\n"
        "---\n\n"
        f"# {task_pattern[:300]}\n\n"
        "## Purpose\n\nSee the bounded machine record below.\n\n"
        "## Capability interface\n\nSee the bounded machine record below.\n\n"
        "## Verified workflow\n\nSee the bounded machine record below.\n\n"
        "## Reuse constraints\n\nSee the bounded machine record below.\n\n"
        "## Execution contract\n\nTreat the verified order as reference evidence; "
        "adapt through current observations and honor applicability checks.\n\n"
        f"{_RECORD_START}\n```json\n{payload}\n```\n{_RECORD_END}\n"
    )


def _source_task_summary(skill: SkillRecord) -> str:
    parts = (
        _one_line(skill.task_pattern),
        _one_line(skill.reuse_constraints.semantic_scope),
        _one_line(skill.guideline),
    )
    unique: list[str] = []
    seen: set[str] = set()
    for part in parts:
        key = part.casefold()
        if part and key not in seen:
            seen.add(key)
            unique.append(part)
    summary = "; ".join(unique) or "reusable learned workflow"
    return summary[:MAX_SOURCE_TASK_SUMMARY_CHARS].rstrip()


def parse_skill_document(text: str) -> SkillRecord:
    """Restore a ``SkillRecord`` exactly from an EvoAgent SKILL.md."""
    match = _RECORD_RE.search(text)
    if match is None:
        raise ValueError("missing EvoAgent machine-readable workflow record")
    value: Any = json.loads(match.group(1))
    if not isinstance(value, dict):
        raise TypeError("EvoAgent workflow record must be a JSON object")
    # One-way migration: new documents expose only executable/experience;
    # historical documents remain readable without restoring old routing.
    canonical = value.get("usage_mode")
    if canonical == "executable":
        value["usage_mode"] = USAGE_DIRECT
    elif canonical == "experience":
        value["usage_mode"] = USAGE_CONDITIONED_ONLY
    return SkillRecord.from_dict(value)


def load_skill_document(path: Path | str) -> SkillRecord:
    return parse_skill_document(Path(path).read_text(encoding="utf-8"))


def _one_line(value: str) -> str:
    return " ".join(str(value or "").split())


def _workflow_lines(template: list[dict[str, Any]]) -> str:
    if not template:
        return "This Skill is guidance-only and has no executable tool sequence."
    lines = []
    for index, step in enumerate(template, 1):
        tool = str(step.get("tool", "")).strip() or "unknown-tool"
        args = step.get("args") or {}
        lines.append(
            f"{index}. Verified use of `{tool}` with argument roles: "
            f"`{', '.join(map(str, args)) or 'no arguments'}`."
        )
    return "\n".join(lines)


def _interface_lines(skill: SkillRecord) -> str:
    lines: list[str] = []
    for label, ports in (
        ("Input", skill.capability_interface.inputs),
        ("Output", skill.capability_interface.outputs),
    ):
        for port in ports:
            required = "required" if port.required else "optional"
            lines.append(f"- {label} `{port.name}` ({required}): {port.description}")
    return "\n".join(lines) or "- No explicit capability ports were recorded."


def _constraint_lines(skill: SkillRecord) -> str:
    value = skill.reuse_constraints
    lines = [
        f"- Semantic scope: {value.semantic_scope or 'same-family capability and inputs'}",
        f"- Fallback mode: `{value.fallback_mode or 'dynamic'}`",
    ]
    for label, items in (
        ("Hard precondition", value.hard_preconditions),
        ("Soft constraint", value.soft_constraints),
        ("Postcondition", value.postconditions),
    ):
        lines.extend(f"- {label}: {item}" for item in items)
    return "\n".join(lines)
