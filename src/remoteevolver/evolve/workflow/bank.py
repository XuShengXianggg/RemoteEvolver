"""Persistent Skill bank with task-first, four-field semantic retrieval."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from ...infra.paths import runtime_path
from typing import Any

from ...llm.base import BaseLLM
from .record import SkillRecord, USAGE_CONDITIONED_ONLY, USAGE_DIRECT, USAGE_HYBRID
from .retrieval_profile import (
    EMBEDDING_MODEL, PROFILE_VERSION, embed_documents, field_documents,
    generate_profile, similarity,
)
from ...llm.base import Message, generate_required_tool
from .skill_document import load_skill_document, render_skill_document, skill_name

DEFAULT_BANK_DIR = runtime_path("memory/skills")
_PROFILE_INDEX_VERSION = 1

_JOINT_GATE_TOOL = {
    "type": "function",
    "function": {
        "name": "report_skill_applicability",
        "description": "Independently report whether each present Skill kind fits the new task.",
        "parameters": {
            "type": "object", "additionalProperties": False,
            "required": [
                "executable_applicable", "experience_applicable",
                "executable_confidence", "experience_confidence",
                "executable_reason", "experience_reason",
            ],
            "properties": {
                "executable_applicable": {"type": "boolean"},
                "experience_applicable": {"type": "boolean"},
                "executable_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "experience_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "executable_reason": {"type": "string"},
                "experience_reason": {"type": "string"},
            },
        },
    },
}

_JOINT_GATE_PROMPT = """You are a pre-ReAct applicability checker. Independently judge the present Executable and Experience from one retrieved source. They are optional guidance for a free ReAct run, not fixed replay instructions.

For the Executable, return true only when the new task belongs to the same workflow family: input roles/modalities, essential operations and dependencies, measurement/unit semantics, aggregation/comparison family, parameter roles, and requested output derivation must be compatible enough to materially reduce tool-planning effort.

For the Experience, return true only when its specific reasoning lesson and stated applicability limits are directly useful for the new task. Shared domain vocabulary or a broadly related goal is insufficient. Reject it when the new task has an incompatible input layout, method, scope, reduction, output meaning, or when applying the lesson would require assumptions absent from the new task.

Judge the two kinds separately: rejecting the Executable must not imply accepting the Experience, and accepting one must not determine the other. If a candidate kind is absent, return false with zero confidence for that kind. Treat paths, places, dates, thresholds, radii, object/category names, grouping intervals, and selection direction as substitutions only when the same operation or reasoning role supports them.

Do not solve the task or infer hidden inputs or steps. Use only the supplied new-task statement/input basenames and candidate cards. Call the reporting tool exactly once."""


@dataclass
class ScoredSkill:
    skill: SkillRecord
    score: float
    weighted: float
    field_scores: dict[str, float] = field(default_factory=dict)
    source_key: str = ""


@dataclass
class SourceMatch:
    source_key: str
    score: float
    field_scores: dict[str, float]
    executable: ScoredSkill | None = None
    experience: ScoredSkill | None = None

    @property
    def relevant(self) -> ScoredSkill | None:
        return self.executable or self.experience


class SkillBank:
    """Skill store; ``SKILL.md`` remains authoritative for Skill content."""

    def __init__(self, bank_dir: Path | str = DEFAULT_BANK_DIR, *, llm: BaseLLM | None = None) -> None:
        self._dir = Path(bank_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._llm = llm
        self._skills: dict[str, SkillRecord] = self._load()
        self._profile_path = self._dir / "_retrieval" / "profiles.json"
        self._profiles: dict[str, dict[str, Any]] = self._load_profiles()
        self._question_override: str | None = None
        self._input_names: tuple[str, ...] = ()
        self._query_cache: dict[str, dict[str, Any]] = {}
        self._last_task = ""
        self._last_scored: list[ScoredSkill] = []
        self._last_matches: list[SourceMatch] = []
        self.last_retrieval_diagnostics: dict[str, Any] = {}

    def configure_profiler(self, llm: BaseLLM) -> None:
        self._llm = llm

    def set_retrieval_context(self, question: str, input_names: list[str] | tuple[str, ...]) -> None:
        """Set the visible pre-ReAct boundary for the next task."""
        self._question_override = str(question)
        self._input_names = tuple(Path(str(name)).name for name in input_names)
        self._last_task = ""
        self._last_scored = []
        self._last_matches = []
        self.last_retrieval_diagnostics = {}

    def all(self) -> list[SkillRecord]:
        return list(self._skills.values())

    def get(self, skill_id: str) -> SkillRecord | None:
        return self._skills.get(skill_id)

    def count(self) -> int:
        return len(self._skills)

    def find_by_cluster(self, cluster_key: str, *, only_live: bool = True) -> SkillRecord | None:
        if not cluster_key:
            return None
        for skill in self._skills.values():
            if skill.cluster_key == cluster_key and (not only_live or skill.active):
                return skill
        return None

    def same_cluster(self, cluster_key: str) -> list[SkillRecord]:
        return [s for s in self._skills.values() if s.cluster_key == cluster_key and s.active]

    def same_cluster_all(self, cluster_key: str) -> list[SkillRecord]:
        return [s for s in self._skills.values() if s.cluster_key == cluster_key]

    def best_source_match(self, task: str) -> SourceMatch | None:
        self._scored(task)
        return self._last_matches[0] if self._last_matches else None

    def source_matches(self, task: str, top_k: int = 3) -> list[SourceMatch]:
        """Return ranked source candidates without flattening their Skill kinds."""
        self._scored(task)
        return list(self._last_matches[:top_k])

    @staticmethod
    def _gate_card(candidate: ScoredSkill | None) -> dict[str, Any] | None:
        if candidate is None:
            return None
        record = candidate.skill.to_dict()
        template = []
        for step in record.get("call_template") or []:
            args = step.get("args") if isinstance(step, dict) else {}
            template.append({
                "tool": str(step.get("tool", "")),
                "argument_roles": sorted(str(name) for name in (args or {})),
            })
        return {
            "task_pattern": record.get("task_pattern", ""),
            "guideline": record.get("guideline", ""),
            "abstract_stages": record.get("abstract_stages") or [],
            "reuse_constraints": record.get("reuse_constraints") or {},
            "capability_interface": record.get("capability_interface") or {},
            "verified_tool_structure": template,
        }

    def applicability_gate(
        self,
        task: str,
        match: SourceMatch,
        *,
        check_executable: bool,
        check_experience: bool,
    ) -> tuple[bool, bool]:
        """Judge same-source Executable and Experience independently in one call."""
        if self._llm is None:
            return False, False
        executable = match.executable if check_executable else None
        experience = match.experience if check_experience else None
        payload = {
            "new_task": {
                "question": self._question_override if self._question_override is not None else task,
                "input_basenames": list(self._input_names),
            },
            "source_score": match.score,
            "candidate_executable": self._gate_card(executable),
            "candidate_experience": self._gate_card(experience),
        }
        event: dict[str, Any] = {
            "source_key": match.source_key,
            "source_score": match.score,
            "executable_skill_id": executable.skill.skill_id if executable else "",
            "experience_skill_id": experience.skill.skill_id if experience else "",
            "status": "started",
        }
        try:
            response = generate_required_tool(
                self._llm,
                [Message(role="system", content=_JOINT_GATE_PROMPT),
                 Message(role="user", content=json.dumps(payload, ensure_ascii=False))],
                _JOINT_GATE_TOOL,
            )
            calls = [call for call in response.tool_calls if call.name == "report_skill_applicability"]
            if len(calls) != 1:
                raise ValueError("joint gate did not return exactly one report")
            verdict = calls[0].arguments
            executable_ok = bool(executable) and verdict.get("executable_applicable") is True
            experience_ok = bool(experience) and verdict.get("experience_applicable") is True
            event.update(
                status="accepted" if executable_ok or experience_ok else "rejected",
                executable_applicable=executable_ok,
                experience_applicable=experience_ok,
                executable_confidence=float(verdict.get("executable_confidence", 0.0)),
                experience_confidence=float(verdict.get("experience_confidence", 0.0)),
                executable_reason=str(verdict.get("executable_reason", ""))[:500],
                experience_reason=str(verdict.get("experience_reason", ""))[:500],
            )
            return executable_ok, experience_ok
        except Exception as exc:
            event.update(
                status="error", executable_applicable=False,
                experience_applicable=False, error_type=type(exc).__name__,
                error_message=str(exc)[:500],
            )
            return False, False
        finally:
            self.last_retrieval_diagnostics.setdefault("applicability_gates", []).append(event)
            # Preserve the old diagnostic key for downstream readers during rollout.
            self.last_retrieval_diagnostics["executable_applicability_gate"] = event

    def last_candidates(self, top_k: int = 3) -> list[ScoredSkill]:
        return self._last_scored[:top_k]

    def _scored(self, task: str) -> list[ScoredSkill]:
        if task == self._last_task and self.last_retrieval_diagnostics:
            return list(self._last_scored)
        self._last_task = task
        self._last_scored = []
        self._last_matches = []
        self.last_retrieval_diagnostics = {
            "profile_version": PROFILE_VERSION,
            "embedding_model": EMBEDDING_MODEL,
            "status": "started",
        }
        if self._llm is None:
            self.last_retrieval_diagnostics.update(status="direct", reason="profile_llm_unavailable")
            return []
        active = [skill for skill in self._skills.values() if skill.active]
        question = self._question_override if self._question_override is not None else task
        query_key = self._fingerprint(question, self._input_names)
        try:
            query_record = self._query_cache.get(query_key)
            if query_record is None:
                query_record = {
                    "profile": generate_profile(self._llm, question, self._input_names),
                    "vectors": {}, "question_hash": query_key,
                    "input_names": list(self._input_names),
                }
                self._query_cache[query_key] = query_record
            ignored_without_profile = sum(
                skill.skill_id not in self._profiles for skill in active
            )
            persistent_vectors_missing = any(
                not self._profiles[skill.skill_id].get("vectors", {}).get(field_name)
                for skill in active if skill.skill_id in self._profiles
                for field_name in ("goal", "input", "operation", "output")
            )
            records = [query_record] + [
                self._profiles[skill.skill_id] for skill in active
                if skill.skill_id in self._profiles
            ]
            embedding_usage = self._ensure_vectors(records)
            if persistent_vectors_missing:
                self._save_profiles()
        except Exception as exc:  # retrieval must never abort the task
            self.last_retrieval_diagnostics.update(
                status="direct", reason="profile_or_embedding_error",
                error_type=type(exc).__name__, error_message=str(exc)[:500],
            )
            return []

        grouped: dict[str, list[SkillRecord]] = {}
        for skill in active:
            record = self._profiles.get(skill.skill_id)
            if record is not None:
                grouped.setdefault(str(record["source_key"]), []).append(skill)
        matches: list[SourceMatch] = []
        for source_key, skills in grouped.items():
            record = self._profiles[skills[0].skill_id]
            score, fields = similarity(
                query_record["profile"], record["profile"],
                query_record["vectors"], record["vectors"],
            )
            match = SourceMatch(source_key=source_key, score=score, field_scores=fields)
            for skill in skills:
                item = ScoredSkill(
                    skill=skill, score=score, weighted=score,
                    field_scores=dict(fields), source_key=source_key,
                )
                if self._is_executable(skill):
                    if match.executable is None or skill.utility > match.executable.skill.utility:
                        match.executable = item
                elif skill.usage_mode == USAGE_CONDITIONED_ONLY:
                    if match.experience is None or skill.utility > match.experience.skill.utility:
                        match.experience = item
            if match.relevant is not None:
                matches.append(match)
        matches.sort(
            key=lambda row: (row.score, max(
                row.executable.skill.utility if row.executable else 0.0,
                row.experience.skill.utility if row.experience else 0.0,
            )), reverse=True,
        )
        scored: list[ScoredSkill] = []
        for match in matches:
            scored.extend(item for item in (match.executable, match.experience) if item is not None)
        self._last_matches = matches
        self._last_scored = scored
        self.last_retrieval_diagnostics.update(
            status="ok", query_profile=query_record["profile"],
            candidate_source_count=len(matches), embedding_usage=embedding_usage,
            ignored_skills_without_profile=ignored_without_profile,
            top_matches=[{
                "source_key": row.source_key, "score": row.score,
                "field_scores": row.field_scores,
                "executable_skill_id": row.executable.skill.skill_id if row.executable else "",
                "experience_skill_id": row.experience.skill.skill_id if row.experience else "",
            } for row in matches[:3]],
        )
        return list(scored)

    @staticmethod
    def _is_executable(skill: SkillRecord) -> bool:
        return (
            skill.usage_mode in {USAGE_DIRECT, USAGE_HYBRID}
            and skill.execution_enabled
            and (skill.usage_mode != USAGE_DIRECT or skill.direct_eligible)
        )

    def retrieve_tiers(self, task: str, top_k: int = 3) -> tuple[list[ScoredSkill], list[ScoredSkill]]:
        """Compatibility API; both tiers come from one source-task ranking."""
        executable: list[ScoredSkill] = []
        experience: list[ScoredSkill] = []
        self._scored(task)
        for match in self._last_matches:
            if match.executable is not None and len(executable) < top_k:
                executable.append(match.executable)
            if match.experience is not None and len(experience) < top_k:
                experience.append(match.experience)
        return executable, experience

    def retrieve(self, task: str, top_k: int = 3) -> list[ScoredSkill]:
        return self._scored(task)[:top_k]

    def experience_for_last_match(self, task: str, threshold: float) -> ScoredSkill | None:
        self._scored(task)
        if not self._last_matches:
            return None
        top = self._last_matches[0]
        return top.experience if top.score > threshold else None

    def add(self, skill: SkillRecord) -> Path:
        path = self._save(skill)
        self._skills[skill.skill_id] = skill
        # The source task has just completed, so reuse its pre-ReAct profile.
        if self._last_task == skill.source_task and self._last_task:
            question = self._question_override if self._question_override is not None else skill.source_task
            query_key = self._fingerprint(question, self._input_names)
            record = self._query_cache.get(query_key)
            if record is not None:
                self._profiles[skill.skill_id] = self._profile_entry(
                    skill.source_task, record["profile"], record.get("vectors", {}), self._input_names,
                )
                self._save_profiles()
        return path

    def save(self, skill: SkillRecord) -> Path:
        return self.add(skill)

    def suppress(self, skill_id: str) -> bool:
        skill = self._skills.get(skill_id)
        if not skill:
            return False
        skill.status = "suppressed"
        self._save(skill)
        return True

    @staticmethod
    def _profile_entry(
        source_task: str, profile: dict[str, Any], vectors: dict[str, list[float]],
        input_names: tuple[str, ...],
    ) -> dict[str, Any]:
        return {
            "version": PROFILE_VERSION, "embedding_model": EMBEDDING_MODEL,
            "source_key": SkillBank._source_key(source_task),
            "source_task_hash": hashlib.sha256(source_task.encode("utf-8")).hexdigest(),
            "input_names": list(input_names), "profile": profile, "vectors": vectors,
        }

    @staticmethod
    def _source_key(source_task: str) -> str:
        return hashlib.sha256(source_task.strip().encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _fingerprint(question: str, inputs: tuple[str, ...]) -> str:
        payload = json.dumps([question.strip(), list(inputs)], ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _ensure_vectors(records: list[dict[str, Any]]) -> dict[str, int]:
        pending: list[tuple[dict[str, Any], str, str]] = []
        for record in records:
            vectors = record.setdefault("vectors", {})
            for field_name, document in field_documents(record["profile"]).items():
                if not vectors.get(field_name):
                    pending.append((record, field_name, document))
        if not pending:
            return {}
        vectors, usage = embed_documents([row[2] for row in pending])
        if len(vectors) != len(pending):
            raise RuntimeError("embedding response length mismatch")
        for (record, field_name, _), vector in zip(pending, vectors):
            record["vectors"][field_name] = vector
        return usage

    def _path_of(self, skill_id: str) -> Path:
        skill = self._skills.get(skill_id)
        if skill is None:
            raise KeyError(skill_id)
        return self._dir / skill_name(skill) / "SKILL.md"

    def _save(self, skill: SkillRecord) -> Path:
        document = render_skill_document(skill)
        path = self._dir / skill_name(skill) / "SKILL.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name("SKILL.md.tmp")
        temporary.write_text(document, encoding="utf-8")
        temporary.replace(path)
        return path

    def _save_profiles(self) -> None:
        self._profile_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "index_version": _PROFILE_INDEX_VERSION, "profile_version": PROFILE_VERSION,
            "embedding_model": EMBEDDING_MODEL, "skills": self._profiles,
        }
        temporary = self._profile_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        temporary.replace(self._profile_path)

    def _load_profiles(self) -> dict[str, dict[str, Any]]:
        if not self._profile_path.is_file():
            return {}
        try:
            payload = json.loads(self._profile_path.read_text(encoding="utf-8"))
            if (
                payload.get("index_version") != _PROFILE_INDEX_VERSION
                or payload.get("profile_version") != PROFILE_VERSION
                or payload.get("embedding_model") != EMBEDDING_MODEL
            ):
                return {}
            return {
                str(key): value for key, value in (payload.get("skills") or {}).items()
                if isinstance(value, dict)
            }
        except (json.JSONDecodeError, OSError, TypeError):
            return {}

    def _load(self) -> dict[str, SkillRecord]:
        skills: dict[str, SkillRecord] = {}
        skipped = 0
        for path in sorted(self._dir.glob("*/SKILL.md")):
            try:
                skill = load_skill_document(path)
                skills[skill.skill_id] = skill
            except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
                skipped += 1
        for path in sorted(self._dir.glob("*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                skill = SkillRecord.from_dict(value)
                skills.setdefault(skill.skill_id, skill)
            except (json.JSONDecodeError, KeyError, OSError, TypeError):
                skipped += 1
        if skipped:
            print(f"[workflow] {self._dir}: skipped {skipped} unreadable Skill files")
        return skills
