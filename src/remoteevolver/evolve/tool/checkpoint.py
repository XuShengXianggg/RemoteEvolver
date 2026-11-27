"""Durable checkpoints for resumable tool evolution.

The checkpoint contains only explicit execution state.  It deliberately does
not ask an LLM to reconstruct a previous plan after an infrastructure failure.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from ...infra.paths import runtime_path
from typing import Any

CHECKPOINT_VERSION = 1
DEFAULT_CHECKPOINT_DIR = runtime_path("tool-checkpoints")


def _root() -> Path:
    configured = os.environ.get("EVOAGENT_EVOLVE_CHECKPOINT_DIR", "").strip()
    return Path(configured) if configured else DEFAULT_CHECKPOINT_DIR


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def capability_resume_key(task: str, capability_id: str) -> str:
    """Stable key tying one unfinished capability to its original task plan."""
    return f"{_digest(task)}:{capability_id}"


def repository_checkpoint_path(resume_key: str) -> Path:
    return _root() / "repositories" / f"{_digest(resume_key)}.json"


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    os.replace(temporary, path)


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def save_repository_checkpoint(
    resume_key: str, *, capability: str, repository: str,
) -> Path:
    path = repository_checkpoint_path(resume_key)
    _atomic_write(path, {
        "version": CHECKPOINT_VERSION,
        "resume_key": resume_key,
        "capability": capability,
        "repository": repository,
    })
    return path


def load_repository_checkpoint(
    resume_key: str, *, capability: str,
) -> str | None:
    value = _read(repository_checkpoint_path(resume_key))
    if not value or value.get("version") != CHECKPOINT_VERSION:
        return None
    if value.get("resume_key") != resume_key:
        return None
    if value.get("capability") != capability:
        return None
    repository = value.get("repository")
    return repository if isinstance(repository, str) and repository else None


def clear_repository_checkpoint(resume_key: str) -> None:
    repository_checkpoint_path(resume_key).unlink(missing_ok=True)
