"""Persistent engineering-verification records and cache validity checks.

These records prove that a wrapper passed an isolated smoke call, deterministic
artifact-health checks, and a fresh-process MCP probe.  They deliberately make
no claim that a synthetic evaluator proved the tool's domain semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from ...infra.paths import runtime_path
import platform
import sys
import uuid
from typing import Any

from .capability_contract import CapabilityContract, fingerprint_json, fingerprint_text


DEFAULT_CERTIFICATION_DIR = runtime_path("tools/.certifications")


@dataclass
class CertificationRecord:
    tool_name: str
    tool_fingerprint: str
    wrapper_fingerprint: str
    schema_fingerprint: str
    capability_contract: dict[str, Any]
    contract_fingerprint: str
    smoke_passed: bool
    invariants_passed: bool
    probe_passed: bool
    status: str
    verified_at: str
    evidence: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    failure_reason: str = ""
    invalidated_reason: str = ""

    @classmethod
    def create(
        cls,
        *,
        tool_name: str,
        tool_code: str,
        wrapper_code: str,
        schema: dict[str, Any],
        contract: CapabilityContract | None,
        smoke_passed: bool,
        invariants_passed: bool,
        probe_passed: bool,
        evidence: dict[str, Any] | None = None,
        failure_reason: str = "",
    ) -> "CertificationRecord":
        passed = smoke_passed and invariants_passed and probe_passed
        status = "verified" if passed else "pending"
        contract_data = contract.to_dict() if contract is not None else {}
        return cls(
            tool_name=tool_name,
            tool_fingerprint=fingerprint_text(tool_code),
            wrapper_fingerprint=fingerprint_text(wrapper_code),
            schema_fingerprint=fingerprint_json(schema),
            capability_contract=contract_data,
            contract_fingerprint=(contract.fingerprint if contract is not None else ""),
            smoke_passed=smoke_passed,
            invariants_passed=invariants_passed,
            probe_passed=probe_passed,
            status=status,
            verified_at=datetime.now(timezone.utc).isoformat(),
            evidence=dict(evidence or {}),
            environment=environment_fingerprint(),
            failure_reason=failure_reason,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CertificationRecord":
        return cls(
            tool_name=str(value.get("tool_name", "")),
            tool_fingerprint=str(value.get("tool_fingerprint", "")),
            wrapper_fingerprint=str(value.get("wrapper_fingerprint", "")),
            schema_fingerprint=str(value.get("schema_fingerprint", "")),
            capability_contract=dict(value.get("capability_contract", {}) or {}),
            contract_fingerprint=str(value.get("contract_fingerprint", "")),
            smoke_passed=bool(value.get("smoke_passed", False)),
            invariants_passed=bool(value.get("invariants_passed", False)),
            probe_passed=bool(value.get("probe_passed", False)),
            status=str(value.get("status", "")),
            verified_at=str(value.get("verified_at", "")),
            evidence=dict(value.get("evidence", {}) or {}),
            environment=dict(value.get("environment", {}) or {}),
            failure_reason=str(value.get("failure_reason", "")),
            invalidated_reason=str(value.get("invalidated_reason", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CertificationCheck:
    valid: bool
    reason: str
    needs_revalidation: bool = True


class CertificationStore:
    def __init__(self, directory: Path | str = DEFAULT_CERTIFICATION_DIR) -> None:
        self.directory = Path(directory)

    def path_for(self, tool_name: str) -> Path:
        safe = tool_name.split("__")[-1]
        if not safe.isidentifier():
            raise ValueError(f"invalid tool name for certification: {tool_name!r}")
        return self.directory / f"{safe}.json"

    def load(self, tool_name: str) -> CertificationRecord | None:
        path = self.path_for(tool_name)
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                return None
            return CertificationRecord.from_dict(value)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None

    def save(self, record: CertificationRecord) -> Path:
        path = self.path_for(record.tool_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(record.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temp.write_text(payload, encoding="utf-8")
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
        return path

    def invalidate(self, tool_name: str, reason: str) -> None:
        record = self.load(tool_name)
        if record is None:
            return
        record.status = "invalidated"
        record.invalidated_reason = reason
        self.save(record)


def check_certification(
    record: CertificationRecord | None,
    *,
    tool_code: str,
    wrapper_code: str,
    schema: dict[str, Any],
    required_contract: CapabilityContract | None = None,
) -> CertificationCheck:
    del required_contract  # capability descriptions do not certify implementations
    if record is None:
        return CertificationCheck(False, "missing certification")
    if record.status != "verified" or not (
        record.smoke_passed and record.invariants_passed and record.probe_passed
    ):
        return CertificationCheck(False, record.invalidated_reason or record.failure_reason or record.status)
    if record.tool_fingerprint != fingerprint_text(tool_code):
        return CertificationCheck(False, "tool code changed")
    if record.wrapper_fingerprint != fingerprint_text(wrapper_code):
        return CertificationCheck(False, "wrapper changed")
    if record.schema_fingerprint != fingerprint_json(schema):
        return CertificationCheck(False, "schema changed")
    if record.environment != environment_fingerprint():
        return CertificationCheck(False, "relevant environment changed")
    return CertificationCheck(True, "valid cached engineering verification", needs_revalidation=False)


def environment_fingerprint() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for module_name in ("numpy", "rasterio"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except Exception:  # noqa: BLE001
            versions[module_name] = "unavailable"
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.system(),
        "dependencies": versions,
    }
