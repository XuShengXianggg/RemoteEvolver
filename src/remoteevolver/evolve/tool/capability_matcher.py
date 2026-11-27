"""Deterministic capability-only checks for tool selection.

Concrete parameter names, JSON/Python types and container encodings are
handled by schema-aware ReAct calls and repository wrappers.  This module only
decides whether the required and offered behaviours describe the same
capability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any, Literal

from .capability_contract import CapabilityContract
from ...llm.base import BaseLLM, Message


@dataclass
class CapabilityMatchResult:
    matched: bool
    confidence: float = 0.0
    covered_requirements: list[str] = field(default_factory=list)
    uncovered_requirements: list[str] = field(default_factory=list)
    hard_conflicts: list[str] = field(default_factory=list)
    reason: str = ""
    operation_judgement: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OperationJudgement:
    """Directional verdict about whether an offered operation covers a requirement."""

    decision: Literal["covers", "incompatible", "uncertain"]
    confidence: float = 0.0
    reason: str = ""
    covered_required_behaviors: tuple[str, ...] = ()
    missing_required_behaviors: tuple[str, ...] = ()
    explicit_conflicts: tuple[str, ...] = ()


@dataclass(frozen=True)
class CapabilityEquivalenceResult:
    """Conservative verdict used only to suppress duplicate evolution."""

    equivalent: bool
    confidence: float = 0.0
    reason: str = ""
    hard_conflicts: tuple[str, ...] = ()


def judge_capability_equivalence(
    required: CapabilityContract,
    offered: CapabilityContract,
    llm: BaseLLM,
    *,
    confidence_threshold: float = 0.8,
) -> CapabilityEquivalenceResult:
    """Judge bidirectional capability equivalence without using tool names.

    Explicit interface-shape conflicts are authoritative. The LLM resolves
    only the remaining operation semantics. Uncertainty permits evolution so
    that a genuinely new capability is not suppressed.
    """
    if required.fingerprint == offered.fingerprint:
        return CapabilityEquivalenceResult(True, 1.0, "identical capability contract")

    conflicts = tuple(_hard_shape_conflicts(required, offered))
    if conflicts:
        return CapabilityEquivalenceResult(
            False, 0.0, "explicit input/output capability conflict", conflicts,
        )
    if not _has_behavioral_evidence(required) or not _has_behavioral_evidence(offered):
        return CapabilityEquivalenceResult(
            False, 0.0, "insufficient contract evidence for duplicate suppression",
        )

    prompt = (
        "Determine whether the two capability contracts are bidirectionally equivalent, "
        "not merely related or useful for the same task. Do not use or infer equivalence "
        "from tool names. Operation, aggregation axis, input cardinality, output kind, "
        "and required artifact must all mean the same thing. A pixel-wise mean raster "
        "and one scalar mean value are not equivalent. Return only JSON: "
        '{"equivalent":false,"confidence":0.0,"reason":""}.\n'
        f"REQUIRED={json.dumps(required.to_dict(), ensure_ascii=False, sort_keys=True)}\n"
        f"OFFERED={json.dumps(offered.to_dict(), ensure_ascii=False, sort_keys=True)}"
    )
    try:
        response = llm.generate([Message(role="user", content=prompt)])
        value = _extract_json_object(response.text or "") or {}
        confidence = max(0.0, min(1.0, float(value.get("confidence", 0.0))))
    except Exception as exc:  # noqa: BLE001 - uncertainty must not block evolution
        return CapabilityEquivalenceResult(
            False, 0.0, f"semantic equivalence judgement unavailable: {type(exc).__name__}",
        )
    semantic_yes = value.get("equivalent") is True
    equivalent = semantic_yes and confidence >= confidence_threshold
    reason = str(value.get("reason", "")).strip()[:500]
    if semantic_yes and not equivalent:
        reason = reason or "semantic equivalence confidence below threshold"
    return CapabilityEquivalenceResult(equivalent, confidence, reason)


def _has_behavioral_evidence(contract: CapabilityContract) -> bool:
    return bool(
        contract.description or contract.formula_or_rule or contract.inputs
        or contract.outputs or contract.artifacts or contract.postconditions
    )


def _hard_shape_conflicts(
    required: CapabilityContract, offered: CapabilityContract,
) -> list[str]:
    conflicts: list[str] = []
    required_input = _field_shape(required.inputs, output=False)
    offered_input = _field_shape(offered.inputs, output=False)
    required_output = _field_shape(required.outputs, output=True)
    offered_output = _field_shape(offered.outputs, output=True)

    if "single_raster" in required_input and "raster_collection" in offered_input:
        conflicts.append("input cardinality differs: single raster versus raster collection")
    if "raster_collection" in required_input and "single_raster" in offered_input:
        conflicts.append("input cardinality differs: raster collection versus single raster")
    for left, right, label in (
        ("scalar", "raster_artifact", "output differs: scalar versus raster artifact"),
        ("scalar", "collection", "output differs: scalar versus collection"),
    ):
        if (left in required_output and right in offered_output) or (
            right in required_output and left in offered_output
        ):
            conflicts.append(label)

    required_artifacts = {item.type.casefold() for item in required.artifacts if item.required}
    offered_artifacts = {item.type.casefold() for item in offered.artifacts if item.required}
    if required_artifacts and offered_artifacts and required_artifacts != offered_artifacts:
        conflicts.append("required artifact types differ")
    return conflicts


def _field_shape(fields: tuple[Any, ...], *, output: bool) -> set[str]:
    shape: set[str] = set()
    for item in fields:
        type_name = item.type.casefold()
        text = f"{item.name} {item.semantic_role} {item.description}".casefold()
        if type_name in {"number", "integer", "boolean"}:
            shape.add("scalar")
        if type_name in {"array", "file_list", "raster_list"}:
            shape.add("collection")
        if type_name == "raster_list":
            shape.add("raster_collection")
        elif type_name == "raster":
            shape.add("raster_artifact" if output else "single_raster")
        if output and type_name == "file":
            shape.add("artifact")
        if any(word in text for word in ("scalar", "numeric value", "single value", "标量", "数值")):
            shape.add("scalar")
        if output and any(word in text for word in ("raster", "geotiff", ".tif", "栅格", "影像")):
            shape.add("raster_artifact")
        if any(word in text for word in ("raster list", "rasters", "image list", "栅格列表", "多景")):
            shape.add("raster_collection")
    return shape


def _extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def match_capability_contract(
    required: CapabilityContract,
    offered: CapabilityContract,
    *,
    operation_judgement: OperationJudgement | None = None,
) -> CapabilityMatchResult:
    """Match by LLM-judged operation coverage and treat detailed policies as hints.

    Formula wording, units, ranges, boundary policies, NoData conventions,
    spatial metadata and artifact declarations are useful ranking evidence but
    are not reliable enough to reject a runnable tool.  Concrete interface
    compatibility remains the caller and generated wrapper's responsibility.
    """
    covered: list[str] = []
    uncovered: list[str] = []
    conflicts: list[str] = []

    operation_resolved = False
    if operation_judgement is None:
        if required.operation and offered.operation and required.operation != offered.operation:
            conflicts.append(
                "operation coverage incompatible: operation identifiers differ"
            )
        else:
            uncovered.append("operation_coverage:unjudged")
    elif operation_judgement.decision == "covers":
        operation_resolved = True
        covered.append("operation:llm_covers")
    elif operation_judgement.decision == "incompatible":
        conflicts.append(
            "operation coverage incompatible: " + operation_judgement.reason[:300]
        )
    else:
        uncovered.append("operation_coverage:uncertain")

    advisory_pairs = (
        ("formula_or_rule", required.formula_or_rule, offered.formula_or_rule),
        ("boundary_policy", required.boundary_policy, offered.boundary_policy),
        ("nodata_policy", required.nodata_policy, offered.nodata_policy),
        ("unit", required.unit, offered.unit),
    )
    for name, wanted, declared in advisory_pairs:
        if not wanted:
            continue
        if not declared:
            uncovered.append(f"advisory:{name}:undeclared")
        elif wanted == declared or _rules_relation(str(wanted), str(declared)) == "compatible":
            covered.append(f"advisory:{name}")
        else:
            uncovered.append(f"advisory:{name}:different")

    if required.value_range is not None:
        if offered.value_range is not None and offered.value_range[0] <= required.value_range[0] \
                and offered.value_range[1] >= required.value_range[1]:
            covered.append("advisory:value_range")
        else:
            uncovered.append("advisory:value_range")
    for name in required.spatial_requirements.__dataclass_fields__:
        if getattr(required.spatial_requirements, name):
            marker = f"advisory:spatial_requirements.{name}"
            (covered if getattr(offered.spatial_requirements, name) else uncovered).append(marker)
    offered_artifacts = {item.type.casefold() for item in offered.artifacts if item.required}
    for artifact in required.artifacts:
        if artifact.required:
            marker = f"advisory:artifact:{artifact.type}"
            (covered if artifact.type.casefold() in offered_artifacts else uncovered).append(marker)

    matched = operation_resolved and not conflicts
    agreement_count = len([item for item in covered if item.startswith("advisory:")])
    advisory_count = agreement_count + len(uncovered)
    # LLM confidence has no calibrated objective meaning. Keep it only inside
    # operation_judgement for audit; candidate ordering uses deterministic
    # agreement with advisory declarations instead.
    confidence = 0.5 if matched else 0.0
    if matched and advisory_count:
        confidence += (1.0 - confidence) * agreement_count / advisory_count
    if conflicts:
        reason = "operation coverage conflict"
    elif not operation_resolved:
        reason = "operation coverage is unjudged or uncertain"
    elif uncovered:
        reason = (
            "core operation matched; detailed declarations are advisory and concrete "
            "arguments must follow the exposed schema"
        )
    else:
        reason = "core operation matched; concrete arguments must follow the exposed schema"
    judgement_evidence = asdict(operation_judgement) if operation_judgement else {}
    return CapabilityMatchResult(
        matched, confidence, covered, uncovered, conflicts, reason,
        operation_judgement=judgement_evidence,
    )


def _rules_relation(required: str, offered: str) -> str:
    req = re.sub(r"\s+", "", required.casefold())
    got = re.sub(r"\s+", "", offered.casefold())
    directions = (("<", ">"), ("greater", "less"), ("above", "below"))
    for left, right in directions:
        if (left in req and right in got) or (right in req and left in got):
            return "conflict"
    if req == got or req in got or got in req:
        return "compatible"
    req_tokens = set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", required.casefold()))
    got_tokens = set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", offered.casefold()))
    union = req_tokens | got_tokens
    if union and len(req_tokens & got_tokens) / len(union) >= 0.6:
        return "compatible"
    return "unknown"
