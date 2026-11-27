"""Per-invocation input preconditions, explicit adapters, and output invariants."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import re
from typing import Any

from .capability_contract import CapabilityContract, ContractField


@dataclass
class RuntimeContractResult:
    passed: bool
    category: str = ""
    violations: list[str] = field(default_factory=list)
    adapted_arguments: dict[str, Any] = field(default_factory=dict)
    adapters_applied: list[dict[str, str]] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_invocation_preconditions(
    contract: CapabilityContract,
    arguments: dict[str, Any],
    *,
    adapters: dict[str, str] | None = None,
    enforce_interface: bool = True,
) -> RuntimeContractResult:
    args = dict(arguments)
    applied: list[dict[str, str]] = []
    for name, adapter in (adapters or {}).items():
        if name not in args:
            continue
        before = type(args[name]).__name__
        if adapter == "csv_to_list" and isinstance(args[name], str):
            args[name] = [part.strip() for part in args[name].split(",") if part.strip()]
        elif adapter == "list_to_csv" and isinstance(args[name], (list, tuple)):
            args[name] = ",".join(map(str, args[name]))
        else:
            return RuntimeContractResult(False, "input_contract_failure",
                                         [f"adapter {adapter!r} cannot convert argument {name!r}"], args, applied)
        applied.append({"argument": name, "adapter": adapter, "from": before, "to": type(args[name]).__name__})

    violations: list[str] = []
    raster_groups: list[tuple[str, list[Path]]] = []
    fields = {field.name: field for field in contract.inputs} if enforce_interface else {}
    for field in contract.inputs if enforce_interface else ():
        if field.required and field.name not in args:
            violations.append(f"missing required argument: {field.name}")
            continue
        if field.name not in args:
            continue
        value = args[field.name]
        if not _value_matches_type(value, field.type):
            violations.append(f"argument {field.name} expects {field.type}, got {type(value).__name__}")
            continue
        paths = _paths_for(field, value)
        if field.type in {"file", "raster"} and not paths:
            violations.append(f"input file does not exist: {value}")
        if field.type == "directory" and not Path(value).is_dir():
            violations.append(f"input directory does not exist: {value}")
        if field.type in {"file_list", "raster_list"} and not paths:
            violations.append(f"input file list is empty or contains no existing files: {field.name}")
        if field.type in {"raster", "raster_list"} and paths:
            raster_groups.append((field.semantic_role or field.name, paths))

    if not enforce_interface:
        for name, value in args.items():
            paths = _as_existing_paths(value)
            raster_paths = [path for path in paths if path.suffix.casefold() in {".tif", ".tiff"}]
            if raster_paths:
                raster_groups.append((str(name), raster_paths))

    if contract.spatial_requirements.same_crs or contract.spatial_requirements.same_shape or contract.spatial_requirements.same_resolution:
        violations.extend(_check_raster_compatibility(raster_groups, contract))
    violations.extend(_check_date_pairing(raster_groups))
    return RuntimeContractResult(
        not violations, "" if not violations else "input_contract_failure", violations,
        args, applied, {"arguments": _summarise(args), "declared_fields": sorted(fields)},
    )


def check_invocation_postconditions(
    contract: CapabilityContract,
    result_repr: str,
    *,
    arguments: dict[str, Any] | None = None,
    enforce_interface: bool = True,
) -> RuntimeContractResult:
    value = _literal(result_repr)
    violations: list[str] = []
    category = "output_contract_failure"
    if enforce_interface and contract.outputs:
        out = contract.outputs[0]
        if not _value_matches_type(value, out.type):
            # MCP serialises some structured results as text. A declared file/raster
            # may still be represented by a path string and is handled below.
            if out.type not in {"file", "raster"} or not isinstance(value, str):
                violations.append(f"return value expects {out.type}, got {type(value).__name__}")

    numeric = _extract_number(value)
    if contract.value_range is not None and numeric is not None:
        lo, hi = contract.value_range
        if not math.isfinite(numeric) or numeric < lo or numeric > hi:
            category = "unit_failure" if contract.unit in {"fraction", "percent"} else "output_contract_failure"
            violations.append(f"numeric result {numeric} outside declared range [{lo}, {hi}]")

    paths = _paths_from_result(value)
    required_artifacts = [item for item in contract.artifacts if item.required]
    if required_artifacts and not paths:
        category = "artifact_failure"
        violations.append("required output artifact path is missing from return value")
    raster_paths = [path for path in paths if path.suffix.lower() in {".tif", ".tiff"}]
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            category = "artifact_failure"
            violations.append(f"output artifact missing or empty: {path}")
    for path in raster_paths:
        raster_violations = _check_output_raster(path, contract, arguments or {})
        if raster_violations:
            category = "spatial_metadata_failure" if any("metadata" in item for item in raster_violations) else "output_contract_failure"
            violations.extend(raster_violations)
    return RuntimeContractResult(
        not violations, "" if not violations else category, violations,
        dict(arguments or {}), [], {"result": _summarise_value(value), "artifacts": [str(p) for p in paths]},
    )


def _value_matches_type(value: Any, type_name: str) -> bool:
    if type_name == "any":
        return True
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "array":
        return isinstance(value, (list, tuple))
    if type_name == "object":
        return isinstance(value, dict)
    if type_name in {"file", "raster", "directory"}:
        return isinstance(value, (str, Path))
    if type_name in {"file_list", "raster_list"}:
        return isinstance(value, (list, tuple))
    return False


def _paths_for(field: ContractField, value: Any) -> list[Path]:
    raw = list(value) if isinstance(value, (list, tuple)) else [value]
    return [Path(item) for item in raw if isinstance(item, (str, Path)) and Path(item).is_file()]


def _check_raster_compatibility(groups: list[tuple[str, list[Path]]], contract: CapabilityContract) -> list[str]:
    try:
        import rasterio
    except Exception:  # noqa: BLE001
        return ["raster metadata cannot be verified because rasterio is unavailable"]
    metadata: list[tuple[Path, Any, tuple[int, int], Any]] = []
    for _, paths in groups:
        for path in paths:
            try:
                with rasterio.open(path) as ds:
                    metadata.append((path, ds.crs, (ds.height, ds.width), ds.res))
            except Exception as exc:  # noqa: BLE001
                return [f"input raster unreadable: {path} ({exc})"]
    if len(metadata) < 2:
        return []
    first = metadata[0]
    violations: list[str] = []
    for current in metadata[1:]:
        if contract.spatial_requirements.same_crs and current[1] != first[1]:
            violations.append(f"CRS mismatch: {first[0]} vs {current[0]}")
        if contract.spatial_requirements.same_shape and current[2] != first[2]:
            violations.append(f"raster shape mismatch: {first[2]} vs {current[2]}")
        if contract.spatial_requirements.same_resolution and current[3] != first[3]:
            violations.append(f"raster resolution mismatch: {first[3]} vs {current[3]}")
    return violations


def _check_date_pairing(groups: list[tuple[str, list[Path]]]) -> list[str]:
    keyed = [(role, {_date_key(path) for path in paths}) for role, paths in groups
             if any(word in role.casefold() for word in ("green", "swir", "band"))]
    if len(keyed) < 2:
        return []
    first_role, first_keys = keyed[0]
    for role, keys in keyed[1:]:
        if keys != first_keys:
            return [f"raster date pairing mismatch: {first_role}={sorted(first_keys)}, {role}={sorted(keys)}"]
    return []


def _date_key(path: Path) -> str:
    match = re.search(r"(?:19|20)\d{2}[-_]?\d{2}[-_]?\d{2}", path.name)
    return re.sub(r"[-_]", "", match.group(0)) if match else path.stem


def _check_output_raster(path: Path, contract: CapabilityContract, arguments: dict[str, Any]) -> list[str]:
    try:
        import numpy as np
        import rasterio
        with rasterio.open(path) as ds:
            band = ds.read(1, masked=True)
            data = np.ma.asarray(band).astype("float64").filled(np.nan)
            finite = data[np.isfinite(data)]
            if finite.size == 0:
                return [f"output raster is all NaN/NoData: {path}"]
            if contract.value_range is not None:
                lo, hi = contract.value_range
                if float(finite.min()) < lo or float(finite.max()) > hi:
                    return [f"output raster range [{finite.min()}, {finite.max()}] exceeds [{lo}, {hi}]"]
            if contract.spatial_requirements.preserve_metadata:
                input_paths = [p for value in arguments.values() for p in _as_existing_paths(value)]
                if input_paths:
                    with rasterio.open(input_paths[0]) as source:
                        if ds.crs != source.crs or ds.transform != source.transform or ds.shape != source.shape:
                            return ["output raster metadata does not preserve input CRS/transform/shape"]
    except Exception as exc:  # noqa: BLE001
        return [f"output raster unreadable: {path} ({exc})"]
    return []


def _as_existing_paths(value: Any) -> list[Path]:
    values = value if isinstance(value, (list, tuple)) else [value]
    return [Path(item) for item in values if isinstance(item, (str, Path)) and Path(item).is_file()]


def _literal(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value.strip().strip("'\"")


def _extract_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _paths_from_result(value: Any) -> list[Path]:
    if isinstance(value, dict):
        values = list(value.values())
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    paths: list[Path] = []
    for item in values:
        if not isinstance(item, str):
            continue
        for part in item.split(","):
            part = part.strip()
            if part and len(part) < 1024 and Path(part).suffix:
                paths.append(Path(part))
    return paths


def _summarise(arguments: dict[str, Any]) -> dict[str, Any]:
    return {key: _summarise_value(value) for key, value in arguments.items()}


def _summarise_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "count": len(value)}
    if isinstance(value, dict):
        return {"type": "object", "keys": sorted(map(str, value.keys()))[:20]}
    if isinstance(value, str):
        return value[:300]
    return value
