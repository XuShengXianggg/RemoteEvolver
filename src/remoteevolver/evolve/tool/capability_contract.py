"""Validated, serialisable capability contracts shared by all tool checks.

Contracts describe behaviour, not a particular benchmark or implementation.  The
same object is used when selecting an existing tool, certifying a generated
wrapper, and checking an individual real invocation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import ast
import hashlib
import json
import math
from pathlib import PurePath
from typing import Any, Mapping


_VALID_TYPES = {
    "any", "string", "number", "integer", "boolean", "array", "object",
    "file", "file_list", "raster", "raster_list", "directory",
}
_VALID_UNITS = {"", "fraction", "percent", "index", "meter", "degree", "count"}


class ContractValidationError(ValueError):
    """Raised when a capability contract is structurally invalid."""


@dataclass(frozen=True)
class ContractField:
    name: str
    type: str = "any"
    semantic_role: str = ""
    required: bool = True
    description: str = ""

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, output: bool = False) -> "ContractField":
        if not isinstance(value, Mapping):
            raise ContractValidationError("contract input/output entries must be objects")
        name = str(value.get("name", "")).strip()
        if not name:
            raise ContractValidationError("contract input/output field requires name")
        type_name = _normalise_type(value.get("type", "any"))
        role = str(value.get("semantic_role", value.get("description", ""))).strip()
        if type_name == "file" and "directory" in f"{name} {role}".casefold():
            type_name = "directory"
        return cls(
            name=name,
            type=type_name,
            semantic_role=role,
            required=bool(value.get("required", True)),
            description=str(value.get("description", "")).strip(),
        )


@dataclass(frozen=True)
class ArtifactSpec:
    type: str
    required: bool = True

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | str) -> "ArtifactSpec":
        if isinstance(value, str):
            artifact_type = _infer_artifact_type(value)
            return cls(artifact_type, True)
        if not isinstance(value, Mapping):
            raise ContractValidationError("artifact requires a non-empty type")
        raw_type = value.get("type") or value.get("format") or value.get("media_type")
        if not str(raw_type or "").strip():
            raw_type = _infer_artifact_type(
                value.get("path") or value.get("name") or value.get("description") or ""
            )
        return cls(str(raw_type).strip().lower(), bool(value.get("required", True)))


@dataclass(frozen=True)
class SpatialRequirements:
    same_crs: bool = False
    same_shape: bool = False
    same_resolution: bool = False
    preserve_metadata: bool = False

    @classmethod
    def from_dict(cls, value: Any) -> "SpatialRequirements":
        value = value or {}
        if isinstance(value, (str, list, tuple)):
            text = " ".join(map(str, value)) if not isinstance(value, str) else value
            text = text.casefold().replace("_", " ")
            return cls(
                same_crs=("crs" in text and any(word in text for word in ("same", "match", "align", "consistent"))),
                same_shape=(any(word in text for word in ("shape", "dimension", "extent"))
                            and any(word in text for word in ("same", "match", "align", "consistent"))),
                same_resolution=("resolution" in text
                                 and any(word in text for word in ("same", "match", "align", "consistent"))),
                preserve_metadata=("metadata" in text and any(word in text for word in ("preserve", "retain", "copy"))),
            )
        if not isinstance(value, Mapping):
            raise ContractValidationError("spatial_requirements must be an object")
        return cls(**{name: bool(value.get(name, False)) for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class CapabilityContract:
    operation: str
    description: str = ""
    inputs: tuple[ContractField, ...] = ()
    outputs: tuple[ContractField, ...] = ()
    formula_or_rule: str = ""
    unit: str = ""
    value_range: tuple[float, float] | None = None
    boundary_policy: str = ""
    nodata_policy: str = ""
    spatial_requirements: SpatialRequirements = field(default_factory=SpatialRequirements)
    artifacts: tuple[ArtifactSpec, ...] = ()
    preconditions: tuple[str, ...] = ()
    postconditions: tuple[str, ...] = ()
    testable_assertions: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CapabilityContract":
        if not isinstance(value, Mapping):
            raise ContractValidationError("capability contract must be a JSON object")
        operation = _normalise_operation(value.get("operation", ""))
        if not operation:
            raise ContractValidationError("capability contract requires operation")
        inputs_raw = _normalise_fields(value.get("inputs", []), field_name="inputs")
        outputs_raw = _normalise_fields(value.get("outputs", []), field_name="outputs")
        artifacts_raw = _normalise_artifacts(value.get("artifacts", []))
        unit = _normalise_unit(value.get("unit", ""))
        if unit not in _VALID_UNITS:
            raise ContractValidationError(f"unsupported unit: {unit!r}")
        inputs = tuple(ContractField.from_dict(v) for v in inputs_raw)
        outputs = tuple(ContractField.from_dict(v, output=True) for v in outputs_raw)
        formula = str(value.get("formula_or_rule", "")).strip()
        spatial = SpatialRequirements.from_dict(value.get("spatial_requirements"))
        nodata_policy = str(value.get("nodata_policy", "")).strip().lower()
        if operation in {
            "directory_listing", "file_selection", "raster_mean_scalar",
            "threshold_statistics", "mask_percentage",
        }:
            # These capabilities return references or scalars; CRS/shape and
            # metadata preservation are not semantic output guarantees.
            spatial = SpatialRequirements()
        if operation in {"directory_listing", "file_selection"}:
            # Pixel-level NoData semantics do not apply to discovering or
            # selecting file references and must not block a listing tool.
            nodata_policy = ""
        return cls(
            operation=operation,
            description=str(value.get("description", "")).strip(),
            inputs=inputs,
            outputs=outputs,
            formula_or_rule=formula,
            unit=unit,
            value_range=_parse_range(value.get("value_range")),
            boundary_policy=str(value.get("boundary_policy", "")).strip().lower(),
            nodata_policy=nodata_policy,
            spatial_requirements=spatial,
            artifacts=tuple(ArtifactSpec.from_dict(v) for v in artifacts_raw),
            preconditions=_strings(value.get("preconditions")),
            postconditions=_strings(value.get("postconditions")),
            testable_assertions=_strings(value.get("testable_assertions")),
        )

    @classmethod
    def from_task_requirement(cls, value: Mapping[str, Any]) -> "CapabilityContract":
        """Compile LLM task requirements without weakening tool declarations.

        Operation and behavioural evidence remain unchanged.  Only unstable
        presentation details (field names/types, units and spatial container
        shape) are normalised so they cannot abort semantic tool selection.
        The strict ``from_dict`` path remains authoritative for tool-side
        declarations and certifications.
        """
        if not isinstance(value, Mapping):
            raise ContractValidationError("capability contract must be a JSON object")
        prepared = dict(value)
        prepared["inputs"] = _normalise_task_requirement_fields(
            value.get("inputs", []), field_name="inputs",
        )
        prepared["outputs"] = _normalise_task_requirement_fields(
            value.get("outputs", []), field_name="outputs",
        )
        unit = _normalise_unit(value.get("unit", ""))
        prepared["unit"] = unit if unit in _VALID_UNITS else ""
        try:
            SpatialRequirements.from_dict(value.get("spatial_requirements"))
        except (TypeError, ValueError):
            prepared["spatial_requirements"] = {}
        return cls.from_dict(prepared)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["inputs"] = list(data["inputs"])
        data["outputs"] = list(data["outputs"])
        data["artifacts"] = list(data["artifacts"])
        data["preconditions"] = list(data["preconditions"])
        data["postconditions"] = list(data["postconditions"])
        data["testable_assertions"] = list(data["testable_assertions"])
        data["value_range"] = list(self.value_range) if self.value_range is not None else ""
        return data

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())


def validate_contract_json(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a canonical JSON-compatible contract."""
    return CapabilityContract.from_dict(value).to_dict()


def fingerprint_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fingerprint_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def contract_from_capability_spec(spec: Any) -> CapabilityContract:
    """Convert an LLM-authored task requirement without hard-coding its ABI.

    Capability fields can come from an explicit ReAct tool-gap request.
    Descriptive types such as ``dataset references`` or
    ``numeric percentage`` therefore take the relaxed task-requirement path:
    semantic wording is retained while unknown presentation types become
    ``any``. Tool-side declarations remain strict.
    """
    raw_outputs = list(getattr(spec, "outputs", []) or [])
    primary_outputs = [
        item for item in raw_outputs
        if isinstance(item, Mapping) and item.get("primary") is True
    ]
    if len(primary_outputs) == 1:
        primary = primary_outputs[0]
        raw_outputs = [primary] + [item for item in raw_outputs if item is not primary]
    raw = {
        "operation": getattr(spec, "operation", "") or getattr(spec, "capability", ""),
        "description": getattr(spec, "capability", ""),
        "inputs": getattr(spec, "inputs", []) or [],
        "outputs": raw_outputs,
        "formula_or_rule": getattr(spec, "formula_or_rule", ""),
        "unit": getattr(spec, "unit", ""),
        "value_range": getattr(spec, "value_range", ""),
        "boundary_policy": getattr(spec, "boundary_policy", ""),
        "nodata_policy": getattr(spec, "nodata_policy", ""),
        "spatial_requirements": getattr(spec, "spatial_requirements", {}) or {},
        "artifacts": getattr(spec, "artifacts", []) or [],
        "preconditions": getattr(spec, "preconditions", []) or [],
        "postconditions": getattr(spec, "postconditions", []) or [],
        "testable_assertions": getattr(spec, "testable_assertions", []) or [],
    }
    return CapabilityContract.from_task_requirement(raw)


def function_schema_from_wrapper(wrapper_code: str, function_name: str = "") -> dict[str, Any]:
    """Build the subset of MCP JSON schema needed for fingerprints and tests."""
    tree = ast.parse(wrapper_code)
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    function = next((node for node in functions if node.name == function_name), None) if function_name else None
    function = function or (functions[0] if functions else None)
    if function is None:
        raise ContractValidationError("wrapper contains no function")
    properties: dict[str, Any] = {}
    required: list[str] = []
    defaults_from = len(function.args.args) - len(function.args.defaults)
    for index, argument in enumerate(function.args.args):
        annotation = ast.unparse(argument.annotation) if argument.annotation else "Any"
        properties[argument.arg] = {"type": _json_type_from_annotation(annotation)}
        if index < defaults_from:
            required.append(argument.arg)
    return {
        "name": function.name,
        "description": ast.get_docstring(function) or "",
        "parameters": {"type": "object", "properties": properties, "required": required},
    }


def contract_io_from_schema(schema: Mapping[str, Any]) -> list[dict[str, Any]]:
    params = schema.get("parameters", {}) if isinstance(schema, Mapping) else {}
    properties = params.get("properties", {}) if isinstance(params, Mapping) else {}
    required = set(params.get("required", []) if isinstance(params, Mapping) else [])
    output: list[dict[str, Any]] = []
    for name, spec in (properties.items() if isinstance(properties, Mapping) else []):
        type_name = spec.get("type", "any") if isinstance(spec, Mapping) else "any"
        output.append({
            "name": str(name), "type": _normalise_type(type_name),
            "semantic_role": str(name), "required": name in required,
        })
    return output


def _normalise_operation(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", " ")
    return "_".join(text.split())[:120]


def _normalise_type(value: Any) -> str:
    text = str(value or "any").strip().lower().replace(" ", "_")
    aliases = {
        "str": "string", "float": "number", "int": "integer", "bool": "boolean",
        "list": "array", "list[str]": "file_list", "geotiff": "raster",
        "tiff": "raster", "path": "file", "paths": "file_list", "scalar": "number",
        "geotiff_list": "raster_list", "list_of_rasters": "raster_list",
        "raster_stack": "raster_list", "raster_series": "raster_list",
        "scalar_percent": "number", "percentage": "number", "percent": "number",
        "fraction": "number", "ratio": "number", "index": "number",
        "raster[]": "raster_list", "raster_(boolean)": "raster",
        "raster(boolean)": "raster",
        "files": "file_list", "raster_collection": "raster_list",
        "string_(path)": "file", "string(path)": "file",
    }
    text = aliases.get(text, text)
    # LLMs occasionally return descriptive semantic shapes despite the enum
    # instruction.  Normalise only explicit lexical evidence; never infer a
    # domain operation or alter values here.
    if text not in _VALID_TYPES:
        if any(word in text for word in ("directory", "folder", "目录")):
            text = "directory"
        elif any(word in text for word in ("scalar", "标量", "numeric_value", "数值")):
            text = "number"
        elif any(word in text for word in ("raster", "image", "栅格", "影像")):
            text = "raster_list" if any(word in text for word in (
                "list", "collection", "series", "stack", "集合", "序列", "多幅", "按时间",
                "temporal", "multi",
            )) else "raster"
        elif any(word in text for word in ("file", "path", "文件", "路径")):
            text = "file_list" if any(word in text for word in (
                "list", "collection", "group", "集合", "清单", "分组", "多个",
            )) else "file"
        elif any(word in text for word in (
            "criteria", "condition", "parameter", "descriptor", "metadata",
            "条件", "参数", "描述", "元数据",
        )):
            text = "object"
    if text not in _VALID_TYPES and "raster" in text:
        if any(word in text for word in ("series", "stack", "time", "temporal", "collection", "list", "[]")):
            text = "raster_list"
        else:
            text = "raster"
    if text not in _VALID_TYPES and ("list" in text or "array" in text):
        if any(word in text for word in ("raster", "geotiff", "tiff")):
            text = "raster_list"
        elif any(word in text for word in ("file", "path", "str")):
            text = "file_list"
        else:
            text = "array"
    if text not in _VALID_TYPES:
        raise ContractValidationError(f"unsupported contract type: {text!r}")
    return text


def _normalise_unit(value: Any) -> str:
    text = str(value or "").strip().casefold().replace("_", " ")
    if not text or any(word in text for word in ("boolean", "bool", "not applicable", "n/a")):
        return ""
    if "percent" in text or text == "%":
        return "percent"
    if any(word in text for word in ("unitless", "dimensionless", "index")):
        return "index"
    if any(word in text for word in ("fraction", "ratio", "reflectance")):
        return "fraction"
    aliases = {"metres": "meter", "meters": "meter", "degrees": "degree", "counts": "count"}
    return aliases.get(text, text)


def _normalise_fields(value: Any, *, field_name: str) -> list[Mapping[str, Any]]:
    """Accept common LLM JSON variations without weakening semantic validation."""
    if value in (None, ""):
        return []
    if isinstance(value, Mapping):
        if value.get("name"):
            return [value]
        fields: list[Mapping[str, Any]] = []
        for name, spec in value.items():
            if isinstance(spec, Mapping):
                fields.append({"name": str(name), **spec})
            else:
                fields.append({"name": str(name), "type": spec or "any"})
        return fields
    if not isinstance(value, (list, tuple)):
        raise ContractValidationError(f"{field_name} must be an array or field map")
    fields = []
    for index, item in enumerate(value, 1):
        if isinstance(item, Mapping):
            fields.append(item)
        elif isinstance(item, str) and item.strip():
            # Some models compress a semantic field to a descriptive string.
            # Preserve that text as the role and construct a neutral ABI-free
            # field; no domain semantics are invented.
            fields.append({
                "name": f"{field_name[:-1]}_{index}",
                "type": "any",
                "semantic_role": item.strip(),
            })
        else:
            raise ContractValidationError("contract input/output entries must be objects or descriptions")
    return fields


def _normalise_task_requirement_fields(
    value: Any, *, field_name: str,
) -> list[Mapping[str, Any]]:
    """Preserve semantic roles while neutralising unstable LLM ABI labels."""
    fields = _normalise_fields(value, field_name=field_name)
    output: list[Mapping[str, Any]] = []
    singular = field_name[:-1]
    for index, item in enumerate(fields, 1):
        row = dict(item)
        row["name"] = str(row.get("name", "")).strip() or f"{singular}_{index}"
        raw_type = str(row.get("type", "any") or "any").strip()
        try:
            _normalise_type(raw_type)
        except ContractValidationError:
            role = str(row.get("semantic_role", row.get("description", ""))).strip()
            shape_note = f"declared semantic shape: {raw_type}"
            row["semantic_role"] = f"{role}; {shape_note}" if role else shape_note
            row["type"] = "any"
        output.append(row)
    return output


def _normalise_artifacts(value: Any) -> list[Mapping[str, Any] | str]:
    if value in (None, ""):
        return []
    if isinstance(value, (str, Mapping)):
        return [value]
    if not isinstance(value, (list, tuple)):
        raise ContractValidationError("artifacts must be an array")
    return list(value)


def _infer_artifact_type(value: Any) -> str:
    """Reduce task-specific paths to a stable output kind for the contract."""
    text = str(value or "").strip().lower()
    suffix = PurePath(text).suffix
    if suffix in {".tif", ".tiff", ".img", ".vrt"} or "raster" in text:
        return "raster"
    if suffix in {".json", ".geojson"} or "json" in text:
        return "json"
    if suffix in {".csv", ".tsv"}:
        return "table"
    if suffix in {".txt", ".md", ".log"} or "text" in text:
        return "text"
    if "directory" in text or "folder" in text:
        return "directory"
    return "file"


def _json_type_from_annotation(annotation: str) -> str:
    text = annotation.replace(" ", "").lower()
    if "list" in text or "tuple" in text:
        return "array"
    if "float" in text:
        return "number"
    if "int" in text:
        return "integer"
    if "bool" in text:
        return "boolean"
    if "dict" in text:
        return "object"
    return "string"


def _parse_range(value: Any) -> tuple[float, float] | None:
    if value in (None, "", []):
        return None
    if isinstance(value, str):
        import re
        text = value.strip()
        if "inf" in text.casefold() or text.casefold() in {"any", "unbounded", "none"}:
            return None
        # Treat a bare hyphen/en-dash/em-dash/tilde between two numerals as a
        # range delimiter before the generic signed-number scan. Otherwise
        # common forms such as ``0-100`` are misread as [0, -100]. This also
        # handles negative lower bounds such as ``-1-1`` unambiguously.
        delimited = re.fullmatch(
            r"\s*([-+]?\d+(?:\.\d+)?)\s*[-–—~]\s*([-+]?\d+(?:\.\d+)?)\s*",
            text,
        )
        numeric_parts = (
            list(delimited.groups()) if delimited
            else re.findall(r"[-+]?\d+(?:\.\d+)?", text)
        )
        if len(numeric_parts) == 2:
            lo, hi = float(numeric_parts[0]), float(numeric_parts[1])
            if lo > hi:
                raise ContractValidationError("value_range lower bound exceeds upper bound")
            return lo, hi
        value = value.strip().strip("[]()")
        parts = [p.strip() for p in value.replace("～", ",").replace(" to ", ",").split(",")]
        if len(parts) != 2:
            import re
            parts = re.findall(r"[-+]?\d+(?:\.\d+)?", value)
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    elif isinstance(value, Mapping):
        lower = next((value[key] for key in ("min", "minimum", "lower", "lo") if key in value), None)
        upper = next((value[key] for key in ("max", "maximum", "upper", "hi") if key in value), None)
        if lower is None and upper is None:
            raise ContractValidationError("value_range object requires min/max or lower/upper")
        if lower is None or upper is None:
            return None
        parts = [lower, upper]
    else:
        raise ContractValidationError("value_range must be empty or a two-item array")
    if len(parts) != 2:
        raise ContractValidationError("value_range must contain exactly two values")
    lo, hi = float(parts[0]), float(parts[1])
    if not math.isfinite(lo) or not math.isfinite(hi):
        return None
    if lo > hi:
        raise ContractValidationError("value_range lower bound exceeds upper bound")
    return lo, hi


def _strings(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if not isinstance(value, list):
        raise ContractValidationError("contract condition fields must be arrays")
    return tuple(str(item).strip() for item in value if str(item).strip())
