"""Typed dataflow between string-returning MCP tools."""

from __future__ import annotations

import json
import math
import re
from typing import Any


class DataflowError(ValueError):
    """A tool result cannot satisfy a structured workflow reference."""


_FIELD_START = re.compile(
    r"(?:(?<=^)|(?<=[;\n])|(?<=\s))([A-Za-z_][A-Za-z0-9_]*)\s*="
)
_INT = re.compile(r"[-+]?\d+")
_FLOAT = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def is_tool_error_result(result: Any) -> bool:
    """统一识别框架错误与进化 wrapper 常见的 ``[tool_name error]``。"""
    if not isinstance(result, str):
        return False
    lowered = result.lstrip().lower()
    return lowered.startswith("[tool error]") or (
        lowered.startswith("[") and " error]" in lowered[:100]
    )


def parse_tool_result(result: str) -> dict[str, Any]:
    """Parse JSON or repeated ``key=value`` output into typed named fields."""
    text = str(result or "").strip()
    if is_tool_error_result(text):
        raise DataflowError(f"cannot reference failed tool output: {text[:160]}")
    if not text:
        raise DataflowError("cannot reference empty tool output")

    try:
        decoded = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        decoded = None
    else:
        if isinstance(decoded, dict):
            return {str(k): v for k, v in decoded.items()}
        return {"$value": decoded}

    matches = list(_FIELD_START.finditer(text))
    if not matches:
        return {"$value": text}
    fields: dict[str, Any] = {}
    for pos, match in enumerate(matches):
        start = match.end()
        end = matches[pos + 1].start() if pos + 1 < len(matches) else len(text)
        raw = text[start:end].strip().strip(";,")
        value = _parse_scalar(raw)
        key = match.group(1)
        if key in fields:
            prior = fields[key]
            fields[key] = prior + [value] if isinstance(prior, list) else [prior, value]
        else:
            fields[key] = value
    return fields


def extract_tool_result(result: str, *, field: str | None = None, raw: bool = False) -> Any:
    """Resolve a field-level reference to a typed Python value."""
    if raw:
        if field is not None:
            raise DataflowError("a reference cannot request both $field and $raw")
        return result
    if not field:
        raise DataflowError("structured $from_step requires $field or $raw=true")
    fields = parse_tool_result(result)
    if field not in fields:
        raise DataflowError(f"field {field!r} not found; available fields={sorted(fields)}")
    return fields[field]


def matching_result_field(result: str, real_value: Any) -> str | None:
    """Find the unique parsed field equal to a downstream real argument."""
    try:
        fields = parse_tool_result(result)
    except DataflowError:
        return None
    matches = [name for name, value in fields.items() if _equivalent(value, real_value)]
    return matches[0] if len(matches) == 1 else None


def matching_argument(arguments: dict[str, Any], real_value: Any) -> str | None:
    """Find a unique upstream argument whose value is reused downstream."""
    matches = [
        str(name)
        for name, value in (arguments or {}).items()
        if _equivalent(value, real_value)
    ]
    return matches[0] if len(matches) == 1 else None


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        pass
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none"}:
        return None
    if _INT.fullmatch(value):
        try:
            return int(value)
        except ValueError:
            pass
    if _FLOAT.fullmatch(value):
        try:
            number = float(value)
            return number if math.isfinite(number) else value
        except ValueError:
            pass
    return value


def _equivalent(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(left, str) and isinstance(right, str):
        return left.strip() == right.strip()
    return left == right
