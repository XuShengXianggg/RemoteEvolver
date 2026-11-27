"""Lightweight, auditable dataflow facts captured during ReAct execution.

This module deliberately records only relationships that can be checked against
the concrete task and earlier tool calls.  It never attempts to reconstruct the
model's hidden reasoning or infer a complex transformation from one example.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from .dataflow import matching_argument, matching_result_field, parse_tool_result


def capture_transition_event(
    task: str,
    arguments: dict[str, Any],
    prior_steps: Iterable[Any],
) -> dict[str, Any]:
    """Describe where each concrete tool argument demonstrably came from."""
    steps = list(prior_steps)
    sources: dict[str, dict[str, Any]] = {}
    for name, value in arguments.items():
        source = _prior_source(value, steps)
        if source is None and _appears_verbatim_in_task(value, task):
            source = {"source": "task_input", "mode": "bind"}
        if source is None:
            source = {"source": "react", "mode": "semantic"}
        sources[str(name)] = source
    return {
        "version": 1,
        "kind": "tool_transition",
        "after_step": len(steps) - 1,
        "before_step": len(steps),
        "argument_sources": sources,
    }


def attach_output_shape(event: dict[str, Any], result: Any) -> dict[str, Any]:
    """Add a compact output interface observation without retaining its value."""
    event = dict(event)
    event["output_shape"] = _output_shape(result)
    return event


def _prior_source(value: Any, steps: list[Any]) -> dict[str, Any] | None:
    for index in range(len(steps) - 1, -1, -1):
        step = steps[index]
        result = str(getattr(step, "result", ""))
        field = matching_result_field(result, value)
        if field is not None:
            return {
                "source": "previous_result",
                "source_step": index,
                "selector": {"field": field},
                "mode": "direct",
            }
        upstream_args = getattr(step, "arguments", {}) or {}
        argument = matching_argument(upstream_args, value)
        if argument is not None:
            return {
                "source": "previous_argument",
                "source_step": index,
                "selector": {"arg": argument},
                "mode": "direct",
            }
        if result == value:
            return {
                "source": "previous_result",
                "source_step": index,
                "selector": {"raw": True},
                "mode": "direct",
            }
    return None


def _appears_verbatim_in_task(value: Any, task: str) -> bool:
    if isinstance(value, bool) or value is None:
        return str(value).lower() in task.lower()
    if isinstance(value, (int, float)):
        token = re.escape(str(value))
        return re.search(rf"(?<![\w.]){token}(?![\w.])", task) is not None
    if isinstance(value, str):
        stripped = value.strip()
        return len(stripped) >= 2 and stripped in task
    if isinstance(value, (list, dict)):
        try:
            compact = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return False
        return compact in task.replace(" ", "")
    return False


def _output_shape(result: Any) -> dict[str, Any]:
    text = str(result or "")
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        try:
            fields = parse_tool_result(text)
        except Exception:  # the shape is optional audit metadata
            fields = {"$value": text}
        if set(fields) != {"$value"}:
            return {"type": "object", "fields": sorted(fields)}
        value = fields.get("$value")
    if isinstance(value, dict):
        return {"type": "object", "fields": sorted(str(key) for key in value)}
    if isinstance(value, list):
        kinds = sorted({type(item).__name__ for item in value[:20]})
        return {"type": "array", "item_types": kinds}
    if isinstance(value, bool):
        kind = "boolean"
    elif isinstance(value, (int, float)):
        kind = "number"
    elif value is None:
        kind = "null"
    else:
        kind = "string"
    return {"type": kind}
