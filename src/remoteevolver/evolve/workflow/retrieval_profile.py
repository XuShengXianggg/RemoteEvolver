"""Pre-ReAct task profiles and four-field semantic similarity.

Only the task question and visible input basenames enter the profile.  Tool
catalogues, traces, answers, Skill text and dataset labels are deliberately
outside this boundary.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from typing import Any, Callable

from ...llm.base import BaseLLM, Message, generate_required_tool


PROFILE_VERSION = "pre-react-semantic-profile-v5.2-four-fields"
RETRIEVAL_VARIANT = "input-output-multiset-micro-v1"
EMBEDDING_MODEL = os.getenv(
    "RSEVO_EMBEDDING_MODEL", "",
).strip()
FIELD_WEIGHTS = {
    "goal": 0.30,
    "input": 0.20,
    "operation": 0.35,
    "output": 0.15,
}
OUTPUT_TYPES = {
    "numeric_scalar", "coordinate_or_numeric_vector", "integer_count",
    "boolean", "categorical_label", "entity", "entity_mapping",
    "entity_list", "ranked_list", "numeric_series", "raster", "mask",
    "map", "annotated_image", "chart", "text_explanation",
}

_PROFILE_PROMPT = r"""
Create a compact retrieval profile for one task before any ReAct execution.

Use only the supplied question and supplied input names/descriptions. Do not use
or infer from reference answers, tool catalogs, execution traces, Skills,
dataset labels, class labels, or family labels. Represent the request; do not
solve it and do not choose tools.

Produce exactly four semantic fields:
1. task_goal: a concise normalized statement of what must be determined,
   measured, compared, selected, or produced. Include the subject and intended
   result. Combine primary and genuinely requested secondary results here.
2. input_roles: the semantic role of each distinct supplied input-file group.
   Describe reusable roles supported by the supplied filenames, not filenames
   themselves or literal places/dates. Use an empty list when no input files are
   supplied. When this list is empty, do not add an input_roles evidence item
   and do not cite the input-list formatting marker. Never turn a place, time
   range, search area, requested entity, or
   externally obtainable data mentioned only in the question into an input role.
   Do not collapse distinct required input channels, layers, images, or entity
   sets merely because they share the same data kind. Preserve their
   multiplicity and different semantic roles.
3. operation_chain: ordered question-level logical operations that can be
   inferred before execution. Keep it abstract. Never emit tool names, API
   calls, file paths, or hidden implementation steps.
   Explicitly preserve every requested final selection, aggregation,
   comparison, measurement, assignment, or presentation operation, together
   with its order and scope.
4. output_contract: only final results the user asks to receive or present. Do
   not include an intermediate raster, mask, statistic, or entity list merely
   because it may be useful internally. Preserve compound outputs.

Interpret words and numbers in context. A contextual search/buffer radius limits
the area in which entities are obtained; it is not an entity-to-entity distance
constraint. A walking-time cutoff is a time condition, not a distance condition.
Ground sampling distance is a measurement scale, not a requested length; an
assumed average speed does not request an average result.

Preserve relation direction and distinguish per-subject nearest assignment from
a single global closest/farthest pair. Express this directly in task_goal and
operation_chain. Put thresholds and restrictions into the operation that uses
them, but normalize away literal values. If a named method or index is explicitly
required, include it in the relevant operation.

output_contract.type must be one of: numeric_scalar,
coordinate_or_numeric_vector, integer_count, boolean, categorical_label,
entity, entity_mapping, entity_list, ranked_list, numeric_series, raster, mask,
map, annotated_image, chart, or text_explanation. A coordinate pair is
coordinate_or_numeric_vector; a requested association from each source entity
to a target is entity_mapping. Scientific derived rasters are intermediate
unless the user asks to return, map, preview, save, or otherwise present them.
Raster inputs alone never justify a raster output. A yes/no question requires
boolean; a requested category or direction also requires categorical_label.

Normalize reusable descriptions as short lower_snake_case phrases. Do not put
incidental proper names, dates, or literal numeric values in semantic fields.
If genuinely ambiguous, record the ambiguity in quality.uncertainties rather
than inventing missing semantics. Evidence quotes must be short, contiguous,
verbatim substrings of the question or an input name. Return only the requested
structured object.
""".strip()


def profile_tool() -> dict[str, Any]:
    item = {
        "type": "object", "additionalProperties": False,
        "required": ["data_kind", "role"],
        "properties": {"data_kind": {"type": "string"}, "role": {"type": "string"}},
    }
    operation = {
        "type": "object", "additionalProperties": False,
        "required": ["step", "operation", "object"],
        "properties": {
            "step": {"type": "integer", "minimum": 1},
            "operation": {"type": "string"}, "object": {"type": "string"},
        },
    }
    output = {
        "type": "object", "additionalProperties": False,
        "required": ["type", "content"],
        "properties": {
            "type": {"type": "string", "enum": sorted(OUTPUT_TYPES)},
            "content": {"type": "string"},
        },
    }
    evidence = {
        "type": "object", "additionalProperties": False,
        "required": ["field", "quote"],
        "properties": {"field": {"type": "string"}, "quote": {"type": "string"}},
    }
    return {
        "type": "function",
        "function": {
            "name": "submit_pre_react_profile",
            "description": "Submit the four-field pre-ReAct retrieval profile.",
            "strict": True,
            "parameters": {
                "type": "object", "additionalProperties": False,
                "required": [
                    "task_goal", "input_roles", "operation_chain",
                    "output_contract", "quality",
                ],
                "properties": {
                    "task_goal": {"type": "string"},
                    "input_roles": {"type": "array", "items": item},
                    "operation_chain": {"type": "array", "items": operation},
                    "output_contract": {"type": "array", "minItems": 1, "items": output},
                    "quality": {
                        "type": "object", "additionalProperties": False,
                        "required": ["confidence", "uncertainties", "evidence"],
                        "properties": {
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "uncertainties": {"type": "array", "items": {"type": "string"}},
                            "evidence": {"type": "array", "items": evidence},
                        },
                    },
                },
            },
        },
    }


def generate_profile(
    llm: BaseLLM, question: str, input_names: tuple[str, ...], *, attempts: int = 2,
) -> dict[str, Any]:
    inputs = "\n".join(f"- {name}" for name in input_names) or "[]"
    messages = [
        Message(role="system", content=_PROFILE_PROMPT),
        Message(role="user", content=f"Question:\n{question.strip()}\n\nSupplied input names:\n{inputs}"),
    ]
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            response = generate_required_tool(llm, messages, profile_tool())
            calls = [
                call for call in response.tool_calls
                if call.name == "submit_pre_react_profile"
            ]
            if response.tool_calls:
                if len(calls) != 1 or len(response.tool_calls) != 1:
                    raise ValueError("profile response must contain exactly one required tool call")
                value = calls[0].arguments
            else:
                value = json.loads(response.text)
            errors = validate_profile(value, question, input_names)
            if errors:
                raise ValueError("; ".join(errors))
            return value
        except Exception as exc:  # bounded complete retry, then fail closed
            last_error = exc
    raise RuntimeError(f"profile generation failed: {last_error}") from last_error


def validate_profile(
    profile: Any, question: str, input_names: tuple[str, ...],
) -> list[str]:
    if not isinstance(profile, dict):
        return ["profile is not an object"]
    required = {"task_goal", "input_roles", "operation_chain", "output_contract", "quality"}
    errors: list[str] = []
    if set(profile) != required:
        errors.append("top-level fields differ from the required schema")
    if not isinstance(profile.get("task_goal"), str) or not profile.get("task_goal", "").strip():
        errors.append("task_goal is empty")
    roles = profile.get("input_roles")
    if not isinstance(roles, list) or any(
        not isinstance(row, dict) or set(row) != {"data_kind", "role"}
        or not all(isinstance(row[key], str) and row[key].strip() for key in row)
        for row in (roles if isinstance(roles, list) else [])
    ):
        errors.append("invalid input_roles")
    if not input_names and roles:
        errors.append("input_roles must be empty without supplied inputs")
    chain = profile.get("operation_chain")
    if not isinstance(chain, list):
        errors.append("invalid operation_chain")
    else:
        invalid_rows = [
            row for row in chain
            if not isinstance(row, dict)
            or set(row) != {"step", "operation", "object"}
            or not isinstance(row.get("operation"), str)
            or not row.get("operation", "").strip()
            or not isinstance(row.get("object"), str)
            or not row.get("object", "").strip()
        ]
        if invalid_rows:
            errors.append("invalid operation_chain rows")
        steps = [row.get("step") for row in chain if isinstance(row, dict)]
        if len(steps) != len(chain) or steps != list(range(1, len(chain) + 1)):
            errors.append("operation_chain steps must be consecutive from one")
    outputs = profile.get("output_contract")
    if not isinstance(outputs, list) or not outputs or any(
        not isinstance(row, dict) or set(row) != {"type", "content"}
        or row.get("type") not in OUTPUT_TYPES
        or not isinstance(row.get("content"), str) or not row["content"].strip()
        for row in (outputs if isinstance(outputs, list) else [])
    ):
        errors.append("invalid output_contract")
    quality = profile.get("quality")
    if (
        not isinstance(quality, dict)
        or set(quality) != {"confidence", "uncertainties", "evidence"}
        or not isinstance(quality.get("confidence"), (int, float))
        or isinstance(quality.get("confidence"), bool)
        or not 0 <= quality.get("confidence", -1) <= 1
        or not isinstance(quality.get("uncertainties"), list)
        or any(not isinstance(item, str) for item in quality.get("uncertainties", []))
        or not isinstance(quality.get("evidence"), list)
    ):
        errors.append("invalid quality")
        return errors
    visible = " ".join((question + "\n" + "\n".join(input_names)).casefold().split())
    for row in quality.get("evidence", []):
        quote = str(row.get("quote", "")).strip() if isinstance(row, dict) else ""
        normalized_quote = " ".join(quote.casefold().split())
        if (
            not isinstance(row, dict) or set(row) != {"field", "quote"}
            or not isinstance(row.get("field"), str) or not row["field"].strip()
            or not quote or normalized_quote not in visible
        ):
            errors.append(f"non-verbatim evidence: {quote!r}")
    return errors


def _normalize(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").split())


def field_documents(profile: dict[str, Any]) -> dict[str, str]:
    return {
        "goal": _normalize(profile["task_goal"]),
        "input": _normalize(" ".join(
            f"{row['data_kind']} {row['role']}" for row in profile["input_roles"]
        )),
        "operation": _normalize(" ".join(
            f"step_{row['step']} {row['operation']} {row['object']}"
            for row in profile["operation_chain"]
        )),
        "output": _normalize(" ".join(
            f"{row['type']} {row['content']}" for row in profile["output_contract"]
        )),
    }


def embed_documents(
    documents: list[str], *, client_factory: Callable[..., Any] | None = None,
) -> tuple[list[list[float]], dict[str, int]]:
    key = (
        os.getenv("RSEVO_EMBEDDING_API_KEY")
        or ""
    ).strip()
    base = (
        os.getenv("RSEVO_EMBEDDING_API_BASE_URL")
        or ""
    ).strip()
    if not key or not base or not EMBEDDING_MODEL:
        raise RuntimeError("embedding relay configuration is unavailable")
    if client_factory is None:
        from openai import OpenAI
        client_factory = OpenAI
    client = client_factory(api_key=key, base_url=base, timeout=float(os.getenv("REMOTEEVOLVER_EMBEDDING_TIMEOUT", "180")), max_retries=int(os.getenv("REMOTEEVOLVER_EMBEDDING_RETRIES", "1")))
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text or "[no supplied input]" for text in documents],
    )
    vectors = [_unit(list(item.embedding)) for item in response.data]
    usage_obj = getattr(response, "usage", None)
    usage = {
        "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
        "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
    }
    return vectors, usage


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm > 0 else vector


def _cosine(left: list[float], right: list[float]) -> float:
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(left, right))))


def _jaccard(left: set[str], right: set[str], *, both_empty: float = 1.0) -> float:
    if not left and not right:
        return both_empty
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _multiset_jaccard(
    left: list[str], right: list[str], *, both_empty: float = 1.0,
) -> float:
    """Jaccard similarity that preserves repeated semantic roles/contracts."""
    if not left and not right:
        return both_empty
    if not left or not right:
        return 0.0
    left_counts = Counter(left)
    right_counts = Counter(right)
    keys = left_counts.keys() | right_counts.keys()
    intersection = sum(min(left_counts[key], right_counts[key]) for key in keys)
    union = sum(max(left_counts[key], right_counts[key]) for key in keys)
    return intersection / union if union else 0.0


def similarity(
    query: dict[str, Any], candidate: dict[str, Any],
    query_vectors: dict[str, list[float]], candidate_vectors: dict[str, list[float]],
) -> tuple[float, dict[str, float]]:
    fields = {
        name: _cosine(query_vectors[name], candidate_vectors[name])
        for name in FIELD_WEIGHTS
    }
    q_inputs = [_normalize(row["data_kind"]) for row in query["input_roles"]]
    c_inputs = [_normalize(row["data_kind"]) for row in candidate["input_roles"]]
    if not q_inputs and not c_inputs:
        fields["input"] = 1.0
    elif not q_inputs or not c_inputs:
        fields["input"] = 0.0
    else:
        fields["input"] = (
            0.80 * fields["input"]
            + 0.20 * _multiset_jaccard(q_inputs, c_inputs)
        )
    q_outputs = [row["type"] for row in query["output_contract"]]
    c_outputs = [row["type"] for row in candidate["output_contract"]]
    fields["output"] = (
        0.80 * fields["output"]
        + 0.20 * _multiset_jaccard(q_outputs, c_outputs, both_empty=0.0)
    )
    score = sum(FIELD_WEIGHTS[name] * fields[name] for name in FIELD_WEIGHTS)
    return max(0.0, min(1.0, score)), fields
