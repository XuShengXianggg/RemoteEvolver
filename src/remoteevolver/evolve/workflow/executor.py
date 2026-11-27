"""Shared Skill argument resolution and answer synthesis used by compilation and ReAct."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Protocol

from ...llm.base import BaseLLM, Message
from .dataflow import extract_tool_result

_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_DERIVED_OPERATIONS = {"add", "subtract", "multiply", "divide"}
_MAX_DERIVE_DEPTH = 8


class ToolCaller(Protocol):
    """executor 只依赖一个同步 call;MCPClient 正好满足。"""
    def call(self, qualified_name: str, arguments: dict[str, Any]) -> str: ...




# ---- 参数代入 -----------------------------------------------------------

def _resolve_value(
    value: Any,
    bindings: dict[str, Any],
    prior_results: list[str],
    prior_arguments: list[dict[str, Any]],
    *,
    _derive_depth: int = 0,
) -> Any:
    """把一个参数值解析成具体可执行的值。来源:
      - {"$from_step": N, "$field": K} → 取第 N 步输出中的类型化字段 K
      - {"$from_step": N, "$raw": true} → 显式传递第 N 步完整原始字符串
      - {"$from_step": N, "$arg": K} → 复用第 N 步已解析的调用参数 K
      - "{slot}" / 内嵌    → 用 bindings 代入(整体匹配保留类型,内嵌转 str)
      - {"$derive": ...}   → 对已解析值执行白名单二元算术(add/subtract/multiply/divide)
      - list / dict        → **递归**解析每个元素 / 值(否则嵌在列表里的
                             {"$from_step": N}、{slot} 不会被替换,原样传给工具
                             会被 schema 拒,典型如 file_list=[{"$from_step":N}])
      - 其它               → 原样
    N 还没执行到 / 越界时抛 ValueError，由调用方处理。
    """
    if isinstance(value, dict) and "$derive" in value:
        if set(value) != {"$derive"} or not isinstance(value["$derive"], dict):
            raise ValueError("$derive must be the only key and contain an object")
        if _derive_depth >= _MAX_DERIVE_DEPTH:
            raise ValueError(f"$derive nesting exceeds {_MAX_DERIVE_DEPTH}")
        expression = value["$derive"]
        if set(expression) != {"op", "args"}:
            raise ValueError("$derive requires exactly op and args")
        operation = expression.get("op")
        operands = expression.get("args")
        if operation not in _DERIVED_OPERATIONS:
            raise ValueError(f"unsupported $derive operation: {operation!r}")
        if not isinstance(operands, list) or len(operands) != 2:
            raise ValueError("$derive args must be an array of exactly two operands")
        resolved = [
            _resolve_value(
                operand,
                bindings,
                prior_results,
                prior_arguments,
                _derive_depth=_derive_depth + 1,
            )
            for operand in operands
        ]
        left, right = (_finite_number(item) for item in resolved)
        if operation == "add":
            result = left + right
        elif operation == "subtract":
            result = left - right
        elif operation == "multiply":
            result = left * right
        else:
            if right == 0:
                raise ValueError("division by zero in $derive")
            result = left / right
        if not math.isfinite(result):
            raise ValueError("non-finite result in $derive")
        return result
    if isinstance(value, dict) and "$from_step" in value:
        idx = value.get("$from_step")
        if not isinstance(idx, int) or idx < 0 or idx >= len(prior_results):
            raise ValueError(f"$from_step={idx} 不可用(已执行 {len(prior_results)} 步)")
        argument = value.get("$arg")
        if argument is not None:
            if not isinstance(argument, str) or not argument:
                raise ValueError("$arg must be a non-empty string")
            if argument not in prior_arguments[idx]:
                raise ValueError(
                    f"argument {argument!r} not found in step {idx}; "
                    f"available arguments={sorted(prior_arguments[idx])}"
                )
            return prior_arguments[idx][argument]
        return extract_tool_result(
            prior_results[idx],
            field=value.get("$field"),
            raw=value.get("$raw") is True,
        )
    if isinstance(value, str):
        m = _PLACEHOLDER.fullmatch(value.strip())
        if m and m.group(1) in bindings:
            return bindings[m.group(1)]                # 整体替换,保留类型
        return _PLACEHOLDER.sub(
            lambda mo: str(bindings.get(mo.group(1), mo.group(0))), value
        )
    if isinstance(value, list):
        return [
            _resolve_value(
                v, bindings, prior_results, prior_arguments,
                _derive_depth=_derive_depth,
            )
            for v in value
        ]
    if isinstance(value, dict):
        return {
            k: _resolve_value(
                v, bindings, prior_results, prior_arguments,
                _derive_depth=_derive_depth,
            )
            for k, v in value.items()
        }
    return value


def _finite_number(value: Any) -> float:
    """Coerce a tool-produced JSON/scalar number without evaluating expressions."""
    if isinstance(value, bool):
        raise ValueError("boolean is not a numeric $derive operand")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError as exc:
            raise ValueError(f"non-numeric $derive operand: {value!r}") from exc
    else:
        raise ValueError(f"non-numeric $derive operand type: {type(value).__name__}")
    if not math.isfinite(number):
        raise ValueError("non-finite $derive operand")
    return number




def slots_in_value(value: Any) -> set[str]:
    """Return placeholder names without interpreting or executing the value."""
    found: set[str] = set()
    if isinstance(value, str):
        found.update(_PLACEHOLDER.findall(value))
    elif isinstance(value, list):
        for item in value:
            found.update(slots_in_value(item))
    elif isinstance(value, dict):
        for item in value.values():
            found.update(slots_in_value(item))
    return found


# ---- 重放 ---------------------------------------------------------------











# ---- 绑定 + 合成 --------------------------------------------------------





_SYNTH_PROMPT = """\
Task: {task}

This is an independently executed Skill replay. Its tools produced these results:
{results}

Generate the final answer in English using only these replay results. Do not use a
prior answer, trajectory, memory, or unsupported assumptions. If results are
insufficient, say so explicitly and do not invent values, files, or conclusions.
Treat every declared result meaning as binding: reconcile the task's requested
quantity, comparison direction, and unit with what each tool actually returned.
Apply a conversion or complement only when it is valid from the declared meaning.
Return only the final answer.
"""


def synthesize(task: str, results: list[str], llm: BaseLLM) -> str:
    """把重放得到的工具结果整理成最终答案(单次 LLM 调用)。"""
    def view(result: str, limit: int = 2400) -> str:
        text = str(result)
        if len(text) <= limit:
            return text
        return (
            f"{text[:1600]}\n... <truncated; chars={len(text)}> ...\n"
            f"{text[-800:]}"
        )

    joined = "\n".join(f"[{i}] {view(r)}" for i, r in enumerate(results, 1)) or "(no results)"
    prompt = _SYNTH_PROMPT.format(task=task.strip(), results=joined)
    resp = llm.generate([Message(role="user", content=prompt)], tools=None)
    return (resp.text or "").strip()
