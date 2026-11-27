"""
mcpx.schema —— MCP tool → LLM function-call schema 转换

MCP server 用 JSON Schema 描述工具入参;OpenAI / Qwen 的 function-calling
接口需要 {"type":"function","function":{...}} 的包装。这里只做这一层转换。
"""

from __future__ import annotations

from typing import Any


def to_function_schema(
    qualified_name: str,
    description: str,
    input_schema: dict[str, Any] | None,
) -> dict[str, Any]:
    """把一个 MCP 工具描述转成 OpenAI function-call schema。"""
    return {
        "type": "function",
        "function": {
            "name": qualified_name,
            "description": description or "",
            "parameters": input_schema or {"type": "object", "properties": {}},
        },
    }
