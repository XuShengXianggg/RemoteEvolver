"""Cold-start MCP integration probe for a newly generated tool."""

from __future__ import annotations

import sys
import re
from typing import Any

from ...infra.config import MCPServerConfig
from ...mcpx.client import MCPClient


def _looks_like_error_result(result: str) -> bool:
    return bool(re.match(
        r"^\s*\[(?:tool|[A-Za-z_]\w*)\s+error\]", result, re.IGNORECASE,
    ))


def probe_registered_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Prove discovery, ReAct-visible schema and real MCP invocation in a fresh process."""
    qualified = f"local__{tool_name}"
    client = MCPClient([MCPServerConfig(name="local", command=sys.executable,
                                        args=["-m", "remoteevolver.mcpx"])])
    try:
        client.start()
        if qualified not in client.list_names():
            raise RuntimeError(f"cold-start MCP did not discover {qualified}")
        schemas = client.schemas(whitelist=[qualified])
        if not schemas or schemas[0].get("function", {}).get("name") != qualified:
            raise RuntimeError(f"{qualified} is not visible as a ReAct function schema")
        result = client.call(qualified, arguments)
        if _looks_like_error_result(result):
            raise RuntimeError(f"real MCP invocation failed: {result[:500]}")
        return {"qualified_name": qualified, "schema": schemas[0], "result": result}
    finally:
        client.stop()
