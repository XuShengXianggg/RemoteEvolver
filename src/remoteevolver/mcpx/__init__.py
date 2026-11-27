"""
mcpx —— 本项目的 MCP 协议层。

包含两块:
  - server 侧:暴露 mcp = FastMCP("local") 单例;tools/ 下的工具文件通过
    `from mcpx import mcp` + `@mcp.tool()` 完成注册;
    入口 `python -m mcpx` 启动 stdio server。
  - client 侧:MCPClient 给 agent 调用任意 MCP server(自家 + 外部)。

不混业务、不混基建,只讲一件事:"项目里跟 MCP 协议打交道的代码"。
"""

from __future__ import annotations

from fastmcp import FastMCP

# 共享单例 —— 工具文件靠 `from mcpx import mcp` 拿到它,然后 @mcp.tool() 注册。
# 名字 "local" 与 configs/default.yaml 里 mcp_servers[].name 对齐,
# 用作工具的 server 前缀(qualified_name = "local__<tool>")。
mcp = FastMCP("local")

# client 在子模块里实现;在这里 re-export 给 agent 用更顺手。
from .client import MCPClient, RemoteTool, NAME_SEP  # noqa: E402

__all__ = ["mcp", "MCPClient", "RemoteTool", "NAME_SEP"]
