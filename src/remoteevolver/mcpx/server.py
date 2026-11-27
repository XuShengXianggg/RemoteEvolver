"""
remoteevolver.mcpx.server —— 自家 MCP server 启动逻辑

`python -m remoteevolver.mcpx` 会跑到这里的 main()。流程很短:
  1) import remoteevolver.tools(由 tools/__init__.py 触发各工具模块 import,
     每个工具靠 @mcp.tool() 直接挂到 mcpx 里的 mcp 单例上)
  2) mcp.run() 起 stdio loop

启动报告打 stderr 而不是 stdout —— stdio 上 stdout 是 MCP 协议通道,
任何污染都会让客户端解析炸。
"""

from __future__ import annotations

import sys

from . import mcp


def main() -> None:
    # 触发工具注册;装饰器副作用,必须在 mcp.run() 之前 import
    from .. import tools  # noqa: F401

    # 工具数量不在这里读 —— FastMCP 的 list_tools() 是 async,
    # 在同步入口里跑会跟 mcp.run() 起冲突。客户端连上后自己 list_tools() 会知道。
    print("[remoteevolver.mcpx] FastMCP up; entering stdio loop", file=sys.stderr)

    mcp.run()
