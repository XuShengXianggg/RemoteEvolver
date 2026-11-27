"""
mcpx.client —— MCP 客户端(基于 fastmcp 2.x)

按 config.mcp_servers 启动所有 stdio MCP server,自动发现其工具,
对外暴露同步的 schemas() / call() / list_names() API。

实现要点:
- fastmcp.Client 是 asyncio 接口。这里在后台线程里跑一个事件循环,
  对外保持同步接口,避免污染 agent 主循环。
- 工具名通过 "<server>__<tool>" 命名空间化,跨 server 不冲突;
  fastmcp 多 server 模式自带的单下划线前缀(servername_toolname)在这里
  绕开 —— 我们给每个 server 起独立 Client,自己拼前缀。
- 所有 Client 的 enter / exit 都发生在 **同一个长跑协程** 里,
  避免 "Attempted to exit cancel scope in a different task" 这类
  anyio 限制。
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from ..infra.config import MCPServerConfig
from .schema import to_function_schema

NAME_SEP = "__"


def _resolve_stdio_command(command: str) -> str:
    """Keep unqualified Python MCP servers in the Agent environment.

    A bare ``python`` is PATH-dependent and can silently select system Python or
    Conda instead of the interpreter running EvoAgent. Explicit executable paths
    and non-Python commands remain untouched.
    """
    if command.strip().lower() in {"python", "python3"}:
        return sys.executable
    return command


def _stdio_env(configured: dict[str, str]) -> dict[str, str]:
    """Apply deterministic, non-interactive defaults to MCP subprocesses.

    FastMCP's cosmetic banner performs a user-cache read and network version
    check before the JSON-RPC handshake. Those side effects can fail in a
    sandbox even though the tool server itself is healthy. User configuration
    wins when it explicitly chooses different values.
    """
    defaults = {
        "FASTMCP_SHOW_SERVER_BANNER": "false",
        "FASTMCP_CHECK_FOR_UPDATES": "off",
        "PYTHONUTF8": "1",
    }
    # StdioTransport treats an explicit env mapping as the complete child
    # environment. Preserve task-scoped values (notably EVOAGENT_OUTPUT_DIR)
    # set by the evaluation worker, then apply deterministic defaults and
    # finally explicit per-server overrides.
    return {**os.environ, **defaults, **(configured or {})}


@dataclass
class RemoteTool:
    server_name: str
    tool_name: str          # MCP 服务端的原始工具名
    description: str
    input_schema: dict[str, Any]

    @property
    def qualified_name(self) -> str:
        return f"{self.server_name}{NAME_SEP}{self.tool_name}"


class MCPClient:
    def __init__(self, servers: list[MCPServerConfig]) -> None:
        self._servers = servers
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._clients: dict[str, Any] = {}              # server_name -> fastmcp.Client
        self._tools: dict[str, RemoteTool] = {}         # qualified_name -> RemoteTool
        self._started = False

        # lifecycle 同步原语
        self._ready = threading.Event()                 # 启动完成(成功或失败)
        self._stop_event: asyncio.Event | None = None   # 通知 lifecycle 协程退出
        self._lifecycle_future: Any | None = None       # concurrent.futures.Future
        self._start_error: BaseException | None = None
        self._start_timeout: float = float(os.environ.get("REMOTEEVOLVER_MCP_START_TIMEOUT", "15"))               # 单 server 握手超时(秒)
        # A malformed/evolved tool must not consume the whole task wall-clock
        # budget. Keep this below EarthAgent's strict per-task timeout and make
        # it overrideable for legitimately long geoprocessing operations.
        self._call_timeout: float = float(
            os.environ.get("EVOAGENT_TOOL_CALL_TIMEOUT", "180")
        )
        if self._call_timeout <= 0:
            raise ValueError("EVOAGENT_TOOL_CALL_TIMEOUT must be > 0")

    # ---- 生命周期 -------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        if not self._servers:
            self._started = True
            return

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        self._ready.clear()
        self._start_error = None
        self._lifecycle_future = asyncio.run_coroutine_threadsafe(
            self._lifecycle(), self._loop
        )

        # 总超时 = 单 server 握手超时 × server 数,下限 15s。子进程缺依赖 / 路径
        # 错 / 不回 JSON-RPC 握手时,宁可响亮报错也别静默僵死。
        total_timeout = max(self._start_timeout * max(len(self._servers), 1), 15.0)
        if not self._ready.wait(timeout=total_timeout):
            if self._loop is not None and self._stop_event is not None:
                self._loop.call_soon_threadsafe(self._stop_event.set)
            self._stop_loop()
            raise TimeoutError(
                f"MCP server 启动超时({total_timeout:.0f}s)。常见原因:\n"
                f"  - 子进程缺依赖:用 cfg.command 手动跑一遍 cfg.args 看 traceback\n"
                f"    例如:python -m mcpx\n"
                f"  - 工作目录错:args 是相对路径,必须在仓库根运行入口脚本\n"
                f"  - 'command: python' 指向的是 PATH 里的 python,未必是当前 venv —"
                f" 建议在 yaml 里写绝对路径或换成 sys.executable 等价物"
            )

        if self._start_error is not None:
            self._stop_loop()
            raise self._start_error
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        try:
            if self._loop is not None and self._stop_event is not None:
                self._loop.call_soon_threadsafe(self._stop_event.set)
            if self._lifecycle_future is not None:
                try:
                    self._lifecycle_future.result(timeout=10)
                except Exception:
                    # lifecycle 抛错时容忍,stop 不应该把这个再抛一次
                    pass
        finally:
            self._stop_loop()
            self._clients.clear()
            self._tools.clear()
            self._stop_event = None
            self._lifecycle_future = None
            self._started = False

    def restart(self) -> None:
        """重启 MCP 子进程,让新加进 remoteevolver/tools/ 的工具被识别。

        用法:工具自进化注册新文件到 remoteevolver/tools/ 后调用,无需重启整个 Agent。
        实现:stop → 把 _started 重置 → start(它会重启 lifecycle 协程 + 子进程)。
        旧子进程的内存里 import 的旧 tools/__init__.py 会随进程死亡而清空,新子进程
        会重新 `from . import echo, ..., <新工具>`。
        """
        self.stop()
        self.start()

    def __enter__(self) -> "MCPClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ---- 对外同步 API ---------------------------------------------------

    def schemas(
        self,
        whitelist: list[str] | None = None,
        blacklist: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        names = self._filter_names(whitelist, blacklist)
        return [
            to_function_schema(
                qualified_name=t.qualified_name,
                description=t.description,
                input_schema=t.input_schema,
            )
            for t in self._tools.values()
            if t.qualified_name in names
        ]

    def list_names(self) -> list[str]:
        return sorted(self._tools)

    def call(self, qualified_name: str, arguments: dict[str, Any]) -> str:
        if qualified_name not in self._tools:
            raise KeyError(
                f"未注册工具:{qualified_name};已知:{self.list_names()}"
            )
        tool = self._tools[qualified_name]
        client = self._clients[tool.server_name]
        assert self._loop is not None, "MCPClient 尚未 start()"
        fut = asyncio.run_coroutine_threadsafe(
            self._async_call(client, tool.tool_name, arguments), self._loop
        )
        try:
            return fut.result(timeout=self._call_timeout)
        except FutureTimeoutError as exc:
            # Prevent a late result from a cancelled call entering a newer
            # ReAct step as if it belonged to that step.
            fut.cancel()
            raise TimeoutError(
                f"tool call timed out after {self._call_timeout:.0f}s: "
                f"{qualified_name}"
            ) from exc

    # ---- 内部异步实现 ---------------------------------------------------

    async def _lifecycle(self) -> None:
        """启动所有 server,等 stop_event,再统一关停。

        所有 Client 的 enter / exit 都在本协程的同一 task 中完成,
        避免 anyio 的跨 task cancel scope 限制。
        """
        from fastmcp import Client
        from fastmcp.client.transports import StdioTransport

        self._stop_event = asyncio.Event()
        try:
            async with AsyncExitStack() as stack:
                for cfg in self._servers:
                    transport = StdioTransport(
                        command=_resolve_stdio_command(cfg.command),
                        args=list(cfg.args),
                        env=_stdio_env(cfg.env),
                        # This MCPClient owns the process lifecycle. FastMCP's
                        # default keep_alive=True leaves stdio servers behind
                        # after Client.__aexit__, especially after a cancelled
                        # tool call. Always disconnect so mcp's stdio cleanup
                        # can close stdin and terminate the process tree.
                        keep_alive=False,
                    )
                    client = await stack.enter_async_context(Client(transport))
                    self._clients[cfg.name] = client

                    tools = await client.list_tools()
                    for t in tools:
                        rt = RemoteTool(
                            server_name=cfg.name,
                            tool_name=t.name,
                            description=t.description or "",
                            input_schema=t.inputSchema or {"type": "object", "properties": {}},
                        )
                        self._tools[rt.qualified_name] = rt

                # 启动成功,通知 start() 解除阻塞
                self._ready.set()
                # 阻塞在这里直到 stop() 触发;async with 退出时统一清理
                await self._stop_event.wait()
        except BaseException as e:  # noqa: BLE001
            self._start_error = e
            # 确保 start() 不会卡死
            if not self._ready.is_set():
                self._ready.set()
            # 启动阶段出错时,stack 会自然清理;这里不再 re-raise,
            # 避免 future.result() 在 stop() 中再抛一次

    async def _async_call(
        self, client, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        result = await client.call_tool(tool_name, arguments)
        chunks: list[str] = []
        for c in (result.content or []):
            text = getattr(c, "text", None)
            if text:
                chunks.append(text)
        text = "\n".join(chunks).strip() or "(empty result)"
        if getattr(result, "isError", False):
            text = f"[tool error] {text}"
        return text

    # ---- 工具方法 -------------------------------------------------------

    def _filter_names(
        self, whitelist: list[str] | None, blacklist: list[str] | None
    ) -> set[str]:
        names = set(self._tools)
        if whitelist:
            names &= set(whitelist)
        if blacklist:
            names -= set(blacklist)
        return names

    def _stop_loop(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._loop = None
        self._thread = None
