"""
remoteevolver.llm.base —— LLM 客户端抽象 + 通用消息 / 响应数据结构

所有 provider 都实现 BaseLLM 接口；agent 主代码只依赖这个抽象。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None    # role=="tool" 时填


@dataclass
class Response:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)


class BaseLLM(ABC):
    """LLM provider 统一接口。"""

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        stream: bool = False,
    ) -> Response:
        """根据消息历史 + 可用工具 schema 生成下一条响应。"""
        ...

    def generate_required_tool(
        self,
        messages: list[Message],
        tool: dict,
        stream: bool = False,
    ) -> Response:
        """Generate a response that must submit the single supplied tool.

        Providers that support a native ``tool_choice`` should override this
        method.  The fallback keeps lightweight/legacy test doubles working.
        """
        return self.generate(messages, tools=[tool], stream=stream)


def generate_required_tool(
    llm: Any,
    messages: list[Message],
    tool: dict,
    stream: bool = False,
) -> Response:
    """Use the required-tool contract without requiring duck-typed fakes to implement it."""
    method = getattr(llm, "generate_required_tool", None)
    if callable(method):
        return method(messages, tool, stream=stream)
    return llm.generate(messages, tools=[tool], stream=stream)
