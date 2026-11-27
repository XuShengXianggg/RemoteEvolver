"""
llm —— LLM 客户端模块

本包持有 LLM provider 的注册表(LLM_REGISTRY),并暴露三个便捷符号:
  - register_llm —— 装饰器,各 provider 文件用 @register_llm("xxx")
  - get_llm      —— 按 name 取 provider 类
  - list_llms    —— 列举所有已注册 provider name

import 本包即触发所有 provider 自注册。
"""

from __future__ import annotations

from ..infra.registry import Registry

from .base import BaseLLM, Message, Response, ToolCall, generate_required_tool

# LLM provider 的注册表 —— "LLM 这件事的注册中心"应归 LLM 自己,
# 不再放在 infra/ 里(infra 不应知道有 LLM 这种业务概念)
LLM_REGISTRY: Registry[BaseLLM] = Registry()
register_llm = LLM_REGISTRY.register
get_llm = LLM_REGISTRY.get
list_llms = LLM_REGISTRY.names

# 触发自注册:每个 provider 文件 import 时把自己塞进 LLM_REGISTRY
from . import openai_client  # noqa: F401, E402


__all__ = [
    "BaseLLM", "Message", "Response", "ToolCall", "generate_required_tool",
    "LLM_REGISTRY", "register_llm", "get_llm", "list_llms",
]
