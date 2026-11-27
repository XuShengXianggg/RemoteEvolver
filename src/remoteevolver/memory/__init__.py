"""
memory —— 短期 + 长期记忆

本包持有 memory 后端的注册表(MEMORY_BACKEND_REGISTRY),并暴露:
  - register_memory_backend —— 装饰器,各后端文件用 @register_memory_backend("xxx")
  - get_memory_backend      —— 按 name 取后端类
  - list_memory_backends    —— 列举所有已注册后端 name

import 本包即触发各后端自注册。
"""

from __future__ import annotations

from ..infra.registry import Registry

from .base import BaseMemoryBackend, Entry, EntryType
from .long_term import LongTermMemory
from .short_term import ShortTermMemory

# memory 后端的注册表 —— 业务概念归业务模块管,不在 infra/ 写死白名单
MEMORY_BACKEND_REGISTRY: Registry[BaseMemoryBackend] = Registry()
register_memory_backend = MEMORY_BACKEND_REGISTRY.register
get_memory_backend = MEMORY_BACKEND_REGISTRY.get
list_memory_backends = MEMORY_BACKEND_REGISTRY.names

# 触发后端自注册
from . import backends  # noqa: F401, E402

__all__ = [
    "BaseMemoryBackend", "Entry", "EntryType",
    "ShortTermMemory", "LongTermMemory",
    "MEMORY_BACKEND_REGISTRY",
    "register_memory_backend", "get_memory_backend", "list_memory_backends",
]
