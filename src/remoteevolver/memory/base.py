"""
remoteevolver.memory.base —— 记忆抽象 + 条目数据结构

长期记忆里两种条目：
  - ExecutionRecord：每次任务的完整执行结果（成功/失败、轨迹、报错）
  - Experience：反思提炼出来的可复用经验（特别是失败 → fix_strategy）

memory 只负责「存与取」，不解释、不提炼（提炼是 reflection 的职责）。
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

EntryType = Literal["execution_record", "experience"]


@dataclass
class Entry:
    """长期记忆的统一条目。"""
    id: str
    type: EntryType
    content: str                          # 用于全文检索的扁平文本
    payload: dict[str, Any] = field(default_factory=dict)  # 结构化字段
    created_at: float = field(default_factory=time.time)

    @staticmethod
    def new(type: EntryType, content: str, payload: dict[str, Any] | None = None) -> "Entry":
        return Entry(
            id=str(uuid.uuid4()),
            type=type,
            content=content,
            payload=payload or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Entry":
        return Entry(
            id=d["id"],
            type=d["type"],
            content=d["content"],
            payload=d.get("payload", {}),
            created_at=d.get("created_at", time.time()),
        )


class BaseMemoryBackend(ABC):
    """长期记忆后端抽象（文件 / 向量库 ...）。"""

    @abstractmethod
    def add(self, entry: Entry) -> None: ...

    @abstractmethod
    def search(self, query: str, top_k: int = 3, type: EntryType | None = None) -> list[Entry]: ...

    @abstractmethod
    def all(self) -> list[Entry]: ...

    @abstractmethod
    def clear(self) -> None: ...
