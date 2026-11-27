"""
remoteevolver.memory.long_term —— 长期记忆门面

把后端的 add / search / all / clear 封一层，提供按条目类型筛选与
便捷写入 ExecutionRecord / Experience 的工厂方法。
"""

from __future__ import annotations

from typing import Any

from .base import BaseMemoryBackend, Entry, EntryType


class LongTermMemory:
    def __init__(self, backend: BaseMemoryBackend) -> None:
        self._backend = backend

    # ---- 写入 -----------------------------------------------------------

    def add_execution_record(
        self,
        task: str,
        success: bool,
        summary: str,
        payload: dict[str, Any] | None = None,
    ) -> Entry:
        """每次任务结束写一条（不管成功失败）。"""
        content = f"[task] {task}\n[result] {'OK' if success else 'FAIL'}\n[summary] {summary}"
        entry = Entry.new(
            type="execution_record",
            content=content,
            payload={"task": task, "success": success, "summary": summary, **(payload or {})},
        )
        self._backend.add(entry)
        return entry

    def add_experience(
        self,
        task_signature: str,
        failure_mode: str,
        fix_strategy: str,
        evidence: str = "",
    ) -> Entry:
        """反思提炼出的修复经验。"""
        content = (
            f"[signature] {task_signature}\n"
            f"[failure] {failure_mode}\n"
            f"[fix] {fix_strategy}"
        )
        entry = Entry.new(
            type="experience",
            content=content,
            payload={
                "task_signature": task_signature,
                "failure_mode": failure_mode,
                "fix_strategy": fix_strategy,
                "evidence": evidence,
            },
        )
        self._backend.add(entry)
        return entry

    # ---- 读取 -----------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 3,
        type: EntryType | None = None,
    ) -> list[Entry]:
        return self._backend.search(query, top_k=top_k, type=type)

    def search_experiences(self, query: str, top_k: int = 3) -> list[Entry]:
        return self._backend.search(query, top_k=top_k, type="experience")

    def all(self) -> list[Entry]:
        return self._backend.all()

    def clear(self) -> None:
        self._backend.clear()
