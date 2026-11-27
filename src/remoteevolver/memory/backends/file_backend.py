"""
remoteevolver.memory.backends.file_backend —— JSON 文件后端

最简单的长期记忆持久化：所有条目存到一个 JSON 文件，
search 用关键词重叠 + 时间近度做粗排（够 v0.2 验证闭环）。
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path

from .. import register_memory_backend
from ..base import BaseMemoryBackend, Entry, EntryType

_WORD = re.compile(r"\w+", re.UNICODE)


@register_memory_backend("file")
class FileBackend(BaseMemoryBackend):
    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[Entry] = self._load()

    def add(self, entry: Entry) -> None:
        self._entries.append(entry)
        self._save()

    def search(
        self,
        query: str,
        top_k: int = 3,
        type: EntryType | None = None,
    ) -> list[Entry]:
        pool = [e for e in self._entries if type is None or e.type == type]
        if not pool:
            return []

        q_tokens = _tokenize(query)
        if not q_tokens:
            # 查询为空 → 返回最近的 top_k 条
            return sorted(pool, key=lambda e: e.created_at, reverse=True)[:top_k]

        def score(e: Entry) -> float:
            tokens = _tokenize(e.content)
            if not tokens:
                return 0.0
            overlap = len(q_tokens & tokens)
            return overlap + 0.001 * e.created_at   # 平手时偏新

        ranked = sorted(pool, key=score, reverse=True)
        return [e for e in ranked[:top_k] if score(e) > 0]

    def all(self) -> list[Entry]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self._save()

    # ---- internals ------------------------------------------------------

    def _load(self) -> list[Entry]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        entries: list[Entry] = []
        skipped = 0
        for d in data:
            try:
                entries.append(Entry.from_dict(d))
            except (KeyError, TypeError):
                skipped += 1
        if skipped:
            print(
                f"[memory] {self._path}: 跳过 {skipped} 条与当前 schema 不兼容的旧记录"
            )
        return entries

    def _save(self) -> None:
        data = [e.to_dict() for e in self._entries]
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        # A cold-start harness can replace its isolated state directory between
        # workers. Re-create the parent at the actual write boundary and publish
        # atomically so readers never observe a missing/partial memory file.
        # Retry once if a concurrent directory replacement lands between mkdir
        # and replace.
        last_error: FileNotFoundError | None = None
        for _ in range(2):
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temp = self._path.with_name(
                f".{self._path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            try:
                temp.write_text(payload, encoding="utf-8")
                os.replace(temp, self._path)
                return
            except FileNotFoundError as exc:
                last_error = exc
                temp.unlink(missing_ok=True)
        assert last_error is not None
        raise last_error


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD.finditer(text or "")}
