"""Short-term model context with task-safe interaction compaction."""

from __future__ import annotations

from ..llm.base import Message


_SUMMARY_PREFIX = "[Earlier tool interactions were compacted]"
_SUMMARY_MAX_CHARS = 2000


class ShortTermMemory:
    def __init__(self, max_messages: int = 50) -> None:
        if max_messages < 2:
            raise ValueError("max_messages must allow at least two messages")
        self._max = max_messages
        self._msgs: list[Message] = []

    def append(self, msg: Message) -> None:
        self._msgs.append(msg)
        self._truncate()

    def extend(self, msgs: list[Message]) -> None:
        self._msgs.extend(msgs)
        self._truncate()

    def as_messages(self) -> list[Message]:
        return list(self._msgs)

    def clear(self) -> None:
        self._msgs.clear()

    def replace(self, msgs: list[Message]) -> None:
        """Replace the model-visible window without touching the external trace."""
        self._msgs = list(msgs)
        self._truncate()

    def _truncate(self) -> None:
        if len(self._msgs) <= self._max:
            return

        # The initial user message is the task contract. Dropping it makes
        # OpenAI-compatible chat templates reject or misinterpret long ReAct
        # sessions, so keep it alongside the leading system prompt.
        anchors: list[Message] = []
        anchor_ids: set[int] = set()
        if self._msgs and self._msgs[0].role == "system":
            anchors.append(self._msgs[0])
            anchor_ids.add(id(self._msgs[0]))
        original_user = next((msg for msg in self._msgs if msg.role == "user"), None)
        if original_user is not None and id(original_user) not in anchor_ids:
            anchors.append(original_user)
            anchor_ids.add(id(original_user))

        body = [msg for msg in self._msgs if id(msg) not in anchor_ids]
        previous_summary = next(
            (
                msg.content or ""
                for msg in body
                if msg.role == "user"
                and (msg.content or "").startswith(_SUMMARY_PREFIX)
            ),
            "",
        )
        body = [
            msg for msg in body
            if not (
                msg.role == "user"
                and (msg.content or "").startswith(_SUMMARY_PREFIX)
            )
        ]
        chunks = _interaction_chunks(body)

        # Reserve one message for a deterministic summary when old chunks must
        # be removed. Whole chunks are selected newest-first so an assistant
        # tool request is never separated from its tool responses.
        available = max(0, self._max - len(anchors))
        payload_capacity = max(0, available - 1)
        selected: list[list[Message]] = []
        selected_size = 0
        for chunk in reversed(chunks):
            if selected_size + len(chunk) > payload_capacity:
                break
            selected.append(chunk)
            selected_size += len(chunk)
        selected.reverse()
        dropped_count = len(chunks) - len(selected)

        kept_messages = [msg for chunk in selected for msg in chunk]
        dropped_chunks = chunks[:dropped_count]
        summary = _compact_summary(previous_summary, dropped_chunks)
        summary_message = (
            [Message(role="user", content=summary)]
            if summary and available > selected_size else []
        )
        self._msgs = anchors + summary_message + kept_messages


def _interaction_chunks(messages: list[Message]) -> list[list[Message]]:
    """Group each assistant tool request with all immediately following results."""
    chunks: list[list[Message]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role == "tool":
            # A stray result cannot be sent without its assistant tool call.
            index += 1
            continue
        chunk = [message]
        index += 1
        if message.role == "assistant" and message.tool_calls:
            call_ids = {call.id for call in message.tool_calls}
            while (
                index < len(messages)
                and messages[index].role == "tool"
                and messages[index].tool_call_id in call_ids
            ):
                chunk.append(messages[index])
                index += 1
        chunks.append(chunk)
    return chunks


def _compact_summary(previous: str, chunks: list[list[Message]]) -> str:
    if not previous and not chunks:
        return ""
    lines = [_SUMMARY_PREFIX]
    if previous:
        old = previous.removeprefix(_SUMMARY_PREFIX).strip()
        if old:
            lines.append(old)
    for chunk in chunks:
        first = chunk[0]
        if first.role == "assistant" and first.tool_calls:
            names_by_id = {call.id: call.name for call in first.tool_calls}
            for result in chunk[1:]:
                name = names_by_id.get(result.tool_call_id or "", "unknown_tool")
                text = " ".join((result.content or "").split())
                status = "failed" if text.startswith("[tool error]") else "completed"
                lines.append(f"- {name}: {status}; result={text[:240] or '(empty)'}")
            missing = set(names_by_id) - {
                result.tool_call_id for result in chunk[1:] if result.tool_call_id
            }
            for call_id in missing:
                lines.append(f"- {names_by_id[call_id]}: result not recorded")
        elif first.role in ("user", "assistant"):
            text = " ".join((first.content or "").split())
            if text:
                lines.append(f"- earlier {first.role}: {text[:240]}")
    summary = "\n".join(lines)
    if len(summary) > _SUMMARY_MAX_CHARS:
        summary = summary[: _SUMMARY_MAX_CHARS - 16].rstrip() + "\n[summary clipped]"
    return summary
