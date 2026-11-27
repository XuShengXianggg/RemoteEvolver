"""
remoteevolver.llm.openai_client —— OpenAI / 兼容 OpenAI 协议的后端

适用：OpenAI 官方 API；任何提供 /v1/chat/completions 兼容接口的服务
（vLLM / Ollama OpenAI-compat / 各种中转）—— 切换只需改 base_url。
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from typing import Any

from . import register_llm
from .base import BaseLLM, Message, Response, ToolCall


class TokenUsageRecorder:
    """Keep a process-local audit trail for every logical LLM request.

    The OpenAI-compatible response is the authoritative source for token
    counts.  Failed requests are still recorded, but their token fields stay
    ``None`` because the provider did not return billable usage metadata.
    """

    def _ensure_usage_recorder(self) -> None:
        if not hasattr(self, "_usage_events"):
            self._usage_events: list[dict[str, Any]] = []
            self._usage_sequence = 0
            self._usage_lock = threading.Lock()

    def usage_cursor(self) -> int:
        self._ensure_usage_recorder()
        with self._usage_lock:
            return int(self._usage_sequence)

    def _record_usage(
        self, *, response: Response | None, elapsed_seconds: float,
        api_attempts: int, tool_count: int, error: BaseException | None = None,
    ) -> None:
        self._ensure_usage_recorder()
        raw_usage = dict(getattr(response, "usage", None) or {})
        available = bool(raw_usage)
        prompt_tokens = (
            int(raw_usage.get("prompt_tokens", 0) or 0) if available else None
        )
        completion_tokens = (
            int(raw_usage.get("completion_tokens", 0) or 0) if available else None
        )
        total_tokens = (
            int(raw_usage.get(
                "total_tokens", (prompt_tokens or 0) + (completion_tokens or 0),
            ) or 0)
            if available else None
        )
        with self._usage_lock:
            self._usage_sequence += 1
            self._usage_events.append({
                "request_index": self._usage_sequence,
                "model": str(getattr(self, "_model", "")),
                "success": error is None,
                "api_attempts": max(1, int(api_attempts)),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "usage_reported": available,
                "elapsed_seconds": round(float(elapsed_seconds), 6),
                "tool_count": int(tool_count),
                "finish_reason": (
                    str(getattr(response, "finish_reason", ""))
                    if response is not None else ""
                ),
                **({"error_type": type(error).__name__} if error is not None else {}),
            })

    def usage_since(self, cursor: int = 0) -> dict[str, Any]:
        """Return per-request records and exact aggregates after ``cursor``."""
        self._ensure_usage_recorder()
        with self._usage_lock:
            requests = [
                dict(event) for event in self._usage_events
                if int(event["request_index"]) > int(cursor)
            ]
        reported = [row for row in requests if row.get("usage_reported")]
        return {
            "request_count": len(requests),
            "successful_request_count": sum(bool(row.get("success")) for row in requests),
            "failed_request_count": sum(not bool(row.get("success")) for row in requests),
            "api_attempt_count": sum(int(row.get("api_attempts", 1)) for row in requests),
            "usage_reported_request_count": len(reported),
            "usage_missing_request_count": len(requests) - len(reported),
            "prompt_tokens": sum(int(row.get("prompt_tokens", 0) or 0) for row in reported),
            "completion_tokens": sum(
                int(row.get("completion_tokens", 0) or 0) for row in reported
            ),
            "total_tokens": sum(int(row.get("total_tokens", 0) or 0) for row in reported),
            "requests": requests,
        }


class EmptyCompletionError(RuntimeError):
    """An OpenAI-compatible gateway returned no assistant content or action."""


class MissingRequiredToolCallError(RuntimeError):
    """A required-tool request completed without an actual tool call."""


def _call_with_wall_timeout(call, *, timeout_s: float | None):
    """Bound a blocking SDK call even when its transport timeout is ignored."""
    if timeout_s is None or timeout_s <= 0:
        return call()
    result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def target() -> None:
        try:
            result_queue.put((True, call()))
        except BaseException as exc:  # noqa: BLE001
            result_queue.put((False, exc))

    worker = threading.Thread(target=target, daemon=True, name="remoteevolver-llm-call")
    worker.start()
    try:
        ok, value = result_queue.get(timeout=timeout_s)
    except queue.Empty as exc:
        raise TimeoutError(
            f"LLM request exceeded wall-clock limit of {timeout_s:g}s"
        ) from exc
    if ok:
        return value
    raise value


def _is_transient_api_error(exc: BaseException) -> bool:
    # OpenAI-compatible gateways can occasionally return a truncated/non-JSON
    # HTTP body.  The SDK surfaces that as JSONDecodeError before a completion
    # object exists, so retrying the side-effect-free request is appropriate.
    if isinstance(exc, json.JSONDecodeError):
        return True
    if isinstance(exc, EmptyCompletionError):
        return True
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in (408, 409, 425, 429) or status >= 500
    return any(
        marker in name or marker in message
        for marker in (
            "connection", "timeout", "timed out", "rate limit",
            "temporarily unavailable", "bad gateway", "gateway timeout",
            "unexpected_eof", "eof occurred in violation of protocol",
        )
    )


def _is_timeout_error(exc: BaseException) -> bool:
    """Recognize builtin and OpenAI-compatible timeout exceptions."""
    if isinstance(exc, TimeoutError):
        return True
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return "timeout" in name or "timed out" in message


def _record_infra_signal(kind: str, exc: BaseException) -> None:
    """Append to the harness infra-event file (runtime_isolation/infra_signals.py)."""
    path = os.environ.get("RSEVO_INFRA_EVENTS_PATH", "").strip()
    if not path:
        return
    try:
        line = json.dumps({"time": time.time(), "pid": os.getpid(), "kind": kind,
                           "detail": f"{type(exc).__name__}: {exc}"[:1000]})
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(line + "\n")
    except OSError:
        pass


def _create_with_transient_retries(
    create,
    kwargs: dict[str, Any],
    *,
    attempts: int = 2,
    backoff_s: tuple[float, ...] = (2.0, 5.0),
    sleep=time.sleep,
    retry_timeouts: bool = True,
):
    """对单次无副作用的模型请求做显式瞬态重试，并留下可分析日志。"""
    if attempts <= 0:
        raise ValueError("attempts must be > 0")
    for attempt in range(1, attempts + 1):
        try:
            return create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            transient = _is_transient_api_error(exc)
            if transient:
                _record_infra_signal("api_transient", exc)
            if _is_timeout_error(exc) and not retry_timeouts:
                _record_infra_signal("api_failed", exc)
                raise
            if not transient or attempt == attempts:
                if transient:
                    _record_infra_signal("api_failed", exc)
                raise
            delay = backoff_s[min(attempt - 1, len(backoff_s) - 1)] if backoff_s else 0.0
            print(
                f"[llm] TRANSIENT_RETRY attempt {attempt}/{attempts}: "
                f"{type(exc).__name__}: {exc}; backoff={delay:.1f}s",
                flush=True,
            )
            sleep(delay)
    raise AssertionError("unreachable")


@register_llm("openai")
class OpenAIClient(TokenUsageRecorder, BaseLLM):
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
        max_retries: int = 0,
        call_attempts: int = 2,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("OpenAIClient 需要 openai 包：pip install openai") from e

        self._model = model
        self._temperature = temperature
        self._call_attempts = int(os.environ.get("REMOTEEVOLVER_MAIN_ATTEMPTS", str(call_attempts)))
        self._ensure_usage_recorder()
        transport_timeout_raw = os.environ.get(
            "EVOAGENT_LLM_TRANSPORT_TIMEOUT_SECONDS", str(timeout)
        ).strip()
        self._transport_timeout_s = float(transport_timeout_raw or str(timeout))
        wall_timeout_raw = os.environ.get(
            "EVOAGENT_LLM_CALL_WALL_TIMEOUT_SECONDS", "0"
        ).strip()
        self._wall_timeout_s = float(wall_timeout_raw or "0")
        self._retry_timeouts = os.environ.get(
            "EVOAGENT_LLM_RETRY_TIMEOUTS", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        max_tokens_raw = os.environ.get(
            "EVOAGENT_LLM_MAX_COMPLETION_TOKENS", "0"
        ).strip()
        self._max_completion_tokens = int(max_tokens_raw or "0")
        if self._max_completion_tokens < 0:
            raise ValueError("EVOAGENT_LLM_MAX_COMPLETION_TOKENS must be non-negative")
        chat_template_raw = os.environ.get(
            "EVOAGENT_LLM_CHAT_TEMPLATE_KWARGS", ""
        ).strip()
        self._chat_template_kwargs: dict[str, Any] = {}
        if chat_template_raw:
            parsed = json.loads(chat_template_raw)
            if not isinstance(parsed, dict):
                raise ValueError("EVOAGENT_LLM_CHAT_TEMPLATE_KWARGS must be a JSON object")
            self._chat_template_kwargs = parsed
        self._required_tool_thinking_policy = os.environ.get(
            "EVOAGENT_LLM_REQUIRED_TOOL_THINKING_POLICY", "preserve"
        ).strip().lower()
        if self._required_tool_thinking_policy not in {"preserve", "disable", "auto"}:
            raise ValueError(
                "EVOAGENT_LLM_REQUIRED_TOOL_THINKING_POLICY must be one of: "
                "preserve, disable, auto"
            )
        # 重试只由本客户端的 call_attempts 统一管理。SDK 重试默认关闭，
        # 避免“外层两次 × SDK 两次”的嵌套放大。
        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def generate(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        stream: bool = False,
    ) -> Response:
        return self._generate(messages, tools=tools, stream=stream)

    def generate_required_tool(
        self,
        messages: list[Message],
        tool: dict,
        stream: bool = False,
    ) -> Response:
        return self._generate(
            messages, tools=[tool], stream=stream, tool_choice="required",
            require_tool_call=True,
        )

    def _generate(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        stream: bool = False,
        tool_choice: str | dict[str, Any] | None = None,
        require_tool_call: bool = False,
    ) -> Response:
        if stream:
            raise NotImplementedError("Streaming not implemented in v0.2")

        # Constructor-free subclasses and lightweight tests predate the
        # per-request timeout field.  Preserve the historical 60-second value
        # for those clients instead of requiring every implementation to grow
        # an OpenAIClient-private attribute.
        transport_timeout = float(getattr(self, "_transport_timeout_s", 60.0))
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [_to_openai_message(m) for m in messages],
            "temperature": self._temperature,
            "timeout": transport_timeout,
        }
        # Pass this per request so auxiliary callers (for example the
        # non-fatal reflector) can temporarily tighten the transport budget
        # without rebuilding the SDK client.  ``getattr`` preserves backward
        # compatibility with lightweight subclasses and test doubles created
        # without running this concrete client's constructor.
        if tools:
            kwargs["tools"] = tools
        if self._max_completion_tokens:
            kwargs["max_tokens"] = self._max_completion_tokens
        chat_template_kwargs = dict(getattr(self, "_chat_template_kwargs", {}))
        effective_tool_choice = tool_choice
        required_tool_policy = getattr(
            self, "_required_tool_thinking_policy", "preserve"
        )
        if tool_choice == "required" and required_tool_policy == "disable":
            chat_template_kwargs["enable_thinking"] = False
            chat_template_kwargs.pop("reasoning_effort", None)
        elif tool_choice == "required" and required_tool_policy == "auto":
            effective_tool_choice = "auto"
        if chat_template_kwargs:
            kwargs["extra_body"] = {
                "chat_template_kwargs": chat_template_kwargs,
            }
        if effective_tool_choice is not None:
            kwargs["tool_choice"] = effective_tool_choice

        api_attempts = 0

        def create_and_decode(**request_kwargs: Any) -> Response:
            nonlocal api_attempts
            api_attempts += 1
            raw = _call_with_wall_timeout(
                lambda: self._client.chat.completions.create(**request_kwargs),
                timeout_s=self._wall_timeout_s,
            )
            if raw is None or not getattr(raw, "choices", None):
                raise EmptyCompletionError(
                    "OpenAI-compatible gateway returned no completion object or choices"
                )
            response = _from_openai_response(raw)
            diagnostics = _completion_diagnostics(raw)
            if require_tool_call and not response.tool_calls:
                raise MissingRequiredToolCallError(
                    "Required-tool completion contained no tool call; " + diagnostics
                )
            if not response.text.strip() and not response.tool_calls:
                raise EmptyCompletionError(
                    "OpenAI-compatible gateway returned an empty completion; "
                    + diagnostics
                )
            return response

        started = time.monotonic()
        input_chars = sum(
            len(str(message.get("content", ""))) for message in kwargs["messages"]
        )
        print(
            f"[llm] REQUEST_START model={self._model} messages={len(kwargs['messages'])} "
            f"input_chars={input_chars} tools={len(tools or [])} "
            f"attempts={self._call_attempts} "
            f"transport_timeout={transport_timeout:g} "
            f"wall_timeout={self._wall_timeout_s:g}",
            flush=True,
        )
        try:
            response = _create_with_transient_retries(
                create_and_decode, kwargs, attempts=self._call_attempts,
                retry_timeouts=bool(getattr(self, "_retry_timeouts", True)),
            )
        except Exception as exc:
            self._record_usage(
                response=None, elapsed_seconds=time.monotonic() - started,
                api_attempts=api_attempts, tool_count=len(tools or []), error=exc,
            )
            print(
                f"[llm] REQUEST_FAIL model={self._model} "
                f"elapsed={time.monotonic() - started:.3f}s "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )
            raise
        self._record_usage(
            response=response, elapsed_seconds=time.monotonic() - started,
            api_attempts=api_attempts, tool_count=len(tools or []),
        )
        print(
            f"[llm] REQUEST_END model={self._model} "
            f"elapsed={time.monotonic() - started:.3f}s",
            flush=True,
        )
        return response


# ---- helpers shared with qwen (兼容 OpenAI 协议) ------------------------

def _to_openai_message(m: Message) -> dict[str, Any]:
    if m.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": m.tool_call_id,
            "content": m.content or "",
        }
    d: dict[str, Any] = {"role": m.role, "content": m.content or ""}
    if m.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                },
            }
            for tc in m.tool_calls
        ]
        d["content"] = m.content or None
    return d


def _from_openai_response(raw: Any) -> Response:
    choice = raw.choices[0]
    msg = choice.message

    tcs: list[ToolCall] = []
    for tc in (msg.tool_calls or []):
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {"_raw": tc.function.arguments}
        tcs.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

    usage: dict[str, int] = {}
    if raw.usage:
        usage = {
            "prompt_tokens": raw.usage.prompt_tokens,
            "completion_tokens": raw.usage.completion_tokens,
            "total_tokens": getattr(
                raw.usage, "total_tokens",
                raw.usage.prompt_tokens + raw.usage.completion_tokens,
            ),
        }

    return Response(
        text=msg.content or "",
        tool_calls=tcs,
        finish_reason=choice.finish_reason or "stop",
        usage=usage,
    )


def _completion_diagnostics(raw: Any) -> str:
    """Return safe metadata for malformed compatible-gateway responses."""
    try:
        choice = raw.choices[0]
        msg = choice.message
        content = getattr(msg, "content", None) or ""
        raw_calls = getattr(msg, "tool_calls", None) or []
        reasoning = (
            getattr(msg, "reasoning", None)
            or getattr(msg, "reasoning_content", None)
            or ""
        )
        if not reasoning and hasattr(msg, "model_dump"):
            dumped = msg.model_dump()
            reasoning = dumped.get("reasoning") or dumped.get("reasoning_content") or ""
        return (
            f"finish_reason={getattr(choice, 'finish_reason', None)!r}, "
            f"content_chars={len(str(content))}, reasoning_chars={len(str(reasoning))}, "
            f"tool_calls={len(raw_calls)}"
        )
    except Exception:  # noqa: BLE001
        return "response_diagnostics=unavailable"
