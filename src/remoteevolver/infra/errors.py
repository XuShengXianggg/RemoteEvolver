"""Shared classification for failures of external infrastructure."""

from __future__ import annotations

import json


class InfrastructureFailure(RuntimeError):
    """An external service failed; benchmark accuracy must not absorb it."""

    def __init__(self, phase: str, cause: BaseException | str) -> None:
        self.phase = phase
        self.cause = cause
        super().__init__(f"INFRASTRUCTURE_FAILURE phase={phase}: {cause}")


def is_transient_external_error(exc: BaseException) -> bool:
    """Return whether retrying an LLM/GitHub/git/network operation may help."""
    if isinstance(exc, InfrastructureFailure):
        return True
    # At this boundary a JSONDecodeError is raised by an HTTP/SDK client while
    # decoding an OpenAI-compatible response.  Model-content JSON is parsed and
    # handled inside the individual generators, so it must not reach here.
    if isinstance(exc, json.JSONDecodeError):
        return True
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in (408, 409, 425, 429) or status >= 500
    markers = (
        "apiconnectionerror", "apitimeouterror", "ratelimiterror",
        "internalservererror", "serviceunavailableerror", "connection",
        "connecterror", "timeout", "timed out", "temporarily unavailable",
        "rate limit", "service unavailable", "remote end closed",
        "connection reset", "unexpected_eof", "unexpected eof",
        "eof occurred in violation of protocol", "bad gateway",
        "gateway timeout", "network is unreachable", "recv failure",
        "failed to connect", "tls", "ssl",
        "emptycompletionerror", "empty completion",
    )
    return any(marker in name or marker in message for marker in markers)
