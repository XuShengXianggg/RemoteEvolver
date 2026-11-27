"""Console output helpers that never crash an agent because of terminal encoding."""

from __future__ import annotations

import builtins
import sys
from typing import Any


def safe_print(*values: Any, **kwargs: Any) -> None:
    """Behave like ``print`` but replace unsupported glyphs on legacy consoles.

    Windows terminals or redirected log streams may still use GBK. Diagnostic
    decorations must never abort task execution with ``UnicodeEncodeError``.
    """
    try:
        builtins.print(*values, **kwargs)
        return
    except UnicodeEncodeError:
        pass

    stream = kwargs.get("file") or sys.stdout
    encoding = (
        getattr(stream, "encoding", None)
        or getattr(getattr(stream, "_stream", None), "encoding", None)
        or "ascii"
    )

    def clean(value: Any) -> str:
        text = str(value)
        return text.encode(encoding, errors="replace").decode(encoding, errors="replace")

    builtins.print(*(clean(value) for value in values), **kwargs)
