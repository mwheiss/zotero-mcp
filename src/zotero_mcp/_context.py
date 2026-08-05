"""FastMCP context compatibility helpers for synchronous Zotero tools."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("zotero_mcp.tools")

try:
    from fastmcp import Context
except ImportError:
    class Context:  # type: ignore[no-redef]
        def info(self, message: str) -> None: pass
        def warning(self, message: str) -> None: pass
        def error(self, message: str) -> None: pass


def _context_log(ctx: Any, level: str, message: object) -> None:
    """Send a context log from a sync tool without leaking a coroutine.

    FastMCP 3 context logging is async while this package deliberately keeps
    its blocking Zotero tools synchronous. Waiting on the server event loop
    from one of those tools can deadlock an in-process or saturated transport,
    so async context methods are mirrored to the server logger instead.
    """
    rendered = str(message)
    method: Callable[..., Any] | None = getattr(ctx, level, None)
    if method is None:
        logger.log(getattr(logging, level.upper(), logging.INFO), rendered)
        return

    if inspect.iscoroutinefunction(method):
        logger.log(getattr(logging, level.upper(), logging.INFO), rendered)
        return

    result = method(rendered)
    if inspect.iscoroutine(result):
        # Defensive support for wrappers whose call signature hides that the
        # return value is async. Closing it prevents RuntimeWarning on GC.
        result.close()
        logger.log(getattr(logging, level.upper(), logging.INFO), rendered)


def context_info(ctx: Any, message: object) -> None:
    _context_log(ctx, "info", message)


def context_warning(ctx: Any, message: object) -> None:
    _context_log(ctx, "warning", message)


def context_error(ctx: Any, message: object) -> None:
    _context_log(ctx, "error", message)


def context_debug(ctx: Any, message: object) -> None:
    _context_log(ctx, "debug", message)
