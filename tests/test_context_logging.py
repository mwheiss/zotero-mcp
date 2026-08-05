"""Regression tests for logging from synchronous MCP tool functions."""

from zotero_mcp._context import context_error, context_info, context_warning


class _SyncContext:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))

    def error(self, message):
        self.messages.append(("error", message))


def test_sync_context_receives_messages():
    context = _SyncContext()

    context_info(context, "started")
    context_warning(context, "careful")
    context_error(context, "failed")

    assert context.messages == [
        ("info", "started"),
        ("warning", "careful"),
        ("error", "failed"),
    ]


def test_async_context_is_logged_without_creating_coroutine(caplog):
    class AsyncContext:
        async def info(self, message):
            raise AssertionError("sync tools must not call an async context method")

    with caplog.at_level("INFO", logger="zotero_mcp.tools"):
        context_info(AsyncContext(), "indexed")

    assert "indexed" in caplog.text


def test_hidden_async_wrapper_result_is_closed(caplog):
    async def emit():
        return None

    class WrappedContext:
        def info(self, message):
            return emit()

    with caplog.at_level("INFO", logger="zotero_mcp.tools"):
        context_info(WrappedContext(), "wrapped")

    assert "wrapped" in caplog.text
