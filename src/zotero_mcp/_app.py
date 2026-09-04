"""FastMCP application instance and server lifecycle."""

import json
import logging
import math
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastmcp import FastMCP

from zotero_mcp.tool_contract import ToolContractMiddleware
from zotero_mcp.tool_profiles import ToolProfileMiddleware

# Configure logging from environment variable
# Set ZOTERO_MCP_LOG_LEVEL=DEBUG in Claude Desktop config to enable debug logs
_log_level = os.environ.get("ZOTERO_MCP_LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.WARNING),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)


def _bounded_number(
    value: object,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return max(minimum, min(parsed, maximum))


def _sync_schema_refresh(config_path: Path | None = None) -> None:
    """Run the opt-in logical metadata-schema refresh when it is due."""
    config_path = config_path or (
        Path.home() / ".config" / "zotero-mcp" / "config.json"
    )
    if not config_path.exists():
        return

    try:
        with open(config_path, encoding="utf-8") as config_file:
            config = json.load(config_file)
        refresh_config = config.get("schema_refresh", {})
    except Exception:
        return
    if (
        not isinstance(refresh_config, dict)
        or refresh_config.get("auto_refresh") is not True
    ):
        return

    source = refresh_config.get("source", "auto")
    if source not in {"auto", "local", "web"}:
        source = "auto"
    interval_days = _bounded_number(
        refresh_config.get("interval_days", 7),
        default=7,
        minimum=1 / 24,
        maximum=365,
    )
    failure_backoff_hours = _bounded_number(
        refresh_config.get("failure_backoff_hours", 24),
        default=24,
        minimum=1,
        maximum=7 * 24,
    )

    from zotero_mcp import schema

    result = schema.refresh_if_due(
        source=source,
        interval_seconds=interval_days * 24 * 60 * 60,
        failure_backoff_seconds=failure_backoff_hours * 60 * 60,
    )
    version = result.get("schema_version")
    version_text = f"schema {version}" if version else "built-in fallback"
    message = (
        "Automatic Zotero metadata schema refresh: "
        f"{result['status']} ({result['source']}, {version_text})"
    )
    if result["status"] == "error":
        message += f"; {result.get('error', 'unknown error')}"
    sys.stderr.write(message + "\n")


def _sync_semantic_update(config_path: Path | None = None) -> None:
    """Check for and run semantic search auto-update (called in a worker thread)."""
    from zotero_mcp.config_light import (
        auto_update_embedding_concurrency,
        should_update,
    )

    config_path = config_path or (
        Path.home() / ".config" / "zotero-mcp" / "config.json"
    )
    if not config_path.exists():
        return

    # Avoid initializing ChromaDB on every server startup when semantic
    # auto-update is disabled. This also avoids racing a foreground
    # zotero_semantic_search call for the same persisted ChromaDB directory.
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        semantic_cfg = cfg.get("semantic_search", {})
        update_cfg = semantic_cfg.get("update_config", {})
    except Exception:
        return

    if not should_update(update_cfg):
        return

    from zotero_mcp.semantic_search import create_semantic_search

    search = create_semantic_search(str(config_path))
    if not search.should_update_database():
        return

    sys.stderr.write("Auto-updating semantic search database...\n")
    stats = search.update_database(
        fulltext=bool(semantic_cfg.get("fulltext", False)),
        embedding_concurrency=auto_update_embedding_concurrency(update_cfg),
    )
    sys.stderr.write(
        f"Database update completed: {stats.get('processed_items', 0)} items processed\n"
    )


@asynccontextmanager
async def server_lifespan(server: FastMCP):
    """Manage server startup and shutdown lifecycle.

    Logical-schema and semantic-search maintenance are offloaded to a daemon
    thread so they cannot block the event loop.
    The previous synchronous call prevented FastMCP from responding
    to the MCP ``initialize`` request within the 60-second client
    timeout.

    On shutdown the worker thread is left to finish on its own. ChromaDB
    (SQLite WAL) is crash-safe, so an unfinished update simply resumes on the
    next startup. A dedicated daemon avoids making event-loop shutdown wait on
    Python's default executor.
    """
    sys.stderr.write("Starting Zotero MCP server...\n")

    def _background_update():
        try:
            _sync_schema_refresh()
        except Exception as e:
            sys.stderr.write(
                f"Warning: Could not check Zotero metadata schema refresh: {e}\n"
            )
        try:
            _sync_semantic_update()
        except Exception as e:
            sys.stderr.write(f"Warning: Could not check semantic search auto-update: {e}\n")

    threading.Thread(
        target=_background_update,
        name="zotero-mcp-background-maintenance",
        daemon=True,
    ).start()

    yield {}

    sys.stderr.write("Shutting down Zotero MCP server...\n")


# Create an MCP server (fastmcp 2.14+ no longer accepts `dependencies`).
mcp = FastMCP(
    "Zotero",
    instructions=(
        "Start with zotero_get_capabilities when availability is uncertain. "
        "Use zotero_search_items for known metadata, zotero_semantic_search for "
        "concepts, then zotero_get_semantic_context for evidence. Retrieve whole "
        "full text only when document-wide reading is necessary. Destructive "
        "semantic rebuilds require explicit user confirmation."
    ),
    lifespan=server_lifespan,
)
mcp.add_middleware(ToolProfileMiddleware())
mcp.add_middleware(ToolContractMiddleware())
