"""FastMCP application instance and server lifecycle."""

import json
import logging
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


def _sync_semantic_update() -> None:
    """Check for and run semantic search auto-update (called in a worker thread)."""
    from zotero_mcp.config_light import should_update

    config_path = Path.home() / ".config" / "zotero-mcp" / "config.json"
    if not config_path.exists():
        return

    # Avoid initializing ChromaDB on every server startup when semantic
    # auto-update is disabled. This also avoids racing a foreground
    # zotero_semantic_search call for the same persisted ChromaDB directory.
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        update_cfg = cfg.get("semantic_search", {}).get("update_config", {})
    except Exception:
        return

    if not should_update(update_cfg):
        return

    from zotero_mcp.semantic_search import create_semantic_search

    search = create_semantic_search(str(config_path))
    if not search.should_update_database():
        return

    sys.stderr.write("Auto-updating semantic search database...\n")
    stats = search.update_database()
    sys.stderr.write(
        f"Database update completed: {stats.get('processed_items', 0)} items processed\n"
    )


@asynccontextmanager
async def server_lifespan(server: FastMCP):
    """Manage server startup and shutdown lifecycle.

    Semantic search initialization (ChromaDB + embedding model) is
    offloaded to a daemon thread so it cannot block the event loop.
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
            _sync_semantic_update()
        except Exception as e:
            sys.stderr.write(f"Warning: Could not check semantic search auto-update: {e}\n")

    threading.Thread(
        target=_background_update,
        name="zotero-mcp-semantic-auto-update",
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
