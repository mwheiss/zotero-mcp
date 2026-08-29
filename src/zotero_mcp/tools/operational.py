"""Operational discovery and read-only semantic database diagnostics."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from zotero_mcp import client as _client
from zotero_mcp import utils as _utils
from zotero_mcp._app import mcp
from zotero_mcp._context import Context, context_error, context_info
from zotero_mcp.tool_profiles import (
    WRITE_TOOLS,
    effective_tool_profile,
    requested_tool_profile,
    tool_visible,
)


@mcp.tool(
    name="zotero_get_capabilities",
    description=(
        "Inspect the active Zotero MCP contract before choosing a workflow. "
        "Reports the requested/effective tool profile, active library, local "
        "versus Web API mode, write availability, semantic/PDF optional "
        "dependencies, full-text source availability, and whether local paths "
        "are exposed. Read-only and available in every profile."
    ),
)
def get_capabilities(*, ctx: Context) -> str:
    context_info(ctx, "Inspecting Zotero MCP capabilities")
    library = _client.get_current_library()
    local_mode = _utils.is_local_mode()
    api_key = bool(os.getenv("ZOTERO_API_KEY"))
    semantic_available = importlib.util.find_spec("chromadb") is not None
    pdf_available = importlib.util.find_spec("fitz") is not None
    requested = requested_tool_profile()
    effective = effective_tool_profile()
    local_paths = tool_visible("zotero_get_attachment_path", effective)
    local_write_available = (
        _client.get_local_write_zotero_client() is not None
        if local_mode
        else False
    )
    write_usable = (local_write_available or api_key) and any(
        tool_visible(tool_name, effective) for tool_name in WRITE_TOOLS
    )

    lines = [
        "# Zotero MCP Capabilities",
        "",
        f"**Tool profile:** {effective} (requested: {requested})",
        f"**Active library:** {library.get('library_type', 'user')}:{library.get('library_id', '')}",
        f"**Zotero access:** {'local desktop API' if local_mode else 'Web API'}",
        f"**Write transport:** "
        f"{'local desktop API' if local_write_available else 'Web API fallback' if api_key else 'unavailable'}",
        f"**Write tools usable:** {'yes' if write_usable else 'no'}",
        f"**Semantic search installed:** {'yes' if semantic_available else 'no'}",
        f"**PDF page/outline support:** {'yes' if pdf_available else 'no'}",
        f"**Local full-text extraction:** {'yes' if local_mode else 'no'}",
        f"**Local paths exposed:** {'yes' if local_paths else 'no'}",
        "",
        "Use metadata search for known titles/authors, semantic search for topics, "
        "semantic context for matched evidence, and whole-item full text only for "
        "document-wide reading.",
    ]
    return "\n".join(lines)


@mcp.tool(
    name="zotero_get_search_database_health",
    description=(
        "Run a read-only integrity audit of the active library's semantic "
        "collection. Verifies SQLite, Chroma records, passage groups, content "
        "hashes, HNSW replay state, orphan segment directories, and optional "
        "coverage against the local Zotero snapshot. It never repairs or "
        "deletes data. quick=True skips the slower full SQLite integrity check; "
        "compare_zotero=False skips source coverage comparison."
    ),
)
def get_search_database_health(
    quick: bool = True,
    compare_zotero: bool = True,
    *,
    ctx: Context,
) -> str:
    context_info(ctx, "Auditing semantic search database health")
    try:
        from zotero_mcp.db_health import (
            audit_semantic_database,
            format_health_report,
        )

        config_path = Path.home() / ".config" / "zotero-mcp" / "config.json"
        report = audit_semantic_database(
            config_path,
            full_integrity=not quick,
            compare_zotero=compare_zotero,
            library=_client.get_current_library(),
        )
        return format_health_report(report)
    except Exception as exc:
        context_error(ctx, f"Semantic database health audit failed: {exc}")
        return f"Error: Semantic database health audit failed: {exc}"
