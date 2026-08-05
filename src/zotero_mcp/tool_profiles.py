"""Runtime MCP tool profiles and capability-aware tool filtering."""

from __future__ import annotations

import importlib.util
import os
from typing import Literal

from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware, MiddlewareContext

ToolProfile = Literal["auto", "research", "full", "admin", "connector", "all"]

CONNECTOR_TOOLS = {"search", "fetch"}
LOCAL_PATH_TOOLS = {"zotero_get_attachment_path"}
PDF_TOOLS = {
    "zotero_read_pdf_pages",
    "zotero_get_pdf_outline",
    "zotero_get_page_layout",
    "zotero_create_area_annotation",
}
SEMANTIC_TOOLS = {
    "zotero_semantic_search",
    "zotero_get_semantic_context",
    "zotero_update_search_database",
    "zotero_get_search_database_status",
    "zotero_get_search_database_health",
}
FEED_TOOLS = {"zotero_list_feeds", "zotero_get_feed_items"}
WRITE_TOOLS = {
    "zotero_create_note",
    "zotero_update_note",
    "zotero_delete_note",
    "zotero_create_annotation",
    "zotero_create_area_annotation",
    "zotero_update_annotation",
    "zotero_delete_annotation",
    "zotero_batch_update_tags",
    "zotero_batch_update_extra",
    "zotero_create_collection",
    "zotero_delete_collection",
    "zotero_manage_collections",
    "zotero_add_by_doi",
    "zotero_add_by_url",
    "zotero_add_by_isbn",
    "zotero_update_item",
    "zotero_delete_item",
    "zotero_merge_duplicates",
    "zotero_add_from_file",
    "zotero_add_item_relation",
    "zotero_remove_item_relation",
    "zotero_add_by_bibtex",
    "zotero_add_by_csl_json",
}
ADMIN_TOOLS = {
    "zotero_get_capabilities",
    "zotero_get_search_database_status",
    "zotero_get_search_database_health",
    "zotero_update_search_database",
}


def requested_tool_profile() -> ToolProfile:
    value = os.getenv("ZOTERO_MCP_TOOL_PROFILE", "auto").strip().lower()
    if value not in {"auto", "research", "full", "admin", "connector", "all"}:
        return "auto"
    return value  # type: ignore[return-value]


def effective_tool_profile() -> str:
    requested = requested_tool_profile()
    if requested != "auto":
        return requested
    return "full" if os.getenv("ZOTERO_API_KEY") else "research"


def tool_visible(name: str, profile: str | None = None) -> bool:
    profile = profile or effective_tool_profile()
    if name == "zotero_get_capabilities":
        return True
    if profile == "all":
        return True
    if profile == "connector":
        return name in CONNECTOR_TOOLS
    if profile == "admin":
        visible = name in ADMIN_TOOLS
    else:
        visible = True
    if name in CONNECTOR_TOOLS:
        return False
    if name in SEMANTIC_TOOLS and importlib.util.find_spec("chromadb") is None:
        return False
    if name in PDF_TOOLS and importlib.util.find_spec("fitz") is None:
        return False
    if name in FEED_TOOLS and os.getenv("ZOTERO_LOCAL", "").lower() not in {
        "1",
        "true",
        "yes",
    }:
        return False
    if name in LOCAL_PATH_TOOLS and os.getenv(
        "ZOTERO_MCP_EXPOSE_LOCAL_PATHS", ""
    ).lower() not in {"1", "true", "yes"}:
        return False
    if name in LOCAL_PATH_TOOLS and os.getenv("ZOTERO_LOCAL", "").lower() not in {
        "1",
        "true",
        "yes",
    }:
        return False
    if name in WRITE_TOOLS and (
        profile == "research" or not os.getenv("ZOTERO_API_KEY")
    ):
        return False
    return visible


class ToolProfileMiddleware(Middleware):
    """Filter and enforce the selected tool profile for every MCP session."""

    async def on_list_tools(self, context, call_next):
        tools = await call_next(context)
        profile = effective_tool_profile()
        return [tool for tool in tools if tool_visible(tool.name, profile)]

    async def on_list_prompts(self, context, call_next):
        prompts = await call_next(context)
        return [] if effective_tool_profile() in {"connector", "admin"} else prompts

    async def on_list_resources(self, context, call_next):
        resources = await call_next(context)
        return [] if effective_tool_profile() in {"connector", "admin"} else resources

    async def on_list_resource_templates(self, context, call_next):
        templates = await call_next(context)
        return [] if effective_tool_profile() in {"connector", "admin"} else templates

    async def on_get_prompt(self, context, call_next):
        if effective_tool_profile() in {"connector", "admin"}:
            raise ToolError("Prompts are unavailable in this tool profile.")
        return await call_next(context)

    async def on_read_resource(self, context, call_next):
        if effective_tool_profile() in {"connector", "admin"}:
            raise ToolError("Resources are unavailable in this tool profile.")
        return await call_next(context)

    async def on_call_tool(
        self,
        context: MiddlewareContext,
        call_next,
    ):
        name = context.message.name
        profile = effective_tool_profile()
        if not tool_visible(name, profile):
            raise ToolError(
                f"Tool {name!r} is unavailable in the {profile!r} profile. "
                "Call zotero_get_capabilities to inspect the active contract."
            )
        return await call_next(context)
