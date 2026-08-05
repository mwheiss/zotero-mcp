"""Consistent MCP schemas and structured results for Zotero tools."""

from __future__ import annotations

import copy
import re
from typing import Any

from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.tools.tool import ToolResult

from zotero_mcp.tool_profiles import CONNECTOR_TOOLS

RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean", "description": "Whether the operation completed as requested."},
        "status": {
            "type": "string",
            "enum": ["success", "empty", "blocked", "partial", "error"],
            "description": "Machine-readable outcome class.",
        },
        "text": {"type": "string", "description": "The complete human-readable markdown result."},
        "warnings": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Warning lines extracted from the result.",
        },
        "errors": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Error lines extracted from the result.",
        },
    },
    "required": ["ok", "status", "text", "warnings", "errors"],
    "additionalProperties": False,
}

PARAMETER_DESCRIPTIONS = {
    "item_key": "Exact 8-character Zotero item key from a prior tool result.",
    "item_keys": "List of exact Zotero item keys.",
    "attachment_key": "Exact child attachment key; when omitted the tool selects one deterministically.",
    "collection_key": "Exact Zotero collection key.",
    "collections": "Collection keys, names, or slash-separated paths.",
    "query": "Search text or natural-language semantic query.",
    "limit": "Maximum number of results or items to process.",
    "tag": "Tag filter or list of tag filters.",
    "tags": "Tags to apply to created or reused items.",
    "fallback_mode": "Search widening policy: none, relaxed metadata/full-text, or semantic.",
    "filters": "Exact-match semantic metadata filters as an object or JSON object string.",
    "if_exists": "Existing-item policy: reuse, merge, duplicate, or legacy aliases skip/file.",
    "attach_mode": "Attachment policy; see the tool description for supported modes.",
    "force_rebuild": "Request a complete replacement semantic index; requires explicit confirmation.",
    "force_clear": "Clear the live index before a confirmed force rebuild instead of staged replacement.",
    "confirmation": "Exact user-provided confirmation required by a destructive safeguard.",
}


def _result_text(result: ToolResult) -> str:
    parts = []
    for block in result.content:
        value = getattr(block, "text", None)
        if isinstance(value, str):
            parts.append(value)
    return "\n".join(parts)


def classify_result(text: str, is_error: bool = False) -> dict[str, Any]:
    stripped = text.strip()
    lowered = stripped.lower()
    errors = [
        line.strip()
        for line in stripped.splitlines()
        if re.match(r"^(?:error|failed)\b", line.strip(), re.I)
        or re.search(r"\bpartial failure\b", line, re.I)
    ]
    warnings = [
        line.strip()
        for line in stripped.splitlines()
        if re.match(r"^(?:warning|warn:)\b", line.strip(), re.I)
    ]
    has_partial_failure = "partial failure" in lowered
    if is_error or lowered.startswith(("error", "failed")):
        status = "error"
    elif has_partial_failure:
        status = "partial"
    elif "not started" in lowered or "requires explicit" in lowered:
        status = "blocked"
    elif lowered.startswith("no ") or "\n\nno " in lowered:
        status = "empty"
    else:
        status = "success"
    return {
        "ok": status in {"success", "empty"},
        "status": status,
        "text": text,
        "warnings": warnings,
        "errors": errors,
    }


class ToolContractMiddleware(Middleware):
    """Advertise and return one stable result envelope for Zotero tools."""

    async def on_list_tools(self, context, call_next):
        tools = await call_next(context)
        normalized = []
        for tool in tools:
            if tool.name in CONNECTOR_TOOLS:
                normalized.append(tool)
                continue
            parameters = copy.deepcopy(tool.parameters)
            for name, schema in parameters.get("properties", {}).items():
                if isinstance(schema, dict) and not schema.get("description"):
                    schema["description"] = PARAMETER_DESCRIPTIONS.get(
                        name,
                        f"Value for {name.replace('_', ' ')}.",
                    )
            normalized.append(
                tool.model_copy(
                    update={
                        "parameters": parameters,
                        "output_schema": RESULT_SCHEMA,
                    }
                )
            )
        return normalized

    async def on_call_tool(
        self,
        context: MiddlewareContext,
        call_next,
    ):
        result = await call_next(context)
        if context.message.name in CONNECTOR_TOOLS:
            return result
        text = _result_text(result)
        return ToolResult(
            content=result.content,
            structured_content=classify_result(text, result.is_error),
            meta=result.meta,
            is_error=result.is_error,
        )
