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
        "data": {
            "description": (
                "Native machine-readable result data when the tool provides it; "
                "null for legacy text-only tools."
            ),
        },
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
    "required": ["ok", "status", "text", "data", "warnings", "errors"],
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


def _marker_text(line: str) -> str:
    """Strip common Markdown decoration before inspecting outcome markers."""
    value = re.sub(r"^[\s>#*+-]+", "", line).strip()
    value = value.replace("**", "").replace("__", "").strip()
    return value


def _is_error_marker(line: str) -> bool:
    marker = _marker_text(line)
    return bool(
        re.match(
            r"^(?:\[(?:error|fail(?:ed|ure)?)\]|"
            r"(?:(?:input|semantic search)\s+)?error\b|"
            r"fail(?:ed|ure)?\b|"
            r"collection not found\b|"
            r"invalid\b|"
            r"could not\b|"
            r"unable to\b|"
            r"unsupported\b|"
            r"missing\b)",
            marker,
            re.I,
        )
        or re.search(r":\s*fail(?:ed|ure)?\b", marker, re.I)
    )


def _is_warning_marker(line: str) -> bool:
    return bool(
        re.match(
            r"^(?:\[warn(?:ing)?\]|warnings?\b|warn:)",
            _marker_text(line),
            re.I,
        )
    )


def _is_blocked_marker(line: str) -> bool:
    """Recognize requests that cannot run until a capability is available."""
    marker = _marker_text(line)
    return bool(
        re.match(
            r"^(?:update not started\b|.*\brequires?\s+(?:local mode|web api credentials|explicit confirmation)\b|"
            r".*\bcredentials required\b|.*\bis required for\b|.*\bis required\.?(?:\s|$))",
            marker,
            re.I,
        )
    )


def classify_result(
    text: str,
    is_error: bool = False,
    data: Any = None,
) -> dict[str, Any]:
    stripped = text.strip()
    lowered = stripped.lower()
    blocked = [line.strip() for line in stripped.splitlines() if _is_blocked_marker(line)]
    errors = [
        line.strip()
        for line in stripped.splitlines()
        if not _is_blocked_marker(line)
        and (
            _is_error_marker(line)
            or re.search(r"\bpartial failure\b", line, re.I)
        )
    ]
    warnings = [
        line.strip()
        for line in stripped.splitlines()
        if _is_warning_marker(line)
    ]
    has_partial_failure = "partial failure" in lowered
    has_positive_result = bool(
        re.search(
            r"\b(?:successfully|created|updated|deleted|reused|restored|attached)\b",
            lowered,
        )
    )
    if has_partial_failure or (errors and has_positive_result):
        status = "partial"
    elif is_error or errors:
        status = "error"
    elif blocked:
        status = "blocked"
    elif lowered.startswith("no ") or "\n\nno " in lowered:
        status = "empty"
    else:
        status = "success"
    return {
        "ok": status in {"success", "empty"},
        "status": status,
        "text": text,
        "data": data,
        "warnings": warnings,
        "errors": errors,
    }


def _result_data(result: ToolResult, text: str) -> Any:
    """Preserve native structured data while dropping FastMCP's string wrapper."""
    structured = result.structured_content
    if isinstance(structured, dict) and structured == {"result": text}:
        return None
    return copy.deepcopy(structured)


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
        meta = copy.deepcopy(result.meta) if result.meta else {}
        # The underlying string-returning function is marked as a wrapped
        # FastMCP result. We replace that payload with our envelope, so clients
        # must validate the envelope itself rather than look for a nonexistent
        # ``structuredContent.result`` value.
        fastmcp_meta = meta.setdefault("fastmcp", {})
        if isinstance(fastmcp_meta, dict):
            fastmcp_meta["wrap_result"] = False
        structured = classify_result(
            text,
            result.is_error,
            data=_result_data(result, text),
        )
        return ToolResult(
            content=result.content,
            structured_content=structured,
            meta=meta,
            is_error=result.is_error or structured["status"] == "error",
        )
