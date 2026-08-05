"""Consistent MCP schemas and structured results for Zotero tools."""

from __future__ import annotations

import copy
import json
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
                "Machine-readable result data. Native structures are preserved; "
                "legacy markdown tools expose parsed fields and identifiers."
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
            r"missing\b|"
            r"(?:collection|feed|group)\b.*\bnot found\b|"
            r"arxiv api error\b|"
            r".*\bcurrently unreachable\b|"
            r"semantic chunk\b.*\bchanged after\b)",
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
            r"^(?:.*\bnot started\b|semantic search\b.*\bnot available\b|"
            r"rss feed items?\b.*\bonly accessible\b|rss feeds?\b.*\bonly accessible\b|"
            r".*\brequires?\s+(?:local mode|web api credentials|explicit confirmation)\b|"
            r".*\bcredentials required\b|.*\bis required for\b|.*\bis required\.?(?:\s|$))",
            marker,
            re.I,
        )
    )


def _is_empty_marker(line: str) -> bool:
    marker = _marker_text(line)
    return bool(
        re.match(
            r"^(?:no\b|none of\b|doi\b.*\bnot found\b|isbn\b.*\bnot found\b|"
            r"relation\b.*\bnot found\b)",
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
    empty = [line.strip() for line in stripped.splitlines() if _is_empty_marker(line)]
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
    elif empty:
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
    """Preserve native data or derive useful structure from legacy markdown."""
    structured = result.structured_content
    if not (isinstance(structured, dict) and structured == {"result": text}):
        if structured is not None:
            return copy.deepcopy(structured)
    return _legacy_result_data(text)


def _field_name(label: str) -> str:
    """Normalize a human-facing markdown label into a stable data key."""
    value = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
    return value or "field"


def _field_value(value: str) -> str:
    """Remove presentation-only markdown around a field value."""
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] == "`":
        cleaned = cleaned[1:-1]
    return cleaned.strip()


def _append_value(target: dict[str, Any], key: str, value: str) -> None:
    existing = target.get(key)
    if existing is None:
        target[key] = value
    elif isinstance(existing, list):
        if value not in existing:
            existing.append(value)
    elif existing != value:
        target[key] = [existing, value]


def _legacy_result_data(text: str) -> Any:
    """Derive a conservative machine view without changing legacy tool text."""
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    fields: dict[str, Any] = {}
    identifiers: dict[str, Any] = {}
    field_pattern = re.compile(
        r"^\s*(?:[-*]\s+)?\*\*([^*:\n]+):\*\*\s*(.*?)\s*$"
    )
    identifier_fields = {
        "item_key": "item_keys",
        "attachment_key": "attachment_keys",
        "collection_key": "collection_keys",
        "chunk_id": "chunk_ids",
        "chunk_hash": "chunk_hashes",
        "doi": "dois",
        "citation_key": "citation_keys",
    }
    prose_fields = {
        "abstract",
        "content",
        "full_text",
        "matched_passage",
        "note",
        "supporting_passages",
        "text",
    }

    for line in stripped.splitlines():
        match = field_pattern.match(line)
        if not match:
            continue
        key = _field_name(match.group(1))
        value = _field_value(match.group(2))
        if not value or key in prose_fields:
            continue
        _append_value(fields, key, value)
        identifier_key = identifier_fields.get(key)
        if identifier_key:
            _append_value(identifiers, identifier_key, value)

    # Capture identifiers embedded in links or compact prose even when a tool
    # does not render them as labelled fields.
    for key in re.findall(r"zotero://select/(?:library|groups/\d+)/items/([A-Z0-9]{8})", stripped):
        _append_value(identifiers, "item_keys", key)
    for doi in re.findall(r"https?://doi\.org/([^\s)>]+)", stripped, re.I):
        _append_value(identifiers, "dois", doi.rstrip(".,;"))

    data: dict[str, Any] = {}
    if fields:
        data["fields"] = fields
    if identifiers:
        data["identifiers"] = identifiers
    return data


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
