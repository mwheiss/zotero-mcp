"""Consistent MCP schemas and structured results for Zotero tools."""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.tools.tool import ToolResult
from mcp.types import ToolAnnotations

from zotero_mcp.tool_profiles import CONNECTOR_TOOLS, FEED_TOOLS, WRITE_TOOLS

CONTENT_BEARING_TOOLS = {
    *FEED_TOOLS,
    "scite_check_retractions",
    "scite_enrich_item",
    "scite_enrich_search",
    "zotero_advanced_search",
    "zotero_export_bibliography",
    "zotero_find_duplicates",
    "zotero_find_related_papers",
    "zotero_get_annotations",
    "zotero_get_attachment",
    "zotero_get_collections",
    "zotero_get_collection_items",
    "zotero_get_item_children",
    "zotero_get_item_fulltext",
    "zotero_get_document_text",
    "zotero_get_item_metadata",
    "zotero_get_item_related",
    "zotero_get_items_children",
    "zotero_get_notes",
    "zotero_get_page_layout",
    "zotero_get_pdf_outline",
    "zotero_get_recent",
    "zotero_get_semantic_context",
    "zotero_get_tags",
    "zotero_library_coverage",
    "zotero_list_libraries",
    "zotero_list_attachments",
    "zotero_read_pdf_pages",
    "zotero_search_by_citation_key",
    "zotero_search_by_tag",
    "zotero_search_collections",
    "zotero_search_items",
    "zotero_search_notes",
    "zotero_semantic_search",
    "zotero_synthesize_annotations",
}

STRUCTURED_JSON_TEXT_TOOLS = {
    "zotero_get_annotations",
    "zotero_get_attachment",
    "zotero_list_attachments",
}

DESTRUCTIVE_TOOLS = {
    "zotero_batch_update_extra",
    "zotero_batch_update_tags",
    "zotero_delete_annotation",
    "zotero_delete_collection",
    "zotero_delete_item",
    "zotero_delete_note",
    "zotero_manage_collections",
    "zotero_merge_duplicates",
    "zotero_put_attachment",
    "zotero_remove_item_relation",
    "zotero_set_attachment_trashed",
    "zotero_update_annotation",
    "zotero_update_attachment",
    "zotero_update_item",
    "zotero_update_note",
    "zotero_update_search_database",
}

NON_IDEMPOTENT_TOOLS = {
    "zotero_add_by_bibtex",
    "zotero_add_by_csl_json",
    "zotero_add_by_doi",
    "zotero_add_by_isbn",
    "zotero_add_by_url",
    "zotero_add_from_file",
    "zotero_create_annotation",
    "zotero_create_area_annotation",
    "zotero_create_collection",
    "zotero_create_note",
    "zotero_merge_duplicates",
    "zotero_prepare_attachment_change",
    "zotero_prepare_attachment_upload",
}

OPEN_WORLD_TOOLS = {
    "scite_check_retractions",
    "scite_enrich_item",
    "scite_enrich_search",
    "zotero_add_by_bibtex",
    "zotero_add_by_csl_json",
    "zotero_add_by_doi",
    "zotero_add_by_isbn",
    "zotero_add_by_url",
    "zotero_add_from_file",
    "zotero_find_related_papers",
    "zotero_search_items",
    "zotero_semantic_search",
    "zotero_update_search_database",
}


def _tool_annotations(name: str) -> ToolAnnotations:
    """Return complete, conservative MCP behavior hints for one tool."""
    read_only = name not in WRITE_TOOLS and name != "zotero_switch_library"
    return ToolAnnotations(
        readOnlyHint=read_only,
        destructiveHint=name in DESTRUCTIVE_TOOLS,
        idempotentHint=name not in NON_IDEMPOTENT_TOOLS,
        openWorldHint=name in OPEN_WORLD_TOOLS,
    )

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
                "legacy markdown tools expose parsed fields and identifiers, "
                "with every parsed category represented as an array."
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
    "abstract": "Replacement abstract text for the Zotero item.",
    "access_date": "Replacement access date in a Zotero-supported date format.",
    "add_tags": "Tags to add while preserving tags already present.",
    "add_to": "Collection keys, names, or slash-separated paths to add the items to.",
    "adjacent_chunks": "Number of neighboring indexed chunks to include on each side (0-2).",
    "annotation_key": "Exact 8-character Zotero annotation key.",
    "item_key": "Exact 8-character Zotero item key from a prior tool result.",
    "item_keys": "List of exact Zotero item keys.",
    "attachment_key": "Exact child attachment key; when omitted the tool selects one deterministically.",
    "action": "Prepared attachment mutation: replace, trash, or reparent.",
    "bibtex": "Inline BibTeX containing one or more entries; mutually exclusive with file_path.",
    "book_title": "Replacement containing-book title for a book section or chapter.",
    "chunk_id": "Exact Chunk ID returned by zotero_semantic_search; do not infer or construct it.",
    "collection": "Collection name or exact 8-character key to scan.",
    "collection_key": "Exact Zotero collection key.",
    "collection_names": "Replacement collection names, keys, or slash-separated paths.",
    "collections": "Collection keys, names, or slash-separated paths.",
    "color": "Replacement annotation color as a hexadecimal value such as #ffd400.",
    "comment": "Replacement annotation comment; an empty string clears it.",
    "compare_zotero": "Compare indexed coverage with the current local Zotero library snapshot.",
    "confirm": "Set true only after the user has explicitly approved the destructive operation.",
    "confirmation_token": "Short-lived operation-bound token returned by the attachment change preview.",
    "content_type": "IANA MIME type for the attachment binary.",
    "create_missing_collections": "Create unresolved collection paths instead of rejecting the operation.",
    "creators": "Replacement Zotero creator list, as structured creator objects or supported text input.",
    "csl_json": "Inline CSL JSON object or array; mutually exclusive with file_path.",
    "date": "Replacement publication date in a Zotero-supported date format.",
    "detail": "Response detail level: keys_only, summary, or full metadata including abstracts.",
    "direction": "Citation direction to traverse: references, citations, or both.",
    "doi": "DOI with or without a DOI URL prefix.",
    "duplicate_keys": "Exact Zotero item keys to merge into the keeper item and move to Trash.",
    "edition": "Replacement edition statement for a book or comparable item.",
    "expected_hash": "Optional Chunk Hash from the same search result; rejects stale indexed context.",
    "extra": "Replacement contents of Zotero's Extra field.",
    "query": "Search text or natural-language semantic query.",
    "file_path": "Absolute path to the local source file accepted by this tool.",
    "filename": "Safe filename for the attachment binary.",
    "identifier": "Seed paper as an 8-character Zotero item key, DOI, or DOI URL.",
    "include_trashed": "Include collections currently in Zotero Trash.",
    "include_subcollections": "Include items filed in descendant collections as well as the named collection.",
    "inline": "Embed base64 only within the configured inline limit (1 MiB by default).",
    "idempotency_key": "Caller-generated stable key that makes attachment creation safe to retry.",
    "isbn": "ISBN-10 or ISBN-13, with optional hyphens or URL/isbn prefix.",
    "issn": "Replacement ISSN for the Zotero item.",
    "issue": "Replacement publication issue identifier.",
    "item_type": "Zotero item type to assign, such as document or journalArticle.",
    "keeper_key": "Exact key of the Zotero item that will retain the merged metadata and children.",
    "language": "Replacement language value for the Zotero item.",
    "limit": "Maximum number of results or items to process.",
    "offset": "Zero-based result offset used to continue a paginated listing.",
    "method": "Duplicate matching method: title, DOI, or both.",
    "name": "Name for the new Zotero collection.",
    "pages": "Replacement page range or article number.",
    "parent_collection": "Parent collection as an exact key or unambiguous name; omit for top level.",
    "parent_item_key": "Exact parent bibliographic item key for the attachment.",
    "plan_exact_doi": "Return a read-only exact-DOI batch plan with deterministic keeper recommendations.",
    "plan_id": "Short-lived duplicate-merge plan identifier returned by the dry-run preview.",
    "plan_token": "One-use version-bound duplicate-merge token returned by the dry-run preview.",
    "publication_title": "Replacement journal, magazine, or publication title.",
    "publisher": "Replacement publisher name.",
    "quick": "Skip the slower full SQLite integrity check while retaining other health checks.",
    "remove_from": "Collection keys, names, or slash-separated paths to remove the items from.",
    "remove_tags": "Tags to remove while preserving all other tags.",
    "operation_id": "Short-lived attachment mutation ID returned by the change preview.",
    "sha256": "Expected lowercase hexadecimal SHA-256 digest of the complete binary.",
    "size": "Exact binary size in bytes.",
    "short_title": "Replacement short title for the Zotero item.",
    "search_all_libraries": "Search the personal library and all locally available group libraries.",
    "tag": "Tag filter or list of tag filters.",
    "tags": "Tags to apply to created or reused items.",
    "text": "Replacement highlighted annotation text; an empty string clears it.",
    "title": "Title to assign to the Zotero item.",
    "trashed": "True moves the attachment to Trash; false restores it.",
    "upload_id": "Staged upload ID returned by zotero_prepare_attachment_upload.",
    "url": "Source URL or replacement Zotero URL field, as described by the tool.",
    "volume": "Replacement publication volume identifier.",
    "fallback_mode": "Search widening policy: none, relaxed metadata/full-text, or semantic.",
    "filters": "Exact-match semantic metadata filters as an object or JSON object string.",
    "if_exists": "Existing-item policy: reuse, merge, duplicate, or legacy aliases skip/file.",
    "attach_mode": "Attachment policy; see the tool description for supported modes.",
    "force_rebuild": "Re-embed the complete semantic index with resumable item replacement; requires explicit confirmation.",
    "force_clear": "Clear the live index before a confirmed force rebuild.",
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
            r"file download failed\b|"
            r".*\bcurrently unreachable\b|"
            r"semantic chunk\b.*\bchanged after\b)",
            marker,
            re.I,
        )
        or re.match(
            r"^(?:attachments?|collections?|items?|relations?|tags?):\s*"
            r"fail(?:ed|ure)?\b",
            marker,
            re.I,
        )
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
            r"^(?:no\s+(?:matching\s+)?(?:annotations?|attachments?|"
            r"bibliography entries|changes|child items|collections?|duplicates?|"
            r"feed|full[- ]?text|items?|notes?|pdf attachment|pdf outline|related items|"
            r"results?|rss feeds?|scite data|semantically similar items|"
            r"suitable attachment|tags?)\b|"
            r"none of\b|doi\b.*\bnot found\b|isbn\b.*\bnot found\b|"
            r"relation\b.*\bnot found\b|openalex has no record\b)",
            marker,
            re.I,
        )
    )


def _is_markdown_heading(line: str) -> bool:
    return bool(re.match(r"^\s{0,3}#{1,6}\s+", line))


def _is_explicit_control_marker(line: str) -> bool:
    """Return whether a non-leading line deliberately reports tool state."""
    marker = _marker_text(line)
    return _is_partial_marker(line) or bool(
        re.match(
            r"^(?:\[(?:error|fail(?:ed|ure)?|warn(?:ing)?)\]|"
            r"(?:(?:input|semantic search)\s+)?error\s*:|"
            r"warnings?\s*:|warn\s*:|"
            r"file download failed\.?$|no suitable attachment found\b|"
            r".*:\s*fail(?:ed|ure)?\b)",
            marker,
            re.I,
        )
    )


def _is_partial_marker(line: str) -> bool:
    """Recognize generated partial-result markers without scanning user text."""
    marker = _marker_text(line)
    return bool(
        re.match(r"^partial failure\s*:", marker, re.I)
        or re.match(r"^partial coverage\s*:", marker, re.I)
        or re.match(r"^partially updated\b", marker, re.I)
        or re.match(r"^pdf\s*:.*;\s*partial failure\s*:", marker, re.I)
        or re.match(
            r"^note\s*:\s*search stopped\b.*\bresults are partial\.?$",
            marker,
            re.I,
        )
    )


def _control_lines(text: str, *, content_bearing: bool = False) -> list[str]:
    """Return only lines that may describe the operation rather than content."""
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return []

    control: list[str] = []
    first = lines[0]
    # A heading is normally a paper/result title. The destructive safeguard is
    # the one control outcome intentionally rendered as a heading.
    if not _is_markdown_heading(first) or "not started" in _marker_text(first).lower():
        control.append(first)
    if content_bearing:
        # Later lines may be arbitrary papers, notes, abstracts, or quoted
        # passages. Only attachment fallback messages have a guaranteed
        # control meaning inside those otherwise content-bearing responses.
        safe_followups = re.compile(
            r"^(?:file download failed\.?$|no suitable attachment found\b|"
            r"error accessing attachment\s*:|partial coverage\s*:|"
            r"note\s*:\s*search stopped\b.*\bresults are partial\.?$)",
            re.I,
        )
        control.extend(
            line
            for line in lines[1:]
            if safe_followups.match(_marker_text(line))
        )
    else:
        control.extend(
            line
            for line in lines[1:]
            if _is_explicit_control_marker(line)
        )
    return control


def classify_result(
    text: str,
    is_error: bool = False,
    data: Any = None,
    tool_name: str | None = None,
) -> dict[str, Any]:
    stripped = text.strip()
    control_lines = _control_lines(
        stripped,
        content_bearing=tool_name in CONTENT_BEARING_TOOLS,
    )
    blocked = [line.strip() for line in control_lines if _is_blocked_marker(line)]
    errors = [
        line.strip()
        for line in control_lines
        if not _is_blocked_marker(line)
        and (
            _is_error_marker(line)
            or _is_partial_marker(line)
        )
    ]
    warnings = [
        line.strip()
        for line in control_lines
        if _is_warning_marker(line)
    ]
    if tool_name in STRUCTURED_JSON_TEXT_TOOLS and isinstance(data, dict):
        data_error = data.get("error")
        if data_error:
            errors.append(f"Error: {data_error}")
    empty = [line.strip() for line in control_lines[:1] if _is_empty_marker(line)]
    has_partial_failure = any(
        _is_partial_marker(line)
        for line in control_lines
    )
    leading_text = _marker_text(control_lines[0]).lower() if control_lines else ""
    has_positive_result = bool(
        re.search(
            r"\b(?:successfully|created|updated|deleted|reused|restored|attached)\b",
            leading_text,
        )
    )
    if has_partial_failure or (errors and has_positive_result):
        status = "partial"
    elif is_error or errors:
        status = "error"
    elif blocked:
        status = "blocked"
    elif (
        tool_name == "zotero_list_attachments"
        and isinstance(data, dict)
        and data.get("count") == 0
    ):
        status = "empty"
    elif (
        tool_name == "zotero_get_annotations"
        and isinstance(data, list)
        and not data
        and not errors
    ):
        status = "empty"
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


def _result_data(result: ToolResult, text: str, *, tool_name: str | None = None) -> Any:
    """Preserve native data or derive useful structure from legacy markdown."""
    structured = result.structured_content
    if not (isinstance(structured, dict) and structured == {"result": text}):
        if structured is not None:
            return copy.deepcopy(structured)
    if tool_name in STRUCTURED_JSON_TEXT_TOOLS:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            if tool_name == "zotero_get_annotations" and "\n\n" in text:
                try:
                    return json.loads(text.rsplit("\n\n", 1)[1])
                except json.JSONDecodeError:
                    pass
    return _legacy_result_data(
        text,
        content_bearing=tool_name in CONTENT_BEARING_TOOLS,
    )


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


def _append_value(target: dict[str, list[str]], key: str, value: str) -> None:
    values = target.setdefault(key, [])
    if value not in values:
        values.append(value)


def _legacy_result_data(text: str, *, content_bearing: bool = False) -> Any:
    """Derive a conservative machine view without changing legacy tool text."""
    stripped = text.strip()
    if not content_bearing and stripped.startswith(("{", "[")):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    fields: dict[str, list[str]] = {}
    identifiers: dict[str, list[str]] = {}
    field_pattern = re.compile(
        r"^\s*(?:[-*]\s+)?\*\*([^*:\n]+):\*\*\s*(.*?)\s*$"
    )
    identifier_fields = {
        "item_key": "item_keys",
        "requested_item_key": "item_keys",
        "note_key": "note_keys",
        "attachment_key": "attachment_keys",
        "collection_key": "collection_keys",
        "chunk_id": "chunk_ids",
        "requested_chunk": "chunk_ids",
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
    safe_content_fields = {
        "attachment_key",
        "authors",
        "chunk_hash",
        "chunk_id",
        "citation_key",
        "collection_key",
        "date",
        "doi",
        "indexed_source",
        "item_key",
        "location",
        "note_key",
        "relevance",
        "requested_chunk",
        "requested_item_key",
        "selected_attachment",
        "selected_source",
        "type",
    }

    for line in stripped.splitlines():
        match = field_pattern.match(line)
        if not match:
            continue
        key = _field_name(match.group(1))
        value = _field_value(match.group(2))
        if (
            not value
            or key in prose_fields
            or (content_bearing and key not in safe_content_fields)
        ):
            continue
        _append_value(fields, key, value)
        identifier_key = identifier_fields.get(key)
        if identifier_key:
            _append_value(identifiers, identifier_key, value)

    if not content_bearing:
        plain_identifier_pattern = re.compile(
            r"^\s*(?:[-*]\s+)?(Item|Note|Attachment|Collection|Citation)\s+key:\s*`?([^`\s]+)`?\s*$",
            re.I,
        )
        plain_doi_pattern = re.compile(r"^\s*(?:[-*]\s+)?DOI:\s*(\S+)\s*$", re.I)
        for line in stripped.splitlines():
            match = plain_identifier_pattern.match(line)
            if match:
                key = _field_name(f"{match.group(1)} key")
                value = _field_value(match.group(2))
                _append_value(fields, key, value)
                _append_value(identifiers, identifier_fields[key], value)
                continue
            doi_match = plain_doi_pattern.match(line)
            if doi_match:
                value = _field_value(doi_match.group(1))
                _append_value(fields, "doi", value)
                _append_value(identifiers, "dois", value)

        # Capture identifiers embedded in links or compact prose even when a
        # non-content tool does not render them as labelled fields.
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
            annotations = _tool_annotations(tool.name)
            if tool.name in CONNECTOR_TOOLS:
                normalized.append(tool.model_copy(update={"annotations": annotations}))
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
                        "annotations": annotations,
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
            data=_result_data(result, text, tool_name=context.message.name),
            tool_name=context.message.name,
        )
        return ToolResult(
            content=result.content,
            structured_content=structured,
            meta=meta,
            is_error=result.is_error or structured["status"] == "error",
        )
