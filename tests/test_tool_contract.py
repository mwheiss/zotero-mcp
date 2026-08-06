"""Tests for the MCP-visible Zotero tool contract."""

import asyncio
from types import SimpleNamespace

from fastmcp.tools.tool import ToolResult

from zotero_mcp.server import mcp
from zotero_mcp.tool_contract import (
    CONTENT_BEARING_TOOLS,
    RESULT_SCHEMA,
    ToolContractMiddleware,
    _legacy_result_data,
    classify_result,
)
from zotero_mcp.tool_profiles import (
    ADMIN_TOOLS,
    CONNECTOR_TOOLS,
    FEED_TOOLS,
    LOCAL_PATH_TOOLS,
    WRITE_TOOLS,
)


def _normalized_tools():
    async def run():
        middleware = ToolContractMiddleware()

        async def registered(_context):
            return await mcp.list_tools()

        return await middleware.on_list_tools(None, registered)

    return asyncio.run(run())


def test_zotero_tools_advertise_described_parameters_and_result_schema():
    tools = _normalized_tools()
    zotero_tools = [tool for tool in tools if tool.name.startswith("zotero_")]

    assert zotero_tools
    for tool in zotero_tools:
        assert tool.output_schema == RESULT_SCHEMA
        for schema in tool.parameters.get("properties", {}).values():
            assert schema.get("description")


def test_connector_tools_keep_the_connector_output_contract(monkeypatch):
    monkeypatch.setenv("ZOTERO_MCP_TOOL_PROFILE", "all")
    tools = {tool.name: tool for tool in _normalized_tools()}

    assert tools["search"].output_schema != RESULT_SCHEMA
    assert tools["fetch"].output_schema != RESULT_SCHEMA


def test_every_non_content_tool_has_an_explicit_operational_role(monkeypatch):
    monkeypatch.setenv("ZOTERO_MCP_TOOL_PROFILE", "all")
    monkeypatch.setenv("ZOTERO_API_KEY", "test-key")
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "1")
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setenv("ZOTERO_MCP_EXPOSE_LOCAL_PATHS", "true")
    tool_names = {tool.name for tool in _normalized_tools()}
    permitted_non_content = (
        CONNECTOR_TOOLS
        | WRITE_TOOLS
        | ADMIN_TOOLS
        | LOCAL_PATH_TOOLS
        | {"zotero_switch_library"}
    )

    assert tool_names - CONTENT_BEARING_TOOLS <= permitted_non_content


def test_result_classification_distinguishes_empty_blocked_and_partial():
    assert classify_result("No matching items found.")["status"] == "empty"
    assert classify_result("Update not started: requires explicit confirmation.")[
        "status"
    ] == "blocked"
    assert classify_result("Created metadata.\nPartial failure: PDF unavailable.")[
        "status"
    ] == "partial"


def test_result_classification_recognizes_formatted_failures_and_warnings():
    failures = [
        "# Database\n\n**Error:** embedding failed",
        "Input error: invalid item key",
        "Collections: FAILED to update",
        "[ERROR] content hash mismatch",
        "Semantic search error: encoder unavailable",
        "Collection not found: 'ABCD1234'.",
        "Invalid library_type 'wat'. Must be 'user', 'group', or 'feed'.",
        "Could not reach Scite API - try again later.",
        "Unable to read the requested attachment.",
        "Unsupported operation 'containsAny'.",
        "Missing item key.",
        "Group '999' not found. Available groups: 123.",
        "Collection 'missing' not found.",
        "arXiv API error for 1234.5678: timeout",
        "arXiv is currently unreachable and its fallback failed.",
        "Semantic chunk `ABCD1234#0` changed after the search result was produced.",
    ]

    for failure in failures:
        outcome = classify_result(failure)
        assert outcome["ok"] is False
        assert outcome["status"] == "error"
        assert outcome["errors"]

    warning = classify_result("[WARN] fulltext unavailable")
    assert warning["status"] == "success"
    assert warning["warnings"] == ["[WARN] fulltext unavailable"]


def test_result_classification_recognizes_blocked_capabilities():
    blocked = [
        "PyMuPDF is required for PDF page reading. Install the PDF extra.",
        "Error: zotero_get_attachment_path requires local mode (set ZOTERO_LOCAL=true).",
        "Error: Web API credentials required for creating annotations.",
        "Update not started: requires explicit confirmation.",
        "# Force Rebuild Not Started\n\nThe user has not confirmed it.",
        "Semantic search is not available. Install the semantic extra.",
        "RSS feeds are only accessible in local mode (ZOTERO_LOCAL=true).",
    ]

    for message in blocked:
        outcome = classify_result(message)
        assert outcome["ok"] is False
        assert outcome["status"] == "blocked"


def test_result_classification_recognizes_empty_resolver_results():
    empty = [
        "No matching items found.",
        "None of the 10 items have DOIs - cannot check Scite.",
        "DOI not found on CrossRef: 10.1000/missing",
        "ISBN not found on Open Library or Google Books: 1234567890",
        "Relation not found: 'ABCD1234' is not related to 'WXYZ5678'.",
        "OpenAlex has no record for DOI '10.1000/missing', or the lookup failed.",
    ]

    for message in empty:
        outcome = classify_result(message)
        assert outcome["ok"] is True
        assert outcome["status"] == "empty"


def test_document_content_cannot_change_tool_outcome():
    successful_documents = [
        "# Full Text\n\nNo significant difference was observed between cohorts.",
        "# Full Text\n\nMissing values were imputed using the median.",
        "# Failure Analysis of Detector Readout\n\nUseful scientific content.",
        "# Full Text\n\nCalibration is required for accurate spectroscopy.",
        "# Search Result\n\nPartial failure: a term used in the quoted paper.",
        "No significant difference was observed between cohorts.",
    ]

    for document in successful_documents:
        outcome = classify_result(document, tool_name="zotero_get_item_fulltext")
        assert outcome["ok"] is True
        assert outcome["status"] == "success"
        assert outcome["errors"] == []

    for tool_name in CONTENT_BEARING_TOOLS:
        outcome = classify_result(
            "# Retrieved content\n\nError: this sentence is quoted source material.",
            tool_name=tool_name,
        )
        assert outcome["status"] == "success", tool_name

    assert FEED_TOOLS <= CONTENT_BEARING_TOOLS


def test_write_titles_cannot_change_tool_outcome():
    successful_writes = [
        "Successfully added: **Failure Analysis of Detector Readout**",
        "Successfully added: **Partial failure: a statistical perspective**",
        "Successfully updated: **Missing values in longitudinal studies**",
    ]

    for result in successful_writes:
        outcome = classify_result(result, tool_name="zotero_add_by_doi")
        assert outcome["status"] == "success"
        assert outcome["errors"] == []

    partial = classify_result(
        "Successfully added: **Ordinary title**\n"
        "PDF: unavailable; Partial failure: required PDF attachment was not satisfied",
        tool_name="zotero_add_by_doi",
    )
    assert partial["status"] == "partial"


def test_explicit_control_lines_after_a_heading_are_still_recognized():
    error = classify_result("# Database\n\n**Error:** embedding failed")
    assert error["status"] == "error"

    partial = classify_result(
        "Successfully created metadata.\nPartial failure: PDF unavailable."
    )
    assert partial["status"] == "partial"

    empty = classify_result("# Item metadata\n\nNo suitable attachment found for this item.")
    assert empty["status"] == "empty"

    download = classify_result(
        "# Paper title\n\nFile download failed.\n\nAttempted sources: none",
        tool_name="zotero_get_item_fulltext",
    )
    assert download["status"] == "error"


def test_call_middleware_preserves_text_and_adds_structured_result():
    async def run():
        middleware = ToolContractMiddleware()
        context = SimpleNamespace(message=SimpleNamespace(name="zotero_example"))

        async def call_next(_context):
            return ToolResult(
                content="No matching items found.",
                meta={"fastmcp": {"wrap_result": True}},
            )

        return await middleware.on_call_tool(context, call_next)

    result = asyncio.run(run())

    assert result.content[0].text == "No matching items found."
    assert result.structured_content == {
        "ok": True,
        "status": "empty",
        "text": "No matching items found.",
        "data": {},
        "warnings": [],
        "errors": [],
    }
    assert result.meta["fastmcp"]["wrap_result"] is False


def test_call_middleware_preserves_native_data_and_sets_error_bit():
    async def run():
        middleware = ToolContractMiddleware()
        context = SimpleNamespace(message=SimpleNamespace(name="zotero_example"))

        async def call_next(_context):
            return ToolResult(
                content="**Error:** write rejected",
                structured_content={"item_key": "ABCD1234"},
            )

        return await middleware.on_call_tool(context, call_next)

    result = asyncio.run(run())

    assert result.is_error is True
    assert result.structured_content["status"] == "error"
    assert result.structured_content["data"] == {"item_key": "ABCD1234"}


def test_call_middleware_derives_fields_and_identifiers_from_legacy_markdown():
    async def run():
        middleware = ToolContractMiddleware()
        context = SimpleNamespace(message=SimpleNamespace(name="zotero_semantic_search"))

        async def call_next(_context):
            return ToolResult(
                content=(
                    "### Result one\n"
                    "**Item Key:** `ABCD1234`\n"
                    "**Chunk ID:** `ABCD1234:0007`\n"
                    "**Chunk Hash:** `abc123`\n"
                    "**DOI:** `10.1000/example`\n\n"
                    "### Result two\n"
                    "**Item Key:** `WXYZ5678`\n"
                    "**Chunk ID:** `WXYZ5678:0002`"
                ),
                meta={"fastmcp": {"wrap_result": True}},
            )

        return await middleware.on_call_tool(context, call_next)

    data = asyncio.run(run()).structured_content["data"]

    assert data["fields"]["item_key"] == ["ABCD1234", "WXYZ5678"]
    assert data["identifiers"] == {
        "item_keys": ["ABCD1234", "WXYZ5678"],
        "chunk_ids": ["ABCD1234:0007", "WXYZ5678:0002"],
        "chunk_hashes": ["abc123"],
        "dois": ["10.1000/example"],
    }


def test_call_middleware_uses_json_text_as_native_data():
    async def run():
        middleware = ToolContractMiddleware()
        context = SimpleNamespace(message=SimpleNamespace(name="zotero_example"))

        async def call_next(_context):
            return ToolResult(content='{"items":[{"key":"ABCD1234"}]}')

        return await middleware.on_call_tool(context, call_next)

    data = asyncio.run(run()).structured_content["data"]
    assert data == {"items": [{"key": "ABCD1234"}]}


def test_content_bearing_json_remains_document_text_not_structured_data():
    data = _legacy_result_data(
        '{"error":"quoted source material","body":"paper text"}',
        content_bearing=True,
    )

    assert data == {}


def test_legacy_data_captures_plain_write_identifiers_as_arrays():
    data = _legacy_result_data(
        "Successfully created an item.\n"
        "Item key: `ABCD1234`\n"
        "Note key: WXYZ5678\n"
        "DOI: 10.1000/example"
    )

    assert data["identifiers"] == {
        "item_keys": ["ABCD1234"],
        "note_keys": ["WXYZ5678"],
        "dois": ["10.1000/example"],
    }


def test_content_data_ignores_arbitrary_paper_fields():
    data = _legacy_result_data(
        "# Paper\n"
        "**Item Key:** `ABCD1234`\n"
        "**Methods:** Missing values were imputed.\n"
        "**Result:** No effect.",
        content_bearing=True,
    )

    assert data == {
        "fields": {"item_key": ["ABCD1234"]},
        "identifiers": {"item_keys": ["ABCD1234"]},
    }
