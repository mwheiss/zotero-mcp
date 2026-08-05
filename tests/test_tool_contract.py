"""Tests for the MCP-visible Zotero tool contract."""

import asyncio
from types import SimpleNamespace

from fastmcp.tools.tool import ToolResult

from zotero_mcp.server import mcp
from zotero_mcp.tool_contract import (
    RESULT_SCHEMA,
    ToolContractMiddleware,
    classify_result,
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


def test_result_classification_distinguishes_empty_blocked_and_partial():
    assert classify_result("No matching items found.")["status"] == "empty"
    assert classify_result("Update not started: requires explicit confirmation.")[
        "status"
    ] == "blocked"
    assert classify_result("Created metadata.\nPartial failure: PDF unavailable.")[
        "status"
    ] == "partial"


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
        "warnings": [],
        "errors": [],
    }
    assert result.meta["fastmcp"]["wrap_result"] is False
