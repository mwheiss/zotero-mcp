import asyncio

import pytest
from conftest import DummyContext
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import MiddlewareContext
from mcp.types import CallToolRequestParams

from zotero_mcp.server import get_capabilities, get_search_database_health, mcp
from zotero_mcp.tool_profiles import (
    ToolProfileMiddleware,
    effective_tool_profile,
    tool_visible,
)


def _listed_tool_names() -> set[str]:
    async def run():
        middleware = ToolProfileMiddleware()

        async def list_registered(_context):
            return await mcp.list_tools()

        tools = await middleware.on_list_tools(None, list_registered)
        return {tool.name for tool in tools}

    return asyncio.run(run())


def test_auto_profile_uses_research_without_api_key(monkeypatch):
    monkeypatch.setenv("ZOTERO_MCP_TOOL_PROFILE", "auto")
    monkeypatch.setenv("ZOTERO_LOCAL", "false")
    monkeypatch.delenv("ZOTERO_API_KEY", raising=False)

    assert effective_tool_profile() == "research"
    assert tool_visible("zotero_semantic_search")
    assert not tool_visible("zotero_add_by_doi")


def test_auto_profile_uses_full_with_api_key(monkeypatch):
    monkeypatch.setenv("ZOTERO_MCP_TOOL_PROFILE", "auto")
    monkeypatch.setenv("ZOTERO_API_KEY", "test-key")

    assert effective_tool_profile() == "full"
    assert tool_visible("zotero_add_by_doi")
    assert not tool_visible("search")


def test_auto_profile_exposes_local_write_tools_without_cloud_key(monkeypatch):
    monkeypatch.setenv("ZOTERO_MCP_TOOL_PROFILE", "auto")
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.delenv("ZOTERO_API_KEY", raising=False)

    assert effective_tool_profile() == "full"
    assert tool_visible("zotero_add_by_doi")


def test_connector_profile_is_small_and_coherent(monkeypatch):
    monkeypatch.setenv("ZOTERO_MCP_TOOL_PROFILE", "connector")

    assert _listed_tool_names() == {
        "search",
        "fetch",
        "zotero_get_capabilities",
    }


def test_research_profile_hides_writes_connectors_and_paths(monkeypatch):
    monkeypatch.setenv("ZOTERO_MCP_TOOL_PROFILE", "research")
    monkeypatch.delenv("ZOTERO_MCP_EXPOSE_LOCAL_PATHS", raising=False)

    names = _listed_tool_names()

    assert "zotero_semantic_search" in names
    assert "zotero_get_search_database_health" in names
    assert "zotero_add_by_doi" not in names
    assert "zotero_get_attachment_path" not in names
    assert "search" not in names
    assert "fetch" not in names


def test_all_profile_still_enforces_path_and_write_capabilities(monkeypatch):
    monkeypatch.setenv("ZOTERO_MCP_TOOL_PROFILE", "all")
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.delenv("ZOTERO_MCP_EXPOSE_LOCAL_PATHS", raising=False)
    monkeypatch.delenv("ZOTERO_API_KEY", raising=False)

    names = _listed_tool_names()

    assert "search" in names
    assert "fetch" in names
    assert "zotero_get_attachment_path" not in names
    assert "zotero_add_from_file" not in names
    assert "zotero_add_by_doi" in names


def test_all_profile_exposes_paths_only_with_explicit_local_opt_in(monkeypatch):
    monkeypatch.setenv("ZOTERO_MCP_TOOL_PROFILE", "all")
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setenv("ZOTERO_MCP_EXPOSE_LOCAL_PATHS", "true")

    names = _listed_tool_names()
    assert "zotero_get_attachment_path" in names
    assert "zotero_add_from_file" in names


def test_citation_import_local_path_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("ZOTERO_MCP_TOOL_PROFILE", "full")
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setenv("ZOTERO_MCP_WRITE_SECRET", "correct-secret")
    monkeypatch.delenv("ZOTERO_MCP_EXPOSE_LOCAL_PATHS", raising=False)
    middleware = ToolProfileMiddleware()

    async def unexpected(_context):
        raise AssertionError("local file path reached the tool without opt-in")

    context = MiddlewareContext(
        message=CallToolRequestParams(
            name="zotero_add_by_bibtex",
            arguments={
                "file_path": "/srv/imports/references.bib",
                "write_secret": "correct-secret",
            },
        ),
        method="tools/call",
    )

    with pytest.raises(ToolError, match="ZOTERO_MCP_EXPOSE_LOCAL_PATHS"):
        asyncio.run(middleware.on_call_tool(context, unexpected))


def test_citation_import_inline_input_does_not_require_path_opt_in(monkeypatch):
    monkeypatch.setenv("ZOTERO_MCP_TOOL_PROFILE", "full")
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setenv("ZOTERO_MCP_WRITE_SECRET", "correct-secret")
    monkeypatch.delenv("ZOTERO_MCP_EXPOSE_LOCAL_PATHS", raising=False)
    middleware = ToolProfileMiddleware()
    seen = {}

    async def execute(context):
        seen.update(context.message.arguments)
        return "ok"

    context = MiddlewareContext(
        message=CallToolRequestParams(
            name="zotero_add_by_bibtex",
            arguments={
                "bibtex": "@article{x, title={Inline}}",
                "write_secret": "correct-secret",
            },
        ),
        method="tools/call",
    )

    assert asyncio.run(middleware.on_call_tool(context, execute)) == "ok"
    assert seen == {"bibtex": "@article{x, title={Inline}}"}


def test_annotation_authoring_requires_pdf_or_epub_dependency(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setattr(
        "zotero_mcp.tool_profiles.importlib.util.find_spec",
        lambda _name: None,
    )

    assert not tool_visible("zotero_create_annotation", "full")
    assert not tool_visible("zotero_create_area_annotation", "full")


def test_annotation_authoring_is_visible_with_epub_only(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setattr(
        "zotero_mcp.tool_profiles.importlib.util.find_spec",
        lambda name: object() if name == "ebooklib" else None,
    )

    assert tool_visible("zotero_create_annotation", "full")
    assert not tool_visible("zotero_create_area_annotation", "full")


def test_capabilities_reports_effective_contract(monkeypatch):
    monkeypatch.setenv("ZOTERO_MCP_TOOL_PROFILE", "research")
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "0")
    monkeypatch.delenv("ZOTERO_API_KEY", raising=False)
    monkeypatch.setattr(
        "zotero_mcp.tools.operational._client.get_local_write_zotero_client",
        lambda: object(),
    )

    result = get_capabilities(ctx=DummyContext())

    assert "**Tool profile:** research" in result
    assert "**Active library:** user:0" in result
    assert "**Write tools usable:** no" in result
    assert "**Write transport:** server-local" in result


def test_capabilities_reports_remote_local_write_target(monkeypatch):
    monkeypatch.setenv("ZOTERO_MCP_TOOL_PROFILE", "full")
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setenv("ZOTERO_MCP_WRITE_SECRET", "admin-secret")
    monkeypatch.setattr(
        "zotero_mcp.tools.operational._client.get_local_write_zotero_client",
        lambda: type(
            "RemoteClient",
            (),
            {"local_endpoint_role": "remote-local"},
        )(),
    )

    result = get_capabilities(ctx=DummyContext())

    assert "**Write transport:** remote-local" in result
    assert "**Write tools usable:** yes" in result


def test_capabilities_path_report_matches_visible_tools(monkeypatch):
    monkeypatch.setenv("ZOTERO_MCP_TOOL_PROFILE", "all")
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.delenv("ZOTERO_MCP_EXPOSE_LOCAL_PATHS", raising=False)

    result = get_capabilities(ctx=DummyContext())

    assert "**Local paths exposed:** no" in result
    assert not tool_visible("zotero_get_attachment_path", "all")


def test_health_tool_scopes_audit_to_active_library(monkeypatch):
    from zotero_mcp import db_health

    captured = {}
    sentinel = object()

    def audit(*args, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(db_health, "audit_semantic_database", audit)
    monkeypatch.setattr(db_health, "format_health_report", lambda report: "healthy")
    monkeypatch.setattr(
        "zotero_mcp.tools.operational._client.get_current_library",
        lambda: {"library_id": "5910265", "library_type": "group"},
    )

    result = get_search_database_health(ctx=DummyContext())

    assert result == "healthy"
    assert captured["library"] == {
        "library_id": "5910265",
        "library_type": "group",
    }
    assert captured["full_integrity"] is False


def test_connector_and_admin_profiles_hide_non_tool_surfaces(monkeypatch):
    middleware = ToolProfileMiddleware()

    async def listed(_context):
        return ["registered"]

    for profile in ("connector", "admin"):
        monkeypatch.setenv("ZOTERO_MCP_TOOL_PROFILE", profile)
        assert asyncio.run(middleware.on_list_prompts(None, listed)) == []
        assert asyncio.run(middleware.on_list_resources(None, listed)) == []
        assert asyncio.run(middleware.on_list_resource_templates(None, listed)) == []


def test_research_profile_keeps_prompts_and_resources(monkeypatch):
    monkeypatch.setenv("ZOTERO_MCP_TOOL_PROFILE", "research")
    middleware = ToolProfileMiddleware()

    async def listed(_context):
        return ["registered"]

    assert asyncio.run(middleware.on_list_prompts(None, listed)) == ["registered"]
    assert asyncio.run(middleware.on_list_resources(None, listed)) == ["registered"]


def test_hidden_resource_and_prompt_calls_are_rejected(monkeypatch):
    monkeypatch.setenv("ZOTERO_MCP_TOOL_PROFILE", "connector")
    middleware = ToolProfileMiddleware()

    async def call_next(_context):
        raise AssertionError("hidden surface must not be reached")

    with pytest.raises(ToolError, match="Prompts are unavailable"):
        asyncio.run(middleware.on_get_prompt(None, call_next))
    with pytest.raises(ToolError, match="Resources are unavailable"):
        asyncio.run(middleware.on_read_resource(None, call_next))


def test_every_write_tool_schema_requires_undisclosed_secret(monkeypatch):
    monkeypatch.setenv("ZOTERO_MCP_TOOL_PROFILE", "full")
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setenv("ZOTERO_MCP_WRITE_SECRET", "never-disclose-this")
    middleware = ToolProfileMiddleware()

    async def listed(_context):
        return await mcp.list_tools(run_middleware=False)

    tools = asyncio.run(middleware.on_list_tools(None, listed))
    write_tool = next(tool for tool in tools if tool.name == "zotero_add_by_doi")

    assert "write_secret" in write_tool.parameters["required"]
    assert write_tool.parameters["properties"]["write_secret"]["type"] == "string"
    assert "never-disclose-this" not in str(write_tool.parameters)
    assert "never-disclose-this" not in (write_tool.description or "")


def test_write_gate_rejects_missing_or_wrong_secret(monkeypatch):
    monkeypatch.setenv("ZOTERO_MCP_TOOL_PROFILE", "full")
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setenv("ZOTERO_MCP_WRITE_SECRET", "correct-secret")
    middleware = ToolProfileMiddleware()

    async def unexpected(_context):
        raise AssertionError("unauthorized write reached the tool")

    for arguments in ({"doi": "10.1/test"}, {"doi": "10.1/test", "write_secret": "wrong"}):
        context = MiddlewareContext(
            message=CallToolRequestParams(name="zotero_add_by_doi", arguments=arguments),
            method="tools/call",
        )
        with pytest.raises(ToolError, match="write-admin secret"):
            asyncio.run(middleware.on_call_tool(context, unexpected))


def test_write_gate_strips_valid_secret_before_tool_execution(monkeypatch):
    monkeypatch.setenv("ZOTERO_MCP_TOOL_PROFILE", "full")
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setenv("ZOTERO_MCP_WRITE_SECRET", "correct-secret")
    middleware = ToolProfileMiddleware()
    seen = {}

    async def execute(context):
        seen.update(context.message.arguments)
        return "ok"

    context = MiddlewareContext(
        message=CallToolRequestParams(
            name="zotero_add_by_doi",
            arguments={"doi": "10.1/test", "write_secret": "correct-secret"},
        ),
        method="tools/call",
    )

    assert asyncio.run(middleware.on_call_tool(context, execute)) == "ok"
    assert seen == {"doi": "10.1/test"}
