"""Connector links follow the effective request library."""

import asyncio
import json

import pytest

from zotero_mcp import client as zclient
from zotero_mcp.server import mcp
from zotero_mcp.tools import connectors


class DummyContext:
    def error(self, *_args, **_kwargs):
        return None


class FakeZotero:
    def item(self, key):
        return {"key": key, "data": {"title": "Test item", "itemType": "journalArticle"}}


def test_item_urls_use_current_personal_and_group_library(monkeypatch):
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "0")
    monkeypatch.setenv("ZOTERO_LIBRARY_TYPE", "user")

    assert connectors._item_urls("ABCD1234") == (
        "zotero://select/library/items/ABCD1234",
        "",
    )
    assert connectors._citation_url("ABCD1234") == (
        "zotero://select/library/items/ABCD1234"
    )
    assert connectors._citation_url(
        "ABCD1234",
        {"library": {"type": "user", "id": 20765677}},
    ) == "https://www.zotero.org/users/20765677/items/ABCD1234"

    zclient.set_active_library("5910265", "group")
    try:
        assert connectors._item_urls("ABCD1234") == (
            "zotero://select/groups/5910265/items/ABCD1234",
            "https://www.zotero.org/groups/5910265/items/ABCD1234",
        )
        assert connectors._citation_url("ABCD1234") == (
            "https://www.zotero.org/groups/5910265/items/ABCD1234"
        )
    finally:
        zclient.clear_active_library()


def test_fetch_uses_runtime_library_in_citation_urls(monkeypatch):
    monkeypatch.setattr(connectors._client, "get_zotero_client", lambda: FakeZotero())
    monkeypatch.setattr(
        connectors,
        "get_item_fulltext",
        lambda item_key, ctx: "# Test item\n\n## Full Text\n\nUseful body text for citation.",
    )

    zclient.set_active_library("5910265", "group")
    try:
        payload = connectors.connector_fetch(
            "ABCD1234", ctx=DummyContext()
        ).model_dump()
    finally:
        zclient.clear_active_library()

    expected = "https://www.zotero.org/groups/5910265/items/ABCD1234"
    assert payload["url"] == expected
    assert payload["metadata"]["web_url"] == expected
    assert payload["metadata"]["zotero_url"] == (
        "zotero://select/groups/5910265/items/ABCD1234"
    )


def test_search_uses_the_preferred_group_citation_url(monkeypatch):
    from zotero_mcp import semantic_search

    class FakeSearch:
        def search(self, **_kwargs):
            return {
                "results": [
                    {
                        "item_key": "ABCD1234",
                        "zotero_item": {"data": {"title": "Test item"}},
                    }
                ]
            }

    monkeypatch.setattr(
        semantic_search,
        "create_semantic_search",
        lambda *_args, **_kwargs: FakeSearch(),
    )

    zclient.set_active_library("5910265", "group")
    try:
        search_payload = connectors.chatgpt_connector_search(
            "test", ctx=DummyContext()
        ).model_dump()
    finally:
        zclient.clear_active_library()

    expected = "https://www.zotero.org/groups/5910265/items/ABCD1234"
    assert search_payload["results"][0]["url"] == expected


def test_search_omits_results_without_fetchable_item_keys(monkeypatch):
    from zotero_mcp import semantic_search

    class FakeSearch:
        def search(self, **_kwargs):
            return {
                "results": [
                    {"item_key": "", "zotero_item": {"data": {"title": "Missing"}}},
                    {"item_key": "TOO-LONG-KEY", "zotero_item": {"data": {"title": "Bad"}}},
                    {"item_key": "ABCD1234", "zotero_item": {"data": {"title": "Good"}}},
                ]
            }

    monkeypatch.setattr(
        semantic_search,
        "create_semantic_search",
        lambda *_args, **_kwargs: FakeSearch(),
    )

    payload = connectors.chatgpt_connector_search(
        "test", ctx=DummyContext()
    ).model_dump()

    assert [result["id"] for result in payload["results"]] == ["ABCD1234"]


def test_fetch_rejects_malformed_item_key_before_zotero_access(monkeypatch):
    monkeypatch.setattr(
        connectors._client,
        "get_zotero_client",
        lambda: (_ for _ in ()).throw(AssertionError("Zotero must not be called")),
    )

    with pytest.raises(ValueError, match="invalid Zotero item key"):
        connectors.connector_fetch("../INVALID", ctx=DummyContext())


def test_connector_result_is_structured_and_mirrored_as_json_text():
    payload = connectors.ConnectorFetchOutput(
        id="ABCD1234",
        title="Test",
        text="Body",
        url="https://example.test/item",
        metadata={"source": "test"},
    )
    tools = asyncio.run(mcp.list_tools(run_middleware=False))
    tool = next(value for value in tools if value.name == "fetch")

    result = tool.convert_result(payload)

    assert result.structured_content == payload.model_dump()
    assert json.loads(result.content[0].text) == payload.model_dump()


def test_connector_search_backend_failure_is_not_an_empty_success(monkeypatch):
    from zotero_mcp import semantic_search

    monkeypatch.setattr(
        semantic_search,
        "create_semantic_search",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("semantic backend down")
        ),
    )

    with pytest.raises(RuntimeError, match="Connector search failed"):
        connectors.chatgpt_connector_search("test", ctx=DummyContext())
