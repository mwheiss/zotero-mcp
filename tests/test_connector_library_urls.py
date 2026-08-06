"""Connector links follow the effective request library."""

import json

from zotero_mcp import client as zclient
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
        payload = json.loads(connectors.connector_fetch("ABCD1234", ctx=DummyContext()))
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
        search_payload = json.loads(
            connectors.chatgpt_connector_search("test", ctx=DummyContext())
        )
    finally:
        zclient.clear_active_library()

    expected = "https://www.zotero.org/groups/5910265/items/ABCD1234"
    assert search_payload["results"][0]["url"] == expected
