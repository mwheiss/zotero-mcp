"""Tests for local, API, and metadata-only full-text source selection."""

import sys
from types import SimpleNamespace

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

from zotero_mcp import semantic_search


class FakeChromaClient:
    def __init__(self):
        self.embedding_max_tokens = 8000

    def get_existing_ids(self, ids):
        return set()

    def upsert_documents(self, documents, metadatas, ids):
        pass

    def reset_collection(self):
        pass


def test_get_items_from_source_aborts_when_fulltext_without_local_mode(monkeypatch):
    """Requesting fulltext extraction without local mode should raise RuntimeError."""
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: False)
    search = semantic_search.ZoteroSemanticSearch(chroma_client=FakeChromaClient())

    with pytest.raises(RuntimeError, match="ZOTERO_LOCAL"):
        search._get_items_from_source(fulltext_source="local")


def test_get_items_from_source_proceeds_when_fulltext_with_local_mode(monkeypatch):
    """Requesting fulltext extraction with local mode should not raise."""
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: True)

    # Mock _get_items_from_local_db to avoid needing a real DB
    search = semantic_search.ZoteroSemanticSearch(chroma_client=FakeChromaClient())
    monkeypatch.setattr(search, "_get_items_from_local_db", lambda *a, **kw: [])

    # Should not raise
    result = search._get_items_from_source(fulltext_source="local")
    assert result == []


def test_get_items_from_source_metadata_only_uses_api_items(monkeypatch):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: False)

    search = semantic_search.ZoteroSemanticSearch(chroma_client=FakeChromaClient())
    captured = {}
    monkeypatch.setattr(
        search,
        "_get_items_from_api",
        lambda *a, **kw: captured.update(kw) or [],
    )
    result = search._get_items_from_source(fulltext_source="none")
    assert result == []
    assert captured["include_fulltext"] is False


def test_get_items_from_source_api_requests_cached_fulltext(monkeypatch):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    search = semantic_search.ZoteroSemanticSearch(chroma_client=FakeChromaClient())
    captured = {}
    monkeypatch.setattr(
        search,
        "_get_items_from_api",
        lambda *a, **kw: captured.update(kw) or [],
    )

    assert search._get_items_from_source(fulltext_source="api") == []
    assert captured["include_fulltext"] is True


def test_get_items_from_source_rejects_unknown_mode(monkeypatch):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    search = semantic_search.ZoteroSemanticSearch(chroma_client=FakeChromaClient())

    with pytest.raises(ValueError, match="api, local, none"):
        search._get_items_from_source(fulltext_source="surprise")


def test_local_fulltext_overlays_complete_api_metadata(monkeypatch):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    search = semantic_search.ZoteroSemanticSearch(
        chroma_client=FakeChromaClient()
    )
    local_item = SimpleNamespace(
        key="ITEM1",
        item_type="journalArticle",
        title="Stale local title",
        abstract="Stale local abstract",
        extra="",
        date_added="2020-01-01",
        date_modified="2020-01-02",
        creators="Local, Author",
        notes="local workflow note",
        fulltext="Selected BetterIssa body",
        fulltext_source="betterissa-indexing",
        _fulltext_attempted=False,
        _attachment_keys="ATT1,ATT2",
        _attachment_signature="signature-v1",
    )
    api_item = {
        "key": "ITEM1",
        "version": 42,
        "data": {
            "key": "ITEM1",
            "itemType": "bookSection",
            "title": "Canonical API title",
            "abstractNote": "Canonical API abstract",
            "bookTitle": "Canonical API book",
            "date": "2026",
            "dateModified": "2026-07-31T00:00:00Z",
            "DOI": "10.1234/example",
            "url": "https://example.test/paper",
            "tags": [{"tag": "canonical-tag"}],
            "creators": [
                {
                    "firstName": "Canonical",
                    "lastName": "Author",
                    "creatorType": "author",
                }
            ],
        },
    }

    merged = search._merge_local_fulltext_with_api_metadata(
        local_item,
        api_item,
        extract_fulltext=True,
    )

    assert merged["version"] == 42
    assert merged["data"]["title"] == "Canonical API title"
    assert merged["data"]["bookTitle"] == "Canonical API book"
    assert merged["data"]["DOI"] == "10.1234/example"
    assert merged["data"]["tags"] == [{"tag": "canonical-tag"}]
    assert merged["data"]["fulltext"] == "Selected BetterIssa body"
    assert merged["data"]["fulltextSource"] == "betterissa-indexing"
    assert merged["data"]["attachmentKeys"] == "ATT1,ATT2"
    assert merged["data"]["attachmentSignature"] == "signature-v1"
    assert "fulltext" not in api_item["data"]
