"""Tests for opt-in local full-text selection."""

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
        search._get_items_from_source(fulltext=True)


def test_get_items_from_source_proceeds_when_fulltext_with_local_mode(monkeypatch):
    """Requesting fulltext extraction with local mode should not raise."""
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: True)

    # Mock _get_items_from_local_db to avoid needing a real DB
    search = semantic_search.ZoteroSemanticSearch(chroma_client=FakeChromaClient())
    monkeypatch.setattr(search, "_get_items_from_local_db", lambda *a, **kw: [])

    # Should not raise
    result = search._get_items_from_source(fulltext=True)
    assert result == []


def test_get_items_from_source_metadata_only_uses_api_items(monkeypatch):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: False)

    search = semantic_search.ZoteroSemanticSearch(chroma_client=FakeChromaClient())
    captured = []
    monkeypatch.setattr(
        search,
        "_get_items_from_api",
        lambda *a, **kw: captured.append((a, kw)) or [],
    )
    result = search._get_items_from_source(fulltext=False)
    assert result == []
    assert captured == [((None,), {})]


def test_get_items_from_source_rejects_non_boolean_mode(monkeypatch):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    search = semantic_search.ZoteroSemanticSearch(chroma_client=FakeChromaClient())

    with pytest.raises(ValueError, match="true or false"):
        search._get_items_from_source(fulltext="local")


def test_local_fulltext_read_failure_does_not_fall_back_to_api(monkeypatch):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(
        semantic_search,
        "LocalZoteroReader",
        lambda **kwargs: (_ for _ in ()).throw(OSError("locked")),
    )
    search = semantic_search.ZoteroSemanticSearch(chroma_client=FakeChromaClient())
    api_calls = []
    monkeypatch.setattr(search, "_get_items_from_api", lambda *args: api_calls.append(args) or [])

    with pytest.raises(RuntimeError, match="metadata-only API records"):
        search._get_items_from_local_db(extract_fulltext=True)

    # The first call loads canonical metadata before opening SQLite. A fallback
    # would make a second API call from the outer exception handler.
    assert api_calls == [()]


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


def test_local_embedding_queue_prefers_best_selected_fulltext_stably():
    items = [
        SimpleNamespace(key="PDF", fulltext="pdf", fulltext_selection_priority=8),
        SimpleNamespace(key="METADATA", fulltext=None, fulltext_selection_priority=None),
        SimpleNamespace(key="INDEXING_1", fulltext="one", fulltext_selection_priority=0),
        SimpleNamespace(key="OCR", fulltext="ocr", fulltext_selection_priority=2),
        SimpleNamespace(key="INDEXING_2", fulltext="two", fulltext_selection_priority=0),
    ]

    ordered = semantic_search._sort_local_items_by_fulltext_priority(items)

    assert [item.key for item in ordered] == [
        "METADATA",
        "INDEXING_1",
        "INDEXING_2",
        "OCR",
        "PDF",
    ]


@pytest.mark.parametrize(
    ("completed_signature", "expected_extractions"),
    [
        ("same-DONE", ["PENDING"]),
        ("changed-DONE", ["DONE", "PENDING"]),
    ],
)
def test_local_rebuild_resume_extracts_pending_and_changed_items_only(
    monkeypatch,
    completed_signature,
    expected_extractions,
):
    marker = "rebuild-pending:test"

    def local_item(item_id, key):
        return SimpleNamespace(
            item_id=item_id,
            key=key,
            title=f"Title {key}",
            creators="Author, A",
            fulltext=None,
            fulltext_source=None,
            fulltext_selection_priority=None,
            date_modified="2026-08-10T00:00:00Z",
        )

    class Reader:
        def __init__(self):
            self.extract_calls = []
            self.last_extraction_details = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def resolve_library_id(self, *_args):
            return 0

        def get_all_item_keys(self, _library_id):
            return {"DONE", "PENDING"}

        def get_items_with_text(self, **_kwargs):
            return [local_item(1, "DONE"), local_item(2, "PENDING")]

        def get_fulltext_meta_for_item(self, item_id, *_args):
            return [[f"ATT{item_id}", f"storage:{item_id}.txt", "text/plain"]]

        def get_attachment_signature(self, item_id, *_args):
            if item_id == 1:
                return completed_signature
            return "same-PENDING"

        def extract_fulltext_for_item(self, item_id, *_args):
            key = "DONE" if item_id == 1 else "PENDING"
            self.extract_calls.append(key)
            self.last_extraction_details = {"selection_priority": 0}
            return f"Body {key}", "betterissa-indexing"

    class ResumeChroma(FakeChromaClient):
        def __init__(self, search):
            super().__init__()
            self.search = search

        def get_document_metadata(self, key):
            return {
                "has_fulltext": "failed" if key == "PENDING" else True,
                "date_modified": "2026-08-10T00:00:00Z",
                "attachment_signature": f"same-{key}",
                "index_layout_signature": self.search._index_layout_signature,
                "embedding_content_sha256": marker if key == "PENDING" else "current",
            }

    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: True)
    search = semantic_search.ZoteroSemanticSearch(
        chroma_client=FakeChromaClient()
    )
    chroma = ResumeChroma(search)
    reader = Reader()
    monkeypatch.setattr(
        semantic_search,
        "LocalZoteroReader",
        lambda **_kwargs: reader,
    )
    api_items = [
        {
            "key": key,
            "data": {
                "key": key,
                "itemType": "journalArticle",
                "title": f"Title {key}",
                "abstractNote": "Abstract",
                "dateModified": "2026-08-10T00:00:00Z",
                "creators": [],
            },
        }
        for key in ("DONE", "PENDING")
    ]

    def api_scan():
        search._last_api_attachment_keys_by_parent = None
        return api_items

    monkeypatch.setattr(search, "_get_items_from_api", api_scan)

    items = search._get_items_from_local_db(
        extract_fulltext=True,
        chroma_client=chroma,
        rebuild_marker=marker,
    )

    assert reader.extract_calls == expected_extractions
    assert [item["key"] for item in items] == expected_extractions
