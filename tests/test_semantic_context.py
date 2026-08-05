"""Tests for exact retrieval of stored semantic-search chunks."""

import hashlib

from conftest import DummyContext

from zotero_mcp import semantic_search as semantic_module
from zotero_mcp.tools import search as search_tools


class _FakeChromaClient:
    def __init__(self, records):
        self.records = records
        self.calls = []

    def get_records(self, ids):
        self.calls.append(list(ids))
        return {record_id: self.records[record_id] for record_id in ids if record_id in self.records}


class _FakeSearch:
    def __init__(self, records):
        self.chroma_client = _FakeChromaClient(records)


def _record(item_key, chunk_index, text, *, n_chunks=3):
    return {
        "document": text,
        "metadata": {
            "parent_item_key": item_key,
            "chunk_index": chunk_index,
            "n_chunks": n_chunks,
            "title": "A useful paper",
            "fulltext_source": "betterissa-indexing",
            "embedding_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        },
    }


def _install_search(monkeypatch, records):
    fake = _FakeSearch(records)
    monkeypatch.setattr(
        semantic_module,
        "create_semantic_search",
        lambda _config_path: fake,
    )
    return fake


def test_get_semantic_context_returns_exact_chunk_and_neighbors(monkeypatch):
    records = {
        "PAPER001#0": _record("PAPER001", 0, "first stored passage"),
        "PAPER001#1": _record("PAPER001", 1, "target stored passage"),
        "PAPER001#2": _record("PAPER001", 2, "last stored passage"),
    }
    fake = _install_search(monkeypatch, records)
    expected_hash = hashlib.sha256(b"target stored passage").hexdigest()

    result = search_tools.get_semantic_context(
        chunk_id="PAPER001#1",
        adjacent_chunks=1,
        expected_hash=expected_hash,
        ctx=DummyContext(),
    )

    assert fake.chroma_client.calls == [["PAPER001#0", "PAPER001#1", "PAPER001#2"]]
    assert "**Requested Chunk:** `PAPER001#1`" in result
    assert "**Location:** passage 2/3" in result
    assert "**Indexed Source:** betterissa-indexing" in result
    assert result.index("first stored passage") < result.index("target stored passage")
    assert result.index("target stored passage") < result.index("last stored passage")


def test_get_semantic_context_rejects_stale_hash_without_returning_text(monkeypatch):
    fake = _install_search(
        monkeypatch,
        {"PAPER001#1": _record("PAPER001", 1, "new private passage")},
    )

    result = search_tools.get_semantic_context(
        chunk_id="PAPER001#1",
        expected_hash="0" * 64,
        ctx=DummyContext(),
    )

    assert len(fake.chroma_client.calls) == 1
    assert "changed after the search result" in result
    assert "new private passage" not in result


def test_get_semantic_context_uses_document_hash_not_stored_metadata(monkeypatch):
    record = _record("PAPER001", 1, "stored passage")
    record["metadata"]["embedding_content_sha256"] = "0" * 64
    fake = _install_search(monkeypatch, {"PAPER001#1": record})
    actual_hash = hashlib.sha256(b"stored passage").hexdigest()

    result = search_tools.get_semantic_context(
        chunk_id="PAPER001#1",
        expected_hash=actual_hash,
        ctx=DummyContext(),
    )

    assert fake.chroma_client.calls
    assert f"**Chunk Hash:** `{actual_hash}`" in result
    assert "stored passage" in result


def test_get_semantic_context_reports_missing_chunk(monkeypatch):
    fake = _install_search(monkeypatch, {})

    result = search_tools.get_semantic_context(
        chunk_id="PAPER001#4",
        adjacent_chunks=2,
        ctx=DummyContext(),
    )

    assert fake.chroma_client.calls == [[
        "PAPER001#2",
        "PAPER001#3",
        "PAPER001#4",
        "PAPER001#5",
        "PAPER001#6",
    ]]
    assert "No indexed semantic chunk found" in result


def test_get_semantic_context_rejects_invalid_inputs_before_opening_index(monkeypatch):
    calls = []
    monkeypatch.setattr(
        semantic_module,
        "create_semantic_search",
        lambda _config_path: calls.append(True),
    )
    ctx = DummyContext()

    assert "chunk_id must be" in search_tools.get_semantic_context(
        chunk_id="PAPER001", ctx=ctx
    )
    assert "chunk_id must be" in search_tools.get_semantic_context(
        chunk_id=123, ctx=ctx
    )
    assert "integer from 0 through 2" in search_tools.get_semantic_context(
        chunk_id="PAPER001#0", adjacent_chunks=3, ctx=ctx
    )
    assert "64-character Chunk Hash" in search_tools.get_semantic_context(
        chunk_id="PAPER001#0", expected_hash="short", ctx=ctx
    )
    assert "64-character Chunk Hash" in search_tools.get_semantic_context(
        chunk_id="PAPER001#0", expected_hash=123, ctx=ctx
    )
    assert calls == []


def test_get_semantic_context_rejects_inconsistent_target_provenance(monkeypatch):
    record = _record("OTHER", 1, "wrong item passage")
    _install_search(monkeypatch, {"PAPER001#1": record})

    result = search_tools.get_semantic_context(
        chunk_id="PAPER001#1",
        ctx=DummyContext(),
    )

    assert "inconsistent provenance" in result
    assert "wrong item passage" not in result


def test_get_semantic_context_omits_neighbor_from_a_different_item_version(monkeypatch):
    target = _record("PAPER001", 1, "target passage")
    target["metadata"]["date_modified"] = "2026-08-05"
    neighbor = _record("PAPER001", 2, "stale neighbor passage")
    neighbor["metadata"]["date_modified"] = "2026-08-04"
    _install_search(
        monkeypatch,
        {"PAPER001#1": target, "PAPER001#2": neighbor},
    )

    result = search_tools.get_semantic_context(
        chunk_id="PAPER001#1",
        adjacent_chunks=1,
        ctx=DummyContext(),
    )

    assert "target passage" in result
    assert "stale neighbor passage" not in result


def test_semantic_search_formats_retrievable_chunk_references(monkeypatch):
    class _ResultSearch:
        def search(self, **_kwargs):
            return {
                "results": [
                    {
                        "item_key": "PAPER001",
                        "similarity_score": 0.9,
                        "chunk_id": "PAPER001#4",
                        "content_hash": "a" * 64,
                        "chunk_index": 4,
                        "n_chunks": 8,
                        "matched_passage": "primary evidence",
                        "matched_passages": [
                            {
                                "chunk_id": "PAPER001#4",
                                "content_hash": "a" * 64,
                                "matched_passage": "primary evidence",
                            },
                            {
                                "chunk_id": "PAPER001#7",
                                "content_hash": "b" * 64,
                                "matched_passage": "supporting evidence",
                            },
                        ],
                        "zotero_item": {
                            "key": "PAPER001",
                            "data": {
                                "title": "A useful paper",
                                "itemType": "journalArticle",
                                "creators": [],
                            },
                        },
                    }
                ]
            }

    monkeypatch.setattr(
        semantic_module,
        "create_semantic_search",
        lambda _config_path: _ResultSearch(),
    )
    monkeypatch.setattr(search_tools, "_maybe_fire_presearch_sync", lambda _search: None)

    result = search_tools.semantic_search(
        query="useful evidence",
        ctx=DummyContext(),
    )

    assert "**Chunk ID:** `PAPER001#4`" in result
    assert f"**Chunk Hash:** `{'a' * 64}`" in result
    assert f"`PAPER001#7`: `{'b' * 64}`" in result
    assert "supporting evidence" in result
