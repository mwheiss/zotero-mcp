"""Tests for passage-level chunked retrieval (Tier-1 grounded search).

Covers the pure splitter, page mapping, the best-snippet extractor, the
chunk-aware indexing path in `_process_item_batch`, and chunk grouping in
`_enrich_search_results`. Chunking is opt-in; these tests force it on via a
config file or by setting `_chunking_config` directly.
"""

import json
import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb relies on pydantic v1 paths incompatible with Python 3.14+",
        allow_module_level=True,
    )

from zotero_mcp import semantic_search
from zotero_mcp.semantic_search import (
    _page_for_offset,
    best_snippet,
    split_into_passages,
)

# ---------------------------------------------------------------------------
# Pure splitter
# ---------------------------------------------------------------------------


def test_split_empty_returns_nothing():
    assert split_into_passages("", 100, 20, 10) == []
    assert split_into_passages("   ", 100, 20, 10) == []


def test_split_short_text_single_passage():
    out = split_into_passages("a short body", chunk_size=100, overlap=20, max_chunks=10)
    assert len(out) == 1
    text, start, end = out[0]
    assert text == "a short body"
    assert start == 0


def test_split_offsets_are_monotonic_and_cover_text():
    body = ("Sentence one. " * 200).strip()
    out = split_into_passages(body, chunk_size=200, overlap=40, max_chunks=50)
    assert len(out) > 1
    # Offsets strictly advance and each passage is non-empty.
    prev_start = -1
    for text, start, end in out:
        assert text.strip()
        assert start > prev_start
        assert end > start
        prev_start = start


def test_split_respects_max_chunks():
    body = "word " * 5000
    out = split_into_passages(body, chunk_size=100, overlap=10, max_chunks=7)
    assert len(out) == 7


def test_default_chunk_cap_covers_very_large_document():
    body = "word " * 525_000
    out = split_into_passages(body, chunk_size=6000, overlap=750)

    assert 96 < len(out) < semantic_search.DEFAULT_MAX_CHUNKS_PER_ITEM
    assert out[-1][2] == len(body.strip())


def test_split_overlap_larger_than_chunk_is_tolerated():
    body = "x" * 1000
    out = split_into_passages(body, chunk_size=100, overlap=500, max_chunks=20)
    # Must still terminate and make progress.
    assert out
    assert all(end > start for _, start, end in out)


# ---------------------------------------------------------------------------
# Page mapping
# ---------------------------------------------------------------------------


def test_page_for_offset_without_separators_is_none():
    assert _page_for_offset("no form feeds here", 5) is None


def test_page_for_offset_counts_form_feeds():
    text = "page one\fpage two\fpage three"
    assert _page_for_offset(text, 0) == 1
    assert _page_for_offset(text, text.index("page two")) == 2
    assert _page_for_offset(text, text.index("page three")) == 3


# ---------------------------------------------------------------------------
# Best-snippet extractor
# ---------------------------------------------------------------------------


def test_best_snippet_short_text_returned_whole():
    snippet, off = best_snippet("anything", "tiny doc")
    assert snippet == "tiny doc"
    assert off == 0


def test_best_snippet_centers_on_query_terms():
    head = "irrelevant filler. " * 40
    target = "mindfulness based cognitive therapy reduces relapse"
    text = head + target + " more filler here." * 40
    snippet, off = best_snippet("mindfulness cognitive therapy", text, width=120)
    assert "mindfulness" in snippet.lower()
    assert off > 0  # not the head of the document


def test_best_snippet_no_terms_falls_back_to_head():
    text = "alpha beta gamma delta " * 50
    snippet, off = best_snippet("zzz", text, width=80)
    assert off == 0
    assert snippet.startswith("alpha")


# ---------------------------------------------------------------------------
# Chunk-aware indexing
# ---------------------------------------------------------------------------


class ChunkingFakeChroma:
    """Chroma stub that records upserts and supports chunk bookkeeping."""

    def __init__(self, existing=None):
        self.upserted_ids = []
        self.upserted_docs = []
        self.upserted_metas = []
        self.deleted_parents = []
        self.reconciled = []
        self.events = []
        self.embedding_max_tokens = 8000
        self._existing = set(existing or [])

    def get_existing_ids(self, ids):
        return self._existing & set(ids)

    def delete_item_chunks(self, item_key):
        self.deleted_parents.append(item_key)

    def upsert_documents(self, documents, metadatas, ids):
        self.events.append("upsert")
        self.upserted_docs.extend(documents)
        self.upserted_metas.extend(metadatas)
        self.upserted_ids.extend(ids)

    def reconcile_item_records(self, item_key, expected_ids):
        self.events.append("reconcile")
        self.reconciled.append((item_key, set(expected_ids)))

    def truncate_text(self, text, max_tokens=None):
        return text


def _chunking_search(monkeypatch, config=None, existing=None):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    s = semantic_search.ZoteroSemanticSearch(chroma_client=ChunkingFakeChroma(existing))
    s._chunking_config = {
        "enabled": True,
        "chunk_size": 120,
        "overlap": 20,
        "max_chunks_per_item": 10,
    }
    if config:
        s._chunking_config.update(config)
    return s


def _long_item(key="ITEM0001"):
    body = "Mindfulness reduces relapse. " * 60
    return {
        "key": key,
        "data": {
            "title": "Mindfulness Paper",
            "itemType": "journalArticle",
            "abstractNote": "An abstract.",
            "creators": [],
            "fulltext": body,
        },
    }


def test_chunking_emits_multiple_passage_ids(monkeypatch):
    s = _chunking_search(monkeypatch)
    stats = s._process_item_batch([_long_item("ITEM0001")], force_rebuild=True)

    # One *item* processed, but many chunk ids upserted.
    assert stats["processed"] == 1
    assert len(s.chroma_client.upserted_ids) > 1
    assert all(cid.startswith("ITEM0001#") for cid in s.chroma_client.upserted_ids)
    # Each chunk carries passage provenance in its metadata.
    meta0 = s.chroma_client.upserted_metas[0]
    assert meta0["parent_item_key"] == "ITEM0001"
    assert meta0["chunk_index"] == 0
    assert meta0["n_chunks"] == len(s.chroma_client.upserted_ids)
    assert "char_start" in meta0 and "char_end" in meta0
    assert meta0["index_layout_signature"] == "chunks-v1:120:20:10"


def test_fallback_fulltext_gets_separate_metadata_and_body_passages(monkeypatch):
    s = _chunking_search(monkeypatch)
    item = _long_item("FALLBACK")
    item["data"].update(
        {
            "fulltextSource": "api:attachment:PDF1",
            "publicationTitle": "Ignored Journal",
            "tags": [{"tag": "ignored-tag"}],
            "note": "ignored note",
        }
    )

    prepared = s._prepare_item_batch([item])

    assert prepared.documents[0] == "Mindfulness Paper\n\nAn abstract."
    assert prepared.metadatas[0]["passage_kind"] == "metadata"
    assert all(
        meta["passage_kind"] == "body"
        for meta in prepared.metadatas[1:]
    )
    assert "Mindfulness reduces relapse" in prepared.documents[1]
    combined = "\n".join(prepared.documents)
    assert "Ignored Journal" not in combined
    assert "ignored-tag" not in combined
    assert "ignored note" not in combined


def test_betterissa_indexing_text_is_not_prefixed_or_duplicated(monkeypatch):
    s = _chunking_search(monkeypatch)
    item = _long_item("BETTERISSA")
    item["data"].update(
        {
            "title": "Duplicate Zotero Title",
            "abstractNote": "Duplicate Zotero abstract.",
            "fulltext": (
                "Canonical BetterIssa Title\n\n"
                "Canonical abstract.\n\n"
                + "Filtered body passage. " * 30
            ),
            "fulltextSource": "betterissa-indexing",
        }
    )

    prepared = s._prepare_item_batch([item])

    assert all(
        meta["passage_kind"] == "body"
        for meta in prepared.metadatas
    )
    combined = "\n".join(prepared.documents)
    assert "Canonical BetterIssa Title" in combined
    assert "Duplicate Zotero Title" not in combined
    assert "Duplicate Zotero abstract" not in combined


def test_no_fulltext_indexes_only_title_and_abstract(monkeypatch):
    s = _chunking_search(monkeypatch)
    item = _long_item("METADATA")
    item["data"].update(
        {
            "fulltext": "",
            "creators": [
                {
                    "firstName": "Ada",
                    "lastName": "Author",
                    "creatorType": "author",
                }
            ],
            "tags": [{"tag": "ignored-tag"}],
            "note": "ignored note",
        }
    )

    prepared = s._prepare_item_batch([item])

    assert prepared.documents == ["Mindfulness Paper\n\nAn abstract."]
    assert prepared.metadatas[0]["passage_kind"] == "metadata"
    assert prepared.metadatas[0]["index_content_signature"] == (
        "lean-paper-content-v2"
    )


def test_chunking_added_vs_updated_is_item_granular(monkeypatch):
    # Pretend the item already exists (its chunk #0 is present).
    s = _chunking_search(monkeypatch, existing={"ITEM0001#0"})
    stats = s._process_item_batch([_long_item("ITEM0001")], force_rebuild=False)
    assert stats["updated"] == 1
    assert stats["added"] == 0
    assert s.chroma_client.events == ["upsert", "reconcile"]
    assert s.chroma_client.reconciled[0][0] == "ITEM0001"
    assert all(
        doc_id.startswith("ITEM0001#")
        for doc_id in s.chroma_client.reconciled[0][1]
    )


def test_chunk_layout_migrations_probe_and_reconcile_both_layouts(monkeypatch):
    chunked = _chunking_search(monkeypatch, existing={"ITEM0001"})
    chunk_stats = chunked._process_item_batch([_long_item()], force_rebuild=False)
    assert chunk_stats["updated"] == 1
    assert chunked.chroma_client.reconciled[0][0] == "ITEM0001"

    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    item_level_client = ChunkingFakeChroma(existing={"ITEM0001#0"})
    item_level = semantic_search.ZoteroSemanticSearch(
        chroma_client=item_level_client
    )
    item_stats = item_level._process_item_batch(
        [_long_item()],
        force_rebuild=False,
    )
    assert item_stats["updated"] == 1
    assert item_level_client.reconciled == [("ITEM0001", {"ITEM0001"})]


def test_default_path_still_item_level(monkeypatch):
    # No chunking config -> ids are bare item keys (regression guard).
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    s = semantic_search.ZoteroSemanticSearch(chroma_client=ChunkingFakeChroma())
    stats = s._process_item_batch([_long_item("ITEM0001")], force_rebuild=True)
    assert stats["processed"] == 1
    assert s.chroma_client.upserted_ids == ["ITEM0001"]


def test_chunking_config_loaded_from_file(monkeypatch, tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"semantic_search": {"chunking": {"enabled": True, "chunk_size": 256}}}))
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(
        semantic_search,
        "create_chroma_client",
        lambda _p, **kwargs: ChunkingFakeChroma(),
    )
    s = semantic_search.ZoteroSemanticSearch(config_path=str(cfg))
    assert s._chunking_enabled is True
    assert s._chunking_config["chunk_size"] == 256
    assert (
        s._chunking_config["max_chunks_per_item"]
        == semantic_search.DEFAULT_MAX_CHUNKS_PER_ITEM
    )


def test_changed_chunk_layout_requires_incremental_migration(monkeypatch):
    s = _chunking_search(monkeypatch)
    assert s._index_layout_changed({}) is True
    assert (
        s._index_layout_changed({"index_layout_signature": "chunks-v1:120:20:10"})
        is False
    )
    assert (
        s._index_layout_changed({"index_layout_signature": "chunks-v1:6000:750:96"})
        is True
    )


# ---------------------------------------------------------------------------
# Chunk grouping in enrichment
# ---------------------------------------------------------------------------


class _ZotItemStub:
    def item(self, key):
        return {"key": key, "data": {"title": f"Title {key}"}}


def test_enrich_groups_chunks_back_to_items(monkeypatch):
    s = _chunking_search(monkeypatch)
    s.zotero_client = _ZotItemStub()

    chroma_results = {
        "ids": [["PAP1#3", "PAP1#0", "PAP2#1"]],
        "distances": [[0.10, 0.20, 0.30]],
        "documents": [
            [
                "mindfulness relapse passage from paper one chunk three",
                "intro passage from paper one chunk zero",
                "unrelated passage from paper two",
            ]
        ],
        "metadatas": [
            [
                {
                    "parent_item_key": "PAP1",
                    "chunk_index": 3,
                    "n_chunks": 5,
                    "char_start": 900,
                    "char_end": 1020,
                    "page": 4,
                },
                {"parent_item_key": "PAP1", "chunk_index": 0, "n_chunks": 5, "char_start": 0, "char_end": 120},
                {"parent_item_key": "PAP2", "chunk_index": 1, "n_chunks": 2, "char_start": 100, "char_end": 220},
            ]
        ],
    }

    enriched = s._enrich_search_results(chroma_results, "mindfulness relapse", limit=10)

    # Two distinct items, PAP1 represented by its best (first) passage.
    assert [r["item_key"] for r in enriched] == ["PAP1", "PAP2"]
    best = enriched[0]
    assert best["chunk_index"] == 3
    assert best["page"] == 4
    assert best["char_start"] == 900
    assert best["matched_passage"]
    assert best["zotero_item"]["data"]["title"] == "Title PAP1"
    assert best["best_chunk_similarity_score"] == pytest.approx(0.9)
    assert best["supporting_chunk_count"] == 1
    assert len(best["matched_passages"]) == 2
    assert best["similarity_score"] > best["best_chunk_similarity_score"]


def test_enrich_rejects_overlapping_support_passages(monkeypatch):
    s = _chunking_search(monkeypatch)
    s.zotero_client = _ZotItemStub()
    repeated = "mindfulness relapse " * 30
    chroma_results = {
        "ids": [["PAP1#0", "PAP1#1", "PAP1#4"]],
        "distances": [[0.10, 0.11, 0.12]],
        "documents": [[repeated, repeated, "independent evidence about relapse prevention"]],
        "metadatas": [
            [
                {"parent_item_key": "PAP1", "chunk_index": 0, "char_start": 0},
                {"parent_item_key": "PAP1", "chunk_index": 1, "char_start": 20},
                {"parent_item_key": "PAP1", "chunk_index": 4, "char_start": 1000},
            ]
        ],
    }

    [enriched] = s._enrich_search_results(chroma_results, "mindfulness relapse")

    assert enriched["supporting_chunk_count"] == 1
    assert [p["chunk_index"] for p in enriched["matched_passages"]] == [0, 4]


def test_metadata_and_body_passages_do_not_overlap_by_offset():
    metadata = {
        "passage": "Title and abstract",
        "passage_start": 0,
        "passage_end": 18,
        "meta": {"passage_kind": "metadata"},
    }
    body = {
        "passage": "Body opening text",
        "passage_start": 0,
        "passage_end": 17,
        "meta": {"passage_kind": "body"},
    }

    assert semantic_search._passages_overlap(metadata, body) is False


def test_search_adaptively_fetches_until_it_has_distinct_papers(monkeypatch):
    class AdaptiveChroma(ChunkingFakeChroma):
        def __init__(self):
            super().__init__()
            self.fetches = []

        def search(self, query_texts, n_results, where=None):
            self.fetches.append(n_results)
            count = n_results
            if count <= 12:
                ids = [f"A#{i}" for i in range(count)]
            else:
                ids = [f"A#{i}" for i in range(12)]
                ids.extend(f"P{i}#0" for i in range(1, count - 11))
            return {
                "ids": [ids],
                "distances": [[0.1 + i / 1000 for i in range(len(ids))]],
                "documents": [[f"passage {i}" for i in range(len(ids))]],
                "metadatas": [[{"parent_item_key": raw.split("#")[0]} for raw in ids]],
            }

    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: _ZotItemStub())
    s = semantic_search.ZoteroSemanticSearch(chroma_client=AdaptiveChroma())
    s._chunking_config = {
        "enabled": True,
        "chunk_size": 6000,
        "overlap": 750,
        "max_chunks_per_item": 96,
    }

    result = s.search("query", limit=3)

    assert s.chroma_client.fetches == [12, 24]
    assert len(result["results"]) == 3


def test_search_can_expand_past_old_fixed_ceiling_for_large_books(monkeypatch):
    class BookHeavyChroma(ChunkingFakeChroma):
        def __init__(self):
            super().__init__()
            self.fetches = []

        def count_documents(self):
            return 600

        def search(self, query_texts, n_results, where=None):
            self.fetches.append(n_results)
            count = min(n_results, 600)
            ids = [f"BOOK#{i}" for i in range(min(count, 300))]
            if count > 300:
                ids.extend(f"PAPER{i}#0" for i in range(count - 300))
            return {
                "ids": [ids],
                "distances": [[0.1 + i / 10000 for i in range(len(ids))]],
                "documents": [[f"passage {i}" for i in range(len(ids))]],
                "metadatas": [[
                    {"parent_item_key": raw.split("#")[0]}
                    for raw in ids
                ]],
            }

    monkeypatch.setattr(
        semantic_search,
        "get_zotero_client",
        lambda: _ZotItemStub(),
    )
    search = semantic_search.ZoteroSemanticSearch(
        chroma_client=BookHeavyChroma()
    )
    search._chunking_config = {
        "enabled": True,
        "chunk_size": 6000,
        "overlap": 750,
        "max_chunks_per_item": 768,
    }

    result = search.search("query", limit=3)

    assert search.chroma_client.fetches == [12, 24, 48, 96, 192, 384]
    assert len(result["results"]) == 3




def test_enrich_caps_at_limit(monkeypatch):
    s = _chunking_search(monkeypatch)
    s.zotero_client = _ZotItemStub()
    chroma_results = {
        "ids": [["A#0", "B#0", "C#0", "D#0"]],
        "distances": [[0.1, 0.2, 0.3, 0.4]],
        "documents": [["a", "b", "c", "d"]],
        "metadatas": [[{"parent_item_key": x} for x in ("A", "B", "C", "D")]],
    }
    enriched = s._enrich_search_results(chroma_results, "q", limit=2)
    assert [r["item_key"] for r in enriched] == ["A", "B"]
