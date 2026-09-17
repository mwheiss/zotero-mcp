"""Tests for passage-level chunked retrieval (Tier-1 grounded search).

Covers the pure splitter, page mapping, the best-snippet extractor, the
chunk-aware indexing path in `_process_item_batch`, and chunk grouping in
`_enrich_search_results`. Chunking is opt-in; these tests force it on via a
config file or by setting `_chunking_config` directly.
"""

import hashlib
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


def test_split_balances_chunks_instead_of_leaving_a_small_tail():
    body = "x" * 6575

    out = split_into_passages(body, chunk_size=6000, overlap=750)

    assert len(out) == 2
    lengths = [end - start for _text, start, end in out]
    assert max(lengths) - min(lengths) <= 1
    assert out[1][1] == out[0][2] - 750
    assert out[-1][2] == len(body)


def test_balanced_split_snaps_to_nearby_paragraph_boundary():
    body = "a" * 3400 + "\n\n" + "b" * 3173

    out = split_into_passages(body, chunk_size=6000, overlap=750)

    assert len(out) == 2
    assert out[0][2] == 3402
    assert out[1][1] == 2652
    assert out[-1][2] == len(body)


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


class ReusingChunkingFakeChroma(ChunkingFakeChroma):
    def __init__(self):
        super().__init__()
        self.records = {}
        self.metadata_updates = []

    def get_records(self, ids):
        return {
            doc_id: self.records[doc_id]
            for doc_id in ids
            if doc_id in self.records
        }

    def update_metadatas(self, ids, metadatas):
        self.events.append("metadata")
        self.metadata_updates.extend(zip(ids, metadatas, strict=True))


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
    assert meta0["index_layout_signature"] == "chunks-v2-balanced:120:20:10"


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


def test_unchanged_chunk_text_reuses_vectors_and_refreshes_metadata(monkeypatch):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    client = ReusingChunkingFakeChroma()
    search = semantic_search.ZoteroSemanticSearch(chroma_client=client)
    search._chunking_config = {
        "enabled": True,
        "chunk_size": 120,
        "overlap": 20,
        "max_chunks_per_item": 10,
    }
    item = _long_item("REUSE")
    prepared = search._prepare_item_batch([item])
    client._existing = set(prepared.ids)
    client.records = {
        doc_id: {"document": document, "metadata": {"date_modified": "old"}}
        for doc_id, document in zip(
            prepared.ids,
            prepared.documents,
            strict=True,
        )
    }

    stats = search._process_item_batch([item])

    assert client.upserted_ids == []
    assert len(client.metadata_updates) == len(prepared.expected_ids)
    assert stats["reused_embeddings"] == len(prepared.expected_ids)
    assert stats["updated"] == 1
    assert all(
        metadata["embedding_content_sha256"]
        for _doc_id, metadata in client.metadata_updates
    )
    assert client.reconciled == [("REUSE", set(prepared.expected_ids))]


def test_only_changed_chunks_are_reembedded(monkeypatch):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    client = ReusingChunkingFakeChroma()
    search = semantic_search.ZoteroSemanticSearch(chroma_client=client)
    search._chunking_config = {
        "enabled": True,
        "chunk_size": 120,
        "overlap": 20,
        "max_chunks_per_item": 10,
    }
    item = _long_item("PARTIAL")
    prepared = search._prepare_item_batch([item])
    client._existing = set(prepared.ids)
    client.records = {
        doc_id: {"document": document, "metadata": {}}
        for doc_id, document in zip(
            prepared.ids,
            prepared.documents,
            strict=True,
        )
    }
    changed_id = prepared.ids[1]
    client.records[changed_id]["document"] = "previous chunk contents"

    stats = search._process_item_batch([item])

    assert client.upserted_ids == [changed_id]
    assert len(client.metadata_updates) == len(prepared.expected_ids) - 1
    assert stats["reused_embeddings"] == len(prepared.expected_ids) - 1
    assert client.reconciled == [("PARTIAL", set(prepared.expected_ids))]


def test_force_rebuild_never_reuses_stored_vectors(monkeypatch):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    client = ReusingChunkingFakeChroma()
    search = semantic_search.ZoteroSemanticSearch(chroma_client=client)
    item = _long_item("FORCE")
    prepared = search._prepare_item_batch([item])
    client.records = {
        doc_id: {"document": document, "metadata": metadata}
        for doc_id, document, metadata in zip(
            prepared.ids,
            prepared.documents,
            prepared.metadatas,
            strict=True,
        )
    }

    stats = search._process_item_batch([item], force_rebuild=True)

    assert client.upserted_ids == prepared.expected_ids
    assert client.metadata_updates == []
    assert stats["reused_embeddings"] == 0


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
        s._index_layout_changed({"index_layout_signature": "chunks-v2-balanced:120:20:10"})
        is False
    )
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
    assert best["chunk_id"] == "PAP1#3"
    assert len(best["content_hash"]) == 64
    assert best["page"] == 4
    assert best["char_start"] == 900
    assert best["matched_passage"]
    assert best["zotero_item"]["data"]["title"] == "Title PAP1"
    assert best["best_chunk_similarity_score"] == pytest.approx(0.9)
    assert best["supporting_chunk_count"] == 1
    assert len(best["matched_passages"]) == 2
    assert best["matched_passages"][0]["chunk_id"] == "PAP1#3"
    assert len(best["matched_passages"][0]["content_hash"]) == 64
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


class _RankedChunkSearchFake(ChunkingFakeChroma):
    def __init__(self, ids, *, distances=None, count_available=True, overlapping=True):
        super().__init__()
        self.ids = list(ids)
        self.distances = list(distances or [0.10 + i / 100 for i in range(len(ids))])
        self.count_available = count_available
        self.overlapping = overlapping
        self.fetches = []
        self.filters = []

    def count_documents(self):
        if not self.count_available:
            raise RuntimeError("count unavailable")
        return len(self.ids)

    def distance_to_similarity(self, distance):
        return 1.0 - distance

    def search(self, query_texts, n_results, where=None):
        self.fetches.append(n_results)
        self.filters.append(where)
        ids = self.ids[:n_results]
        return {
            "ids": [ids],
            "distances": [self.distances[:n_results]],
            "documents": [[f"stored text for {raw_id}" for raw_id in ids]],
            "metadatas": [
                [
                    {
                        "parent_item_key": raw_id.split("#", 1)[0],
                        "chunk_index": int(raw_id.split("#", 1)[1]),
                        "n_chunks": 99,
                        "passage_kind": "body",
                        # Identical offsets prove boundary-qualified hits are
                        # not discarded merely because they overlap.
                        "char_start": 100 if self.overlapping else i * 1000,
                        "char_end": 200 if self.overlapping else i * 1000 + 100,
                        "page": 7,
                    }
                    for i, raw_id in enumerate(ids)
                ]
            ],
        }


def _ranked_chunk_search(monkeypatch, client, *, max_chunks=96):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: _ZotItemStub())
    search = semantic_search.ZoteroSemanticSearch(chroma_client=client)
    search._chunking_config = {
        "enabled": True,
        "chunk_size": 6000,
        "overlap": 750,
        "max_chunks_per_item": max_chunks,
    }
    return search


def test_search_uses_limit_plus_one_parent_as_exact_chunk_boundary(monkeypatch):
    client = _RankedChunkSearchFake(
        ["A#0", "A#1", "B#0", "A#2", "B#1", "A#3", "C#0", "A#4"]
    )
    search = _ranked_chunk_search(monkeypatch, client)

    result = search.search("query", limit=2)

    assert client.fetches == [8]
    assert [entry["item_key"] for entry in result["results"]] == ["A", "B"]
    paper_a, paper_b = result["results"]
    assert [p["chunk_id"] for p in paper_a["matched_passages"]] == [
        "A#0",
        "A#1",
        "A#2",
        "A#3",
    ]
    assert [p["chunk_id"] for p in paper_b["matched_passages"]] == ["B#0", "B#1"]
    assert paper_a["supporting_chunk_count"] == 3
    assert paper_a["similarity_score"] == pytest.approx(
        paper_a["best_chunk_similarity_score"]
    )
    first_passage = paper_a["matched_passages"][0]
    assert first_passage["content_hash"] == hashlib.sha256(
        b"stored text for A#0"
    ).hexdigest()
    assert {
        "chunk_index": 0,
        "n_chunks": 99,
        "passage_kind": "body",
        "char_start": 100,
        "char_end": 200,
        "page": 7,
    }.items() <= first_passage.items()
    surfaced = {
        passage["chunk_id"]
        for entry in result["results"]
        for passage in entry["matched_passages"]
    }
    assert "C#0" not in surfaced
    assert "A#4" not in surfaced


def test_search_widens_until_limit_plus_one_parent_is_visible(monkeypatch):
    client = _RankedChunkSearchFake(
        [
            "A#0",
            "A#1",
            "A#2",
            "B#0",
            "A#3",
            "B#1",
            "A#4",
            "B#2",
            "A#5",
            "C#0",
            "A#6",
            "D#0",
        ]
    )
    search = _ranked_chunk_search(monkeypatch, client)

    result = search.search("query", limit=2)

    assert client.fetches == [8, 12]
    assert [entry["item_key"] for entry in result["results"]] == ["A", "B"]
    assert [p["chunk_id"] for p in result["results"][0]["matched_passages"]] == [
        "A#0",
        "A#1",
        "A#2",
        "A#3",
        "A#4",
        "A#5",
    ]


def test_search_widens_past_dominant_parent_when_count_is_unavailable(monkeypatch):
    # max_chunks_per_item body chunks plus one metadata chunk can all belong to
    # A. The fallback ceiling must still fetch B#0 to establish the boundary.
    client = _RankedChunkSearchFake(
        ["A#0", "A#1", "A#2", "A#3", "A#4", "A#5", "B#0"],
        count_available=False,
    )
    search = _ranked_chunk_search(monkeypatch, client, max_chunks=5)

    result = search.search("query", limit=1)

    assert client.fetches == [4, 7]
    assert [entry["item_key"] for entry in result["results"]] == ["A"]
    assert [p["chunk_id"] for p in result["results"][0]["matched_passages"]] == [
        "A#0",
        "A#1",
        "A#2",
        "A#3",
        "A#4",
        "A#5",
    ]


def test_fetched_tail_cannot_add_support_or_break_cutoff_ties(monkeypatch):
    # All distances tie. Ranked position still makes C#0 the first excluded
    # parent; A#1 and B#1 in the oversized reservoir must remain invisible.
    client = _RankedChunkSearchFake(
        ["A#0", "B#0", "C#0", "A#1", "B#1", "D#0", "A#2", "B#2"],
        distances=[0.1] * 8,
    )
    search = _ranked_chunk_search(monkeypatch, client)

    result = search.search("query", limit=2)

    assert [entry["item_key"] for entry in result["results"]] == ["A", "B"]
    assert [entry["supporting_chunk_count"] for entry in result["results"]] == [0, 0]
    assert [
        passage["chunk_id"]
        for entry in result["results"]
        for passage in entry["matched_passages"]
    ] == ["A#0", "B#0"]


@pytest.mark.parametrize(
    ("limit", "ids", "expected_parents"),
    [
        (2, ["A#0", "A#1", "A#2", "A#3", "B#0"], ["A", "B"]),
        (3, ["A#0", "A#1", "A#2", "A#3", "B#0"], ["A", "B"]),
    ],
)
def test_search_without_excluded_parent_preserves_enrichment_fallback(
    monkeypatch,
    limit,
    ids,
    expected_parents,
):
    client = _RankedChunkSearchFake(ids, overlapping=False)
    search = _ranked_chunk_search(monkeypatch, client)

    result = search.search("query", limit=limit)

    assert [entry["item_key"] for entry in result["results"]] == expected_parents
    paper_a = result["results"][0]
    # Historical no-boundary behavior retains at most two supporting passages
    # and their bounded relevance bonus.
    assert [p["chunk_id"] for p in paper_a["matched_passages"]] == [
        "A#0",
        "A#1",
        "A#2",
    ]
    assert paper_a["similarity_score"] > paper_a["best_chunk_similarity_score"]


def test_filtered_single_parent_search_has_no_invented_cutoff(monkeypatch):
    client = _RankedChunkSearchFake(["A#0", "A#1", "A#2"])
    search = _ranked_chunk_search(monkeypatch, client)
    filters = {"item_key": {"$eq": "A"}}

    result = search.search("query", limit=2, filters=filters)

    assert client.filters == [filters]
    assert [entry["item_key"] for entry in result["results"]] == ["A"]
    assert [p["chunk_id"] for p in result["results"][0]["matched_passages"]] == ["A#0"]
