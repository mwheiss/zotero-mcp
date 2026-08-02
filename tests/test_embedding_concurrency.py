"""Tests for opt-in concurrent realtime embedding during update-db."""

import threading
import time

from zotero_mcp import semantic_search


def _items(count: int) -> list[dict]:
    return [
        {
            "key": f"ITEM{i:04d}",
            "data": {
                "title": f"Item {i}",
                "itemType": "journalArticle",
                "abstractNote": "Embedding concurrency test.",
                "creators": [],
            },
        }
        for i in range(count)
    ]


class _ZoteroClient:
    def last_modified_version(self):
        return 123


class _ConcurrentChroma:
    embedding_model = "openai"
    embedding_max_tokens = 8000

    def __init__(self):
        self._lock = threading.Lock()
        self.active_embeddings = 0
        self.max_active_embeddings = 0
        self.upserted_batches: list[list[str]] = []
        self.writer_threads: set[int] = set()
        self.sequential_upserts = 0

    def reset_collection(self):
        pass

    def begin_staged_rebuild(self):
        pass

    def commit_staged_rebuild(self):
        pass

    def abort_staged_rebuild(self):
        pass

    def truncate_text(self, text, max_tokens=None):
        return text

    def get_existing_ids(self, ids):
        return set()

    def get_all_ids(self):
        return set()

    def embed_documents(self, documents):
        with self._lock:
            self.active_embeddings += 1
            self.max_active_embeddings = max(
                self.max_active_embeddings,
                self.active_embeddings,
            )
        time.sleep(0.05)
        with self._lock:
            self.active_embeddings -= 1
        return [[float(i), 1.0] for i, _ in enumerate(documents)]

    def upsert_embeddings(self, documents, metadatas, ids, embeddings):
        assert len(documents) == len(metadatas) == len(ids) == len(embeddings)
        self.writer_threads.add(threading.get_ident())
        self.upserted_batches.append(list(ids))

    def upsert_documents(self, documents, metadatas, ids):
        self.sequential_upserts += 1
        self.upserted_batches.append(list(ids))


def _search(monkeypatch, chroma, items):
    monkeypatch.setenv("ZOTERO_MCP_FORCE_UPDATE", "1")
    monkeypatch.setattr(
        semantic_search,
        "get_zotero_client",
        lambda: _ZoteroClient(),
    )
    search = semantic_search.ZoteroSemanticSearch(chroma_client=chroma)
    monkeypatch.setattr(
        search,
        "_get_items_from_source",
        lambda **kwargs: items,
    )
    monkeypatch.setattr(search, "_save_update_config", lambda **kwargs: None)
    return search


def test_concurrent_embeddings_overlap_with_per_entry_serialized_writes(monkeypatch):
    items = _items(51)
    chroma = _ConcurrentChroma()
    search = _search(monkeypatch, chroma, items)
    caller_thread = threading.get_ident()

    stats = search.update_database(
        force_full_rebuild=True,
        embedding_concurrency=2,
    )

    assert stats["errors"] == 0
    assert stats["processed_items"] == 51
    assert stats["embedding_concurrency"] == 2
    assert chroma.max_active_embeddings == 2
    assert chroma.writer_threads == {caller_thread}
    assert all(len(batch_ids) == 1 for batch_ids in chroma.upserted_batches)
    assert {
        item_id
        for batch_ids in chroma.upserted_batches
        for item_id in batch_ids
    } == {item["key"] for item in items}


def test_default_update_path_writes_each_entry_immediately(
    monkeypatch,
    capsys,
):
    chroma = _ConcurrentChroma()
    search = _search(monkeypatch, chroma, _items(2))

    stats = search.update_database(force_full_rebuild=True)

    assert stats["errors"] == 0
    assert chroma.sequential_upserts == 2
    assert chroma.upserted_batches == [["ITEM0000"], ["ITEM0001"]]
    assert chroma.max_active_embeddings == 0
    assert "embedding_concurrency" not in stats
    progress = capsys.readouterr().err
    assert "| ETA " in progress
    assert "2/2 finished" in progress
    assert "| Last: Item 1" in progress


class _SkewedChroma(_ConcurrentChroma):
    def __init__(self):
        super().__init__()
        self.slow_started = threading.Event()
        self.replacement_started = threading.Event()
        self.slow_observed_replacement = False

    def embed_documents(self, documents):
        document = documents[0]
        if "Item 0" in document:
            self.slow_started.set()
            self.slow_observed_replacement = self.replacement_started.wait(1)
        elif "Item 1" in document:
            assert self.slow_started.wait(1)
        elif "Item 2" in document:
            self.replacement_started.set()
        return [[float(i), 1.0] for i, _ in enumerate(documents)]


class _ReusingConcurrentChroma(_ConcurrentChroma):
    def __init__(self):
        super().__init__()
        self.records = {}
        self.metadata_updates = []
        self.embedding_calls = 0

    def get_records(self, ids):
        return {
            doc_id: self.records[doc_id]
            for doc_id in ids
            if doc_id in self.records
        }

    def update_metadatas(self, ids, metadatas):
        self.metadata_updates.extend(zip(ids, metadatas, strict=True))

    def embed_documents(self, documents):
        self.embedding_calls += 1
        return super().embed_documents(documents)


def test_fast_worker_takes_next_entry_while_other_worker_is_slow(monkeypatch):
    chroma = _SkewedChroma()
    search = _search(monkeypatch, chroma, _items(3))

    stats = search.update_database(
        force_full_rebuild=True,
        embedding_concurrency=2,
    )

    assert stats["errors"] == 0
    assert chroma.slow_observed_replacement is True
    assert all(len(batch_ids) == 1 for batch_ids in chroma.upserted_batches)


def test_concurrent_metadata_only_updates_do_not_call_encoder(monkeypatch):
    items = _items(3)
    chroma = _ReusingConcurrentChroma()
    search = _search(monkeypatch, chroma, items)
    prepared = search._prepare_item_batch(items)
    chroma.records = {
        doc_id: {"document": document, "metadata": {}}
        for doc_id, document in zip(
            prepared.ids,
            prepared.documents,
            strict=True,
        )
    }

    stats = search.update_database(
        embedding_concurrency=2,
    )

    assert stats["errors"] == 0
    assert stats["reused_embeddings"] == 3
    assert chroma.embedding_calls == 0
    assert len(chroma.metadata_updates) == 3
    assert chroma.upserted_batches == []


def test_concurrency_rejects_non_openai_embedding_models(monkeypatch):
    chroma = _ConcurrentChroma()
    chroma.embedding_model = "default"
    search = _search(monkeypatch, chroma, [])

    stats = search.update_database(embedding_concurrency=2)

    assert "requires realtime OpenAI-compatible embeddings" in stats["error"]


def test_concurrency_rejects_openai_batch_mode(monkeypatch):
    chroma = _ConcurrentChroma()
    search = _search(monkeypatch, chroma, [])
    monkeypatch.setattr(search, "_resolve_openai_batch_enabled", lambda value: True)

    stats = search.update_database(embedding_concurrency=2)

    assert "cannot be combined with OpenAI Batch mode" in stats["error"]
