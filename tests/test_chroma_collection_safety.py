"""Regression tests for non-destructive embedding-model mismatches."""

import hashlib
import json
import threading
from types import SimpleNamespace

import pytest

from zotero_mcp import chroma_client


class _FakePersistentClient:
    def __init__(self, *, conflict=False, metadata=None):
        self.conflict = conflict
        if metadata is None:
            metadata = {"hnsw:space": "cosine"}
        self.collection = SimpleNamespace(metadata=metadata)
        self.deleted = []
        self._sysdb = SimpleNamespace(get_collections=lambda **kwargs: [])

    def get_or_create_collection(self, **kwargs):
        if self.conflict:
            raise ValueError("embedding function conflict")
        return self.collection

    def get_collection(self, **kwargs):
        return self.collection

    def delete_collection(self, **kwargs):
        self.deleted.append(kwargs["name"])


def _build_client(monkeypatch, fake, **kwargs):
    monkeypatch.setattr(chroma_client.chromadb, "PersistentClient", lambda **unused: fake)
    monkeypatch.setattr(
        chroma_client.ChromaClient,
        "_create_embedding_function",
        lambda self: SimpleNamespace(model_name="api-alias"),
    )
    return chroma_client.ChromaClient(
        persist_directory="/tmp/chroma-safety-test",
        embedding_model="openai",
        embedding_config={
            "model_name": "api-alias",
            "model_identity": "Qwen3-Embedding-8B-Q8_0",
        },
        **kwargs,
    )


def test_embedding_conflict_never_deletes_collection(monkeypatch):
    fake = _FakePersistentClient(conflict=True)

    with pytest.raises(RuntimeError, match="left untouched"):
        _build_client(monkeypatch, fake)

    assert fake.deleted == []


def test_confirmed_rebuild_can_open_conflicting_collection(monkeypatch):
    fake = _FakePersistentClient(conflict=True)

    client = _build_client(
        monkeypatch,
        fake,
        allow_embedding_mismatch=True,
    )

    assert client.collection is fake.collection
    assert client._pending_embedding_mismatch is True
    assert fake.deleted == []


def test_explicit_proxy_model_identity_detects_backend_change(monkeypatch):
    fake = _FakePersistentClient(
        metadata={"zotero_mcp_embedding_identity": "old-backend"}
    )

    with pytest.raises(RuntimeError, match="old-backend"):
        _build_client(monkeypatch, fake)

    assert fake.deleted == []


def test_reset_stamps_embedding_identity(monkeypatch):
    fake = _FakePersistentClient()
    created = {}

    def create_collection(**kwargs):
        created.update(kwargs)
        return fake.collection

    fake.create_collection = create_collection
    client = _build_client(monkeypatch, fake)

    client.reset_collection()

    assert created["metadata"]["zotero_mcp_embedding_identity"] == (
        "Qwen3-Embedding-8B-Q8_0"
    )
    assert created["metadata"]["hnsw:space"] == "cosine"


def _staging_client(tmp_path):
    persist_directory = tmp_path / "chroma-staging"
    raw_client = chroma_client.chromadb.PersistentClient(path=str(persist_directory))
    collection = raw_client.create_collection(
        "zotero_library",
        metadata={
            "hnsw:space": "cosine",
            "zotero_mcp_embedding_identity": "test:model",
        },
    )
    collection.add(ids=["OLD"], embeddings=[[1.0, 0.0]])
    client = chroma_client.ChromaClient.__new__(chroma_client.ChromaClient)
    client.client = raw_client
    client.collection = collection
    client.collection_name = "zotero_library"
    client.embedding_function = None
    client.embedding_identity = "test:model"
    client.persist_directory = str(persist_directory)
    marker_token = hashlib.sha256(b"zotero_library").hexdigest()[:16]
    client._rebuild_marker_path = (
        persist_directory / f".zotero-mcp-rebuild-{marker_token}.json"
    )
    client._pending_embedding_mismatch = False
    client._rebuild_original_collection = None
    client._rebuild_staging_name = None
    return client, raw_client


def test_staged_rebuild_swaps_only_after_replacement_is_ready(tmp_path):
    client, raw_client = _staging_client(tmp_path)

    client.begin_staged_rebuild()
    assert raw_client.get_collection("zotero_library").get()["ids"] == ["OLD"]
    client.collection.add(ids=["NEW"], embeddings=[[0.0, 1.0]])

    client.commit_staged_rebuild()

    assert raw_client.get_collection("zotero_library").get()["ids"] == ["NEW"]


def test_prune_orphan_segments_preserves_references_and_removes_orphans(tmp_path):
    client, raw_client = _staging_client(tmp_path)
    persist_directory = tmp_path / "chroma-staging"
    old_orphan = persist_directory / "44444444-4444-4444-4444-444444444444"
    fresh_orphan = persist_directory / "55555555-5555-5555-5555-555555555555"
    old_orphan.mkdir()
    fresh_orphan.mkdir()
    old_file = old_orphan / "index.bin"
    old_file.write_bytes(b"retired")
    (fresh_orphan / "index.bin").write_bytes(b"possibly in use")
    ledger_path = persist_directory / ".zotero-mcp-orphan-segments.json"
    ledger_path.write_text(json.dumps({old_orphan.name: 0}))

    result = client.prune_orphan_segment_directories()

    assert result["removed_directories"] == 2
    assert result["removed_bytes"] == len(b"retiredpossibly in use")
    assert not old_orphan.exists()
    assert not fresh_orphan.exists()
    assert not ledger_path.exists()
    assert raw_client.get_collection("zotero_library").count() == 1


def test_prune_orphan_segments_refuses_during_collection_swap(tmp_path):
    client, _raw_client = _staging_client(tmp_path)
    orphan = (
        tmp_path
        / "chroma-staging"
        / "66666666-6666-6666-6666-666666666666"
    )
    orphan.mkdir()
    client._rebuild_original_collection = client.collection
    client._rebuild_marker_path.write_text("{}")

    result = client.prune_orphan_segment_directories()

    assert result["skipped_reason"] == "collection_swap_in_progress"
    assert orphan.exists()


def test_exclusive_lifecycle_lock_waits_for_active_reader(tmp_path):
    persist_directory = tmp_path / "lifecycle-lock"
    reader_started = threading.Event()
    release_reader = threading.Event()
    writer_acquired = threading.Event()
    errors = []

    class BlockingCollection:
        def count(self):
            reader_started.set()
            if not release_reader.wait(timeout=5):
                raise TimeoutError("reader was not released")
            return 1

    client = chroma_client.ChromaClient.__new__(chroma_client.ChromaClient)
    client.persist_directory = str(persist_directory)
    client.collection = BlockingCollection()

    def read():
        try:
            assert client.count_documents() == 1
        except Exception as error:
            errors.append(error)

    def write():
        try:
            with chroma_client.index_lifecycle_lock(
                persist_directory,
                exclusive=True,
            ):
                writer_acquired.set()
        except Exception as error:
            errors.append(error)

    reader = threading.Thread(target=read)
    writer = threading.Thread(target=write)
    reader.start()
    assert reader_started.wait(timeout=2)
    writer.start()
    assert not writer_acquired.wait(timeout=0.1)

    release_reader.set()
    reader.join(timeout=2)
    writer.join(timeout=2)

    assert not reader.is_alive()
    assert not writer.is_alive()
    assert writer_acquired.is_set()
    assert errors == []


def test_chroma_operations_declare_lifecycle_lock_mode():
    shared = {
        "add_documents",
        "upsert_documents",
        "upsert_embeddings",
        "get_records",
        "update_metadatas",
        "update_item_metadata",
        "search",
        "delete_documents",
        "delete_item_chunks",
        "get_item_chunk_ids",
        "get_item_embedding_hashes",
        "get_item_record_hashes",
        "reconcile_item_records",
        "delete_item_records",
        "get_collection_info",
        "count_documents",
        "document_exists",
        "get_document_metadata",
        "get_existing_ids",
        "get_all_ids",
    }
    exclusive = {
        "reset_collection",
        "prune_orphan_segment_directories",
        "begin_staged_rebuild",
        "commit_staged_rebuild",
        "abort_staged_rebuild",
    }

    for method_name in shared:
        method = getattr(chroma_client.ChromaClient, method_name)
        assert method._zotero_mcp_lifecycle_exclusive is False
    for method_name in exclusive:
        method = getattr(chroma_client.ChromaClient, method_name)
        assert method._zotero_mcp_lifecycle_exclusive is True


def test_aborted_staged_rebuild_leaves_live_collection_untouched(tmp_path):
    client, raw_client = _staging_client(tmp_path)

    client.begin_staged_rebuild()
    client.collection.add(ids=["PARTIAL"], embeddings=[[0.5, 0.5]])
    client.abort_staged_rebuild()

    assert raw_client.get_collection("zotero_library").get()["ids"] == ["OLD"]


def test_existing_client_refreshes_handle_after_cross_process_swap(tmp_path):
    persist_directory = str(tmp_path / "cross-process")
    reader = chroma_client.ChromaClient(persist_directory=persist_directory)
    reader.upsert_embeddings(["old"], [{"title": "Old"}], ["OLD"], [[1.0, 0.0]])
    updater = chroma_client.ChromaClient(persist_directory=persist_directory)

    updater.begin_staged_rebuild()
    updater.upsert_embeddings(
        ["new"], [{"title": "New"}], ["NEW"], [[0.0, 1.0]]
    )
    updater.commit_staged_rebuild()

    assert reader.get_all_ids() == {"NEW"}


def test_interrupted_swap_restores_backup_before_opening(tmp_path):
    client, raw_client = _staging_client(tmp_path)
    client.begin_staged_rebuild()
    client.collection.add(ids=["PARTIAL"], embeddings=[[0.5, 0.5]])
    staging_name = client._rebuild_staging_name
    backup_name = "zotero_library__replaced_interrupted"
    client._write_rebuild_marker(staging_name, backup_name)
    marker = json.loads(client._rebuild_marker_path.read_text())
    marker["pid"] = 999_999_999
    client._rebuild_marker_path.write_text(json.dumps(marker))
    client._rebuild_original_collection.modify(name=backup_name)

    assert client._recover_interrupted_rebuild() is False

    assert set(raw_client.get_collection("zotero_library").get()["ids"]) == {"OLD"}
    names = {collection.name for collection in raw_client.list_collections()}
    assert "zotero_library" in names
    assert staging_name not in names
    assert not client._rebuild_marker_path.exists()


def test_interrupted_swap_keeps_activated_replacement(tmp_path):
    client, raw_client = _staging_client(tmp_path)
    client.begin_staged_rebuild()
    client.collection.add(ids=["NEW"], embeddings=[[0.0, 1.0]])
    staging_name = client._rebuild_staging_name
    backup_name = "zotero_library__replaced_interrupted"
    client._write_rebuild_marker(staging_name, backup_name)
    marker = json.loads(client._rebuild_marker_path.read_text())
    marker["pid"] = 999_999_999
    client._rebuild_marker_path.write_text(json.dumps(marker))
    client._rebuild_original_collection.modify(name=backup_name)
    client.collection.modify(name="zotero_library")

    assert client._recover_interrupted_rebuild() is False

    assert set(raw_client.get_collection("zotero_library").get()["ids"]) == {"NEW"}
    names = {collection.name for collection in raw_client.list_collections()}
    assert backup_name in names
    assert not client._rebuild_marker_path.exists()


class _RecordCollection:
    def __init__(self, ids):
        self.ids = set(ids)
        self.deleted = []

    def get(self, ids=None, where=None, include=None):
        if where is not None:
            parent = where["parent_item_key"]
            found = sorted(
                doc_id for doc_id in self.ids if doc_id.startswith(f"{parent}#")
            )
        else:
            found = sorted(self.ids & set(ids or []))
        return {"ids": found}

    def delete(self, ids=None, where=None):
        if where is not None:
            parent = where["parent_item_key"]
            ids = [doc_id for doc_id in self.ids if doc_id.startswith(f"{parent}#")]
        removed = set(ids or [])
        self.deleted.append(removed)
        self.ids.difference_update(removed)


def test_reconcile_removes_stale_chunks_and_opposite_layout():
    client = chroma_client.ChromaClient.__new__(chroma_client.ChromaClient)
    client.collection = _RecordCollection({"ITEM", "ITEM#0", "ITEM#1", "ITEM#2"})

    client.reconcile_item_records("ITEM", {"ITEM#0", "ITEM#1"})

    assert client.collection.ids == {"ITEM#0", "ITEM#1"}


def test_delete_item_records_removes_bare_and_chunked_ids():
    client = chroma_client.ChromaClient.__new__(chroma_client.ChromaClient)
    client.collection = _RecordCollection({"ITEM", "ITEM#0", "ITEM#1", "OTHER#0"})

    client.delete_item_records("ITEM")

    assert client.collection.ids == {"OTHER#0"}


def test_collection_id_read_errors_are_not_treated_as_empty():
    class FailingCollection:
        def get(self, **kwargs):
            raise OSError("database unavailable")

    client = chroma_client.ChromaClient.__new__(chroma_client.ChromaClient)
    client.collection = FailingCollection()

    with pytest.raises(OSError, match="database unavailable"):
        client.get_all_ids()


def test_record_reads_and_metadata_only_updates_preserve_documents():
    class Collection:
        def __init__(self):
            self.updated = []

        def get(self, **kwargs):
            assert kwargs["include"] == ["documents", "metadatas"]
            return {
                "ids": ["A", "B"],
                "documents": ["alpha", "beta"],
                "metadatas": [{"old": 1}, {"old": 2}],
            }

        def update(self, **kwargs):
            self.updated.append(kwargs)

    client = chroma_client.ChromaClient.__new__(chroma_client.ChromaClient)
    client.client = SimpleNamespace(get_max_batch_size=lambda: 100)
    client.collection = Collection()

    records = client.get_records(["A", "B"])
    client.update_metadatas(["A", "B"], [{"new": 1}, {"new": 2}])

    assert records == {
        "A": {"document": "alpha", "metadata": {"old": 1}},
        "B": {"document": "beta", "metadata": {"old": 2}},
    }
    assert client.collection.updated == [{
        "ids": ["A", "B"],
        "metadatas": [{"new": 1}, {"new": 2}],
    }]


def test_real_chroma_metadata_update_preserves_document_and_vector(tmp_path):
    raw_client = chroma_client.chromadb.PersistentClient(
        path=str(tmp_path / "chroma")
    )
    collection = raw_client.create_collection(
        "metadata_update",
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=["A"],
        documents=["unchanged document"],
        metadatas=[{"title": "Old"}],
        embeddings=[[0.25, 0.75]],
    )
    before = collection.get(
        ids=["A"],
        include=["documents", "metadatas", "embeddings"],
    )
    client = chroma_client.ChromaClient.__new__(chroma_client.ChromaClient)
    client.client = raw_client
    client.collection = collection

    client.update_metadatas(["A"], [{"title": "New"}])

    after = collection.get(
        ids=["A"],
        include=["documents", "metadatas", "embeddings"],
    )
    assert after["documents"] == before["documents"]
    assert after["embeddings"].tolist() == before["embeddings"].tolist()
    assert after["metadatas"] == [{"title": "New"}]


def test_item_metadata_merge_preserves_fulltext_chunk_fields(tmp_path):
    raw_client = chroma_client.chromadb.PersistentClient(
        path=str(tmp_path / "chroma-item-merge")
    )
    collection = raw_client.create_collection(
        "item_metadata_merge",
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=["ITEM#0", "ITEM#1"],
        documents=["first body passage", "second body passage"],
        metadatas=[
            {
                "parent_item_key": "ITEM",
                "has_fulltext": True,
                "chunk_index": 0,
                "title": "Old",
            },
            {
                "parent_item_key": "ITEM",
                "has_fulltext": True,
                "chunk_index": 1,
                "title": "Old",
            },
        ],
        embeddings=[[0.25, 0.75], [0.5, 0.5]],
    )
    before = collection.get(
        ids=["ITEM#0", "ITEM#1"],
        include=["documents", "metadatas", "embeddings"],
    )
    client = chroma_client.ChromaClient.__new__(chroma_client.ChromaClient)
    client.client = raw_client
    client.collection = collection

    updated = client.update_item_metadata("ITEM", {"title": "New"})

    after = collection.get(
        ids=["ITEM#0", "ITEM#1"],
        include=["documents", "metadatas", "embeddings"],
    )
    assert updated == 2
    assert after["documents"] == before["documents"]
    assert after["embeddings"].tolist() == before["embeddings"].tolist()
    assert [metadata["title"] for metadata in after["metadatas"]] == [
        "New",
        "New",
    ]
    assert all(metadata["has_fulltext"] is True for metadata in after["metadatas"])
    assert [metadata["chunk_index"] for metadata in after["metadatas"]] == [0, 1]


def test_distance_conversion_uses_cosine_distance():
    client = chroma_client.ChromaClient.__new__(chroma_client.ChromaClient)

    assert client.distance_to_similarity(0.4) == pytest.approx(0.6)


def test_non_cosine_collection_requires_confirmed_rebuild(monkeypatch):
    fake = _FakePersistentClient(metadata={"hnsw:space": "l2"})

    with pytest.raises(RuntimeError, match="not marked as a cosine collection"):
        _build_client(monkeypatch, fake)


def test_confirmed_rebuild_can_replace_non_cosine_collection(monkeypatch):
    fake = _FakePersistentClient(metadata={"hnsw:space": "l2"})

    client = _build_client(monkeypatch, fake, allow_embedding_mismatch=True)

    assert client._pending_embedding_mismatch is True
