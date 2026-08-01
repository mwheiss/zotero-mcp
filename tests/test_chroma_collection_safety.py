"""Regression tests for non-destructive embedding-model mismatches."""

import json
from types import SimpleNamespace

import pytest

from zotero_mcp import chroma_client


class _FakePersistentClient:
    def __init__(self, *, conflict=False, metadata=None):
        self.conflict = conflict
        self.collection = SimpleNamespace(metadata=metadata or {})
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


@pytest.mark.parametrize(
    ("metric", "distance", "expected"),
    [
        ("l2", 0.4, 0.8),
        ("cosine", 0.4, 0.6),
        ("ip", 0.4, 0.6),
    ],
)
def test_distance_conversion_respects_collection_metric(
    metric,
    distance,
    expected,
):
    client = chroma_client.ChromaClient.__new__(chroma_client.ChromaClient)
    client.collection = SimpleNamespace(metadata={"hnsw:space": metric})

    assert client.distance_to_similarity(distance) == pytest.approx(expected)


def test_metric_detection_uses_raw_config_without_loading_embedding_function():
    class Collection:
        metadata = {}
        _model = SimpleNamespace(
            configuration_json=json.dumps({"hnsw": {"space": "cosine"}})
        )

        @property
        def configuration(self):
            raise AssertionError("must not reconstruct the embedding function")

    client = chroma_client.ChromaClient.__new__(chroma_client.ChromaClient)
    client.collection = Collection()

    assert client.distance_metric == "cosine"
