"""Regression tests for non-destructive embedding-model mismatches."""

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
