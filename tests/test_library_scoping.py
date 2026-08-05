import chromadb
from chromadb.config import Settings

from zotero_mcp.chroma_client import (
    ChromaClient,
    _existing_collection_owner,
    _NoEmbeddingFunction,
    scoped_collection_name,
)


def test_default_library_keeps_historical_collection(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)
    monkeypatch.setenv("ZOTERO_LIBRARY_TYPE", "user")

    assert scoped_collection_name("zotero_library", scope_identity="user:0") == "zotero_library"


def test_switched_library_gets_isolated_collection(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)

    assert scoped_collection_name("zotero_library", scope_identity="group:5910265") == "zotero_library__group_5910265"


def test_persisted_legacy_owner_survives_default_library_change(monkeypatch):
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "5910265")
    monkeypatch.setenv("ZOTERO_LIBRARY_TYPE", "group")

    assert scoped_collection_name(
        "zotero_library",
        scope_identity="user:0",
        legacy_owner_identity="user:0",
    ) == "zotero_library"
    assert scoped_collection_name(
        "zotero_library",
        scope_identity="group:5910265",
        legacy_owner_identity="user:0",
    ) == "zotero_library__group_5910265"


def test_new_collection_metadata_records_library_owner():
    client = object.__new__(ChromaClient)
    client.embedding_identity = "openai:test"
    client.library_identity = "group:5910265"

    assert client._new_collection_metadata()["zotero_mcp_library_identity"] == (
        "group:5910265"
    )


def test_claiming_legacy_owner_preserves_cosine_configuration(tmp_path):
    persist_directory = str(tmp_path / "chroma")
    raw_client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )
    raw_client.get_or_create_collection(
        "zotero_library",
        embedding_function=_NoEmbeddingFunction(),
        metadata={"hnsw:space": "cosine", "marker": "preserved"},
    )

    owner = _existing_collection_owner(
        "zotero_library",
        persist_directory,
        claim_identity="user:0",
    )
    collection = raw_client.get_collection(
        "zotero_library",
        embedding_function=_NoEmbeddingFunction(),
    )

    assert owner == "user:0"
    assert collection.metadata == {
        "marker": "preserved",
        "zotero_mcp_library_identity": "user:0",
    }
    assert collection.configuration_json["hnsw"]["space"] == "cosine"
