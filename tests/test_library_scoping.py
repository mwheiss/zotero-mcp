from zotero_mcp.chroma_client import ChromaClient, scoped_collection_name


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
