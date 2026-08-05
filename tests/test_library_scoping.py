from zotero_mcp.chroma_client import scoped_collection_name


def test_default_library_keeps_historical_collection(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)
    monkeypatch.setenv("ZOTERO_LIBRARY_TYPE", "user")

    assert scoped_collection_name("zotero_library", scope_identity="user:0") == "zotero_library"


def test_switched_library_gets_isolated_collection(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)

    assert scoped_collection_name("zotero_library", scope_identity="group:5910265") == "zotero_library__group_5910265"
