"""Tests for the public Zotero library identity contract."""

from conftest import DummyContext

from zotero_mcp import client
from zotero_mcp.tools import retrieval


class _LocalReader:
    def get_libraries(self):
        return [
            {"type": "user", "libraryID": 1, "itemCount": 42},
            {
                "type": "group",
                "libraryID": 2,
                "groupID": 1234,
                "groupName": "Research Group",
                "groupDescription": "",
                "itemCount": 7,
            },
            {"type": "feed", "libraryID": 9, "feedName": "Updates", "itemCount": 3},
        ]

    def close(self):
        return None


def test_local_library_listing_uses_public_api_identities(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)
    monkeypatch.setattr("zotero_mcp.local_db.LocalZoteroReader", _LocalReader)

    result = retrieval.list_libraries(ctx=DummyContext())

    assert "**Active library:** user:0" in result
    assert "`library_id=0`, `library_type=user`" in result
    assert "`library_id=1234`, `library_type=group`" in result
    assert "`library_id=9`, `library_type=feed`" in result
    assert "libraryID=1" not in result


def test_local_default_identity_ignores_cloud_user_id(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setenv("ZOTERO_LIBRARY_TYPE", "user")
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "20765677")

    assert client.get_default_library() == {
        "library_id": "0",
        "library_type": "user",
    }


def test_local_switch_rejects_sqlite_user_library_id(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")

    error = retrieval.validate_library_switch("1", "user")

    assert error is not None
    assert "library_id='0'" in error
    assert "SQLite-internal" in error


def test_local_switch_accepts_api_user_library_id(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setattr("zotero_mcp.local_db.LocalZoteroReader", _LocalReader)

    assert retrieval.validate_library_switch("0", "user") is None
