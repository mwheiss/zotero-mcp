"""Regression coverage for selectively ported upstream behavior."""

from __future__ import annotations

import sqlite3
import subprocess
import threading

from pyzotero.zotero_errors import PreConditionFailedError

from zotero_mcp import client as zotero_client
from zotero_mcp import updater
from zotero_mcp.citation_import import csl_json_to_zotero
from zotero_mcp.html_metadata import extract_embedded_metadata
from zotero_mcp.local_db import LocalZoteroReader
from zotero_mcp.tools import _helpers, annotations, search, write
from zotero_mcp.utils import (
    _paginate,
    get_search_backend,
    item_display_date,
    item_display_title,
    use_sqlite_search,
)


def test_updater_never_offers_downgrade(monkeypatch):
    monkeypatch.setattr(updater, "get_current_version", lambda: "1.3.1")
    monkeypatch.setattr(updater, "get_latest_version", lambda: "0.11.0")
    result = updater.update_zotero_mcp(check_only=True)
    assert result["success"] is True
    assert result["needs_update"] is False
    assert "ahead" in result["message"]


def test_version_order_is_numeric():
    assert updater.is_newer_version("0.9.0", "0.10.0") is True
    assert updater.is_newer_version("1.3.1", "0.11.0") is False


def test_paginate_fetches_every_page_and_trims_output():
    values = list(range(230))

    def page(*, start, limit):
        return values[start : start + limit]

    assert _paginate(page) == values
    assert _paginate(page, max_items=125) == values[:125]


def test_paginate_supports_simple_test_double():
    assert _paginate(lambda: [1, 2, 3], max_items=2) == [1, 2]


def test_collection_descendants_are_cycle_safe():
    collections = [
        {"key": "ROOT", "data": {"parentCollection": "CHILD"}},
        {"key": "CHILD", "data": {"parentCollection": "ROOT"}},
    ]
    assert _helpers.collection_descendants(collections, "ROOT") == ["ROOT", "CHILD"]


def test_identifier_lock_normalizes_equivalent_isbns():
    assert _helpers.identifier_lock_key("isbn", "0-306-40615-2") == (
        _helpers.identifier_lock_key("isbn", "9780306406157")
    )


def test_identifier_lock_serializes_same_identifier():
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def first():
        with _helpers.identifier_lock("doi", "10.1234/same"):
            first_entered.set()
            release_first.wait(timeout=2)

    def second():
        first_entered.wait(timeout=2)
        with _helpers.identifier_lock("doi", "https://doi.org/10.1234/same"):
            second_entered.set()

    one = threading.Thread(target=first)
    two = threading.Thread(target=second)
    one.start()
    two.start()
    assert first_entered.wait(timeout=2)
    assert not second_entered.wait(timeout=0.05)
    release_first.set()
    one.join(timeout=2)
    two.join(timeout=2)
    assert second_entered.is_set()


def test_update_retries_stale_versions():
    class Zotero:
        attempts = 0

        def item(self, key):
            return {"key": key, "data": {"extra": "old"}}

        def update_item(self, item):
            self.attempts += 1
            assert item["data"]["extra"] == "new"
            if self.attempts == 1:
                raise PreConditionFailedError()
            return True

    client = Zotero()
    result = _helpers._update_item_with_version_retry(
        client,
        "ITEM0001",
        lambda item: item["data"].update(extra="new"),
    )
    assert result is True
    assert client.attempts == 2


def test_pdf_outline_subprocess_uses_tagged_payload(monkeypatch):
    class Process:
        returncode = 0
        stdout = None
        stderr = None

        def communicate(self, timeout):
            assert timeout == 30
            return ("library noise\n@@ZOTERO_MCP_TOC@@[[1, \"Intro\", 1]]", "")

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: Process())
    outcome = write._extract_pdf_toc("paper.pdf")
    assert outcome.status == "ok"
    assert outcome.toc == [[1, "Intro", 1]]


def test_embedded_publisher_metadata():
    metadata = extract_embedded_metadata(
        """
        <head>
          <meta name="citation_title" content="A useful paper">
          <meta name="citation_author" content="Doe, Jane">
          <meta name="citation_journal_title" content="Journal of Tests">
          <meta name="citation_doi" content="https://doi.org/10.1234/example">
          <meta name="citation_firstpage" content="10">
          <meta name="citation_lastpage" content="19">
        </head>
        """
    )
    assert metadata.title == "A useful paper"
    assert metadata.authors == [("Jane", "Doe")]
    assert metadata.doi == "10.1234/example"
    assert metadata.pages == "10-19"
    assert metadata.looks_like_article()


def test_crossref_shaped_csl_arrays_are_accepted():
    def template(item_type):
        return {
            "itemType": item_type,
            "title": "",
            "publicationTitle": "",
            "ISSN": "",
            "ISBN": "",
            "creators": [],
            "tags": [],
            "extra": "",
        }

    item = csl_json_to_zotero(
        {
            "type": "journal-article",
            "title": ["Array title"],
            "container-title": ["Array journal"],
            "ISSN": ["1234-5678", "8765-4321"],
        },
        template,
    )
    assert item["itemType"] == "journalArticle"
    assert item["title"] == "Array title"
    assert item["publicationTitle"] == "Array journal"
    assert item["ISSN"] == "1234-5678"


def test_type_specific_display_fields_and_note_titles():
    assert item_display_title({"itemType": "case", "caseName": "Example v Test"}) == "Example v Test"
    assert item_display_date({"itemType": "case", "dateDecided": "2024-05-01"}) == "2024-05-01"
    assert item_display_title({"itemType": "note", "note": "<p>Heading</p><p>Body</p>"}) == "Heading"


def test_update_routes_base_fields_to_type_specific_keys(monkeypatch):
    class Zotero:
        updated = None

        def item(self, key):
            return {
                "key": key,
                "version": 1,
                "data": {
                    "key": key,
                    "version": 1,
                    "itemType": "case",
                    "caseName": "Old name",
                    "dateDecided": "2020",
                    "creators": [],
                    "tags": [],
                    "collections": [],
                    "relations": {},
                },
            }

        def item_template(self, _item_type):
            return {"caseName": "", "dateDecided": ""}

        def update_item(self, item):
            self.updated = item
            return True

    client = Zotero()
    monkeypatch.setattr(
        _helpers, "_get_write_client", lambda _ctx: (client, client)
    )
    result = write.update_item(
        "CASE0001", title="New name", date="2026", ctx=object()
    )
    assert client.updated["data"]["caseName"] == "New name"
    assert client.updated["data"]["dateDecided"] == "2026"
    assert result.startswith("Successfully updated")


def test_get_notes_accepts_a_note_key(monkeypatch):
    class Zotero:
        def item(self, key):
            return {
                "key": key,
                "data": {
                    "itemType": "note",
                    "note": "<p>Heading</p><p>Complete body</p>",
                },
            }

    monkeypatch.setattr(
        "zotero_mcp.client.get_zotero_client", lambda: Zotero()
    )
    result = annotations.get_notes("NOTE0001", truncate=False, ctx=object())
    assert "Heading\n\nComplete body" in result


def test_batch_inputs_are_deduplicated():
    called = []
    results = write._run_batch(
        ["10.1234/A", "https://doi.org/10.1234/A"],
        _helpers._normalize_doi,
        lambda value: called.append(value) or "ok",
    )
    assert called == ["10.1234/A"]
    assert results[1].startswith("Repeated input")


def test_doi_batch_uses_one_crossref_request(monkeypatch):
    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "message": {
                    "items": [
                        {"DOI": "10.1234/a", "type": "journal-article", "title": ["A"]},
                        {"DOI": "10.1234/b", "type": "journal-article", "title": ["B"]},
                    ]
                }
            }

    calls = []

    def request(url, **kwargs):
        calls.append((url, kwargs))
        assert url == "https://api.crossref.org/works"
        return Response()

    class Zotero:
        created = 0

        def item_template(self, item_type):
            return {
                "itemType": item_type,
                "title": "",
                "creators": [],
                "date": "",
                "DOI": "",
                "url": "",
                "volume": "",
                "issue": "",
                "pages": "",
                "publisher": "",
                "ISSN": "",
                "publicationTitle": "",
                "abstractNote": "",
                "tags": [],
                "collections": [],
            }

        def create_items(self, _items):
            self.created += 1
            return {"success": {"0": f"ITEM{self.created:04d}"}}

    client = Zotero()
    monkeypatch.setattr(write.requests, "get", request)
    monkeypatch.setattr(_helpers, "_get_write_client", lambda _ctx: (client, client))
    monkeypatch.setattr(_helpers, "find_existing_items", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(_helpers, "ensure_collection_membership", lambda *_args, **_kwargs: [])
    result = write.add_by_doi(
        ["10.1234/a", "10.1234/b"], attach_mode="none", ctx=object()
    )
    assert len(calls) == 1
    assert calls[0][1]["params"]["rows"] == 2
    assert client.created == 2
    assert "succeeded: 2" in result


def test_failed_attachment_both_retries_and_verifies_bytes(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "zotero_mcp.webdav.is_webdav_configured", lambda: False
    )
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-test")

    class Zotero:
        uploaded = False

        def attachment_both(self, *_args, **_kwargs):
            return {"failure": [{"message": "rejected"}]}

        def item_template(self, *_args, **_kwargs):
            return {"title": "", "filename": "", "parentItem": ""}

        def create_items(self, _items):
            return {"success": {"0": "ATTACH01"}, "successVersions": {"0": 1}}

        def item(self, _key):
            return {
                "version": 1,
                "data": {
                    "filename": "paper.pdf",
                    "md5": "abc" if self.uploaded else "",
                },
            }

        def upload_attachments(self, _items):
            self.uploaded = True
            return {"success": [{"key": "ATTACH01"}]}

        def delete_item(self, _item):
            raise AssertionError("successful fallback must not be deleted")

    ok, _detail, key = _helpers._attach_and_verify(
        Zotero(), "paper.pdf", str(source), "PARENT01", object()
    )
    assert ok is True
    assert key == "ATTACH01"


def test_linked_attachment_is_copied_from_local_storage(monkeypatch, tmp_path):
    source = tmp_path / "linked.pdf"
    source.write_bytes(b"linked bytes")

    class Reader:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get_attachment_by_key(self, key):
            return {"zotero_path": str(source), "content_type": "application/pdf"}

        def _resolve_attachment_path(self, _key, _path):
            return source

    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setattr("zotero_mcp.local_db.LocalZoteroReader", Reader)
    destination = tmp_path / "download"
    result = zotero_client.download_attachment_file(
        "ATTACH01", destination, "copy.pdf", enable_webdav=False
    )
    assert result.source == "Local storage"
    assert result.path.read_bytes() == b"linked bytes"
    assert result.path != source


def _create_search_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE libraries (libraryID INTEGER PRIMARY KEY, type TEXT);
        CREATE TABLE groups (libraryID INTEGER, groupID INTEGER, name TEXT);
        CREATE TABLE itemTypes (itemTypeID INTEGER PRIMARY KEY, typeName TEXT);
        CREATE TABLE items (itemID INTEGER PRIMARY KEY, key TEXT, libraryID INTEGER,
                            itemTypeID INTEGER, dateAdded TEXT, dateModified TEXT);
        CREATE TABLE deletedItems (itemID INTEGER);
        CREATE TABLE fields (fieldID INTEGER PRIMARY KEY, fieldName TEXT);
        CREATE TABLE itemData (itemID INTEGER, fieldID INTEGER, valueID INTEGER);
        CREATE TABLE itemDataValues (valueID INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE creatorTypes (creatorTypeID INTEGER PRIMARY KEY, creatorType TEXT);
        CREATE TABLE creators (creatorID INTEGER PRIMARY KEY, firstName TEXT, lastName TEXT);
        CREATE TABLE itemCreators (itemID INTEGER, creatorID INTEGER,
                                   creatorTypeID INTEGER, orderIndex INTEGER);
        CREATE TABLE tags (tagID INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE itemTags (itemID INTEGER, tagID INTEGER);
        CREATE TABLE collections (collectionID INTEGER PRIMARY KEY, key TEXT,
                                  libraryID INTEGER, parentCollectionID INTEGER);
        CREATE TABLE collectionItems (itemID INTEGER, collectionID INTEGER);
        CREATE TABLE itemNotes (itemID INTEGER, parentItemID INTEGER, note TEXT);
        INSERT INTO libraries VALUES (1, 'user'), (2, 'group');
        INSERT INTO groups VALUES (2, 42, 'Research Group');
        INSERT INTO itemTypes VALUES (1, 'journalArticle');
        INSERT INTO fields VALUES (1, 'title'), (2, 'date');
        INSERT INTO items VALUES
          (1, 'USER0001', 1, 1, '2025-01-01', '2025-01-02'),
          (2, 'GROUP001', 2, 1, '2025-01-01', '2025-01-03');
        INSERT INTO itemDataValues VALUES
          (1, 'Alpha user paper'), (2, '2025-00-00 2025'),
          (3, 'Alpha group paper'), (4, '2026-00-00 2026');
        INSERT INTO itemData VALUES
          (1, 1, 1), (1, 2, 2), (2, 1, 3), (2, 2, 4);
        """
    )
    conn.commit()
    conn.close()


def test_sqlite_search_can_cover_all_libraries(tmp_path):
    db_path = tmp_path / "zotero.sqlite"
    _create_search_db(db_path)
    with LocalZoteroReader(db_path=str(db_path)) as reader:
        personal = reader.search_items_sql("Alpha", group_id=0)
        global_results = reader.search_items_sql("Alpha", group_id=None)
    assert [item["key"] for item in personal] == ["USER0001"]
    assert {item["key"] for item in global_results} == {"USER0001", "GROUP001"}
    assert {item["library"]["name"] for item in global_results} == {
        "My Library",
        "Research Group",
    }


def test_global_search_routes_only_through_sqlite(monkeypatch):
    class Reader:
        def search_items_sql(self, *_args, **kwargs):
            assert kwargs["group_id"] is None
            return [
                {
                    "key": "GROUP001",
                    "data": {
                        "itemType": "journalArticle",
                        "title": "Group result",
                        "date": "2026",
                        "creators": [],
                        "tags": [],
                    },
                    "library": {"id": 42, "type": "group", "name": "Research Group"},
                }
            ]

        def close(self):
            pass

    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setenv("ZOTERO_SEARCH_BACKEND", "sqlite")
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: object())
    monkeypatch.setattr("zotero_mcp.local_db.get_local_zotero_reader", lambda: Reader())
    result = search.search_items(
        "Group", search_all_libraries=True, ctx=object()
    )
    assert "Group result" in result
    assert "Research Group (groupID=42)" in result


def test_split_search_routing_is_default(monkeypatch):
    monkeypatch.delenv("ZOTERO_SEARCH_BACKEND", raising=False)
    assert get_search_backend() == "split"
    assert use_sqlite_search("keyword") is False
    assert use_sqlite_search("tag") is False
    assert use_sqlite_search("advanced") is True
    assert use_sqlite_search("keyword", search_all_libraries=True) is True


def test_search_routing_overrides(monkeypatch):
    monkeypatch.setenv("ZOTERO_SEARCH_BACKEND", "api")
    assert use_sqlite_search("advanced") is False
    assert use_sqlite_search("advanced", search_all_libraries=True) is False
    monkeypatch.setenv("ZOTERO_SEARCH_BACKEND", "sqlite")
    assert use_sqlite_search("keyword") is True
    assert use_sqlite_search("tag") is True
    assert use_sqlite_search("advanced") is True
