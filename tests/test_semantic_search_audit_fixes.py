"""Regression tests for semantic/search correctness audit findings."""

from __future__ import annotations

import copy
import json
import sys
from types import SimpleNamespace

import pytest

from zotero_mcp import cli, db_health, search_semantics, semantic_search
from zotero_mcp.local_db import LocalZoteroReader
from zotero_mcp.tool_contract import classify_result
from zotero_mcp.tools import search as search_tools


class _CollectionLookupConnection:
    def __init__(self):
        self.final_query_executed = False

    def execute(self, query, params=()):
        if "SELECT collectionID FROM collections WHERE key" in query:
            return _Rows([(1,)] if params[0] == "VALID" else [])
        if "SELECT collectionID FROM collections WHERE parentCollectionID" in query:
            return _Rows([])
        self.final_query_executed = True
        return _Rows([])


class _Rows(list):
    def fetchone(self):
        return self[0] if self else None

    def fetchall(self):
        return self


@pytest.mark.parametrize(
    "configured",
    [["MISSING"], ["VALID", "MISSING"], "VALID", ["VALID", ""]],
)
def test_collection_filter_fails_closed_for_invalid_configuration(configured):
    reader = object.__new__(LocalZoteroReader)
    connection = _CollectionLookupConnection()
    reader._connection = connection

    with pytest.raises(ValueError, match="collection"):
        reader.get_items_with_text(collection_keys=configured, library_id=1)

    assert connection.final_query_executed is False


def test_sqlite_advanced_search_uses_multipart_year_and_excludes_fieldless_negation():
    reader = object.__new__(LocalZoteroReader)
    items = [
        {
            "key": "DATED",
            "data": {
                "title": "Dated paper",
                "date": "October 1, 2016",
                "_dateISO": "2016-10-01",
                "creators": [{"firstName": "Jane", "lastName": "Doe"}],
                "tags": [],
                "collections": [],
            },
        },
        {
            "key": "FIELDLESS",
            "data": {
                "title": "No creator",
                "date": "2020",
                "creators": [],
                "tags": [],
                "collections": [],
            },
        },
    ]
    reader._metadata_search_items = lambda _group_id: copy.deepcopy(items)

    year = reader.advanced_search_sql(
        [{"field": "year", "operation": "is", "value": "2016"}]
    )
    negative_creator = reader.advanced_search_sql(
        [
            {
                "field": "creator",
                "operation": "doesNotContain",
                "value": "Smith",
            }
        ]
    )

    assert [item["key"] for item in year] == ["DATED"]
    assert [item["key"] for item in negative_creator] == ["DATED"]


def test_missing_scalar_fields_keep_upstream_negated_semantics():
    condition = {"field": "DOI", "operation": "isNot", "value": "10.1/x"}
    reader = object.__new__(LocalZoteroReader)
    reader._metadata_search_items = lambda _group_id: [
        {"key": "NO_DOI", "data": {"title": "No DOI", "DOI": ""}}
    ]
    assert [item["key"] for item in reader.advanced_search_sql([condition])] == [
        "NO_DOI"
    ]

    # Truly multivalued fields (creators/tags) use an empty candidate set and
    # therefore satisfy no condition, including negation.
    assert not search_semantics.matches([], "10.1/x", "isNot")


def test_sqlite_keyword_search_defers_unsupported_filter_syntax():
    reader = object.__new__(LocalZoteroReader)
    reader._metadata_search_items = lambda _group_id: pytest.fail(
        "unsupported syntax must be rejected before scanning"
    )

    assert (
        reader.search_items_sql("x", item_type="book || journalArticle")
        is None
    )
    assert reader.search_items_sql("x", tag=["review*"]) is None


class _TimeoutReader:
    def __init__(self):
        self.extract_calls = 0
        self.last_extraction_details = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def resolve_library_id(self, *_args):
        return 1

    def get_all_item_keys(self, _library_id):
        return {f"ITEM{i:04d}" for i in range(7)}

    def get_items_with_text(self, **_kwargs):
        return [
            SimpleNamespace(
                item_id=i,
                key=f"ITEM{i:04d}",
                item_type="journalArticle",
                title=f"Paper {i}",
                abstract="",
                extra="",
                doi=None,
                notes=None,
                creators="Doe, Jane",
                date_added="2026-01-01",
                date_modified="2026-01-01",
                fulltext=None,
                fulltext_source=None,
                fulltext_selection_priority=None,
            )
            for i in range(7)
        ]

    def get_fulltext_meta_for_item(self, item_id, *_args):
        return [[f"ATT{item_id:04d}", "storage:paper.pdf", "application/pdf"]]

    def get_attachment_signature(self, item_id, *_args):
        return f"signature-{item_id}"

    def extract_fulltext_for_item(self, *_args):
        self.extract_calls += 1
        return "__EXTRACTION_TIMEOUT__", "timeout"


def test_pdf_timeout_breaker_defers_instead_of_emitting_metadata_only(monkeypatch):
    reader = _TimeoutReader()
    search = object.__new__(semantic_search.ZoteroSemanticSearch)
    search.library = {"library_id": "0", "library_type": "user"}
    search.library_identity = "user:0"
    search._uses_legacy_state = True
    search.db_path = None
    search.config_path = None
    search._last_api_attachment_keys_by_parent = None
    monkeypatch.setattr(
        semantic_search, "LocalZoteroReader", lambda **_kwargs: reader
    )
    monkeypatch.setattr(search, "_get_items_from_api", lambda: [])

    result = search._get_items_from_local_db(extract_fulltext=True)

    assert result == []
    assert reader.extract_calls == 5
    assert search._last_scan_extraction_complete is False


def test_single_pdf_timeout_also_marks_scan_incomplete(monkeypatch):
    reader = _TimeoutReader()
    original_items = reader.get_items_with_text
    reader.get_items_with_text = lambda **kwargs: original_items(**kwargs)[:1]
    search = object.__new__(semantic_search.ZoteroSemanticSearch)
    search.library = {"library_id": "0", "library_type": "user"}
    search.library_identity = "user:0"
    search._uses_legacy_state = True
    search.db_path = None
    search.config_path = None
    search._last_api_attachment_keys_by_parent = None
    monkeypatch.setattr(
        semantic_search, "LocalZoteroReader", lambda **_kwargs: reader
    )
    monkeypatch.setattr(search, "_get_items_from_api", lambda: [])

    assert search._get_items_from_local_db(extract_fulltext=True) == []
    assert reader.extract_calls == 1
    assert search._last_scan_extraction_complete is False


def _minimal_update_search(config_path):
    search = object.__new__(semantic_search.ZoteroSemanticSearch)
    search.library = {"library_id": "0", "library_type": "user"}
    search.library_identity = "user:0"
    search._uses_legacy_state = True
    search.config_path = str(config_path)
    search.db_path = None
    search.update_config = {"auto_update": False, "update_frequency": "manual"}
    search._chunking_config = {"enabled": False}
    search.chroma_client = SimpleNamespace(
        get_collection_info=lambda: {"count": 0},
        get_all_ids=lambda: set(),
        set_embedding_cancel_event=lambda _event: None,
    )
    search.zotero_client = SimpleNamespace(last_modified_version=lambda: 12)
    return search


def test_partial_extraction_scan_does_not_advance_watermark(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "semantic_search": {
                    "last_sync_version": 10,
                    "update_config": {
                        "auto_update": False,
                        "update_frequency": "manual",
                    },
                }
            }
        )
    )
    search = _minimal_update_search(config_path)

    def partial_scan(**_kwargs):
        search._last_scan_extraction_complete = False
        search._last_scan_indexable_keys = set()
        return []

    monkeypatch.setattr(search, "_get_items_from_source", partial_scan)
    stats = search.update_database(fulltext=True)

    assert stats["errors"] == 0
    saved = json.loads(config_path.read_text())
    assert saved["semantic_search"]["last_sync_version"] == 10


def test_invalid_falsy_collection_configuration_preserves_index_and_watermark(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "semantic_search": {
                    "collection_keys": "",
                    "last_sync_version": 10,
                    "update_config": {
                        "auto_update": False,
                        "update_frequency": "manual",
                    },
                }
            }
        )
    )
    search = _minimal_update_search(config_path)
    monkeypatch.setattr(
        search,
        "_get_items_from_source",
        lambda **_kwargs: pytest.fail("invalid config must fail before scanning"),
    )

    stats = search.update_database(fulltext=False)

    assert stats["errors"] == 1
    assert "collection_keys" in stats["error"]
    assert json.loads(config_path.read_text())["semantic_search"][
        "last_sync_version"
    ] == 10


@pytest.mark.parametrize(
    ("text", "tool_name", "expected"),
    [
        (
            "No semantically similar items found for query: 'x'",
            "zotero_semantic_search",
            "empty",
        ),
        (
            "# Semantic results\n\n> **Partial coverage:** group:7 failed",
            "zotero_semantic_search",
            "partial",
        ),
        (
            "# Advanced Search Results\n\n"
            "> **Note:** Search stopped after 20s having examined 100 items; "
            "results are partial.",
            "zotero_advanced_search",
            "partial",
        ),
    ],
)
def test_search_result_contract_reports_empty_and_partial_truthfully(
    text, tool_name, expected
):
    assert classify_result(text, tool_name=tool_name)["status"] == expected


def test_tag_search_states_when_no_collection_scope_was_applied(monkeypatch):
    class Zotero:
        def add_parameters(self, **_kwargs):
            pass

        def items(self):
            return [
                {
                    "key": "ITEM0001",
                    "data": {
                        "itemType": "journalArticle",
                        "title": "Tagged paper",
                        "creators": [],
                        "tags": [{"tag": "review"}],
                    },
                }
            ]

    monkeypatch.setattr(search_tools._client, "get_zotero_client", lambda: Zotero())
    monkeypatch.setattr(search_tools._utils, "use_sqlite_search", lambda *_a, **_kw: False)
    result = search_tools.search_by_tag(tag=["review"], ctx=object())

    assert "entire active library" in result
    assert "no collection scope applied" in result


def test_db_inspect_filter_scans_later_pages(monkeypatch, capsys):
    class Collection:
        calls = []

        def get(self, *, limit, offset, include):
            self.calls.append((limit, offset, tuple(include)))
            count = min(limit, max(0, 501 - offset))
            ids = [f"id-{offset + index}" for index in range(count)]
            metadata = [
                {
                    "title": (
                        "needle in final record"
                        if offset + index == 500
                        else "ordinary"
                    ),
                    "creators": "",
                }
                for index in range(count)
            ]
            return {"ids": ids, "metadatas": metadata}

    collection = Collection()
    client = SimpleNamespace(
        collection=collection,
        get_collection_info=lambda: {"count": 501},
    )
    monkeypatch.setattr(cli, "setup_zotero_environment", lambda: None)
    monkeypatch.setattr(
        semantic_search,
        "create_semantic_search",
        lambda _path: SimpleNamespace(chroma_client=client),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["zotero-mcp", "db-inspect", "--filter", "needle", "--limit", "1"],
    )

    cli.main()

    assert "needle in final record" in capsys.readouterr().out
    assert [call[1] for call in collection.calls] == [0, 500]


def test_db_health_cli_passes_effective_library(monkeypatch):
    report = db_health.DatabaseHealthReport("/tmp/chroma.sqlite3", "library")
    captured = {}
    monkeypatch.setattr(cli, "setup_zotero_environment", lambda: None)
    monkeypatch.setattr(
        "zotero_mcp.client.get_current_library",
        lambda: {"library_id": "42", "library_type": "group"},
    )

    def audit(*_args, **kwargs):
        captured.update(kwargs)
        return report

    monkeypatch.setattr(db_health, "audit_semantic_database", audit)
    monkeypatch.setattr(sys, "argv", ["zotero-mcp", "db-health", "--quick"])
    with pytest.raises(SystemExit):
        cli.main()

    assert captured["library"] == {
        "library_id": "42",
        "library_type": "group",
    }
