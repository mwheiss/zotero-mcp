"""Tests for the strictly read-only semantic database health audit."""

import hashlib
import json
import pickle
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from zotero_mcp import cli, db_health


@pytest.fixture(autouse=True)
def _isolate_health_tests_from_live_update_lock(monkeypatch):
    monkeypatch.setattr(
        db_health,
        "_acquire_shared_update_lock",
        lambda: (None, True),
    )


def _create_fixture(tmp_path, *, bad_hash=False, queue_vector=True):
    persist_directory = tmp_path / "chroma_db"
    persist_directory.mkdir()
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "semantic_search": {
                    "collection_name": "zotero_library",
                }
            }
        )
    )
    database_path = persist_directory / "chroma.sqlite3"
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """
        CREATE TABLE collections (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            dimension INTEGER
        );
        CREATE TABLE segments (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            scope TEXT NOT NULL,
            collection TEXT REFERENCES collection(id) NOT NULL
        );
        CREATE TABLE embeddings (
            id INTEGER PRIMARY KEY,
            segment_id TEXT NOT NULL,
            embedding_id TEXT NOT NULL,
            seq_id BLOB NOT NULL,
            UNIQUE(segment_id, embedding_id)
        );
        CREATE TABLE embedding_metadata (
            id INTEGER REFERENCES embeddings(id),
            key TEXT NOT NULL,
            string_value TEXT,
            int_value INTEGER,
            float_value REAL,
            bool_value INTEGER,
            PRIMARY KEY(id, key)
        );
        CREATE TABLE embeddings_queue (
            seq_id INTEGER PRIMARY KEY,
            operation INTEGER NOT NULL,
            topic TEXT NOT NULL,
            id TEXT NOT NULL,
            vector BLOB,
            metadata TEXT
        );
        CREATE TABLE max_seq_id (
            segment_id TEXT PRIMARY KEY,
            seq_id INTEGER
        );
        """
    )
    collection_id = "11111111-1111-1111-1111-111111111111"
    metadata_segment = "22222222-2222-2222-2222-222222222222"
    vector_segment = "33333333-3333-3333-3333-333333333333"
    connection.execute(
        "INSERT INTO collections VALUES (?, ?, ?)",
        (collection_id, "zotero_library", 3),
    )
    connection.executemany(
        "INSERT INTO segments VALUES (?, ?, ?, ?)",
        [
            (metadata_segment, "metadata", "METADATA", collection_id),
            (vector_segment, "vector", "VECTOR", collection_id),
        ],
    )
    documents = {"OLD#0": "old document", "NEW#0": "new document"}
    for row_id, (record_id, document) in enumerate(documents.items(), 1):
        item_key = record_id.split("#", 1)[0]
        connection.execute(
            "INSERT INTO embeddings VALUES (?, ?, ?, ?)",
            (row_id, metadata_segment, record_id, b"1"),
        )
        content_hash = hashlib.sha256(document.encode()).hexdigest()
        if bad_hash and record_id == "NEW#0":
            content_hash = "not-the-document-hash"
        connection.executemany(
            "INSERT INTO embedding_metadata VALUES (?, ?, ?, ?, ?, ?)",
            [
                (row_id, "item_key", item_key, None, None, None),
                (row_id, "chunk_index", None, 0, None, None),
                (row_id, "n_chunks", None, 1, None, None),
                (row_id, "chroma:document", document, None, None, None),
                (
                    row_id,
                    "embedding_content_sha256",
                    content_hash,
                    None,
                    None,
                    None,
                ),
            ],
        )
    connection.executemany(
        "INSERT INTO max_seq_id VALUES (?, ?)",
        [(metadata_segment, 11), (vector_segment, 10)],
    )
    connection.execute(
        "INSERT INTO embeddings_queue VALUES (?, ?, ?, ?, ?, ?)",
        (
            11,
            2,
            f"persistent://default/default/{collection_id}",
            "NEW#0",
            b"vector" if queue_vector else None,
            "{}",
        ),
    )
    connection.commit()
    connection.close()

    vector_directory = persist_directory / vector_segment
    vector_directory.mkdir()
    (vector_directory / "index_metadata.pickle").write_bytes(
        pickle.dumps(
            {
                "id_to_label": {"OLD#0": 1},
                "label_to_id": {1: "OLD#0"},
            }
        )
    )
    return config_path, persist_directory


def _finding(report, check):
    return next(finding for finding in report.findings if finding.check == check)


def test_health_audit_replays_queue_without_modifying_files(tmp_path):
    config_path, persist_directory = _create_fixture(tmp_path)
    tracked = [
        persist_directory / "chroma.sqlite3",
        persist_directory
        / "33333333-3333-3333-3333-333333333333"
        / "index_metadata.pickle",
    ]
    before = [(path.stat().st_mtime_ns, path.read_bytes()) for path in tracked]

    report = db_health.audit_semantic_database(
        config_path,
        persist_directory=persist_directory,
        compare_zotero=False,
    )

    assert report.healthy is True
    assert report.metrics["passages"] == 2
    assert report.metrics["items"] == 2
    assert report.metrics["hnsw_replayed_operations"] == 1
    assert _finding(report, "vector_replay").level == "ok"
    assert _finding(report, "foreign_keys").level == "ok"
    after = [(path.stat().st_mtime_ns, path.read_bytes()) for path in tracked]
    assert after == before


def test_health_audit_detects_hash_mismatch(tmp_path):
    config_path, persist_directory = _create_fixture(tmp_path, bad_hash=True)

    report = db_health.audit_semantic_database(
        config_path,
        persist_directory=persist_directory,
        compare_zotero=False,
        full_integrity=False,
    )

    assert report.healthy is False
    assert _finding(report, "content_hashes").level == "error"


def test_health_audit_detects_missing_replay_vector(tmp_path):
    config_path, persist_directory = _create_fixture(
        tmp_path,
        queue_vector=False,
    )

    report = db_health.audit_semantic_database(
        config_path,
        persist_directory=persist_directory,
        compare_zotero=False,
        full_integrity=False,
    )

    assert report.healthy is False
    assert _finding(report, "vector_replay").level == "error"
    assert "1 missing" in _finding(report, "vector_replay").detail


def test_health_audit_does_not_call_other_collection_segments_orphans(tmp_path):
    config_path, persist_directory = _create_fixture(tmp_path)
    database_path = persist_directory / "chroma.sqlite3"
    connection = sqlite3.connect(database_path)
    other_collection = "44444444-4444-4444-4444-444444444444"
    other_segment = "55555555-5555-5555-5555-555555555555"
    connection.execute(
        "INSERT INTO collections VALUES (?, ?, ?)",
        (other_collection, "other_collection", 3),
    )
    connection.execute(
        "INSERT INTO segments VALUES (?, ?, ?, ?)",
        (other_segment, "vector", "VECTOR", other_collection),
    )
    connection.commit()
    connection.close()
    (persist_directory / other_segment).mkdir()

    report = db_health.audit_semantic_database(
        config_path,
        persist_directory=persist_directory,
        compare_zotero=False,
        full_integrity=False,
    )

    assert report.metrics["orphan_segment_directories"] == 0
    assert _finding(report, "orphan_segments").level == "ok"


def test_zotero_key_deduplication_matches_update_type_priority():
    unknown = SimpleNamespace(
        key="UNKNOWN",
        doi="10.1/shared",
        title="Shared",
        item_type="report",
    )
    preprint = SimpleNamespace(
        key="PREPRINT",
        doi="10.1/shared",
        title="Shared",
        item_type="preprint",
    )
    journal = SimpleNamespace(
        key="JOURNAL",
        doi="10.1/shared",
        title="Shared",
        item_type="journalArticle",
    )

    result = db_health._deduplicate_index_items([unknown, preprint, journal])

    assert [item.key for item in result] == ["UNKNOWN", "JOURNAL"]


def test_health_audit_refuses_snapshot_during_update(monkeypatch, tmp_path):
    config_path, persist_directory = _create_fixture(tmp_path)
    monkeypatch.setattr(
        db_health,
        "_acquire_shared_update_lock",
        lambda: (None, False),
    )

    report = db_health.audit_semantic_database(
        config_path,
        persist_directory=persist_directory,
        compare_zotero=False,
    )

    assert report.healthy is False
    assert _finding(report, "update_lock").level == "error"


def test_db_health_cli_returns_report_status(monkeypatch, capsys):
    report = db_health.DatabaseHealthReport("/tmp/chroma.sqlite3", "library")
    report.add("ok", "sqlite", "fine")
    monkeypatch.setattr(cli, "setup_zotero_environment", lambda: None)
    monkeypatch.setattr(db_health, "audit_semantic_database", lambda *_a, **_kw: report)
    monkeypatch.setattr(sys, "argv", ["zotero-mcp", "db-health", "--quick"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    assert "Status: HEALTHY" in capsys.readouterr().out


def test_db_health_cli_returns_nonzero_for_errors(monkeypatch, capsys):
    report = db_health.DatabaseHealthReport("/tmp/chroma.sqlite3", "library")
    report.add("error", "sqlite", "broken")
    monkeypatch.setattr(cli, "setup_zotero_environment", lambda: None)
    monkeypatch.setattr(db_health, "audit_semantic_database", lambda *_a, **_kw: report)
    monkeypatch.setattr(sys, "argv", ["zotero-mcp", "db-health", "--json"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "unhealthy"
