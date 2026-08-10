"""Tests for the strictly read-only semantic database health audit."""

import hashlib
import json
import pickle
import sqlite3
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from zotero_mcp import cli, db_health, local_db, semantic_search

_REAL_ACQUIRE_SHARED_UPDATE_LOCK = db_health._acquire_shared_update_lock


@pytest.fixture(autouse=True)
def _isolate_health_tests_from_live_update_lock(monkeypatch):
    monkeypatch.setattr(
        db_health,
        "_acquire_shared_update_lock",
        lambda: (None, True),
    )


def _create_fixture(
    tmp_path,
    *,
    bad_hash=False,
    queue_vector=True,
    rebuild_marker=None,
    save_rebuild_state=True,
):
    persist_directory = tmp_path / "chroma_db"
    persist_directory.mkdir()
    config_path = tmp_path / "config.json"
    semantic_config = {"collection_name": "zotero_library"}
    if rebuild_marker and save_rebuild_state:
        semantic_config["library_states"] = {
            "user:0": {
                "rebuild_in_progress": {"marker": rebuild_marker}
            }
        }
    config_path.write_text(json.dumps({"semantic_search": semantic_config}))
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
        if rebuild_marker and record_id == "NEW#0":
            content_hash = rebuild_marker
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


def test_health_audit_accepts_hashes_invalidated_by_active_rebuild(tmp_path):
    marker = "rebuild-pending:test-run"
    config_path, persist_directory = _create_fixture(
        tmp_path,
        rebuild_marker=marker,
    )

    report = db_health.audit_semantic_database(
        config_path,
        persist_directory=persist_directory,
        compare_zotero=False,
        full_integrity=False,
    )

    assert report.healthy is True
    assert report.metrics["pending_rebuild_passages"] == 1
    assert _finding(report, "content_hashes").level == "warning"
    assert "active rebuild" in _finding(report, "content_hashes").detail


def test_health_audit_rejects_orphaned_rebuild_hash_marker(tmp_path):
    marker = "rebuild-pending:orphaned"
    config_path, persist_directory = _create_fixture(
        tmp_path,
        rebuild_marker=marker,
        save_rebuild_state=False,
    )

    report = db_health.audit_semantic_database(
        config_path,
        persist_directory=persist_directory,
        compare_zotero=False,
        full_integrity=False,
    )

    assert report.healthy is False
    assert _finding(report, "content_hashes").level == "error"
    assert "without a matching active rebuild" in _finding(
        report, "content_hashes"
    ).detail


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


def test_health_audit_reports_genuinely_unowned_segments(tmp_path):
    config_path, persist_directory = _create_fixture(tmp_path)
    connection = sqlite3.connect(persist_directory / "chroma.sqlite3")
    connection.execute(
        "INSERT INTO segments VALUES (?, ?, ?, ?)",
        (
            "77777777-7777-7777-7777-777777777777",
            "vector",
            "VECTOR",
            "88888888-8888-8888-8888-888888888888",
        ),
    )
    connection.commit()
    connection.close()

    report = db_health.audit_semantic_database(
        config_path,
        persist_directory=persist_directory,
        compare_zotero=False,
        full_integrity=False,
    )

    assert report.healthy is False
    assert _finding(report, "foreign_keys").level == "error"


def test_zotero_coverage_keeps_preprints_and_journal_versions(monkeypatch):
    items = [
        SimpleNamespace(key="PREPRINT", item_type="preprint"),
        SimpleNamespace(key="JOURNAL", item_type="journalArticle"),
    ]

    class Reader:
        db_path = "/tmp/zotero.sqlite"

        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def get_items_with_text(self, **_kwargs):
            return items

    monkeypatch.setattr(local_db, "LocalZoteroReader", Reader)

    keys, _path = db_health._read_zotero_keys(None, None)

    assert keys == {"PREPRINT", "JOURNAL"}


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


def test_health_lock_is_created_and_blocks_first_updater(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    lock_file, available = _REAL_ACQUIRE_SHARED_UPDATE_LOCK()
    try:
        assert available is True
        assert lock_file is not None
        lock_path = tmp_path / ".config" / "zotero-mcp" / "update.lock"
        assert lock_path.exists()
        with semantic_search._acquire_update_lock(lock_path) as acquired:
            assert acquired is False
    finally:
        if lock_file is not None:
            lock_file.close()


def test_health_audit_holds_lifecycle_lock(monkeypatch, tmp_path):
    config_path, persist_directory = _create_fixture(tmp_path)
    entered_audit = threading.Event()
    release_audit = threading.Event()
    exclusive_acquired = threading.Event()
    original_table_names = db_health._table_names

    def blocking_table_names(connection):
        entered_audit.set()
        assert release_audit.wait(timeout=5)
        return original_table_names(connection)

    monkeypatch.setattr(db_health, "_table_names", blocking_table_names)
    audit = threading.Thread(
        target=db_health.audit_semantic_database,
        args=(config_path,),
        kwargs={
            "persist_directory": persist_directory,
            "compare_zotero": False,
        },
    )

    def acquire_exclusive():
        with db_health.index_lifecycle_lock(
            persist_directory,
            exclusive=True,
        ):
            exclusive_acquired.set()

    writer = threading.Thread(target=acquire_exclusive)
    audit.start()
    assert entered_audit.wait(timeout=2)
    writer.start()
    assert not exclusive_acquired.wait(timeout=0.1)
    release_audit.set()
    audit.join(timeout=2)
    writer.join(timeout=2)

    assert not audit.is_alive()
    assert not writer.is_alive()
    assert exclusive_acquired.is_set()


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
