"""Read-only health checks for the persisted semantic search database."""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ._file_lock import acquire_file_lock, release_file_lock
from .chroma_client import index_lifecycle_lock, scoped_collection_name


@dataclass
class HealthFinding:
    level: str
    check: str
    detail: str


@dataclass
class DatabaseHealthReport:
    database_path: str
    collection_name: str
    findings: list[HealthFinding] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def add(self, level: str, check: str, detail: str) -> None:
        self.findings.append(HealthFinding(level, check, detail))

    @property
    def healthy(self) -> bool:
        return not any(finding.level == "error" for finding in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "healthy" if self.healthy else "unhealthy",
            "database_path": self.database_path,
            "collection_name": self.collection_name,
            "metrics": self.metrics,
            "findings": [asdict(finding) for finding in self.findings],
        }


class _PersistentDataView:
    """Safe stand-in for older Chroma PersistentData pickles."""


class _SafePersistentDataUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (
            module
            == "chromadb.segment.impl.vector.local_persistent_hnsw"
            and name == "PersistentData"
        ):
            return _PersistentDataView
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def _read_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    with config_path.open(encoding="utf-8") as config_file:
        return json.load(config_file)


def _read_only_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"file:{path}?mode=ro",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    return connection


def _acquire_shared_update_lock() -> tuple[Any, bool]:
    """Hold a shared flock so an update cannot mutate the audited snapshot."""
    lock_path = Path.home() / ".config" / "zotero-mcp" / "update.lock"
    lock_file = acquire_file_lock(
        lock_path,
        exclusive=False,
        blocking=False,
    )
    return lock_file, lock_file is not None


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }


def _directory_size(path: Path) -> int:
    return sum(
        entry.stat().st_size
        for entry in path.rglob("*")
        if entry.is_file()
    )


def _load_persisted_ids(metadata_path: Path) -> set[str]:
    with metadata_path.open("rb") as metadata_file:
        data = _SafePersistentDataUnpickler(metadata_file).load()
    id_to_label = (
        data.get("id_to_label")
        if isinstance(data, dict)
        else getattr(data, "id_to_label", None)
    )
    if not isinstance(id_to_label, dict):
        raise ValueError("HNSW metadata has no id_to_label mapping")
    return set(id_to_label)


def _deduplicate_index_items(items: list[Any]) -> list[Any]:
    """Mirror the update path's preprint/journal duplicate preference."""

    def normalized(value: str | None) -> str | None:
        return "".join(value.lower().split()) if value else None

    best: dict[tuple[str, str], Any] = {}
    type_priority = {"journalArticle": 2, "preprint": 1}
    for item in items:
        keys = []
        if doi := normalized(getattr(item, "doi", None)):
            keys.append(("doi", doi))
        if title := normalized(getattr(item, "title", None)):
            keys.append(("title", title))
        for key in keys:
            current = best.get(key)
            if current is None or type_priority.get(
                getattr(item, "item_type", None), 0
            ) > type_priority.get(getattr(current, "item_type", None), 0):
                best[key] = item

    result = []
    for item in items:
        if getattr(item, "item_type", None) != "preprint":
            result.append(item)
            continue
        keys = []
        if doi := normalized(getattr(item, "doi", None)):
            keys.append(("doi", doi))
        if title := normalized(getattr(item, "title", None)):
            keys.append(("title", title))
        if any(
            (winner := best.get(key)) is not None
            and winner is not item
            and getattr(winner, "item_type", None) == "journalArticle"
            for key in keys
        ):
            continue
        result.append(item)
    return result


def _read_zotero_keys(
    db_path: str | None,
    collection_keys: list[str] | None,
    library: dict[str, str] | None = None,
) -> tuple[set[str], Path]:
    from .local_db import LocalZoteroReader

    with LocalZoteroReader(db_path=db_path) as reader:
        sqlite_library_id = None
        if library is not None:
            sqlite_library_id = reader.resolve_library_id(
                library.get("library_id", ""),
                library.get("library_type", "user"),
            )
            if sqlite_library_id is None:
                raise ValueError(
                    "Active library is not present in the local Zotero snapshot"
                )
        read_kwargs = dict(
            include_fulltext=False,
            collection_keys=collection_keys,
        )
        if library is not None:
            read_kwargs["library_id"] = sqlite_library_id
        items = reader.get_items_with_text(**read_kwargs)
        resolved_path = Path(reader.db_path)
    return {item.key for item in _deduplicate_index_items(items)}, resolved_path


def audit_semantic_database(
    config_path: str | Path,
    *,
    persist_directory: str | Path | None = None,
    zotero_db_path: str | None = None,
    full_integrity: bool = True,
    compare_zotero: bool = True,
    library: dict[str, str] | None = None,
) -> DatabaseHealthReport:
    """Audit Chroma and Zotero state without opening a writable DB handle."""
    config_path = Path(config_path)
    try:
        config = _read_config(config_path)
    except Exception as error:
        report = DatabaseHealthReport("unknown", "zotero_library")
        report.add("error", "configuration", f"Could not read config: {error}")
        return report

    semantic_config = config.get("semantic_search", {})
    collection_name = semantic_config.get("collection_name", "zotero_library")
    if library is not None:
        from .client import library_identity

        collection_name = scoped_collection_name(
            collection_name,
            scope_identity=library_identity(library),
        )
    if persist_directory is None:
        persist_directory = Path.home() / ".config" / "zotero-mcp" / "chroma_db"
    persist_directory = Path(persist_directory)
    database_path = persist_directory / "chroma.sqlite3"
    report = DatabaseHealthReport(str(database_path), collection_name)

    update_lock, snapshot_available = _acquire_shared_update_lock()
    if not snapshot_available:
        report.add(
            "error",
            "update_lock",
            "A semantic database update is running; retry the health audit after it stops.",
        )
        return report
    report.add("ok", "update_lock", "No update can start during this audit snapshot.")

    lifecycle_lock = index_lifecycle_lock(persist_directory, exclusive=False)
    try:
        lifecycle_lock.__enter__()
    except Exception as error:
        if update_lock is not None:
            release_file_lock(update_lock)
        report.add(
            "error",
            "lifecycle_lock",
            f"Could not lock the Chroma lifecycle for auditing: {error}",
        )
        return report
    report.add(
        "ok",
        "lifecycle_lock",
        "Collection swaps and physical cleanup are blocked during the audit.",
    )

    if not database_path.is_file():
        report.add("error", "database", f"Database does not exist: {database_path}")
        lifecycle_lock.__exit__(None, None, None)
        if update_lock is not None:
            release_file_lock(update_lock)
        return report

    markers = sorted(persist_directory.glob(".zotero-mcp-rebuild-*.json"))
    if markers:
        report.add(
            "warning",
            "rebuild_marker",
            f"Found {len(markers)} staged-rebuild recovery marker(s).",
        )
    else:
        report.add("ok", "rebuild_marker", "No staged-rebuild marker is present.")

    try:
        connection = _read_only_connection(database_path)
    except Exception as error:
        report.add("error", "database", f"Could not open database read-only: {error}")
        lifecycle_lock.__exit__(None, None, None)
        if update_lock is not None:
            release_file_lock(update_lock)
        return report

    try:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        if quick_check == "ok":
            report.add("ok", "sqlite_quick_check", "SQLite quick check passed.")
        else:
            report.add("error", "sqlite_quick_check", str(quick_check))

        if full_integrity:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity == "ok":
                report.add("ok", "sqlite_integrity", "Full SQLite integrity check passed.")
            else:
                report.add("error", "sqlite_integrity", str(integrity))
        else:
            report.add("skipped", "sqlite_integrity", "Skipped by --quick.")

        tables = _table_names(connection)
        required_tables = {
            "collections",
            "segments",
            "embeddings",
            "embedding_metadata",
            "embeddings_queue",
            "max_seq_id",
        }
        missing_tables = required_tables - tables
        if missing_tables:
            report.add(
                "error",
                "schema",
                "Missing Chroma tables: " + ", ".join(sorted(missing_tables)),
            )
            return report

        foreign_key_rows = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_rows:
            targets = {
                row[2]
                for row in connection.execute("PRAGMA foreign_key_list(segments)")
            }
            unowned_segments = connection.execute(
                """
                SELECT COUNT(*) FROM segments AS segment
                LEFT JOIN collections AS collection
                  ON collection.id = segment.collection
                WHERE collection.id IS NULL
                """
            ).fetchone()[0]
            vendor_schema_only = (
                targets == {"collection"}
                and "collection" not in tables
                and all(row[0] == "segments" for row in foreign_key_rows)
                and unowned_segments == 0
            )
            level = "ok" if vendor_schema_only else "error"
            detail = (
                "Chroma's segments table references the non-existent singular "
                "table name 'collection', but every segment belongs to an existing "
                "collection. No repair is required."
                if vendor_schema_only
                else f"SQLite reports {len(foreign_key_rows)} foreign-key violation(s)."
            )
            report.add(level, "foreign_keys", detail)
        else:
            report.add("ok", "foreign_keys", "No foreign-key violations found.")

        collections = connection.execute(
            "SELECT id, name, dimension FROM collections WHERE name = ?",
            (collection_name,),
        ).fetchall()
        if len(collections) != 1:
            report.add(
                "error",
                "active_collection",
                f"Expected one collection named {collection_name!r}; found {len(collections)}.",
            )
            return report
        collection = collections[0]
        collection_id = collection["id"]
        report.metrics["embedding_dimension"] = collection["dimension"]

        segment_rows = connection.execute(
            "SELECT id, type, scope FROM segments WHERE collection = ?",
            (collection_id,),
        ).fetchall()
        metadata_segments = [row for row in segment_rows if row["scope"] == "METADATA"]
        vector_segments = [row for row in segment_rows if row["scope"] == "VECTOR"]
        if len(metadata_segments) != 1 or len(vector_segments) != 1:
            report.add(
                "error",
                "segments",
                f"Expected one metadata and one vector segment; found "
                f"{len(metadata_segments)} and {len(vector_segments)}.",
            )
            return report
        report.add("ok", "segments", "Active collection has one metadata and one vector segment.")

        metadata_segment = metadata_segments[0]["id"]
        vector_segment = vector_segments[0]["id"]
        current_ids = {
            row[0]
            for row in connection.execute(
                "SELECT embedding_id FROM embeddings WHERE segment_id = ?",
                (metadata_segment,),
            )
        }
        report.metrics["passages"] = len(current_ids)

        item_keys = {
            row[0]
            for row in connection.execute(
                """
                SELECT DISTINCT metadata.string_value
                FROM embeddings AS embedding
                JOIN embedding_metadata AS metadata ON metadata.id = embedding.id
                WHERE embedding.segment_id = ? AND metadata.key = 'item_key'
                """,
                (metadata_segment,),
            )
        }
        item_keys.discard(None)
        report.metrics["items"] = len(item_keys)

        duplicate_ids = connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT embedding_id FROM embeddings
                WHERE segment_id = ? GROUP BY embedding_id HAVING COUNT(*) > 1
            )
            """,
            (metadata_segment,),
        ).fetchone()[0]
        orphan_metadata = connection.execute(
            """
            SELECT COUNT(*) FROM embedding_metadata AS metadata
            LEFT JOIN embeddings AS embedding ON embedding.id = metadata.id
            WHERE embedding.id IS NULL
            """
        ).fetchone()[0]
        empty_documents = connection.execute(
            """
            SELECT COUNT(*) FROM embeddings AS embedding
            LEFT JOIN embedding_metadata AS document
              ON document.id = embedding.id AND document.key = 'chroma:document'
            WHERE embedding.segment_id = ?
              AND (document.string_value IS NULL OR length(trim(document.string_value)) = 0)
            """,
            (metadata_segment,),
        ).fetchone()[0]
        if duplicate_ids or orphan_metadata or empty_documents:
            report.add(
                "error",
                "records",
                f"duplicates={duplicate_ids}, orphan metadata={orphan_metadata}, "
                f"empty documents={empty_documents}.",
            )
        else:
            report.add(
                "ok",
                "records",
                f"{len(current_ids)} unique, non-empty passage records are internally consistent.",
            )

        chunk_metadata_count = connection.execute(
            """
            SELECT COUNT(*) FROM embeddings AS embedding
            JOIN embedding_metadata AS metadata ON metadata.id = embedding.id
            WHERE embedding.segment_id = ? AND metadata.key = 'chunk_index'
            """,
            (metadata_segment,),
        ).fetchone()[0]
        chunk_rows = connection.execute(
            """
            WITH passage AS (
                SELECT embedding.id,
                       embedding.embedding_id,
                       MAX(CASE WHEN metadata.key = 'item_key' THEN metadata.string_value END) AS item_key,
                       MAX(CASE WHEN metadata.key = 'chunk_index' THEN metadata.int_value END) AS chunk_index,
                       MAX(CASE WHEN metadata.key = 'n_chunks' THEN metadata.int_value END) AS n_chunks
                FROM embeddings AS embedding
                LEFT JOIN embedding_metadata AS metadata ON metadata.id = embedding.id
                WHERE embedding.segment_id = ?
                GROUP BY embedding.id
            )
            SELECT item_key, COUNT(*) AS actual,
                   MIN(chunk_index) AS min_chunk,
                   MAX(chunk_index) AS max_chunk,
                   MIN(n_chunks) AS min_count,
                   MAX(n_chunks) AS max_count,
                   SUM(embedding_id != item_key || '#' || chunk_index) AS malformed_ids
            FROM passage GROUP BY item_key
            """,
            (metadata_segment,),
        ).fetchall()
        if chunk_metadata_count == 0:
            bad_groups = [
                row
                for row in chunk_rows
                if row["item_key"] is None or row["actual"] != 1
            ]
        elif chunk_metadata_count != len(current_ids):
            bad_groups = list(chunk_rows)
        else:
            bad_groups = [
                row
                for row in chunk_rows
                if row["item_key"] is None
                or row["min_chunk"] != 0
                or row["max_chunk"] != row["actual"] - 1
                or row["min_count"] != row["max_count"]
                or row["max_count"] != row["actual"]
                or row["malformed_ids"]
            ]
        if bad_groups:
            report.add(
                "error",
                "chunk_layout",
                f"Found {len(bad_groups)} incomplete or malformed item chunk group(s).",
            )
        else:
            report.add(
                "ok",
                "chunk_layout",
                f"All {len(chunk_rows)} item chunk groups are complete and contiguous.",
            )

        hash_rows = connection.execute(
            """
            SELECT document.string_value, content_hash.string_value
            FROM embeddings AS embedding
            JOIN embedding_metadata AS document
              ON document.id = embedding.id AND document.key = 'chroma:document'
            JOIN embedding_metadata AS content_hash
              ON content_hash.id = embedding.id
             AND content_hash.key = 'embedding_content_sha256'
            WHERE embedding.segment_id = ?
            """,
            (metadata_segment,),
        )
        hash_count = 0
        hash_mismatches = 0
        for document, expected_hash in hash_rows:
            hash_count += 1
            actual_hash = hashlib.sha256(document.encode("utf-8")).hexdigest()
            hash_mismatches += actual_hash != expected_hash
        report.metrics["content_hashes"] = hash_count
        report.metrics["legacy_passages_without_hash"] = len(current_ids) - hash_count
        if hash_mismatches:
            report.add(
                "error",
                "content_hashes",
                f"Found {hash_mismatches} stored content-hash mismatch(es).",
            )
        elif hash_count < len(current_ids):
            report.add(
                "warning",
                "content_hashes",
                f"All {hash_count} available hashes match; {len(current_ids) - hash_count} "
                "legacy passages rely on exact stored-document comparison.",
            )
        else:
            report.add("ok", "content_hashes", f"All {hash_count} content hashes match.")

        failed_fulltext = connection.execute(
            """
            SELECT COUNT(DISTINCT metadata.string_value)
            FROM embeddings AS embedding
            JOIN embedding_metadata AS metadata ON metadata.id = embedding.id
            JOIN embedding_metadata AS failure ON failure.id = embedding.id
            WHERE embedding.segment_id = ?
              AND metadata.key = 'item_key'
              AND failure.key = 'fulltext_error'
            """,
            (metadata_segment,),
        ).fetchone()[0]
        report.metrics["items_with_fulltext_errors"] = failed_fulltext
        if failed_fulltext:
            report.add(
                "warning",
                "fulltext",
                f"{failed_fulltext} item(s) are indexed from metadata after full-text extraction failed.",
            )
        else:
            report.add("ok", "fulltext", "No indexed item reports a full-text extraction failure.")

        vector_path = persist_directory / vector_segment
        metadata_path = vector_path / "index_metadata.pickle"
        if not metadata_path.is_file() and not current_ids:
            report.add(
                "ok",
                "vector_checkpoint",
                "The empty collection does not require an HNSW checkpoint.",
            )
        elif not metadata_path.is_file():
            report.add("error", "vector_checkpoint", f"Missing {metadata_path}.")
        else:
            try:
                replay_ids = _load_persisted_ids(metadata_path)
                checkpoint_row = connection.execute(
                    "SELECT seq_id FROM max_seq_id WHERE segment_id = ?",
                    (vector_segment,),
                ).fetchone()
                if checkpoint_row is None:
                    raise ValueError("vector segment has no max_seq_id checkpoint")
                checkpoint = checkpoint_row[0]
                replayed = 0
                topic = f"%/{collection_id}"
                for row in connection.execute(
                    """
                    SELECT operation, id, vector IS NOT NULL AS has_vector
                    FROM embeddings_queue
                    WHERE seq_id > ? AND topic LIKE ? ORDER BY seq_id
                    """,
                    (checkpoint, topic),
                ):
                    replayed += 1
                    operation, record_id, has_vector = row
                    if operation in (0, 2) and has_vector:
                        replay_ids.add(record_id)
                    elif operation == 3:
                        replay_ids.discard(record_id)
                report.metrics["hnsw_checkpoint"] = checkpoint
                report.metrics["hnsw_replayed_operations"] = replayed
                missing_vectors = current_ids - replay_ids
                extra_vectors = replay_ids - current_ids
                if missing_vectors or extra_vectors:
                    report.add(
                        "error",
                        "vector_replay",
                        f"Replay differs from metadata: {len(missing_vectors)} missing, "
                        f"{len(extra_vectors)} extra passage IDs.",
                    )
                else:
                    report.add(
                        "ok",
                        "vector_replay",
                        f"Persisted HNSW state plus {replayed} queued operation(s) "
                        f"reconstruct all {len(current_ids)} passage IDs.",
                    )
            except Exception as error:
                report.add("error", "vector_checkpoint", f"Could not verify HNSW replay: {error}")

        referenced_segment_ids = {
            row[0] for row in connection.execute("SELECT id FROM segments")
        }
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )
        orphan_directories = [
            path
            for path in persist_directory.iterdir()
            if path.is_dir()
            and uuid_pattern.match(path.name)
            and path.name not in referenced_segment_ids
        ]
        orphan_bytes = sum(_directory_size(path) for path in orphan_directories)
        report.metrics["orphan_segment_directories"] = len(orphan_directories)
        report.metrics["orphan_segment_bytes"] = orphan_bytes
        if orphan_directories:
            report.add(
                "warning",
                "orphan_segments",
                f"{len(orphan_directories)} unreferenced segment "
                f"{'directory' if len(orphan_directories) == 1 else 'directories'} use "
                f"{orphan_bytes / (1024 * 1024):.1f} MiB. The next update-db "
                "reclaims them under the exclusive index lifecycle lock; this "
                "read-only audit removed nothing.",
            )
        else:
            report.add("ok", "orphan_segments", "No unreferenced segment directories found.")

        if compare_zotero:
            configured_db_path = zotero_db_path or semantic_config.get("zotero_db_path")
            try:
                zotero_keys, resolved_zotero_path = _read_zotero_keys(
                    configured_db_path,
                    semantic_config.get("collection_keys"),
                    library,
                )
                report.metrics["zotero_items"] = len(zotero_keys)
                zotero_wal = resolved_zotero_path.with_name(
                    resolved_zotero_path.name + "-wal"
                )
                if zotero_wal.exists():
                    report.add(
                        "warning",
                        "zotero_coverage",
                        "Zotero has a WAL; immutable coverage may omit its newest changes.",
                    )
                else:
                    missing_items = zotero_keys - item_keys
                    extra_items = item_keys - zotero_keys
                    if missing_items or extra_items:
                        report.add(
                            "error",
                            "zotero_coverage",
                            f"Index coverage differs from Zotero: {len(missing_items)} missing, "
                            f"{len(extra_items)} extra item key(s).",
                        )
                    else:
                        report.add(
                            "ok",
                            "zotero_coverage",
                            f"All {len(zotero_keys)} current Zotero item keys are represented.",
                        )
            except FileNotFoundError:
                report.add("skipped", "zotero_coverage", "No local Zotero database was found.")
            except Exception as error:
                report.add("warning", "zotero_coverage", f"Could not compare Zotero keys: {error}")
        else:
            report.add("skipped", "zotero_coverage", "Skipped by --no-zotero-compare.")
    except Exception as error:
        report.add("error", "audit", f"Health audit failed: {error}")
    finally:
        connection.close()
        lifecycle_lock.__exit__(None, None, None)
        if update_lock is not None:
            release_file_lock(update_lock)

    return report


def format_health_report(report: DatabaseHealthReport) -> str:
    """Render a concise terminal health report."""
    status = "HEALTHY" if report.healthy else "UNHEALTHY"
    lines = [
        "=== Semantic Search Database Health ===",
        f"Status: {status}",
        "Mode: read-only (no repair actions)",
        f"Collection: {report.collection_name}",
        f"Database: {report.database_path}",
    ]
    if report.metrics:
        lines.append("\nMetrics:")
        labels = {
            "items": "Indexed items",
            "zotero_items": "Zotero items",
            "passages": "Indexed passages",
            "embedding_dimension": "Embedding dimension",
            "content_hashes": "Verified content hashes",
            "legacy_passages_without_hash": "Legacy passages without hashes",
            "items_with_fulltext_errors": "Items with full-text errors",
            "hnsw_replayed_operations": "Queued HNSW operations",
            "orphan_segment_directories": "Orphan segment directories",
            "orphan_segment_bytes": "Orphan segment bytes",
        }
        for key, label in labels.items():
            if key in report.metrics:
                lines.append(f"- {label}: {report.metrics[key]}")
    lines.append("\nChecks:")
    symbols = {"ok": "OK", "warning": "WARN", "error": "ERROR", "skipped": "SKIP"}
    for finding in report.findings:
        lines.append(
            f"[{symbols.get(finding.level, finding.level.upper())}] "
            f"{finding.check}: {finding.detail}"
        )
    return "\n".join(lines)
