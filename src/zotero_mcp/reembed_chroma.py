"""Controlled, embedding-only migration of an existing Chroma index.

This module is intentionally separate from ``update-db``. It never reads
Zotero, reparses documents, rechunks text, or supplies documents/metadata to a
Chroma write. Its sole mutation is ``collection.update(ids=..., embeddings=...)``.

Run ``python -m zotero_mcp.reembed_chroma --help`` for the guarded CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from chromadb.config import Settings

from ._file_lock import advisory_file_lock
from .chroma_client import (
    OpenAIEmbeddingFunction,
    _NoEmbeddingFunction,
    chromadb,
    index_lifecycle_lock,
)

PRODUCTION_CHROMA = Path("/home/mheiss/.config/zotero-mcp/chroma_db")
MIGRATION_CONCURRENCY = 2
ALLOWED_ENDPOINTS = {
    "http://127.0.0.1:8032/v1",
    "http://localhost:8032/v1",
}
DOCUMENT_METADATA_KEY = "chroma:document"
CONTENT_HASH_KEY = "embedding_content_sha256"
METADATA_HASH_KEY = "embedding_metadata_sha256"


@dataclass(frozen=True)
class TargetSpec:
    config_path: Path
    chroma_path: Path
    progress_path: Path
    endpoint: str
    model_name: str
    model_identity: str
    embedding_config: dict[str, Any]


@dataclass(frozen=True)
class CollectionInventory:
    name: str
    dimension: int | None
    ids: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.ids)


@dataclass(frozen=True)
class StoredRecord:
    collection_name: str
    record_id: str
    document: str
    metadata: dict[str, Any]

    @property
    def document_sha256(self) -> str:
        return hashlib.sha256(self.document.encode("utf-8")).hexdigest()

    @property
    def content_hash(self) -> str:
        return str(self.metadata.get(CONTENT_HASH_KEY) or "")


def _resolved(path: str | Path, *, strict: bool) -> Path:
    return Path(path).expanduser().resolve(strict=strict)


def _contains_nemotron(path: Path) -> bool:
    return "nemotron" in str(path).casefold()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_target(
    config_path: str | Path,
    chroma_path: str | Path,
    progress_path: str | Path | None = None,
) -> TargetSpec:
    """Load explicit target settings and enforce the migration safety fence."""
    config = _resolved(config_path, strict=True)
    chroma = _resolved(chroma_path, strict=True)
    production = PRODUCTION_CHROMA.resolve(strict=True)

    if not _contains_nemotron(config):
        raise ValueError(f"config path must contain 'nemotron': {config}")
    if not _contains_nemotron(chroma):
        raise ValueError(f"Chroma path must contain 'nemotron': {chroma}")
    if chroma == production:
        raise ValueError(f"refusing production Chroma target: {chroma}")

    try:
        raw_config = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read target config {config}: {exc}") from exc
    semantic = raw_config.get("semantic_search") or {}
    configured_chroma_raw = semantic.get("persist_directory")
    if not configured_chroma_raw:
        raise ValueError("shadow config must set semantic_search.persist_directory explicitly")
    configured_chroma = _resolved(configured_chroma_raw, strict=True)
    if configured_chroma != chroma:
        raise ValueError(
            "explicit Chroma target does not match config: "
            f"argument={chroma}, config={configured_chroma}"
        )

    if semantic.get("embedding_model") != "openai":
        raise ValueError("controlled re-embedding supports only the OpenAI-compatible backend")
    embedding_config = dict(semantic.get("embedding_config") or {})
    endpoint = str(embedding_config.get("base_url") or "").rstrip("/")
    normalized_allowed = {value.rstrip("/") for value in ALLOWED_ENDPOINTS}
    parsed = urlparse(endpoint)
    if (
        endpoint not in normalized_allowed
        or parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.port != 8032
    ):
        raise ValueError(f"refusing unexpected embedding endpoint: {endpoint or '<missing>'}")

    model_name = str(embedding_config.get("model_name") or "").strip()
    model_identity = str(
        embedding_config.get("model_identity") or model_name
    ).strip()
    if "nemotron" not in model_identity.casefold():
        raise ValueError(f"model identity must name Nemotron: {model_identity or '<missing>'}")
    if embedding_config.get("query_instruction"):
        raise ValueError("Nemotron shadow config must not retain a Qwen query_instruction")
    if embedding_config.get("query_prefix") != "query: ":
        raise ValueError("Nemotron shadow query_prefix must be exactly 'query: '")
    if embedding_config.get("document_prefix") != "passage: ":
        raise ValueError("Nemotron shadow document_prefix must be exactly 'passage: '")

    progress = _resolved(
        progress_path or config.parent / "reembed-progress.sqlite3",
        strict=False,
    )
    if not _contains_nemotron(progress):
        raise ValueError(f"progress path must contain 'nemotron': {progress}")
    if progress == chroma or _is_within(progress, chroma):
        raise ValueError("progress database must be outside the Chroma directory")

    return TargetSpec(
        config_path=config,
        chroma_path=chroma,
        progress_path=progress,
        endpoint=endpoint,
        model_name=model_name,
        model_identity=model_identity,
        embedding_config=embedding_config,
    )


def _sqlite_uri(database: Path) -> str:
    return f"file:{database}?mode=ro"


def _readonly_sqlite(chroma_path: Path) -> sqlite3.Connection:
    database = chroma_path / "chroma.sqlite3"
    if not database.is_file():
        raise ValueError(f"Chroma SQLite database is missing: {database}")
    connection = sqlite3.connect(_sqlite_uri(database), uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def read_inventory(chroma_path: Path) -> list[CollectionInventory]:
    """Read collection names, dimensions, and stable IDs without opening Chroma."""
    with _readonly_sqlite(chroma_path) as connection:
        collections = connection.execute(
            "SELECT id, name, dimension FROM collections ORDER BY name"
        ).fetchall()
        inventories = []
        for collection_id, name, dimension in collections:
            rows = connection.execute(
                """
                SELECT e.embedding_id
                  FROM segments AS s
                  JOIN embeddings AS e ON e.segment_id = s.id
                 WHERE s.collection = ? AND s.scope = 'METADATA'
                 ORDER BY e.embedding_id
                """,
                (collection_id,),
            ).fetchall()
            inventories.append(
                CollectionInventory(
                    name=str(name),
                    dimension=int(dimension) if dimension is not None else None,
                    ids=tuple(str(row[0]) for row in rows),
                )
            )
    return inventories


def print_target(spec: TargetSpec, inventory: list[CollectionInventory], limit: int | None) -> None:
    total = sum(collection.count for collection in inventory)
    selected = min(total, limit) if limit is not None else total
    print("=" * 72)
    print("CONTROLLED EMBEDDING-ONLY CHROMA MIGRATION")
    print(f"TARGET CONFIG:       {spec.config_path}")
    print(f"TARGET CHROMA:       {spec.chroma_path}")
    print(f"EMBEDDING ENDPOINT:  {spec.endpoint}")
    print(f"MODEL IDENTITY:      {spec.model_identity}")
    print("COLLECTIONS:")
    for collection in inventory:
        print(
            f"  - {collection.name}: {collection.count} passages, "
            f"dimension={collection.dimension}"
        )
    if limit is None:
        print(f"PASSAGE COUNT:       {total}")
    else:
        print(f"PASSAGE COUNT:       {total} total; {selected} selected by --limit")
    print(f"CONCURRENCY:         {MIGRATION_CONCURRENCY}")
    print(f"PROGRESS DATABASE:   {spec.progress_path}")
    print("WRITE CONTRACT:      embeddings only; no documents or metadata")
    print("=" * 72)


class ProgressStore:
    """Durable per-record completion state stored outside Chroma."""

    def __init__(self, path: Path, spec: TargetSpec):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = FULL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS completed (
                collection_name TEXT NOT NULL,
                record_id TEXT NOT NULL,
                document_sha256 TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                PRIMARY KEY (collection_name, record_id)
            )
            """
        )
        expected = {
            "target_chroma": str(spec.chroma_path),
            "embedding_endpoint": spec.endpoint,
            "model_identity": spec.model_identity,
            "document_prefix": str(spec.embedding_config.get("document_prefix") or ""),
        }
        existing = dict(self.connection.execute("SELECT key, value FROM settings"))
        if existing and existing != expected:
            raise ValueError(
                "progress database belongs to different migration settings: "
                f"existing={existing}, expected={expected}"
            )
        if not existing:
            self.connection.executemany(
                "INSERT INTO settings(key, value) VALUES (?, ?)",
                sorted(expected.items()),
            )
            self.connection.commit()

    def matches(self, record: StoredRecord) -> bool:
        row = self.connection.execute(
            """
            SELECT document_sha256, content_hash
              FROM completed
             WHERE collection_name = ? AND record_id = ?
            """,
            (record.collection_name, record.record_id),
        ).fetchone()
        return bool(
            row
            and row[0] == record.document_sha256
            and row[1] == record.content_hash
        )

    def mark_completed(self, record: StoredRecord) -> None:
        completed_at = datetime.now(timezone.utc).isoformat()
        self.connection.execute(
            """
            INSERT INTO completed(
                collection_name, record_id, document_sha256,
                content_hash, completed_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(collection_name, record_id) DO UPDATE SET
                document_sha256 = excluded.document_sha256,
                content_hash = excluded.content_hash,
                completed_at = excluded.completed_at
            """,
            (
                record.collection_name,
                record.record_id,
                record.document_sha256,
                record.content_hash,
                completed_at,
            ),
        )
        # Ordering guarantee: Chroma update succeeds first, then this commit.
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


def create_embedding_function(spec: TargetSpec) -> OpenAIEmbeddingFunction:
    config = spec.embedding_config
    return OpenAIEmbeddingFunction(
        model_name=spec.model_name,
        api_key=config.get("api_key"),
        base_url=spec.endpoint,
        request_batch_size=1,
        rate_limit_rps=config.get("rate_limit_rps"),
        query_instruction=config.get("query_instruction"),
        query_prefix=config.get("query_prefix"),
        document_prefix=config.get("document_prefix"),
    )


def _selected_inventory(
    inventory: list[CollectionInventory], limit: int | None
) -> list[CollectionInventory]:
    remaining = limit
    selected = []
    for collection in inventory:
        ids = collection.ids
        if remaining is not None:
            ids = ids[:remaining]
            remaining -= len(ids)
        selected.append(
            CollectionInventory(collection.name, collection.dimension, ids)
        )
        if remaining == 0:
            selected.extend(
                CollectionInventory(other.name, other.dimension, ())
                for other in inventory[len(selected) :]
            )
            break
    return selected


def _iter_records(
    client: Any,
    inventory: list[CollectionInventory],
    *,
    fetch_size: int = 64,
) -> Iterator[tuple[Any, int | None, StoredRecord]]:
    for collection_info in inventory:
        if not collection_info.ids:
            continue
        collection = client.get_collection(
            name=collection_info.name,
            embedding_function=_NoEmbeddingFunction(),
        )
        for offset in range(0, len(collection_info.ids), fetch_size):
            requested_ids = collection_info.ids[offset : offset + fetch_size]
            result = collection.get(
                ids=list(requested_ids),
                include=["documents", "metadatas"],
            )
            documents = dict(zip(result.get("ids") or [], result.get("documents") or []))
            metadatas = dict(zip(result.get("ids") or [], result.get("metadatas") or []))
            for record_id in requested_ids:
                if record_id not in documents:
                    raise RuntimeError(
                        f"record disappeared during migration: {collection_info.name}/{record_id}"
                    )
                document = documents[record_id]
                if not isinstance(document, str) or not document:
                    raise RuntimeError(
                        f"record has no nonempty stored document: {collection_info.name}/{record_id}"
                    )
                metadata = metadatas.get(record_id)
                if not isinstance(metadata, dict):
                    metadata = {}
                yield (
                    collection,
                    collection_info.dimension,
                    StoredRecord(
                        collection_name=collection_info.name,
                        record_id=record_id,
                        document=document,
                        metadata=metadata,
                    ),
                )


def _embed_one(
    embedding_function: Callable[[list[str]], list[list[float]]],
    record: StoredRecord,
) -> list[float]:
    embeddings = embedding_function([record.document])
    if len(embeddings) != 1:
        raise RuntimeError(
            f"embedding endpoint returned {len(embeddings)} vectors for one record"
        )
    vector = embeddings[0]
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]


def _update_embedding_only(collection: Any, record_id: str, vector: list[float]) -> None:
    """Commit one vector without supplying replacement record content."""
    collection.update(ids=[record_id], embeddings=[vector])


def execute_migration(
    spec: TargetSpec,
    inventory: list[CollectionInventory],
    *,
    limit: int | None = None,
    embedding_function: Callable[[list[str]], list[list[float]]] | None = None,
) -> dict[str, int]:
    """Replace only embeddings with a bounded two-request pipeline."""
    selected = _selected_inventory(inventory, limit)
    embedder = embedding_function or create_embedding_function(spec)
    progress = ProgressStore(spec.progress_path, spec)
    stats = {"selected": sum(item.count for item in selected), "updated": 0, "skipped": 0}
    # Reuse the shadow namespace's semantic updater lock so this tool cannot
    # race the prepared shadow update-db wrapper.
    migration_lock_path = spec.config_path.parent / "update.lock"

    try:
        with advisory_file_lock(
            migration_lock_path,
            exclusive=True,
            blocking=False,
        ) as migration_lock:
            if migration_lock is None:
                raise RuntimeError(f"another migration holds {migration_lock_path}")
            with index_lifecycle_lock(spec.chroma_path, exclusive=False):
                client = chromadb.PersistentClient(
                    path=str(spec.chroma_path),
                    settings=Settings(anonymized_telemetry=False, allow_reset=True),
                )
                record_iter = _iter_records(client, selected)
                with ThreadPoolExecutor(
                    max_workers=MIGRATION_CONCURRENCY,
                    thread_name_prefix="nemotron-reembed",
                ) as executor:
                    pending: dict[Future[list[float]], tuple[Any, int | None, StoredRecord]] = {}

                    def fill_queue() -> None:
                        while len(pending) < MIGRATION_CONCURRENCY:
                            try:
                                collection, dimension, record = next(record_iter)
                            except StopIteration:
                                return
                            if progress.matches(record):
                                stats["skipped"] += 1
                                continue
                            future = executor.submit(_embed_one, embedder, record)
                            pending[future] = (collection, dimension, record)

                    fill_queue()
                    while pending:
                        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in completed:
                            collection, dimension, record = pending.pop(future)
                            vector = future.result()
                            if dimension is not None and len(vector) != dimension:
                                raise RuntimeError(
                                    f"dimension mismatch for {record.collection_name}/"
                                    f"{record.record_id}: expected {dimension}, got {len(vector)}"
                                )
                            _update_embedding_only(
                                collection,
                                record.record_id,
                                vector,
                            )
                            progress.mark_completed(record)
                            stats["updated"] += 1
                        fill_queue()
    finally:
        progress.close()
    return stats


def _logical_snapshot(chroma_path: Path) -> dict[str, Any]:
    """Return record-visible state only; vector/index state is excluded."""
    with _readonly_sqlite(chroma_path) as connection:
        collections = {
            str(name): {"dimension": dimension, "records": {}}
            for name, dimension in connection.execute(
                "SELECT name, dimension FROM collections ORDER BY name"
            )
        }
        rows = connection.execute(
            """
            SELECT c.name, e.embedding_id, m.key,
                   m.string_value, m.int_value, m.float_value, m.bool_value
              FROM collections AS c
              JOIN segments AS s
                ON s.collection = c.id AND s.scope = 'METADATA'
              JOIN embeddings AS e ON e.segment_id = s.id
              LEFT JOIN embedding_metadata AS m ON m.id = e.id
             ORDER BY c.name, e.embedding_id, m.key
            """
        )
        for collection_name, record_id, key, string, integer, floating, boolean in rows:
            record = collections[str(collection_name)]["records"].setdefault(
                str(record_id), {"document": None, "metadata": {}}
            )
            if key is None:
                continue
            value = (string, integer, floating, boolean)
            if key == DOCUMENT_METADATA_KEY:
                record["document"] = string
            else:
                record["metadata"][str(key)] = value
        return collections


def validate_against_reference(target: Path, reference: Path) -> dict[str, Any]:
    """Compare every record-visible field while intentionally ignoring vectors."""
    target_state = _logical_snapshot(target)
    reference_state = _logical_snapshot(reference)
    target_collections = set(target_state)
    reference_collections = set(reference_state)
    report: dict[str, Any] = {
        "collection_set_match": target_collections == reference_collections,
        "collection_dimension_mismatches": [],
        "record_count_mismatches": [],
        "id_mismatches": [],
        "document_mismatches": 0,
        "metadata_mismatches": 0,
        "content_hash_mismatches": 0,
        "metadata_hash_mismatches": 0,
    }
    for name in sorted(target_collections | reference_collections):
        target_collection = target_state.get(name, {"dimension": None, "records": {}})
        reference_collection = reference_state.get(name, {"dimension": None, "records": {}})
        if target_collection["dimension"] != reference_collection["dimension"]:
            report["collection_dimension_mismatches"].append(name)
        target_records = target_collection["records"]
        reference_records = reference_collection["records"]
        if len(target_records) != len(reference_records):
            report["record_count_mismatches"].append(name)
        if set(target_records) != set(reference_records):
            report["id_mismatches"].append(name)
        for record_id in set(target_records) & set(reference_records):
            target_record = target_records[record_id]
            reference_record = reference_records[record_id]
            if target_record["document"] != reference_record["document"]:
                report["document_mismatches"] += 1
            if target_record["metadata"] != reference_record["metadata"]:
                report["metadata_mismatches"] += 1
            for key, report_key in (
                (CONTENT_HASH_KEY, "content_hash_mismatches"),
                (METADATA_HASH_KEY, "metadata_hash_mismatches"),
            ):
                if target_record["metadata"].get(key) != reference_record["metadata"].get(key):
                    report[report_key] += 1
    report["ok"] = bool(
        report["collection_set_match"]
        and not report["collection_dimension_mismatches"]
        and not report["record_count_mismatches"]
        and not report["id_mismatches"]
        and report["document_mismatches"] == 0
        and report["metadata_mismatches"] == 0
        and report["content_hash_mismatches"] == 0
        and report["metadata_hash_mismatches"] == 0
    )
    return report


def _positive_limit(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("limit must be at least 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely replace only stored Chroma embeddings in a Nemotron shadow",
    )
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--chroma-path", required=True)
    parser.add_argument("--progress-path")
    parser.add_argument("--limit", type=_positive_limit)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform embedding-only writes; otherwise inventory/dry-run only",
    )
    parser.add_argument(
        "--validate-reference",
        help="read-only Chroma path whose IDs/documents/metadata must match",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        spec = resolve_target(args.config_path, args.chroma_path, args.progress_path)
        inventory = read_inventory(spec.chroma_path)
        print_target(spec, inventory, args.limit)
        if not args.execute:
            print("DRY RUN ONLY: no Chroma or progress writes were performed.")
            stats = None
        else:
            stats = execute_migration(spec, inventory, limit=args.limit)
            print(f"Migration result: {json.dumps(stats, sort_keys=True)}")

        if args.validate_reference:
            reference = _resolved(args.validate_reference, strict=True)
            report = validate_against_reference(spec.chroma_path, reference)
            print(f"Validation result: {json.dumps(report, sort_keys=True)}")
            if not report["ok"]:
                return 1
        return 0
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
