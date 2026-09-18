#!/usr/bin/env python3
"""Activate the already re-embedded Nemotron shadow Chroma index.

Dry-run by default. With --execute this changes only:
  * the two persisted schema embedding-function descriptors per live semantic
    collection; and
  * collection_metadata.zotero_mcp_embedding_identity.

It does not touch record IDs, documents, record metadata, embeddings, HNSW
files, Zotero, or config_json_str.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import sqlite3
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_CONFIG = Path("/home/mheiss/.config/zotero-mcp-nemotron/config.json")
DEFAULT_CHROMA = Path("/home/mheiss/.config/zotero-mcp-nemotron/chroma_db_nemotron")
DEFAULT_REPO = Path("/home/mheiss/zotero-mcp")
PRODUCTION_CHROMA = Path("/home/mheiss/.config/zotero-mcp/chroma_db")
IDENTITY_KEY = "zotero_mcp_embedding_identity"
ALLOWED_ENDPOINTS = {"http://127.0.0.1:8032/v1", "http://localhost:8032/v1"}


def resolved(path: Path, *, strict: bool = True) -> Path:
    return path.expanduser().resolve(strict=strict)


def load_target(config_path: Path, chroma_path: Path, repo_root: Path) -> dict[str, Any]:
    config_path = resolved(config_path)
    chroma_path = resolved(chroma_path)
    repo_root = resolved(repo_root)
    production = resolved(PRODUCTION_CHROMA, strict=False)

    if "nemotron" not in str(config_path).casefold() or "nemotron" not in str(chroma_path).casefold():
        raise RuntimeError("refusing non-Nemotron config/Chroma path")
    if chroma_path == production:
        raise RuntimeError(f"refusing production Chroma path: {chroma_path}")

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    semantic = cfg.get("semantic_search") or {}
    configured_path = semantic.get("persist_directory")
    if not configured_path or resolved(Path(configured_path)) != chroma_path:
        raise RuntimeError("--chroma-path does not match semantic_search.persist_directory")
    if semantic.get("embedding_model") != "openai":
        raise RuntimeError("embedding_model must be 'openai'")

    ec = dict(semantic.get("embedding_config") or {})
    endpoint = str(ec.get("base_url") or "").rstrip("/")
    parsed = urlparse(endpoint)
    if (
        endpoint not in {x.rstrip("/") for x in ALLOWED_ENDPOINTS}
        or parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.port != 8032
    ):
        raise RuntimeError(f"refusing unexpected endpoint: {endpoint or '<missing>'}")

    model_name = str(ec.get("model_name") or "").strip()
    model_identity = str(ec.get("model_identity") or "").strip()
    if not model_name or "nemotron" not in model_identity.casefold():
        raise RuntimeError("Nemotron model_name/model_identity missing")
    if ec.get("query_instruction"):
        raise RuntimeError("query_instruction must be absent for Nemotron")
    if ec.get("query_prefix") != "query: " or ec.get("document_prefix") != "passage: ":
        raise RuntimeError("expected query_prefix='query: ' and document_prefix='passage: '")
    if not (repo_root / "src/zotero_mcp/chroma_client.py").is_file():
        raise RuntimeError(f"not a Zotero-MCP source tree: {repo_root}")

    return {
        "config_path": config_path,
        "chroma_path": chroma_path,
        "repo_root": repo_root,
        "base_collection": str(semantic.get("collection_name") or "zotero_library"),
        "endpoint": endpoint,
        "model_name": model_name,
        "model_identity": model_identity,
        "embedding_config": ec,
    }


def build_descriptor(target: dict[str, Any]) -> dict[str, Any]:
    """Serialize through the current working tree's actual EF implementation."""
    src = str(target["repo_root"] / "src")
    sys.path.insert(0, src)
    try:
        from zotero_mcp.chroma_client import OpenAIEmbeddingFunction  # type: ignore

        ec = target["embedding_config"]
        ef = OpenAIEmbeddingFunction(
            model_name=target["model_name"],
            api_key=ec.get("api_key"),
            base_url=target["endpoint"],
            request_batch_size=ec.get("request_batch_size"),
            rate_limit_rps=ec.get("rate_limit_rps"),
            query_instruction=ec.get("query_instruction"),
            query_prefix=ec.get("query_prefix"),
            document_prefix=ec.get("document_prefix"),
        )
        serialized = ef.get_config()
        if ef.name() != "openai":
            raise RuntimeError(f"unexpected EF name: {ef.name()!r}")
    finally:
        sys.path.remove(src)

    expected = {
        "model_name": target["model_name"],
        "base_url": target["endpoint"],
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
    }
    for key, value in expected.items():
        if serialized.get(key) != value:
            raise RuntimeError(f"current EF serialized unexpected {key}: {serialized.get(key)!r}")
    if "query_instruction" in serialized:
        raise RuntimeError("current EF unexpectedly persists query_instruction")
    return {"type": "known", "name": "openai", "config": serialized}


@contextmanager
def lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def schema_slots(schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        a = schema["defaults"]["float_list"]["vector_index"]["config"]
        b = schema["keys"]["#embedding"]["float_list"]["vector_index"]["config"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("unexpected schema_str: dense-vector descriptor paths missing") from exc
    if not isinstance(a, dict) or not isinstance(b, dict) or "embedding_function" not in a or "embedding_function" not in b:
        raise RuntimeError("unexpected schema_str embedding-function shape")
    return a, b


def live_collection(name: str, base: str) -> bool:
    if name == base:
        return True
    if not name.startswith(base + "__"):
        return False
    return not (name.startswith(base + "__rebuild_") or name.startswith(base + "__replaced_"))


def schema_without_embedding_functions(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy with only the two mutable EF descriptors masked."""
    masked = copy.deepcopy(schema)
    a, b = schema_slots(masked)
    a["embedding_function"] = "<embedding-function>"
    b["embedding_function"] = "<embedding-function>"
    return masked


def snapshot_untouched(conn: sqlite3.Connection) -> dict[str, Any]:
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    counts = {}
    for table in ("segments", "embeddings", "embedding_metadata", "embeddings_queue"):
        if table in tables:
            counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return {
        "collections": [tuple(r) for r in conn.execute(
            "SELECT id,name,dimension,database_id,config_json_str FROM collections ORDER BY id"
        )],
        "other_collection_metadata": [tuple(r) for r in conn.execute(
            "SELECT * FROM collection_metadata WHERE key<>? ORDER BY collection_id,key", (IDENTITY_KEY,)
        )],
        "record_table_counts": counts,
    }


def activate_database(
    conn: sqlite3.Connection,
    target: dict[str, Any],
    descriptor: dict[str, Any],
    *,
    execute: bool,
    backup_path: Path | None,
) -> list[dict[str, Any]]:
    collection_cols = {r[1] for r in conn.execute("PRAGMA table_info(collections)")}
    metadata_cols = {r[1] for r in conn.execute("PRAGMA table_info(collection_metadata)")}
    if not {"id", "name", "dimension", "database_id", "config_json_str", "schema_str"} <= collection_cols:
        raise RuntimeError("unexpected Chroma collections table layout")
    if not {"collection_id", "key", "str_value"} <= metadata_cols:
        raise RuntimeError("unexpected Chroma collection_metadata table layout")

    rows = conn.execute(
        "SELECT id,name,dimension,config_json_str,schema_str FROM collections ORDER BY name"
    ).fetchall()
    rows = [r for r in rows if live_collection(str(r["name"]), target["base_collection"])]
    if not rows:
        raise RuntimeError("no live Zotero semantic collections found")

    plan = []
    for row in rows:
        schema = json.loads(str(row["schema_str"] or ""))
        a, b = schema_slots(schema)
        identities = conn.execute(
            "SELECT str_value FROM collection_metadata WHERE collection_id=? AND key=?",
            (row["id"], IDENTITY_KEY),
        ).fetchall()
        if len(identities) != 1 or identities[0][0] is None:
            raise RuntimeError(f"{row['name']}: expected exactly one string {IDENTITY_KEY} row")
        old_identity = str(identities[0][0])
        if old_identity != target["model_identity"] and "qwen" not in old_identity.casefold():
            raise RuntimeError(f"{row['name']}: refusing unexpected identity {old_identity!r}")
        plan.append({
            "id": str(row["id"]),
            "name": str(row["name"]),
            "old_identity": old_identity,
            "old_a": copy.deepcopy(a["embedding_function"]),
            "old_b": copy.deepcopy(b["embedding_function"]),
            "schema_str": str(row["schema_str"]),
            "schema_without_ef": schema_without_embedding_functions(schema),
        })

    if not execute:
        return plan

    if backup_path is None:
        raise RuntimeError("execute mode requires backup_path")
    backup = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "chroma_path": str(target["chroma_path"]),
        "model_identity": target["model_identity"],
        "collections": [
            {"id": p["id"], "name": p["name"], "schema_str": p["schema_str"], "embedding_identity": p["old_identity"]}
            for p in plan
        ],
    }
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    with backup_path.open("x", encoding="utf-8") as fh:
        json.dump(backup, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())

    before = snapshot_untouched(conn)
    conn.execute("BEGIN IMMEDIATE")
    try:
        for p in plan:
            schema = json.loads(p["schema_str"])
            a, b = schema_slots(schema)
            a["embedding_function"] = copy.deepcopy(descriptor)
            b["embedding_function"] = copy.deepcopy(descriptor)
            conn.execute(
                "UPDATE collections SET schema_str=? WHERE id=?",
                (json.dumps(schema, separators=(",", ":")), p["id"]),
            )
            cur = conn.execute(
                "UPDATE collection_metadata SET str_value=? WHERE collection_id=? AND key=?",
                (target["model_identity"], p["id"], IDENTITY_KEY),
            )
            if cur.rowcount != 1:
                raise RuntimeError(f"{p['name']}: identity update touched {cur.rowcount} rows")

        if snapshot_untouched(conn) != before:
            raise RuntimeError("state outside schema_str / embedding identity changed")
        for p in plan:
            row = conn.execute("SELECT schema_str FROM collections WHERE id=?", (p["id"],)).fetchone()
            schema = json.loads(str(row[0]))
            a, b = schema_slots(schema)
            if schema_without_embedding_functions(schema) != p["schema_without_ef"]:
                raise RuntimeError(f"{p['name']}: schema outside embedding-function descriptors changed")
            if a["embedding_function"] != descriptor or b["embedding_function"] != descriptor:
                raise RuntimeError(f"{p['name']}: descriptor verification failed")
            identity = conn.execute(
                "SELECT str_value FROM collection_metadata WHERE collection_id=? AND key=?",
                (p["id"], IDENTITY_KEY),
            ).fetchone()[0]
            if identity != target["model_identity"]:
                raise RuntimeError(f"{p['name']}: identity verification failed")
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("SQLite integrity_check failed")
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--chroma-path", type=Path, default=DEFAULT_CHROMA)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--backup-path", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    target = load_target(args.config_path, args.chroma_path, args.repo_root)
    descriptor = build_descriptor(target)
    backup_path = resolved(args.backup_path, strict=False) if args.backup_path else (
        target["config_path"].parent
        / f"nemotron-activation-backup-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )

    db = target["chroma_path"] / "chroma.sqlite3"
    update_lock = target["config_path"].parent / "update.lock"
    lifecycle_lock = target["chroma_path"] / ".zotero-mcp-index-lifecycle.lock"
    with ExitStack() as stack:
        stack.enter_context(lock(update_lock))
        stack.enter_context(lock(lifecycle_lock))
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        stack.callback(conn.close)
        plan = activate_database(
            conn, target, descriptor, execute=args.execute,
            backup_path=backup_path if args.execute else None,
        )

    print("=" * 72)
    print("NEMOTRON SHADOW ACTIVATION")
    print(f"mode:       {'EXECUTE' if args.execute else 'DRY RUN'}")
    print(f"config:     {target['config_path']}")
    print(f"chroma:     {target['chroma_path']}")
    print(f"identity:   {target['model_identity']}")
    print(f"endpoint:   {target['endpoint']}")
    print("descriptor:")
    print(json.dumps(descriptor, indent=2, sort_keys=True))
    print("collections:")
    for p in plan:
        old_types = [
            x.get("type", "<missing>") if isinstance(x, dict) else "<invalid>"
            for x in (p["old_a"], p["old_b"])
        ]
        print(f"  {p['name']}: identity {p['old_identity']!r} -> {target['model_identity']!r}; EF {old_types} -> known/openai")
    if args.execute:
        print(f"committed + verified; rollback metadata: {backup_path}")
    else:
        print("dry-run only; no writes performed. Re-run with --execute to commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
