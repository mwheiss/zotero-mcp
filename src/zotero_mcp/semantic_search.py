"""
Semantic search functionality for Zotero MCP.

This module provides semantic search capabilities by integrating ChromaDB
with the existing Zotero client to enable vector-based similarity search
over research libraries.
"""

import contextlib
import hashlib
import json
import logging
import math
import os
import re
import signal
import statistics
import sys
import threading
import time
import unicodedata
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any

try:
    import tiktoken

    _tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    tiktoken = None
    _tokenizer = None


from . import openai_batch
from ._atomic_io import atomic_write_json
from ._file_lock import advisory_file_lock
from .chroma_client import ChromaClient, create_chroma_client, record_state_hash
from .client import (
    get_current_library,
    get_default_library,
    get_zotero_client,
    library_identity,
)
from .local_db import LocalZoteroReader
from .utils import format_creators, is_local_mode, suppress_stdout

logger = logging.getLogger(__name__)

_CONTENT_CONTRACT_SIGNATURE = "lean-paper-content-v2"
_SELF_CONTAINED_FULLTEXT_SOURCES = {"betterissa-indexing"}
DEFAULT_MAX_CHUNKS_PER_ITEM = 768
_IMMEDIATE_METADATA_FIELDS = {
    "item_key",
    "library_identity",
    "library_id",
    "library_type",
    "item_type",
    "title",
    "date",
    "date_added",
    "date_modified",
    "creators",
    "publication",
    "url",
    "doi",
    "tags",
    "citation_key",
}


def _sort_local_items_by_fulltext_priority(items: list[Any]) -> list[Any]:
    """Queue cheap metadata-only items, then full text from best to worst."""

    def priority(item: Any) -> tuple[int, int | float]:
        if not getattr(item, "fulltext", None):
            return (0, 0)
        value = getattr(item, "fulltext_selection_priority", None)
        return (1, value if isinstance(value, int) else math.inf)

    return sorted(items, key=priority)


@dataclass
class _PreparedIndexBatch:
    """An embedding batch prepared in memory but not yet written to ChromaDB."""

    documents: list[str]
    metadatas: list[dict[str, Any]]
    ids: list[str]
    expected_ids: list[str]
    metadata_only_ids: list[str]
    metadata_only_metadatas: list[dict[str, Any]]
    item_keys: list[str]
    stats: dict[str, int]


class _MedianETA:
    """Estimate remaining time from all observed per-entry durations."""

    def __init__(
        self,
        total: int,
        parallelism: int = 1,
    ):
        self.total = max(0, total)
        self.parallelism = max(1, parallelism)
        self._durations: list[float] = []

    def record(self, duration: float) -> None:
        """Record one completed entry's processing duration."""
        self._durations.append(max(0.0, duration))

    def estimate(self, completed: int) -> float | None:
        """Return ETA in seconds using the median of all recorded entries."""
        completed = min(max(0, completed), self.total)
        if completed >= self.total:
            return 0.0
        if not self._durations:
            return None
        typical_duration = statistics.median(self._durations)
        return typical_duration * (self.total - completed) / self.parallelism


class _CumulativeETA:
    """Estimate remaining time from cumulative wall-clock throughput."""

    def __init__(self, total: int, clock=None):
        self.total = max(0, total)
        self._clock = clock or time.monotonic
        self._started = self._clock()

    def estimate(self, completed: int) -> float | None:
        """Return ETA using elapsed time across all completed entries."""
        completed = min(max(0, completed), self.total)
        if completed >= self.total:
            return 0.0
        elapsed = self._clock() - self._started
        if completed <= 0 or elapsed <= 0:
            return None
        return elapsed * (self.total - completed) / completed


class _UpdateInterruptController:
    """Turn SIGINT into a graceful stop, then an explicit forced exit."""

    def __init__(self, stream=None):
        self.stop_requested = threading.Event()
        self._stream = stream or sys.stderr
        self._old_handler: Any = None
        self._installed = False
        self._interrupt_count = 0

    def install(self) -> None:
        """Install the handler when update_database runs on the main thread."""
        if threading.current_thread() is not threading.main_thread():
            return
        self._old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_sigint)
        self._installed = True

    def restore(self) -> None:
        """Restore the caller's SIGINT behavior after indexing finishes."""
        if not self._installed:
            return
        signal.signal(signal.SIGINT, self._old_handler)
        self._installed = False

    def _write(self, message: str) -> None:
        try:
            self._stream.write(message)
            self._stream.flush()
        except Exception:
            pass

    def _handle_sigint(self, _signum, _frame) -> None:
        self._interrupt_count += 1
        if self._interrupt_count == 1:
            self.stop_requested.set()
            self._write(
                "\n\nInterrupt received: stopping new work and waiting for "
                "active model calls to finish.\n"
                "Press Ctrl+C again to exit immediately; this may interrupt "
                "an in-progress database write.\n"
            )
            return

        self._write("\nSecond interrupt received: forcing immediate exit (status 130).\n")
        os._exit(130)


def _format_eta(seconds: float | None) -> str:
    """Format an ETA compactly for a single-line terminal progress display."""
    if seconds is None:
        return "calculating"
    remaining = max(0, math.ceil(seconds))
    days, remaining = divmod(remaining, 86400)
    hours, remaining = divmod(remaining, 3600)
    minutes, secs = divmod(remaining, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _embedding_content_hash(document: str) -> str:
    """Fingerprint the exact text passed to the embedding provider."""
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def _publication_year(value: str | None) -> str:
    """Extract a display year from Zotero's free-form publication date."""
    if not value:
        return ""
    match = re.search(r"(?<!\d)(?:1\d{3}|20\d{2}|21\d{2})(?!\d)", value)
    return match.group(0) if match else ""


def _display_width(text: str) -> int:
    """Return the number of terminal columns occupied by *text*."""
    width = 0
    for char in text:
        category = unicodedata.category(char)
        if unicodedata.combining(char) or category in {"Mn", "Me", "Cf", "Cc"}:
            continue
        width += 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
    return width


def _truncate_display(text: str, max_columns: int) -> str:
    """Truncate text to a terminal-column budget without splitting wide glyphs."""
    max_columns = max(0, max_columns)
    if _display_width(text) <= max_columns:
        return text

    ellipsis = "..." if max_columns >= 3 else ""
    budget = max_columns - len(ellipsis)
    result = []
    used = 0
    for char in text:
        char_width = _display_width(char)
        if used + char_width > budget:
            break
        result.append(char)
        used += char_width
    return "".join(result) + ellipsis


def _terminal_columns(stream) -> int:
    """Return terminal columns for a stream, with a conservative fallback."""
    try:
        return max(2, os.get_terminal_size(stream.fileno()).columns)
    except (AttributeError, OSError, ValueError):
        return 80


def _write_progress_line(stream, text: str) -> None:
    """Clear and repaint one non-wrapping terminal progress line."""
    max_columns = _terminal_columns(stream) - 1
    line = _truncate_display(text, max_columns)
    is_tty = bool(getattr(stream, "isatty", lambda: False)())
    stream.write("\r")
    if is_tty:
        stream.write("\x1b[2K")
    stream.write(line)
    if not is_tty:
        stream.write(" " * max(0, max_columns - _display_width(line)))
    stream.flush()


def _clear_progress_line(stream) -> None:
    """Erase the current progress line and return the cursor to column zero."""
    is_tty = bool(getattr(stream, "isatty", lambda: False)())
    stream.write("\r")
    if is_tty:
        stream.write("\x1b[2K")
    else:
        stream.write(" " * (_terminal_columns(stream) - 1))
        stream.write("\r")
    stream.flush()


def _pid_is_alive(pid: int) -> bool:
    """Best-effort liveness check for a process id (POSIX ``kill(pid, 0)``)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    except Exception:
        return False
    return True


def read_lock_holder(lock_path: Path) -> tuple[int | None, bool]:
    """Return ``(holder_pid, alive)`` for the update lock, for diagnostics.

    ``holder_pid`` is None when the file is missing/unparseable. ``alive`` says
    whether that pid still exists — a held lock whose holder is dead would be a
    genuinely stale lock (flock releases those automatically on POSIX, so this
    is purely to make the user-facing "skipped" message precise).
    """
    try:
        raw = lock_path.read_text().strip()
        pid = int(raw)
    except Exception:
        return None, False
    return pid, _pid_is_alive(pid)


@contextlib.contextmanager
def _acquire_update_lock(lock_path: Path):
    """Non-blocking exclusive flock over an update-database run.

    Yields True if the lock was acquired (caller should proceed), False if
    another process already holds it (caller should skip). This prevents the
    MCP server's auto-update in ``server_lifespan`` from racing a manual
    ``zotero-mcp update-db`` invocation on the same ChromaDB collection.

    Kernel locks are released automatically when a process exits, so there is
    deliberately no override that can admit a second live updater.
    """
    with advisory_file_lock(
        lock_path,
        exclusive=True,
        blocking=False,
    ) as lock_file:
        if lock_file is None:
            yield False
            return
        # Record our pid so a concurrent invocation can report the holder.
        try:
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.write(str(os.getpid()).encode("ascii"))
            lock_file.flush()
        except Exception:
            pass
        yield True


def _truncate_to_tokens(text: str, max_tokens: int = 8000) -> str:
    """Truncate text to fit within embedding model token limit.

    Uses tiktoken for accurate token counting when available,
    falls back to conservative character-based estimation.
    """
    if _tokenizer is not None:
        tokens = _tokenizer.encode(text, disallowed_special=())
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            text = _tokenizer.decode(tokens)
    else:
        # Fallback: conservative char limit (~1.5 chars/token for non-Latin scripts)
        max_chars = max_tokens * 2
        if len(text) > max_chars:
            text = text[:max_chars]
    return text


_DEFAULT_UPDATE_CONFIG = {
    "auto_update": False,
    "update_frequency": "manual",
    "last_update": None,
    "update_days": 7,
}


def load_update_config(config_path: str | None) -> dict[str, Any]:
    """Read the semantic-search ``update_config`` block from disk.

    Pure file read with no ChromaDB or embedding-model side effects, so it is
    safe on the read-only status path. Returns defaults when the file is
    missing or unreadable.
    """
    config = dict(_DEFAULT_UPDATE_CONFIG)
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path) as f:
                file_config = json.load(f)
            config.update(file_config.get("semantic_search", {}).get("update_config", {}))
        except Exception as e:
            logger.warning(f"Error loading update config: {e}")
    return config


_DEFAULT_RERANKER_CONFIG: dict[str, Any] = {
    "enabled": False,
    "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "candidate_multiplier": 3,
}


def load_reranker_config(config_path: str | None) -> dict[str, Any]:
    """Read the semantic-search ``reranker`` block from disk.

    Pure file read with no model load, so the server can consult it (e.g. to
    decide whether to warm up) without paying the cross-encoder cost.
    """
    config = dict(_DEFAULT_RERANKER_CONFIG)
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path) as f:
                file_config = json.load(f)
            config.update(file_config.get("semantic_search", {}).get("reranker", {}))
        except Exception as e:
            logger.warning(f"Error loading reranker config: {e}")
    return config


def warmup_reranker(config_path: str | None = None) -> bool:
    """Preload the configured reranker into the process-wide cache.

    Lets the server pay the cross-encoder load cost once at startup (off the
    request path) so the first real ``zotero_semantic_search`` is fast too
    (issue #283). Returns ``True`` if a model was warmed, ``False`` if the
    reranker is disabled. Never raises — a failed warmup must not crash startup.
    """
    cfg = load_reranker_config(config_path)
    if not cfg.get("enabled", False):
        return False
    model = cfg.get("model", _DEFAULT_RERANKER_CONFIG["model"])
    try:
        get_cached_reranker(model)
        return True
    except Exception as e:
        logger.warning(f"Reranker warmup failed for '{model}': {e}")
        return False


def should_update(update_config: dict[str, Any]) -> bool:
    """Decide whether an auto-update is due from ``update_config`` alone.

    Pure function of the config dict (and the wall clock) — no I/O, no model
    load — so both :class:`ZoteroSemanticSearch` and the status tool can share
    one source of truth.
    """
    if not update_config.get("auto_update", False):
        return False

    frequency = update_config.get("update_frequency", "manual")

    if frequency == "manual":
        return False
    elif frequency == "startup":
        return True
    elif frequency == "daily":
        last_update = update_config.get("last_update")
        if not last_update:
            return True
        return datetime.now() - datetime.fromisoformat(last_update) >= timedelta(days=1)
    elif frequency.startswith("every_"):
        try:
            days = int(frequency.split("_")[1])
            last_update = update_config.get("last_update")
            if not last_update:
                return True
            return datetime.now() - datetime.fromisoformat(last_update) >= timedelta(days=days)
        except (ValueError, IndexError):
            return False

    return False


# ---------------------------------------------------------------------------
# Passage-level chunking (Tier-1 grounded retrieval)
# ---------------------------------------------------------------------------

# Sentinel separating PDF pages in extracted fulltext, when present. Page-aware
# extractors may insert a form-feed between pages; the chunker uses it to map a
# character offset back to a 1-indexed page. Absent it, only char offsets are
# reported and ``page`` is omitted from passage metadata.
_PAGE_SEPARATOR = "\f"


def split_into_passages(
    text: str,
    chunk_size: int = 1500,
    overlap: int = 200,
    max_chunks: int = DEFAULT_MAX_CHUNKS_PER_ITEM,
) -> list[tuple[str, int, int]]:
    """Split *text* into balanced, overlapping natural passages.

    Pure function (no I/O, no model load) so it is unit-testable in isolation.
    Returns a list of ``(passage_text, char_start, char_end)`` tuples with
    character offsets into the original string. The required passage count is
    calculated first from ``chunk_size`` and ``overlap``; internal boundaries
    are then distributed evenly and nudged within a narrow tolerance toward a
    paragraph, sentence, line, or word boundary. This avoids a greedy first
    passage leaving a disproportionately small final passage. Consecutive
    windows overlap by ``overlap`` source characters so a relevant span
    straddling a boundary is still captured whole in one of them. At most
    ``max_chunks`` passages are produced (a guard against pathologically long
    documents inflating the index).
    """
    text = (text or "").strip()
    if not text:
        return []
    chunk_size = max(1, int(chunk_size))
    overlap = max(0, int(overlap))
    max_chunks = max(1, int(max_chunks))
    if overlap >= chunk_size:
        overlap = chunk_size // 4

    n = len(text)
    if n <= chunk_size:
        return [(text, 0, n)]

    stride = chunk_size - overlap
    required_chunks = max(1, math.ceil((n - overlap) / stride))
    chunk_count = min(required_chunks, max_chunks)
    # Preserve the historical cap contract: if the configured cap cannot
    # cover the document at the requested size, index only that bounded prefix
    # rather than creating passages larger than the embedding budget.
    covered_end = min(n, chunk_count * chunk_size - (chunk_count - 1) * overlap)
    ideal_length = (covered_end + (chunk_count - 1) * overlap) / chunk_count
    snap_tolerance = max(1, int(ideal_length * 0.10))

    def natural_boundary(ideal: int, lower: int, upper: int) -> int:
        """Return a coherent boundary near *ideal*, preferring paragraphs."""
        lower = max(1, lower)
        upper = min(covered_end - 1, upper)
        if lower > upper:
            return max(1, min(ideal, covered_end - 1))
        for separator in ("\n\n", ". ", ".\n", "\n", " "):
            candidates: list[int] = []
            cursor = text.find(separator, lower, upper + 1)
            while cursor != -1:
                boundary = cursor + len(separator)
                if boundary <= upper:
                    candidates.append(boundary)
                cursor = text.find(separator, cursor + 1, upper + 1)
            if candidates:
                return min(candidates, key=lambda boundary: (abs(boundary - ideal), boundary))
        return max(lower, min(ideal, upper))

    ends: list[int] = []
    previous_end = 0
    for index in range(1, chunk_count):
        ideal_end = round(index * (covered_end - overlap) / chunk_count + overlap)
        start = max(0, previous_end - overlap)
        remaining_chunks = chunk_count - index
        lower = max(
            start + 1,
            ideal_end - snap_tolerance,
            covered_end - remaining_chunks * stride,
        )
        upper = min(start + chunk_size, ideal_end + snap_tolerance)
        end = natural_boundary(ideal_end, lower, upper)
        ends.append(end)
        previous_end = end
    ends.append(covered_end)

    passages: list[tuple[str, int, int]] = []
    previous_end = 0
    for end in ends:
        start = max(0, previous_end - overlap) if passages else 0
        chunk = text[start:end].strip()
        if chunk:
            passages.append((chunk, start, end))
        previous_end = end
    return passages


def _page_for_offset(text: str, offset: int) -> int | None:
    """Return the 1-indexed page containing *offset*, or None if unknowable.

    Only meaningful when *text* carries ``_PAGE_SEPARATOR`` form-feed page
    breaks (page-aware extraction). Returns None otherwise so callers can omit
    a page field rather than report a misleading one.
    """
    if _PAGE_SEPARATOR not in text:
        return None
    return text.count(_PAGE_SEPARATOR, 0, max(0, offset)) + 1


def best_snippet(query: str, text: str, width: int = 320) -> tuple[str, int]:
    """Return the ``width``-char window of *text* richest in query terms.

    Used to surface a *grounded* quote — the part of a matched document that
    actually overlaps the query — instead of a blind head-truncation. Returns
    ``(snippet, char_start)``. Falls back to the head of the text when no query
    term appears. Pure and dependency-free (lexical overlap only).
    """
    text = text or ""
    if not text.strip():
        return "", 0
    if len(text) <= width:
        return text.strip(), 0
    terms = [t for t in re.findall(r"\w+", (query or "").lower()) if len(t) > 2]
    if not terms:
        return text[:width].strip(), 0
    lowered = text.lower()
    # Score each candidate window anchored at a query-term hit; keep the best.
    best_start = 0
    best_score = -1
    for m in re.finditer(r"\w+", lowered):
        if m.group(0) not in terms:
            continue
        start = max(0, m.start() - width // 3)
        window = lowered[start : start + width]
        score = sum(window.count(t) for t in terms)
        if score > best_score:
            best_score = score
            best_start = start
    snippet = text[best_start : best_start + width].strip()
    return snippet, best_start


def _passages_overlap(candidate: dict[str, Any], selected: dict[str, Any]) -> bool:
    """Return True when two matched snippets repeat substantially."""
    candidate_kind = candidate["meta"].get("passage_kind")
    selected_kind = selected["meta"].get("passage_kind")
    if candidate_kind and selected_kind and candidate_kind != selected_kind:
        return False

    candidate_text = " ".join(candidate["passage"].lower().split())
    selected_text = " ".join(selected["passage"].lower().split())
    if candidate_text and candidate_text == selected_text:
        return True

    start = max(candidate["passage_start"], selected["passage_start"])
    end = min(candidate["passage_end"], selected["passage_end"])
    overlap = max(0, end - start)
    shorter = min(
        candidate["passage_end"] - candidate["passage_start"],
        selected["passage_end"] - selected["passage_start"],
    )
    return shorter > 0 and overlap / shorter >= 0.5


def _passage_result(candidate: dict[str, Any]) -> dict[str, Any]:
    """Build the public supporting-passage representation."""
    result = {
        "chunk_id": candidate["raw_id"],
        "matched_passage": candidate["passage"],
        "similarity_score": candidate["similarity"],
    }
    meta = candidate["meta"]
    result["content_hash"] = _embedding_content_hash(candidate["document"])
    for key in (
        "chunk_index",
        "n_chunks",
        "passage_kind",
        "char_start",
        "char_end",
        "page",
    ):
        if key in meta:
            result[key] = meta[key]
    if "char_start" not in result and candidate["passage_offset"]:
        result["passage_offset"] = candidate["passage_offset"]
    return result


class CrossEncoderReranker:
    """Optional cross-encoder re-ranker for semantic search results."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[int]:
        """Re-rank documents by relevance to query.

        Returns indices of top_k documents in descending relevance order.
        """
        return [idx for idx, _ in self.rerank_with_scores(query, documents, top_k)]

    def rerank_with_scores(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        """Re-rank documents, returning ``(index, score)`` pairs.

        Scores are the raw cross-encoder relevance logits, surfaced so search
        results can report *why* an item ranked where it did, not just an
        opaque order.
        """
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, float(scores[i])) for i in ranked[:top_k]]


# Process-wide reranker cache (issue #283).
#
# The MCP search path builds a fresh ``ZoteroSemanticSearch`` per request, so a
# reranker held on the instance (``self._reranker``) was reloaded from disk on
# *every* call — the cross-encoder load dominates at ~tens of seconds and blew
# past client timeouts. The weights are immutable for a given ``model_name``, so
# caching the loaded reranker at module scope keeps it warm across requests and
# instances. The lock prevents two concurrent first-calls from double-loading.
_RERANKER_CACHE: dict[str, CrossEncoderReranker] = {}
_RERANKER_CACHE_LOCK = threading.Lock()


def get_cached_reranker(model_name: str) -> CrossEncoderReranker:
    """Return a process-wide cached reranker, loading it once per ``model_name``."""
    cached = _RERANKER_CACHE.get(model_name)
    if cached is not None:
        return cached
    with _RERANKER_CACHE_LOCK:
        # Re-check under the lock: another thread may have loaded it while we
        # waited, and the model load is far too expensive to repeat.
        cached = _RERANKER_CACHE.get(model_name)
        if cached is None:
            cached = CrossEncoderReranker(model_name=model_name)
            _RERANKER_CACHE[model_name] = cached
        return cached


class ZoteroSemanticSearch:
    """Semantic search interface for Zotero libraries using ChromaDB."""

    def __init__(
        self,
        chroma_client: ChromaClient | None = None,
        config_path: str | None = None,
        db_path: str | None = None,
        allow_embedding_mismatch: bool = False,
    ):
        """
        Initialize semantic search.

        Args:
            chroma_client: Optional ChromaClient instance
            config_path: Path to configuration file
            db_path: Optional path to Zotero database (overrides config file)
        """
        self.library = get_current_library()
        self.library_identity = library_identity(self.library)
        self.chroma_client = chroma_client or create_chroma_client(
            config_path,
            allow_embedding_mismatch=allow_embedding_mismatch,
            scope_identity=self.library_identity,
        )
        configured_collection = "zotero_library"
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path) as config_file:
                    configured_collection = (
                        json.load(config_file)
                        .get("semantic_search", {})
                        .get("collection_name", configured_collection)
                    )
            except Exception:
                pass
        actual_collection = getattr(self.chroma_client, "collection_name", None)
        self._uses_legacy_state = (
            actual_collection == configured_collection
            if actual_collection
            else self.library_identity == library_identity(get_default_library())
        )
        self.zotero_client = get_zotero_client()
        self.config_path = config_path
        self.db_path = db_path  # CLI override for Zotero database path
        # Item keys seen by the most recent local sqlite scan (set by
        # _get_items_from_local_db); used to verify watermark promotion.
        self._last_scan_snapshot_keys: set[str] | None = None
        # Top-level keys that belong in the configured semantic corpus. Unlike
        # the snapshot set, this excludes deleted, child, and collection-
        # filtered items and can be reconciled against ChromaDB.
        self._last_scan_indexable_keys: set[str] | None = None
        # Active attachment keys grouped by parent from the canonical API item
        # listing. Local immutable SQLite reads can retain attachments that
        # Zotero has already deleted from its live state.
        self._last_api_attachment_keys_by_parent: dict[str, set[str]] | None = None
        self._last_scan_attachment_snapshot_complete = True

        # Load update configuration
        self.update_config = self._load_update_config()
        library_state = self._load_library_state()
        if "last_update" in library_state:
            self.update_config["last_update"] = library_state["last_update"]

        # Reranker (lazy-initialized on first search)
        self._reranker: CrossEncoderReranker | None = None
        self._reranker_config = self._load_reranker_config()

        # Passage-level chunking (opt-in; default off preserves item-level
        # indexing and existing collections byte-for-byte).
        self._chunking_config = self._load_chunking_config()
        self._active_fulltext = self._load_fulltext_setting()

    def _load_chunking_config(self) -> dict[str, Any]:
        """Load passage-chunking configuration from file or use defaults.

        When ``enabled`` is true, each item is indexed as several overlapping
        passages (id ``<item_key>#<n>``) instead of one item-level vector, so
        semantic search returns grounded passage quotes and long PDFs are
        searchable past the single-vector truncation limit. Off by default.
        """
        config: dict[str, Any] = {
            "enabled": False,
            "chunk_size": 1500,
            "overlap": 200,
            "max_chunks_per_item": DEFAULT_MAX_CHUNKS_PER_ITEM,
        }
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    file_config = json.load(f)
                    config.update(file_config.get("semantic_search", {}).get("chunking", {}))
            except Exception as e:
                logger.warning(f"Error loading chunking config: {e}")
        return config

    @property
    def _chunking_enabled(self) -> bool:
        return bool(self._chunking_config.get("enabled", False))

    @property
    def _index_layout_signature(self) -> str:
        """Identify the chunk layout that produced the stored vectors."""
        if not self._chunking_enabled:
            return "item-v1"
        return "chunks-v2-balanced:{size}:{overlap}:{maximum}".format(
            size=int(self._chunking_config.get("chunk_size", 1500)),
            overlap=int(self._chunking_config.get("overlap", 200)),
            maximum=int(self._chunking_config.get("max_chunks_per_item", DEFAULT_MAX_CHUNKS_PER_ITEM)),
        )

    def _index_layout_changed(self, metadata: dict[str, Any]) -> bool:
        """Return whether an existing item needs migration to this layout."""
        stored = metadata.get("index_layout_signature")
        current = self._index_layout_signature
        if self._chunking_enabled and isinstance(stored, str):
            # Balanced chunking changes boundaries, not the retrieval schema.
            # Treat the prior algorithm as read-compatible when all configured
            # dimensions match, so deploying this improvement cannot silently
            # trigger a full-library re-embedding. Any item selected for a
            # normal content refresh is rewritten with the current signature;
            # an explicitly confirmed force rebuild migrates the rest.
            legacy_compatible = current.replace("chunks-v2-balanced:", "chunks-v1:", 1)
            if stored == legacy_compatible:
                return False
        # Old item-level indexes did not carry a signature. Preserve their
        # historical skip behavior, while an enabled chunk layout migrates old
        # records once. A stamped chunked index also migrates when disabled.
        return (self._chunking_enabled or stored is not None) and (stored != current)

    def _load_reranker_config(self) -> dict[str, Any]:
        """Load reranker configuration from file or use defaults."""
        return load_reranker_config(self.config_path)

    def _get_reranker(self) -> CrossEncoderReranker | None:
        """Get the reranker, reusing the process-wide cache if enabled.

        Each MCP request builds a new ``ZoteroSemanticSearch``, so the model is
        fetched from :func:`get_cached_reranker` (loaded once per process) rather
        than reloaded per instance (issue #283).
        """
        if not self._reranker_config.get("enabled", False):
            return None
        if self._reranker is None:
            model = self._reranker_config.get("model", _DEFAULT_RERANKER_CONFIG["model"])
            self._reranker = get_cached_reranker(model)
        return self._reranker

    def _load_update_config(self) -> dict[str, Any]:
        """Load update configuration from file or use defaults."""
        return load_update_config(self.config_path)

    def _library_scope(self) -> tuple[dict[str, str], str, bool]:
        """Return the captured library scope and legacy-state ownership."""
        library = getattr(self, "library", None) or get_current_library()
        identity = getattr(self, "library_identity", None) or library_identity(library)
        uses_legacy_state = getattr(self, "_uses_legacy_state", None)
        if uses_legacy_state is None:
            uses_legacy_state = identity == library_identity(get_default_library())
        return library, identity, bool(uses_legacy_state)

    def _load_library_state(self) -> dict[str, Any]:
        """Return persisted state for this library identity."""
        if not self.config_path or not os.path.exists(self.config_path):
            return {}
        try:
            with open(self.config_path) as f:
                semantic = json.load(f).get("semantic_search", {})
            _, identity, _ = self._library_scope()
            state = semantic.get("library_states", {}).get(identity, {})
            return state if isinstance(state, dict) else {}
        except Exception as e:
            logger.warning(f"Error loading library-scoped semantic state: {e}")
            return {}

    def _load_fulltext_setting(self) -> bool:
        """Load whether local full-text extraction is enabled."""
        if not self.config_path or not os.path.exists(self.config_path):
            return False
        try:
            with open(self.config_path) as f:
                file_config = json.load(f)
                value = file_config.get("semantic_search", {}).get("fulltext", False)
                if not isinstance(value, bool):
                    raise ValueError("fulltext must be true or false")
                return value
        except Exception as e:
            logger.warning(f"Error loading fulltext setting: {e}")
            return False

    def _load_openai_batch_enabled(self) -> bool:
        """Whether OpenAI Batch API indexing is enabled by semantic config."""
        if not self.config_path or not os.path.exists(self.config_path):
            return False
        try:
            with open(self.config_path) as f:
                file_config = json.load(f)
                value = file_config.get("semantic_search", {}).get("openai_batch", {}).get("enabled", False)
                return bool(value)
        except Exception as e:
            logger.warning(f"Error loading OpenAI batch setting: {e}")
            return False

    def _resolve_openai_batch_enabled(self, use_openai_batch: bool | None) -> bool:
        """Resolve CLI override + config default for OpenAI batch indexing."""
        requested = self._load_openai_batch_enabled() if use_openai_batch is None else use_openai_batch
        return bool(requested and self.chroma_client.embedding_model == "openai")

    def _load_last_sync_version(self) -> int:
        """Last Zotero library version fully indexed into ChromaDB.

        Zero means "no prior successful sync; bootstrap required". Used to
        drive since-based incremental ingest via pyzotero's
        `item_versions(since=V)` and `new_fulltext(since=V)`.
        """
        if not self.config_path or not os.path.exists(self.config_path):
            return 0
        try:
            with open(self.config_path) as f:
                file_config = json.load(f)
                semantic = file_config.get("semantic_search", {})
                _, identity, is_default = self._library_scope()
                state = semantic.get("library_states", {}).get(identity, {})
                value = state.get(
                    "last_sync_version",
                    semantic.get("last_sync_version", 0) if is_default else 0,
                )
                return int(value) if value is not None else 0
        except Exception as e:
            logger.warning(f"Error loading last_sync_version: {e}")
            return 0

    def _load_indexed_fulltext(self) -> bool | None:
        """Return whether the completed index contains maintained full text."""
        if not self.config_path or not os.path.exists(self.config_path):
            return None
        try:
            with open(self.config_path) as f:
                semantic = json.load(f).get("semantic_search", {})
                _, identity, is_default = self._library_scope()
                state = semantic.get("library_states", {}).get(identity, {})
                value = state.get(
                    "indexed_fulltext",
                    semantic.get("indexed_fulltext") if is_default else None,
                )
            return value if isinstance(value, bool) else None
        except Exception as e:
            logger.warning(f"Error loading indexed_fulltext: {e}")
            return None

    def _load_indexed_content_signature(self) -> str | None:
        """Return the embedding-content contract used by the complete index."""
        if not self.config_path or not os.path.exists(self.config_path):
            return None
        try:
            with open(self.config_path) as f:
                semantic = json.load(f).get("semantic_search", {})
                _, identity, is_default = self._library_scope()
                state = semantic.get("library_states", {}).get(identity, {})
                value = state.get(
                    "indexed_content_signature",
                    semantic.get("indexed_content_signature") if is_default else None,
                )
            return value if isinstance(value, str) and value else None
        except Exception as e:
            logger.warning(f"Error loading indexed_content_signature: {e}")
            return None

    def _load_rebuild_state(self) -> dict[str, Any] | None:
        """Return the durable contract for an interrupted in-place rebuild."""
        transient = getattr(self, "_transient_rebuild_state", None)
        if not self.config_path or not os.path.exists(self.config_path):
            return dict(transient) if isinstance(transient, dict) else None
        try:
            state = self._load_library_state().get("rebuild_in_progress")
            return dict(state) if isinstance(state, dict) else None
        except Exception as e:
            logger.warning("Error loading rebuild state: %s", e)
            return None

    def _save_rebuild_state(self, state: dict[str, Any] | None) -> None:
        """Persist or clear the library-scoped in-place rebuild contract."""
        self._transient_rebuild_state = dict(state) if state is not None else None
        if not self.config_path:
            return
        full_config: dict[str, Any] = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path) as config_file:
                    full_config = json.load(config_file)
            except Exception as e:
                raise RuntimeError(
                    f"Cannot update unreadable configuration {self.config_path}: {e}"
                ) from e
        semantic = full_config.setdefault("semantic_search", {})
        library, identity, _ = self._library_scope()
        library_state = semantic.setdefault("library_states", {}).setdefault(
            identity, {}
        )
        library_state["library_id"] = library.get("library_id", "")
        library_state["library_type"] = library.get("library_type", "user")
        if state is None:
            library_state.pop("rebuild_in_progress", None)
        else:
            library_state["rebuild_in_progress"] = dict(state)
        atomic_write_json(self.config_path, full_config, indent=2)

    def _make_rebuild_state(
        self,
        *,
        fulltext: bool,
        force_clear: bool,
    ) -> dict[str, Any]:
        """Create the compatibility contract carried across rebuild resumes."""
        token = uuid.uuid4().hex
        embedding_identity = getattr(
            self.chroma_client,
            "embedding_identity",
            getattr(self.chroma_client, "embedding_model", "unknown"),
        )
        return {
            "version": 1,
            "phase": "preparing",
            "token": token,
            "marker": f"rebuild-pending:{token}",
            "fulltext": fulltext,
            "force_clear": force_clear,
            "embedding_identity": embedding_identity,
            "library_identity": self.library_identity,
            "index_layout_signature": self._index_layout_signature,
            "content_signature": _CONTENT_CONTRACT_SIGNATURE,
        }

    def _validate_rebuild_state(self, state: dict[str, Any]) -> None:
        """Reject resume under a different vector or content contract."""
        embedding_identity = getattr(
            self.chroma_client,
            "embedding_identity",
            getattr(self.chroma_client, "embedding_model", "unknown"),
        )
        expected = {
            "version": 1,
            "embedding_identity": embedding_identity,
            "library_identity": self.library_identity,
            "index_layout_signature": self._index_layout_signature,
            "content_signature": _CONTENT_CONTRACT_SIGNATURE,
        }
        mismatched = [
            key for key, value in expected.items() if state.get(key) != value
        ]
        marker = state.get("marker")
        if not isinstance(marker, str) or not marker.startswith("rebuild-pending:"):
            mismatched.append("marker")
        if not isinstance(state.get("fulltext"), bool):
            mismatched.append("fulltext")
        if not isinstance(state.get("force_clear"), bool):
            mismatched.append("force_clear")
        if state.get("phase") not in {"preparing", "active"}:
            mismatched.append("phase")
        if mismatched:
            raise RuntimeError(
                "Cannot resume the semantic rebuild because its saved contract "
                f"differs in: {', '.join(sorted(set(mismatched)))}. Start a new "
                "explicitly confirmed force rebuild instead."
            )

    def _save_update_config(
        self,
        last_sync_version: int | None = None,
        indexed_fulltext: bool | None = None,
        indexed_content_signature: str | None = None,
    ) -> None:
        """Save update configuration and completed-corpus state."""
        if not self.config_path:
            return

        config_dir = Path(self.config_path).parent
        config_dir.mkdir(parents=True, exist_ok=True)

        # Never replace an unreadable config with a partial default document.
        full_config = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    full_config = json.load(f)
            except Exception as e:
                raise RuntimeError(f"Cannot update unreadable configuration {self.config_path}: {e}") from e

        # Update semantic search config
        if "semantic_search" not in full_config:
            full_config["semantic_search"] = {}

        full_config["semantic_search"]["update_config"] = self.update_config
        library, identity, is_default = self._library_scope()
        library_states = full_config["semantic_search"].setdefault("library_states", {})
        state = library_states.setdefault(identity, {})
        state["library_id"] = library.get("library_id", "")
        state["library_type"] = library.get("library_type", "user")
        collection_name = getattr(
            getattr(self, "chroma_client", None), "collection_name", None
        )
        if collection_name:
            state["collection_name"] = collection_name
        state["last_update"] = self.update_config.get("last_update")
        if last_sync_version is not None:
            state["last_sync_version"] = int(last_sync_version)
            if is_default:
                full_config["semantic_search"]["last_sync_version"] = int(last_sync_version)
        if indexed_fulltext is not None:
            state["indexed_fulltext"] = indexed_fulltext
            if is_default:
                full_config["semantic_search"]["indexed_fulltext"] = indexed_fulltext
        if indexed_content_signature is not None:
            state["indexed_content_signature"] = indexed_content_signature
            if is_default:
                full_config["semantic_search"]["indexed_content_signature"] = indexed_content_signature

        atomic_write_json(self.config_path, full_config, indent=2)

    def _create_document_text(self, item: dict[str, Any]) -> str:
        """
        Create searchable text from a Zotero item.

        Args:
            item: Zotero item dictionary

        Returns:
            Combined text for embedding
        """
        data = item.get("data", {})
        item_type = data.get("itemType", "")

        # Annotations have no title / creators / abstract — they have
        # ``annotationText`` (the highlighted passage) and an optional
        # ``annotationComment``. The previous "title + creators + abstract"
        # template fell back to ``format_creators([])`` → "No authors
        # listed", which then embedded identically for every annotation in
        # the library, collapsing them all to a single vector and
        # dominating every semantic-search result (#287).
        if item_type == "annotation":
            return self._create_annotation_document_text(data)

        # Keep similarity focused on the paper's subject matter. Bibliographic
        # identifiers, creators, tags, notes, and workflow state remain
        # available as result metadata or through Zotero's exact search tools,
        # but do not influence vector similarity.
        title = data.get("title", "")
        abstract = data.get("abstractNote", "")
        return "\n\n".join(part.strip() for part in (title, abstract) if part and part.strip())

    def _create_annotation_document_text(self, data: dict[str, Any]) -> str:
        """Build the embedding text for an annotation item.

        Combines ``annotationText`` (highlighted passage) and
        ``annotationComment`` (user's commentary), plus any tags. Returns
        the empty string when nothing meaningful is present so the caller
        can decide to skip the item rather than embedding noise.
        """
        parts: list[str] = []
        if highlighted := (data.get("annotationText") or "").strip():
            parts.append(highlighted)
        if comment := (data.get("annotationComment") or "").strip():
            parts.append(comment)
        if tags := data.get("tags"):
            tag_text = " ".join(t.get("tag", "") for t in tags if t.get("tag"))
            if tag_text:
                parts.append(tag_text)
        return " ".join(parts)

    def _create_metadata(self, item: dict[str, Any]) -> dict[str, Any]:
        """
        Create metadata for a Zotero item.

        Args:
            item: Zotero item dictionary

        Returns:
            Metadata dictionary for ChromaDB
        """
        data = item.get("data", {})

        library, identity, _ = self._library_scope()
        metadata = {
            "item_key": item.get("key", ""),
            "library_identity": identity,
            "library_id": library.get("library_id", ""),
            "library_type": library.get("library_type", "user"),
            "item_type": data.get("itemType", ""),
            "title": data.get("title", ""),
            "date": data.get("date", ""),
            "date_added": data.get("dateAdded", ""),
            "date_modified": data.get("dateModified", ""),
            "creators": format_creators(data.get("creators", [])),
            "publication": data.get("publicationTitle", ""),
            "url": data.get("url", ""),
            "doi": data.get("DOI", ""),
            "embedding_metadata_sha256": _embedding_content_hash(self._create_document_text(item).strip()),
        }
        # If fulltext was extracted (or attempted), mark it so incremental
        # updates don't keep re-trying items that failed extraction
        if data.get("fulltext"):
            metadata["has_fulltext"] = True
            if data.get("fulltextSource"):
                metadata["fulltext_source"] = data.get("fulltextSource")
        elif data.get("fulltext_attempted"):
            # Extraction was attempted but failed (timeout, empty, etc.)
            # Mark so we don't retry on every incremental update
            metadata["has_fulltext"] = "failed"
            if data.get("fulltextError"):
                metadata["fulltext_error"] = data["fulltextError"]

        # Record the attachment-key set (local mode only) so update runs can
        # retry a "failed" item once its attachments change — attaching a file
        # does not bump the parent's dateModified.
        if (att_keys := data.get("attachmentKeys")) is not None:
            metadata["attachment_keys"] = att_keys
        if (attachment_signature := data.get("attachmentSignature")) is not None:
            metadata["attachment_signature"] = attachment_signature

        # Add tags as a single string
        if tags := data.get("tags"):
            metadata["tags"] = " ".join([tag.get("tag", "") for tag in tags])
        else:
            metadata["tags"] = ""

        # Add citation key if available
        extra = data.get("extra", "")
        citation_key = ""
        for line in extra.split("\n"):
            if line.lower().startswith(("citation key:", "citationkey:")):
                citation_key = line.split(":", 1)[1].strip()
                break
        metadata["citation_key"] = citation_key

        return metadata

    def should_update_database(self) -> bool:
        """Check if the database should be updated based on configuration."""
        return should_update(self.update_config)

    def _refresh_bibliographic_metadata(
        self,
        items: list[dict[str, Any]],
    ) -> tuple[int, int]:
        """Refresh result metadata without changing embedding state or vectors."""
        get_metadata = getattr(self.chroma_client, "get_document_metadata", None)
        update_metadata = getattr(self.chroma_client, "update_item_metadata", None)
        if not callable(get_metadata) or not callable(update_metadata):
            return 0, 0

        refreshed_items = 0
        refreshed_records = 0
        for item in items:
            item_key = item.get("key", "")
            if not item_key:
                continue
            existing = get_metadata(item_key)
            if not existing:
                continue
            current = self._create_metadata(item)
            updates = {key: current.get(key, "") for key in _IMMEDIATE_METADATA_FIELDS}
            if all(existing.get(key, "") == value for key, value in updates.items()):
                continue
            updated_records = update_metadata(item_key, updates)
            if updated_records:
                refreshed_items += 1
                refreshed_records += updated_records
        return refreshed_items, refreshed_records

    def _get_items_from_source(
        self,
        limit: int | None = None,
        fulltext: bool = False,
        local_scan: bool = False,
        chroma_client: ChromaClient | None = None,
        force_rebuild: bool = False,
        retry_failed_fulltext: bool = False,
        pre_extraction_callback: Callable[[list[dict[str, Any]], set[str], set[str]], None] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get canonical API metadata with optional local full text.

        Full-text extraction requires local mode (ZOTERO_LOCAL=true). Without
        it, this returns API metadata only and never requests cached full text.

        Args:
            limit: Optional limit on number of items
            fulltext: Whether to select and extract a local attachment
            local_scan: Use the filtered local corpus without extracting text
            chroma_client: ChromaDB client to check for existing documents (None to skip checks)
            force_rebuild: Whether to force extraction even if item exists

        Returns:
            List of items in API-compatible format
        """
        if not isinstance(fulltext, bool):
            raise ValueError("fulltext must be true or false")
        if fulltext or local_scan:
            if not is_local_mode():
                raise RuntimeError(
                    "Local semantic indexing requires local mode but ZOTERO_LOCAL is not enabled. "
                    "Set ZOTERO_LOCAL=true or run 'zotero-mcp setup' to enable local mode."
                )
            return self._get_items_from_local_db(
                limit,
                extract_fulltext=fulltext,
                chroma_client=chroma_client,
                force_rebuild=force_rebuild,
                retry_failed_fulltext=retry_failed_fulltext,
                pre_extraction_callback=pre_extraction_callback,
            )
        return self._get_items_from_api(limit)

    def _get_items_from_local_db(
        self,
        limit: int | None = None,
        extract_fulltext: bool = False,
        chroma_client: ChromaClient | None = None,
        force_rebuild: bool = False,
        retry_failed_fulltext: bool = False,
        pre_extraction_callback: Callable[[list[dict[str, Any]], set[str], set[str]], None] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get items from local Zotero database.

        Args:
            limit: Optional limit on number of items
            extract_fulltext: Whether to extract fulltext content
            chroma_client: ChromaDB client to check for existing documents (None to skip checks)
            force_rebuild: Whether to force extraction even if item exists

        Returns:
            List of items in API-compatible format
        """
        logger.info("Fetching items from local Zotero database...")
        self._last_scan_indexable_keys = None
        self._last_scan_attachment_snapshot_complete = True

        try:
            api_metadata_by_key: dict[str, dict[str, Any]] = {}
            self._last_api_metadata_snapshot_complete = False
            try:
                api_metadata_by_key = {
                    item.get("key", ""): item for item in self._get_items_from_api() if item.get("key")
                }
                self._last_api_metadata_snapshot_complete = True
            except Exception as e:
                logger.warning(
                    "Could not load canonical Zotero API metadata for local "
                    "full-text extraction (%s); using reduced SQLite metadata.",
                    e,
                )

            # Load per-run config, including extraction limits and db path if provided
            pdf_max_pages = None
            pdf_timeout = 30
            zotero_db_path = self.db_path  # CLI override takes precedence
            collection_keys = None
            # If semantic_search config file exists, prefer its setting
            try:
                if self.config_path and os.path.exists(self.config_path):
                    with open(self.config_path) as _f:
                        _cfg = json.load(_f)
                        semantic_cfg = _cfg.get("semantic_search", {})
                        extraction_cfg = semantic_cfg.get("extraction", {})
                        pdf_max_pages = extraction_cfg.get("pdf_max_pages")
                        pdf_timeout = extraction_cfg.get("pdf_timeout", 30)
                        collection_keys = semantic_cfg.get("collection_keys")
                        # Use config db_path only if no CLI override
                        if not zotero_db_path:
                            zotero_db_path = semantic_cfg.get("zotero_db_path")
            except Exception:
                pass

            with (
                suppress_stdout(),
                LocalZoteroReader(
                    db_path=zotero_db_path, pdf_max_pages=pdf_max_pages, pdf_timeout=pdf_timeout
                ) as reader,
            ):
                library, _, _ = self._library_scope()
                sqlite_library_id = reader.resolve_library_id(
                    library.get("library_id", ""),
                    library.get("library_type", "user"),
                )
                if sqlite_library_id is None:
                    raise RuntimeError(
                        "The active Zotero library is not present in the local "
                        "SQLite snapshot; refusing an unscoped semantic scan."
                    )
                # Capture the snapshot's full key set on the SAME connection
                # this scan uses. The staleness check after the (potentially
                # long) extraction must compare against what this scan could
                # actually see — a fresh read taken later could already
                # include rows from a WAL checkpoint that landed mid-scan.
                self._last_scan_snapshot_keys = reader.get_all_item_keys(sqlite_library_id)
                # Phase 1: fetch metadata only (fast)
                sys.stderr.write("Scanning local Zotero database for items...\n")
                if collection_keys:
                    sys.stderr.write(f"Filtering to collections: {collection_keys}\n")
                local_items = reader.get_items_with_text(
                    limit=limit,
                    include_fulltext=False,
                    collection_keys=collection_keys,
                    library_id=sqlite_library_id,
                )
                candidate_count = len(local_items)
                sys.stderr.write(f"Found {candidate_count} candidate items.\n")

                self._last_scan_indexable_keys = {it.key for it in local_items}
                if pre_extraction_callback is not None:
                    preview_items = [
                        self._merge_local_fulltext_with_api_metadata(
                            item,
                            api_metadata_by_key.get(item.key),
                            extract_fulltext=False,
                        )
                        for item in local_items
                    ]
                    pre_extraction_callback(
                        preview_items,
                        set(self._last_scan_indexable_keys),
                        set(api_metadata_by_key),
                    )
                try:
                    sys.stderr.write("Extracting content...\n")
                except Exception:
                    pass

                # Phase 2: selectively extract fulltext only when requested
                if extract_fulltext:
                    extracted = 0
                    skipped_existing = 0
                    updated_existing = 0
                    items_to_process = []

                    consecutive_timeouts = 0
                    MAX_CONSECUTIVE_TIMEOUTS = 5
                    _extraction_stopped = False  # Set True when circuit breaker trips

                    total_local = len(local_items)
                    _failed_extractions: list[tuple[str, str]] = []
                    _skipped_failed: list[tuple[str, str]] = []
                    _truncated_pdfs: list[tuple[str, int, int]] = []
                    _deferred_attachments: list[tuple[str, int]] = []

                    # Show startup note
                    try:
                        sys.stderr.write(
                            "\n  Note: Most papers take 1-3 seconds. Some larger or complex PDFs\n"
                            "  may take up to 30 seconds. Password-protected or corrupted files\n"
                            "  will be skipped automatically. The system moves on to the next\n"
                            "  paper if a file can't be processed in time.\n\n"
                        )
                        sys.stderr.flush()
                    except Exception:
                        pass

                    # Temporarily suppress local_db logger to prevent timeout warnings
                    # from disrupting the progress line — we collect them ourselves
                    _local_db_logger = logging.getLogger("zotero_mcp.local_db")
                    _prev_level = _local_db_logger.level
                    _local_db_logger.setLevel(logging.CRITICAL)
                    extraction_eta = _CumulativeETA(total_local)

                    for item_idx, it in enumerate(local_items, 1):
                        # Build display string: Author (Year) — Title
                        title = getattr(it, "title", "") or ""
                        creators = getattr(it, "creators", "") or ""
                        canonical_data = api_metadata_by_key.get(it.key, {}).get("data", {})
                        publication_date = canonical_data.get("date", "")
                        first_author = ""
                        if creators:
                            first_author = creators.split(";")[0].split(",")[0].strip()
                            if first_author:
                                first_author += " et al." if ";" in creators else ""
                        year = _publication_year(publication_date)
                        citation = ""
                        if first_author and year:
                            citation = f"{first_author} ({year}) — "
                        elif first_author:
                            citation = f"{first_author} — "
                        display = f"{citation}{title}"

                        # Single-line progress with \r overwrite
                        # MUST fit within terminal width to prevent wrapping
                        try:
                            status_parts = []
                            if skipped_existing > 0:
                                status_parts.append(f"{skipped_existing} up to date")
                            if extracted > 0:
                                status_parts.append(f"{extracted} extracted")
                            status = f" ({', '.join(status_parts)})" if status_parts else ""
                            eta = _format_eta(extraction_eta.estimate(item_idx - 1))
                            prefix = f"  Processing {item_idx}/{total_local}{status} | ETA {eta} — "
                            line = f"{prefix}{display or 'working...'}"
                            _write_progress_line(sys.stderr, line)
                        except Exception:
                            pass

                        should_extract = True

                        local_attachment_rows = reader.get_fulltext_meta_for_item(it.item_id)
                        local_attachment_keys = {key for key, _path, _ctype in local_attachment_rows}
                        api_attachment_map = self._last_api_attachment_keys_by_parent
                        allowed_attachment_keys: set[str] | None = None
                        if api_attachment_map is not None:
                            api_attachment_keys = api_attachment_map.get(it.key, set())
                            missing_local_keys = api_attachment_keys - local_attachment_keys
                            if missing_local_keys:
                                self._last_scan_attachment_snapshot_complete = False
                                _deferred_attachments.append((display or f"item {it.key}", len(missing_local_keys)))
                                continue
                            # Exclude SQLite rows that the live Zotero API has
                            # already deleted, even before WAL checkpointing.
                            allowed_attachment_keys = local_attachment_keys & api_attachment_keys

                        # Current active attachment-key set, stored in metadata so a
                        # later run can detect attachment changes. Attaching a
                        # file does NOT bump the parent's dateModified, so the
                        # date check alone never clears a "failed" marker.
                        active_attachment_rows = (
                            local_attachment_rows
                            if allowed_attachment_keys is None
                            else reader.get_fulltext_meta_for_item(it.item_id, allowed_attachment_keys)
                        )
                        att_keys = ",".join(sorted(k for k, _p, _c in active_attachment_rows))
                        it._attachment_keys = att_keys
                        if hasattr(reader, "get_attachment_signature"):
                            attachment_signature = (
                                reader.get_attachment_signature(it.item_id)
                                if allowed_attachment_keys is None
                                else reader.get_attachment_signature(it.item_id, allowed_attachment_keys)
                            )
                        else:
                            # Compatibility for custom/legacy readers. The key
                            # set still detects added and removed attachments.
                            attachment_signature = att_keys
                        it._attachment_signature = attachment_signature

                        # CHECK IF ITEM ALREADY EXISTS (unless force_rebuild or no client)
                        if chroma_client and not force_rebuild:
                            # With passage-chunking the stored ids are
                            # "<key>#<n>"; get_document_metadata falls back to
                            # chunk 0 so chunked items are still recognized.
                            existing_metadata = chroma_client.get_document_metadata(it.key)
                            if existing_metadata:
                                chroma_has_fulltext = existing_metadata.get("has_fulltext", False)
                                local_has_fulltext = bool(att_keys)
                                chroma_date = existing_metadata.get("date_modified", "")
                                item_date = canonical_data.get("dateModified") or getattr(it, "date_modified", "") or ""
                                metadata_changed = chroma_date != item_date
                                stored_signature = existing_metadata.get("attachment_signature")
                                signature_changed = (
                                    stored_signature is not None and stored_signature != attachment_signature
                                )
                                layout_changed = self._index_layout_changed(existing_metadata)

                                # Skip if extraction previously failed AND neither the item
                                # nor its attachment set has changed since (handles both a
                                # replaced bad PDF and a PDF newly attached to an item that
                                # was indexed metadata-only)
                                if chroma_has_fulltext == "failed":
                                    if retry_failed_fulltext:
                                        updated_existing += 1
                                        attachment_unchanged = False
                                    else:
                                        stored_att_keys = existing_metadata.get("attachment_keys")
                                        attachment_unchanged = (
                                            stored_signature == attachment_signature
                                            if stored_signature is not None
                                            else stored_att_keys == att_keys
                                        )
                                    if (
                                        not retry_failed_fulltext
                                        and not metadata_changed
                                        and attachment_unchanged
                                        and not layout_changed
                                    ):
                                        # Nothing changed since the failure — don't retry
                                        should_extract = False
                                        skipped_existing += 1
                                        reason = existing_metadata.get("fulltext_error") or (
                                            "The previous attempt returned no usable text; "
                                            "a detailed reason was not recorded."
                                        )
                                        _skipped_failed.append((display or f"item {it.key}", reason))
                                    elif not retry_failed_fulltext:
                                        # Item or its attachments changed since last
                                        # failure (legacy records without attachment_keys
                                        # retry once, then converge) — retry
                                        updated_existing += 1
                                elif chroma_has_fulltext:
                                    # Successful legacy records have no signature.
                                    # Reindex them once to establish a baseline.
                                    if (
                                        metadata_changed
                                        or signature_changed
                                        or stored_signature is None
                                        or layout_changed
                                    ):
                                        updated_existing += 1
                                    else:
                                        should_extract = False
                                        skipped_existing += 1
                                elif metadata_changed or signature_changed or local_has_fulltext or layout_changed:
                                    # Metadata changed, attachment content changed, or
                                    # a metadata-only item gained extractable content.
                                    updated_existing += 1
                                else:
                                    should_extract = False
                                    skipped_existing += 1

                        if should_extract:
                            # Extract fulltext if item doesn't have it yet
                            # (skip if circuit breaker has tripped)
                            if not getattr(it, "fulltext", None) and not _extraction_stopped:
                                text = (
                                    reader.extract_fulltext_for_item(it.item_id)
                                    if allowed_attachment_keys is None
                                    else reader.extract_fulltext_for_item(it.item_id, allowed_attachment_keys)
                                )
                                # Circuit breaker: stop PDF extraction after consecutive timeouts
                                if isinstance(text, tuple) and len(text) == 2 and text[1] == "timeout":
                                    failure_reason = f"PDF extraction timed out after {pdf_timeout} seconds."
                                    it._fulltext_error = failure_reason
                                    _failed_extractions.append((display or f"item {it.key}", failure_reason))
                                    consecutive_timeouts += 1
                                    if consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
                                        logger.warning(
                                            f"Stopping PDF extraction after {MAX_CONSECUTIVE_TIMEOUTS} "
                                            f"consecutive timeouts — remaining items will use metadata only"
                                        )
                                        try:
                                            sys.stderr.write(
                                                f"\n  Warning: PDF extraction stopped after {MAX_CONSECUTIVE_TIMEOUTS} "
                                                f"consecutive timeouts. Remaining items will be indexed with "
                                                f"metadata only (titles, abstracts, authors).\n\n"
                                            )
                                        except Exception:
                                            pass
                                        _extraction_stopped = True
                                    # Don't skip the item — still add it with metadata only
                                    it._fulltext_attempted = True  # Mark so metadata knows extraction was tried
                                else:
                                    # Reset counter on successful extraction
                                    if text:
                                        consecutive_timeouts = 0
                                    if text:
                                        # Support new (text, source) return format
                                        if isinstance(text, tuple) and len(text) == 2:
                                            it.fulltext, it.fulltext_source = text[0], text[1]
                                        else:
                                            it.fulltext = text
                                        details = getattr(reader, "last_extraction_details", None) or {}
                                        it.fulltext_selection_priority = details.get("selection_priority")
                                        page_count = details.get("page_count")
                                        page_cap = details.get("page_cap")
                                        if (
                                            isinstance(page_count, int)
                                            and isinstance(page_cap, int)
                                            and page_count > page_cap
                                        ):
                                            _truncated_pdfs.append(
                                                (
                                                    display or f"item {it.key}",
                                                    page_count,
                                                    page_cap,
                                                )
                                            )
                                    else:
                                        # Extraction returned empty — mark as attempted
                                        it._fulltext_attempted = True
                                        failure_reason = (
                                            "No selected attachment yielded usable text. "
                                            "The file may be missing, unsupported, empty, "
                                            "image-only, encrypted, or corrupt."
                                            if att_keys
                                            else "No candidate full-text attachment was available."
                                        )
                                        it._fulltext_error = failure_reason
                                        _failed_extractions.append((display or f"item {it.key}", failure_reason))
                            extracted += 1
                            items_to_process.append(it)

                            # (progress shown inline above via \r)

                    # Restore local_db logger
                    _local_db_logger.setLevel(_prev_level)

                    # Clear progress line and show extraction summary
                    try:
                        _clear_progress_line(sys.stderr)
                        parts = [f"  Extraction complete: {extracted} items to index"]
                        if skipped_existing > 0:
                            parts.append(f"{skipped_existing} already up to date")
                        sys.stderr.write(", ".join(parts) + "\n")
                        if updated_existing > 0:
                            sys.stderr.write(f"  ({updated_existing} existing items selected for full-text refresh)\n")
                        if _failed_extractions:
                            sys.stderr.write(f"  Full-text extraction failed for {len(_failed_extractions)} item(s):\n")
                            for name, reason in _failed_extractions:
                                sys.stderr.write(f"    - {name}\n")
                                sys.stderr.write(f"      Reason: {reason}\n")
                        if _truncated_pdfs:
                            sys.stderr.write(
                                "  Warning: the PDF page cap truncated full-text "
                                f"extraction for {len(_truncated_pdfs)} item(s):\n"
                            )
                            for name, page_count, page_cap in _truncated_pdfs[:10]:
                                sys.stderr.write(f"    - {name} ({page_count} pages; indexed first {page_cap})\n")
                            if len(_truncated_pdfs) > 10:
                                sys.stderr.write(f"    ... and {len(_truncated_pdfs) - 10} more\n")
                        if _deferred_attachments:
                            sys.stderr.write(
                                f"  Warning: deferred {len(_deferred_attachments)} item(s) "
                                "because active API attachments are not yet visible in "
                                "the local SQLite snapshot:\n"
                            )
                            for name, count in _deferred_attachments[:10]:
                                sys.stderr.write(f"    - {name} ({count} pending attachment(s))\n")
                            if len(_deferred_attachments) > 10:
                                sys.stderr.write(f"    ... and {len(_deferred_attachments) - 10} more\n")
                            sys.stderr.write(
                                "  Their existing index records were left unchanged; the next update will retry them.\n"
                            )
                        if _skipped_failed:
                            sys.stderr.write(
                                f"  {len(_skipped_failed)} item(s) not retried "
                                "because a previous full-text extraction failed:\n"
                            )
                            for name, reason in _skipped_failed[:5]:
                                sys.stderr.write(f"    - {name}\n")
                                sys.stderr.write(f"      Reason: {reason}\n")
                            if len(_skipped_failed) > 5:
                                sys.stderr.write(f"    ... and {len(_skipped_failed) - 5} more\n")
                            sys.stderr.write(
                                "  Retry only these cached failures with:\n"
                                "    zotero-mcp update-db --fulltext "
                                "--retry-failed-fulltext\n"
                            )
                    except Exception:
                        pass

                    # Dispatch the best selected artifacts first. This is a
                    # stable sort, so Zotero's existing order is preserved
                    # within each attachment-preference tier.
                    local_items = _sort_local_items_by_fulltext_priority(items_to_process)
                else:
                    # Skip fulltext extraction for faster processing
                    for it in local_items:
                        it.fulltext = None
                        it.fulltext_source = None

                # Convert to API-compatible format
                api_items = []
                for item in local_items:
                    api_items.append(
                        self._merge_local_fulltext_with_api_metadata(
                            item,
                            api_metadata_by_key.get(item.key),
                            extract_fulltext=extract_fulltext,
                        )
                    )

                logger.info(f"Retrieved {len(api_items)} items from local database")
                return api_items

        except Exception as e:
            self._last_scan_indexable_keys = None
            logger.error(f"Error reading from local database: {e}")
            raise RuntimeError(
                "Local full-text extraction failed; the semantic index was not "
                "updated with metadata-only API records. Fix the local Zotero "
                "database/filesystem error and retry."
            ) from e

    def _merge_local_fulltext_with_api_metadata(
        self,
        item: Any,
        api_item: dict[str, Any] | None,
        *,
        extract_fulltext: bool,
    ) -> dict[str, Any]:
        """Overlay local extraction state onto canonical Zotero API metadata."""
        if api_item:
            merged = dict(api_item)
            data = dict(api_item.get("data", {}))
            merged["data"] = data
            merged["key"] = api_item.get("key") or item.key
        else:
            data = {
                "key": item.key,
                "itemType": getattr(item, "item_type", None) or "journalArticle",
                "title": item.title or "",
                "abstractNote": item.abstract or "",
                "extra": item.extra or "",
                "dateAdded": item.date_added,
                "dateModified": item.date_modified,
                "creators": (self._parse_creators_string(item.creators) if item.creators else []),
            }
            if item.notes:
                data["notes"] = item.notes
            merged = {"key": item.key, "version": 0, "data": data}

        data["fulltext"] = getattr(item, "fulltext", None) or "" if extract_fulltext else ""
        data["fulltextSource"] = getattr(item, "fulltext_source", None) or "" if extract_fulltext else ""
        data["fulltext_attempted"] = getattr(item, "_fulltext_attempted", False)
        if (att := getattr(item, "_attachment_keys", None)) is not None:
            data["attachmentKeys"] = att
        if (signature := getattr(item, "_attachment_signature", None)) is not None:
            data["attachmentSignature"] = signature
        if (error := getattr(item, "_fulltext_error", None)) is not None:
            data["fulltextError"] = error
        return merged

    def _parse_creators_string(self, creators_str: str) -> list[dict[str, str]]:
        """
        Parse creators string from local DB into API format.

        Args:
            creators_str: String like "Smith, John; Doe, Jane"

        Returns:
            List of creator objects
        """
        if not creators_str:
            return []

        creators = []
        for creator in creators_str.split(";"):
            creator = creator.strip()
            if not creator:
                continue

            if "," in creator:
                last, first = creator.split(",", 1)
                creators.append({"creatorType": "author", "firstName": first.strip(), "lastName": last.strip()})
            else:
                creators.append({"creatorType": "author", "name": creator})

        return creators

    def _get_items_from_api(self, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Get canonical item metadata from the Zotero API.

        Args:
            limit: Optional limit on number of items

        Returns:
            List of items from API
        """
        logger.info("Fetching items from Zotero API...")
        self._last_api_attachment_keys_by_parent = None
        attachment_keys_by_parent: dict[str, set[str]] = {}

        # Fetch items in batches to handle large libraries
        batch_size = 100
        start = 0
        all_items = []

        while True:
            batch_params = {"start": start, "limit": batch_size}
            if limit and len(all_items) >= limit:
                break

            try:
                items = self.zotero_client.items(**batch_params)
            except Exception as e:
                if "Connection refused" in str(e):
                    error_msg = (
                        "Cannot connect to Zotero local API. Please ensure:\n"
                        "1. Zotero is running\n"
                        "2. Local API is enabled in Zotero Preferences > Advanced > Enable HTTP server\n"
                        "3. The local API port (default 23119) is not blocked"
                    )
                    raise Exception(error_msg) from e
                else:
                    raise Exception(f"Zotero API connection error: {e}") from e
            if not items:
                break

            for item in items:
                data = item.get("data", {})
                if data.get("itemType") != "attachment":
                    continue
                parent_key = data.get("parentItem")
                attachment_key = item.get("key") or data.get("key")
                if parent_key and attachment_key and not data.get("deleted", False):
                    attachment_keys_by_parent.setdefault(parent_key, set()).add(attachment_key)

            # Child artifacts are represented only through their parent item.
            filtered_items = [
                item
                for item in items
                if item.get("data", {}).get("itemType") not in {"attachment", "note", "annotation"}
            ]

            all_items.extend(filtered_items)
            start += batch_size

            if len(items) < batch_size:
                break

        if limit:
            all_items = all_items[:limit]

        self._last_api_attachment_keys_by_parent = attachment_keys_by_parent

        logger.info(f"Retrieved {len(all_items)} items from API")
        return all_items

    def _get_changed_items_from_api(self, since_version: int) -> tuple[list[dict[str, Any]], set[str] | None]:
        """Fetch only items changed in the Zotero library since a given version.

        Uses pyzotero's `item_versions(since=V)` to discover changed top-level
        item keys.

        Returns:
            (changed_items, all_current_top_level_keys). The second element
            powers deletion detection: any id present in the ChromaDB
            collection but absent from it has been removed from the library.
            It is None when the current library keys could not be enumerated;
            callers must skip deletion in that case.
        """
        logger.info(f"Fetching changed items since library version {since_version}...")
        self._last_api_attachment_keys_by_parent = None
        try:
            changed_versions = self.zotero_client.item_versions(since=since_version) or {}
        except Exception as e:
            raise Exception(f"Failed to fetch item_versions(since={since_version}): {e}") from e

        discovery_complete = True
        changed_keys = set(changed_versions.keys())
        deleted_method = getattr(self.zotero_client, "deleted", None)
        if callable(deleted_method):
            try:
                deleted = deleted_method(since=since_version) or {}
                changed_keys.update(deleted.get("items", []))
            except Exception as e:
                logger.warning(
                    "Failed to fetch deletions since version %s: %s",
                    since_version,
                    e,
                )
                discovery_complete = False

        try:
            current_versions = self.zotero_client.item_versions() or {}
        except Exception as e:
            logger.warning(f"Failed to fetch current item_versions for deletion check: {e}")
            current_versions = None
            discovery_complete = False
        current_keys = set(current_versions.keys()) if current_versions is not None else None

        if not changed_keys:
            return [], current_keys if discovery_complete else None

        changed_items_by_key: dict[str, dict[str, Any]] = {}
        changed_parent_keys: set[str] = set()
        for key in changed_keys:
            try:
                item = self.zotero_client.item(key)
            except Exception as e:
                logger.debug(f"item({key}) failed during incremental fetch: {e}")
                discovery_complete = False
                continue
            if not item:
                continue
            item_type = item.get("data", {}).get("itemType")
            # Don't index attachments/notes as standalone entries; only
            # top-level research items participate in semantic search.
            if item_type == "attachment":
                if parent_key := item.get("data", {}).get("parentItem"):
                    changed_parent_keys.add(parent_key)
                continue
            if item.get("data", {}).get("deleted", False):
                continue
            if item_type in {"note", "annotation"}:
                continue
            changed_items_by_key[key] = item

        # Attachment changes do not reliably bump the parent item's version.
        # Reconcile the parent so an existing full-text record can be refreshed
        # or downgraded when its selected source changes or is deleted.
        for parent_key in changed_parent_keys - changed_items_by_key.keys():
            try:
                parent = self.zotero_client.item(parent_key)
            except Exception as e:
                logger.debug(
                    "item(%s) failed while resolving changed attachment parent: %s",
                    parent_key,
                    e,
                )
                discovery_complete = False
                continue
            if parent:
                changed_items_by_key[parent_key] = parent

        return list(changed_items_by_key.values()), (current_keys if discovery_complete else None)

    def _verify_local_snapshot_version(self, target_sync_version: int) -> int | None:
        """Decide whether the local sqlite snapshot supports promoting the
        API-derived sync watermark.

        The local-extraction scan reads zotero.sqlite with `immutable=1`,
        which cannot see rows still sitting in an un-checkpointed WAL file.
        The API (served by the running Zotero) *does* see them, so its
        library version may cover items the scan never returned. Promoting
        `last_sync_version` in that state makes every later incremental
        update skip those items forever (issue #292).

        Returns:
            `target_sync_version` if every item key known to the API is
            present in the sqlite snapshot, otherwise None (keep the
            previous watermark so the next update re-covers the gap).
        """
        if not getattr(self, "_last_scan_attachment_snapshot_complete", True):
            logger.warning(
                "Active Zotero API attachments are missing from the local "
                "sqlite snapshot; keeping previous sync watermark so the next "
                "update can pick them up."
            )
            return None

        try:
            api_keys = set((self.zotero_client.item_versions() or {}).keys())

            # Prefer the key set captured by the scan's own connection: a
            # fresh read here could already see rows from a WAL checkpoint
            # that landed mid-scan, masking the very staleness we check for.
            snapshot_keys = getattr(self, "_last_scan_snapshot_keys", None)
            if snapshot_keys is None:
                zotero_db_path = self.db_path  # CLI override takes precedence
                if not zotero_db_path and self.config_path and os.path.exists(self.config_path):
                    try:
                        with open(self.config_path) as f:
                            zotero_db_path = json.load(f).get("semantic_search", {}).get("zotero_db_path")
                    except Exception:
                        pass
                with LocalZoteroReader(db_path=zotero_db_path) as reader:
                    library, _, _ = self._library_scope()
                    sqlite_library_id = reader.resolve_library_id(
                        library.get("library_id", ""),
                        library.get("library_type", "user"),
                    )
                    if sqlite_library_id is None:
                        raise RuntimeError(
                            "active library is not present in the local Zotero snapshot"
                        )
                    snapshot_keys = reader.get_all_item_keys(
                        library_id=sqlite_library_id
                    )
        except Exception as e:
            logger.warning(f"Could not verify local snapshot completeness ({e}); keeping previous sync watermark.")
            return None

        missing = api_keys - snapshot_keys
        if missing:
            logger.warning(
                f"{len(missing)} item(s) are visible via the Zotero API but "
                "missing from the local sqlite snapshot (immutable reads "
                "cannot see un-checkpointed WAL data); keeping previous sync "
                "watermark so the next update can pick them up."
            )
            return None
        return target_sync_version

    def _delete_missing_index_items(self, current_item_keys: set[str]) -> int:
        """Delete indexed items absent from an authoritative current-key set."""
        stored_ids = self.chroma_client.get_all_ids()
        stored_item_keys = {doc_id.split("#", 1)[0] for doc_id in stored_ids}
        to_delete_keys = sorted(k for k in (stored_item_keys - current_item_keys) if k)
        if not to_delete_keys:
            return 0

        for key in to_delete_keys:
            if hasattr(self.chroma_client, "delete_item_records"):
                self.chroma_client.delete_item_records(key)
            elif self._chunking_enabled and hasattr(self.chroma_client, "delete_item_chunks"):
                self.chroma_client.delete_item_chunks(key)
            else:
                self.chroma_client.delete_documents([key])
        return len(to_delete_keys)

    def _maintain_existing_fulltext(
        self,
        items: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int, int, bool]:
        """Maintain already-enriched items during a metadata-first update.

        Unchanged attachment signatures preserve documents and vectors while
        refreshing Zotero metadata. Changed signatures trigger extraction and
        normal payload-hash reconciliation, which may upgrade, refresh, or
        legitimately downgrade the item if its source disappeared.
        """
        remaining = []
        preserved_items = 0
        preserved_records = 0
        snapshot_complete = True
        update_item_metadata = getattr(self.chroma_client, "update_item_metadata", None)
        if not callable(update_item_metadata):
            return items, 0, 0, True

        def preserve(item: dict[str, Any]) -> bool:
            nonlocal preserved_items, preserved_records
            current = self._create_metadata(item)
            updated_records = update_item_metadata(
                item.get("key", ""),
                {key: current.get(key, "") for key in _IMMEDIATE_METADATA_FIELDS},
            )
            if not updated_records:
                return False
            preserved_items += 1
            preserved_records += updated_records
            return True

        fulltext_items = []
        for item in items:
            item_key = item.get("key", "")
            existing = self.chroma_client.get_document_metadata(item_key) if item_key else None
            if existing and existing.get("has_fulltext") is True:
                fulltext_items.append((item, existing))
            else:
                remaining.append(item)

        if not fulltext_items:
            return remaining, 0, 0, True

        # Without local attachment access, absence of full text in the API item
        # is not evidence of removal. Preserve the best known representation.
        if not is_local_mode():
            for item, _existing in fulltext_items:
                if not preserve(item):
                    remaining.append(item)
            return remaining, preserved_items, preserved_records, True

        pdf_max_pages = None
        pdf_timeout = 30
        zotero_db_path = self.db_path
        try:
            if self.config_path and os.path.exists(self.config_path):
                with open(self.config_path) as config_file:
                    semantic_config = json.load(config_file).get("semantic_search", {})
                extraction_config = semantic_config.get("extraction", {})
                pdf_max_pages = extraction_config.get("pdf_max_pages")
                pdf_timeout = extraction_config.get("pdf_timeout", 30)
                if not zotero_db_path:
                    zotero_db_path = semantic_config.get("zotero_db_path")
        except Exception:
            pass

        downgraded = []
        truncated_pdfs = []
        with (
            suppress_stdout(),
            LocalZoteroReader(
                db_path=zotero_db_path,
                pdf_max_pages=pdf_max_pages,
                pdf_timeout=pdf_timeout,
            ) as reader,
        ):
            library, _, _ = self._library_scope()
            resolve_library_id = getattr(reader, "resolve_library_id", None)
            sqlite_library_id = (
                resolve_library_id(
                    library.get("library_id", ""),
                    library.get("library_type", "user"),
                )
                if callable(resolve_library_id)
                else None
            )
            if callable(resolve_library_id) and sqlite_library_id is None:
                return remaining, preserved_items, preserved_records, False
            for item, existing in fulltext_items:
                item_key = item.get("key", "")
                local_item = (
                    reader.get_item_by_key(
                        item_key, library_id=sqlite_library_id
                    )
                    if sqlite_library_id is not None
                    else reader.get_item_by_key(item_key)
                )
                if local_item is None:
                    snapshot_complete = False
                    if not preserve(item):
                        remaining.append(item)
                    continue

                try:
                    if self._last_api_attachment_keys_by_parent is not None:
                        api_attachment_keys = set(self._last_api_attachment_keys_by_parent.get(item_key, set()))
                    else:
                        children = self.zotero_client.children(item_key) or []
                        api_attachment_keys = {
                            child.get("key") or child.get("data", {}).get("key")
                            for child in children
                            if child.get("data", {}).get("itemType") == "attachment"
                            and not child.get("data", {}).get("deleted", False)
                        }
                        api_attachment_keys.discard(None)
                except Exception as exc:
                    logger.warning(
                        "Could not verify attachments for %s (%s); preserving "
                        "its existing full text and sync watermark.",
                        item_key,
                        exc,
                    )
                    snapshot_complete = False
                    if not preserve(item):
                        remaining.append(item)
                    continue

                local_attachment_keys = {
                    key for key, _path, _ctype in reader.get_fulltext_meta_for_item(local_item.item_id)
                }
                if api_attachment_keys - local_attachment_keys:
                    snapshot_complete = False
                    if not preserve(item):
                        remaining.append(item)
                    continue

                allowed_attachment_keys = local_attachment_keys & api_attachment_keys
                signature = reader.get_attachment_signature(
                    local_item.item_id,
                    allowed_attachment_keys,
                )
                if existing.get("attachment_signature") == signature:
                    current_metadata_hash = _embedding_content_hash(self._create_document_text(item).strip())
                    metadata_payload_changed = (
                        existing.get("fulltext_source") not in _SELF_CONTAINED_FULLTEXT_SOURCES
                        and existing.get("embedding_metadata_sha256") != current_metadata_hash
                    )
                    if not metadata_payload_changed:
                        if not preserve(item):
                            remaining.append(item)
                        continue

                extracted = reader.extract_fulltext_for_item(
                    local_item.item_id,
                    allowed_attachment_keys,
                )
                if isinstance(extracted, tuple) and len(extracted) == 2 and extracted[1] == "timeout":
                    snapshot_complete = False
                    if not preserve(item):
                        remaining.append(item)
                    continue

                refreshed = {
                    **item,
                    "data": dict(item.get("data", {})),
                }
                data = refreshed["data"]
                data["attachmentKeys"] = ",".join(sorted(allowed_attachment_keys))
                data["attachmentSignature"] = signature
                if extracted:
                    data["fulltext"], data["fulltextSource"] = extracted
                    details = getattr(reader, "last_extraction_details", None) or {}
                    page_count = details.get("page_count")
                    page_cap = details.get("page_cap")
                    if isinstance(page_count, int) and isinstance(page_cap, int) and page_count > page_cap:
                        truncated_pdfs.append(
                            (
                                data.get("title") or item_key,
                                page_count,
                                page_cap,
                            )
                        )
                else:
                    data["fulltext"] = ""
                    data["fulltextSource"] = ""
                    data["fulltext_attempted"] = True
                    data["fulltextError"] = (
                        "No selected attachment yielded usable text."
                        if allowed_attachment_keys
                        else "No candidate full-text attachment was available."
                    )
                    downgraded.append(data.get("title") or item_key)
                remaining.append(refreshed)

        if downgraded:
            sys.stderr.write(
                f"\nFull-text sources were removed or became unusable for "
                f"{len(downgraded)} changed item(s); rebuilding them from "
                "metadata only:\n"
            )
            for title in downgraded[:10]:
                sys.stderr.write(f"  - {title}\n")
        if truncated_pdfs:
            sys.stderr.write(
                f"\nWarning: the PDF page cap truncated full-text extraction "
                f"for {len(truncated_pdfs)} changed item(s):\n"
            )
            for title, page_count, page_cap in truncated_pdfs[:10]:
                sys.stderr.write(f"  - {title} ({page_count} pages; indexed first {page_cap})\n")

        return remaining, preserved_items, preserved_records, snapshot_complete

    def _prepare_index_records(
        self,
        items: list[dict[str, Any]],
        *,
        force_rebuild: bool = False,
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, int],
        _PreparedIndexBatch,
    ]:
        """Prepare ChromaDB records without embedding or writing them."""
        prepared = self._prepare_item_batch(items)
        prepared = self._partition_reusable_embeddings(
            prepared,
            force_rebuild=force_rebuild,
        )
        records = [
            {"id": doc_id, "document": document, "metadata": metadata}
            for doc_id, document, metadata in zip(
                prepared.ids,
                prepared.documents,
                prepared.metadatas,
                strict=True,
            )
        ]
        stats = {
            "processed": prepared.stats["processed"],
            "skipped": prepared.stats["skipped"],
            "errors": prepared.stats["errors"],
            "reused_embeddings": prepared.stats["reused_embeddings"],
        }
        return records, stats, prepared

    def _submit_openai_batch_index(
        self,
        items: list[dict[str, Any]],
        force_full_rebuild: bool,
        target_sync_version: int | None,
        stats: dict[str, Any],
        indexed_fulltext_state: bool | None = None,
    ) -> dict[str, Any]:
        """Prepare records and submit asynchronous OpenAI embedding batches."""
        if indexed_fulltext_state is None:
            indexed_fulltext_state = self._active_fulltext
        records, prepare_stats, prepared = self._prepare_index_records(
            items,
            force_rebuild=force_full_rebuild,
        )
        stats["processed_items"] += prepare_stats["processed"]
        stats["skipped_items"] += prepare_stats["skipped"]
        stats["errors"] += prepare_stats["errors"]
        stats["reused_embeddings"] = stats.get("reused_embeddings", 0) + (prepare_stats["reused_embeddings"])

        expected_ids_by_item: dict[str, list[str]] = {}
        for doc_id in prepared.expected_ids:
            parent = doc_id.split("#", 1)[0]
            expected_ids_by_item.setdefault(parent, []).append(doc_id)
        expected_ids_by_item = {parent: sorted(ids) for parent, ids in expected_ids_by_item.items()}
        get_item_hashes = getattr(self.chroma_client, "get_item_embedding_hashes", None)
        baseline_embedding_hashes_by_item = (
            {parent: get_item_hashes(parent) for parent in expected_ids_by_item} if callable(get_item_hashes) else {}
        )
        get_item_record_hashes = getattr(self.chroma_client, "get_item_record_hashes", None)
        baseline_record_hashes_by_item = (
            {parent: get_item_record_hashes(parent) for parent in expected_ids_by_item}
            if callable(get_item_record_hashes)
            else {}
        )

        if not records:
            if prepared.metadata_only_ids:
                self.chroma_client.update_metadatas(
                    prepared.metadata_only_ids,
                    prepared.metadata_only_metadatas,
                )
            stats["batch_submitted"] = False
            stats["metadata_only"] = bool(prepared.metadata_only_ids)
            if hasattr(self.chroma_client, "reconcile_item_records"):
                for parent, expected_ids in expected_ids_by_item.items():
                    self.chroma_client.reconcile_item_records(
                        parent,
                        set(expected_ids),
                    )
            self.update_config["last_update"] = datetime.now().isoformat()
            completed_sync_version = target_sync_version if stats["errors"] == 0 else None
            self._save_update_config(
                last_sync_version=completed_sync_version,
                indexed_fulltext=(indexed_fulltext_state if completed_sync_version is not None else None),
                indexed_content_signature=(_CONTENT_CONTRACT_SIGNATURE if completed_sync_version is not None else None),
            )
            return stats

        submitted_parent_keys = {
            (record.get("metadata") or {}).get("parent_item_key")
            or record["id"].split("#", 1)[0]
            for record in records
        }
        if target_sync_version is None:
            raise RuntimeError(
                "OpenAI Batch submission requires a verified Zotero library "
                "version so delayed imports cannot overwrite newer source data"
            )
        model_name = self.chroma_client.embedding_config.get("model_name", "text-embedding-3-small")
        manifest = openai_batch.submit_embedding_batches(
            records=records,
            model_name=model_name,
            embedding_config=self.chroma_client.embedding_config,
            config_path=self.config_path,
            force_full_rebuild=force_full_rebuild,
            target_sync_version=target_sync_version,
            fulltext=indexed_fulltext_state,
            content_signature=_CONTENT_CONTRACT_SIGNATURE,
            expected_ids_by_item=expected_ids_by_item,
            baseline_embedding_hashes_by_item=(baseline_embedding_hashes_by_item),
            baseline_record_hashes_by_item=baseline_record_hashes_by_item,
            metadata_only_records=[
                {"id": doc_id, "metadata": metadata}
                for doc_id, metadata in zip(
                    prepared.metadata_only_ids,
                    prepared.metadata_only_metadatas,
                    strict=True,
                )
            ],
        )
        stats["batch_submitted"] = True
        stats["batch_run_id"] = manifest["run_id"]
        stats["batch_manifest"] = manifest["manifest_path"]
        stats["batch_ids"] = [batch["batch_id"] for batch in manifest.get("batches", [])]
        existing_parent_keys = {
            parent
            for parent in submitted_parent_keys
            if baseline_embedding_hashes_by_item.get(parent)
        }
        stats["submitted_items"] = len(submitted_parent_keys)
        stats["submitted_records"] = len(records)
        stats["estimated_updated_items"] = len(existing_parent_keys)
        stats["estimated_added_items"] = len(
            submitted_parent_keys - existing_parent_keys
        )
        return stats

    def _run_orphan_segment_cleanup(self, stats: dict[str, Any]) -> None:
        """Run lock-protected Chroma storage maintenance and record its result."""
        prune_orphans = getattr(
            self.chroma_client,
            "prune_orphan_segment_directories",
            None,
        )
        if not callable(prune_orphans):
            return
        cleanup = prune_orphans()
        stats["orphan_segment_directories_pruned"] = int(cleanup.get("removed_directories", 0))
        stats["orphan_segment_bytes_pruned"] = int(cleanup.get("removed_bytes", 0))
        stats["orphan_segment_cleanup_errors"] = int(cleanup.get("errors", 0))

    def update_database(
        self,
        force_full_rebuild: bool = False,
        force_clear: bool = False,
        limit: int | None = None,
        fulltext: bool | None = None,
        use_openai_batch: bool | None = None,
        embedding_concurrency: int = 1,
        retry_failed_fulltext: bool = False,
    ) -> dict[str, Any]:
        """
        Update the semantic search database with Zotero items.

        Args:
            force_full_rebuild: Whether to rebuild the entire database
            force_clear: Clear the live collection before a forced realtime
                rebuild instead of replacing items as embeddings complete.
            limit: Limit number of items to process (for testing)
            fulltext: Whether to select and extract one local attachment.
                False indexes API title and abstract only. None uses the
                configured setting, which defaults to false.
            use_openai_batch: Override for OpenAI Batch API indexing. None
                uses `semantic_search.openai_batch.enabled`.
            embedding_concurrency: Number of realtime OpenAI-compatible
                embedding batches to run concurrently. ChromaDB reads and
                writes remain sequential. Defaults to 1.
            retry_failed_fulltext: Retry cached local full-text extraction
                failures without rebuilding unaffected items.

        Returns:
            Update statistics
        """
        logger.info("Starting database update...")
        start_time = datetime.now()

        stats = {
            "total_items": 0,
            "processed_items": 0,
            "added_items": 0,
            "updated_items": 0,
            "reused_embeddings": 0,
            "recovered_items": 0,
            "skipped_items": 0,
            "deleted_items": 0,
            "orphan_segment_directories_pruned": 0,
            "orphan_segment_bytes_pruned": 0,
            "orphan_segment_cleanup_errors": 0,
            "errors": 0,
            "start_time": start_time.isoformat(),
            "duration": None,
        }
        interrupt_controller: _UpdateInterruptController | None = None
        rebuild_state: dict[str, Any] | None = None

        # Guard against concurrent rebuilds: the MCP server auto-launches
        # update_database on startup while the user may also run
        # `zotero-mcp update-db` manually. A cross-process flock avoids
        # double work and potential ChromaDB corruption.
        lock_path = Path.home() / ".config" / "zotero-mcp" / "update.lock"
        lock_cm = _acquire_update_lock(lock_path)
        acquired = lock_cm.__enter__()
        if not acquired:
            lock_cm.__exit__(None, None, None)
            holder_pid, holder_alive = read_lock_holder(lock_path)
            if holder_pid and not holder_alive:
                logger.warning(
                    "Update lock at %s is held by dead pid %s (stale). "
                    "the kernel should have released it; check the filesystem "
                    "if this persists.",
                    lock_path,
                    holder_pid,
                )
            else:
                logger.warning(
                    "Another semantic-search update is already running "
                    "(lock held at %s by pid %s); skipping this invocation. "
                    "This is expected when the MCP server's background sync is "
                    "active.",
                    lock_path,
                    holder_pid if holder_pid else "unknown",
                )
            stats["duration"] = "0:00:00"
            stats["skipped_reason"] = "another_update_in_progress"
            return stats

        try:
            self._run_orphan_segment_cleanup(stats)
            saved_rebuild_state = self._load_rebuild_state()
            resuming_rebuild = (
                saved_rebuild_state is not None and not force_full_rebuild
            )
            complete_rebuild = force_full_rebuild or resuming_rebuild
            if resuming_rebuild:
                self._validate_rebuild_state(saved_rebuild_state)
                fulltext = bool(saved_rebuild_state["fulltext"])
                rebuild_state = saved_rebuild_state
                stats["resumed_rebuild"] = True
                sys.stderr.write(
                    "Resuming the interrupted force rebuild; completed items "
                    "will reuse their current vectors.\n"
                )
            if fulltext is None:
                fulltext = self._load_fulltext_setting()
            if not isinstance(fulltext, bool):
                raise ValueError("fulltext must be true or false")
            extract_fulltext = fulltext
            self._active_fulltext = fulltext
            stats["fulltext"] = fulltext
            indexed_fulltext = self._load_indexed_fulltext()
            indexed_content_signature = self._load_indexed_content_signature()
            collection_has_items = int(self.chroma_client.get_collection_info().get("count", 0)) > 0
            indexed_fulltext_state = (
                fulltext if complete_rebuild or not collection_has_items else indexed_fulltext is True or fulltext
            )
            content_mismatch = (
                not complete_rebuild
                and collection_has_items
                and indexed_content_signature != _CONTENT_CONTRACT_SIGNATURE
            )
            if content_mismatch:
                previous = indexed_content_signature or "legacy/unknown"
                sys.stderr.write(
                    f"WARNING: the existing index uses semantic content contract "
                    f"{previous}; the current contract is "
                    f"{_CONTENT_CONTRACT_SIGNATURE}. This incremental run will "
                    "not re-embed unchanged items. Use an explicitly confirmed "
                    "force rebuild to migrate the complete index.\n"
                )
            use_openai_batch = self._resolve_openai_batch_enabled(use_openai_batch)
            if resuming_rebuild:
                use_openai_batch = False
            configured_collection_keys = None
            try:
                if self.config_path and os.path.exists(self.config_path):
                    with open(self.config_path) as config_file:
                        configured_collection_keys = (
                            json.load(config_file).get("semantic_search", {}).get("collection_keys")
                        )
            except Exception:
                pass
            use_local_source = bool(extract_fulltext or configured_collection_keys)
            if force_clear and not force_full_rebuild:
                raise ValueError("force_clear requires force_full_rebuild")
            if complete_rebuild and limit is not None:
                raise ValueError(
                    "limit cannot be combined with a complete rebuild because "
                    "a partial scan cannot replace the complete index"
                )
            if force_clear and use_openai_batch:
                raise ValueError(
                    "force_clear cannot be combined with OpenAI Batch mode; "
                    "use realtime embeddings for an immediate clear"
                )
            if embedding_concurrency < 1:
                raise ValueError("embedding_concurrency must be at least 1")
            if embedding_concurrency > 1:
                if use_openai_batch:
                    raise ValueError("embedding_concurrency cannot be combined with OpenAI Batch mode")
                if self.chroma_client.embedding_model != "openai":
                    raise ValueError("embedding_concurrency above 1 requires realtime OpenAI-compatible embeddings")
                stats["embedding_concurrency"] = embedding_concurrency
                logger.info(
                    "Using %s concurrent realtime embedding batches",
                    embedding_concurrency,
                )

            if complete_rebuild and not use_openai_batch:
                pending_mismatch = bool(
                    getattr(self.chroma_client, "_pending_embedding_mismatch", False)
                )
                rebuild_force_clear = (
                    bool(rebuild_state["force_clear"])
                    if rebuild_state is not None
                    else force_clear
                )
                if pending_mismatch and not rebuild_force_clear:
                    raise RuntimeError(
                        "The embedding model or collection metric changed. An "
                        "in-place rebuild cannot mix incompatible vectors; start "
                        "a new confirmed rebuild with --force-clear."
                    )
                if rebuild_state is None:
                    rebuild_state = self._make_rebuild_state(
                        fulltext=fulltext,
                        force_clear=force_clear,
                    )
                    self._save_rebuild_state(rebuild_state)
                if rebuild_state["phase"] == "preparing":
                    if rebuild_state["force_clear"]:
                        logger.warning("Force clearing semantic index before rebuild")
                        self.chroma_client.reset_collection()
                    else:
                        invalidated = self.chroma_client.invalidate_embedding_hashes(
                            rebuild_state["marker"]
                        )
                        stats["invalidated_records"] = invalidated
                    rebuild_state = {**rebuild_state, "phase": "active"}
                    self._save_rebuild_state(rebuild_state)

            # Decide whether to use since-based incremental ingest.
            # Incremental requires: not a forced rebuild, not a local-extraction
            # run (incremental path covers web-API metadata and optionally
            # fulltext only), not a test limit, and a known prior sync version.
            last_sync_version = self._load_last_sync_version() if not complete_rebuild else 0
            use_incremental = (
                not complete_rebuild and not use_local_source and limit is None and last_sync_version > 0
            )

            target_sync_version: int | None = None
            all_items: list[dict[str, Any]] = []
            local_snapshot_complete = False
            if use_incremental:
                try:
                    target_sync_version = self.zotero_client.last_modified_version()
                except Exception as e:
                    logger.warning(f"last_modified_version() failed, falling back to full scan: {e}")
                    use_incremental = False

            if use_incremental and target_sync_version == last_sync_version:
                # No changes since last sync; skip ingest but still touch last_update
                try:
                    sys.stderr.write(
                        f"\nLibrary unchanged since last sync (version {last_sync_version}); no items to reindex.\n"
                    )
                except Exception:
                    pass
                self.update_config["last_update"] = datetime.now().isoformat()
                self._save_update_config(
                    last_sync_version=target_sync_version,
                    indexed_fulltext=indexed_fulltext_state,
                    indexed_content_signature=(None if content_mismatch else _CONTENT_CONTRACT_SIGNATURE),
                )
                end_time = datetime.now()
                stats["duration"] = str(end_time - start_time)
                stats["end_time"] = end_time.isoformat()
                return stats

            if use_incremental:
                all_items, current_library_keys = self._get_changed_items_from_api(since_version=last_sync_version)
                # Delete collection entries that are no longer present in the
                # library. Map any chunk ids (``<key>#<n>``) back to item keys
                # so deletion works identically whether or not chunking is on.
                if current_library_keys is None:
                    logger.warning(
                        "Skipping deletion pass and keeping the previous sync "
                        "watermark because incremental discovery was incomplete."
                    )
                    target_sync_version = None
                else:
                    try:
                        deleted = self._delete_missing_index_items(current_library_keys)
                        if deleted:
                            stats["deleted_items"] = deleted
                            try:
                                sys.stderr.write(f"\nDeleted {deleted} items no longer present in Zotero.\n")
                            except Exception:
                                pass
                    except Exception as e:
                        raise RuntimeError(
                            "Pruning removed semantic-index items failed; later refresh phases were not started"
                        ) from e
            else:
                # Full scan: bootstrap or forced rebuild.
                # Capture the library version BEFORE scanning so any changes
                # made during the scan will be picked up by the next
                # incremental run. Skipping this after a force_full_rebuild
                # would leave last_sync_version stale and the next
                # incremental run would miss items that haven't changed
                # since the old watermark (because they were just deleted
                # along with the collection).
                try:
                    target_sync_version = self.zotero_client.last_modified_version()
                except Exception as e:
                    logger.warning(f"last_modified_version() failed: {e}")
                    target_sync_version = None
                if use_local_source:
                    self._last_scan_indexable_keys = None
                local_prephase_complete = False

                def run_local_prephase(
                    preview_items: list[dict[str, Any]],
                    indexable_keys: set[str],
                    api_item_keys: set[str],
                ) -> None:
                    nonlocal local_prephase_complete
                    prune_keys: set[str] | None = None
                    if (
                        limit is None
                        and target_sync_version is not None
                        and self._verify_local_snapshot_version(target_sync_version) is not None
                    ):
                        prune_keys = indexable_keys
                    elif getattr(self, "_last_api_metadata_snapshot_complete", False):
                        # Even when SQLite lags, the complete API parent-key
                        # set safely identifies true Zotero deletions. It does
                        # not enforce local collection/dedup filters until the
                        # local snapshot itself is verified.
                        prune_keys = api_item_keys

                    if prune_keys is not None and not force_clear:
                        deleted = self._delete_missing_index_items(prune_keys)
                        if deleted:
                            stats["deleted_items"] += deleted
                            sys.stderr.write(
                                f"\nDeleted {deleted} items no longer present in the current Zotero corpus.\n"
                            )

                    refreshed_items, refreshed_records = self._refresh_bibliographic_metadata(preview_items)
                    if refreshed_items:
                        stats["metadata_refreshed_items"] = stats.get("metadata_refreshed_items", 0) + refreshed_items
                        stats["metadata_refreshed_records"] = (
                            stats.get("metadata_refreshed_records", 0) + refreshed_records
                        )
                        sys.stderr.write(
                            f"\nRefreshed bibliographic metadata for "
                            f"{refreshed_items} existing item(s) before "
                            "full-text extraction.\n"
                        )
                    local_prephase_complete = True

                all_items = self._get_items_from_source(
                    limit=limit,
                    fulltext=fulltext,
                    local_scan=bool(configured_collection_keys),
                    chroma_client=self.chroma_client if not complete_rebuild else None,
                    force_rebuild=complete_rebuild,
                    retry_failed_fulltext=retry_failed_fulltext,
                    pre_extraction_callback=(run_local_prephase if use_local_source else None),
                )
                # The local-extraction scan may lag behind the API version
                # captured above (immutable sqlite reads skip WAL contents);
                # only promote the watermark if the snapshot was complete.
                if use_local_source and target_sync_version is not None:
                    verified_sync_version = self._verify_local_snapshot_version(target_sync_version)
                    local_snapshot_complete = verified_sync_version is not None
                    target_sync_version = verified_sync_version

                # Local full-text scans return only new/changed items for
                # embedding, so reconcile against the complete indexable key
                # set captured before that filtering. Never prune a partial,
                # stale, or unverifiable sqlite snapshot.
                if (
                    use_local_source
                    and not complete_rebuild
                    and limit is None
                    and local_snapshot_complete
                    and self._last_scan_indexable_keys is not None
                    and not local_prephase_complete
                ):
                    try:
                        deleted = self._delete_missing_index_items(self._last_scan_indexable_keys)
                        if deleted:
                            stats["deleted_items"] = deleted
                            try:
                                sys.stderr.write(
                                    f"\nDeleted {deleted} items no longer present in the local Zotero corpus.\n"
                                )
                            except Exception:
                                pass
                    except Exception as e:
                        logger.warning(f"Local deletion pass failed: {e}")
                        target_sync_version = None

                # API full scans (bootstrap, watermark recovery, and similar
                # non-incremental runs) are authoritative too. Prune before
                # embedding so removed records never wait behind a long queue.
                if not use_local_source and not complete_rebuild and limit is None:
                    try:
                        deleted = self._delete_missing_index_items(
                            {item.get("key", "") for item in all_items if item.get("key")}
                        )
                        if deleted:
                            stats["deleted_items"] = deleted
                            sys.stderr.write(f"\nDeleted {deleted} items no longer present in Zotero.\n")
                    except Exception as e:
                        raise RuntimeError(
                            "Pruning removed semantic-index items failed; later refresh phases were not started"
                        ) from e

                # A forced scan also has an authoritative corpus view. Prune
                # genuinely removed items from the live index before any
                # replacement embeddings are written. Deferred local items are
                # retained because the complete indexable-key set includes
                # them; unverifiable SQLite snapshots never prune.
                force_prune_keys: set[str] | None = None
                if complete_rebuild and limit is None:
                    if use_local_source:
                        if (
                            local_snapshot_complete
                            and self._last_scan_indexable_keys is not None
                            and not local_prephase_complete
                        ):
                            force_prune_keys = self._last_scan_indexable_keys
                    else:
                        force_prune_keys = {item.get("key", "") for item in all_items if item.get("key")}
                if force_prune_keys is not None:
                    try:
                        deleted = self._delete_missing_index_items(force_prune_keys)
                        if deleted:
                            stats["deleted_items"] = deleted
                            sys.stderr.write(
                                f"\nDeleted {deleted} items no longer present in the current Zotero corpus.\n"
                            )
                    except Exception as e:
                        raise RuntimeError(
                            "Pruning removed semantic-index items failed; later refresh phases were not started"
                        ) from e

            preserved_fulltext_items = 0
            if not extract_fulltext and not complete_rebuild and all_items:
                (
                    all_items,
                    preserved_fulltext_items,
                    preserved_fulltext_records,
                    maintenance_snapshot_complete,
                ) = self._maintain_existing_fulltext(all_items)
                if not maintenance_snapshot_complete:
                    target_sync_version = None
                if preserved_fulltext_items:
                    stats["processed_items"] += preserved_fulltext_items
                    stats["updated_items"] += preserved_fulltext_items
                    stats["reused_embeddings"] += preserved_fulltext_records
                    stats["preserved_fulltext_items"] = preserved_fulltext_items
                    sys.stderr.write(
                        f"\nPreserved unchanged indexed full text for "
                        f"{preserved_fulltext_items} changed item(s); "
                        "refreshed Zotero metadata only.\n"
                    )

            metadata_refreshed_items, metadata_refreshed_records = self._refresh_bibliographic_metadata(all_items)
            if metadata_refreshed_items:
                stats["metadata_refreshed_items"] = stats.get("metadata_refreshed_items", 0) + metadata_refreshed_items
                stats["metadata_refreshed_records"] = (
                    stats.get("metadata_refreshed_records", 0) + metadata_refreshed_records
                )
                sys.stderr.write(
                    f"\nRefreshed bibliographic metadata for "
                    f"{metadata_refreshed_items} existing item(s) before "
                    "embedding.\n"
                )

            stats["total_items"] = len(all_items) + preserved_fulltext_items
            logger.info(f"Found {stats['total_items']} items to process")

            if (
                complete_rebuild
                and use_local_source
                and limit is None
                and not local_snapshot_complete
            ):
                message = (
                    "Cannot complete a full-text rebuild because the local "
                    "Zotero SQLite snapshot could not be verified as complete. "
                    "The current index was left active; retry after Zotero "
                    "checkpoints its WAL data."
                )
                logger.warning(message)
                stats["errors"] += 1
                stats["error"] = message
                stats["rebuild_in_progress"] = True
                end_time = datetime.now()
                stats["duration"] = str(end_time - start_time)
                stats["end_time"] = end_time.isoformat()
                return stats

            if use_openai_batch:
                stats["batch_mode"] = True
                try:
                    sys.stderr.write(f"\nSubmitting {len(all_items)} items to OpenAI Batch API...\n")
                    sys.stderr.flush()
                except Exception:
                    pass
                stats = self._submit_openai_batch_index(
                    all_items,
                    force_full_rebuild=force_full_rebuild,
                    target_sync_version=target_sync_version,
                    indexed_fulltext_state=indexed_fulltext_state,
                    stats=stats,
                )
                try:
                    if stats.get("batch_submitted"):
                        batch_ids = ", ".join(stats.get("batch_ids", []))
                        sys.stderr.write(
                            "  Submitted OpenAI embedding batch"
                            f"{'es' if len(stats.get('batch_ids', [])) != 1 else ''}: {batch_ids}\n"
                        )
                        sys.stderr.write("  Run 'zotero-mcp openai-batch-status' to check progress.\n")
                        sys.stderr.write("  Run 'zotero-mcp openai-batch-import' after the batch completes.\n")
                    else:
                        sys.stderr.write("  No embedding batch needed; all vectors were reused.\n")
                except Exception:
                    pass
                end_time = datetime.now()
                stats["duration"] = str(end_time - start_time)
                stats["end_time"] = end_time.isoformat()
                return stats

            # User-friendly progress reporting
            total = len(all_items)
            try:
                sys.stderr.write(f"\nIndexing {total} items...\n\n")
                sys.stderr.flush()
            except Exception:
                pass

            seen_items = 0
            indexing_eta = _MedianETA(
                total,
                parallelism=embedding_concurrency,
            )
            _failed_docs: list[_PreparedIndexBatch] = []
            failed_concurrent_batches: list[_PreparedIndexBatch] = []
            interrupt_controller = _UpdateInterruptController()
            interrupt_controller.install()
            stop_pipeline = interrupt_controller.stop_requested
            set_cancel_event = getattr(
                self.chroma_client,
                "set_embedding_cancel_event",
                None,
            )
            if set_cancel_event is not None:
                set_cancel_event(stop_pipeline)

            def report_item_progress(item: dict[str, Any]) -> None:
                nonlocal seen_items
                seen_items += 1
                title = item.get("data", {}).get("title", "")
                pct = int(seen_items / total * 100) if total else 0
                eta = _format_eta(indexing_eta.estimate(seen_items))
                try:
                    _write_progress_line(
                        sys.stderr,
                        f"  [{pct:3d}%] {seen_items}/{total} finished | ETA {eta} | Last: {title or 'untitled item'}",
                    )
                except Exception:
                    pass

            def accumulate_batch_stats(batch_stats: dict[str, int]) -> None:
                stats["processed_items"] += batch_stats["processed"]
                stats["added_items"] += batch_stats["added"]
                stats["updated_items"] += batch_stats["updated"]
                stats["reused_embeddings"] += batch_stats.get("reused_embeddings", 0)
                stats["skipped_items"] += batch_stats["skipped"]
                stats["errors"] += batch_stats["errors"]

                logger.info(
                    f"Processed {seen_items}/{total} items (added: {stats['added_items']}, skipped: {stats['skipped_items']})"
                )

            if embedding_concurrency == 1:
                for item in all_items:
                    if stop_pipeline.is_set():
                        break
                    item_started = time.monotonic()
                    batch_stats = self._process_item_batch(
                        [item],
                        False,
                        _failed_docs,
                        item_atomic_rebuild=complete_rebuild,
                    )
                    indexing_eta.record(time.monotonic() - item_started)
                    report_item_progress(item)
                    accumulate_batch_stats(batch_stats)
            else:
                queue_depth = embedding_concurrency * 2
                work_queue: Queue[Any] = Queue(maxsize=queue_depth)
                result_queue: Queue[Any] = Queue(maxsize=queue_depth)
                chroma_access_lock = threading.Lock()
                work_done = object()
                worker_done = object()
                producer_errors: list[Exception] = []

                def put_with_backpressure(queue: Queue[Any], value: Any) -> bool:
                    while not stop_pipeline.is_set():
                        try:
                            queue.put(value, timeout=0.1)
                            return True
                        except Full:
                            continue
                    return False

                def prepare_entries() -> None:
                    try:
                        for item in all_items:
                            prepared = self._prepare_item_batch([item])
                            with chroma_access_lock:
                                prepared = self._partition_reusable_embeddings(
                                    prepared,
                                    force_rebuild=False,
                                    item_atomic_rebuild=complete_rebuild,
                                )
                            if not put_with_backpressure(
                                work_queue,
                                (item, prepared),
                            ):
                                return
                    except Exception as e:
                        producer_errors.append(e)
                    finally:
                        for _ in range(embedding_concurrency):
                            if not put_with_backpressure(work_queue, work_done):
                                break

                def embedding_worker() -> None:
                    try:
                        while not stop_pipeline.is_set():
                            try:
                                job = work_queue.get(timeout=0.1)
                            except Empty:
                                continue
                            if job is work_done:
                                return
                            item, prepared = job
                            embeddings = None
                            error = None
                            item_started = time.monotonic()
                            if prepared.documents:
                                try:
                                    embeddings = self.chroma_client.embed_documents(prepared.documents)
                                except Exception as e:
                                    error = e
                            duration = time.monotonic() - item_started
                            if not put_with_backpressure(
                                result_queue,
                                (
                                    item,
                                    prepared,
                                    embeddings,
                                    error,
                                    duration,
                                ),
                            ):
                                return
                    finally:
                        put_with_backpressure(result_queue, worker_done)

                with ThreadPoolExecutor(
                    max_workers=embedding_concurrency,
                    thread_name_prefix="zotero-embedding",
                ) as executor:
                    workers = [executor.submit(embedding_worker) for _ in range(embedding_concurrency)]
                    producer = threading.Thread(
                        target=prepare_entries,
                        name="zotero-embedding-producer",
                        daemon=True,
                    )
                    producer.start()

                    completed_workers = 0
                    try:
                        while completed_workers < embedding_concurrency and not stop_pipeline.is_set():
                            try:
                                result = result_queue.get(timeout=0.1)
                            except Empty:
                                continue
                            if result is worker_done:
                                completed_workers += 1
                                continue

                            (
                                item,
                                prepared,
                                embeddings,
                                error,
                                item_duration,
                            ) = result
                            commit_started = time.monotonic()
                            if error is None:
                                try:
                                    with chroma_access_lock:
                                        self._commit_prepared_batch(
                                            prepared,
                                            force_rebuild=False,
                                            embeddings=embeddings,
                                            replace_item_records=complete_rebuild,
                                        )
                                except Exception as e:
                                    error = e

                            if error is not None:
                                logger.warning(
                                    "Concurrent embedding entry failed (%s), saving it for a sequential retry",
                                    error,
                                )
                                prepared.stats["errors"] += prepared.stats["processed"]
                                failed_concurrent_batches.append(prepared)

                            item_duration += time.monotonic() - commit_started
                            indexing_eta.record(item_duration)
                            report_item_progress(item)
                            accumulate_batch_stats(prepared.stats)
                    finally:
                        if completed_workers < embedding_concurrency:
                            stop_pipeline.set()
                        producer.join()
                        for worker in workers:
                            worker.result()

                    if producer_errors:
                        raise producer_errors[0]

                if stop_pipeline.is_set():
                    raise KeyboardInterrupt

                if failed_concurrent_batches:
                    import time as _retry_time

                    failed_items = sum(batch.stats["processed"] for batch in failed_concurrent_batches)
                    try:
                        sys.stderr.write(f"\n  Retrying {failed_items} failed items sequentially...\n")
                    except Exception:
                        pass
                    _retry_time.sleep(1)

                    retry_ok = 0
                    retry_fail = 0
                    for prepared in failed_concurrent_batches:
                        try:
                            embeddings = self.chroma_client.embed_documents(prepared.documents)
                            added_before = prepared.stats["added"]
                            updated_before = prepared.stats["updated"]
                            self._commit_prepared_batch(
                                prepared,
                                force_rebuild=False,
                                embeddings=embeddings,
                                replace_item_records=complete_rebuild,
                            )
                            stats["added_items"] += prepared.stats["added"] - added_before
                            stats["updated_items"] += prepared.stats["updated"] - updated_before
                            stats["errors"] -= prepared.stats["processed"]
                            recovered = prepared.stats["processed"]
                            stats["recovered_items"] += recovered
                            retry_ok += recovered
                        except Exception as e:
                            retry_fail += prepared.stats["processed"]
                            logger.error(
                                "Sequential retry failed for batch starting %s: %s",
                                prepared.ids[0] if prepared.ids else "unknown",
                                e,
                            )
                    try:
                        sys.stderr.write(f"  Retry: {retry_ok} recovered, {retry_fail} still failed\n")
                    except Exception:
                        pass

            if stop_pipeline.is_set():
                raise KeyboardInterrupt

            # Retry any documents that failed during the main run
            if _failed_docs:
                try:
                    _clear_progress_line(sys.stderr)
                    retry_items = sum(batch.stats["processed"] for batch in _failed_docs)
                    sys.stderr.write(f"\n  Retrying {retry_items} failed items...\n")
                except Exception:
                    pass

                import time as _retry_time

                _retry_time.sleep(1)  # Brief pause before retry

                retry_ok = 0
                retry_fail = 0
                for prepared in _failed_docs:
                    try:
                        added_before = prepared.stats["added"]
                        updated_before = prepared.stats["updated"]
                        self._commit_prepared_batch(
                            prepared,
                            force_rebuild=False,
                            embeddings=(
                                self.chroma_client.embed_documents(prepared.documents)
                                if complete_rebuild and prepared.documents
                                else None
                            ),
                            replace_item_records=complete_rebuild,
                        )
                        stats["added_items"] += prepared.stats["added"] - added_before
                        stats["updated_items"] += prepared.stats["updated"] - updated_before
                        recovered = prepared.stats["processed"]
                        retry_ok += recovered
                        stats["errors"] -= recovered
                        stats["recovered_items"] += recovered
                    except Exception as e2:
                        retry_fail += prepared.stats["processed"]
                        first_id = prepared.ids[0] if prepared.ids else "unknown"
                        logger.error(f"Retry failed for {first_id}: {e2}")

                try:
                    sys.stderr.write(f"  Retry: {retry_ok} recovered, {retry_fail} still failed\n")
                except Exception:
                    pass

            # Clear the progress line and show summary
            if stop_pipeline.is_set():
                raise KeyboardInterrupt
            if complete_rebuild and rebuild_state is not None:
                pending_keys = self.chroma_client.get_pending_rebuild_item_keys(
                    rebuild_state["marker"]
                )
                stats["pending_rebuild_items"] = len(pending_keys)
                if pending_keys and stats["errors"] == 0:
                    stats["errors"] = len(pending_keys)
                    stats["error"] = (
                        f"{len(pending_keys)} item(s) remain marked for rebuild"
                    )
                if stats["errors"]:
                    stats.setdefault(
                        "error",
                        "The rebuild is incomplete; a normal update will continue it",
                    )
                stats["rebuild_in_progress"] = stats["errors"] != 0
            if (
                complete_rebuild
                and rebuild_state is not None
                and stats["errors"] == 0
            ):
                self._save_rebuild_state(None)
                rebuild_state = None

            try:
                _clear_progress_line(sys.stderr)
                summary = (
                    f"  Done: {stats['processed_items']} indexed, "
                    f"{stats['skipped_items']} skipped, "
                    f"{stats['errors']} errors"
                )
                if stats["recovered_items"]:
                    summary += f", {stats['recovered_items']} recovered"
                if stats["reused_embeddings"]:
                    summary += f", {stats['reused_embeddings']} unchanged vectors reused"
                sys.stderr.write(summary + "\n")
            except Exception:
                pass

            # Update last update time, and promote last_sync_version on success
            self.update_config["last_update"] = datetime.now().isoformat()
            completed_sync_version = target_sync_version if stats["errors"] == 0 else None
            completed_fulltext = (
                indexed_fulltext_state
                if stats["errors"] == 0 and limit is None and completed_sync_version is not None
                else None
            )
            completed_content_signature = (
                _CONTENT_CONTRACT_SIGNATURE
                if stats["errors"] == 0
                and limit is None
                and completed_sync_version is not None
                and (complete_rebuild or not collection_has_items or not content_mismatch)
                else None
            )
            self._save_update_config(
                last_sync_version=completed_sync_version,
                indexed_fulltext=completed_fulltext,
                indexed_content_signature=completed_content_signature,
            )

            end_time = datetime.now()
            stats["duration"] = str(end_time - start_time)
            stats["end_time"] = end_time.isoformat()

            logger.info(f"Database update completed in {stats['duration']}")
            return stats

        except Exception as e:
            logger.error(f"Error updating database: {e}")
            stats["errors"] += 1
            stats["error"] = str(e)
            if rebuild_state is not None:
                stats["rebuild_in_progress"] = True
            end_time = datetime.now()
            stats["duration"] = str(end_time - start_time)
            return stats
        finally:
            if interrupt_controller is not None:
                try:
                    interrupt_controller.restore()
                except Exception as e:
                    logger.warning("Could not restore SIGINT handler: %s", e)
                set_cancel_event = getattr(
                    self.chroma_client,
                    "set_embedding_cancel_event",
                    None,
                )
                if set_cancel_event is not None:
                    try:
                        set_cancel_event(None)
                    except Exception as e:
                        logger.warning(
                            "Could not clear embedding cancellation: %s",
                            e,
                        )
            # Release the update flock on every exit path. Paired with the
            # __enter__ call above; the "not acquired" branch releases
            # separately before its early return, so this finally only runs
            # for the path where we actually hold the lock.
            lock_cm.__exit__(None, None, None)

    def _prepare_item_batch(
        self,
        items: list[dict[str, Any]],
    ) -> _PreparedIndexBatch:
        """Prepare an item batch without embedding or mutating ChromaDB."""
        stats = {
            "processed": 0,
            "added": 0,
            "updated": 0,
            "skipped": 0,
            "errors": 0,
            "reused_embeddings": 0,
        }

        chunking = self._chunking_enabled
        chunk_size = int(self._chunking_config.get("chunk_size", 1500))
        overlap = int(self._chunking_config.get("overlap", 200))
        max_chunks = int(self._chunking_config.get("max_chunks_per_item", DEFAULT_MAX_CHUNKS_PER_ITEM))

        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ids: list[str] = []
        # One entry per *item* successfully prepared (not per chunk) so add/
        # update accounting stays item-granular regardless of chunking.
        item_keys_order: list[str] = []

        for item in items:
            try:
                item_key = item.get("key", "")
                if not item_key:
                    stats["skipped"] += 1
                    continue

                data = item.get("data", {})
                fulltext = (data.get("fulltext") or "").strip()
                metadata_text = self._create_document_text(item).strip()
                selected_fulltext_source = data.get("fulltextSource", "")
                self_contained = selected_fulltext_source in _SELF_CONTAINED_FULLTEXT_SOURCES
                metadata = self._create_metadata(item)
                metadata["index_layout_signature"] = self._index_layout_signature
                metadata["index_fulltext"] = bool(fulltext)
                metadata["index_content_signature"] = _CONTENT_CONTRACT_SIGNATURE

                if chunking:
                    passages: list[tuple[str, int, int, str, str]] = []
                    if metadata_text and not self_contained:
                        passages.append(
                            (
                                metadata_text,
                                0,
                                len(metadata_text),
                                "metadata",
                                metadata_text,
                            )
                        )
                    if fulltext:
                        passages.extend(
                            (
                                text,
                                start,
                                end,
                                "body",
                                fulltext,
                            )
                            for text, start, end in split_into_passages(
                                fulltext,
                                chunk_size,
                                overlap,
                                max_chunks,
                            )
                        )
                    elif metadata_text and not passages:
                        passages.append(
                            (
                                metadata_text,
                                0,
                                len(metadata_text),
                                "metadata",
                                metadata_text,
                            )
                        )
                    if not passages:
                        stats["skipped"] += 1
                        continue
                    n_chunks = len(passages)
                    for ci, (
                        chunk_text,
                        c0,
                        c1,
                        passage_kind,
                        source_text,
                    ) in enumerate(passages):
                        cmeta = dict(metadata)
                        cmeta["parent_item_key"] = item_key
                        cmeta["chunk_index"] = ci
                        cmeta["n_chunks"] = n_chunks
                        cmeta["passage_kind"] = passage_kind
                        cmeta["char_start"] = c0
                        cmeta["char_end"] = c1
                        page = _page_for_offset(source_text, c0) if passage_kind == "body" else None
                        if page is not None:
                            cmeta["page"] = page
                        document = self.chroma_client.truncate_text(chunk_text)
                        cmeta["embedding_content_sha256"] = _embedding_content_hash(document)
                        documents.append(document)
                        metadatas.append(cmeta)
                        ids.append(f"{item_key}#{ci}")
                else:
                    if fulltext:
                        doc_text = fulltext if self_contained or not metadata_text else f"{metadata_text}\n\n{fulltext}"
                    else:
                        doc_text = metadata_text
                    if not doc_text:
                        stats["skipped"] += 1
                        continue
                    # Truncate to fit the configured embedding model's token limit
                    document = self.chroma_client.truncate_text(doc_text)
                    metadata["embedding_content_sha256"] = _embedding_content_hash(document)
                    documents.append(document)
                    metadatas.append(metadata)
                    ids.append(item_key)

                item_keys_order.append(item_key)
                stats["processed"] += 1

            except Exception as e:
                logger.error(f"Error processing item {item.get('key', 'unknown')}: {e}")
                stats["errors"] += 1

        return _PreparedIndexBatch(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            expected_ids=list(ids),
            metadata_only_ids=[],
            metadata_only_metadatas=[],
            item_keys=item_keys_order,
            stats=stats,
        )

    def _partition_reusable_embeddings(
        self,
        prepared: _PreparedIndexBatch,
        *,
        force_rebuild: bool,
        item_atomic_rebuild: bool = False,
    ) -> _PreparedIndexBatch:
        """Move text-identical existing records to metadata-only updates."""
        if (
            force_rebuild
            or not prepared.ids
            or not hasattr(self.chroma_client, "get_records")
            or not hasattr(self.chroma_client, "update_metadatas")
        ):
            return prepared

        existing = self.chroma_client.get_records(prepared.ids)
        if item_atomic_rebuild:
            item_keys = list(dict.fromkeys(prepared.item_keys))
            if len(item_keys) != 1:
                raise ValueError(
                    "In-place rebuild batches must contain exactly one Zotero item"
                )
            item_key = item_keys[0]
            current_ids = set()
            if hasattr(self.chroma_client, "get_item_chunk_ids"):
                current_ids.update(
                    self.chroma_client.get_item_chunk_ids(item_key)
                )
            current_ids.update(self.chroma_client.get_existing_ids([item_key]))
            expected_ids = set(prepared.expected_ids)
            complete = current_ids == expected_ids
            for doc_id, document, metadata in zip(
                prepared.ids,
                prepared.documents,
                prepared.metadatas,
                strict=True,
            ):
                stored = existing.get(doc_id) or {}
                stored_metadata = stored.get("metadata") or {}
                stored_hash = stored_metadata.get("embedding_content_sha256")
                if stored_hash != metadata["embedding_content_sha256"]:
                    complete = False
                    break
            if complete:
                prepared.metadata_only_ids.extend(prepared.ids)
                prepared.metadata_only_metadatas.extend(prepared.metadatas)
                prepared.stats["reused_embeddings"] += len(prepared.ids)
                prepared.ids = []
                prepared.documents = []
                prepared.metadatas = []
            return prepared

        embed_documents: list[str] = []
        embed_metadatas: list[dict[str, Any]] = []
        embed_ids: list[str] = []

        for doc_id, document, metadata in zip(
            prepared.ids,
            prepared.documents,
            prepared.metadatas,
            strict=True,
        ):
            stored = existing.get(doc_id) or {}
            stored_metadata = stored.get("metadata") or {}
            current_hash = metadata["embedding_content_sha256"]
            stored_hash = stored_metadata.get("embedding_content_sha256")
            unchanged = (
                stored_hash == current_hash
                or (
                    not str(stored_hash or "").startswith("rebuild-pending:")
                    and stored.get("document") == document
                )
            )
            if unchanged:
                prepared.metadata_only_ids.append(doc_id)
                prepared.metadata_only_metadatas.append(metadata)
            else:
                embed_ids.append(doc_id)
                embed_documents.append(document)
                embed_metadatas.append(metadata)

        prepared.ids = embed_ids
        prepared.documents = embed_documents
        prepared.metadatas = embed_metadatas
        prepared.stats["reused_embeddings"] += len(prepared.metadata_only_ids)
        return prepared

    def _commit_prepared_batch(
        self,
        prepared: _PreparedIndexBatch,
        force_rebuild: bool = False,
        _failed_docs: list | None = None,
        embeddings: list[list[float]] | None = None,
        replace_item_records: bool = False,
    ) -> dict[str, int]:
        """Write one prepared batch, optionally using precomputed embeddings."""
        documents = prepared.documents
        metadatas = prepared.metadatas
        ids = prepared.ids
        expected_ids = prepared.expected_ids
        metadata_only_ids = prepared.metadata_only_ids
        metadata_only_metadatas = prepared.metadata_only_metadatas
        item_keys_order = prepared.item_keys
        stats = prepared.stats

        if not documents and not metadata_only_ids:
            return stats
        if embeddings is not None and len(embeddings) != len(documents):
            raise ValueError(f"Embedding provider returned {len(embeddings)} vectors for {len(documents)} documents")

        # Probe both layouts so migrations are classified as updates. Existing
        # records remain searchable until the replacement upsert succeeds.
        existing_item_keys: set[str] = set()
        if not force_rebuild:
            unique_keys = list(dict.fromkeys(item_keys_order))
            probe_ids = unique_keys + [f"{key}#0" for key in unique_keys]
            existing_records = self.chroma_client.get_existing_ids(probe_ids)
            existing_item_keys = {record_id.split("#", 1)[0] for record_id in existing_records}

        expected_ids_by_item: dict[str, set[str]] = {key: set() for key in item_keys_order}
        for doc_id in expected_ids:
            expected_ids_by_item.setdefault(doc_id.split("#", 1)[0], set()).add(doc_id)

        try:
            if documents:
                if replace_item_records:
                    if embeddings is None:
                        raise ValueError(
                            "In-place rebuild replacement requires precomputed embeddings"
                        )
                    if not hasattr(self.chroma_client, "delete_item_records"):
                        raise RuntimeError(
                            "The Chroma client cannot replace complete item records"
                        )
                    for key in dict.fromkeys(item_keys_order):
                        self.chroma_client.delete_item_records(key)
                if embeddings is None:
                    self.chroma_client.upsert_documents(documents, metadatas, ids)
                else:
                    self.chroma_client.upsert_embeddings(
                        documents,
                        metadatas,
                        ids,
                        embeddings,
                    )
            if metadata_only_ids:
                self.chroma_client.update_metadatas(
                    metadata_only_ids,
                    metadata_only_metadatas,
                )
            if not force_rebuild and hasattr(self.chroma_client, "reconcile_item_records"):
                for key, expected_ids in expected_ids_by_item.items():
                    self.chroma_client.reconcile_item_records(key, expected_ids)
            for key in item_keys_order:
                if key in existing_item_keys:
                    stats["updated"] += 1
                else:
                    stats["added"] += 1
        except Exception as e:
            # Batch failed — collect failures for end-of-run retry.
            # ChromaDB's ONNX tokenizer can fail intermittently in bursts;
            # retrying immediately usually fails too. Collecting failures
            # and retrying after all batches are done is more effective.
            logger.warning(f"Batch upsert failed ({e}), saving for retry")
            if _failed_docs is not None:
                _failed_docs.append(prepared)
                stats["errors"] += stats["processed"]
            else:
                raise

        return stats

    def _process_item_batch(
        self,
        items: list[dict[str, Any]],
        force_rebuild: bool = False,
        _failed_docs: list | None = None,
        item_atomic_rebuild: bool = False,
    ) -> dict[str, int]:
        """Prepare, embed, and write a batch through the sequential path."""
        prepared = self._prepare_item_batch(items)
        prepared = self._partition_reusable_embeddings(
            prepared,
            force_rebuild=force_rebuild,
            item_atomic_rebuild=item_atomic_rebuild,
        )
        embeddings = None
        if item_atomic_rebuild and prepared.documents:
            embeddings = self.chroma_client.embed_documents(prepared.documents)
        return self._commit_prepared_batch(
            prepared,
            force_rebuild=force_rebuild,
            _failed_docs=_failed_docs,
            embeddings=embeddings,
            replace_item_records=item_atomic_rebuild,
        )

    def get_openai_batch_status(self, batch_ids: list[str] | None = None) -> dict[str, Any]:
        """Refresh and return OpenAI Batch API status for the latest run or selected batches."""
        selected_ids = set(batch_ids or [])
        manifest = openai_batch.find_manifest(
            config_path=self.config_path,
            batch_id=next(iter(selected_ids), None),
        )
        manifest = openai_batch.refresh_manifest_status(
            manifest,
            embedding_config=self.chroma_client.embedding_config,
            batch_ids=selected_ids or None,
        )
        batches = [
            batch for batch in manifest.get("batches", []) if not selected_ids or batch.get("batch_id") in selected_ids
        ]
        missing_ids = selected_ids - {batch.get("batch_id") for batch in batches}
        if missing_ids:
            raise FileNotFoundError(f"No OpenAI batch manifest entries found for: {', '.join(sorted(missing_ids))}")
        return {
            "run_id": manifest.get("run_id"),
            "manifest_path": manifest.get("manifest_path"),
            "model": manifest.get("model"),
            "force_full_rebuild": manifest.get("force_full_rebuild", False),
            "batches": batches,
        }

    def _openai_batch_submitted_hashes(
        self,
        manifest: dict[str, Any],
    ) -> dict[str, dict[str, str]]:
        """Load submitted payload hashes grouped by parent item."""
        hashes: dict[str, dict[str, str]] = {}
        paths = [Path(batch["records_path"]) for batch in manifest.get("batches", [])]
        metadata_only_path = manifest.get("metadata_only_records_path")
        if metadata_only_path:
            paths.append(Path(metadata_only_path))
        for path in paths:
            for record in openai_batch.read_jsonl(path):
                doc_id = record["id"]
                metadata = record.get("metadata") or {}
                parent = metadata.get("parent_item_key") or doc_id.split("#", 1)[0]
                content_hash = metadata.get("embedding_content_sha256")
                if content_hash:
                    hashes.setdefault(parent, {})[doc_id] = str(content_hash)
        return hashes

    def _openai_batch_submitted_record_hashes(
        self,
        manifest: dict[str, Any],
    ) -> dict[str, dict[str, str]]:
        """Load submitted complete-record identities grouped by parent item."""
        hashes: dict[str, dict[str, str]] = {}
        paths = [Path(batch["records_path"]) for batch in manifest.get("batches", [])]
        metadata_only_path = manifest.get("metadata_only_records_path")
        if metadata_only_path:
            paths.append(Path(metadata_only_path))
        for path in paths:
            for record in openai_batch.read_jsonl(path):
                doc_id = record["id"]
                metadata = record.get("metadata") or {}
                parent = metadata.get("parent_item_key") or doc_id.split("#", 1)[0]
                hashes.setdefault(parent, {})[doc_id] = record_state_hash(
                    record.get("document"),
                    metadata,
                )
        return hashes

    def _validate_openai_batch_baseline(
        self,
        manifest: dict[str, Any],
    ) -> None:
        """Reject a batch that would overwrite vectors changed after submission."""
        if "baseline_embedding_hashes_by_item" not in manifest:
            logger.warning(
                "OpenAI batch manifest predates stale-write protection; importing without a baseline comparison."
            )
            return
        get_item_hashes = getattr(self.chroma_client, "get_item_embedding_hashes", None)
        if not callable(get_item_hashes):
            raise RuntimeError("The Chroma client cannot validate this batch against the current index")

        baseline = manifest.get("baseline_embedding_hashes_by_item") or {}
        submitted = self._openai_batch_submitted_hashes(manifest)
        expected = manifest.get("expected_ids_by_item") or {}
        conflicts = []
        for parent in sorted(set(baseline) | set(submitted) | set(expected)):
            baseline_hashes = baseline.get(parent) or {}
            submitted_hashes = submitted.get(parent) or {}
            current_hashes = get_item_hashes(parent)
            expected_ids = set(expected.get(parent) or submitted_hashes)
            submitted_complete = (
                bool(expected_ids)
                and expected_ids.issubset(current_hashes)
                and all(current_hashes.get(doc_id) == submitted_hashes.get(doc_id) for doc_id in expected_ids)
            )

            if set(baseline_hashes) - set(current_hashes) and not submitted_complete:
                conflicts.append(parent)
                continue
            allowed_ids = set(baseline_hashes) | set(submitted_hashes)
            if set(current_hashes) - allowed_ids:
                conflicts.append(parent)
                continue
            for doc_id, current_hash in current_hashes.items():
                allowed_hashes = {
                    value
                    for value in (
                        baseline_hashes.get(doc_id),
                        submitted_hashes.get(doc_id),
                    )
                    if value
                }
                if allowed_hashes and current_hash not in allowed_hashes:
                    conflicts.append(parent)
                    break

        if conflicts:
            sample = ", ".join(conflicts[:10])
            suffix = "" if len(conflicts) <= 10 else f" and {len(conflicts) - 10} more"
            raise RuntimeError(
                "OpenAI batch import is stale for item(s) changed after "
                f"submission: {sample}{suffix}. Run a new update instead."
            )

        if "baseline_record_hashes_by_item" not in manifest:
            logger.warning(
                "OpenAI batch manifest predates metadata stale-write "
                "protection; importing after payload-only validation."
            )
            return
        get_item_record_hashes = getattr(self.chroma_client, "get_item_record_hashes", None)
        if not callable(get_item_record_hashes):
            raise RuntimeError("The Chroma client cannot validate batch metadata against the current index")

        baseline_records = manifest.get("baseline_record_hashes_by_item") or {}
        submitted_records = self._openai_batch_submitted_record_hashes(manifest)
        record_conflicts = []
        for parent in sorted(set(baseline_records) | set(submitted_records)):
            baseline_hashes = baseline_records.get(parent) or {}
            submitted_hashes = submitted_records.get(parent) or {}
            current_hashes = get_item_record_hashes(parent)
            for doc_id, current_hash in current_hashes.items():
                allowed_hashes = {
                    value
                    for value in (
                        baseline_hashes.get(doc_id),
                        submitted_hashes.get(doc_id),
                    )
                    if value
                }
                if allowed_hashes and current_hash not in allowed_hashes:
                    record_conflicts.append(parent)
                    break

        if record_conflicts:
            sample = ", ".join(record_conflicts[:10])
            suffix = "" if len(record_conflicts) <= 10 else f" and {len(record_conflicts) - 10} more"
            raise RuntimeError(
                "OpenAI batch import is stale for item metadata changed after "
                f"submission: {sample}{suffix}. Run a new update instead."
            )

    def _validate_openai_batch_source(self, manifest: dict[str, Any]) -> None:
        """Reject delayed results when their Zotero source items changed."""
        if not manifest.get("source_guard_version"):
            logger.warning(
                "OpenAI batch manifest predates Zotero source-version "
                "validation; importing against the index baseline only."
            )
            return
        target_version = manifest.get("target_sync_version")
        if target_version is None:
            raise RuntimeError("OpenAI batch manifest has no verified Zotero source version")
        try:
            current_version = self.zotero_client.last_modified_version()
        except Exception as e:
            raise RuntimeError("Could not verify Zotero source state before OpenAI batch import") from e
        if int(current_version) <= int(target_version):
            return
        if manifest.get("force_full_rebuild"):
            raise RuntimeError(
                "Zotero changed after this force-rebuild batch was submitted; "
                "submit a new rebuild so the replacement corpus is complete"
            )

        changed_items, current_keys = self._get_changed_items_from_api(int(target_version))
        if current_keys is None:
            raise RuntimeError("Could not completely verify Zotero changes before OpenAI batch import")
        expected_parents = set((manifest.get("expected_ids_by_item") or {}).keys())
        changed_parents = {item.get("key", "") for item in changed_items if item.get("key")}
        deleted_method = getattr(self.zotero_client, "deleted", None)
        deleted_keys: set[str] = set()
        if callable(deleted_method):
            try:
                deleted_keys = set((deleted_method(since=int(target_version)) or {}).get("items", []))
            except Exception as e:
                raise RuntimeError("Could not verify Zotero deletions before OpenAI batch import") from e
        affected = (expected_parents - current_keys) | (expected_parents & changed_parents)
        if affected or deleted_keys:
            sample_keys = sorted(affected or deleted_keys)
            sample = ", ".join(sample_keys[:10])
            suffix = "" if len(sample_keys) <= 10 else f" and {len(sample_keys) - 10} more"
            raise RuntimeError(
                f"Zotero source data changed after OpenAI batch submission ({sample}{suffix}); run a new update instead"
            )

    def import_openai_batch(self, batch_ids: list[str] | None = None) -> dict[str, Any]:
        """Import completed OpenAI Batch API embeddings into ChromaDB."""
        lock_path = Path.home() / ".config" / "zotero-mcp" / "update.lock"
        lock_cm = _acquire_update_lock(lock_path)
        acquired = lock_cm.__enter__()
        if not acquired:
            lock_cm.__exit__(None, None, None)
            raise RuntimeError(
                "Another semantic-search update is already running "
                f"(lock held at {lock_path})"
            )
        try:
            return self._import_openai_batch_locked(batch_ids)
        finally:
            lock_cm.__exit__(None, None, None)

    def _import_openai_batch_locked(
        self,
        batch_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Import a batch while the semantic update lock is held."""
        selected_ids = set(batch_ids or [])
        manifest = openai_batch.find_manifest(
            config_path=self.config_path,
            batch_id=next(iter(selected_ids), None),
        )
        manifest = openai_batch.refresh_manifest_status(
            manifest,
            embedding_config=self.chroma_client.embedding_config,
            batch_ids=selected_ids or None,
        )

        all_batches = manifest.get("batches", [])
        batches = [batch for batch in all_batches if not selected_ids or batch.get("batch_id") in selected_ids]
        missing_ids = selected_ids - {batch.get("batch_id") for batch in batches}
        if missing_ids:
            raise FileNotFoundError(f"No OpenAI batch manifest entries found for: {', '.join(sorted(missing_ids))}")
        if not batches:
            raise ValueError("No matching OpenAI batches found in the local manifest")
        if manifest.get("force_full_rebuild") and selected_ids and len(batches) != len(all_batches):
            raise RuntimeError("Force-rebuild OpenAI batch runs must be imported as a complete run")
        if manifest.get("force_full_rebuild"):
            incomplete = [
                batch.get("batch_id")
                for batch in all_batches
                if not batch.get("imported_at") and batch.get("status") != "completed"
            ]
            if incomplete:
                raise RuntimeError(
                    "Force-rebuild OpenAI batch runs can only be imported after all batches complete: "
                    + ", ".join(incomplete)
                )

        stats = {
            "run_id": manifest.get("run_id"),
            "manifest_path": manifest.get("manifest_path"),
            "batches_seen": len(batches),
            "batches_imported": 0,
            "batches_skipped": 0,
            "imported_items": 0,
            "added_items": 0,
            "updated_items": 0,
            "failed_items": 0,
            "missing_items": 0,
            "orphan_segment_directories_pruned": 0,
            "orphan_segment_bytes_pruned": 0,
            "orphan_segment_cleanup_errors": 0,
            "errors": [],
        }

        staged_batch_rebuild_active = False
        try:
            self._run_orphan_segment_cleanup(stats)
            self._validate_openai_batch_source(manifest)
            self._validate_openai_batch_baseline(manifest)
            already_imported = any(batch.get("imported_at") for batch in all_batches)
            if (
                manifest.get("force_full_rebuild")
                and already_imported
                and not all(batch.get("imported_at") for batch in all_batches)
            ):
                raise RuntimeError(
                    "Cannot safely resume a partially imported legacy force rebuild; submit a new force rebuild"
                )
            client = openai_batch.create_openai_client(self.chroma_client.embedding_config)
            prepared_imports = []
            for batch in batches:
                if batch.get("imported_at"):
                    stats["batches_skipped"] += 1
                    continue
                if batch.get("status") != "completed":
                    stats["batches_skipped"] += 1
                    stats["errors"].append(
                        {
                            "batch_id": batch.get("batch_id"),
                            "error": f"Batch status is {batch.get('status')}, not completed",
                        }
                    )
                    continue
                output_file_id = batch.get("output_file_id")
                if not output_file_id:
                    stats["batches_skipped"] += 1
                    stats["errors"].append({"batch_id": batch.get("batch_id"), "error": "Missing output_file_id"})
                    continue

                output_path = Path(batch["records_path"]).with_name(Path(batch["records_path"]).stem + "-output.jsonl")
                if output_path.exists():
                    output_text = output_path.read_text(encoding="utf-8")
                else:
                    output_text = openai_batch.download_file_text(client, output_file_id, output_path)
                embeddings_by_id, row_failures = openai_batch.parse_embedding_output(output_text)

                if batch.get("error_file_id"):
                    error_path = Path(batch["records_path"]).with_name(
                        Path(batch["records_path"]).stem + "-errors.jsonl"
                    )
                    error_text = openai_batch.download_file_text(client, batch["error_file_id"], error_path)
                    row_failures.extend(openai_batch.parse_error_output(error_text))

                records = {record["id"]: record for record in openai_batch.read_jsonl(Path(batch["records_path"]))}
                ids = [doc_id for doc_id in embeddings_by_id if doc_id in records]
                unexpected_output_ids = [doc_id for doc_id in embeddings_by_id if doc_id not in records]
                failure_ids = {failure.get("custom_id") for failure in row_failures if failure.get("custom_id")}
                missing_result_ids = [
                    doc_id for doc_id in records if doc_id not in embeddings_by_id and doc_id not in failure_ids
                ]
                missing_errors = [
                    {"custom_id": doc_id, "error": "Batch output returned an embedding for an unknown record"}
                    for doc_id in unexpected_output_ids
                ] + [
                    {"custom_id": doc_id, "error": "No embedding or error row returned for batch record"}
                    for doc_id in missing_result_ids
                ]
                stats["missing_items"] += len(unexpected_output_ids) + len(missing_result_ids)
                stats["failed_items"] += len(row_failures)
                stats["errors"].extend(row_failures)
                stats["errors"].extend(missing_errors)

                batch_errors = row_failures + missing_errors
                if batch_errors:
                    batch["import_error_at"] = datetime.now().isoformat()
                    batch["import_error_count"] = len(batch_errors)
                    continue

                prepared_imports.append((batch, records, ids, embeddings_by_id))

            # A force rebuild must validate the entire replacement before the
            # old collection is discarded. Partial output leaves the existing
            # index untouched and the manifest retryable.
            pending_selected = [batch for batch in batches if not batch.get("imported_at")]
            if manifest.get("force_full_rebuild") and len(prepared_imports) != len(pending_selected):
                openai_batch.save_manifest(manifest)
                return stats

            if manifest.get("force_full_rebuild") and not already_imported and prepared_imports:
                self.chroma_client.begin_staged_rebuild()
                staged_batch_rebuild_active = True

            for batch, records, ids, embeddings_by_id in prepared_imports:
                if ids:
                    existing_ids = self.chroma_client.get_existing_ids(ids)
                    self.chroma_client.upsert_embeddings(
                        documents=[records[doc_id]["document"] for doc_id in ids],
                        metadatas=[records[doc_id]["metadata"] for doc_id in ids],
                        ids=ids,
                        embeddings=[embeddings_by_id[doc_id] for doc_id in ids],
                    )
                    stats["imported_items"] += len(ids)
                    stats["updated_items"] += len(existing_ids)
                    stats["added_items"] += len(ids) - len(existing_ids)

                batch["imported_at"] = datetime.now().isoformat()
                batch["imported_count"] = len(ids)
                batch.pop("import_error_at", None)
                batch.pop("import_error_count", None)
                stats["batches_imported"] += 1

            if staged_batch_rebuild_active:
                self.chroma_client.commit_staged_rebuild()
                staged_batch_rebuild_active = False

            openai_batch.save_manifest(manifest)
            if all(batch.get("imported_at") for batch in all_batches):
                # Every expected record is now present. Reconcile only at this
                # point because one item's chunks may span multiple batch files.
                metadata_only_path = manifest.get("metadata_only_records_path")
                if metadata_only_path:
                    metadata_only_records = openai_batch.read_jsonl(Path(metadata_only_path))
                    if metadata_only_records:
                        self.chroma_client.update_metadatas(
                            [record["id"] for record in metadata_only_records],
                            [record["metadata"] for record in metadata_only_records],
                        )
                        stats["reused_embeddings"] = len(metadata_only_records)
                manifest_expected = manifest.get("expected_ids_by_item") or {}
                expected_ids_by_item: dict[str, set[str]] = {
                    parent: set(ids) for parent, ids in manifest_expected.items()
                }
                if not expected_ids_by_item:
                    for batch in all_batches:
                        for record in openai_batch.read_jsonl(Path(batch["records_path"])):
                            doc_id = record["id"]
                            parent = record.get("metadata", {}).get("parent_item_key") or doc_id.split("#", 1)[0]
                            expected_ids_by_item.setdefault(parent, set()).add(doc_id)
                if hasattr(self.chroma_client, "reconcile_item_records"):
                    for parent, expected_ids in expected_ids_by_item.items():
                        self.chroma_client.reconcile_item_records(
                            parent,
                            expected_ids,
                        )
                self.update_config["last_update"] = datetime.now().isoformat()
                self._save_update_config(
                    last_sync_version=manifest.get("target_sync_version"),
                    indexed_fulltext=manifest.get("fulltext"),
                    indexed_content_signature=manifest.get("content_signature"),
                )
            return stats
        finally:
            if staged_batch_rebuild_active:
                try:
                    self.chroma_client.abort_staged_rebuild()
                except Exception as e:
                    logger.warning(
                        "Could not discard staged OpenAI batch rebuild: %s",
                        e,
                    )

    def search(self, query: str, limit: int = 10, filters: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Perform semantic search over the Zotero library.

        Args:
            query: Search query text
            limit: Maximum number of results to return
            filters: Optional metadata filters

        Returns:
            Search results with Zotero item details
        """
        try:
            # Over-fetch candidates when re-ranking and/or chunking are on.
            reranker = self._get_reranker()
            fetch_limit = limit
            if self._chunking_enabled:
                # Passages are grouped back to items downstream, so fetch
                # several chunks per desired item to still surface ~limit
                # distinct papers.
                fetch_limit = max(fetch_limit, limit * 4)
            if reranker:
                multiplier = self._reranker_config.get("candidate_multiplier", 3)
                fetch_limit = max(fetch_limit, limit * multiplier)

            # Perform semantic search. If several strong chunks belong to the
            # same papers, widen the candidate window until it contains enough
            # distinct parent items or reaches a bounded ceiling.
            results = self.chroma_client.search(query_texts=[query], n_results=fetch_limit, where=filters)
            if self._chunking_enabled:
                # A single book can legitimately own hundreds of highly ranked
                # chunks. Keep widening to the collection boundary so those
                # chunks cannot consume a fixed candidate ceiling and hide all
                # other parents. Most searches still stop after the first or
                # second small query once enough distinct items are present.
                try:
                    max_fetch = max(
                        fetch_limit,
                        self.chroma_client.count_documents(),
                    )
                except Exception:
                    max_chunks = int(
                        self._chunking_config.get(
                            "max_chunks_per_item",
                            DEFAULT_MAX_CHUNKS_PER_ITEM,
                        )
                    )
                    max_fetch = max(fetch_limit, limit * max_chunks)
                while fetch_limit < max_fetch:
                    result_ids = (results.get("ids") or [[]])[0]
                    distinct_items = {raw_id.split("#", 1)[0] for raw_id in result_ids}
                    if len(distinct_items) >= limit or len(result_ids) < fetch_limit:
                        break
                    fetch_limit = min(fetch_limit * 2, max_fetch)
                    results = self.chroma_client.search(
                        query_texts=[query],
                        n_results=fetch_limit,
                        where=filters,
                    )

            # Re-rank results with cross-encoder if enabled. With chunking we
            # rerank ALL candidates (grouping to `limit` items happens in
            # enrichment); without chunking we keep the historical top-k=limit.
            if reranker and results.get("documents") and results["documents"][0]:
                documents = results["documents"][0]
                top_k = len(documents) if self._chunking_enabled else limit
                ranked_indices = reranker.rerank(query, documents, top_k=top_k)
                for key in ["ids", "distances", "documents", "metadatas"]:
                    if results.get(key) and results[key][0]:
                        results[key][0] = [results[key][0][i] for i in ranked_indices]

            # Enrich results with full Zotero item data, grouping passages back
            # to their parent items and capping at `limit` distinct papers.
            enriched_results = self._enrich_search_results(
                results,
                query,
                limit,
                preserve_input_order=reranker is not None,
            )

            return {
                "query": query,
                "limit": limit,
                "filters": filters,
                "results": enriched_results,
                "total_found": len(enriched_results),
            }

        except Exception as e:
            logger.error(f"Error performing semantic search: {e}")
            return {
                "query": query,
                "limit": limit,
                "filters": filters,
                "results": [],
                "total_found": 0,
                "error": str(e),
            }

    def _enrich_search_results(
        self,
        chroma_results: dict[str, Any],
        query: str,
        limit: int | None = None,
        preserve_input_order: bool = False,
    ) -> list[dict[str, Any]]:
        """Enrich ChromaDB results with full Zotero item data.

        Chunk-aware: when the collection is indexed as passages, ids look like
        ``<item_key>#<n>``. Results are grouped back to their parent item. The
        first (best-ranked) passage supplies the primary result, while up to two
        independent supporting passages can add a bounded relevance bonus.
        Item-level collections (ids without ``#``) flow through unchanged.
        """
        enriched: list[dict[str, Any]] = []

        if not chroma_results.get("ids") or not chroma_results["ids"][0]:
            return enriched

        ids = chroma_results["ids"][0]
        distances = chroma_results.get("distances", [[]])[0]
        documents = chroma_results.get("documents", [[]])[0]
        metadatas = chroma_results.get("metadatas", [[]])[0]

        grouped: dict[str, list[dict[str, Any]]] = {}
        for i, raw_id in enumerate(ids):
            item_key = raw_id.split("#", 1)[0]
            distance = distances[i] if i < len(distances) else None
            document = documents[i] if i < len(documents) else ""
            meta = metadatas[i] if i < len(metadatas) else {}
            passage, passage_offset = best_snippet(query, document)
            if distance is None:
                similarity = 0
            elif hasattr(self.chroma_client, "distance_to_similarity"):
                similarity = self.chroma_client.distance_to_similarity(distance)
            else:
                similarity = 1 - distance
            meta = meta if isinstance(meta, dict) else {}
            global_start = (
                int(meta["char_start"]) + passage_offset if isinstance(meta.get("char_start"), int) else passage_offset
            )
            grouped.setdefault(item_key, []).append(
                {
                    "raw_id": raw_id,
                    "document": document,
                    "meta": meta,
                    "passage": passage,
                    "passage_offset": passage_offset,
                    "passage_start": global_start,
                    "passage_end": global_start + len(passage),
                    "similarity": similarity,
                }
            )

        for item_key, candidates in grouped.items():
            best = candidates[0]
            best_score = best["similarity"]
            selected = [best]
            is_chunked = "#" in best["raw_id"] or "chunk_index" in best["meta"]

            if is_chunked and best_score > 0:
                for candidate in candidates[1:]:
                    if len(selected) >= 3:
                        break
                    if candidate["similarity"] < max(0, best_score - 0.15):
                        continue
                    if any(_passages_overlap(candidate, chosen) for chosen in selected):
                        continue
                    selected.append(candidate)

            support_weights = (0.10, 0.05)
            support_bonus = (
                sum(
                    weight * min(1.0, max(0.0, candidate["similarity"]) / best_score)
                    for weight, candidate in zip(support_weights, selected[1:])
                )
                if best_score > 0
                else 0.0
            )
            aggregate_score = min(1.0, best_score * (1.0 + min(0.15, support_bonus)))

            enriched_result: dict[str, Any] = {
                "item_key": item_key,
                "similarity_score": aggregate_score,
                "matched_text": best["document"],
                "matched_passage": best["passage"],
                "metadata": best["meta"],
                "query": query,
            }
            if is_chunked:
                enriched_result["chunk_id"] = best["raw_id"]
                enriched_result["content_hash"] = _embedding_content_hash(best["document"])
                enriched_result["best_chunk_similarity_score"] = best_score
                enriched_result["supporting_chunk_count"] = len(selected) - 1
                enriched_result["matched_passages"] = [_passage_result(candidate) for candidate in selected]
            # Passage provenance — present only on a chunk-indexed collection.
            for mk in ("chunk_index", "n_chunks", "char_start", "char_end", "page"):
                if mk in best["meta"]:
                    enriched_result[mk] = best["meta"][mk]
            if "char_start" not in enriched_result and best["passage_offset"]:
                enriched_result["passage_offset"] = best["passage_offset"]

            try:
                enriched_result["zotero_item"] = self.zotero_client.item(item_key)
            except Exception as e:
                logger.error(f"Error enriching result for item {item_key}: {e}")
                enriched_result["error"] = f"Could not fetch full item data: {e}"

            enriched.append(enriched_result)

        if not preserve_input_order:
            enriched.sort(key=lambda result: result["similarity_score"], reverse=True)
        if limit:
            enriched = enriched[:limit]

        return enriched

    def get_database_status(self) -> dict[str, Any]:
        """Get status information about the semantic search database."""
        collection_info = self.chroma_client.get_collection_info()
        rebuild_state = self._load_rebuild_state()
        rebuild_status = None
        if rebuild_state is not None:
            rebuild_status = {
                "phase": rebuild_state.get("phase"),
                "fulltext": rebuild_state.get("fulltext"),
                "force_clear": rebuild_state.get("force_clear"),
                "embedding_identity": rebuild_state.get("embedding_identity"),
                "index_layout_signature": rebuild_state.get(
                    "index_layout_signature"
                ),
            }

        return {
            "collection_info": collection_info,
            "update_config": self.update_config,
            "openai_batch": {
                "enabled": self._load_openai_batch_enabled(),
                "active": self._resolve_openai_batch_enabled(None),
            },
            "should_update": self.should_update_database(),
            "last_update": self.update_config.get("last_update"),
            "rebuild_in_progress": rebuild_status,
        }

    def delete_item(self, item_key: str) -> bool:
        """Delete an item from the semantic search database."""
        try:
            if hasattr(self.chroma_client, "delete_item_records"):
                self.chroma_client.delete_item_records(item_key)
            else:
                self.chroma_client.delete_documents([item_key])
            return True
        except Exception as e:
            logger.error(f"Error deleting item {item_key}: {e}")
            return False


def create_semantic_search(
    config_path: str | None = None,
    db_path: str | None = None,
    *,
    allow_embedding_mismatch: bool = False,
) -> ZoteroSemanticSearch:
    """
    Create a ZoteroSemanticSearch instance.

    Args:
        config_path: Path to configuration file
        db_path: Optional path to Zotero database (overrides config file)

    Returns:
        Configured ZoteroSemanticSearch instance
    """
    return ZoteroSemanticSearch(
        config_path=config_path,
        db_path=db_path,
        allow_embedding_mismatch=allow_embedding_mismatch,
    )
