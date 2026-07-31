"""
Semantic search functionality for Zotero MCP.

This module provides semantic search capabilities by integrating ChromaDB
with the existing Zotero client to enable vector-based similarity search
over research libraries.
"""

import contextlib
import json
import logging
import math
import os
import re
import statistics
import sys
import threading
import time
import unicodedata
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
from .chroma_client import ChromaClient, create_chroma_client
from .client import get_zotero_client
from .local_db import LocalZoteroReader
from .utils import format_creators, is_local_mode, suppress_stdout

logger = logging.getLogger(__name__)

_CONTENT_CONTRACT_SIGNATURE = "lean-paper-content-v1"
_SELF_CONTAINED_FULLTEXT_SOURCES = {"betterissa-indexing"}


@dataclass
class _PreparedIndexBatch:
    """An embedding batch prepared in memory but not yet written to ChromaDB."""

    documents: list[str]
    metadatas: list[dict[str, Any]]
    ids: list[str]
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


def _force_update_requested() -> bool:
    """Whether the user asked to bypass the cross-process update lock."""
    return os.getenv("ZOTERO_MCP_FORCE_UPDATE", "").strip().lower() in {"1", "true", "yes"}


@contextlib.contextmanager
def _acquire_update_lock(lock_path: Path):
    """Non-blocking exclusive flock over an update-database run.

    Yields True if the lock was acquired (caller should proceed), False if
    another process already holds it (caller should skip). This prevents the
    MCP server's auto-update in ``server_lifespan`` from racing a manual
    ``zotero-mcp update-db`` invocation on the same ChromaDB collection.

    Setting ``ZOTERO_MCP_FORCE_UPDATE=1`` bypasses the lock entirely — an
    escape hatch for the rare case where a lock appears stuck (e.g. a crashed
    holder on a filesystem with quirky flock semantics) and the user knowingly
    accepts the small double-work risk.

    Windows lacks ``fcntl``; on that platform the function degrades to a
    no-op and yields True so behaviour matches pre-lock releases.
    """
    if _force_update_requested():
        yield True
        return

    try:
        import fcntl
    except ImportError:
        yield True
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    try:
        fd = open(lock_path, "w")
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        # Record our pid so a concurrent invocation can report the holder.
        try:
            fd.seek(0)
            fd.truncate()
            fd.write(str(os.getpid()))
            fd.flush()
        except Exception:
            pass
        yield True
    finally:
        if fd is not None:
            fd.close()


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
    max_chunks: int = 20,
) -> list[tuple[str, int, int]]:
    """Split *text* into overlapping passages on natural boundaries.

    Pure function (no I/O, no model load) so it is unit-testable in isolation.
    Returns a list of ``(passage_text, char_start, char_end)`` tuples with
    character offsets into the original string. Each window targets
    ``chunk_size`` characters but is snapped back to the nearest paragraph or
    sentence boundary in its second half so passages read as coherent quotes.
    Consecutive windows overlap by ``overlap`` characters so a relevant span
    straddling a boundary is still captured whole in one of them. At most
    ``max_chunks`` passages are produced (a guard against pathologically long
    documents inflating the index).
    """
    text = (text or "").strip()
    if not text:
        return []
    if overlap >= chunk_size:
        overlap = chunk_size // 4

    passages: list[tuple[str, int, int]] = []
    start = 0
    n = len(text)
    while start < n and len(passages) < max_chunks:
        end = min(n, start + chunk_size)
        if end < n:
            window = text[start:end]
            for sep in ("\n\n", ". ", ".\n", "\n", " "):
                idx = window.rfind(sep)
                if idx != -1 and idx >= int(chunk_size * 0.5):
                    end = start + idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            passages.append((chunk, start, min(end, n)))
        if end >= n:
            break
        new_start = end - overlap
        # Guarantee forward progress even when overlap is large.
        start = new_start if new_start > start else end
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
        "matched_passage": candidate["passage"],
        "similarity_score": candidate["similarity"],
    }
    meta = candidate["meta"]
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
        self, chroma_client: ChromaClient | None = None, config_path: str | None = None, db_path: str | None = None
    ):
        """
        Initialize semantic search.

        Args:
            chroma_client: Optional ChromaClient instance
            config_path: Path to configuration file
            db_path: Optional path to Zotero database (overrides config file)
        """
        self.chroma_client = chroma_client or create_chroma_client(config_path)
        self.zotero_client = get_zotero_client()
        self.config_path = config_path
        self.db_path = db_path  # CLI override for Zotero database path
        # Item keys seen by the most recent local sqlite scan (set by
        # _get_items_from_local_db); used to verify watermark promotion.
        self._last_scan_snapshot_keys: set[str] | None = None
        # Top-level keys that belong in the configured semantic corpus. Unlike
        # the snapshot set, this excludes deleted, child, filtered, and
        # deduplicated items and can be reconciled against ChromaDB.
        self._last_scan_indexable_keys: set[str] | None = None

        # Load update configuration
        self.update_config = self._load_update_config()

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
            "max_chunks_per_item": 20,
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
        return "chunks-v1:{size}:{overlap}:{maximum}".format(
            size=int(self._chunking_config.get("chunk_size", 1500)),
            overlap=int(self._chunking_config.get("overlap", 200)),
            maximum=int(self._chunking_config.get("max_chunks_per_item", 20)),
        )

    def _index_layout_changed(self, metadata: dict[str, Any]) -> bool:
        """Return whether an existing item needs migration to this layout."""
        stored = metadata.get("index_layout_signature")
        # Old item-level indexes did not carry a signature. Preserve their
        # historical skip behavior, while an enabled chunk layout migrates old
        # records once. A stamped chunked index also migrates when disabled.
        return (self._chunking_enabled or stored is not None) and (
            stored != self._index_layout_signature
        )

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

    def _load_fulltext_setting(self) -> bool:
        """Load whether local full-text extraction is enabled."""
        if not self.config_path or not os.path.exists(self.config_path):
            return False
        try:
            with open(self.config_path) as f:
                file_config = json.load(f)
                value = file_config.get("semantic_search", {}).get(
                    "fulltext", False
                )
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
                value = (
                    file_config
                    .get("semantic_search", {})
                    .get("openai_batch", {})
                    .get("enabled", False)
                )
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
                value = file_config.get("semantic_search", {}).get("last_sync_version", 0)
                return int(value) if value is not None else 0
        except Exception as e:
            logger.warning(f"Error loading last_sync_version: {e}")
            return 0

    def _load_indexed_fulltext(self) -> bool | None:
        """Return the full-text mode used for the last complete update."""
        if not self.config_path or not os.path.exists(self.config_path):
            return None
        try:
            with open(self.config_path) as f:
                value = (
                    json.load(f)
                    .get("semantic_search", {})
                    .get("indexed_fulltext")
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
                value = (
                    json.load(f)
                    .get("semantic_search", {})
                    .get("indexed_content_signature")
                )
            return value if isinstance(value, str) and value else None
        except Exception as e:
            logger.warning(f"Error loading indexed_content_signature: {e}")
            return None

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

        # Load existing config or create new one
        full_config = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    full_config = json.load(f)
            except Exception:
                pass

        # Update semantic search config
        if "semantic_search" not in full_config:
            full_config["semantic_search"] = {}

        full_config["semantic_search"]["update_config"] = self.update_config
        if last_sync_version is not None:
            full_config["semantic_search"]["last_sync_version"] = int(last_sync_version)
        if indexed_fulltext is not None:
            full_config["semantic_search"]["indexed_fulltext"] = (
                indexed_fulltext
            )
        if indexed_content_signature is not None:
            full_config["semantic_search"]["indexed_content_signature"] = (
                indexed_content_signature
            )

        try:
            with open(self.config_path, "w") as f:
                json.dump(full_config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving update config: {e}")

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
        return "\n\n".join(
            part.strip()
            for part in (title, abstract)
            if part and part.strip()
        )

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

        metadata = {
            "item_key": item.get("key", ""),
            "item_type": data.get("itemType", ""),
            "title": data.get("title", ""),
            "date": data.get("date", ""),
            "date_added": data.get("dateAdded", ""),
            "date_modified": data.get("dateModified", ""),
            "creators": format_creators(data.get("creators", [])),
            "publication": data.get("publicationTitle", ""),
            "url": data.get("url", ""),
            "doi": data.get("DOI", ""),
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

    def _get_items_from_source(
        self,
        limit: int | None = None,
        fulltext: bool = False,
        chroma_client: ChromaClient | None = None,
        force_rebuild: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Get canonical API metadata with optional local full text.

        Full-text extraction requires local mode (ZOTERO_LOCAL=true). Without
        it, this returns API metadata only and never requests cached full text.

        Args:
            limit: Optional limit on number of items
            fulltext: Whether to select and extract a local attachment
            chroma_client: ChromaDB client to check for existing documents (None to skip checks)
            force_rebuild: Whether to force extraction even if item exists

        Returns:
            List of items in API-compatible format
        """
        if not isinstance(fulltext, bool):
            raise ValueError("fulltext must be true or false")
        if fulltext:
            if not is_local_mode():
                raise RuntimeError(
                    "Fulltext extraction requires local mode but ZOTERO_LOCAL is not enabled. "
                    "Set ZOTERO_LOCAL=true or run 'zotero-mcp setup' to enable local mode."
                )
            return self._get_items_from_local_db(
                limit,
                extract_fulltext=True,
                chroma_client=chroma_client,
                force_rebuild=force_rebuild,
            )
        return self._get_items_from_api(limit)

    def _get_items_from_local_db(
        self,
        limit: int | None = None,
        extract_fulltext: bool = False,
        chroma_client: ChromaClient | None = None,
        force_rebuild: bool = False,
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

        try:
            api_metadata_by_key: dict[str, dict[str, Any]] = {}
            try:
                api_metadata_by_key = {
                    item.get("key", ""): item
                    for item in self._get_items_from_api()
                    if item.get("key")
                }
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
                # Capture the snapshot's full key set on the SAME connection
                # this scan uses. The staleness check after the (potentially
                # long) extraction must compare against what this scan could
                # actually see — a fresh read taken later could already
                # include rows from a WAL checkpoint that landed mid-scan.
                self._last_scan_snapshot_keys = reader.get_all_item_keys()
                # Phase 1: fetch metadata only (fast)
                sys.stderr.write("Scanning local Zotero database for items...\n")
                if collection_keys:
                    sys.stderr.write(f"Filtering to collections: {collection_keys}\n")
                local_items = reader.get_items_with_text(limit=limit, include_fulltext=False, collection_keys=collection_keys)
                candidate_count = len(local_items)
                sys.stderr.write(f"Found {candidate_count} candidate items.\n")

                # Optional deduplication: if preprint and journalArticle share a DOI/title, keep journalArticle
                # Build index by (normalized DOI or normalized title)
                def norm(s: str | None) -> str | None:
                    if not s:
                        return None
                    return "".join(s.lower().split())

                key_to_best = {}
                for it in local_items:
                    doi_key = ("doi", norm(getattr(it, "doi", None))) if getattr(it, "doi", None) else None
                    title_key = ("title", norm(getattr(it, "title", None))) if getattr(it, "title", None) else None

                    def consider(k):
                        if not k:
                            return
                        cur = key_to_best.get(k)
                        # Prefer journalArticle over preprint; otherwise keep first
                        if cur is None:
                            key_to_best[k] = it
                        else:
                            prefer_types = {"journalArticle": 2, "preprint": 1}
                            cur_score = prefer_types.get(getattr(cur, "item_type", ""), 0)
                            new_score = prefer_types.get(getattr(it, "item_type", ""), 0)
                            if new_score > cur_score:
                                key_to_best[k] = it

                    consider(doi_key)
                    consider(title_key)

                # If a preprint loses against a journal article for same DOI/title, drop it
                filtered_items = []
                for it in local_items:
                    # If there is a journalArticle alternative for same DOI or title, and this is preprint, drop
                    if getattr(it, "item_type", None) == "preprint":
                        k_doi = ("doi", norm(getattr(it, "doi", None))) if getattr(it, "doi", None) else None
                        k_title = ("title", norm(getattr(it, "title", None))) if getattr(it, "title", None) else None
                        drop = False
                        for k in (k_doi, k_title):
                            if not k:
                                continue
                            best = key_to_best.get(k)
                            if (
                                best is not None
                                and best is not it
                                and getattr(best, "item_type", None) == "journalArticle"
                            ):
                                drop = True
                                break
                        if drop:
                            continue
                    filtered_items.append(it)

                local_items = filtered_items
                self._last_scan_indexable_keys = {it.key for it in local_items}
                total_to_extract = len(local_items)
                if total_to_extract != candidate_count:
                    try:
                        sys.stderr.write(
                            f"After filtering/dedup: {total_to_extract} items to process. Extracting content...\n"
                        )
                    except Exception:
                        pass
                else:
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
                    _skipped_pdfs = []  # Collect timeout/error names for summary
                    _skipped_failed = []  # Items skipped because extraction previously failed

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
                        date = getattr(it, "date_added", "") or ""
                        first_author = ""
                        if creators:
                            first_author = creators.split(";")[0].split(",")[0].strip()
                            if first_author:
                                first_author += " et al." if ";" in creators else ""
                        year = ""
                        if date and len(date) >= 4:
                            year = date[:4]
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
                            prefix = (
                                f"  Processing {item_idx}/{total_local}{status} "
                                f"| ETA {eta} — "
                            )
                            line = f"{prefix}{display or 'working...'}"
                            _write_progress_line(sys.stderr, line)
                        except Exception:
                            pass

                        should_extract = True

                        # Current attachment-key set, stored in metadata so a
                        # later run can detect attachment changes. Attaching a
                        # file does NOT bump the parent's dateModified, so the
                        # date check alone never clears a "failed" marker.
                        att_keys = ",".join(
                            sorted(k for k, _p, _c in reader.get_fulltext_meta_for_item(it.item_id))
                        )
                        it._attachment_keys = att_keys
                        if hasattr(reader, "get_attachment_signature"):
                            attachment_signature = reader.get_attachment_signature(it.item_id)
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
                                canonical_data = api_metadata_by_key.get(
                                    it.key, {}
                                ).get("data", {})
                                item_date = (
                                    canonical_data.get("dateModified")
                                    or getattr(it, "date_modified", "")
                                    or ""
                                )
                                metadata_changed = chroma_date != item_date
                                stored_signature = existing_metadata.get("attachment_signature")
                                signature_changed = (
                                    stored_signature is not None
                                    and stored_signature != attachment_signature
                                )
                                layout_changed = self._index_layout_changed(existing_metadata)

                                # Skip if extraction previously failed AND neither the item
                                # nor its attachment set has changed since (handles both a
                                # replaced bad PDF and a PDF newly attached to an item that
                                # was indexed metadata-only)
                                if chroma_has_fulltext == "failed":
                                    stored_att_keys = existing_metadata.get("attachment_keys")
                                    attachment_unchanged = (
                                        stored_signature == attachment_signature
                                        if stored_signature is not None
                                        else stored_att_keys == att_keys
                                    )
                                    if (
                                        not metadata_changed
                                        and attachment_unchanged
                                        and not layout_changed
                                    ):
                                        # Nothing changed since the failure — don't retry
                                        should_extract = False
                                        skipped_existing += 1
                                        _skipped_failed.append(display or f"item {it.key}")
                                    else:
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
                                elif (
                                    metadata_changed
                                    or signature_changed
                                    or local_has_fulltext
                                    or layout_changed
                                ):
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
                                text = reader.extract_fulltext_for_item(it.item_id)
                                # Circuit breaker: stop PDF extraction after consecutive timeouts
                                if isinstance(text, tuple) and len(text) == 2 and text[1] == "timeout":
                                    _skipped_pdfs.append(display or f"item {it.key}")
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
                                    else:
                                        # Extraction returned empty — mark as attempted
                                        it._fulltext_attempted = True
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
                            sys.stderr.write(f"  ({updated_existing} items updated with new fulltext)\n")
                        if _skipped_pdfs:
                            sys.stderr.write(f"  Skipped {len(_skipped_pdfs)} PDF(s) (timed out):\n")
                            for name in _skipped_pdfs:
                                sys.stderr.write(f"    - {name}\n")
                        if _skipped_failed:
                            sys.stderr.write(
                                f"  {len(_skipped_failed)} item(s) skipped (PDF extraction previously failed):\n"
                            )
                            for name in _skipped_failed[:5]:  # Show first 5
                                sys.stderr.write(f"    - {name}\n")
                            if len(_skipped_failed) > 5:
                                sys.stderr.write(f"    ... and {len(_skipped_failed) - 5} more\n")
                            sys.stderr.write(
                                "  (To retry these, attach or replace the PDF, or run with --force-rebuild)\n"
                            )
                    except Exception:
                        pass

                    # Replace local_items with filtered list
                    local_items = items_to_process
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
            logger.info("Falling back to API...")
            return self._get_items_from_api(limit)

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
                "itemType": getattr(item, "item_type", None)
                or "journalArticle",
                "title": item.title or "",
                "abstractNote": item.abstract or "",
                "extra": item.extra or "",
                "dateAdded": item.date_added,
                "dateModified": item.date_modified,
                "creators": (
                    self._parse_creators_string(item.creators)
                    if item.creators
                    else []
                ),
            }
            if item.notes:
                data["notes"] = item.notes
            merged = {"key": item.key, "version": 0, "data": data}

        data["fulltext"] = (
            getattr(item, "fulltext", None) or ""
            if extract_fulltext
            else ""
        )
        data["fulltextSource"] = (
            getattr(item, "fulltext_source", None) or ""
            if extract_fulltext
            else ""
        )
        data["fulltext_attempted"] = getattr(
            item, "_fulltext_attempted", False
        )
        if (att := getattr(item, "_attachment_keys", None)) is not None:
            data["attachmentKeys"] = att
        if (
            signature := getattr(item, "_attachment_signature", None)
        ) is not None:
            data["attachmentSignature"] = signature
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

    def _get_items_from_api(
        self, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Get canonical item metadata from the Zotero API.

        Args:
            limit: Optional limit on number of items

        Returns:
            List of items from API
        """
        logger.info("Fetching items from Zotero API...")

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

            # Child artifacts are represented only through their parent item.
            filtered_items = [
                item
                for item in items
                if item.get("data", {}).get("itemType")
                not in {"attachment", "note", "annotation"}
            ]

            all_items.extend(filtered_items)
            start += batch_size

            if len(items) < batch_size:
                break

        if limit:
            all_items = all_items[:limit]

        logger.info(f"Retrieved {len(all_items)} items from API")
        return all_items

    def _get_changed_items_from_api(
        self, since_version: int
    ) -> tuple[list[dict[str, Any]], set[str] | None]:
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
        try:
            changed_versions = self.zotero_client.item_versions(since=since_version) or {}
        except Exception as e:
            raise Exception(f"Failed to fetch item_versions(since={since_version}): {e}") from e

        discovery_complete = True
        changed_keys = set(changed_versions.keys())

        try:
            current_versions = self.zotero_client.item_versions() or {}
        except Exception as e:
            logger.warning(f"Failed to fetch current item_versions for deletion check: {e}")
            current_versions = None
            discovery_complete = False
        current_keys = set(current_versions.keys()) if current_versions is not None else None

        if not changed_keys:
            return [], current_keys if discovery_complete else None

        changed_items: list[dict[str, Any]] = []
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
            if item_type in {"attachment", "note", "annotation"}:
                continue
            changed_items.append(item)

        return changed_items, current_keys if discovery_complete else None

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
                            zotero_db_path = (
                                json.load(f).get("semantic_search", {}).get("zotero_db_path")
                            )
                    except Exception:
                        pass
                with LocalZoteroReader(db_path=zotero_db_path) as reader:
                    snapshot_keys = reader.get_all_item_keys()
        except Exception as e:
            logger.warning(
                f"Could not verify local snapshot completeness ({e}); "
                "keeping previous sync watermark."
            )
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

        if self._chunking_enabled and hasattr(self.chroma_client, "delete_item_chunks"):
            for key in to_delete_keys:
                self.chroma_client.delete_item_chunks(key)
        else:
            self.chroma_client.delete_documents(to_delete_keys)
        return len(to_delete_keys)

    def _prepare_index_records(self, items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Prepare ChromaDB records without embedding or writing them."""
        prepared = self._prepare_item_batch(items)
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
        }
        return records, stats

    def _submit_openai_batch_index(
        self,
        items: list[dict[str, Any]],
        force_full_rebuild: bool,
        target_sync_version: int | None,
        stats: dict[str, Any],
    ) -> dict[str, Any]:
        """Prepare records and submit asynchronous OpenAI embedding batches."""
        records, prepare_stats = self._prepare_index_records(items)
        stats["processed_items"] += prepare_stats["processed"]
        stats["skipped_items"] += prepare_stats["skipped"]
        stats["errors"] += prepare_stats["errors"]

        if not records:
            stats["batch_submitted"] = False
            stats["batch_error"] = "No documents were prepared for OpenAI Batch API submission"
            return stats

        ids = [record["id"] for record in records]
        existing_ids = self.chroma_client.get_existing_ids(ids) if ids and not force_full_rebuild else set()
        model_name = self.chroma_client.embedding_config.get("model_name", "text-embedding-3-small")
        manifest = openai_batch.submit_embedding_batches(
            records=records,
            model_name=model_name,
            embedding_config=self.chroma_client.embedding_config,
            config_path=self.config_path,
            force_full_rebuild=force_full_rebuild,
            target_sync_version=target_sync_version,
            fulltext=self._active_fulltext,
            content_signature=_CONTENT_CONTRACT_SIGNATURE,
        )
        stats["batch_submitted"] = True
        stats["batch_run_id"] = manifest["run_id"]
        stats["batch_manifest"] = manifest["manifest_path"]
        stats["batch_ids"] = [batch["batch_id"] for batch in manifest.get("batches", [])]
        stats["submitted_items"] = len(records)
        stats["estimated_updated_items"] = len(existing_ids)
        stats["estimated_added_items"] = len(ids) - len(existing_ids)
        return stats

    def update_database(
        self,
        force_full_rebuild: bool = False,
        limit: int | None = None,
        fulltext: bool | None = None,
        use_openai_batch: bool | None = None,
        embedding_concurrency: int = 1,
    ) -> dict[str, Any]:
        """
        Update the semantic search database with Zotero items.

        Args:
            force_full_rebuild: Whether to rebuild the entire database
            limit: Limit number of items to process (for testing)
            fulltext: Whether to select and extract one local attachment.
                False indexes API title and abstract only. None uses the
                configured setting, which defaults to false.
            use_openai_batch: Override for OpenAI Batch API indexing. None
                uses `semantic_search.openai_batch.enabled`.
            embedding_concurrency: Number of realtime OpenAI-compatible
                embedding batches to run concurrently. ChromaDB reads and
                writes remain sequential. Defaults to 1.

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
            "recovered_items": 0,
            "skipped_items": 0,
            "deleted_items": 0,
            "errors": 0,
            "start_time": start_time.isoformat(),
            "duration": None,
        }

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
                    "flock should have released it; set ZOTERO_MCP_FORCE_UPDATE=1 "
                    "to bypass if this persists.",
                    lock_path,
                    holder_pid,
                )
            else:
                logger.warning(
                    "Another semantic-search update is already running "
                    "(lock held at %s by pid %s); skipping this invocation. "
                    "This is expected when the MCP server's background sync is "
                    "active. Set ZOTERO_MCP_FORCE_UPDATE=1 to override.",
                    lock_path,
                    holder_pid if holder_pid else "unknown",
                )
            stats["duration"] = "0:00:00"
            stats["skipped_reason"] = "another_update_in_progress"
            return stats

        try:
            if fulltext is None:
                fulltext = self._load_fulltext_setting()
            if not isinstance(fulltext, bool):
                raise ValueError("fulltext must be true or false")
            extract_fulltext = fulltext
            self._active_fulltext = fulltext
            stats["fulltext"] = fulltext
            indexed_fulltext = self._load_indexed_fulltext()
            indexed_content_signature = self._load_indexed_content_signature()
            try:
                collection_has_items = (
                    int(self.chroma_client.get_collection_info().get("count", 0)) > 0
                )
            except Exception:
                collection_has_items = False
            mode_mismatch = (
                not force_full_rebuild
                and collection_has_items
                and indexed_fulltext != fulltext
            )
            if mode_mismatch:
                previous = (
                    "enabled"
                    if indexed_fulltext is True
                    else "disabled"
                    if indexed_fulltext is False
                    else "unknown"
                )
                requested = "enabled" if fulltext else "disabled"
                sys.stderr.write(
                    f"WARNING: requested full-text mode ({requested}) differs "
                    f"from the existing index ({previous}). This incremental "
                    "run will not re-embed unchanged items; the index may "
                    "remain mixed until an explicitly confirmed force rebuild.\n"
                )
            content_mismatch = (
                not force_full_rebuild
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
            if embedding_concurrency < 1:
                raise ValueError("embedding_concurrency must be at least 1")
            if embedding_concurrency > 1:
                if use_openai_batch:
                    raise ValueError(
                        "embedding_concurrency cannot be combined with OpenAI Batch mode"
                    )
                if self.chroma_client.embedding_model != "openai":
                    raise ValueError(
                        "embedding_concurrency above 1 requires realtime "
                        "OpenAI-compatible embeddings"
                    )
                stats["embedding_concurrency"] = embedding_concurrency
                logger.info(
                    "Using %s concurrent realtime embedding batches",
                    embedding_concurrency,
                )

            # In batch mode, defer destructive rebuilds until import so the
            # existing search index remains usable while the batch runs.
            if force_full_rebuild and not use_openai_batch:
                logger.info("Force rebuilding database...")
                self.chroma_client.reset_collection()

            # Decide whether to use since-based incremental ingest.
            # Incremental requires: not a forced rebuild, not a local-extraction
            # run (incremental path covers web-API metadata and optionally
            # fulltext only), not a test limit, and a known prior sync version.
            last_sync_version = self._load_last_sync_version() if not force_full_rebuild else 0
            use_incremental = (
                not force_full_rebuild
                and not extract_fulltext
                and limit is None
                and last_sync_version > 0
            )

            # When a collection filter is configured, skip the API-based
            # incremental path: it fetches changed items from the WHOLE
            # library and its deletion pass compares against all library
            # keys, both of which would bypass the filter. The local
            # full-scan path applies collection_keys and skips
            # already-indexed items, so filtered updates stay cheap.
            configured_collection_keys = None
            try:
                if self.config_path and os.path.exists(self.config_path):
                    with open(self.config_path) as _f:
                        configured_collection_keys = (
                            json.load(_f).get("semantic_search", {}).get("collection_keys")
                        )
            except Exception:
                pass
            if configured_collection_keys and use_incremental:
                use_incremental = False
                try:
                    sys.stderr.write(
                        f"Collection filter active ({configured_collection_keys}); "
                        "using local full scan instead of API incremental update.\n"
                    )
                except Exception:
                    pass

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
                    indexed_fulltext=(
                        None if mode_mismatch else fulltext
                    ),
                    indexed_content_signature=(
                        None
                        if content_mismatch
                        else _CONTENT_CONTRACT_SIGNATURE
                    ),
                )
                end_time = datetime.now()
                stats["duration"] = str(end_time - start_time)
                stats["end_time"] = end_time.isoformat()
                return stats

            if use_incremental:
                all_items, current_library_keys = self._get_changed_items_from_api(
                    since_version=last_sync_version
                )
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
                        logger.warning(f"Deletion pass failed: {e}")
                        target_sync_version = None
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
                if extract_fulltext:
                    self._last_scan_indexable_keys = None
                all_items = self._get_items_from_source(
                    limit=limit,
                    fulltext=fulltext,
                    chroma_client=self.chroma_client if not force_full_rebuild else None,
                    force_rebuild=force_full_rebuild,
                )
                # The local-extraction scan may lag behind the API version
                # captured above (immutable sqlite reads skip WAL contents);
                # only promote the watermark if the snapshot was complete.
                if extract_fulltext and target_sync_version is not None:
                    verified_sync_version = self._verify_local_snapshot_version(
                        target_sync_version
                    )
                    local_snapshot_complete = verified_sync_version is not None
                    target_sync_version = verified_sync_version

                # Local full-text scans return only new/changed items for
                # embedding, so reconcile against the complete indexable key
                # set captured before that filtering. Never prune a partial,
                # stale, or unverifiable sqlite snapshot.
                if (
                    extract_fulltext
                    and not force_full_rebuild
                    and limit is None
                    and local_snapshot_complete
                    and self._last_scan_indexable_keys is not None
                ):
                    try:
                        deleted = self._delete_missing_index_items(
                            self._last_scan_indexable_keys
                        )
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

            stats["total_items"] = len(all_items)
            logger.info(f"Found {stats['total_items']} items to process")

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
                    stats=stats,
                )
                try:
                    batch_ids = ", ".join(stats.get("batch_ids", []))
                    sys.stderr.write(
                        "  Submitted OpenAI embedding batch"
                        f"{'es' if len(stats.get('batch_ids', [])) != 1 else ''}: {batch_ids}\n"
                    )
                    sys.stderr.write("  Run 'zotero-mcp openai-batch-status' to check progress.\n")
                    sys.stderr.write("  Run 'zotero-mcp openai-batch-import' after the batch completes.\n")
                except Exception:
                    pass
                end_time = datetime.now()
                stats["duration"] = str(end_time - start_time)
                stats["end_time"] = end_time.isoformat()
                return stats

            # User-friendly progress reporting
            total = stats["total_items"] = len(all_items)
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
            _failed_docs = []  # Collect failures for end-of-run retry
            failed_concurrent_batches: list[_PreparedIndexBatch] = []

            def report_item_progress(item: dict[str, Any]) -> None:
                nonlocal seen_items
                seen_items += 1
                title = item.get("data", {}).get("title", "")
                pct = int(seen_items / total * 100) if total else 0
                eta = _format_eta(indexing_eta.estimate(seen_items))
                try:
                    _write_progress_line(
                        sys.stderr,
                        f"  [{pct:3d}%] {seen_items}/{total} "
                        f"| ETA {eta} — {title or 'processing...'}",
                    )
                except Exception:
                    pass

            def accumulate_batch_stats(batch_stats: dict[str, int]) -> None:
                stats["processed_items"] += batch_stats["processed"]
                stats["added_items"] += batch_stats["added"]
                stats["updated_items"] += batch_stats["updated"]
                stats["skipped_items"] += batch_stats["skipped"]
                stats["errors"] += batch_stats["errors"]

                logger.info(
                    f"Processed {seen_items}/{total} items (added: {stats['added_items']}, skipped: {stats['skipped_items']})"
                )

            if embedding_concurrency == 1:
                for item in all_items:
                    item_started = time.monotonic()
                    batch_stats = self._process_item_batch(
                        [item],
                        force_full_rebuild,
                        _failed_docs,
                    )
                    indexing_eta.record(time.monotonic() - item_started)
                    report_item_progress(item)
                    accumulate_batch_stats(batch_stats)
            else:
                queue_depth = embedding_concurrency * 2
                work_queue: Queue[Any] = Queue(maxsize=queue_depth)
                result_queue: Queue[Any] = Queue(maxsize=queue_depth)
                stop_pipeline = threading.Event()
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
                                    embeddings = (
                                        self.chroma_client.embed_documents(
                                            prepared.documents
                                        )
                                    )
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
                    workers = [
                        executor.submit(embedding_worker)
                        for _ in range(embedding_concurrency)
                    ]
                    producer = threading.Thread(
                        target=prepare_entries,
                        name="zotero-embedding-producer",
                        daemon=True,
                    )
                    producer.start()

                    completed_workers = 0
                    try:
                        while completed_workers < embedding_concurrency:
                            result = result_queue.get()
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
                                    if prepared.documents:
                                        self._commit_prepared_batch(
                                            prepared,
                                            force_rebuild=force_full_rebuild,
                                            embeddings=embeddings,
                                        )
                                except Exception as e:
                                    error = e

                            if error is not None:
                                logger.warning(
                                    "Concurrent embedding entry failed (%s), "
                                    "saving it for a sequential retry",
                                    error,
                                )
                                prepared.stats["errors"] += len(
                                    prepared.documents
                                )
                                failed_concurrent_batches.append(prepared)

                            item_duration += time.monotonic() - commit_started
                            indexing_eta.record(item_duration)
                            report_item_progress(item)
                            accumulate_batch_stats(prepared.stats)
                    finally:
                        stop_pipeline.set()
                        producer.join()
                        for worker in workers:
                            worker.result()

                    if producer_errors:
                        raise producer_errors[0]

                if failed_concurrent_batches:
                    import time as _retry_time

                    failed_items = sum(
                        batch.stats["processed"]
                        for batch in failed_concurrent_batches
                    )
                    try:
                        sys.stderr.write(
                            f"\n  Retrying {failed_items} failed items sequentially...\n"
                        )
                    except Exception:
                        pass
                    _retry_time.sleep(1)

                    retry_ok = 0
                    retry_fail = 0
                    for prepared in failed_concurrent_batches:
                        try:
                            embeddings = self.chroma_client.embed_documents(
                                prepared.documents
                            )
                            self._commit_prepared_batch(
                                prepared,
                                force_rebuild=force_full_rebuild,
                                embeddings=embeddings,
                            )
                            stats["errors"] -= len(prepared.documents)
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
                        sys.stderr.write(
                            f"  Retry: {retry_ok} recovered, "
                            f"{retry_fail} still failed\n"
                        )
                    except Exception:
                        pass

            # Retry any documents that failed during the main run
            if _failed_docs:
                try:
                    _clear_progress_line(sys.stderr)
                    sys.stderr.write(f"\n  Retrying {len(_failed_docs)} failed items...\n")
                except Exception:
                    pass

                import time as _retry_time

                _retry_time.sleep(1)  # Brief pause before retry

                retry_ok = 0
                retry_fail = 0
                for doc, meta, doc_id in _failed_docs:
                    try:
                        self.chroma_client.upsert_documents([doc], [meta], [doc_id])
                        retry_ok += 1
                        stats["errors"] -= 1  # Remove from error count
                        # Don't classify as added vs updated — when the
                        # original batch failed, the add/update lookup never
                        # ran, so we don't know which category it belongs in.
                        # Track recovered items in their own bucket.
                        stats["recovered_items"] += 1
                    except Exception as e2:
                        retry_fail += 1
                        logger.error(f"Retry failed for {doc_id}: {e2}")

                try:
                    sys.stderr.write(f"  Retry: {retry_ok} recovered, {retry_fail} still failed\n")
                except Exception:
                    pass

            # Clear the progress line and show summary
            try:
                _clear_progress_line(sys.stderr)
                summary = (
                    f"  Done: {stats['processed_items']} indexed, "
                    f"{stats['skipped_items']} skipped, "
                    f"{stats['errors']} errors"
                )
                if stats["recovered_items"]:
                    summary += f", {stats['recovered_items']} recovered"
                sys.stderr.write(summary + "\n")
            except Exception:
                pass

            # Update last update time, and promote last_sync_version on success
            self.update_config["last_update"] = datetime.now().isoformat()
            completed_sync_version = target_sync_version if stats["errors"] == 0 else None
            completed_fulltext = (
                fulltext
                if stats["errors"] == 0
                and limit is None
                and completed_sync_version is not None
                and (
                    force_full_rebuild
                    or not collection_has_items
                    or not mode_mismatch
                )
                else None
            )
            completed_content_signature = (
                _CONTENT_CONTRACT_SIGNATURE
                if stats["errors"] == 0
                and limit is None
                and completed_sync_version is not None
                and (
                    force_full_rebuild
                    or not collection_has_items
                    or not content_mismatch
                )
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
            stats["error"] = str(e)
            end_time = datetime.now()
            stats["duration"] = str(end_time - start_time)
            return stats
        finally:
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
        stats = {"processed": 0, "added": 0, "updated": 0, "skipped": 0, "errors": 0}

        chunking = self._chunking_enabled
        chunk_size = int(self._chunking_config.get("chunk_size", 1500))
        overlap = int(self._chunking_config.get("overlap", 200))
        max_chunks = int(self._chunking_config.get("max_chunks_per_item", 20))

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
                self_contained = (
                    selected_fulltext_source
                    in _SELF_CONTAINED_FULLTEXT_SOURCES
                )
                metadata = self._create_metadata(item)
                metadata["index_layout_signature"] = self._index_layout_signature
                metadata["index_fulltext"] = self._active_fulltext
                metadata["index_content_signature"] = (
                    _CONTENT_CONTRACT_SIGNATURE
                )

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
                        page = (
                            _page_for_offset(source_text, c0)
                            if passage_kind == "body"
                            else None
                        )
                        if page is not None:
                            cmeta["page"] = page
                        documents.append(self.chroma_client.truncate_text(chunk_text))
                        metadatas.append(cmeta)
                        ids.append(f"{item_key}#{ci}")
                else:
                    if fulltext:
                        doc_text = (
                            fulltext
                            if self_contained or not metadata_text
                            else f"{metadata_text}\n\n{fulltext}"
                        )
                    else:
                        doc_text = metadata_text
                    if not doc_text:
                        stats["skipped"] += 1
                        continue
                    # Truncate to fit the configured embedding model's token limit
                    documents.append(self.chroma_client.truncate_text(doc_text))
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
            item_keys=item_keys_order,
            stats=stats,
        )

    def _commit_prepared_batch(
        self,
        prepared: _PreparedIndexBatch,
        force_rebuild: bool = False,
        _failed_docs: list | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> dict[str, int]:
        """Write one prepared batch, optionally using precomputed embeddings."""
        documents = prepared.documents
        metadatas = prepared.metadatas
        ids = prepared.ids
        item_keys_order = prepared.item_keys
        stats = prepared.stats

        if not documents:
            return stats
        if embeddings is not None and len(embeddings) != len(documents):
            raise ValueError(
                "Embedding provider returned "
                f"{len(embeddings)} vectors for {len(documents)} documents"
            )

        # Which items already existed (drives added-vs-updated). When
        # chunking, also clear an item's stale passages before re-adding so
        # a shrinking document never leaves orphaned chunks behind.
        existing_item_keys: set[str] = set()
        if not force_rebuild:
            if self._chunking_enabled:
                probe_ids = [f"{k}#0" for k in item_keys_order]
                existing_chunk0 = self.chroma_client.get_existing_ids(probe_ids)
                existing_item_keys = {
                    cid.split("#", 1)[0] for cid in existing_chunk0
                }
                if hasattr(self.chroma_client, "delete_item_chunks"):
                    for key in dict.fromkeys(item_keys_order):
                        try:
                            self.chroma_client.delete_item_chunks(key)
                        except Exception as e:
                            logger.debug(
                                "delete_item_chunks(%s) failed: %s",
                                key,
                                e,
                            )
            else:
                existing_item_keys = self.chroma_client.get_existing_ids(ids)

        try:
            if embeddings is None:
                self.chroma_client.upsert_documents(documents, metadatas, ids)
            else:
                self.chroma_client.upsert_embeddings(
                    documents,
                    metadatas,
                    ids,
                    embeddings,
                )
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
                for j in range(len(documents)):
                    _failed_docs.append((documents[j], metadatas[j], ids[j]))
                stats["errors"] += len(documents)
            else:
                raise

        return stats

    def _process_item_batch(
        self,
        items: list[dict[str, Any]],
        force_rebuild: bool = False,
        _failed_docs: list | None = None,
    ) -> dict[str, int]:
        """Prepare, embed, and write a batch through the sequential path."""
        prepared = self._prepare_item_batch(items)
        return self._commit_prepared_batch(
            prepared,
            force_rebuild=force_rebuild,
            _failed_docs=_failed_docs,
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
            batch for batch in manifest.get("batches", [])
            if not selected_ids or batch.get("batch_id") in selected_ids
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

    def import_openai_batch(self, batch_ids: list[str] | None = None) -> dict[str, Any]:
        """Import completed OpenAI Batch API embeddings into ChromaDB."""
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
        batches = [
            batch for batch in all_batches
            if not selected_ids or batch.get("batch_id") in selected_ids
        ]
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
            "errors": [],
        }

        lock_path = Path.home() / ".config" / "zotero-mcp" / "update.lock"
        lock_cm = _acquire_update_lock(lock_path)
        acquired = lock_cm.__enter__()
        if not acquired:
            lock_cm.__exit__(None, None, None)
            raise RuntimeError(f"Another semantic-search update is already running (lock held at {lock_path})")

        try:
            already_imported = any(batch.get("imported_at") for batch in all_batches)
            if (
                manifest.get("force_full_rebuild")
                and not already_imported
                and any(not batch.get("imported_at") for batch in batches)
            ):
                self.chroma_client.reset_collection()

            client = openai_batch.create_openai_client(self.chroma_client.embedding_config)
            for batch in batches:
                if batch.get("imported_at"):
                    stats["batches_skipped"] += 1
                    continue
                if batch.get("status") != "completed":
                    stats["batches_skipped"] += 1
                    stats["errors"].append({
                        "batch_id": batch.get("batch_id"),
                        "error": f"Batch status is {batch.get('status')}, not completed",
                    })
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
                    error_path = Path(batch["records_path"]).with_name(Path(batch["records_path"]).stem + "-errors.jsonl")
                    error_text = openai_batch.download_file_text(client, batch["error_file_id"], error_path)
                    row_failures.extend(openai_batch.parse_error_output(error_text))

                records = {record["id"]: record for record in openai_batch.read_jsonl(Path(batch["records_path"]))}
                ids = [doc_id for doc_id in embeddings_by_id if doc_id in records]
                unexpected_output_ids = [doc_id for doc_id in embeddings_by_id if doc_id not in records]
                failure_ids = {
                    failure.get("custom_id")
                    for failure in row_failures
                    if failure.get("custom_id")
                }
                missing_result_ids = [
                    doc_id
                    for doc_id in records
                    if doc_id not in embeddings_by_id and doc_id not in failure_ids
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
                stats["batches_imported"] += 1

            openai_batch.save_manifest(manifest)
            if all(batch.get("imported_at") for batch in all_batches):
                self.update_config["last_update"] = datetime.now().isoformat()
                self._save_update_config(
                    last_sync_version=manifest.get("target_sync_version"),
                    indexed_fulltext=manifest.get("fulltext"),
                    indexed_content_signature=manifest.get(
                        "content_signature"
                    ),
                )
            return stats
        finally:
            lock_cm.__exit__(None, None, None)

    def search(self,
               query: str,
               limit: int = 10,
               filters: dict[str, Any] | None = None) -> dict[str, Any]:
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
                max_fetch = max(fetch_limit, limit * 20)
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
            similarity = (1 - distance) if distance is not None else 0
            meta = meta if isinstance(meta, dict) else {}
            global_start = (
                int(meta["char_start"]) + passage_offset
                if isinstance(meta.get("char_start"), int)
                else passage_offset
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
                    if any(
                        _passages_overlap(candidate, chosen)
                        for chosen in selected
                    ):
                        continue
                    selected.append(candidate)

            support_weights = (0.10, 0.05)
            support_bonus = sum(
                weight * min(1.0, max(0.0, candidate["similarity"]) / best_score)
                for weight, candidate in zip(support_weights, selected[1:])
            ) if best_score > 0 else 0.0
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
                enriched_result["best_chunk_similarity_score"] = best_score
                enriched_result["supporting_chunk_count"] = len(selected) - 1
                enriched_result["matched_passages"] = [
                    _passage_result(candidate) for candidate in selected
                ]
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

        return {
            "collection_info": collection_info,
            "update_config": self.update_config,
            "openai_batch": {
                "enabled": self._load_openai_batch_enabled(),
                "active": self._resolve_openai_batch_enabled(None),
            },
            "should_update": self.should_update_database(),
            "last_update": self.update_config.get("last_update"),
        }

    def delete_item(self, item_key: str) -> bool:
        """Delete an item from the semantic search database."""
        try:
            self.chroma_client.delete_documents([item_key])
            return True
        except Exception as e:
            logger.error(f"Error deleting item {item_key}: {e}")
            return False


def create_semantic_search(config_path: str | None = None, db_path: str | None = None) -> ZoteroSemanticSearch:
    """
    Create a ZoteroSemanticSearch instance.

    Args:
        config_path: Path to configuration file
        db_path: Optional path to Zotero database (overrides config file)

    Returns:
        Configured ZoteroSemanticSearch instance
    """
    return ZoteroSemanticSearch(config_path=config_path, db_path=db_path)
