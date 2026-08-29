"""
Zotero client wrapper for MCP server.
"""

import functools
import math
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from markitdown import MarkItDown
from pyzotero import zotero

from zotero_mcp._file_lock import acquire_file_lock, release_file_lock
from zotero_mcp.local_api import (
    LocalApiHttpClient,
    LocalApiWriteUnavailableError,
    LocalWriteZotero,
    local_api_endpoint,
    writable_local_api_endpoints,
)
from zotero_mcp.utils import format_creators
from zotero_mcp.webdav import (
    WebDAVNotConfiguredError,
    download_attachment_from_webdav,
)

# Load environment variables
load_dotenv()

# Serialize all Zotero API access. The local API (port 23119) is single-threaded;
# concurrent requests from parallel MCP tool threads or a separate CLI process
# queue at the network layer and risk hitting pyzotero's 30s timeout. The RLock
# handles threads and a file lock coordinates processes. Nested calls from the
# same thread acquire both as one reentrant critical section.
_zotero_api_lock = threading.RLock()
_zotero_api_lock_state = threading.local()

# Bound how long a tool will WAIT to acquire the lock before giving up. Without a
# bound, a single slow/stuck op (e.g. a hung cloud write or PDF upload) holds the
# lock and every other tool — reads included — blocks behind it until FastMCP's
# ~60s client timeout fires, surfacing as an opaque "-32001 Request timed out" on
# every queued call. A bounded acquire turns that into a fast, actionable error
# for the *waiters* while leaving the in-flight op untouched. Keep this safely
# below the client timeout. Override via ZOTERO_MCP_LOCK_TIMEOUT (seconds; <=0
# restores the old unbounded behaviour).
_DEFAULT_LOCK_TIMEOUT = 45.0


class _SerializedCallProxy:
    """Serialize callable access to a wrapped Zotero/httpx client.

    Wrapping the clients makes the fundamental guarantee true even for new
    tools and direct ``client.patch`` calls. Locking each request, rather than
    an entire tool workflow, keeps model calls, file conversion, and external
    downloads from blocking unrelated Zotero reads for minutes or hours.
    """

    __slots__ = ("_wrapped",)

    def __init__(self, wrapped: Any):
        object.__setattr__(self, "_wrapped", wrapped)

    def __getattr__(self, name: str) -> Any:
        value = getattr(object.__getattribute__(self, "_wrapped"), name)
        if name == "client" and value is not None:
            return _SerializedCallProxy(value)
        if not callable(value):
            return value

        @functools.wraps(value)
        def locked_call(*args, **kwargs):
            return _call_with_zotero_api_lock(value, *args, **kwargs)

        return locked_call

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_wrapped"), name, value)

    def __repr__(self) -> str:
        return repr(object.__getattribute__(self, "_wrapped"))


def _lock_timeout() -> float:
    raw = os.getenv("ZOTERO_MCP_LOCK_TIMEOUT", "").strip()
    if not raw:
        return _DEFAULT_LOCK_TIMEOUT
    try:
        timeout = float(raw)
    except ValueError:
        return _DEFAULT_LOCK_TIMEOUT
    if not math.isfinite(timeout) or timeout > threading.TIMEOUT_MAX:
        return _DEFAULT_LOCK_TIMEOUT
    return timeout


def _api_lock_path() -> Path:
    configured = os.getenv("ZOTERO_MCP_API_LOCK_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".config" / "zotero-mcp" / "zotero-api.lock"


class ZoteroApiBusyError(RuntimeError):
    """Raised when the cross-process Zotero API lock can't be acquired in time.

    Signals that another Zotero operation is still in flight (likely slow or
    stuck) — not that this call itself failed. Callers should surface a clear,
    retryable message rather than letting the request hang to a timeout.
    """


def _busy_error(timeout: float) -> ZoteroApiBusyError:
    return ZoteroApiBusyError(
        f"Another Zotero API operation is still in progress and did not "
        f"release within {timeout:.0f}s. This usually means a previous "
        f"call is slow or stuck (e.g. a large PDF upload or an "
        f"unreachable Zotero cloud). Please retry shortly; if it "
        f"persists, restart the Zotero MCP server."
    )


def _acquire_cross_process_api_lock(
    *,
    timeout: float,
    deadline: float | None,
):
    """Acquire the API file lock within the caller's shared timeout budget."""
    path = _api_lock_path()
    if timeout <= 0:
        return acquire_file_lock(path, exclusive=True, blocking=True)

    while True:
        lock_file = acquire_file_lock(path, exclusive=True, blocking=False)
        if lock_file is not None:
            return lock_file
        remaining = (deadline or time.monotonic()) - time.monotonic()
        if remaining <= 0:
            raise _busy_error(timeout)
        time.sleep(min(0.05, remaining))


def _call_with_zotero_api_lock(func, *args, **kwargs):
    """Call ``func`` while holding the bounded, cross-process API lock."""
    timeout = _lock_timeout()
    deadline = time.monotonic() + timeout if timeout > 0 else None
    if timeout <= 0:
        _zotero_api_lock.acquire()
        acquired = True
    else:
        acquired = _zotero_api_lock.acquire(timeout=timeout)
    if not acquired:
        raise _busy_error(timeout)

    depth = getattr(_zotero_api_lock_state, "depth", 0)
    try:
        try:
            if depth == 0:
                _zotero_api_lock_state.file = _acquire_cross_process_api_lock(
                    timeout=timeout,
                    deadline=deadline,
                )
            _zotero_api_lock_state.depth = depth + 1
            return func(*args, **kwargs)
        finally:
            if getattr(_zotero_api_lock_state, "depth", 0) > 0:
                _zotero_api_lock_state.depth -= 1
                if _zotero_api_lock_state.depth == 0:
                    lock_file = getattr(_zotero_api_lock_state, "file", None)
                    _zotero_api_lock_state.file = None
                    if lock_file is not None:
                        release_file_lock(lock_file)
    finally:
        _zotero_api_lock.release()


def with_zotero_api_lock(func):
    """Explicitly serialize a short callable across MCP tool threads.

    API clients already serialize individual requests through
    :class:`_SerializedCallProxy`. This decorator remains for callers that need
    to group a deliberately short critical section; it must not wrap model,
    conversion, download, or other potentially long-running work.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return _call_with_zotero_api_lock(func, *args, **kwargs)

    return wrapper


def call_with_zotero_api_lock(func, *args, **kwargs):
    """Serialize one direct Zotero HTTP call through the shared API lock."""
    return _call_with_zotero_api_lock(func, *args, **kwargs)


# Direct-call fallback used by the standalone CLI and unit tests, where no MCP
# session exists. Real MCP calls use the bounded session map below.
_active_library_override: dict[str, str] = {}

_session_library_overrides: OrderedDict[str, dict[str, str]] = OrderedDict()
_session_library_lock = threading.Lock()
_MAX_SESSION_LIBRARY_OVERRIDES = 1024


def _current_mcp_session_id() -> str | None:
    """Return the active FastMCP session ID, or ``None`` outside a request."""
    try:
        from fastmcp.server.dependencies import get_context

        return get_context().session_id
    except Exception:
        return None


def _normalize_library_type(library_type: str) -> str:
    value = (library_type or "user").strip().lower()
    return {"users": "user", "groups": "group", "feeds": "feed"}.get(value, value)


def get_default_library() -> dict[str, str]:
    """Return the process configuration's fixed default library identity."""
    local = os.getenv("ZOTERO_LOCAL", "").lower() in {"true", "yes", "1"}
    return {
        "library_id": os.getenv("ZOTERO_LIBRARY_ID") or ("0" if local else ""),
        "library_type": _normalize_library_type(os.getenv("ZOTERO_LIBRARY_TYPE", "user")),
    }


def library_identity(library: dict[str, str] | None = None) -> str:
    """Return a stable identity suitable for index/config namespacing."""
    current = library if library is not None else get_current_library()
    return f"{_normalize_library_type(current.get('library_type', 'user'))}:{current.get('library_id', '')}"


def set_active_library(
    library_id: str,
    library_type: str,
    *,
    session_id: str | None = None,
) -> None:
    """Set a session-local library override, or the direct-call fallback."""
    override = {
        "library_id": str(library_id),
        "library_type": _normalize_library_type(library_type),
    }
    sid = session_id or _current_mcp_session_id()
    if sid is None:
        _active_library_override.clear()
        _active_library_override.update(override)
        return
    with _session_library_lock:
        _session_library_overrides.pop(sid, None)
        _session_library_overrides[sid] = override
        while len(_session_library_overrides) > _MAX_SESSION_LIBRARY_OVERRIDES:
            _session_library_overrides.popitem(last=False)


def clear_active_library(*, session_id: str | None = None) -> None:
    """Clear the current session override and return to configured defaults."""
    sid = session_id or _current_mcp_session_id()
    if sid is None:
        _active_library_override.clear()
        return
    with _session_library_lock:
        _session_library_overrides.pop(sid, None)


def get_active_library(*, session_id: str | None = None) -> dict[str, str]:
    """Return the current session override (empty when defaults are active)."""
    sid = session_id or _current_mcp_session_id()
    if sid is None:
        return dict(_active_library_override)
    with _session_library_lock:
        override = _session_library_overrides.get(sid)
        if override is None:
            return {}
        _session_library_overrides.move_to_end(sid)
        return dict(override)


def get_current_library() -> dict[str, str]:
    """Return the effective library identity for the current request."""
    return get_active_library() or get_default_library()


def _make_local_http_client(
    *,
    authorize_writes: bool = False,
    endpoint: str | None = None,
) -> httpx.Client:
    """Return an httpx.Client pinned to HTTP/1.1 for the local Zotero server.

    Zotero 8's local server (port 23119) only speaks HTTP/1.0. httpx defaults
    to attempting HTTP/2 negotiation, which the local server rejects with 502
    Bad Gateway — every tool call fails even though the MCP starts cleanly
    (#160). Forcing http1=True / http2=False on the transport keeps requests
    on HTTP/1.1 and the local API answers normally.
    """
    return LocalApiHttpClient(
        authorize_writes=authorize_writes,
        endpoint=endpoint,
    )


def _configure_local_endpoint(
    client: zotero.Zotero,
    endpoint: str | None = None,
) -> zotero.Zotero:
    """Point pyzotero at the configured local port."""
    client.endpoint = endpoint or local_api_endpoint()
    return client


@dataclass
class AttachmentDetails:
    """Details about a Zotero attachment."""

    key: str
    title: str
    filename: str
    content_type: str


@dataclass
class AttachmentDownloadResult:
    """Result of downloading an attachment from one of the supported sources."""

    path: Path | None
    source: str | None
    errors: list[str]


def get_zotero_client() -> zotero.Zotero:
    """
    Get authenticated Zotero client using environment variables.

    If a runtime library override is active (via set_active_library()),
    those values take precedence over environment variables.

    Returns:
        A configured Zotero client instance.

    Raises:
        ValueError: If required environment variables are missing.
    """
    # Runtime overrides take precedence over environment variables
    override = get_active_library()
    library_id = override.get("library_id") or os.getenv("ZOTERO_LIBRARY_ID")
    library_type = override.get("library_type") or os.getenv("ZOTERO_LIBRARY_TYPE", "user")
    api_key = os.getenv("ZOTERO_API_KEY")
    local = os.getenv("ZOTERO_LOCAL", "").lower() in ["true", "yes", "1"]

    # For local API, default to user ID 0 if not specified
    if local and not library_id:
        library_id = "0"

    # For remote API, we need both library_id and api_key
    if not local and not (library_id and api_key):
        raise ValueError(
            "Missing required environment variables. Please set ZOTERO_LIBRARY_ID and ZOTERO_API_KEY, "
            "or use ZOTERO_LOCAL=true for local Zotero instance."
        )

    raw_client = zotero.Zotero(
            library_id=library_id,
            library_type=library_type,
            # zotero.org keys and Local API keys are unrelated. Never send a
            # cloud credential to localhost; authenticated local writes obtain
            # their own key through /api/local/authorize.
            api_key=None if local else api_key,
            local=local,
            client=_make_local_http_client() if local else None,
        )
    if local:
        _configure_local_endpoint(raw_client)
    return _SerializedCallProxy(raw_client)


def get_local_zotero_client() -> zotero.Zotero | None:
    """
    Get a local Zotero client for file access (WebDAV/local storage).

    This client connects to the local Zotero instance running on port 23119.
    It's useful for accessing PDF files stored via WebDAV when the main
    client is configured for web API.

    Returns:
        A local Zotero client instance, or None if local Zotero is not available.
    """
    try:
        current = get_current_library()
        library_id = current.get("library_id") or "0"
        library_type = current.get("library_type") or "user"
        # HTTP/1.1-only transport for compatibility with Zotero 8's local
        # server (#160) — httpx default HTTP/2 negotiation returns 502.
        raw_client = _configure_local_endpoint(
            zotero.Zotero(
                library_id=library_id,
                library_type=library_type,
                api_key=None,
                local=True,
                client=_make_local_http_client(),
            )
        )
        client = _SerializedCallProxy(raw_client)
        # Test connection by making a simple request
        client.items(limit=1)
        return client
    except Exception:
        return None


def get_local_write_zotero_client() -> zotero.Zotero | None:
    """Return a remote-first writable Local API client when available.

    A configured remote-local endpoint is probed first. The server-local API is
    used only if the remote endpoint is unreachable or lacks write support.
    Authorization remains lazy until the first actual write request.
    """
    current = get_current_library()
    library_id = current.get("library_id") or "0"
    library_type = current.get("library_type") or "user"
    for endpoint_role, endpoint in writable_local_api_endpoints():
        http_client = None
        try:
            http_client = _make_local_http_client(
                authorize_writes=True,
                endpoint=endpoint,
            )
            server_id = _call_with_zotero_api_lock(http_client.ensure_server_id)
            raw_client = _configure_local_endpoint(
                LocalWriteZotero(
                    library_id=library_id,
                    library_type=library_type,
                    api_key=None,
                    local=True,
                    client=http_client,
                ),
                endpoint,
            )
            raw_client.local_server_id = server_id
            raw_client.local_endpoint_role = endpoint_role
            raw_client.local_endpoint = endpoint
            return _SerializedCallProxy(raw_client)
        except (
            LocalApiWriteUnavailableError,
            httpx.HTTPError,
            OSError,
        ):
            if http_client is not None:
                http_client.close()
            continue
    return None


def get_zotero_server_id(client: zotero.Zotero | None = None) -> str | None:
    """Return the current Local API server ID, probing once when necessary."""
    zot = client or get_zotero_client()
    if not getattr(zot, "local", False):
        return None
    ensure_server_id = getattr(getattr(zot, "client", None), "ensure_server_id", None)
    if not callable(ensure_server_id):
        return None
    try:
        return str(ensure_server_id())
    except Exception:
        return None


def get_web_zotero_client() -> zotero.Zotero | None:
    """
    Get a web API Zotero client for write operations.

    This client connects to the Zotero web API and can create/modify items.
    Requires ZOTERO_API_KEY and ZOTERO_LIBRARY_ID environment variables.

    Returns:
        A web API Zotero client instance, or None if credentials are not available.
    """
    current = get_current_library()
    library_id = current.get("library_id") or os.getenv("ZOTERO_LIBRARY_ID")
    library_type = current.get("library_type") or os.getenv("ZOTERO_LIBRARY_TYPE", "user")
    api_key = os.getenv("ZOTERO_API_KEY")

    if not library_id or not api_key:
        return None

    return _SerializedCallProxy(
        zotero.Zotero(
            library_id=library_id,
            library_type=library_type,
            api_key=api_key,
            local=False,
        )
    )


def is_local_zotero_available() -> bool:
    """Check if local Zotero instance is running and accessible."""
    client = get_local_zotero_client()
    return client is not None


def format_item_metadata(item: dict[str, Any], include_abstract: bool = True) -> str:
    """
    Format a Zotero item's metadata as markdown.

    Args:
        item: A Zotero item dictionary.
        include_abstract: Whether to include the abstract in the output.

    Returns:
        Markdown-formatted metadata.
    """
    data = item.get("data", {})
    item_type = data.get("itemType", "unknown")

    # Basic information
    lines = [
        f"# {data.get('title', 'Untitled')}",
        f"**Type:** {item_type}",
        f"**Item Key:** {data.get('key')}",
    ]

    # Trash status. The Zotero web API returns data.deleted=1 for items in
    # the Trash; prior versions silently rendered trashed items as if live,
    # so agents reasoning about "current" state could cite papers the user
    # had explicitly removed. Surface it near the top where it's hard to miss.
    if data.get("deleted"):
        lines.append("**Status:** 🗑️ In Trash (recoverable from Zotero Trash view)")

    # Date
    if date := data.get("date"):
        lines.append(f"**Date:** {date}")

    # Authors/Creators
    if creators := data.get("creators", []):
        lines.append(f"**Authors:** {format_creators(creators)}")

    # Publication details based on item type
    if item_type == "journalArticle":
        if journal := data.get("publicationTitle"):
            journal_info = f"**Journal:** {journal}"
            if volume := data.get("volume"):
                journal_info += f", Volume {volume}"
            if issue := data.get("issue"):
                journal_info += f", Issue {issue}"
            if pages := data.get("pages"):
                journal_info += f", Pages {pages}"
            lines.append(journal_info)
    elif item_type == "bookSection":
        if book_title := data.get("bookTitle"):
            lines.append(f"**Book:** {book_title}")
        if pages := data.get("pages"):
            lines.append(f"**Pages:** {pages}")

    # Publisher and place — emitted as independent labeled lines for any
    # item type that has them (book, bookSection, thesis, report, etc.).
    # Round-trip parity: agents that read these need a stable, labeled form.
    if publisher := data.get("publisher"):
        lines.append(f"**Publisher:** {publisher}")
    if place := data.get("place"):
        lines.append(f"**Place:** {place}")

    # Identifiers and URL
    if doi := data.get("DOI"):
        lines.append(f"**DOI:** {doi}")
    if isbn := data.get("ISBN"):
        lines.append(f"**ISBN:** {isbn}")
    if issn := data.get("ISSN"):
        lines.append(f"**ISSN:** {issn}")
    if url := data.get("url"):
        lines.append(f"**URL:** {url}")

    # Extra field often holds citation key / misc metadata
    if extra := data.get("extra"):
        lines.extend(["", "## Extra", extra])

        # Try to surface a citation key if present in Extra
        for line in extra.splitlines():
            if "citation key" in line.lower():
                key_part = line.split(":", 1)[1].strip() if ":" in line else line.strip()
                lines.append(f"**Citation Key (from Extra):** {key_part}")
                break

    # Tags
    if tags := data.get("tags"):
        tag_list = [f"`{tag['tag']}`" for tag in tags]
        if tag_list:
            lines.append(f"**Tags:** {' '.join(tag_list)}")

    # Abstract
    if include_abstract and (abstract := data.get("abstractNote")):
        lines.extend(["", "## Abstract", abstract])

    # Related Items (dc:relation URIs → item keys)
    dc_relations = data.get("relations", {}).get("dc:relation", [])
    if isinstance(dc_relations, str):
        dc_relations = [dc_relations]
    if dc_relations:
        related_keys = [uri.rstrip("/").split("/")[-1] for uri in dc_relations]
        lines.extend(["", "## Related Items", *[f"- {k}" for k in related_keys]])

    # Collections — list actual keys rather than a bare count. The Zotero
    # web API does NOT cascade collection-delete to items, so the array
    # can contain dangling references to collections that no longer exist.
    # Showing the keys lets agents verify against zotero_search_collections
    # instead of trusting a potentially stale count.
    if collections := data.get("collections", []):
        lines.append(f"**Collections:** {', '.join(collections)}")

    # Notes - this requires additional API calls, so we just indicate if there are notes
    if "meta" in item and item["meta"].get("numChildren", 0) > 0:
        lines.append(f"**Notes/Attachments:** {item['meta']['numChildren']}")

    return "\n\n".join(lines)


def generate_bibtex(item: dict[str, Any]) -> str:
    """
    Generate BibTeX format for a Zotero item.

    Args:
        item: Zotero item data

    Returns:
        BibTeX formatted string
    """
    data = item.get("data", {})
    item_key = data.get("key")

    # Try Better BibTeX first
    try:
        from zotero_mcp.better_bibtex_client import ZoteroBetterBibTexAPI

        bibtex = ZoteroBetterBibTexAPI()

        if bibtex.is_zotero_running():
            return bibtex.export_bibtex(item_key)

    except Exception:
        # Continue to fallback method if Better BibTeX fails
        pass

    # Fallback to basic BibTeX generation
    item_type = data.get("itemType", "misc")

    if item_type in ["attachment", "note"]:
        raise ValueError(f"Cannot export BibTeX for item type '{item_type}'")

    # Map Zotero item types to BibTeX types
    type_map = {
        "journalArticle": "article",
        "book": "book",
        "bookSection": "incollection",
        "conferencePaper": "inproceedings",
        "thesis": "phdthesis",
        "report": "techreport",
        "webpage": "misc",
        "manuscript": "unpublished",
    }

    # Create citation key
    creators = data.get("creators", [])
    author = ""
    if creators:
        first = creators[0]
        author = first.get("lastName", first.get("name", "").split()[-1] if first.get("name") else "").replace(" ", "")

    year = data.get("date", "")[:4] if data.get("date") else "nodate"
    cite_key = f"{author}{year}_{item_key}"

    # Build BibTeX entry
    bib_type = type_map.get(item_type, "misc")
    lines = [f"@{bib_type}{{{cite_key},"]

    # Add fields
    field_mappings = [
        ("title", "title"),
        ("publicationTitle", "journal"),
        ("bookTitle", "booktitle"),
        ("volume", "volume"),
        ("issue", "number"),
        ("pages", "pages"),
        ("publisher", "publisher"),
        ("place", "address"),
        ("DOI", "doi"),
        ("url", "url"),
        ("abstractNote", "abstract"),
    ]

    for zotero_field, bibtex_field in field_mappings:
        if value := data.get(zotero_field):
            # Escape special characters
            value = value.replace("{", "\\{").replace("}", "\\}")
            lines.append(f"  {bibtex_field} = {{{value}}},")

    # Add authors
    if creators:
        authors = []
        for creator in creators:
            if creator.get("creatorType") == "author":
                if "lastName" in creator and "firstName" in creator:
                    authors.append(f"{creator['lastName']}, {creator['firstName']}")
                elif "name" in creator:
                    authors.append(creator["name"])
        if authors:
            lines.append(f"  author = {{{' and '.join(authors)}}},")

    # Add year
    if year != "nodate":
        lines.append(f"  year = {{{year}}},")

    # Remove trailing comma from last field and close entry
    if lines[-1].endswith(","):
        lines[-1] = lines[-1][:-1]
    lines.append("}")

    return "\n".join(lines)


def get_attachment_details(zot: zotero.Zotero, item: dict[str, Any]) -> AttachmentDetails | None:
    """
    Get attachment details for a Zotero item, finding the most relevant attachment.

    Args:
        zot: A Zotero client instance.
        item: A Zotero item dictionary.

    Returns:
        AttachmentDetails if found, None otherwise.
    """
    data = item.get("data", {})
    item_type = data.get("itemType")
    item_key = data.get("key")

    # Direct attachment
    if item_type == "attachment":
        return AttachmentDetails(
            key=item_key,
            title=data.get("title", "Untitled"),
            filename=data.get("filename", ""),
            content_type=data.get("contentType", ""),
        )

    # For regular items, look for child attachments
    try:
        children = zot.children(item_key)

        # Group attachments by content type
        pdfs = []
        htmls = []
        others = []

        for child in children:
            child_data = child.get("data", {})
            if child_data.get("itemType") == "attachment":
                content_type = child_data.get("contentType", "")
                filename = child_data.get("filename", "")
                title = child_data.get("title", "Untitled")
                key = child.get("key", "")

                # Use MD5 as proxy for size (longer MD5 usually means larger file)
                size_proxy = len(child_data.get("md5", ""))

                attachment = (key, title, filename, content_type, size_proxy)

                if content_type == "application/pdf":
                    pdfs.append(attachment)
                elif content_type.startswith("text/html"):
                    htmls.append(attachment)
                else:
                    others.append(attachment)

        # Return first match in priority order (PDF > HTML > other)
        # Sort each category by size (descending) to get largest/most complete file
        for category in [pdfs, htmls, others]:
            if category:
                category.sort(key=lambda x: x[4], reverse=True)
                key, title, filename, content_type, _ = category[0]
                return AttachmentDetails(
                    key=key,
                    title=title,
                    filename=filename,
                    content_type=content_type,
                )
    except Exception:
        pass

    return None


def get_pdf_attachment_details(
    zot: zotero.Zotero,
    item: dict[str, Any],
) -> AttachmentDetails | None:
    """Resolve one PDF deterministically from a parent or attachment item.

    An explicitly supplied PDF attachment always wins. For a parent item,
    OCR-labeled PDFs are preferred, then full-text-labeled PDFs, then other
    PDFs. Recency and item key provide stable tie breakers.
    """
    data = item.get("data", {})
    if data.get("itemType") == "attachment":
        content_type = data.get("contentType", "")
        filename = data.get("filename", "")
        if content_type != "application/pdf" and not filename.lower().endswith(
            ".pdf"
        ):
            return None
        return AttachmentDetails(
            key=item.get("key") or data.get("key", ""),
            title=data.get("title", "Untitled"),
            filename=filename,
            content_type=content_type,
        )

    candidates = []
    for child in zot.children(item.get("key") or data.get("key", "")):
        child_data = child.get("data", {})
        content_type = child_data.get("contentType", "")
        filename = child_data.get("filename", "")
        if child_data.get("itemType") != "attachment":
            continue
        if content_type != "application/pdf" and not filename.lower().endswith(
            ".pdf"
        ):
            continue
        label = re.sub(
            r"[^a-z0-9]+",
            " ",
            f"{child_data.get('title', '')} {filename}".casefold(),
        ).strip()
        words = set(label.split())
        preference = 2 if "ocr" in words else 1 if (
            "fulltext" in words or {"full", "text"} <= words
        ) else 0
        candidates.append(
            (
                preference,
                child_data.get("dateModified", ""),
                child.get("key", ""),
                child_data,
            )
        )
    if not candidates:
        return None
    _, _, key, selected = max(candidates, key=lambda value: value[:3])
    return AttachmentDetails(
        key=key,
        title=selected.get("title", "Untitled"),
        filename=selected.get("filename", ""),
        content_type=selected.get("contentType", ""),
    )


def download_attachment_file(
    attachment_key: str,
    destination_dir: str | Path,
    filename: str | None = None,
    *,
    local_client: zotero.Zotero | None = None,
    web_client: zotero.Zotero | None = None,
    enable_webdav: bool = True,
) -> AttachmentDownloadResult:
    """
    Download an attachment using the best available source.

    The fallback order is:
    1. local Zotero API (works with local storage or desktop-managed WebDAV)
    2. Direct WebDAV access via environment variables
    3. Zotero Web API (works with Zotero cloud storage)
    """
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    target_name = Path(filename or f"{attachment_key}.bin").name
    target_path = destination / target_name
    errors: list[str] = []

    def _cleanup_target() -> None:
        if target_path.exists() and target_path.stat().st_size == 0:
            target_path.unlink()

    def _try_dump(label: str, zot_client: zotero.Zotero | None) -> AttachmentDownloadResult | None:
        if zot_client is None:
            return None

        try:
            zot_client.dump(attachment_key, filename=target_name, path=str(destination))
            if target_path.exists() and target_path.stat().st_size > 0:
                return AttachmentDownloadResult(
                    path=target_path,
                    source=label,
                    errors=errors,
                )
            errors.append(f"{label}: file was not created")
        except Exception as exc:
            errors.append(f"{label}: {exc}")
        finally:
            _cleanup_target()

        return None

    local_result = _try_dump("Local Zotero", local_client)
    if local_result:
        return local_result

    if enable_webdav:
        try:
            webdav_path = download_attachment_from_webdav(
                attachment_key,
                destination,
                expected_filename=target_name,
            )
            if webdav_path.exists() and webdav_path.stat().st_size > 0:
                return AttachmentDownloadResult(
                    path=webdav_path,
                    source="WebDAV",
                    errors=errors,
                )
            errors.append("WebDAV: downloaded file was empty")
        except WebDAVNotConfiguredError:
            pass
        except Exception as exc:
            errors.append(f"WebDAV: {exc}")

    web_result = _try_dump("Web API", web_client)
    if web_result:
        return web_result

    return AttachmentDownloadResult(path=None, source=None, errors=errors)


def convert_to_markdown(file_path: str | Path) -> str:
    """
    Convert a file to markdown using markitdown library.

    Args:
        file_path: Path to the file to convert.

    Returns:
        Markdown text.
    """
    try:
        md = MarkItDown()
        result = md.convert(str(file_path))
        return result.text_content
    except Exception as e:
        return f"Error converting file to markdown: {str(e)}"
