"""Search-related tool functions for the Zotero MCP server."""

import hashlib
import json
import logging as _logging
import re
import threading as _threading
import time as _time
from pathlib import Path
from typing import Literal

from zotero_mcp import client as _client
from zotero_mcp import utils as _utils
from zotero_mcp._app import mcp
from zotero_mcp._context import Context, context_error, context_info, context_warning
from zotero_mcp.tools import _helpers

_search_logger = _logging.getLogger("zotero_mcp.search")

CASCADE_TIMEOUT = 60  # seconds — total budget for the entire fallback cascade
_ADVANCED_SCAN_BUDGET_SECONDS = 20


def _server_side_filters(
    conditions: list[dict[str, str]], join_mode: str
) -> dict[str, str]:
    """Return exact AND filters that Zotero can safely apply server-side."""
    if join_mode != "all":
        return {}
    filters: dict[str, str] = {}
    for condition in conditions:
        if condition["operation"] != "is":
            continue
        field = condition["field"].lower()
        value = condition["value"]
        if field == "itemtype" and value:
            filters.setdefault("itemType", value)
        elif field in {"tag", "tags"} and value:
            filters.setdefault("tag", value)
    return filters

# Pre-search background sync debounce: at most one fire-and-forget sync per
# this many seconds, shared across all semantic_search tool invocations.
_PRESEARCH_SYNC_MIN_INTERVAL = 60.0
_last_presearch_sync_ts: float = 0.0
_presearch_sync_lock = _threading.Lock()

_SEMANTIC_FILTER_FIELDS = {
    "item_key",
    "item_type",
    "title",
    "date",
    "creators",
    "publication",
    "doi",
    "tags",
    "citation_key",
    "has_fulltext",
    "fulltext_source",
    "library_identity",
    "library_id",
    "library_type",
}


def _maybe_fire_presearch_sync(search) -> None:
    """Schedule a background semantic-search DB update if auto-update is due.

    Runs in a daemon thread so the current tool call returns immediately.
    Intentionally swallows exceptions — a failed background sync must never
    surface as a search-tool error to the user.
    """
    global _last_presearch_sync_ts
    try:
        if not search.should_update_database():
            return
    except Exception:
        return
    now = _time.monotonic()
    with _presearch_sync_lock:
        if now - _last_presearch_sync_ts < _PRESEARCH_SYNC_MIN_INTERVAL:
            return
        _last_presearch_sync_ts = now

    def _run():
        try:
            search.update_database()
        except Exception as e:
            _search_logger.debug(f"Background pre-search sync failed: {e}")

    _threading.Thread(target=_run, daemon=True, name="zmcp-presearch-sync").start()


def _search_with_variants(zot, query: str, qmode: str, limit: int,
                          item_type: str = "-attachment",
                          tag: list[str] | None = None,
                          cascade_start: float | None = None,
                          cascade_timeout: float | None = None) -> list:
    """Search using multiple query variants, deduplicate by key.

    Generates ASCII, dash-to-space, and umlaut-expanded variants of the query
    and searches for each one.  Results are deduplicated by item key.

    All params (including item_type and tag) are explicitly set on every
    add_parameters call to avoid stale accumulated params in pyzotero.

    If cascade_start and cascade_timeout are provided, checks the budget
    before each API call and bails out if exceeded.
    """
    variants = _utils._generate_search_variants(query)
    _search_logger.debug(f"[SEARCH] query='{query}' variants={variants}")

    all_items: list[dict] = []
    seen_keys: set[str] = set()
    attempted = 0
    completed = 0
    last_error: Exception | None = None
    for variant in variants:
        # Check cascade timeout before each API call
        if cascade_start is not None and cascade_timeout is not None:
            if _time.monotonic() - cascade_start > cascade_timeout:
                _search_logger.debug("[SEARCH] Cascade timeout reached, skipping remaining variants")
                break

        params: dict = {
            "q": variant, "qmode": qmode, "limit": limit, "itemType": item_type,
        }
        if tag:
            params["tag"] = tag
        zot.add_parameters(**params)
        attempted += 1
        try:
            t0 = _time.monotonic()
            batch = zot.items()
            completed += 1
            elapsed = _time.monotonic() - t0
            _search_logger.debug(f"[SEARCH] variant='{variant}' qmode={qmode}: {len(batch)} results in {elapsed:.2f}s")
            for item in batch:
                key = item.get("key", "")
                if key and key not in seen_keys:
                    seen_keys.add(key)
                    all_items.append(item)
        except Exception as e:
            last_error = e
            _search_logger.debug(f"[SEARCH] variant='{variant}' failed: {e}")
            continue  # Skip failed variant, try next

    if attempted and completed == 0 and last_error is not None:
        raise RuntimeError(
            f"All {attempted} Zotero search request(s) failed; last error: {last_error}"
        ) from last_error
    return all_items


@mcp.tool(
    name="zotero_search_items",
    description=(
        "Search Zotero items by substring match against metadata (title, "
        "creators, year, and — in 'everything' mode — abstract). Returns "
        "metadata + abstracts as markdown. "
        "IMPORTANT: keep queries SHORT and SIMPLE — 'Author Year' "
        "(e.g. 'Brewer 2011') or just an author name ('Cladder-Micus'). "
        "This is substring matching, not web search: each extra word "
        "NARROWS the match, so adding topic words usually returns fewer "
        "results, not more. For topic discovery, use zotero_semantic_search "
        "instead; for tag filtering use zotero_search_by_tag. "
        "A miss returns no results by default. Set fallback_mode='relaxed' "
        "to try simpler metadata/full-text queries, or 'semantic' to also "
        "try the semantic index. Fallback results preserve item_type and tag "
        "constraints and are explicitly labeled. "
        "query: required substring. qmode: 'titleCreatorYear' (default) "
        "matches only title/authors/year; 'everything' uses Zotero's indexed "
        "all-fields/full-text search while still returning metadata. "
        "item_type: '-attachment' (default) excludes attachments; "
        "pass 'journalArticle', 'book', etc. to filter. tag: optional list "
        "of tag conditions (ANDed). limit: max results (default 10). "
        "collection_key: 8-char key to restrict to a collection (bypasses "
        "the fallback cascade). "
        "Example: zotero_search_items(query='Cladder-Micus') or "
        "zotero_search_items(query='Brewer 2011', limit=5)."
    )
)
def search_items(
    query: str,
    qmode: Literal["titleCreatorYear", "everything"] = "titleCreatorYear",
    item_type: str = "-attachment",  # Exclude attachments by default
    limit: int | str | None = 10,
    tag: list[str] | list[dict] | str | None = None,
    collection_key: str | None = None,
    include_subcollections: bool = False,
    search_all_libraries: bool = False,
    fallback_mode: Literal["none", "relaxed", "semantic"] = "none",
    *,
    ctx: Context
) -> str:
    """
    Search for items in your Zotero library.

    Args:
        query: Search query string
        qmode: Query mode (titleCreatorYear or everything)
        item_type: Type of items to search for. Use "-attachment" to exclude attachments.
        limit: Maximum number of results to return
        tag: Tag filter. Accepts ["tagA", "tagB"] (preferred), a bare string
            "tagA", a JSON-string list '["tagA", "tagB"]', or the dict-shape
            [{"tag": "tagA"}] sometimes emitted by clients that confuse the
            filter form with Zotero's stored-tag form. All are normalized
            internally to the list[str] form pyzotero expects.
        collection_key: Optional collection key to scope the search to a specific collection.
            When provided, bypasses the fallback cascade and searches the collection directly.
        ctx: MCP context

    Returns:
        Markdown-formatted search results
    """
    try:
        if not query.strip():
            return "Error: Search query cannot be empty"

        # Normalize tag across every wire shape clients produce (#237).
        tag = _helpers._normalize_tag_filter(tag)

        tag_condition_str = ""
        if tag:
            tag_condition_str = f" with tags: '{', '.join(tag)}'"

        context_info(ctx, f"Searching Zotero for '{query}'{tag_condition_str}")
        zot = _client.get_zotero_client()

        limit = _helpers._normalize_limit(limit, default=10)

        if search_all_libraries and collection_key:
            return (
                "Error: collection_key cannot be combined with "
                "search_all_libraries because collections are library-specific."
            )
        if search_all_libraries and (error := _helpers.global_search_error()):
            return error

        items = None
        if _utils.use_sqlite_search(
            "keyword", search_all_libraries=search_all_libraries
        ) and not collection_key:
            try:
                from zotero_mcp.local_db import get_local_zotero_reader

                reader = get_local_zotero_reader()
                if reader is not None:
                    try:
                        items = reader.search_items_sql(
                            query,
                            qmode=qmode,
                            item_type=item_type,
                            tag=tag,
                            limit=limit,
                            group_id=(
                                None
                                if search_all_libraries
                                else _client.get_active_group_id()
                            ),
                        )
                    finally:
                        reader.close()
            except Exception as exc:
                _search_logger.debug("SQLite search failed: %s", exc)
                items = None

        if items is not None:
            fallback_strategy = None
        elif search_all_libraries:
            return (
                "Error: global search could not be served by the SQLite "
                "backend; it will not silently fall back to one library."
            )
        elif collection_key:
            # Collection-scoped search — query the collection directly, no cascade needed
            try:
                _col = zot.collection(collection_key)
            except Exception:
                _col = None
            if not _col or _col.get("key") != collection_key:
                return f"Collection not found: '{collection_key}'. Use zotero_get_collections or zotero_search_collections to find valid collection keys."
            items = _helpers.fetch_collection_scope(
                zot, collection_key,
                include_subcollections=include_subcollections,
                q=query, qmode=qmode, itemType=item_type,
                max_items=limit, **({"tag": tag} if tag else {}),
            )
            fallback_strategy = None
        else:
            # --- Initial search with variant generation ---
            _cascade_start = _time.monotonic()
            items = _search_with_variants(zot, query, qmode, limit,
                                          item_type=item_type, tag=tag,
                                          cascade_start=_cascade_start,
                                          cascade_timeout=CASCADE_TIMEOUT)
            _search_logger.debug(f"[CASCADE] initial: {len(items)} results in {_time.monotonic() - _cascade_start:.2f}s")

            # --- Fallback cascade (only if initial search returned nothing) ---
            fallback_strategy = None
            _timed_out = False

            def _check_cascade_timeout():
                nonlocal _timed_out
                if _time.monotonic() - _cascade_start > CASCADE_TIMEOUT:
                    _timed_out = True
                    _search_logger.debug("[CASCADE] Timeout — stopping cascade")
                    context_info(ctx, "Search took too long — returning best results found so far")
                return _timed_out

            if not items and query.strip() and fallback_mode != "none":
                context_info(ctx, "No results with original query, trying fallback strategies...")
                words = query.strip().split()

                # Strategy 1: Simplify to author + year (P2 fix)
                if not _check_cascade_timeout() and not items and len(words) > 2:
                    # Extract year-like token (4 digits between 1800-2099)
                    year_token = next((w for w in words if re.match(r'^(1[89]\d{2}|20\d{2})$', w)), None)
                    # Extract author (first non-numeric word)
                    author_token = next((w for w in words if not re.match(r'^\d+$', w)), None)

                    if author_token and year_token:
                        simple_query = f"{author_token} {year_token}"
                    elif author_token:
                        simple_query = author_token
                    else:
                        simple_query = words[0]

                    t0 = _time.monotonic()
                    context_info(ctx, f"Retry with simplified query: '{simple_query}'")
                    items = _search_with_variants(zot, simple_query, qmode, limit,
                                                  item_type=item_type, tag=tag,
                                                  cascade_start=_cascade_start,
                                                  cascade_timeout=CASCADE_TIMEOUT)
                    _search_logger.debug(f"[CASCADE] strategy 1 (author+year): {len(items)} results in {_time.monotonic() - t0:.2f}s")
                    if items:
                        fallback_strategy = f"simplified to '{simple_query}'"

                # Strategy 2: Author surname only (first non-numeric word)
                if not _check_cascade_timeout() and not items and len(words) >= 2:
                    author_only = next((w for w in words if not re.match(r'^\d+$', w)), words[0])
                    t0 = _time.monotonic()
                    context_info(ctx, f"Retry with author only: '{author_only}'")
                    items = _search_with_variants(zot, author_only, qmode, limit,
                                                  item_type=item_type, tag=tag,
                                                  cascade_start=_cascade_start,
                                                  cascade_timeout=CASCADE_TIMEOUT)
                    _search_logger.debug(f"[CASCADE] strategy 2 (author only): {len(items)} results in {_time.monotonic() - t0:.2f}s")
                    if items:
                        fallback_strategy = f"author only '{author_only}'"

                # Strategy 3: qmode="everything" (searches full text on Zotero's side)
                # Safe — no tokens consumed, only metadata returned
                if not _check_cascade_timeout() and not items and qmode != "everything":
                    t0 = _time.monotonic()
                    context_info(ctx, f"Retry with qmode='everything': '{query}'")
                    items = _search_with_variants(zot, query, "everything", limit,
                                                  item_type=item_type, tag=tag,
                                                  cascade_start=_cascade_start,
                                                  cascade_timeout=CASCADE_TIMEOUT)
                    _search_logger.debug(f"[CASCADE] strategy 3 (everything): {len(items)} results in {_time.monotonic() - t0:.2f}s")
                    if items:
                        fallback_strategy = "full-text search"

                # Strategy 4: Semantic search (if database exists)
                if (
                    fallback_mode == "semantic"
                    and not _check_cascade_timeout()
                    and not items
                ):
                    try:
                        from zotero_mcp.semantic_search import create_semantic_search
                        config_path = Path.home() / ".config" / "zotero-mcp" / "config.json"
                        if config_path.exists():
                            context_info(ctx, f"Retry with semantic search: '{query}'")
                            t0 = _time.monotonic()
                            sem_search = create_semantic_search(str(config_path))
                            _search_logger.debug(f"[CASCADE] semantic init: {_time.monotonic() - t0:.2f}s")
                            t0 = _time.monotonic()
                            semantic_filters = {}
                            if item_type and not item_type.startswith("-"):
                                semantic_filters["item_type"] = item_type
                            sem_results = sem_search.search(
                                query=query,
                                limit=max((limit or 10) * 4, limit or 10),
                                filters=semantic_filters or None,
                            )
                            _search_logger.debug(f"[CASCADE] semantic query: {_time.monotonic() - t0:.2f}s")
                            if sem_results and sem_results.get("results"):
                                seen_keys: set[str] = set()
                                for sr in sem_results["results"]:
                                    zot_item = sr.get("zotero_item", {})
                                    key = sr.get("item_key", zot_item.get("key", ""))
                                    item_data = zot_item.get("data", {})
                                    result_item_type = item_data.get("itemType", "")
                                    if item_type:
                                        if item_type.startswith("-"):
                                            if result_item_type == item_type[1:]:
                                                continue
                                        elif result_item_type != item_type:
                                            continue
                                    item_tags = {
                                        value.get("tag", "")
                                        for value in item_data.get("tags", [])
                                        if isinstance(value, dict)
                                    }
                                    if tag and not all(value in item_tags for value in tag):
                                        continue
                                    if (
                                        key
                                        and key not in seen_keys
                                        and len(items) < limit
                                    ):
                                        seen_keys.add(key)
                                        if "key" not in zot_item:
                                            zot_item["key"] = key
                                        items.append(zot_item)
                                if items:
                                    fallback_strategy = "semantic search"
                    except Exception as e:
                        _search_logger.debug(f"[CASCADE] semantic failed: {e}")
                        context_info(ctx, f"Semantic search fallback failed: {e}")

            _search_logger.debug(f"[CASCADE] total: {_time.monotonic() - _cascade_start:.2f}s, fallback={fallback_strategy}")

        # --- No results after all strategies ---
        if not items:
            return f"No items found matching query: '{query}'{tag_condition_str}"

        # --- Format results as markdown ---
        output = [f"# Search Results for '{query}'", f"{tag_condition_str}", ""]

        for i, item in enumerate(items, 1):
            output.extend(
                _utils.format_item_result(
                    item, index=i, show_library=search_all_libraries
                )
            )

        # Prepend fallback verification note (AFTER output is built)
        if fallback_strategy:
            if fallback_strategy == "semantic search":
                note_text = (
                    f"*Note: Original search for '{query}' returned no results. "
                    f"The following {len(items)} item(s) are semantically related papers found "
                    f"via AI-powered search — they may be ABOUT the same topic but may NOT be "
                    f"the exact paper you're looking for. The target paper may not be in your "
                    f"library. Verify carefully by checking title, authors, and journal.*"
                )
            else:
                note_text = (
                    f"*Note: Original search for '{query}' returned no results. "
                    f"Found {len(items)} item(s) via {fallback_strategy} — verify the correct one "
                    f"by checking title, authors, journal, and year match your original query.*"
                )
            output.insert(1, "")
            output.insert(2, note_text)
            output.insert(3, "")

        return _helpers._prepend_size_warning("\n".join(output))

    except Exception as e:
        context_error(ctx, f"Error searching Zotero: {str(e)}")
        return f"Error searching Zotero: {str(e)}"

@mcp.tool(
    name="zotero_search_by_tag",
    description=(
        "Find items carrying one or more tags, with boolean syntax "
        "support. tag: list of tag strings; each entry is a condition ANDed "
        "with the others, and within an entry you can use ' OR ' for "
        "disjunction and a leading '-' for exclusion. "
        "Example: tag=['methods OR methodology', '-draft'] matches items "
        "tagged 'methods' OR 'methodology' AND NOT tagged 'draft'. "
        "item_type: '-attachment' (default) excludes attachments; pass "
        "'journalArticle', 'book', etc. to filter. "
        "limit: max results (default 10). "
        "collection_key: optional 8-char key to scope to a collection. "
        "Use zotero_get_tags to discover available tag names first. For "
        "free-text content search, use zotero_search_items or "
        "zotero_semantic_search instead. "
        "Example: zotero_search_by_tag(tag=['to-read'], limit=20)."
    )
)
def search_by_tag(
    tag: list[str] | list[dict] | str,
    item_type: str = "-attachment",
    limit: int | str | None = 10,
    collection_key: str | None = None,
    include_subcollections: bool = False,
    search_all_libraries: bool = False,
    *,
    ctx: Context
) -> str:
    """
    Search for items in your Zotero library by tag.
    Conditions are ANDed, each term supports disjunction (`OR`) and exclusion (`-`).

    Args:
        tag: List of tag conditions. Items are returned only if they satisfy
            ALL conditions in the list. Each tag condition can be expressed
            in two ways:
                As alternatives: tag1 OR tag2 (matches items with either tag1 OR tag2)
                As exclusions: -tag (matches items that do NOT have this tag)
            For example, a tag field with ["research OR important", "-draft"] would
            return items that:
                Have either "research" OR "important" tags, AND
                Do NOT have the "draft" tag
        item_type: Type of items to search for. Use "-attachment" to exclude attachments.
        limit: Maximum number of results to return
        collection_key: Optional collection key to scope the search to a specific collection
        ctx: MCP context

    Returns:
        Markdown-formatted search results
    """
    try:
        # Normalize tag across every wire shape clients produce (#237).
        tag = _helpers._normalize_tag_filter(tag)
        if not tag:
            return "Error: Tag cannot be empty"

        context_info(ctx, f"Searching Zotero for tag '{tag}'")
        zot = _client.get_zotero_client()

        limit = _helpers._normalize_limit(limit, default=10)

        if search_all_libraries and collection_key:
            return (
                "Error: collection_key cannot be combined with "
                "search_all_libraries."
            )
        if search_all_libraries and (error := _helpers.global_search_error()):
            return error

        # Search library-wide or scoped to a collection
        if collection_key:
            try:
                _col = zot.collection(collection_key)
            except Exception:
                _col = None
            if not _col or _col.get("key") != collection_key:
                return f"Collection not found: '{collection_key}'. Use zotero_get_collections or zotero_search_collections to find valid collection keys."
            results = _helpers.fetch_collection_scope(
                zot, collection_key,
                include_subcollections=include_subcollections,
                tag=tag, itemType=item_type, max_items=limit,
            )
        elif _utils.use_sqlite_search(
            "tag", search_all_libraries=search_all_libraries
        ):
            from zotero_mcp.local_db import get_local_zotero_reader

            reader = get_local_zotero_reader()
            if reader is None:
                if search_all_libraries:
                    return "Error: local Zotero database is unavailable."
                results = None
            else:
                try:
                    results = reader.search_items_sql(
                        "",
                        item_type=item_type,
                        tag=tag,
                        limit=limit,
                        group_id=(
                            None
                            if search_all_libraries
                            else _client.get_active_group_id()
                        ),
                    )
                finally:
                    reader.close()
            if results is None:
                if search_all_libraries:
                    return "Error: global tag search could not be served safely."
                zot.add_parameters(q="", tag=tag, itemType=item_type, limit=limit)
                results = zot.items()
        else:
            if search_all_libraries:
                return _helpers.global_search_error() or "Error: global search unavailable"
            zot.add_parameters(q="", tag=tag, itemType=item_type, limit=limit)
            results = zot.items()

        if not results:
            return f"No items found with tag: '{tag}'"

        # Format results as markdown
        scope = f" in Collection {collection_key}" if collection_key else ""
        output = [f"# Search Results for Tag: '{tag}'{scope}", ""]

        for i, item in enumerate(results, 1):
            output.extend(
                _utils.format_item_result(
                    item, index=i, show_library=search_all_libraries
                )
            )

        return "\n".join(output)

    except Exception as e:
        context_error(ctx, f"Error searching Zotero: {str(e)}")
        return f"Error searching Zotero: {str(e)}"


@mcp.tool(
    name="zotero_search_by_citation_key",
    description=(
        "Look up a single Zotero item by its BetterBibTeX citation key "
        "(e.g. 'Smith2024' or 'cladderMicus2018'). Returns that one item's "
        "metadata, or a not-found message if no item has that key. "
        "citekey: the citation key exactly as assigned by BetterBibTeX "
        "(case-sensitive). "
        "In local mode: queries the running Better BibTeX plugin via its "
        "HTTP API (Zotero desktop must be running and have BBT installed). "
        "In web mode: scans the 'Extra' field of items for 'Citation Key:' "
        "lines — slower, and may miss items whose keys aren't persisted to "
        "Extra. "
        "Requires the Better BibTeX plugin in the user's Zotero install. "
        "For partial-key or free-text lookup, use zotero_search_items. "
        "Example: zotero_search_by_citation_key(citekey='hasan2026mcp') → "
        "metadata for that single item."
    )
)
def search_by_citation_key(
    citekey: str,
    *,
    ctx: Context
) -> str:
    """
    Look up a Zotero item by its BetterBibTeX citation key.

    Args:
        citekey: The BetterBibTeX citation key to search for (e.g., 'Smith2024')
        ctx: MCP context

    Returns:
        Formatted item details or error message
    """
    try:
        if not citekey.strip():
            return "Error: Citation key cannot be empty"

        citekey = citekey.strip()
        context_info(ctx, f"Looking up citation key: {citekey}")

        # Strategy A: pyzotero search across all fields, then verify via Extra.
        # Note: the previous BetterBibTeX ``item.search`` JSON-RPC call was
        # removed in #293 — that BBT method does not exist in current versions
        # (always returned -32601 Method not found) and the exception handler
        # silently fell through to the same Extra-field search, so the BBT
        # branch only added noise.
        zot = _client.get_zotero_client()
        zot.add_parameters(q=citekey, qmode="everything", itemType="-attachment", limit=25)
        results = zot.items()

        for item in results:
            data = item.get("data", {})
            extra = data.get("extra", "")
            if data.get("citationKey") == citekey or _helpers._extra_has_citekey(extra, citekey):
                return _helpers._format_citekey_result(item, citekey)

        return f"No item found with citation key: '{citekey}'"

    except Exception as e:
        context_error(ctx, f"Error looking up citation key: {str(e)}")
        return f"Error looking up citation key: {str(e)}"


@mcp.tool(
    name="zotero_advanced_search",
    description=(
        "Advanced item search with multiple structured-field conditions "
        "joined by AND or OR. Use this when you need to filter by fields "
        "that zotero_search_items and zotero_search_by_tag can't express "
        "(date ranges, specific itemTypes, etc.). "
        "For plain text use zotero_search_items; for tags use "
        "zotero_search_by_tag; for topic discovery use "
        "zotero_semantic_search. "
        "conditions: list of {field, operation, value} dicts (also accepts "
        "a JSON string). "
        "  Common fields: title, creator, date, dateAdded, dateModified, "
        "tag, itemType, publicationTitle, abstractNote, collection. "
        "  Supported operations (exhaustive): is, isNot, contains, "
        "doesNotContain, beginsWith, endsWith, isGreaterThan, isLessThan, "
        "isBefore, isAfter. "
        "For 'added in the last N days', use field='dateAdded' with "
        "operation='isAfter' and an ISO date value (e.g. '2026-03-22'). "
        "join_mode: 'all' (AND, default) or 'any' (OR). "
        "sort_by: dateAdded, dateModified, title, creator, etc. "
        "sort_direction: 'asc' (default) or 'desc'. "
        "Results include DOI, language, dateAdded, and dateModified in addition "
        "to the standard item summary; unavailable values remain empty. "
        "limit: max results (default 50, max 500). Large result sets can "
        "produce correspondingly large MCP responses. "
        "Example: zotero_advanced_search(conditions=[{'field': 'itemType', "
        "'operation': 'is', 'value': 'preprint'}, {'field': 'dateAdded', "
        "'operation': 'isAfter', 'value': '2026-03-22'}], "
        "join_mode='all')."
    )
)
def advanced_search(
    conditions: list[dict[str, str]] | str,
    join_mode: Literal["all", "any"] = "all",
    sort_by: str | None = None,
    sort_direction: Literal["asc", "desc"] = "asc",
    limit: int | str = 50,
    include_subcollections: bool = False,
    search_all_libraries: bool = False,
    *,
    ctx: Context
) -> str:
    """
    Perform an advanced search with multiple criteria.

    Args:
        conditions: List of search condition dictionaries, each containing:
                   - field: The field to search (title, creator, date, tag, etc.)
                   - operation: The operation to perform (is, isNot, contains, etc.)
                   - value: The value to search for
        join_mode: Whether all conditions must match ("all") or any condition can match ("any")
        sort_by: Field to sort by (dateAdded, dateModified, title, creator, etc.)
        sort_direction: Direction to sort (asc or desc)
        limit: Maximum number of results to return
        ctx: MCP context

    Returns:
        Markdown-formatted search results
    """
    try:
        if isinstance(conditions, str):
            try:
                conditions = json.loads(conditions)
            except json.JSONDecodeError as parse_error:
                return (
                    "Error: conditions must be valid JSON when provided as a string "
                    f"({parse_error})"
                )

        if not isinstance(conditions, list) or not conditions:
            return "Error: No search conditions provided"

        if join_mode not in {"all", "any"}:
            return "Error: join_mode must be either 'all' or 'any'"

        limit = _helpers._normalize_limit(limit, default=50, max_val=500)

        context_info(ctx, f"Performing advanced search with {len(conditions)} conditions")
        zot = _client.get_zotero_client()

        valid_operations = {
            "is",
            "isNot",
            "contains",
            "doesNotContain",
            "beginsWith",
            "endsWith",
            "isGreaterThan",
            "isLessThan",
            "isBefore",
            "isAfter",
        }

        parsed_conditions: list[dict[str, str]] = []
        for i, condition in enumerate(conditions, 1):
            if not isinstance(condition, dict):
                return f"Error: Condition {i} must be an object"
            if "field" not in condition or "operation" not in condition or "value" not in condition:
                return (
                    f"Error: Condition {i} is missing required fields "
                    "(field, operation, value)"
                )

            field = str(condition["field"]).strip()
            operation = str(condition["operation"]).strip()
            value = str(condition["value"]).strip()

            if operation not in valid_operations:
                return (
                    f"Error: Unsupported operation '{operation}' in condition {i}. "
                    f"Supported: {', '.join(sorted(valid_operations))}"
                )
            if not field:
                return f"Error: Condition {i} has an empty field"

            parsed_conditions.append(
                {"field": field, "operation": operation, "value": value}
            )

        if search_all_libraries:
            if error := _helpers.global_search_error():
                return error
            if include_subcollections or any(
                condition["field"].lower() in {"collection", "collections"}
                for condition in parsed_conditions
            ):
                return (
                    "Error: collection conditions and include_subcollections "
                    "cannot be combined with search_all_libraries."
                )

        collection_scopes: dict[str, set[str]] = {}
        if include_subcollections:
            values = {
                condition["value"]
                for condition in parsed_conditions
                if condition["field"].lower() in {"collection", "collections"}
            }
            if values:
                collections = _helpers._paginate(zot.collections)
                collection_scopes = {
                    value: set(
                        _helpers.collection_descendants(collections, value)
                    )
                    for value in values
                }

        def _extract_values(data: dict[str, object], field: str) -> list[str]:
            field_lower = field.lower()

            if field_lower in {"author", "authors", "creator", "creators"}:
                creators = data.get("creators", []) or []
                values: list[str] = []
                for creator in creators:
                    if not isinstance(creator, dict):
                        continue
                    if creator.get("firstName") or creator.get("lastName"):
                        full_name = " ".join(
                            [
                                str(creator.get("firstName", "")).strip(),
                                str(creator.get("lastName", "")).strip(),
                            ]
                        ).strip()
                        if full_name:
                            values.append(full_name)
                    if creator.get("name"):
                        values.append(str(creator.get("name", "")).strip())
                return values

            if field_lower in {"tag", "tags"}:
                tags = data.get("tags", []) or []
                values = []
                for tag in tags:
                    if isinstance(tag, dict) and tag.get("tag"):
                        values.append(str(tag.get("tag", "")).strip())
                return values

            if field_lower == "year":
                date_value = _utils.item_display_date(data)
                match = re.search(r"\b(\d{4})\b", date_value)
                return [match.group(1)] if match else []

            if field_lower in {"collection", "collections"}:
                collections = data.get("collections", []) or []
                return [str(value).strip() for value in collections if value]

            field_aliases = {
                "itemtype": "itemType",
                "dateadded": "dateAdded",
                "datemodified": "dateModified",
                "doi": "DOI",
            }
            source_field = field_aliases.get(field_lower, field)
            if field_lower == "title":
                return [_utils.item_display_title(data)]
            if field_lower == "date":
                return [_utils.item_display_date(data)]
            try:
                from zotero_mcp import schema

                source_field = schema.resolve_field(
                    str(data.get("itemType", "")), source_field
                )
            except Exception:
                pass
            raw_value = data.get(source_field, "")
            if raw_value is None:
                return []
            return [str(raw_value).strip()]

        def _as_float(text: str) -> float | None:
            try:
                return float(text)
            except ValueError:
                return None

        def _compare(candidate: str, expected: str, operation: str) -> bool:
            # Normalize both sides for diacritics/dashes before comparison
            left = _utils._normalize_for_search(candidate).lower()
            right = _utils._normalize_for_search(expected).lower()

            if operation == "is":
                return left == right
            if operation == "isNot":
                return left != right
            if operation == "contains":
                return right in left
            if operation == "doesNotContain":
                return right not in left
            if operation == "beginsWith":
                return left.startswith(right)
            if operation == "endsWith":
                return left.endswith(right)

            left_num = _as_float(left)
            right_num = _as_float(right)
            if (
                operation in {"isGreaterThan", "isLessThan", "isBefore", "isAfter"}
                and left_num is not None
                and right_num is not None
            ):
                if operation in {"isGreaterThan", "isAfter"}:
                    return left_num > right_num
                return left_num < right_num

            if operation in {"isGreaterThan", "isAfter"}:
                return left > right
            return left < right

        def _matches_condition(data: dict[str, object], condition: dict[str, str]) -> bool:
            values = _extract_values(data, condition["field"])
            operation = condition["operation"]
            target = condition["value"]
            scope = (
                collection_scopes.get(target)
                if condition["field"].lower() in {"collection", "collections"}
                else None
            )
            if scope is not None and operation in {"is", "isNot"}:
                matched = bool(set(values) & scope)
                return matched if operation == "is" else not matched
            if not values:
                return operation in {"isNot", "doesNotContain"}
            comparisons = [_compare(value, target, operation) for value in values]

            if operation in {"isNot", "doesNotContain"}:
                return all(comparisons)
            return any(comparisons)

        # Prefer SQLite when enabled. Global queries never fall back to the
        # one-library API path.
        results = None
        scan_warning = None
        if _utils.use_sqlite_search(
            "advanced", search_all_libraries=search_all_libraries
        ) and not collection_scopes:
            try:
                from zotero_mcp.local_db import get_local_zotero_reader

                reader = get_local_zotero_reader()
                if reader is not None:
                    try:
                        results = reader.advanced_search_sql(
                            parsed_conditions,
                            join_mode=join_mode,
                            group_id=(
                                None
                                if search_all_libraries
                                else _client.get_active_group_id()
                            ),
                        )
                    finally:
                        reader.close()
            except Exception as exc:
                _search_logger.debug("SQLite advanced search failed: %s", exc)
                results = None

        if results is None and search_all_libraries:
            return (
                "Error: global advanced search could not be served by SQLite; "
                "it will not silently narrow to the active library."
            )

        if results is None:
            # Execute the bounded one-library API path.
            results = []
            server_filters = _server_side_filters(parsed_conditions, join_mode)
            deadline = _time.monotonic() + _ADVANCED_SCAN_BUDGET_SECONDS
            batch_size = 100
            start = 0
            while True:
                batch = zot.items(start=start, limit=batch_size, **server_filters)
                if not batch:
                    break
                for item in batch:
                    data = item.get("data", {})
                    if data.get("itemType") in {"attachment", "note", "annotation"}:
                        continue
                    checks = [_matches_condition(data, c) for c in parsed_conditions]
                    matched = all(checks) if join_mode == "all" else any(checks)
                    if matched:
                        results.append(item)
                if not sort_by and len(results) >= limit:
                    break
                if len(batch) < batch_size:
                    break
                start += batch_size
                if _time.monotonic() > deadline:
                    scan_warning = (
                        f"Search stopped after {_ADVANCED_SCAN_BUDGET_SECONDS}s "
                        f"having examined {start} items; results are partial."
                    )
                    break

        sort_warning = None
        if sort_by:
            sort_field = sort_by.strip()
            reverse = sort_direction == "desc"

            def _sort_key(item: dict[str, object]) -> str:
                data = item.get("data", {}) if isinstance(item, dict) else {}
                if sort_field in {"creator", "author"}:
                    return _utils.format_creators(data.get("creators", []))
                if sort_field in {"date", "year", "publicationDate"}:
                    meta = item.get("meta", {}) if isinstance(item, dict) else {}
                    parsed = str(meta.get("parsedDate", "") or "").strip()
                    if parsed:
                        return parsed
                    match = re.search(
                        r"\b(\d{4})\b", _utils.item_display_date(data)
                    )
                    return match.group(1) if match else ""
                return str(data.get(sort_field, "")).lower()

            if results and not any(_sort_key(item) for item in results):
                sort_warning = (
                    f"Requested sort by `{sort_field}` was not applied because "
                    "no result carries that field."
                )
            else:
                results.sort(key=_sort_key, reverse=reverse)

        if not results:
            if scan_warning:
                return (
                    "No items found in the portion of the library searched.\n\n"
                    f"> **Note:** {scan_warning}"
                )
            return "No items found matching the search criteria."

        results = results[:limit]

        output = ["# Advanced Search Results", ""]
        output.append(f"Found {len(results)} items matching the search criteria:")
        output.append("")
        output.append("## Search Criteria")
        if search_all_libraries:
            output.append("Scope: all accessible libraries")
        output.append(f"Join mode: {join_mode.upper()}")
        for i, condition in enumerate(parsed_conditions, 1):
            output.append(
                f"{i}. {condition['field']} {condition['operation']} \"{condition['value']}\""
            )
        if sort_warning:
            output.extend(["", f"> **Note:** {sort_warning}"])
        if scan_warning:
            output.extend(["", f"> **Note:** {scan_warning}"])
        output.append("")
        output.append("## Results")

        for i, item in enumerate(results, 1):
            data = item.get("data", {})
            output.extend(
                _utils.format_item_result(
                    item,
                    index=i,
                    show_library=search_all_libraries,
                    extra_fields={
                        "DOI": str(data.get("DOI") or ""),
                        "Language": str(data.get("language") or ""),
                        "Date Added": str(data.get("dateAdded") or ""),
                        "Date Modified": str(data.get("dateModified") or ""),
                    },
                )
            )

        return _helpers._prepend_size_warning(
            "\n".join(output),
            "Narrow the conditions or lower limit to reduce response size.",
        )

    except Exception as e:
        context_error(ctx, f"Error in advanced search: {str(e)}")
        return f"Error in advanced search: {str(e)}"


@mcp.tool(
    name="zotero_semantic_search",
    description=(
        "First-step concept and topic search across the active Zotero "
        "library. Use this to discover relevant papers, claims, methods, or "
        "passages by meaning; do not open papers one by one to search. "
        "Results contain paper metadata, relevance, and a grounded matched "
        "excerpt. With a passage-chunked index they also contain a Chunk ID "
        "and Chunk Hash. Normal follow-up: call "
        "zotero_get_semantic_context with a promising Chunk ID to inspect "
        "the exact indexed evidence. Call zotero_get_item_fulltext only when "
        "the user requests whole-paper reading or the task genuinely needs "
        "information spread across the document; never load every search "
        "result in full. query: natural-language topic, claim, or question. "
        "limit: maximum paper-level results, default 10. item_key: optional "
        "exact parent item key, useful for finding the best indexed passage "
        "inside one known paper. filters: optional exact-match Chroma metadata "
        "constraints as a dict or JSON string. Supported indexed fields include "
        "item_key, item_type, title, date, creators, publication, doi, tags, "
        "citation_key, has_fulltext, fulltext_source, library_identity, "
        "library_id, and library_type. Requires a populated "
        "semantic database; check zotero_get_search_database_status when "
        "readiness is uncertain. Example: "
        "zotero_semantic_search(query='negative muon capture in helium', "
        "limit=5)."
    )
)
def semantic_search(
    query: str,
    limit: int = 10,
    filters: dict[str, object] | str | None = None,
    item_key: str | None = None,
    search_all_libraries: bool = False,
    *,
    ctx: Context
) -> str:
    """
    Perform semantic search over your Zotero library.

    Args:
        query: Search query text - can be concepts, topics, or natural language descriptions
        limit: Maximum number of results to return (default: 10)
        filters: Optional metadata filters as dict or JSON string. Example: {"item_type": "note"}
        ctx: MCP context

    Returns:
        Markdown-formatted search results with similarity scores
    """
    try:
        if not query.strip():
            return "Error: Search query cannot be empty"

        # Parse and validate filters parameter
        if filters is not None:
            # Handle JSON string input
            if isinstance(filters, str):
                try:
                    filters = json.loads(filters)
                    context_info(ctx, f"Parsed JSON string filters: {filters}")
                except json.JSONDecodeError as e:
                    return f"Error: Invalid JSON in filters parameter: {str(e)}"

            # Validate it's a dictionary
            if not isinstance(filters, dict):
                return "Error: filters parameter must be a dictionary or JSON string. Example: {\"item_type\": \"note\"}"

            filters = dict(filters)

            # Automatically translate common field names
            if "itemType" in filters:
                filters["item_type"] = filters.pop("itemType")
                context_info(ctx, f"Automatically translated 'itemType' to 'item_type': {filters}")

        if item_key:
            normalized_key = item_key.strip()
            if not normalized_key:
                return "Error: item_key cannot be blank"
            filters = dict(filters or {})
            existing_key = filters.get("item_key")
            if existing_key is not None and existing_key != normalized_key:
                return "Error: item_key conflicts with filters['item_key']"
            filters["item_key"] = normalized_key

        if filters:
            unsupported = sorted(set(filters) - _SEMANTIC_FILTER_FIELDS)
            if unsupported:
                return (
                    "Error: unsupported semantic filter field(s): "
                    + ", ".join(unsupported)
                )
            if any(isinstance(value, (dict, list)) for value in filters.values()):
                return "Error: semantic filter values must be exact scalar values"
            if len(filters) > 1:
                filters = {
                    "$and": [
                        {field: {"$eq": value}}
                        for field, value in filters.items()
                    ]
                }

        context_info(ctx, f"Performing semantic search for: '{query}'")

        if search_all_libraries and (error := _helpers.global_search_error()):
            return error

        # Import semantic search module
        try:
            from zotero_mcp.semantic_search import create_semantic_search
        except ImportError:
            return (
                "Semantic search is not available. Install the required packages with:\n"
                "  pip install zotero-mcp-server[semantic]\n\n"
                "This installs chromadb, sentence-transformers, and related dependencies."
            )

        # Determine config path
        config_path = Path.home() / ".config" / "zotero-mcp" / "config.json"

        global_warnings: list[str] = []
        if search_all_libraries:
            from zotero_mcp.local_db import LocalZoteroReader

            original_override = _client.get_active_library()
            combined: list[dict] = []
            try:
                with LocalZoteroReader() as reader:
                    libraries = reader.get_libraries()
                for library in libraries:
                    if library.get("type") == "user":
                        library_id, library_type = "0", "user"
                        library_name = "My Library"
                    elif library.get("type") == "group":
                        library_id = str(library.get("groupID") or "")
                        library_type = "group"
                        library_name = library.get("groupName") or f"Group {library_id}"
                    else:
                        continue
                    if not library_id:
                        continue
                    _client.set_active_library(library_id, library_type)
                    try:
                        scoped = create_semantic_search(str(config_path))
                        outcome = scoped.search(
                            query=query, limit=limit, filters=filters
                        )
                    except Exception as exc:
                        global_warnings.append(
                            f"{library_type}:{library_id}: {exc}"
                        )
                        continue
                    if outcome.get("error"):
                        global_warnings.append(
                            f"{library_type}:{library_id}: {outcome['error']}"
                        )
                        continue
                    for result in outcome.get("results", []):
                        item = result.get("zotero_item")
                        if isinstance(item, dict):
                            item["library"] = {
                                "id": 0 if library_type == "user" else int(library_id),
                                "type": library_type,
                                "name": library_name,
                            }
                        combined.append(result)
            finally:
                if original_override:
                    _client.set_active_library(
                        original_override["library_id"],
                        original_override["library_type"],
                    )
                else:
                    _client.clear_active_library()
            combined.sort(
                key=lambda value: float(value.get("similarity_score", 0)),
                reverse=True,
            )
            results = {"results": combined[:limit]}
        else:
            search = create_semantic_search(str(config_path))
            # Fire-and-forget only for the active library. Global search must
            # not trigger background mutations across libraries.
            _maybe_fire_presearch_sync(search)
            results = search.search(query=query, limit=limit, filters=filters)

        if results.get("error"):
            return f"Semantic search error: {results['error']}"

        search_results = results.get("results", [])

        if not search_results:
            return f"No semantically similar items found for query: '{query}'"

        # Format results as markdown
        output = [f"# Semantic Search Results for '{query}'", ""]
        if global_warnings:
            output.extend(
                [
                    "> **Partial coverage:** " + "; ".join(global_warnings),
                    "",
                ]
            )
        output.append(f"Found {len(search_results)} similar items:")
        output.append("")

        for i, result in enumerate(search_results, 1):
            similarity_score = result.get("similarity_score", 0)
            zotero_item = result.get("zotero_item", {})

            # Prefer the grounded passage — the window of the document that
            # actually overlaps the query — over a blind head-truncation, so
            # the agent gets a citable quote rather than the abstract's opening.
            passage = result.get("matched_passage") or result.get("matched_text", "")
            snippet = passage[:400] + "..." if len(passage) > 400 else passage

            # Provenance for citing: page (when the index carries page breaks),
            # else which passage of how many, else an approximate char offset.
            loc_bits = []
            if (page := result.get("page")) is not None:
                loc_bits.append(f"p. {page}")
            if (ci := result.get("chunk_index")) is not None and (nc := result.get("n_chunks")):
                loc_bits.append(f"passage {ci + 1}/{nc}")
            elif off := result.get("char_start", result.get("passage_offset")):
                loc_bits.append(f"char ~{off}")

            if zotero_item:
                extra = {"Relevance": f"{similarity_score:.3f}"}
                if chunk_id := result.get("chunk_id"):
                    extra["Chunk ID"] = f"`{chunk_id}`"
                if content_hash := result.get("content_hash"):
                    extra["Chunk Hash"] = f"`{content_hash}`"
                if loc_bits:
                    extra["Location"] = ", ".join(loc_bits)
                if snippet:
                    extra["Matched Passage"] = snippet
                supporting = result.get("matched_passages", [])[1:]
                if supporting:
                    extra["Supporting Passages"] = "\n\n".join(
                        (
                            f"`{passage['chunk_id']}`: "
                            f"`{passage.get('content_hash', '')}`\n"
                            f"{passage.get('matched_passage', '')[:400]}"
                        )
                        for passage in supporting
                        if passage.get("chunk_id")
                        and passage.get("matched_passage")
                    )
                # Override key from result since it may differ from item["key"]
                zotero_item.setdefault("key", result.get("item_key", ""))
                output.extend(
                    _utils.format_item_result(
                        zotero_item,
                        index=i,
                        extra_fields=extra,
                        show_library=search_all_libraries,
                    )
                )
            else:
                # Fallback if full Zotero item not available
                output.append(f"## {i}. Item {result.get('item_key', 'Unknown')}")
                output.append(f"**Relevance:** {similarity_score:.3f}")
                if chunk_id := result.get("chunk_id"):
                    output.append(f"**Chunk ID:** `{chunk_id}`")
                if content_hash := result.get("content_hash"):
                    output.append(f"**Chunk Hash:** `{content_hash}`")
                if loc_bits:
                    output.append(f"**Location:** {', '.join(loc_bits)}")
                if snippet:
                    output.append(f"**Matched Passage:** {snippet}")
                if error := result.get("error"):
                    output.append(f"**Error:** {error}")
                output.append("")

        return "\n".join(output)

    except Exception as e:
        context_error(ctx, f"Error in semantic search: {str(e)}")
        return f"Error in semantic search: {str(e)}"


_SEMANTIC_CHUNK_ID_RE = re.compile(
    r"^(?P<item_key>[^#\s]+)#(?P<chunk_index>0|[1-9]\d*)$"
)
_MAX_ADJACENT_SEMANTIC_CHUNKS = 2


@mcp.tool(
    name="zotero_get_semantic_context",
    description=(
        "Preferred second step after zotero_semantic_search. Retrieve the "
        "complete stored chunk behind a matched excerpt, with optional "
        "neighboring chunks, so you can inspect or cite relevant evidence "
        "without loading the whole paper. Use it for targeted questions, "
        "claim verification, local context, methods, results, and quotations. "
        "chunk_id: copy the exact Chunk ID from a search result; never infer "
        "or invent one. expected_hash: copy the accompanying Chunk Hash when "
        "available; this makes the call fail safely if the index changed "
        "since the search. adjacent_chunks: context on each side, 0 by "
        "default, 1 when continuity is needed, maximum 2. Start with 0 and "
        "expand only if the passage is incomplete. This reads the indexed "
        "snapshot only and performs no Zotero extraction or embedding. For a "
        "comprehensive summary or analysis spanning the paper, use "
        "zotero_get_item_fulltext instead. Requires a Chunk ID from a "
        "passage-chunked index. Example: "
        "zotero_get_semantic_context(chunk_id='ABCD1234#17', "
        "adjacent_chunks=1)."
    ),
)
def get_semantic_context(
    chunk_id: str,
    adjacent_chunks: int = 0,
    expected_hash: str | None = None,
    *,
    ctx: Context,
) -> str:
    """Return an exact indexed passage and bounded neighboring passages."""
    match = (
        _SEMANTIC_CHUNK_ID_RE.fullmatch(chunk_id.strip())
        if isinstance(chunk_id, str)
        else None
    )
    if not match:
        return (
            "Error: chunk_id must be an exact Chunk ID returned by "
            "zotero_semantic_search (for example, ABCD1234#17)."
        )

    if isinstance(adjacent_chunks, bool) or not isinstance(adjacent_chunks, int):
        return "Error: adjacent_chunks must be an integer from 0 through 2."
    adjacent = adjacent_chunks
    if not 0 <= adjacent <= _MAX_ADJACENT_SEMANTIC_CHUNKS:
        return "Error: adjacent_chunks must be an integer from 0 through 2."

    normalized_hash: str | None = None
    if expected_hash is not None:
        if not isinstance(expected_hash, str):
            return "Error: expected_hash must be the 64-character Chunk Hash from semantic search."
        normalized_hash = expected_hash.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", normalized_hash):
            return "Error: expected_hash must be the 64-character Chunk Hash from semantic search."

    item_key = match.group("item_key")
    chunk_index = int(match.group("chunk_index"))
    first_index = max(0, chunk_index - adjacent)
    requested_ids = [
        f"{item_key}#{index}"
        for index in range(first_index, chunk_index + adjacent + 1)
    ]

    try:
        try:
            from zotero_mcp.semantic_search import create_semantic_search
        except ImportError:
            return (
                "Semantic search is not available. Install "
                "zotero-mcp-server[semantic] first."
            )

        context_info(ctx, f"Retrieving semantic context for {chunk_id}")
        config_path = Path.home() / ".config" / "zotero-mcp" / "config.json"
        search = create_semantic_search(str(config_path))
        records = search.chroma_client.get_records(requested_ids)
        target = records.get(chunk_id)
        if not target:
            return (
                f"No indexed semantic chunk found for `{chunk_id}`. The "
                "index may have changed; run zotero_semantic_search again."
            )

        target_document = target.get("document") or ""
        target_metadata = target.get("metadata") or {}
        if not target_document.strip():
            return f"Error: indexed semantic chunk `{chunk_id}` is empty."
        if not isinstance(target_metadata, dict):
            return (
                f"Error: indexed semantic chunk `{chunk_id}` has inconsistent "
                "provenance. Call zotero_get_search_database_health before using it."
            )
        if (
            target_metadata.get("parent_item_key", item_key) != item_key
            or target_metadata.get("chunk_index", chunk_index) != chunk_index
        ):
            return (
                f"Error: indexed semantic chunk `{chunk_id}` has inconsistent "
                "provenance. Call zotero_get_search_database_health before using it."
            )

        target_hash = hashlib.sha256(target_document.encode("utf-8")).hexdigest()
        if normalized_hash is not None and normalized_hash != target_hash:
            return (
                f"Semantic chunk `{chunk_id}` changed after the search result "
                "was produced. No text was returned; run "
                "zotero_semantic_search again."
            )

        n_chunks = target_metadata.get("n_chunks")
        if isinstance(n_chunks, int) and not 0 <= chunk_index < n_chunks:
            return (
                f"Error: indexed semantic chunk `{chunk_id}` has inconsistent "
                "provenance. Call zotero_get_search_database_health before using it."
            )
        title = target_metadata.get("title") or item_key
        output = [
            f"# Semantic Context: {title}",
            "",
            f"**Item Key:** `{item_key}`",
            f"**Requested Chunk:** `{chunk_id}`",
            f"**Chunk Hash:** `{target_hash}`",
        ]
        if isinstance(n_chunks, int) and n_chunks > 0:
            output.append(f"**Location:** passage {chunk_index + 1}/{n_chunks}")
        if source := target_metadata.get("fulltext_source"):
            output.append(f"**Indexed Source:** {source}")
        output.append("")

        returned = 0
        for record_id in requested_ids:
            record = records.get(record_id)
            if not record:
                continue
            document = record.get("document") or ""
            metadata = record.get("metadata") or {}
            record_match = _SEMANTIC_CHUNK_ID_RE.fullmatch(record_id)
            record_index = int(record_match.group("chunk_index")) if record_match else -1
            shared_fields = (
                "n_chunks",
                "index_layout_signature",
                "index_content_signature",
                "attachment_signature",
                "date_modified",
                "fulltext_source",
            )
            if (
                not document.strip()
                or not isinstance(metadata, dict)
                or metadata.get("parent_item_key", item_key) != item_key
                or metadata.get("chunk_index", record_index) != record_index
                or (
                    isinstance(n_chunks, int)
                    and not 0 <= record_index < n_chunks
                )
                or any(
                    field in target_metadata
                    and metadata.get(field) != target_metadata[field]
                    for field in shared_fields
                )
            ):
                continue
            label = "Requested Chunk" if record_id == chunk_id else "Adjacent Chunk"
            output.extend([f"## {label} `{record_id}`", "", document, ""])
            returned += 1

        if returned == 0:  # Defensive: the validated target should be present.
            return f"Error: no readable semantic context found for `{chunk_id}`."
        return "\n".join(output).rstrip()
    except Exception as exc:
        context_error(ctx, f"Error retrieving semantic context: {exc}")
        return f"Error retrieving semantic context: {exc}"


_MCP_FORCE_REBUILD_CONFIRMATION = "REBUILD ALL ITEMS"


@mcp.tool(
    name="zotero_update_search_database",
    description=(
        "Build or refresh the semantic search embedding database from "
        "Zotero items. Run this: (a) after first install, (b) after adding "
        "items via zotero_add_by_doi / add_by_url / add_from_file, or "
        "(c) when the user has added items directly in Zotero desktop "
        "since the last update. "
        "By default the update is INCREMENTAL — only new or changed items "
        "are re-embedded, so repeated calls are cheap. "
        "force_rebuild=True re-embeds ALL items from scratch and can take "
        "hours or incur substantial API costs. It is rejected unless "
        "confirm_force_rebuild contains the exact safety phrase supplied "
        "verbatim by the user. Never guess, suggest, or disclose that phrase. "
        "Realtime rebuilds replace each item only after its new embeddings are "
        "ready. The next ordinary update continues an interrupted rebuild and "
        "reuses completed items. force_clear=True clears before indexing and "
        "requires force_rebuild=True. "
        "fulltext=True selects one local BetterIssa/PDF attachment per item. "
        "retry_failed_fulltext=True retries only cached local extraction "
        "failures and requires fulltext=True. "
        "The default fulltext=False indexes title and abstract only. "
        "limit: optional cap on items processed (useful for smoke-testing). "
        "Progress is reported via the MCP context; on large libraries an "
        "incremental update is seconds, a full rebuild can take minutes. "
        "Requires the [semantic] optional dependency and a configured "
        "embedding provider (see config.json). Check status with "
        "zotero_get_search_database_status."
    )
)
def update_search_database(
    force_rebuild: bool = False,
    force_clear: bool = False,
    confirm_force_rebuild: str | None = None,
    fulltext: bool = False,
    retry_failed_fulltext: bool = False,
    limit: int | None = None,
    *,
    ctx: Context
) -> str:
    """
    Update the semantic search database.

    Args:
        force_rebuild: Whether to rebuild the entire database from scratch
        force_clear: Clear the live index before a confirmed forced rebuild
        confirm_force_rebuild: Exact confirmation phrase required for a rebuild
        fulltext: Whether to index one locally selected full-text attachment
        retry_failed_fulltext: Retry cached local extraction failures
        limit: Limit number of items to process (useful for testing)
        ctx: MCP context

    Returns:
        Update status and statistics
    """
    if (
        force_rebuild
        and confirm_force_rebuild != _MCP_FORCE_REBUILD_CONFIRMATION
    ):
        context_warning(ctx, "Blocked unconfirmed force rebuild.")
        return (
            "# Force Rebuild Not Started\n\n"
            "A full rebuild re-embeds the entire semantic search "
            "index. It can take hours and may incur substantial API costs.\n\n"
            "The exact confirmation phrase must be supplied verbatim by the "
            "user. It is intentionally not disclosed or suggested by this "
            "tool. Do not retry until the user provides it explicitly."
        )

    if force_clear and not force_rebuild:
        return (
            "# Database Update Not Started\n\n"
            "force_clear requires force_rebuild=True."
        )
    if force_rebuild and limit is not None:
        return (
            "# Database Update Not Started\n\n"
            "limit cannot be combined with force_rebuild because a partial "
            "scan cannot replace the complete index."
        )

    if fulltext and not _utils.is_local_mode():
        return (
            "# Database Update Not Started\n\n"
            "Full-text indexing requires local Zotero mode because attachments "
            "are selected from the local database and filesystem."
        )
    if retry_failed_fulltext and not fulltext:
        return (
            "# Database Update Not Started\n\n"
            "retry_failed_fulltext requires fulltext=True."
        )

    try:
        context_info(ctx, "Starting semantic search database update...")

        # Import semantic search module
        try:
            from zotero_mcp.semantic_search import create_semantic_search
        except ImportError:
            return (
                "Semantic search is not available. Install the required packages with:\n"
                "  pip install zotero-mcp-server[semantic]\n\n"
                "This installs chromadb, sentence-transformers, and related dependencies."
            )

        # Determine config path
        config_path = Path.home() / ".config" / "zotero-mcp" / "config.json"

        # Create semantic search instance
        search = create_semantic_search(
            str(config_path),
            allow_embedding_mismatch=force_rebuild,
        )

        stats = search.update_database(
            force_full_rebuild=force_rebuild,
            force_clear=force_clear,
            limit=limit,
            fulltext=fulltext,
            retry_failed_fulltext=retry_failed_fulltext,
        )

        # Format results
        output = ["# Database Update Results", ""]

        if stats.get("error"):
            output.append(f"**Error:** {stats['error']}")
            if stats.get("rebuild_in_progress"):
                output.append(
                    "**Rebuild In Progress:** yes; run an ordinary database update to continue."
                )
        else:
            output.append(
                f"**Full text:** {'local attachments' if stats.get('fulltext', fulltext) else 'disabled'}"
            )
            output.append(f"**Total items:** {stats.get('total_items', 0)}")
            output.append(f"**Processed:** {stats.get('processed_items', 0)}")
            output.append(f"**Added:** {stats.get('added_items', 0)}")
            output.append(f"**Updated:** {stats.get('updated_items', 0)}")
            output.append(
                f"**Unchanged vectors reused:** "
                f"{stats.get('reused_embeddings', 0)}"
            )
            output.append(f"**Skipped:** {stats.get('skipped_items', 0)}")
            error_count = stats.get("errors", 0)
            output.append(f"**Errors:** {error_count}")
            if error_count:
                output.append(
                    f"**Partial failure:** {error_count} item(s) could not be updated."
                )
            if stats.get("pending_rebuild_items"):
                output.append(
                    f"**Items still pending rebuild:** "
                    f"{stats['pending_rebuild_items']}"
                )
                output.append(
                    "Run an ordinary database update to continue the rebuild."
                )
            output.append(f"**Duration:** {stats.get('duration', 'Unknown')}")

            if stats.get('start_time'):
                output.append(f"**Started:** {stats['start_time']}")
            if stats.get('end_time'):
                output.append(f"**Completed:** {stats['end_time']}")

        return "\n".join(output)

    except Exception as e:
        context_error(ctx, f"Error updating search database: {str(e)}")
        return f"Error updating search database: {str(e)}"


@mcp.tool(
    name="zotero_get_search_database_status",
    description=(
        "Report the active library's semantic database readiness and stats: "
        "distinct item count, vector-record/passage count, index layout, last "
        "successful update time, indexed-fulltext/content-contract state, "
        "active update lock, embedding provider/model, and whether "
        "the [semantic] optional dependency is installed. "
        "Use this to decide whether zotero_semantic_search will return "
        "useful results, or whether the user should run "
        "zotero_update_search_database first. "
        "Takes no parameters; no side effects. "
        "Returns a human-readable status block. If the [semantic] extras "
        "are not installed, returns an install hint instead of stats. "
        "Example: zotero_get_search_database_status() → count, last sync, "
        "provider summary."
    )
)
def get_search_database_status(*, ctx: Context) -> str:
    """
    Get semantic search database status.

    ChromaDB query that never touches the Zotero API, and holding the shared
    lock here would make a slow status read block every other tool. The read
    path below also avoids constructing the embedding function, which for the
    default backend downloads an ONNX model on first use and could otherwise
    hang this call for minutes.

    Args:
        ctx: MCP context

    Returns:
        Database status information
    """
    try:
        context_info(ctx, "Getting semantic search database status...")

        # Import the lightweight, model-free status readers. These live in the
        # semantic-search modules so they share the [semantic] extra's import
        # guard, but neither loads an embedding model or a Zotero client.
        try:
            from zotero_mcp._file_lock import acquire_file_lock, release_file_lock
            from zotero_mcp.chroma_client import read_collection_status
            from zotero_mcp.semantic_search import (
                load_update_config,
                read_lock_holder,
                should_update,
                update_lock_path,
            )
        except ImportError:
            return (
                "Semantic search is not available. Install the required packages with:\n"
                "  pip install zotero-mcp-server[semantic]\n\n"
                "This installs chromadb, sentence-transformers, and related dependencies."
            )

        # Determine config path
        config_path = Path.home() / ".config" / "zotero-mcp" / "config.json"

        # Read status without loading any embedding model (fast, no network).
        library = _client.get_current_library()
        identity = _client.library_identity(library)
        collection_info = read_collection_status(
            str(config_path), scope_identity=identity
        )
        update_config = load_update_config(str(config_path))
        library_state = {}
        semantic_config = {}
        try:
            with open(config_path, encoding="utf-8") as config_file:
                semantic_config = json.load(config_file).get(
                    "semantic_search", {}
                )
                library_state = semantic_config.get("library_states", {}).get(
                    identity, {}
                )
            if library_state.get("last_update"):
                update_config["last_update"] = library_state["last_update"]
        except Exception:
            pass
        base_collection = semantic_config.get("collection_name", "zotero_library")
        uses_legacy_state = collection_info.get("name") == base_collection

        def scoped_state(name: str):
            if name in library_state:
                return library_state[name]
            return semantic_config.get(name) if uses_legacy_state else None

        lock_path = update_lock_path()
        lock_file = acquire_file_lock(
            lock_path,
            exclusive=False,
            blocking=False,
        )
        update_active = lock_file is None
        if lock_file is not None:
            release_file_lock(lock_file)
        holder_pid, holder_alive = (
            read_lock_holder(lock_path) if update_active else (None, False)
        )

        # Format results
        output = ["# Semantic Search Database Status", ""]

        output.append("## Collection Information")
        output.append(f"**Library:** {identity}")
        output.append(f"**Name:** {collection_info.get('name', 'Unknown')}")
        output.append(f"**Indexed Items:** {collection_info.get('item_count', 0)}")
        output.append(
            f"**Vector Records:** {collection_info.get('record_count', collection_info.get('count', 0))}"
        )
        output.append(f"**Passage Records:** {collection_info.get('chunk_count', 0)}")
        output.append(f"**Index Layout:** {collection_info.get('layout', 'unknown')}")
        output.append(
            f"**Collection Owner:** "
            f"{collection_info.get('library_identity') or identity}"
        )
        output.append(f"**Embedding Model:** {collection_info.get('embedding_model', 'Unknown')}")
        output.append(f"**Database Path:** {collection_info.get('persist_directory', 'Unknown')}")

        if collection_info.get("initialized") is False and not collection_info.get("error"):
            output.append("**Status:** Not initialized — run zotero_update_search_database first.")
        if collection_info.get('error'):
            output.append(f"**Error:** {collection_info['error']}")

        output.append("")

        output.append("## Update Configuration")
        output.append(f"**Auto Update:** {update_config.get('auto_update', False)}")
        output.append(f"**Frequency:** {update_config.get('update_frequency', 'manual')}")
        output.append(
            f"**Last Successful Refresh:** "
            f"{update_config.get('last_update', 'Never')}"
        )
        indexed_fulltext = scoped_state("indexed_fulltext")
        output.append(
            "**Indexed Full Text:** "
            + (
                "yes"
                if indexed_fulltext is True
                else "no"
                if indexed_fulltext is False
                else "unknown"
            )
        )
        output.append(
            f"**Content Contract:** "
            f"{scoped_state('indexed_content_signature') or 'unknown'}"
        )
        rebuild = library_state.get("rebuild_in_progress")
        if isinstance(rebuild, dict):
            rebuild_mode = "full text" if rebuild.get("fulltext") else "metadata only"
            output.append(
                f"**Rebuild In Progress:** yes ({rebuild_mode}; the next ordinary update resumes it)"
            )
        else:
            output.append("**Rebuild In Progress:** no")
        output.append(f"**Update Active:** {'yes' if update_active else 'no'}")
        if update_active and holder_pid is not None:
            output.append(
                f"**Update Holder:** PID {holder_pid} "
                f"({'alive' if holder_alive else 'unknown'})"
            )
        output.append(f"**Should Update Now:** {should_update(update_config)}")

        frequency = update_config.get('update_frequency', 'manual')
        if frequency.startswith('every_') and update_config.get('update_days'):
            output.append(f"**Update Interval:** Every {update_config['update_days']} days")

        return "\n".join(output)

    except Exception as e:
        context_error(ctx, f"Error getting database status: {str(e)}")
        return f"Error getting database status: {str(e)}"
