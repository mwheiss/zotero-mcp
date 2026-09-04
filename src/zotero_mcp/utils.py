import os
import re
import sys
from contextlib import contextmanager

from unidecode import unidecode

html_re = re.compile(r"<.*?>")


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout temporarily."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def format_creators(creators: list[dict[str, str] | str]) -> str:
    """
    Format creator names into a string.

    Args:
        creators: List of creator objects from Zotero.  Each element is
            typically a dict with firstName/lastName or name keys, but may
            also be a plain string (e.g. from BetterBibTeX results).

    Returns:
        Formatted string with creator names.
    """
    names = []
    for creator in creators:
        if isinstance(creator, str):
            names.append(creator)
        elif "firstName" in creator and "lastName" in creator:
            names.append(f"{creator['lastName']}, {creator['firstName']}")
        elif "name" in creator:
            names.append(creator["name"])
    return "; ".join(names) if names else "No authors listed"


def is_local_mode() -> bool:
    """Return True if running in local mode.

    Local mode is enabled when environment variable `ZOTERO_LOCAL` is set to a
    truthy value ("true", "yes", or "1", case-insensitive).
    """
    value = os.getenv("ZOTERO_LOCAL", "")
    return value.lower() in {"true", "yes", "1"}


def _paginate(zot_method, *args, max_items=None, **kwargs):
    """Fetch all pages from a pyzotero method, optionally bounding output."""
    items = []
    start = 0
    page_size = 100
    while True:
        try:
            batch = zot_method(*args, start=start, limit=page_size, **kwargs)
        except TypeError as exc:
            # Lightweight third-party clients and old test doubles sometimes
            # expose children(key) without pagination keywords. Preserve
            # compatibility while real pyzotero uses the fully paged path.
            message = str(exc)
            if start == 0 and "unexpected keyword argument" in message:
                batch = zot_method(*args, **kwargs)
                return list(batch or [])[:max_items] if max_items else list(batch or [])
            raise
        if not batch:
            break
        items.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size
        if max_items and len(items) >= max_items:
            break
    return items[:max_items] if max_items else items


_LINE_BREAK_CLOSE_RE = re.compile(r"</(?:li|tr|dt|dd)\s*>", re.IGNORECASE)
_LINE_BREAK_RE = re.compile(r"<(?:br|li|tr|dt|dd)\b[^>]*>", re.IGNORECASE)
_PARA_BREAK_RE = re.compile(
    r"</?(?:p|div|ul|ol|table|h[1-6]|blockquote|pre|section|article)\b[^>]*>",
    re.IGNORECASE,
)


def html_to_text(raw_html: str) -> str:
    """Strip HTML while preserving paragraph and line boundaries."""
    if not raw_html:
        return ""
    text = _PARA_BREAK_RE.sub("\n\n", raw_html)
    text = _LINE_BREAK_CLOSE_RE.sub("", text)
    text = _LINE_BREAK_RE.sub("\n", text)
    text = clean_html(text)
    lines = [line.strip() for line in text.splitlines()]
    output: list[str] = []
    for line in lines:
        if not line and (not output or not output[-1]):
            continue
        output.append(line)
    return "\n".join(output).strip()


def note_title(note_html: str, max_chars: int = 80) -> str:
    text = html_to_text(note_html)
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not first:
        return "Untitled Note"
    return first[:max_chars].rstrip() + ("…" if len(first) > max_chars else "")


def item_display_title(data: dict) -> str:
    item_type = data.get("itemType", "")
    if item_type == "note":
        return note_title(data.get("note", ""))
    try:
        from zotero_mcp import schema

        title = data.get(schema.resolve_field(item_type, "title"))
    except Exception:
        title = data.get("title")
    return str(title or data.get("filename") or "Untitled")


def item_display_date(data: dict) -> str:
    item_type = data.get("itemType", "")
    try:
        from zotero_mcp import schema

        value = data.get(schema.resolve_field(item_type, "date"))
    except Exception:
        value = data.get("date")
    return str(value or "")


def get_search_backend() -> str:
    """Return ``api``, ``split`` (default), or forced ``sqlite`` routing."""
    value = os.getenv("ZOTERO_SEARCH_BACKEND", "split").strip().lower()
    return value if value in {"api", "split", "sqlite"} else "split"


def use_sqlite_search(
    operation: str, *, search_all_libraries: bool = False
) -> bool:
    """Apply the configured split between live API and SQLite search."""
    backend = get_search_backend()
    if search_all_libraries:
        return backend != "api"
    if backend == "sqlite":
        return True
    if backend == "api":
        return False
    return operation == "advanced"


def library_label(item: dict) -> str | None:
    library = item.get("library")
    if not isinstance(library, dict):
        return None
    name = str(library.get("name") or "").strip()
    library_id = library.get("id")
    if library.get("type") in {"user", "users"}:
        return f"{name or 'My Library'} (personal)"
    return f"{name or 'Group'} (groupID={library_id})" if library_id is not None else name or None

def format_item_result(
    item: dict,
    index: int | None = None,
    abstract_len: int | None = 200,
    include_tags: bool = True,
    extra_fields: dict[str, str] | None = None,
    show_library: bool = False,
) -> list[str]:
    """Format a single Zotero item as markdown lines.

    Args:
        item: Zotero item dict (with ``data`` and ``key`` keys).
        index: 1-based position for numbered headings; omit for unnumbered.
        abstract_len: Max characters for abstract (``None`` = full text,
            ``0`` = omit entirely).
        include_tags: Whether to append tags.
        extra_fields: Additional ``**Label:** value`` pairs inserted after
            authors (e.g. ``{"Similarity Score": "0.912"}``).

    Returns:
        List of markdown lines (caller joins with ``"\\n"``).
    """
    data = item.get("data", {})
    # Standalone attachments carry no title — fall back to their filename.
    title = item_display_title(data)
    heading = f"## {index}. {title}" if index is not None else f"## {title}"
    lines: list[str] = [
        heading,
        f"**Type:** {data.get('itemType', 'unknown')}",
        f"**Item Key:** {item.get('key', '')}",
        f"**Date:** {item_display_date(data) or 'No date'}",
        f"**Authors:** {format_creators(data.get('creators', []))}",
    ]

    if show_library and (label := library_label(item)):
        lines.append(f"**Library:** {label}")

    # Trash status. pyzotero's default list endpoints filter trashed items
    # out, but not every call site does (e.g. includeTrashed=1, direct
    # item() lookups routed through this formatter). Defense in depth —
    # surface the flag whenever data.deleted is set so agents never silently
    # reason about a trashed paper as if it were live.
    if data.get("deleted"):
        lines.append("**Status:** 🗑️ In Trash")

    if extra_fields:
        for label, value in extra_fields.items():
            lines.append(f"**{label}:** {value}")

    if abstract_len != 0:
        abstract = data.get("abstractNote", "")
        if abstract:
            if abstract_len and len(abstract) > abstract_len:
                abstract = abstract[:abstract_len] + "..."
            lines.append(f"**Abstract:** {abstract}")

    if include_tags:
        if tags := data.get("tags"):
            tag_list = [f"`{t['tag']}`" for t in tags]
            if tag_list:
                lines.append(f"**Tags:** {' '.join(tag_list)}")

    lines.append("")  # blank separator
    return lines


def clean_html(raw_html: str, collapse_whitespace: bool = False) -> str:
    """Remove HTML/XML tags from a string.

    Args:
        raw_html: String containing HTML content.
        collapse_whitespace: If True, collapse runs of whitespace into a
            single space and strip leading/trailing whitespace. Useful for
            cleaning JATS XML from CrossRef abstracts.
    Returns:
        Cleaned string without HTML tags.
    """
    if not raw_html:
        return ""
    clean_text = re.sub(html_re, "", raw_html)
    if collapse_whitespace:
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text


# ---------------------------------------------------------------------------
# Search normalization utilities
# ---------------------------------------------------------------------------

# German umlaut expansions (common in academic literature)
_UMLAUT_MAP = {
    'ü': 'ue', 'ö': 'oe', 'ä': 'ae', 'ß': 'ss',
    'Ü': 'Ue', 'Ö': 'Oe', 'Ä': 'Ae',
}

# Dash-like Unicode characters to normalize to ASCII hyphen-minus
_DASH_PATTERN = re.compile(r'[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]')

MAX_SEARCH_VARIANTS = 15


def _normalize_for_search(text: str) -> str:
    """Normalize text for fuzzy matching: transliterate to ASCII, normalize dashes.

    Uses ``unidecode`` for broad Unicode transliteration (handles CJK, Greek,
    Cyrillic, diacritics, etc.) and a regex for dash-like characters.
    """
    if not text:
        return text
    result = unidecode(text)
    result = _DASH_PATTERN.sub('-', result)
    return result


def _generate_search_variants(query: str) -> list[str]:
    """Generate variant forms of a search query for fuzzy matching.

    Returns a deduplicated list of query variants, capped at
    ``MAX_SEARCH_VARIANTS``.  Typically produces 2-5 variants for real
    author names.
    """
    if not query or not query.strip():
        return [query] if query else []

    variants: set[str] = {query}

    # ASCII transliteration (Müller → Muller, 王 → Wang)
    ascii_form = _normalize_for_search(query)
    if ascii_form != query:
        variants.add(ascii_form)

    # Dashes to spaces (Cladder-Micus → Cladder Micus)
    dash_to_space = query.replace('-', ' ')
    if dash_to_space != query:
        variants.add(dash_to_space)
    dash_to_space_norm = ascii_form.replace('-', ' ')
    if dash_to_space_norm not in variants:
        variants.add(dash_to_space_norm)

    # German umlaut expansions (Müller → Mueller)
    umlaut_expanded = query
    for char, expansion in _UMLAUT_MAP.items():
        umlaut_expanded = umlaut_expanded.replace(char, expansion)
    if umlaut_expanded != query:
        variants.add(umlaut_expanded)

    # Spaces to dashes (Cladder Micus → Cladder-Micus)
    if ' ' in query and '-' not in query:
        space_to_dash = query.replace(' ', '-')
        variants.add(space_to_dash)

    # Cap variants
    result = list(variants)
    if len(result) > MAX_SEARCH_VARIANTS:
        result = result[:MAX_SEARCH_VARIANTS]

    return result
