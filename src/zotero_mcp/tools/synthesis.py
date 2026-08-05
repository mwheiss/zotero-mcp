"""Synthesis and export tool functions for the Zotero MCP server.

These tools gather and structure existing library content (annotations,
notes, citations) so an LLM agent can synthesize literature summaries or
drop formatted references into a manuscript. They do NOT call an LLM
themselves; they only collect and format.
"""

from typing import Literal

from zotero_mcp import client as _client
from zotero_mcp import utils as _utils
from zotero_mcp._app import mcp
from zotero_mcp._context import Context, context_error, context_info, context_warning
from zotero_mcp.tools import _helpers


def _resolve_paper_identity(
    zot,
    parent_key: str,
    cache: dict[str, tuple[str, str]],
) -> tuple[str, str]:
    """Resolve an annotation/note parent key to (paper key, title).

    Annotations are children of PDF/EPUB attachments, which are children of
    the paper (two hops: annotation.parentItem -> attachment ->
    attachment.parentItem -> paper). Notes are usually direct children of the
    paper (one hop). This helper tolerates either shape and any missing hop,
    falling back to the attachment title or the bare key. Results are cached.
    """
    if parent_key in cache:
        return cache[parent_key]

    paper_key = parent_key
    title = parent_key
    try:
        parent = zot.item(parent_key)
        data = parent.get("data", {}) if parent else {}
        if data.get("itemType") == "attachment" and data.get("parentItem"):
            gp_key = data["parentItem"]
            paper_key = gp_key
            try:
                grandparent = zot.item(gp_key)
                gp_data = grandparent.get("data", {}) if grandparent else {}
                title = gp_data.get("title") or data.get("title") or parent_key
            except Exception:
                title = data.get("title") or parent_key
        else:
            title = data.get("title") or parent_key
    except Exception:
        title = parent_key

    identity = (paper_key, title)
    cache[parent_key] = identity
    return identity


@mcp.tool(
    name="zotero_synthesize_annotations",
    description=(
        "Collect every highlight, annotation comment, and child note across "
        "a scope and organize them into a structured, per-paper digest that "
        "YOU (the agent) can then synthesize into a literature summary. This "
        "tool does NOT call an LLM — it only gathers and groups the raw "
        "material, so the synthesis step is yours. "
        "collection_key: optional 8-character collection key; when given, "
        "only annotations/notes whose resolved paper is a member of that "
        "collection are included. When omitted, the whole active library is "
        "scanned (capped by limit). "
        "tag: optional tag or list of tags to filter items by (accepts a "
        "string, a JSON list, or a list). "
        "limit: combined cap on annotations plus notes processed (default 200) "
        "to keep the "
        "call tractable. "
        "Output: markdown grouped by paper — each paper heading followed by "
        "its highlights (with attached comments) and any note excerpts — "
        "plus a top summary line counting papers, highlights, and notes. "
        "Use this before writing a thematic review so you can spot themes "
        "and contradictions across sources. "
        "Example: zotero_synthesize_annotations(collection_key='MT53KB66')."
    ),
)
def synthesize_annotations(
    collection_key: str | None = None,
    tag: list[str] | str | None = None,
    limit: int | str | None = 200,
    *,
    ctx: Context,
) -> str:
    """Gather annotations and notes into a per-paper digest for synthesis.

    Args:
        collection_key: Optional collection to restrict the digest to.
        tag: Optional tag filter (string, JSON list, or list).
        limit: Maximum annotations/notes to scan.
        ctx: MCP context.

    Returns:
        Markdown digest grouped by paper.
    """
    try:
        context_info(ctx, "Gathering annotations and notes for synthesis")
        zot = _client.get_zotero_client()

        limit = _helpers._normalize_limit(limit, default=200, max_val=5000)
        tags = _helpers._normalize_tag_filter(tag)

        # Determine the set of allowed paper keys if a collection is scoped.
        allowed_keys: set[str] | None = None
        if collection_key:
            try:
                coll_items = _helpers._paginate(
                    zot.collection_items,
                    collection_key,
                    itemType="-attachment",
                )
                allowed_keys = {it.get("key") for it in coll_items if it.get("key")}
            except Exception as e:
                context_warning(ctx, f"Could not load collection items: {e}")
                allowed_keys = set()

        anno_params = {"itemType": "annotation"}
        note_params = {"itemType": "note"}
        if tags:
            anno_params["tag"] = tags
            note_params["tag"] = tags

        try:
            annotations = _helpers._paginate(zot.items, max_items=limit, **anno_params)
        except Exception as e:
            context_warning(ctx, f"Annotation fetch failed: {e}")
            annotations = []
        try:
            notes = _helpers._paginate(zot.items, max_items=limit, **note_params)
        except Exception as e:
            context_warning(ctx, f"Note fetch failed: {e}")
            notes = []

        combined = [("annotation", item) for item in annotations]
        combined.extend(("note", item) for item in notes)
        combined.sort(
            key=lambda pair: (
                pair[1].get("data", {}).get("dateModified", ""),
                pair[1].get("data", {}).get("dateAdded", ""),
                pair[1].get("key", ""),
            ),
            reverse=True,
        )
        combined = combined[:limit]
        annotations = [item for kind, item in combined if kind == "annotation"]
        notes = [item for kind, item in combined if kind == "note"]

        if not annotations and not notes:
            scope = f" in collection {collection_key}" if collection_key else ""
            return f"No annotations or notes found{scope}."

        # Group by stable paper key; distinct papers may share a title.
        title_cache: dict[str, tuple[str, str]] = {}
        papers: dict[str, dict] = {}

        def _bucket(paper_key: str, title: str) -> dict:
            return papers.setdefault(
                paper_key,
                {"title": title, "highlights": [], "notes": []},
            )

        def _in_scope(parent_key: str) -> bool:
            if allowed_keys is None:
                return True
            # Member if the immediate parent, or its grandparent paper, is in scope.
            if parent_key in allowed_keys:
                return True
            try:
                parent = zot.item(parent_key)
                data = parent.get("data", {}) if parent else {}
                gp = data.get("parentItem")
                if gp and gp in allowed_keys:
                    return True
            except Exception:
                pass
            return False

        highlight_count = 0
        for anno in annotations:
            data = anno.get("data", {})
            parent_key = data.get("parentItem")
            text = (data.get("annotationText") or "").strip()
            comment = (data.get("annotationComment") or "").strip()
            if not text and not comment:
                continue
            if parent_key and not _in_scope(parent_key):
                continue
            paper_key, title = (
                _resolve_paper_identity(zot, parent_key, title_cache)
                if parent_key
                else ("unknown", "(unknown source)")
            )
            _bucket(paper_key, title)["highlights"].append((text, comment))
            highlight_count += 1

        note_count = 0
        for note in notes:
            data = note.get("data", {})
            parent_key = data.get("parentItem")
            raw = data.get("note") or ""
            text = _utils.clean_html(raw).strip()
            if not text:
                continue
            if parent_key and not _in_scope(parent_key):
                continue
            paper_key, title = (
                _resolve_paper_identity(zot, parent_key, title_cache)
                if parent_key
                else (note.get("key", "standalone"), "(standalone note)")
            )
            if len(text) > 400:
                text = text[:400] + "..."
            _bucket(paper_key, title)["notes"].append(text)
            note_count += 1

        if not papers:
            scope = f" in collection {collection_key}" if collection_key else ""
            return f"No annotations or notes found{scope}."

        output = [
            "# Annotation & Note Digest",
            "",
            (f"**{len(papers)} papers, {highlight_count} highlights, {note_count} notes**"),
            "",
        ]

        for paper_key, bucket in sorted(
            papers.items(), key=lambda entry: (entry[1]["title"], entry[0])
        ):
            output.append(f"## {bucket['title']} (`{paper_key}`)")
            if bucket["highlights"]:
                output.append("**Highlights:**")
                for text, comment in bucket["highlights"]:
                    line = f"- {text}" if text else "- (comment only)"
                    if comment:
                        line += f" — *{comment}*"
                    output.append(line)
            if bucket["notes"]:
                output.append("**Notes:**")
                for note_text in bucket["notes"]:
                    output.append(f"- {note_text}")
            output.append("")

        output.append(
            "*You can now synthesize themes, agreements, and contradictions across these papers from the digest above.*"
        )

        result = "\n".join(output)
        return _helpers._prepend_size_warning(
            result,
            "Scope to a collection_key or narrow with tag to reduce size.",
        )

    except Exception as e:
        context_error(ctx, f"Error synthesizing annotations: {str(e)}")
        return f"Error synthesizing annotations: {str(e)}"


def _render_entries(rendered) -> list[str]:
    """Normalize pyzotero content output into a list of plain-text entries."""
    if rendered is None:
        return []
    if isinstance(rendered, str):
        return [rendered]
    entries: list[str] = []
    for item in rendered:
        entries.append(item if isinstance(item, str) else str(item))
    return entries


@mcp.tool(
    name="zotero_export_bibliography",
    description=(
        "Render a formatted bibliography or in-text citations for a set of "
        "Zotero items using Zotero's own CSL citation engine, so you can drop "
        "references straight into a manuscript. "
        "item_keys: optional list of 8-character item keys (also accepts a "
        "JSON list string); takes precedence over collection_key. "
        "collection_key: optional collection to export instead; if neither is "
        "given, the active library is exported (capped). "
        "style: CSL style short name (default 'apa'); e.g. 'modern-language-"
        "association', 'chicago-note-bibliography', 'ieee'. Ignored for "
        "bibtex. "
        "export_format: 'bib' (formatted reference-list entries, default), "
        "'citation' (in-text citation strings), or 'bibtex' (raw BibTeX for "
        ".bib files). "
        "Output: markdown naming the style/format, then the rendered entries "
        "(a fenced block for bibtex, a numbered list otherwise). "
        "Requires bibliography rendering support — if the API errors (e.g. "
        "local read-only mode), the tool suggests web API mode. "
        "Example: zotero_export_bibliography(item_keys=['RTKZQI8E'], "
        "style='apa', export_format='bib')."
    ),
)
def export_bibliography(
    item_keys: list[str] | str | None = None,
    collection_key: str | None = None,
    style: str = "apa",
    export_format: Literal["bib", "citation", "bibtex"] = "bib",
    *,
    ctx: Context,
) -> str:
    """Render a formatted bibliography/citations via Zotero's CSL engine.

    Args:
        item_keys: Optional list of item keys (or JSON/comma string).
        collection_key: Optional collection to export.
        style: CSL style short name (default "apa").
        export_format: "bib", "citation", or "bibtex".
        ctx: MCP context.

    Returns:
        Markdown-formatted bibliography or citations.
    """
    try:
        if not isinstance(style, str) or not style.strip():
            style = "apa"
        style = style.strip()

        keys: list[str] = []
        if item_keys is not None:
            keys = _helpers._normalize_str_list_input(item_keys, "item_keys")

        context_info(ctx, f"Exporting bibliography (format={export_format}, style={style})")
        zot = _client.get_zotero_client()

        content = "bibtex" if export_format == "bibtex" else export_format

        try:
            if keys:
                fetch_kwargs = {"itemKey": ",".join(keys), "content": content, "limit": 100}
                if content != "bibtex":
                    fetch_kwargs["style"] = style
                rendered = zot.items(**fetch_kwargs)
            elif collection_key:
                page_kwargs = {"content": content}
                if content != "bibtex":
                    page_kwargs["style"] = style
                rendered = _helpers._paginate(zot.collection_items, collection_key, max_items=100, **page_kwargs)
            else:
                fetch_kwargs = {"content": content, "limit": 100}
                if content != "bibtex":
                    fetch_kwargs["style"] = style
                rendered = zot.items(**fetch_kwargs)
        except Exception as api_error:
            context_error(ctx, f"Bibliography rendering failed: {api_error}")
            return (
                f"Error rendering bibliography: {api_error}\n\n"
                "Bibliography/citation rendering relies on Zotero's web API "
                "CSL engine. If you are running in local read-only mode, "
                "configure web API credentials (ZOTERO_API_KEY and "
                "ZOTERO_LIBRARY_ID) and try again."
            )

        entries = _render_entries(rendered)
        if not entries:
            scope = (
                f" for collection {collection_key}" if collection_key else (" for the requested items" if keys else "")
            )
            return f"No bibliography entries produced{scope}."

        format_label = {
            "bib": "Bibliography",
            "citation": "Citations",
            "bibtex": "BibTeX",
        }[export_format]

        if export_format == "bibtex":
            body = "\n\n".join(e.strip() for e in entries if e.strip())
            return f"# {format_label}\n\n```bibtex\n{body}\n```"

        header = f"# {format_label} ({style})"
        lines = [header, ""]
        for i, entry in enumerate(entries, 1):
            clean = _utils.clean_html(entry).strip()
            if not clean:
                continue
            lines.append(f"{i}. {clean}")
        return "\n".join(lines)

    except Exception as e:
        context_error(ctx, f"Error exporting bibliography: {str(e)}")
        return f"Error exporting bibliography: {str(e)}"
