"""ChatGPT connector tool functions (search & fetch)."""

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from zotero_mcp import client as _client
from zotero_mcp import utils as _utils
from zotero_mcp._app import mcp
from zotero_mcp._context import Context, context_error
from zotero_mcp.tools.retrieval import get_item_fulltext

# These are required for ChatGPT custom MCP servers via web "connectors"
# specific tools required are "search" and "fetch"
# See: https://platform.openai.com/docs/mcp

_ITEM_KEY_RE = re.compile(r"^[A-Z0-9]{8}$")


class ConnectorSearchResult(BaseModel):
    """One result in the OpenAI MCP search compatibility contract."""

    id: str
    title: str
    url: str


class ConnectorSearchOutput(BaseModel):
    """Structured output required by ChatGPT search integrations."""

    results: list[ConnectorSearchResult]


class ConnectorFetchOutput(BaseModel):
    """Structured output required by ChatGPT fetch integrations."""

    id: str
    title: str
    text: str
    url: str
    metadata: dict[str, Any] | None = None


def _item_urls(
    item_key: str,
    item: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Return active-library-aware ``(select_url, web_url)`` links."""
    library = _client.get_current_library()
    library_type = (library.get("library_type") or "user").lower()
    library_id = str(library.get("library_id") or "")
    item_library_id = str(((item or {}).get("library") or {}).get("id") or "")

    if library_type == "group" and library_id:
        select_url = f"zotero://select/groups/{library_id}/items/{item_key}"
        web_url = f"https://www.zotero.org/groups/{library_id}/items/{item_key}"
    elif library_type == "feed" and library_id:
        select_url = f"zotero://select/feeds/{library_id}/items/{item_key}"
        web_url = ""
    else:
        select_url = f"zotero://select/library/items/{item_key}"
        # The local Zotero API identifies My Library as user:0. It has no
        # corresponding web-library URL; a synced Web API client has a real ID.
        web_library_id = library_id if library_id and library_id != "0" else item_library_id
        web_url = (
            f"https://www.zotero.org/users/{web_library_id}/items/{item_key}"
            if web_library_id
            else ""
        )
    return select_url, web_url


def _citation_url(item_key: str, item: dict[str, Any] | None = None) -> str:
    """Return the same preferred citation URL for connector search and fetch."""
    select_url, web_url = _item_urls(item_key, item)
    return web_url or select_url

@mcp.tool(
    name="search",
    output_schema=ConnectorSearchOutput.model_json_schema(),
    description=(
        "ChatGPT custom connector SEARCH endpoint — name is REQUIRED by "
        "the MCP-over-web spec (see platform.openai.com/docs/mcp); "
        "do not rename. Not intended for general MCP clients — in Claude "
        "or other regular MCP contexts use zotero_semantic_search or "
        "zotero_search_items instead, which return richer markdown. "
        "Performs semantic search over the active Zotero library and "
        "returns structured {\"results\":[{\"id\",\"title\",\"url\"}, "
        "...]} data matching the ChatGPT connector citation UI. URLs are "
        "web-library links when available, otherwise active-library-aware "
        "zotero://select deep-links. "
        "query: topic string; natural language works (embedding match). "
        "No limit parameter — fixed at 10 per the connector UI's "
        "expected result-set size. "
        "Requires the semantic search DB populated — run "
        "zotero_update_search_database first if empty. "
        "Backend failures raise an MCP tool error; an empty results array means "
        "the search completed successfully with no matches. "
        "Example (agent-invoked): search(query='mindfulness-based "
        "therapy')."
    )
)
def chatgpt_connector_search(
    query: str = Field(
        description="Natural-language query for semantic search over the active Zotero library."
    ),
    *,
    ctx: Context
) -> ConnectorSearchOutput:
    """
    Return the structured search compatibility object. FastMCP mirrors the
    same value into the text content block for legacy clients.
    """
    try:
        default_limit = 10

        from zotero_mcp.semantic_search import create_semantic_search

        config_path = Path.home() / ".config" / "zotero-mcp" / "config.json"
        search = create_semantic_search(str(config_path))

        result_list: list[ConnectorSearchResult] = []
        results = search.search(query=query, limit=default_limit, filters=None) or {}
        for r in results.get("results", []):
            raw_item_key = r.get("item_key")
            item_key = raw_item_key.strip() if isinstance(raw_item_key, str) else ""
            # Connector search IDs are later passed verbatim to fetch. Returning
            # a synthetic or malformed ID creates an unfetchable citation.
            if not _ITEM_KEY_RE.fullmatch(item_key):
                continue
            title = ""
            zotero_item = r.get("zotero_item") or {}
            if zotero_item:
                data = zotero_item.get("data", {})
                title = data.get("title", "")
            if not title:
                title = f"Zotero Item {item_key}" if item_key else "Zotero Item"
            url = _citation_url(item_key, zotero_item)
            result_list.append(
                ConnectorSearchResult(id=item_key, title=title, url=url)
            )

        return ConnectorSearchOutput(results=result_list)
    except Exception as e:
        context_error(ctx, f"Error in search wrapper: {str(e)}")
        raise RuntimeError(f"Connector search failed: {e}") from e


@mcp.tool(
    name="fetch",
    output_schema=ConnectorFetchOutput.model_json_schema(),
    description=(
        "ChatGPT custom connector FETCH endpoint — name is REQUIRED by "
        "the MCP-over-web spec (see platform.openai.com/docs/mcp); "
        "do not rename. Not intended for general MCP clients — in Claude "
        "or other regular MCP contexts use zotero_get_item_fulltext and "
        "zotero_get_item_metadata, which return richer markdown. "
        "Retrieves a single Zotero item and returns a JSON envelope "
        "{\"id\",\"title\",\"text\",\"url\",\"metadata\":{...}} matching "
        "the ChatGPT connector citation viewer. "
        "id: an 8-char Zotero item key — typically from a previous "
        "`search` call. Blank, malformed, or missing IDs raise a tool error. "
        "url field: Zotero web-library URL when a synced user/group identity is "
        "available; otherwise an active-library-aware zotero://select deep-link. "
        "text field: extracted fulltext via the same path as "
        "zotero_get_item_fulltext; if none can be extracted, falls back "
        "to title + authors + abstract so the connector isn't blank. "
        "metadata field: itemType, date, DOI, authors, tags, both URLs. "
        "Invalid IDs and backend failures raise MCP tool errors rather than "
        "returning a document-shaped false success. "
        "Example (agent-invoked): fetch(id='RTKZQI8E')."
    )
)
def connector_fetch(
    id: str = Field(
        description="Exact 8-character Zotero item key returned by the connector search tool."
    ),
    *,
    ctx: Context
) -> ConnectorFetchOutput:
    """
    Return the structured fetch compatibility object. FastMCP mirrors the
    same value into the text content block for legacy clients.
    """
    try:
        item_key = (id or "").strip()
        if not item_key:
            raise ValueError("missing item key")
        if not _ITEM_KEY_RE.fullmatch(item_key):
            raise ValueError("invalid Zotero item key")

        # Fetch item metadata for title and context
        zot = _client.get_zotero_client()
        try:
            item = zot.item(item_key)
            data = item.get("data", {}) if item else {}
        except Exception as exc:
            raise LookupError(f"Zotero item {item_key} could not be retrieved: {exc}") from exc
        if not item:
            raise LookupError(f"Zotero item {item_key} was not found")

        title = data.get("title", f"Zotero Item {item_key}")
        zotero_url, web_url = _item_urls(item_key, item)
        # Prefer a shareable web URL; local-only and feed libraries use the
        # Zotero desktop deep-link instead.
        url = web_url or zotero_url

        # Use existing tool to get best-effort fulltext/markdown
        text_md = get_item_fulltext(item_key=item_key, ctx=ctx)
        # Extract the actual full text section if present, else keep as-is
        text_clean = text_md
        try:
            marker = "## Full Text"
            pos = text_md.find(marker)
            if pos >= 0:
                text_clean = text_md[pos + len(marker):].lstrip("\n #")
        except Exception:
            pass
        if (not text_clean or len(text_clean.strip()) < 40) and data:
            abstract = data.get("abstractNote", "")
            creators = data.get("creators", [])
            byline = _utils.format_creators(creators)
            text_clean = (f"{title}\n\n" + (f"Authors: {byline}\n" if byline else "") +
                          (f"Abstract:\n{abstract}" if abstract else "")) or text_md

        metadata = {
            "itemType": data.get("itemType", ""),
            "date": data.get("date", ""),
            "key": item_key,
            "doi": data.get("DOI", ""),
            "isbn": data.get("ISBN", ""),
            "issn": data.get("ISSN", ""),
            "publisher": data.get("publisher", ""),
            "place": data.get("place", ""),
            "authors": _utils.format_creators(data.get("creators", [])),
            "tags": [t.get("tag", "") for t in (data.get("tags", []) or [])],
            "zotero_url": zotero_url,
            "web_url": web_url,
            "source": "zotero-mcp"
        }

        return ConnectorFetchOutput(
            id=item_key,
            title=title,
            text=text_clean,
            url=url,
            metadata=metadata,
        )
    except Exception as e:
        context_error(ctx, f"Error in fetch wrapper: {str(e)}")
        raise
