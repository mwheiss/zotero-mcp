"""MCP resources: expose the Zotero library as attachable context.

Importing this module registers each ``@mcp.resource`` with the FastMCP app (a
side effect, mirroring the tool modules). Resources let an MCP host attach
library content (a collection listing, a single item, a collection's items) as
context directly, without the model having to issue a tool call for each.

The Zotero client is reached lazily via module-level attribute access
(``_client.get_zotero_client()``) so tests can monkeypatch it, and every
resource holds the shared API lock while talking to Zotero.
"""

import tempfile

from zotero_mcp import client as _client
from zotero_mcp import utils as _utils
from zotero_mcp._app import mcp
from zotero_mcp.client import with_zotero_api_lock
from zotero_mcp.tools import _helpers


@mcp.resource(
    "zotero://collections",
    name="Zotero collections",
    description="All collections in the active Zotero library (name, key, and parent key).",
    mime_type="text/markdown",
)
@with_zotero_api_lock
def collections_resource() -> str:
    """List every collection in the active library as markdown."""
    try:
        zot = _client.get_zotero_client()
        collections = _helpers._paginate(zot.collections)
        if not collections:
            return "# Zotero Collections\n\nNo collections found."
        lines = ["# Zotero Collections", ""]
        for c in collections:
            data = c.get("data", {})
            name = data.get("name", "Untitled")
            key = c.get("key", "")
            parent = data.get("parentCollection")
            suffix = f" — child of {parent}" if parent else ""
            lines.append(f"- **{name}** (`{key}`){suffix}")
        return "\n".join(lines)
    except Exception as e:
        return f"# Zotero Collections\n\nError loading collections: {e}"


@mcp.resource(
    "zotero://items/{item_key}",
    name="Zotero item",
    description=(
        "Summary bibliographic metadata for one Zotero item by its 8-character "
        "key. This resource does not include attachment full text."
    ),
    mime_type="text/markdown",
)
@with_zotero_api_lock
def item_resource(item_key: str) -> str:
    """Return one item's metadata as markdown."""
    try:
        zot = _client.get_zotero_client()
        try:
            item = zot.item(item_key)
        except Exception:
            return f"# Item {item_key}\n\nNo item found with key: {item_key}"
        if not item:
            return f"# Item {item_key}\n\nNo item found with key: {item_key}"
        lines = [f"# Item {item_key}", ""]
        lines.extend(_utils.format_item_result(item))
        return "\n".join(lines)
    except Exception as e:
        return f"# Item {item_key}\n\nError loading item: {e}"


@mcp.resource(
    "zotero://collections/{collection_key}/items",
    name="Zotero collection items",
    description=(
        "Summary metadata for up to 200 items in a Zotero collection, by "
        "collection key. Attachment full text is not included."
    ),
    mime_type="text/markdown",
)
@with_zotero_api_lock
def collection_items_resource(collection_key: str) -> str:
    """Return the items in a collection as markdown."""
    try:
        zot = _client.get_zotero_client()
        try:
            coll = zot.collection(collection_key)
            coll_name = coll.get("data", {}).get("name", collection_key)
        except Exception:
            coll_name = collection_key
        items = _helpers._paginate(zot.collection_items, collection_key, max_items=200, itemType="-attachment")
        if not items:
            return f"# Collection: {coll_name}\n\nNo items found in this collection."
        lines = [f"# Collection: {coll_name} (`{collection_key}`)", ""]
        for i, item in enumerate(items, 1):
            lines.extend(_utils.format_item_result(item, index=i))
        return _helpers._prepend_size_warning("\n".join(lines))
    except Exception as e:
        return f"# Collection {collection_key}\n\nError loading collection items: {e}"


@mcp.resource(
    "zotero://attachments/{attachment_key}/content",
    name="Zotero attachment binary",
    description=(
        "Exact binary content for one Zotero attachment. Intended for MCP clients "
        "that support binary resources; use zotero_get_attachment for a streaming "
        "download URL when the file is large."
    ),
    mime_type="application/octet-stream",
)
def attachment_binary_resource(attachment_key: str) -> bytes:
    with tempfile.TemporaryDirectory() as temporary:
        zot = _client.get_zotero_client()
        item = zot.item(attachment_key)
        data = item.get("data", {}) if item else {}
        if data.get("itemType") != "attachment":
            raise ValueError(f"No attachment found with key {attachment_key}")
        filename = data.get("filename") or f"{attachment_key}.bin"
        result = _client.download_attachment_file(
            attachment_key,
            temporary,
            filename,
            local_client=_client.get_local_zotero_client(),
            web_client=_client.get_web_zotero_client(),
        )
        if not result.path:
            raise ValueError("Attachment binary unavailable: " + "; ".join(result.errors))
        return result.path.read_bytes()
