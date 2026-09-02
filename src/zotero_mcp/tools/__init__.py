"""Tool modules — importing this package registers all tools with the MCP app."""

# Signed attachment transfer routes share the same HTTP app but are not MCP tools.
from zotero_mcp import attachment_http as attachment_http  # noqa: F401,E402

# Register MCP prompts (research workflows) and resources (library context).
# Importing these is a side effect that binds their @mcp.prompt / @mcp.resource
# handlers, exactly like the tool modules above.
from zotero_mcp import prompts as prompts  # noqa: F401,E402
from zotero_mcp import resources as resources  # noqa: F401,E402
from zotero_mcp.tools import (  # noqa: F401
    annotations,
    attachments,
    connectors,
    discovery,
    operational,
    read_pdf,
    retrieval,
    search,
    synthesis,
    write,
)

# Scite enrichment uses the core HTTP dependency and is always available.
# The empty ``scite`` package extra remains as an installation compatibility alias.
from zotero_mcp.tools import scite as scite  # noqa: F401
