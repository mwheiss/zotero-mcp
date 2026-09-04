"""Zotero MCP package."""

from typing import TYPE_CHECKING

from ._version import __version__ as __version__

if TYPE_CHECKING:  # pragma: no cover
    from .server import mcp as mcp

__all__ = ["__version__", "mcp"]


def __getattr__(name: str):
    """Resolve the MCP application lazily without loading it for submodules."""
    if name == "mcp":
        try:
            from .server import mcp
        except ImportError as exc:
            raise AttributeError(
                f"zotero_mcp.mcp is unavailable: server dependencies missing ({exc})"
            ) from exc
        return mcp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)

# These modules are not imported by default but are available
# pdfannots_helper and pdfannots_downloader
