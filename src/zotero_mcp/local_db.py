"""
Local Zotero database reader for semantic search.

Provides direct SQLite access to Zotero's local database for faster semantic search
when running in local mode.
"""

import hashlib
import json
import logging
import os
import platform
import re
import sqlite3
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import _normalize_for_search, is_local_mode

logger = logging.getLogger(__name__)

# Sentinel returned by _extract_text_from_pdf on timeout
_EXTRACTION_TIMEOUT = "__EXTRACTION_TIMEOUT__"

# Increment when named attachment precedence changes. It is included only in
# signatures for items whose attachments use that precedence policy.
_ATTACHMENT_SELECTION_VERSION = 2

_BETTERISSA_INDEXING_TITLE = "betterissa indexing text"
_BETTERISSA_SEMANTIC_TITLE = "betterissa semantic document"
_BETTERISSA_OCR_TITLE = "betterissa advanced ocr markdown"
_BETTERISSA_READING_VIEW_TITLE = "betterissa reading view"
_BETTERISSA_REFERENCES_TITLE = "betterissa references"
_BETTERISSA_ARTIFACT_TITLES = frozenset(
    {
        _BETTERISSA_INDEXING_TITLE,
        _BETTERISSA_SEMANTIC_TITLE,
        _BETTERISSA_OCR_TITLE,
        _BETTERISSA_READING_VIEW_TITLE,
        _BETTERISSA_REFERENCES_TITLE,
    }
)
_BETTERISSA_SELECTABLE_TITLES = (
    _BETTERISSA_ARTIFACT_TITLES - {_BETTERISSA_REFERENCES_TITLE}
)


def _extract_pdf_worker(file_path: str, maxpages: int, result_queue):
    """Legacy worker — kept for backward compatibility but no longer used.

    The actual extraction now uses subprocess.run (see _extract_text_from_pdf)
    to avoid a deadlock on macOS where multiprocessing's 'spawn' start method
    re-imports the zotero_mcp package, triggering FastMCP server initialization
    in the child process. See https://github.com/54yyyu/zotero-mcp/issues/178
    """
    try:
        import logging as _logging
        _logging.getLogger("pdfminer").setLevel(_logging.ERROR)

        from pdfminer.high_level import extract_text
        text = extract_text(file_path, maxpages=maxpages) or ""
        result_queue.put(text)
    except Exception:
        result_queue.put("")



def _read_string_pref(prefs_path: Path, pref: str) -> str | None:
    """Read a string preference from a Zotero prefs.js file.

    Returns None if the file cannot be read or the preference is absent.
    """
    try:
        text = prefs_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = re.search(
        r'user_pref\("' + re.escape(pref) + r'",\s*"([^"]*)"\)',
        text,
    )
    if not m:
        return None
    raw = m.group(1)
    # prefs.js values are JavaScript string literals; unescape backslash
    # sequences so Windows paths like C:\\Users\\... resolve correctly.
    try:
        return json.loads(f'"{raw}"')
    except ValueError:
        return raw


def _zotero_profiles_dirs() -> list[Path]:
    """Return OS-specific directories that may contain Zotero profiles."""
    system = platform.system()
    home = Path.home()
    if system == "Darwin":
        return [home / "Library" / "Application Support" / "Zotero" / "Profiles"]
    if system == "Windows":
        appdata = os.getenv("APPDATA")
        return [Path(appdata) / "Zotero" / "Zotero" / "Profiles"] if appdata else []
    # Linux and others: profile folders live directly under ~/.zotero/zotero
    return [home / ".zotero" / "zotero"]


def _profile_prefs_files() -> list[Path]:
    """Return prefs.js files from all Zotero profiles found on this system."""
    prefs_files: list[Path] = []
    for profiles_dir in _zotero_profiles_dirs():
        if profiles_dir.is_dir():
            prefs_files.extend(sorted(profiles_dir.glob("*/prefs.js")))
    return prefs_files


def _data_dirs_from_profiles() -> list[Path]:
    """Collect custom data directories declared in Zotero profiles.

    A user-configured data directory is stored as the
    ``extensions.zotero.dataDir`` preference in the profile directory's
    prefs.js — not inside the data directory itself.
    """
    data_dirs: list[Path] = []
    for prefs_path in _profile_prefs_files():
        data_dir = _read_string_pref(prefs_path, "extensions.zotero.dataDir")
        if data_dir:
            data_dirs.append(Path(data_dir))
    return data_dirs


@dataclass
class ZoteroItem:
    """Represents a Zotero item with text content for semantic search."""
    item_id: int
    key: str
    item_type_id: int
    item_type: str | None = None
    doi: str | None = None
    title: str | None = None
    abstract: str | None = None
    creators: str | None = None
    fulltext: str | None = None
    fulltext_source: str | None = None
    notes: str | None = None
    extra: str | None = None
    date_added: str | None = None
    date_modified: str | None = None

    def get_searchable_text(self) -> str:
        """
        Combine all text fields into a single searchable string.

        Returns:
            Combined text content for semantic search indexing.
        """
        parts = []

        if self.title:
            parts.append(f"Title: {self.title}")

        if self.creators:
            parts.append(f"Authors: {self.creators}")

        if self.abstract:
            parts.append(f"Abstract: {self.abstract}")

        if self.extra:
            parts.append(f"Extra: {self.extra}")

        if self.notes:
            parts.append(f"Notes: {self.notes}")

        if self.fulltext:
            # Truncate very long fulltext for simple text search
            max_chars = 50000
            truncated_fulltext = self.fulltext[:max_chars] + "..." if len(self.fulltext) > max_chars else self.fulltext
            parts.append(f"Content: {truncated_fulltext}")

        return "\n\n".join(parts)


@dataclass(frozen=True)
class _AttachmentCandidate:
    """Attachment details used to choose the best full-text source."""

    key: str
    path: str
    content_type: str | None
    title: str
    date_modified: str
    storage_mod_time: int
    item_id: int
    original_index: int


class LocalZoteroReader:
    """
    Direct SQLite reader for Zotero's local database.

    Provides fast access to item metadata and fulltext for semantic search
    without going through the Zotero API.
    """

    def __init__(self, db_path: str | None = None, pdf_max_pages: int | None = None, pdf_timeout: int = 30):
        """
        Initialize the local database reader.

        Args:
            db_path: Optional path to zotero.sqlite. If None, auto-detect.
            pdf_max_pages: Maximum pages to extract from PDFs.
            pdf_timeout: Seconds to wait for PDF extraction before killing the process.
        """
        self.db_path = db_path or self._find_zotero_db()
        self._connection: sqlite3.Connection | None = None
        self.pdf_max_pages: int | None = pdf_max_pages
        self.pdf_timeout: int = pdf_timeout
        # Reduce noise from pdfminer warnings
        try:
            logging.getLogger("pdfminer").setLevel(logging.ERROR)
        except Exception:
            pass

    def _find_zotero_db(self) -> str:
        """
        Auto-detect the Zotero database location.

        Resolution order:
        1. The ``ZOTERO_DB_PATH`` environment variable.
        2. A custom data directory configured in Zotero's preferences
           (``extensions.zotero.dataDir`` in the profile's prefs.js).
        3. The default data directory (``~/Zotero``).

        Returns:
            Path to zotero.sqlite file.

        Raises:
            FileNotFoundError: If database cannot be located.
        """
        env_path = os.getenv("ZOTERO_DB_PATH")
        if env_path:
            db_path = Path(env_path).expanduser()
            if db_path.is_file():
                return str(db_path)
            raise FileNotFoundError(
                f"ZOTERO_DB_PATH is set to {db_path}, but no file exists there."
            )

        # A data directory configured in Zotero's own preferences is
        # authoritative; a leftover ~/Zotero from an old install is not.
        candidates = [
            data_dir / "zotero.sqlite" for data_dir in _data_dirs_from_profiles()
        ]
        candidates.append(Path.home() / "Zotero" / "zotero.sqlite")
        if platform.system() == "Windows":
            # Fallback to XP/2000 location
            candidates.append(
                Path(os.path.expanduser("~/Documents and Settings"))
                / os.getenv("USERNAME", "")
                / "Zotero"
                / "zotero.sqlite"
            )

        seen: set[str] = set()
        checked: list[Path] = []
        for db_path in candidates:
            if str(db_path) in seen:
                continue
            seen.add(str(db_path))
            checked.append(db_path)
            if db_path.is_file():
                return str(db_path)

        raise FileNotFoundError(
            "Could not locate the Zotero database (checked: "
            + ", ".join(str(c) for c in checked)
            + "). If Zotero stores its data in a custom location, set the "
            "ZOTERO_DB_PATH environment variable to your zotero.sqlite file "
            "or configure the path by running `zotero-mcp setup`."
        )

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection, creating if needed."""
        if self._connection is None:
            # Use immutable=1 to bypass locking entirely. Zotero uses rollback
            # journal mode and holds a write lock while running, which blocks
            # even read-only connections. immutable=1 skips all lock checks —
            # safe here since we only read and tolerate slightly stale data.
            uri = f"file:{self.db_path}?immutable=1"
            self._connection = sqlite3.connect(uri, uri=True)
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def _get_storage_dir(self) -> Path:
        """Return the Zotero storage directory path based on database location."""
        # Infer storage directory from database path (same parent directory)
        db_parent = Path(self.db_path).parent
        return db_parent / "storage"

    def _get_base_attachment_path(self) -> Path | None:
        """Read the linked attachment base directory from Zotero's prefs.js.

        Returns the configured ``extensions.zotero.baseAttachmentPath`` or
        ``None`` if the preference is not set or cannot be read. The
        preference lives in the profile directory's prefs.js; a prefs.js
        next to the database is also checked for unusual setups.
        """
        prefs_files = [Path(self.db_path).parent / "prefs.js"]
        prefs_files.extend(_profile_prefs_files())
        for prefs_path in prefs_files:
            if not prefs_path.exists():
                continue
            value = _read_string_pref(
                prefs_path, "extensions.zotero.baseAttachmentPath"
            )
            if value:
                return Path(value)
        return None

    def _iter_parent_attachments(self, parent_item_id: int):
        """Yield tuples (attachment_key, path, content_type) for a parent item."""
        conn = self._get_connection()
        query = (
            """
            SELECT ia.itemID as attachmentItemID,
                   ia.parentItemID as parentItemID,
                   ia.path as path,
                   ia.contentType as contentType,
                   att.key as attachmentKey
            FROM itemAttachments ia
            JOIN items att ON att.itemID = ia.itemID
            WHERE ia.parentItemID = ?
            """
        )
        for row in conn.execute(query, (parent_item_id,)):
            yield row["attachmentKey"], row["path"], row["contentType"]

    def _get_attachment_selection_metadata(
        self, parent_item_id: int
    ) -> dict[str, dict[str, Any]]:
        """Return attachment titles and timestamps without changing the iterator API."""
        try:
            rows = self._get_connection().execute(
                """
                SELECT att.key AS attachmentKey,
                       att.itemID AS attachmentItemID,
                       att.dateModified AS dateModified,
                       ia.storageModTime AS storageModTime,
                       idv.value AS title
                FROM itemAttachments ia
                JOIN items att ON att.itemID = ia.itemID
                LEFT JOIN itemData id
                  ON id.itemID = att.itemID
                 AND id.fieldID = (
                     SELECT fieldID FROM fields WHERE fieldName = 'title'
                 )
                LEFT JOIN itemDataValues idv
                  ON idv.valueID = id.valueID
                WHERE ia.parentItemID = ?
                """,
                (parent_item_id,),
            ).fetchall()
        except (OSError, sqlite3.Error):
            # Some test readers and old/minimal snapshots do not expose all
            # metadata tables. Attachment paths still provide useful labels.
            return {}

        return {
            row["attachmentKey"]: {
                "title": row["title"] or "",
                "date_modified": row["dateModified"] or "",
                "storage_mod_time": row["storageModTime"] or 0,
                "item_id": row["attachmentItemID"] or 0,
            }
            for row in rows
        }

    @staticmethod
    def _normalize_attachment_title(title: str) -> str:
        return re.sub(r"\s+", " ", (title or "").casefold()).strip()

    @staticmethod
    def _uses_named_attachment_precedence(
        title: str, path: str, content_type: str | None
    ) -> bool:
        """Return whether attachment selection policy can affect this item."""
        normalized_title = LocalZoteroReader._normalize_attachment_title(title)
        filename = path.rsplit("/", 1)[-1]
        words = re.sub(
            r"[^a-z0-9]+", " ", f"{title} {filename}".casefold()
        ).split()
        is_pdf = (
            (content_type or "").lower() == "application/pdf"
            or path.lower().endswith(".pdf")
        )
        is_xml = (
            (content_type or "").lower() in {"application/xml", "text/xml"}
            or path.lower().endswith(".xml")
        )
        is_fulltext = "fulltext" in words or (
            "full" in words and "text" in words
        )
        is_betterissa = "betterissa" in words
        return (
            normalized_title in _BETTERISSA_SELECTABLE_TITLES
            or (
                is_betterissa
                and (
                    "indexing" in words
                    or ("semantic" in words and "document" in words)
                    or ("advanced" in words and "ocr" in words)
                    or ("reading" in words and "view" in words)
                )
            )
            or (is_xml and ("grobid" in words or "tei" in words))
            or (not is_pdf and not is_xml and is_fulltext)
            or (is_pdf and ("ocr" in words or is_fulltext))
        )

    def _resolve_attachment_path(self, attachment_key: str, zotero_path: str) -> Path | None:
        """Resolve a Zotero attachment path to a filesystem path.

        Handles four formats:
        - 'storage:filename.pdf' — Zotero-managed storage (most common)
        - 'file:///path/to/file.pdf' — linked file as URL
        - '/absolute/path/to/file.pdf' — linked file as absolute path
        - 'attachments:relative/path.pdf' — Zotero linked attachment base dir
        """
        if not zotero_path:
            return None

        storage_dir = self._get_storage_dir()

        # Zotero-managed storage: 'storage:filename.pdf'
        if zotero_path.startswith("storage:"):
            rel = zotero_path.split(":", 1)[1]
            parts = [p for p in rel.split("/") if p]
            return storage_dir / attachment_key / Path(*parts)

        # Linked file as URL: 'file:///path/to/file.pdf'
        if zotero_path.startswith("file://"):
            from urllib.parse import unquote, urlparse
            parsed = urlparse(zotero_path)
            decoded_path = unquote(parsed.path or "")
            # file:///C:/... on Windows
            if os.name == "nt" and decoded_path.startswith("/") and len(decoded_path) > 2 and decoded_path[2] == ":":
                decoded_path = decoded_path[1:]
            if not decoded_path:
                return None
            return Path(decoded_path)

        # Linked file as absolute path: '/Users/me/papers/file.pdf'
        if os.path.isabs(zotero_path):
            return Path(zotero_path)

        # Zotero 'attachments:' relative path — resolve against the linked
        # attachment base directory configured in Zotero preferences.
        if zotero_path.startswith("attachments:"):
            rel = zotero_path.split(":", 1)[1]
            parts = [p for p in rel.split("/") if p]
            base = self._get_base_attachment_path()
            if base and base.exists():
                return base / Path(*parts)
            # Fallback: cannot resolve without base path
            return None

        return None

    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from a PDF using pdfminer in a subprocess with timeout.

        Uses subprocess.run instead of multiprocessing.Process to avoid a
        deadlock on macOS: multiprocessing's 'spawn' start method re-imports
        the zotero_mcp package in the child process, which triggers FastMCP
        server initialization and blocks forever. subprocess.run starts a
        clean Python process that only imports pdfminer.

        See: https://github.com/54yyyu/zotero-mcp/issues/178


        Returns the extracted text, empty string on failure, or
        _EXTRACTION_TIMEOUT sentinel if the process was killed due to timeout.
        """
        import subprocess
        import sys

        # Page limit (preserve existing fallback chain)
        if isinstance(self.pdf_max_pages, int) and self.pdf_max_pages > 0:
            maxpages = self.pdf_max_pages
        else:
            max_pages_env = os.getenv("ZOTERO_PDF_MAXPAGES")
            try:
                maxpages = int(max_pages_env) if max_pages_env else 10
            except ValueError:
                maxpages = 10

        timeout = self.pdf_timeout or 30

        # Inline pdfminer script — imports ONLY pdfminer, not zotero_mcp,
        # so the child process never triggers FastMCP initialization.
        script = (
            "import sys, logging; "
            "logging.getLogger('pdfminer').setLevel(logging.ERROR); "
            "from pdfminer.high_level import extract_text; "
            "sys.stdout.write(extract_text(sys.argv[1], maxpages=int(sys.argv[2])) or '')"
        )

        # Strip API keys from the child's environment: pdfminer does not need
        # them, and leaking them via crash dumps or /proc/<pid>/environ is
        # needless exposure. Keep the rest of the env so the interpreter still
        # finds system libraries, temp dirs, locale, etc.
        child_env = os.environ.copy()
        for _k in (
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "ANTHROPIC_API_KEY",
            "ZOTERO_API_KEY",
        ):
            child_env.pop(_k, None)

        # Force UTF-8 on the child's stdio. Without this, Windows consoles
        # default to GBK/cp1252 and pdfminer extracting any non-ASCII text
        # raises UnicodeEncodeError when ``sys.stdout.write`` flushes —
        # turning a perfectly readable PDF into a "failed" extraction that
        # the indexer then refuses to retry until force-rebuild (#286).
        child_env.setdefault("PYTHONIOENCODING", "utf-8")
        child_env.setdefault("PYTHONUTF8", "1")

        try:
            result = subprocess.run(
                [sys.executable, "-c", script, str(file_path), str(maxpages)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                env=child_env,
            )
            if result.returncode == 0:
                return result.stdout
            logger.warning(
                f"PDF extraction failed (exit {result.returncode}): {file_path.name}: "
                f"{result.stderr[:200] if result.stderr else 'no error output'}"
            )
            return ""
        except subprocess.TimeoutExpired:
            sys.stderr.write(f"\r{' ' * 120}\r")  # Clear progress line before warning
            logger.warning(f"PDF extraction timed out after {timeout}s: {file_path.name}")
            return _EXTRACTION_TIMEOUT
        except Exception as e:
            sys.stderr.write(f"\r{' ' * 120}\r")  # Clear progress line before warning
            logger.warning(f"PDF extraction failed: {file_path.name}: {e}")
            return ""

    def _extract_text_from_html(self, file_path: Path) -> str:
        """Extract text from HTML using markitdown if available; fallback to stripping tags."""
        # Try markitdown first
        try:
            from markitdown import MarkItDown
            md = MarkItDown()
            result = md.convert(str(file_path))
            return result.text_content or ""
        except Exception:
            pass
        # Fallback using a simple parser
        try:
            from bs4 import BeautifulSoup  # type: ignore
            html = file_path.read_text(errors="ignore")
            return BeautifulSoup(html, "html.parser").get_text(" ")
        except Exception:
            return ""

    # Extensions / MIME types we know we can read as plain text. Used by
    # ``_is_extractable_attachment`` to gate non-PDF/HTML attachments into
    # the fulltext extractor. Binary formats (.docx, .pptx, .epub, video,
    # etc.) are intentionally excluded — ``read_text`` returns garbage for
    # those and we don't want to pollute the semantic index with it.
    _TEXTUAL_SUFFIXES = frozenset({
        ".txt", ".vtt", ".srt", ".sbv", ".md", ".markdown", ".rst",
        ".csv", ".tsv", ".json", ".xml", ".log", ".text",
    })
    _TEXTUAL_CONTENT_TYPES = frozenset({
        "text/plain", "text/vtt", "text/markdown", "text/csv",
        "text/tab-separated-values", "text/srt", "application/json",
        "application/xml", "text/xml",
    })

    @classmethod
    def _is_extractable_attachment(cls, file_path: Path, ctype: str | None) -> bool:
        """Return True when ``_extract_text_from_file`` can return useful text.

        PDF and HTML are handled by their dedicated extractors elsewhere. For
        anything else, accept the attachment iff its MIME type or extension
        is in the textual allowlist — never accept arbitrary binaries.
        """
        normalized_ctype = (ctype or "").lower()
        if normalized_ctype.startswith("text/"):
            return True
        if normalized_ctype in cls._TEXTUAL_CONTENT_TYPES:
            return True
        return file_path.suffix.lower() in cls._TEXTUAL_SUFFIXES

    def _extract_text_from_file(self, file_path: Path) -> str:
        """Extract text content from a file based on extension, with fallbacks."""
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_text_from_pdf(file_path)
        if suffix in {".html", ".htm"}:
            return self._extract_text_from_html(file_path)
        # Generic best-effort
        try:
            return file_path.read_text(errors="ignore")
        except Exception:
            return ""

    @staticmethod
    def _normalize_xml_text(element: ET.Element) -> str:
        """Collapse XML text nodes into readable prose."""
        text = re.sub(r"\s+", " ", " ".join(element.itertext())).strip()
        return re.sub(r"\s+([,.;:!?])", r"\1", text)

    def _extract_grobid_tei(self, file_path: Path) -> str:
        """Extract only abstract and body text from a GROBID TEI document."""
        try:
            root = ET.parse(file_path).getroot()
        except (ET.ParseError, OSError, ValueError) as exc:
            logger.debug("Could not parse GROBID TEI %s: %s", file_path, exc)
            return ""

        def local_name(element: ET.Element) -> str:
            return element.tag.rsplit("}", 1)[-1].lower()

        if local_name(root) != "tei":
            return ""

        abstracts: list[str] = []
        for element in root.iter():
            is_abstract = local_name(element) == "abstract"
            is_abstract_div = (
                local_name(element) == "div"
                and element.attrib.get("type", "").lower() == "abstract"
            )
            if is_abstract or is_abstract_div:
                text = self._normalize_xml_text(element)
                if text and text not in abstracts:
                    abstracts.append(text)

        bodies = [
            self._normalize_xml_text(element)
            for element in root.iter()
            if local_name(element) == "body"
        ]
        bodies = [text for text in bodies if text]
        if not bodies:
            return ""

        parts = []
        if abstracts:
            parts.append("Abstract:\n" + "\n\n".join(abstracts))
        parts.append("Body:\n" + "\n\n".join(bodies))
        return "\n\n".join(parts)

    def _extract_betterissa_semantic_document(self, file_path: Path) -> str:
        """Extract the clean text payload from a BetterIssa semantic document."""
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
            return ""
        if not isinstance(payload, dict):
            return ""
        if payload.get("ok") is False:
            return ""
        if payload.get("status") not in (None, "complete"):
            return ""

        structured = payload.get("structured")
        if isinstance(structured, dict):
            plain_text = structured.get("plain_text")
            if isinstance(plain_text, str) and plain_text.strip():
                return plain_text

        # Compatibility fallback for semantic documents that omit the
        # denormalized ``structured.plain_text`` field.
        document = payload.get("document")
        if not isinstance(document, dict):
            return ""
        parts: list[str] = []
        for field in ("title", "abstract"):
            value = document.get(field)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
        sections = document.get("sections")
        if isinstance(sections, list):
            for section in sections:
                if not isinstance(section, dict):
                    continue
                heading = section.get("heading")
                if isinstance(heading, str) and heading.strip():
                    parts.append(heading.strip())
                chunks = section.get("chunks")
                if not isinstance(chunks, list):
                    continue
                for chunk in chunks:
                    if not isinstance(chunk, dict):
                        continue
                    text = chunk.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
        return "\n\n".join(parts)

    def _extract_betterissa_ocr_markdown(self, file_path: Path) -> str:
        """Read the durable OCR checkpoint without page-marker comments."""
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return re.sub(
            r"\s*<!--\s*BetterIssa\s+page\s+\d+\s*-->\s*",
            "\n\n",
            text,
            flags=re.IGNORECASE,
        ).strip()

    def _get_fulltext_meta_for_item(
        self,
        item_id: int,
        allowed_attachment_keys: set[str] | None = None,
    ):
        meta = []
        for key, path, ctype in self._iter_parent_attachments(item_id):
            if (
                allowed_attachment_keys is not None
                and key not in allowed_attachment_keys
            ):
                continue
            meta.append([key, path, ctype])

        return meta

    def get_attachment_signature(
        self,
        item_id: int,
        allowed_attachment_keys: set[str] | None = None,
    ) -> str:
        """Return a stable change token for an item's extractable content.

        Zotero records hashes/modification times for stored attachments and a
        version for its extracted full-text cache. Linked files do not always
        have those values, so resolved file and cache stats are included as a
        fallback. This avoids reading and hashing every PDF during each scan
        while still detecting normal replacements and full-text reindexing.
        """
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT att.key AS attachmentKey,
                   att.dateModified AS attachmentDateModified,
                   ia.path AS path,
                   ia.contentType AS contentType,
                   ia.storageModTime AS storageModTime,
                   ia.storageHash AS storageHash,
                   ia.lastProcessedModificationTime AS lastProcessedModificationTime,
                   fi.version AS fulltextVersion,
                   fi.indexedChars AS indexedChars,
                   fi.totalChars AS totalChars
            FROM itemAttachments ia
            JOIN items att ON att.itemID = ia.itemID
            LEFT JOIN fulltextItems fi ON fi.itemID = ia.itemID
            WHERE ia.parentItemID = ?
            ORDER BY att.key
            """,
            (item_id,),
        ).fetchall()
        selection_metadata = self._get_attachment_selection_metadata(item_id)

        def stat_token(path: Path | None) -> tuple[int, int] | None:
            if path is None:
                return None
            try:
                stat = path.stat()
            except OSError:
                return None
            return stat.st_size, stat.st_mtime_ns

        attachments = []
        uses_named_precedence = False
        for row in rows:
            key = row["attachmentKey"]
            if (
                allowed_attachment_keys is not None
                and key not in allowed_attachment_keys
            ):
                continue
            content_type = row["contentType"]
            zotero_path = row["path"] or ""
            title = selection_metadata.get(key, {}).get("title", "")
            uses_named_precedence = (
                uses_named_precedence
                or self._uses_named_attachment_precedence(
                    title, zotero_path, content_type
                )
            )
            resolved = self._resolve_attachment_path(key, zotero_path)
            if not resolved or not resolved.exists():
                resolved = self._scan_storage_for_attachment(key, content_type)
            cache_path = self._get_storage_dir() / key / ".zotero-ft-cache"
            attachments.append(
                {
                    "key": key,
                    "path": zotero_path,
                    "content_type": content_type,
                    "date_modified": row["attachmentDateModified"],
                    "storage_mod_time": row["storageModTime"],
                    "storage_hash": row["storageHash"],
                    "last_processed_modification_time": row["lastProcessedModificationTime"],
                    "fulltext_version": row["fulltextVersion"],
                    "indexed_chars": row["indexedChars"],
                    "total_chars": row["totalChars"],
                    "file_stat": stat_token(resolved),
                    "cache_stat": stat_token(cache_path),
                }
            )

        signature_data: Any = attachments
        if uses_named_precedence:
            signature_data = {
                "attachments": attachments,
                "selection_version": _ATTACHMENT_SELECTION_VERSION,
            }
        encoded = json.dumps(
            signature_data, sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _read_zotero_ft_cache(self, attachment_key: str) -> str | None:
        """Return the text in Zotero's ``.zotero-ft-cache`` for an attachment.

        Zotero writes a plain-text full-text cache next to each indexed PDF /
        EPUB at ``storage/<attachment_key>/.zotero-ft-cache``. Using it has
        two upsides:
        - it's already-extracted text (no pdfminer subprocess needed);
        - it doesn't depend on filename matching, so it survives Zotero
          file-naming drift / non-ASCII filename rewrites (#291).

        Returns ``None`` if the cache file is absent, empty, or unreadable.
        """
        try:
            cache_path = self._get_storage_dir() / attachment_key / ".zotero-ft-cache"
        except Exception:
            return None
        if not cache_path.exists():
            return None
        try:
            text = cache_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
        return text or None

    def _scan_storage_for_attachment(
        self, attachment_key: str, ctype: str | None
    ) -> Path | None:
        """Fallback path resolver: find a likely attachment file on disk.

        ``itemAttachments.path`` in the Zotero sqlite is the filename Zotero
        recorded at import time, but the on-disk filename can drift (renames,
        non-ASCII normalization, external sync tools). When the recorded
        path no longer resolves, scan the attachment's own storage folder
        and pick the largest file whose extension is consistent with the
        recorded content type (#291).
        """
        try:
            attachment_dir = self._get_storage_dir() / attachment_key
        except Exception:
            return None
        if not attachment_dir.is_dir():
            return None

        if ctype == "application/pdf":
            wanted_suffixes = {".pdf"}
        elif (ctype or "").startswith("text/html"):
            wanted_suffixes = {".html", ".htm"}
        elif (ctype or "").startswith("application/epub"):
            wanted_suffixes = {".epub"}
        elif (ctype or "").lower() in {"application/xml", "text/xml"}:
            wanted_suffixes = {".xml"}
        elif self._is_extractable_attachment(Path("attachment"), ctype):
            wanted_suffixes = set(self._TEXTUAL_SUFFIXES)
        else:
            return None

        candidates: list[Path] = [
            child for child in attachment_dir.iterdir()
            if child.is_file() and child.suffix.lower() in wanted_suffixes
        ]
        if not candidates:
            return None
        # Largest file wins — for PDFs this is almost always the body content
        # rather than a stub or thumbnail.
        return max(candidates, key=lambda p: p.stat().st_size)

    def _extract_fulltext_for_item(
        self,
        item_id: int,
        allowed_attachment_keys: set[str] | None = None,
    ) -> tuple[str, str] | None:
        """Attempt to extract fulltext and source from the item's best attachment.

        Preference order:
        1. BetterIssa's deliberately filtered indexing text.
        2. BetterIssa semantic-document clean text.
        3. BetterIssa Advanced OCR Markdown.
        4. BetterIssa Reading View.
        5. Legacy GROBID / named fulltext sources.
        6. Any PDF, regardless of its filename or Zotero title.
        7. Any other extractable attachment.

        If the sqlite-recorded filename doesn't resolve on disk, scan the
        attachment's storage folder for a content-type-matching file before
        giving up (#291, #265).
        """
        self.last_extraction_details: dict[str, Any] | None = None
        metadata = self._get_attachment_selection_metadata(item_id)
        candidates = []
        for index, (key, path, ctype) in enumerate(
            self._iter_parent_attachments(item_id)
        ):
            if (
                allowed_attachment_keys is not None
                and key not in allowed_attachment_keys
            ):
                continue
            details = metadata.get(key, {})
            candidates.append(
                _AttachmentCandidate(
                    key=key,
                    path=path or "",
                    content_type=ctype,
                    title=details.get("title", ""),
                    date_modified=details.get("date_modified", ""),
                    storage_mod_time=details.get("storage_mod_time", 0),
                    item_id=details.get("item_id", 0),
                    original_index=index,
                )
            )

        def label(candidate: _AttachmentCandidate) -> str:
            filename = candidate.path.rsplit("/", 1)[-1]
            return re.sub(
                r"[^a-z0-9]+",
                " ",
                f"{candidate.title} {filename}".casefold(),
            ).strip()

        def normalized_title(candidate: _AttachmentCandidate) -> str:
            return self._normalize_attachment_title(candidate.title)

        def is_pdf(candidate: _AttachmentCandidate) -> bool:
            return (
                (candidate.content_type or "").lower() == "application/pdf"
                or candidate.path.lower().endswith(".pdf")
            )

        def is_xml(candidate: _AttachmentCandidate) -> bool:
            return (
                (candidate.content_type or "").lower()
                in {"application/xml", "text/xml"}
                or candidate.path.lower().endswith(".xml")
            )

        def is_json(candidate: _AttachmentCandidate) -> bool:
            return (
                (candidate.content_type or "").lower() == "application/json"
                or candidate.path.lower().endswith(".json")
            )

        def is_html(candidate: _AttachmentCandidate) -> bool:
            return (
                (candidate.content_type or "").lower().startswith("text/html")
                or candidate.path.lower().endswith((".html", ".htm"))
            )

        def is_named(candidate: _AttachmentCandidate, word: str) -> bool:
            return word in label(candidate).split()

        def has_words(candidate: _AttachmentCandidate, *words: str) -> bool:
            candidate_words = set(label(candidate).split())
            return all(word in candidate_words for word in words)

        def has_betterissa_auxiliary_marker(
            candidate: _AttachmentCandidate,
        ) -> bool:
            if normalized_title(candidate) == _BETTERISSA_REFERENCES_TITLE:
                return True
            words = set(label(candidate).split())
            return bool(
                "betterissa" in words
                and (
                    words
                    & {
                        "state",
                        "summary",
                        "references",
                        "sections",
                    }
                    or {"extraction", "metadata"} <= words
                )
            )

        def is_fulltext(candidate: _AttachmentCandidate) -> bool:
            words = label(candidate).split()
            return "fulltext" in words or (
                "full" in words and "text" in words
            )

        def is_betterissa_indexing(candidate: _AttachmentCandidate) -> bool:
            return (
                not is_pdf(candidate)
                and not is_json(candidate)
                and not is_html(candidate)
                and not has_betterissa_auxiliary_marker(candidate)
                and (
                    normalized_title(candidate) == _BETTERISSA_INDEXING_TITLE
                    or has_words(candidate, "betterissa", "indexing")
                )
            )

        def is_betterissa_semantic(candidate: _AttachmentCandidate) -> bool:
            return (
                is_json(candidate)
                and not has_betterissa_auxiliary_marker(candidate)
                and (
                    normalized_title(candidate) == _BETTERISSA_SEMANTIC_TITLE
                    or has_words(
                        candidate, "betterissa", "semantic", "document"
                    )
                )
            )

        def is_betterissa_ocr(candidate: _AttachmentCandidate) -> bool:
            return (
                not is_pdf(candidate)
                and not is_json(candidate)
                and not is_html(candidate)
                and not has_betterissa_auxiliary_marker(candidate)
                and (
                    normalized_title(candidate) == _BETTERISSA_OCR_TITLE
                    or has_words(candidate, "betterissa", "advanced", "ocr")
                )
            )

        def is_betterissa_reading_view(candidate: _AttachmentCandidate) -> bool:
            return (
                is_html(candidate)
                and not has_betterissa_auxiliary_marker(candidate)
                and (
                    normalized_title(candidate)
                    == _BETTERISSA_READING_VIEW_TITLE
                    or has_words(candidate, "betterissa", "reading", "view")
                )
            )

        def is_betterissa_auxiliary(candidate: _AttachmentCandidate) -> bool:
            words = set(label(candidate).split())
            if "betterissa" not in words:
                return False
            if has_betterissa_auxiliary_marker(candidate):
                return True
            return not (
                is_betterissa_indexing(candidate)
                or is_betterissa_semantic(candidate)
                or is_betterissa_ocr(candidate)
                or is_betterissa_reading_view(candidate)
            )

        def recency(candidate: _AttachmentCandidate):
            return (
                candidate.date_modified,
                candidate.storage_mod_time,
                candidate.item_id,
                -candidate.original_index,
            )

        def newest_first(group):
            return sorted(group, key=recency, reverse=True)

        groups = [
            (
                "betterissa-indexing",
                [
                    candidate
                    for candidate in candidates
                    if is_betterissa_indexing(candidate)
                ],
            ),
            (
                "betterissa-semantic",
                [
                    candidate
                    for candidate in candidates
                    if is_betterissa_semantic(candidate)
                ],
            ),
            (
                "betterissa-ocr",
                [
                    candidate
                    for candidate in candidates
                    if is_betterissa_ocr(candidate)
                ],
            ),
            (
                "betterissa-reading-view",
                [
                    candidate
                    for candidate in candidates
                    if is_betterissa_reading_view(candidate)
                ],
            ),
            (
                "grobid-tei",
                [
                    candidate
                    for candidate in candidates
                    if not has_betterissa_auxiliary_marker(candidate)
                    and is_xml(candidate)
                    and (
                        is_named(candidate, "grobid")
                        or is_named(candidate, "tei")
                    )
                ],
            ),
            (
                "fulltext",
                [
                    candidate
                    for candidate in candidates
                    if not has_betterissa_auxiliary_marker(candidate)
                    and not is_pdf(candidate)
                    and not is_xml(candidate)
                    and is_fulltext(candidate)
                ],
            ),
            (
                "ocr-pdf",
                [
                    candidate
                    for candidate in candidates
                    if not has_betterissa_auxiliary_marker(candidate)
                    and is_pdf(candidate)
                    and is_named(candidate, "ocr")
                ],
            ),
            (
                "pdf",
                [
                    candidate
                    for candidate in candidates
                    if not has_betterissa_auxiliary_marker(candidate)
                    and is_pdf(candidate)
                    and is_fulltext(candidate)
                ],
            ),
            (
                "pdf",
                [
                    candidate
                    for candidate in candidates
                    if not has_betterissa_auxiliary_marker(candidate)
                    and is_pdf(candidate)
                ],
            ),
            (
                "file",
                [
                    candidate
                    for candidate in candidates
                    if not is_betterissa_auxiliary(candidate)
                ],
            ),
        ]

        attempted = set()
        for source, group in groups:
            for candidate in newest_first(group):
                if candidate.key in attempted:
                    continue
                attempted.add(candidate.key)

                resolved = self._resolve_attachment_path(
                    candidate.key, candidate.path
                )
                if not resolved or not resolved.exists():
                    resolved = self._scan_storage_for_attachment(
                        candidate.key, candidate.content_type
                    )

                if source == "grobid-tei":
                    if not resolved or not resolved.exists():
                        continue
                    text = self._extract_grobid_tei(resolved)
                elif source == "betterissa-semantic":
                    if not resolved or not resolved.exists():
                        continue
                    text = self._extract_betterissa_semantic_document(resolved)
                elif source == "betterissa-ocr":
                    if not resolved or not resolved.exists():
                        continue
                    text = self._extract_betterissa_ocr_markdown(resolved)
                elif is_pdf(candidate):
                    cached = self._read_zotero_ft_cache(candidate.key)
                    if cached:
                        self.last_extraction_details = {
                            "attachment_key": candidate.key,
                            "is_pdf": True,
                            "used_zotero_cache": True,
                        }
                        return cached, "zotero-cache"
                    if not resolved or not resolved.exists():
                        continue
                    text = self._extract_text_from_file(resolved)
                else:
                    if (
                        not resolved
                        or not resolved.exists()
                        or not self._is_extractable_attachment(
                            resolved, candidate.content_type
                        )
                    ):
                        continue
                    text = self._extract_text_from_file(resolved)

                if text == _EXTRACTION_TIMEOUT:
                    self.last_extraction_details = {
                        "attachment_key": candidate.key,
                        "is_pdf": is_pdf(candidate),
                        "used_zotero_cache": False,
                    }
                    return _EXTRACTION_TIMEOUT, "timeout"
                if text:
                    if source == "file" and resolved:
                        suffix = resolved.suffix.lower()
                        source = "html" if suffix in {".html", ".htm"} else "file"
                    self.last_extraction_details = {
                        "attachment_key": candidate.key,
                        "is_pdf": is_pdf(candidate),
                        "used_zotero_cache": False,
                    }
                    return text, source
        return None

    def close(self):
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_libraries(self) -> list[dict[str, Any]]:
        """Get all libraries (user, group, feed) from the database."""
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT l.libraryID, l.type, l.editable,
                   g.groupID, g.name as groupName, g.description as groupDescription,
                   f.name as feedName, f.url as feedUrl,
                   f.lastCheck as feedLastCheck, f.lastUpdate as feedLastUpdate,
                   (SELECT COUNT(*) FROM items i
                    JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
                    WHERE i.libraryID = l.libraryID
                    AND it.typeName NOT IN ('attachment', 'note', 'annotation')) as itemCount
            FROM libraries l
            LEFT JOIN groups g ON l.libraryID = g.libraryID
            LEFT JOIN feeds f ON l.libraryID = f.libraryID
            ORDER BY l.type, l.libraryID
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def get_groups(self) -> list[dict[str, Any]]:
        """Get all group libraries with item counts."""
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT g.groupID, g.libraryID, g.name, g.description,
                   (SELECT COUNT(*) FROM items i
                    JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
                    WHERE i.libraryID = g.libraryID
                    AND it.typeName NOT IN ('attachment', 'note', 'annotation')) as itemCount
            FROM groups g
            ORDER BY g.name
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def get_feeds(self) -> list[dict[str, Any]]:
        """Get all RSS feed subscriptions with item counts."""
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT f.libraryID, f.name, f.url,
                   f.lastCheck, f.lastUpdate, f.lastCheckError,
                   f.refreshInterval,
                   (SELECT COUNT(*) FROM feedItems fi
                    JOIN items i ON fi.itemID = i.itemID
                    WHERE i.libraryID = f.libraryID) as itemCount
            FROM feeds f
            ORDER BY f.name
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def get_feed_items(
        self, library_id: int, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Get items from a specific RSS feed by its libraryID."""
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT i.itemID, i.key, it.typeName as itemType,
                   i.dateAdded,
                   fi.readTime, fi.translatedTime,
                   title_val.value as title,
                   abstract_val.value as abstract,
                   date_val.value as date,
                   doi_val.value as DOI,
                   url_val.value as url,
                   GROUP_CONCAT(
                       CASE
                           WHEN c.firstName IS NOT NULL AND c.lastName IS NOT NULL
                           THEN c.lastName || ', ' || c.firstName
                           WHEN c.lastName IS NOT NULL THEN c.lastName
                           ELSE NULL
                       END, '; '
                   ) as creators
            FROM feedItems fi
            JOIN items i ON fi.itemID = i.itemID
            JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
            LEFT JOIN itemData title_data ON i.itemID = title_data.itemID AND title_data.fieldID = 1
            LEFT JOIN itemDataValues title_val ON title_data.valueID = title_val.valueID
            LEFT JOIN itemData abstract_data ON i.itemID = abstract_data.itemID AND abstract_data.fieldID = 2
            LEFT JOIN itemDataValues abstract_val ON abstract_data.valueID = abstract_val.valueID
            LEFT JOIN fields date_f ON date_f.fieldName = 'date'
            LEFT JOIN itemData date_data ON i.itemID = date_data.itemID AND date_data.fieldID = date_f.fieldID
            LEFT JOIN itemDataValues date_val ON date_data.valueID = date_val.valueID
            LEFT JOIN fields doi_f ON doi_f.fieldName = 'DOI'
            LEFT JOIN itemData doi_data ON i.itemID = doi_data.itemID AND doi_data.fieldID = doi_f.fieldID
            LEFT JOIN itemDataValues doi_val ON doi_data.valueID = doi_val.valueID
            LEFT JOIN fields url_f ON url_f.fieldName = 'url'
            LEFT JOIN itemData url_data ON i.itemID = url_data.itemID AND url_data.fieldID = url_f.fieldID
            LEFT JOIN itemDataValues url_val ON url_data.valueID = url_val.valueID
            LEFT JOIN itemCreators ic ON i.itemID = ic.itemID
            LEFT JOIN creators c ON ic.creatorID = c.creatorID
            WHERE i.libraryID = ?
            GROUP BY i.itemID
            ORDER BY i.dateAdded DESC
            LIMIT ?
            """,
            (library_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_item_count(self) -> int:
        """
        Get total count of non-attachment items.

        Returns:
            Number of items in the library.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT COUNT(*)
            FROM items i
            JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
            WHERE it.typeName NOT IN ('attachment', 'note', 'annotation')
            AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
            """
        )
        return cursor.fetchone()[0]

    def get_all_item_keys(self) -> set[str]:
        """
        Get the keys of every item in the database, regardless of type.

        Used to verify that the sqlite snapshot is not lagging behind the
        Zotero API (an `immutable=1` read cannot see rows that are still
        in an un-checkpointed WAL file).
        """
        conn = self._get_connection()
        rows = conn.execute("SELECT key FROM items").fetchall()
        return {row[0] for row in rows}

    def get_items_with_text(self, limit: int | None = None, include_fulltext: bool = False, key_filter: str | None = None, collection_keys: list[str] | None = None) -> list[ZoteroItem]:
        """
        Get all items with their text content for semantic search.

        Args:
            limit: Optional limit on number of items to return.
            collection_keys: Optional list of collection keys; when set, only
                items in those collections (or any of their subcollections)
                are returned.

        Returns:
            List of ZoteroItem objects with text content.
        """
        conn = self._get_connection()

        # Query to get items with their text content (simplified for now)
        query = """
        SELECT
            i.itemID,
            i.key,
            i.itemTypeID,
            it.typeName as item_type,
            i.dateAdded,
            i.dateModified,
            title_val.value as title,
            abstract_val.value as abstract,
            extra_val.value as extra,
            doi_val.value as doi,
            GROUP_CONCAT(n.note, ' ') as notes,
            GROUP_CONCAT(
                CASE
                    WHEN c.firstName IS NOT NULL AND c.lastName IS NOT NULL
                    THEN c.lastName || ', ' || c.firstName
                    WHEN c.lastName IS NOT NULL
                    THEN c.lastName
                    ELSE NULL
                END, '; '
            ) as creators
        FROM items i
        JOIN itemTypes it ON i.itemTypeID = it.itemTypeID

        -- Get title
        LEFT JOIN itemData title_data ON i.itemID = title_data.itemID AND title_data.fieldID = 1
        LEFT JOIN itemDataValues title_val ON title_data.valueID = title_val.valueID

        -- Get abstract
        LEFT JOIN itemData abstract_data ON i.itemID = abstract_data.itemID AND abstract_data.fieldID = 2
        LEFT JOIN itemDataValues abstract_val ON abstract_data.valueID = abstract_val.valueID

        -- Get extra field
        LEFT JOIN itemData extra_data ON i.itemID = extra_data.itemID AND extra_data.fieldID = 16
        LEFT JOIN itemDataValues extra_val ON extra_data.valueID = extra_val.valueID

        -- Get DOI field via fields table
        LEFT JOIN fields doi_f ON doi_f.fieldName = 'DOI'
        LEFT JOIN itemData doi_data ON i.itemID = doi_data.itemID AND doi_data.fieldID = doi_f.fieldID
        LEFT JOIN itemDataValues doi_val ON doi_data.valueID = doi_val.valueID

        -- Get notes
        LEFT JOIN itemNotes n ON i.itemID = n.parentItemID OR i.itemID = n.itemID

        -- Get creators
        LEFT JOIN itemCreators ic ON i.itemID = ic.itemID
        LEFT JOIN creators c ON ic.creatorID = c.creatorID

        WHERE it.typeName NOT IN ('attachment', 'note', 'annotation')
        AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
        """

        params = []
        if collection_keys:
            # Restrict the corpus to the configured collections, including
            # all of their subcollections (resolved recursively).
            all_collection_ids = []
            for ckey in collection_keys:
                root = conn.execute("SELECT collectionID FROM collections WHERE key = ?", (ckey,)).fetchone()
                if root:
                    to_process = [root[0]]
                    while to_process:
                        cid = to_process.pop()
                        all_collection_ids.append(cid)
                        for sub in conn.execute("SELECT collectionID FROM collections WHERE parentCollectionID = ?", (cid,)).fetchall():
                            to_process.append(sub[0])
            if all_collection_ids:
                placeholders = ','.join('?' * len(all_collection_ids))
                query += f" AND i.itemID IN (SELECT DISTINCT itemID FROM collectionItems WHERE collectionID IN ({placeholders}))"
                params.extend(all_collection_ids)

        if key_filter:
            query += " AND i.key = ?"
            params.append(key_filter)

        query += """
        GROUP BY i.itemID, i.key, i.itemTypeID, it.typeName, i.dateAdded, i.dateModified,
                 title_val.value, abstract_val.value, extra_val.value

        ORDER BY i.dateModified DESC
        """

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor = conn.execute(query, params)
        items = []

        for row in cursor:
            item = ZoteroItem(
                item_id=row['itemID'],
                key=row['key'],
                item_type_id=row['itemTypeID'],
                item_type=row['item_type'],
                doi=row['doi'],
                title=row['title'],
                abstract=row['abstract'],
                creators=row['creators'],
                fulltext=(res := (self._extract_fulltext_for_item(row['itemID']) if include_fulltext else None)) and res[0],
                fulltext_source=res[1] if include_fulltext and res else None,
                notes=row['notes'],
                extra=row['extra'],
                date_added=row['dateAdded'],
                date_modified=row['dateModified']
            )
            items.append(item)

        return items

    # Public helper to quickly check full text metadata for item.
    # Returns one [key, path, content_type] row per attachment of the item.
    def get_fulltext_meta_for_item(
        self,
        item_id: int,
        allowed_attachment_keys: set[str] | None = None,
    ) -> list[list[str | None]]:
        return self._get_fulltext_meta_for_item(item_id, allowed_attachment_keys)

    # Public helper to extract fulltext on demand for a specific item
    def extract_fulltext_for_item(
        self,
        item_id: int,
        allowed_attachment_keys: set[str] | None = None,
    ) -> tuple[str, str] | None:
        return self._extract_fulltext_for_item(item_id, allowed_attachment_keys)

    def get_attachment_paths(self, parent_key: str) -> list[dict]:
        """Return resolved filesystem paths for a parent item's attachments.

        Each entry has: ``key`` (attachment key), ``content_type``, ``zotero_path``
        (the raw stored path like ``storage:foo.pdf``), ``resolved_path`` (a
        ``Path`` or ``None`` if it could not be resolved), and ``exists`` (bool).
        """
        item = self.get_item_by_key(parent_key)
        if not item:
            return []
        out: list[dict] = []
        for att_key, zotero_path, ctype in self._iter_parent_attachments(item.item_id):
            resolved = self._resolve_attachment_path(att_key, zotero_path or "")
            out.append({
                "key": att_key,
                "content_type": ctype,
                "zotero_path": zotero_path,
                "resolved_path": resolved,
                "exists": bool(resolved and resolved.exists()),
            })
        return out

    def get_item_by_key(self, key: str) -> ZoteroItem | None:
        """
        Get a specific item by its Zotero key.

        Args:
            key: The Zotero item key.

        Returns:
            ZoteroItem if found, None otherwise.
        """
        items = self.get_items_with_text(key_filter=key)
        return items[0] if items else None

    def search_items_by_text(self, query: str, limit: int = 50) -> list[ZoteroItem]:
        """
        Simple text search through item content.

        Args:
            query: Search query string.
            limit: Maximum number of results.

        Returns:
            List of matching ZoteroItem objects.
        """
        items = self.get_items_with_text()
        matching_items = []

        query_lower = _normalize_for_search(query).lower()

        for item in items:
            searchable_text = _normalize_for_search(item.get_searchable_text()).lower()
            if query_lower in searchable_text:
                matching_items.append(item)
                if len(matching_items) >= limit:
                    break

        return matching_items

    def search_notes_local(self, query: str, limit: int = 20) -> list[dict]:
        """Search notes in the local Zotero database by text content."""
        conn = self._get_connection()
        cursor = conn.cursor()
        pattern = f"%{query}%"
        cursor.execute("""
            SELECT i.key, n.note, n.title,
                   pi.key as parentKey,
                   pdv.value as parentTitle
            FROM itemNotes n
            JOIN items i ON n.itemID = i.itemID
            LEFT JOIN items pi ON n.parentItemID = pi.itemID
            LEFT JOIN itemData pd ON pi.itemID = pd.itemID AND pd.fieldID = 1
            LEFT JOIN itemDataValues pdv ON pd.valueID = pdv.valueID
            WHERE n.note LIKE ?
            AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
            LIMIT ?
        """, (pattern, limit))

        results = []
        for row in cursor.fetchall():
            note_html = row[1] or ""
            # Post-filter: skip if query only matches HTML tags, not content
            from zotero_mcp.utils import clean_html
            clean_text = clean_html(note_html)
            if query.lower() not in clean_text.lower():
                continue
            results.append({
                "type": "note",
                "key": row[0],
                "text": note_html,
                "parent_key": row[3],
                "parent_title": row[4] or ("Unknown" if row[3] else None),
                "tags": [],  # Tags require a separate query; omitted for speed
            })
        return results

    def search_annotations_local(self, query: str, limit: int = 20) -> list[dict]:
        """Search annotations in the local Zotero database by text or comment."""
        conn = self._get_connection()
        cursor = conn.cursor()
        pattern = f"%{query}%"
        # Two-hop join: annotation -> attachment -> grandparent item (for title)
        cursor.execute("""
            SELECT i.key, ia.text, ia.comment, ia.type, ia.color, ia.pageLabel,
                   att.key as attachmentKey,
                   gpi.key as parentKey,
                   gpdv.value as parentTitle
            FROM itemAnnotations ia
            JOIN items i ON ia.itemID = i.itemID
            LEFT JOIN items att ON ia.parentItemID = att.itemID
            LEFT JOIN itemAttachments iatt ON ia.parentItemID = iatt.itemID
            LEFT JOIN items gpi ON iatt.parentItemID = gpi.itemID
            LEFT JOIN itemData gpd ON gpi.itemID = gpd.itemID AND gpd.fieldID = 1
            LEFT JOIN itemDataValues gpdv ON gpd.valueID = gpdv.valueID
            WHERE (ia.text LIKE ? OR ia.comment LIKE ?)
            AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
            LIMIT ?
        """, (pattern, pattern, limit))

        # Map integer annotation types to names
        type_map = {1: "highlight", 2: "note", 3: "image", 4: "ink", 5: "underline"}

        results = []
        for row in cursor.fetchall():
            results.append({
                "type": "annotation",
                "key": row[0],
                "text": row[1] or "",
                "comment": row[2] or "",
                "annotation_type": type_map.get(row[3], "unknown"),
                "color": row[4] or "",
                "page_label": row[5] or None,
                "attachment_key": row[6],
                "parent_key": row[7],
                "parent_title": row[8] or ("Unknown" if row[7] else None),
            })
        return results


def get_local_zotero_reader() -> LocalZoteroReader | None:
    """
    Get a LocalZoteroReader instance if in local mode.

    Returns:
        LocalZoteroReader instance if in local mode and database exists,
        None otherwise.
    """
    if not is_local_mode():
        return None

    try:
        return LocalZoteroReader()
    except FileNotFoundError:
        return None


def is_local_db_available() -> bool:
    """
    Check if local Zotero database is available.

    Returns:
        True if local database can be accessed, False otherwise.
    """
    reader = get_local_zotero_reader()
    if reader:
        reader.close()
        return True
    return False
