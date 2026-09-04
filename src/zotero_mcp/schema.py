"""Resolve Zotero base fields to each item type's actual API field."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Literal

from zotero_mcp._atomic_io import atomic_write_json
from zotero_mcp._file_lock import advisory_file_lock

SchemaSource = Literal["auto", "local", "web"]

WEB_SCHEMA_URL = "https://api.zotero.org/schema"

# Zotero's schema uses these type-specific fields for otherwise generic base
# fields. Keeping this table local makes reads and writes correct offline; the
# active item's template remains authoritative for field validity.
_ALIASES: dict[str, dict[str, str]] = {
    "artwork": {"medium": "artworkMedium"},
    "audioRecording": {"medium": "audioRecordingFormat", "publisher": "label"},
    "bill": {"number": "billNumber", "pages": "codePages", "volume": "codeVolume", "authority": "legislativeBody"},
    "blogPost": {"publicationTitle": "blogTitle", "type": "websiteType"},
    "book": {"medium": "format"},
    "bookSection": {"publicationTitle": "bookTitle", "medium": "format"},
    "case": {"title": "caseName", "authority": "court", "date": "dateDecided", "number": "docketNumber", "pages": "firstPage", "volume": "reporterVolume"},
    "computerProgram": {"publisher": "company"},
    "conferencePaper": {"publicationTitle": "proceedingsTitle"},
    "dataset": {"medium": "format", "number": "identifier", "publisher": "repository", "place": "repositoryLocation"},
    "dictionaryEntry": {"publicationTitle": "dictionaryTitle"},
    "email": {"title": "subject"},
    "encyclopediaArticle": {"publicationTitle": "encyclopediaTitle"},
    "film": {"publisher": "distributor", "type": "genre", "medium": "videoRecordingFormat"},
    "forumPost": {"publicationTitle": "forumTitle", "type": "postType"},
    "hearing": {"number": "documentNumber", "authority": "legislativeBody"},
    "interview": {"medium": "interviewMedium"},
    "letter": {"type": "letterType"},
    "manuscript": {"publisher": "institution", "type": "manuscriptType"},
    "map": {"type": "mapType"},
    "patent": {"date": "issueDate", "authority": "issuingAuthority", "status": "legalStatus", "number": "patentNumber", "originalDate": "priorityDate"},
    "podcast": {"medium": "audioFileType", "number": "episodeNumber"},
    "preprint": {"number": "archiveID", "type": "genre", "publisher": "repository"},
    "presentation": {"type": "presentationType", "publicationTitle": "sessionTitle"},
    "radioBroadcast": {"medium": "audioRecordingFormat", "number": "episodeNumber", "publisher": "network", "publicationTitle": "programTitle"},
    "report": {"publisher": "institution", "number": "reportNumber", "type": "reportType"},
    "standard": {"authority": "organization"},
    "statute": {"date": "dateEnacted", "title": "nameOfAct", "number": "publicLawNumber"},
    "thesis": {"type": "thesisType", "publisher": "university"},
    "tvBroadcast": {"number": "episodeNumber", "publisher": "network", "publicationTitle": "programTitle", "medium": "videoRecordingFormat"},
    "videoRecording": {"publisher": "studio", "medium": "videoRecordingFormat"},
    "webpage": {"publicationTitle": "websiteTitle", "type": "websiteType"},
}

_table_cache: dict[str, dict[str, Any]] = {}


def resolve_source(source: SchemaSource = "auto") -> Literal["local", "web"]:
    """Resolve automatic schema routing without mixing local and web caches."""
    if source not in {"auto", "local", "web"}:
        raise ValueError("schema source must be auto, local, or web")
    if source != "auto":
        return source
    local = os.getenv("ZOTERO_LOCAL", "").strip().casefold() in {
        "1",
        "true",
        "yes",
    }
    return "local" if local else "web"


def _cache_directory() -> Path:
    configured = os.getenv("ZOTERO_MCP_SCHEMA_CACHE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    base = Path(os.getenv("XDG_CACHE_HOME", "").strip() or Path.home() / ".cache")
    return base / "zotero-mcp"


def cache_path(source: SchemaSource = "auto") -> Path:
    resolved = resolve_source(source)
    return _cache_directory() / f"schema-{resolved}.json"


def _valid_alias_table(value: Any, *, source: str | None = None) -> bool:
    if not isinstance(value, dict):
        return False
    if source is not None and value.get("source") != source:
        return False
    aliases = value.get("aliases")
    if not isinstance(aliases, dict) or not aliases:
        return False
    for item_type, mappings in aliases.items():
        if not isinstance(item_type, str) or not isinstance(mappings, dict):
            return False
        if any(
            not isinstance(base, str) or not isinstance(actual, str)
            for base, actual in mappings.items()
        ):
            return False
    return True


def _read_cache(source: Literal["local", "web"]) -> dict[str, Any] | None:
    try:
        value = json.loads(cache_path(source).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if _valid_alias_table(value, source=source) else None


def get_table(source: SchemaSource = "auto") -> dict[str, Any]:
    """Return the source-specific cached aliases over the vendored floor."""
    resolved = resolve_source(source)
    if resolved not in _table_cache:
        cached = _read_cache(resolved)
        aliases = {
            item_type: dict(mappings)
            for item_type, mappings in _ALIASES.items()
        }
        if cached is not None:
            for item_type, mappings in cached["aliases"].items():
                aliases.setdefault(item_type, {}).update(mappings)
        _table_cache[resolved] = {
            "source": resolved,
            "version": cached.get("version") if cached else None,
            "zotero_version": cached.get("zotero_version") if cached else None,
            "checked_at": cached.get("checked_at") if cached else None,
            "aliases": aliases,
            "cache_path": str(cache_path(resolved)),
        }
    return _table_cache[resolved]


def _build_table(
    schema_document: dict[str, Any],
    *,
    source: Literal["local", "web"],
    zotero_version: str | None,
    etag: str | None,
) -> dict[str, Any]:
    """Validate a Zotero API schema and retain only base-field aliases."""
    if not isinstance(schema_document, dict):
        raise ValueError("Zotero schema response must be a JSON object")
    version = schema_document.get("version")
    item_types = schema_document.get("itemTypes")
    if not isinstance(version, int) or version <= 0:
        raise ValueError("Zotero schema has no valid integer version")
    if not isinstance(item_types, list) or not item_types:
        raise ValueError("Zotero schema has no itemTypes array")

    aliases: dict[str, dict[str, str]] = {}
    for item in item_types:
        if not isinstance(item, dict):
            raise ValueError("Zotero schema contains a malformed item type")
        item_type = item.get("itemType")
        fields = item.get("fields", [])
        if not isinstance(item_type, str) or not item_type:
            raise ValueError("Zotero schema contains an unnamed item type")
        if item_type in aliases or not isinstance(fields, list):
            raise ValueError(f"Zotero schema item type {item_type!r} is malformed")
        mappings: dict[str, str] = {}
        for field in fields:
            if not isinstance(field, dict):
                raise ValueError(f"Zotero schema field in {item_type!r} is malformed")
            actual = field.get("field")
            base = field.get("baseField")
            if not isinstance(actual, str) or not actual:
                raise ValueError(f"Zotero schema field in {item_type!r} is unnamed")
            if base is not None and (not isinstance(base, str) or not base):
                raise ValueError(
                    f"Zotero schema base field for {item_type}.{actual} is malformed"
                )
            if base and base != actual:
                mappings[base] = actual
        aliases[item_type] = mappings

    # The runtime document must remain at least as complete as our known floor.
    missing = sorted(set(_ALIASES) - set(aliases))
    if missing:
        raise ValueError(
            "Zotero schema is missing known item type(s): " + ", ".join(missing)
        )
    return {
        "source": source,
        "version": version,
        "zotero_version": zotero_version,
        "etag": etag,
        "checked_at": time.time(),
        "aliases": aliases,
    }


def _schema_url(source: Literal["local", "web"]) -> str:
    if source == "web":
        return WEB_SCHEMA_URL
    from zotero_mcp.local_api import local_api_endpoint

    return f"{local_api_endpoint()}/schema"


def _http_get(url: str, headers: dict[str, str]):
    import requests

    from zotero_mcp.client import call_with_zotero_api_lock

    return call_with_zotero_api_lock(
        requests.get,
        url,
        headers=headers,
        timeout=(5.0, 30.0),
    )


def refresh(source: SchemaSource = "auto") -> dict[str, Any]:
    """Manually refresh the local- or web-specific metadata schema cache."""
    global _table_cache
    resolved = resolve_source(source)
    path = cache_path(resolved)
    with advisory_file_lock(path.with_suffix(".lock"), exclusive=True):
        existing = _read_cache(resolved)
        headers = {"Zotero-API-Version": "3", "Accept": "application/json"}
        if resolved == "web" and existing and existing.get("etag"):
            headers["If-None-Match"] = str(existing["etag"])
        response = _http_get(_schema_url(resolved), headers)
        if response.status_code == 304 and existing is not None:
            existing["checked_at"] = time.time()
            atomic_write_json(path, existing, indent=2, mode=0o600)
            _table_cache.pop(resolved, None)
            return {
                "status": "unchanged",
                "source": resolved,
                "schema_version": existing.get("version"),
                "zotero_version": existing.get("zotero_version"),
                "cache_path": str(path),
            }
        response.raise_for_status()
        table = _build_table(
            response.json(),
            source=resolved,
            zotero_version=response.headers.get("X-Zotero-Version"),
            etag=response.headers.get("ETag"),
        )
        status = (
            "unchanged"
            if existing
            and existing.get("version") == table["version"]
            and existing.get("aliases") == table["aliases"]
            else "refreshed"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, table, indent=2, mode=0o600)
        _table_cache.pop(resolved, None)
        return {
            "status": status,
            "source": resolved,
            "schema_version": table["version"],
            "zotero_version": table.get("zotero_version"),
            "cache_path": str(path),
        }


def resolve_field(item_type: str, field: str) -> str:
    """Map a generic Zotero base field to its type-specific key."""
    aliases = get_table()["aliases"]
    return aliases.get(item_type, {}).get(field, field)


def base_field_of(item_type: str, field: str) -> str:
    """Return the generic base field for a type-specific key."""
    aliases = get_table()["aliases"]
    for base, actual in aliases.get(item_type, {}).items():
        if actual == field:
            return base
    return field
