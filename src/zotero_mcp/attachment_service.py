"""Attachment descriptors, signed transfer URLs, staging, and confirmations."""

from __future__ import annotations

import base64
import bisect
import hashlib
import hmac
import json
import math
import mimetypes
import os
import re
import secrets
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

from zotero_mcp._atomic_io import atomic_write_json, atomic_write_text, durable_unlink
from zotero_mcp._file_lock import (
    acquire_file_lock,
    advisory_file_lock,
    release_file_lock,
)

DEFAULT_MAX_UPLOAD_SIZE = 512 * 1024 * 1024
DEFAULT_MAX_INLINE_SIZE = 1 * 1024 * 1024
DEFAULT_MAX_RESOURCE_SIZE = 8 * 1024 * 1024
HARD_MAX_INLINE_SIZE = 32 * 1024 * 1024
HARD_MAX_RESOURCE_SIZE = 64 * 1024 * 1024
DEFAULT_STATE_LOCK_TIMEOUT = 45.0
DEFAULT_TOKEN_TTL = 3600
DEFAULT_UPLOAD_GC_SCAN_LIMIT = 128
DEFAULT_UPLOAD_GC_DELETE_LIMIT = 32
HARD_UPLOAD_GC_SCAN_LIMIT = 4096
HARD_UPLOAD_GC_DELETE_LIMIT = 512
MIN_SECRET_BYTES = 32


def _state_root() -> Path:
    configured = os.getenv("ZOTERO_MCP_ATTACHMENT_STATE_DIR", "").strip()
    return (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".cache" / "zotero-mcp" / "attachments"
    )


def _private_file(path: Path) -> None:
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _secret() -> bytes:
    configured = os.getenv("ZOTERO_MCP_ATTACHMENT_SECRET", "").strip()
    if configured:
        value = configured.encode("utf-8")
        if len(value) < MIN_SECRET_BYTES:
            raise ValueError(
                f"ZOTERO_MCP_ATTACHMENT_SECRET must be at least {MIN_SECRET_BYTES} bytes"
            )
        return value
    path = _state_root() / "secret"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Serializing initialization prevents a losing process from reading a file
    # before the winning process has finished writing it. A short/corrupt
    # persisted secret is replaced atomically rather than becoming a known HMAC
    # key (including the empty key).
    with advisory_file_lock(path.with_suffix(".lock"), exclusive=True):
        try:
            value = path.read_bytes()
        except OSError:
            value = b""
        if len(value) >= MIN_SECRET_BYTES:
            _private_file(path)
            return value
        generated = secrets.token_bytes(MIN_SECRET_BYTES).hex()
        atomic_write_text(path, generated, mode=0o600)
        _private_file(path)
        value = path.read_bytes()
        if len(value) < MIN_SECRET_BYTES:  # defensive fail-closed check
            raise RuntimeError("Attachment capability secret initialization failed")
        return value


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def make_token(action: str, claims: dict[str, Any], ttl: int = DEFAULT_TOKEN_TTL) -> str:
    payload = {
        "v": 1,
        "action": action,
        "exp": int(time.time()) + max(1, int(ttl)),
        "jti": secrets.token_urlsafe(12),
        **claims,
    }
    encoded = _b64encode(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    signature = _b64encode(hmac.new(_secret(), encoded.encode("ascii"), hashlib.sha256).digest())
    return f"v1.{encoded}.{signature}"


def verify_token(token: str, action: str) -> dict[str, Any]:
    try:
        version, encoded, supplied = token.split(".", 2)
        expected = _b64encode(
            hmac.new(_secret(), encoded.encode("ascii"), hashlib.sha256).digest()
        )
        if version != "v1" or not hmac.compare_digest(supplied, expected):
            raise ValueError
        payload = json.loads(_b64decode(encoded))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid capability token") from exc
    if payload.get("action") != action:
        raise ValueError("Capability token has the wrong action")
    if int(payload.get("exp", 0)) < int(time.time()):
        raise ValueError("Capability token has expired")
    return payload


def public_base_url() -> str | None:
    value = os.getenv("ZOTERO_MCP_PUBLIC_BASE_URL", "").strip().rstrip("/")
    return value or None


def public_transfer_url(kind: str, token: str) -> str | None:
    base = public_base_url()
    if not base:
        return None
    return f"{base}/attachments/{kind}/{quote(token, safe='')}"


def attachment_descriptor(item: dict[str, Any]) -> dict[str, Any]:
    data = item.get("data", {}) or {}
    key = item.get("key") or data.get("key", "")
    link_mode = data.get("linkMode", "")
    content_type = data.get("contentType", "") or "application/octet-stream"
    filename = data.get("filename", "")
    return {
        "attachment_key": key,
        "parent_item_key": data.get("parentItem", ""),
        "title": data.get("title", ""),
        "filename": filename,
        "content_type": content_type,
        "size": item.get("meta", {}).get("fileSize") or data.get("filesize"),
        "md5": data.get("md5", ""),
        "version": item.get("version") or data.get("version"),
        "date_modified": data.get("dateModified", ""),
        "link_mode": link_mode,
        "url": data.get("url", ""),
        "binary_readable": link_mode != "linked_url" and bool(filename or content_type),
        "binary_writable": link_mode in {"imported_file", "imported_url", ""},
        "is_trashed": bool(data.get("deleted")),
    }


def attachment_artifact_type(item: dict[str, Any]) -> str | None:
    """Classify one attachment using the canonical document-source contract."""
    data = item.get("data", {}) or {}
    if data.get("itemType") != "attachment":
        return None
    title = str(data.get("title", ""))
    filename = str(data.get("filename", ""))
    content_type = str(data.get("contentType", "")).lower()
    label = re.sub(r"[^a-z0-9]+", " ", f"{title} {filename}".casefold()).strip()
    words = set(label.split())
    is_pdf = content_type == "application/pdf" or filename.lower().endswith(".pdf")
    is_json = content_type == "application/json" or filename.lower().endswith(".json")
    is_html = content_type.startswith("text/html") or filename.lower().endswith((".html", ".htm"))
    if "betterissa" in words and (
        words & {"state", "summary", "references", "sections"}
        or {"extraction", "metadata"} <= words
    ):
        return None
    if not (is_pdf or is_json or is_html) and "betterissa" in words and "indexing" in words:
        return "betterissa-indexing"
    if is_json and {"betterissa", "semantic", "document"} <= words:
        return "betterissa-semantic"
    if not (is_pdf or is_json or is_html) and {"betterissa", "advanced", "ocr"} <= words:
        return "betterissa-ocr"
    if is_html and {"betterissa", "reading", "view"} <= words:
        return "betterissa-reading-view"
    fulltext = "fulltext" in words or {"full", "text"} <= words
    if not is_pdf and fulltext:
        return "fulltext"
    if is_pdf and "ocr" in words:
        return "ocr-pdf"
    if is_pdf and fulltext:
        return "pdf"
    if is_pdf:
        return "pdf"
    if "betterissa" in words:
        return None
    textual_suffixes = {
        ".txt",
        ".md",
        ".markdown",
        ".rst",
        ".csv",
        ".tsv",
        ".json",
        ".xml",
        ".vtt",
        ".srt",
        ".sbv",
        ".log",
        ".html",
        ".htm",
    }
    if content_type.startswith("text/") or filename.lower().endswith(
        tuple(textual_suffixes)
    ):
        return "file"
    return None


_ARTIFACT_PRIORITY = {
    "betterissa-indexing": 0,
    "betterissa-semantic": 1,
    "betterissa-ocr": 2,
    "betterissa-reading-view": 3,
    "fulltext": 4,
    "ocr-pdf": 5,
    "pdf": 7,
    "file": 8,
}


def select_best_attachment(items: list[dict[str, Any]]) -> tuple[dict[str, Any], str] | None:
    candidates = []
    for item in items:
        kind = attachment_artifact_type(item)
        if kind is None:
            continue
        data = item.get("data", {}) or {}
        candidates.append(
            (
                _ARTIFACT_PRIORITY[kind],
                str(data.get("dateModified", "")),
                str(item.get("key") or data.get("key", "")),
                item,
                kind,
            )
        )
    if not candidates:
        return None
    priority = min(value[0] for value in candidates)
    same_priority = [value for value in candidates if value[0] == priority]
    selected = max(same_priority, key=lambda value: (value[1], value[2]))
    return selected[3], selected[4]


def document_text_from_file(path: Path, artifact_type: str) -> str:
    """Convert a downloaded canonical artifact to its model-facing text."""
    if artifact_type == "betterissa-semantic":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return ""
        if not isinstance(payload, dict) or payload.get("ok") is False:
            return ""
        if payload.get("status") not in (None, "complete"):
            return ""
        structured = payload.get("structured")
        if isinstance(structured, dict) and isinstance(structured.get("plain_text"), str):
            return structured["plain_text"].strip()
        return ""
    if artifact_type == "betterissa-ocr":
        text = path.read_text(encoding="utf-8", errors="replace")
        return re.sub(
            r"\s*<!--\s*BetterIssa\s+page\s+\d+\s*-->\s*",
            "\n\n",
            text,
            flags=re.IGNORECASE,
        ).strip()
    if artifact_type in {"betterissa-indexing", "fulltext", "file"}:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
    from markitdown import MarkItDown

    return MarkItDown().convert(str(path)).text_content or ""


def safe_filename(value: str) -> str:
    name = Path(value or "attachment.bin").name.replace("\x00", "")
    return name or "attachment.bin"


def max_upload_size() -> int:
    raw = os.getenv("ZOTERO_MCP_ATTACHMENT_MAX_BYTES", "").strip()
    try:
        value = int(raw) if raw else DEFAULT_MAX_UPLOAD_SIZE
    except ValueError:
        value = DEFAULT_MAX_UPLOAD_SIZE
    return max(1, value)


def _bounded_size_setting(name: str, default: int, hard_max: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(1, min(value, hard_max))


def max_inline_size() -> int:
    """Maximum binary size encoded directly into a tool response."""
    return _bounded_size_setting(
        "ZOTERO_MCP_ATTACHMENT_INLINE_MAX_BYTES",
        DEFAULT_MAX_INLINE_SIZE,
        HARD_MAX_INLINE_SIZE,
    )


def max_resource_size() -> int:
    """Maximum binary size returned through an in-memory MCP resource."""
    return _bounded_size_setting(
        "ZOTERO_MCP_ATTACHMENT_RESOURCE_MAX_BYTES",
        DEFAULT_MAX_RESOURCE_SIZE,
        HARD_MAX_RESOURCE_SIZE,
    )


def prepare_upload(
    filename: str,
    content_type: str,
    size: int,
    sha256: str,
    *,
    library: dict[str, str],
) -> dict[str, Any]:
    garbage_collect_uploads()
    if size < 0 or size > max_upload_size():
        raise ValueError(f"Upload size must be between 0 and {max_upload_size()} bytes")
    digest = sha256.strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("sha256 must be a 64-character hexadecimal digest")
    upload_id = secrets.token_urlsafe(18)
    directory = _state_root() / "uploads" / upload_id
    directory.mkdir(parents=True, mode=0o700)
    manifest = {
        "upload_id": upload_id,
        "filename": safe_filename(filename),
        "content_type": content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream",
        "size": int(size),
        "sha256": digest,
        "library": library,
        "status": "prepared",
        "created_at": int(time.time()),
        "expires_at": int(time.time()) + DEFAULT_TOKEN_TTL,
    }
    manifest_path = directory / "manifest.json"
    atomic_write_json(manifest_path, manifest, indent=2)
    _private_file(manifest_path)
    token = make_token(
        "upload",
        {
            "upload_id": upload_id,
            "size": size,
            "sha256": digest,
        },
    )
    return {
        **manifest,
        "upload_url": public_transfer_url("upload", token),
        "upload_token": token if public_base_url() is None else None,
        "method": "PUT",
    }


def upload_directory(upload_id: str) -> Path:
    if not upload_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in upload_id):
        raise ValueError("Invalid upload ID")
    return _state_root() / "uploads" / upload_id


def upload_lock_path(upload_id: str) -> Path:
    upload_directory(upload_id)  # validate the identifier
    return _state_root() / "upload-locks" / f"{upload_id}.lock"


@contextmanager
def upload_lock(upload_id: str, *, blocking: bool = True):
    """Serialize upload, consumption, and garbage collection for one upload."""
    path = upload_lock_path(upload_id)
    with advisory_file_lock(path, exclusive=True, blocking=blocking) as lock_file:
        yield lock_file


def read_upload(
    upload_id: str,
    *,
    require_ready: bool = True,
    allow_consumed: bool = False,
) -> tuple[dict[str, Any], Path]:
    directory = upload_directory(upload_id)
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Unknown or expired attachment upload") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Attachment upload manifest is invalid")
    try:
        expires_at = int(manifest.get("expires_at", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("Attachment upload manifest has an invalid expiry") from exc
    if expires_at < int(time.time()):
        raise ValueError("Attachment upload has expired")
    status = manifest.get("status")
    if require_ready and status != "ready" and not (
        allow_consumed and status == "consumed"
    ):
        raise ValueError("Attachment upload has not completed")
    return manifest, directory / safe_filename(manifest.get("filename", ""))


def mark_upload_ready(upload_id: str, *, size: int, sha256: str) -> dict[str, Any]:
    manifest, _ = read_upload(upload_id, require_ready=False)
    if size != manifest["size"]:
        raise ValueError(f"Upload size mismatch: expected {manifest['size']}, received {size}")
    if not hmac.compare_digest(sha256.lower(), manifest["sha256"]):
        raise ValueError("Upload SHA-256 mismatch")
    manifest["status"] = "ready"
    manifest["received_at"] = int(time.time())
    path = upload_directory(upload_id) / "manifest.json"
    atomic_write_json(path, manifest, indent=2)
    _private_file(path)
    return manifest


def consume_upload(upload_id: str, *, already_locked: bool = False) -> dict[str, Any]:
    """Remove staged bytes after a successful Zotero mutation.

    A small consumed manifest is retained until its normal expiry so an
    idempotent create retry can still resolve to the previously created item.
    Failed Zotero operations never call this function, leaving their bytes
    available for retry.
    """

    def _consume() -> dict[str, Any]:
        manifest, payload = read_upload(
            upload_id,
            require_ready=True,
            allow_consumed=True,
        )
        if manifest.get("status") == "consumed":
            return manifest
        manifest["status"] = "consumed"
        manifest["consumed_at"] = int(time.time())
        path = upload_directory(upload_id) / "manifest.json"
        atomic_write_json(path, manifest, indent=2)
        _private_file(path)
        # Persist the tombstone before unlinking bytes. A crash can then leave
        # an expired file for GC, but can never leave a ready manifest pointing
        # at bytes that were already removed.
        payload.unlink(missing_ok=True)
        payload.with_suffix(payload.suffix + ".partial").unlink(missing_ok=True)
        return manifest

    if already_locked:
        return _consume()
    with upload_lock(upload_id):
        return _consume()


def _positive_bounded_int(name: str, default: int, hard_max: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(1, min(value, hard_max))


def garbage_collect_uploads(*, now: int | None = None) -> dict[str, int]:
    """Delete expired uploads with fair, bounded manifest/deletion work."""
    root = _state_root() / "uploads"
    scan_limit = _positive_bounded_int(
        "ZOTERO_MCP_ATTACHMENT_GC_SCAN_LIMIT",
        DEFAULT_UPLOAD_GC_SCAN_LIMIT,
        HARD_UPLOAD_GC_SCAN_LIMIT,
    )
    delete_limit = _positive_bounded_int(
        "ZOTERO_MCP_ATTACHMENT_GC_DELETE_LIMIT",
        DEFAULT_UPLOAD_GC_DELETE_LIMIT,
        HARD_UPLOAD_GC_DELETE_LIMIT,
    )
    current = int(time.time()) if now is None else int(now)
    scanned = deleted = 0
    try:
        with os.scandir(root) as entries:
            names = sorted(
                entry.name
                for entry in entries
                if entry.is_dir(follow_symlinks=False)
            )
    except OSError:
        return {"scanned": 0, "deleted": 0}
    if not names:
        return {"scanned": 0, "deleted": 0}

    cursor_path = _state_root() / "upload-gc-cursor"
    try:
        previous = cursor_path.read_text(encoding="utf-8").strip()
    except OSError:
        previous = ""
    start = bisect.bisect_right(names, previous) if previous else 0
    ordered = names[start:] + names[:start]
    candidates = ordered[:scan_limit]

    last_scanned = ""
    for upload_id in candidates:
        if deleted >= delete_limit:
            break
        scanned += 1
        last_scanned = upload_id
        remove_lock = False
        try:
            with upload_lock(upload_id, blocking=False) as lock_file:
                if lock_file is None:
                    continue
                directory = upload_directory(upload_id)
                try:
                    manifest = json.loads(
                        (directory / "manifest.json").read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    # Do not infer expiry for malformed state; an operator can
                    # inspect it without a background call destroying evidence.
                    continue
                if not isinstance(manifest, dict):
                    continue
                try:
                    expires_at = int(manifest.get("expires_at", 0))
                except (TypeError, ValueError):
                    continue
                if expires_at >= current:
                    continue
                shutil.rmtree(directory, ignore_errors=False)
                deleted += 1
                remove_lock = True
            if remove_lock:
                upload_lock_path(upload_id).unlink(missing_ok=True)
        except (OSError, ValueError):
            continue

    # Rotate the next bounded scan past the last candidate, so a stable set of
    # live directories cannot permanently hide expired entries later in the
    # directory. The cursor contains only a random upload identifier.
    if last_scanned:
        atomic_write_text(cursor_path, last_scanned, mode=0o600)
    return {"scanned": scanned, "deleted": deleted}


def operation_path(operation_id: str) -> Path:
    if not operation_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in operation_id):
        raise ValueError("Invalid attachment operation ID")
    return _state_root() / "operations" / f"{operation_id}.json"


def idempotency_path(key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return _state_root() / "idempotency" / f"{digest}.json"


def idempotency_record(key: str) -> dict[str, Any] | None:
    path = idempotency_path(key)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


@contextmanager
def idempotency_lock(key: str):
    """Serialize one idempotency key across threads and server processes."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    path = _state_root() / "idempotency" / f"{digest}.lock"
    raw_timeout = os.getenv("ZOTERO_MCP_ATTACHMENT_LOCK_TIMEOUT", "").strip()
    try:
        timeout = float(raw_timeout) if raw_timeout else DEFAULT_STATE_LOCK_TIMEOUT
    except ValueError:
        timeout = DEFAULT_STATE_LOCK_TIMEOUT
    if not math.isfinite(timeout) or timeout <= 0:
        timeout = DEFAULT_STATE_LOCK_TIMEOUT
    deadline = time.monotonic() + timeout
    lock_file = None
    while lock_file is None:
        lock_file = acquire_file_lock(path, exclusive=True, blocking=False)
        if lock_file is not None:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Attachment idempotency key remained busy for {timeout:.0f}s"
            )
        time.sleep(0.05)
    try:
        yield
    finally:
        release_file_lock(lock_file)


def save_idempotency_record(key: str, value: dict[str, Any]) -> None:
    path = idempotency_path(key)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    atomic_write_json(path, value, indent=2)
    _private_file(path)


def clear_idempotency_record(key: str) -> None:
    """Remove a definitely failed pending record so the request can retry."""
    durable_unlink(idempotency_path(key))


def prepare_operation(action: str, details: dict[str, Any], ttl: int = 900) -> dict[str, Any]:
    operation_id = secrets.token_urlsafe(18)
    expires_at = int(time.time()) + ttl
    digest = hashlib.sha256(
        json.dumps(details, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    record = {
        "operation_id": operation_id,
        "action": action,
        "details": details,
        "digest": digest,
        "expires_at": expires_at,
        "used": False,
    }
    path = operation_path(operation_id)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    atomic_write_json(path, record, indent=2)
    _private_file(path)
    confirmation_token = make_token(
        "confirm",
        {"operation_id": operation_id, "digest": digest, "mutation": action},
        ttl=ttl,
    )
    return {**record, "confirmation_token": confirmation_token}


def consume_operation(
    operation_id: str,
    confirmation_token: str,
    *,
    action: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    path = operation_path(operation_id)
    with advisory_file_lock(path.with_suffix(".lock"), exclusive=True):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Unknown or expired operation") from exc
        token = verify_token(confirmation_token, "confirm")
        digest = hashlib.sha256(
            json.dumps(details, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if record.get("used"):
            raise ValueError("Operation has already been used")
        if int(record.get("expires_at", 0)) < int(time.time()):
            raise ValueError("Operation has expired")
        if record.get("action") != action or record.get("digest") != digest:
            raise ValueError("Operation does not match the requested mutation")
        if token.get("operation_id") != operation_id or token.get("digest") != digest:
            raise ValueError("Confirmation token does not match the operation")
        record["used"] = True
        atomic_write_json(path, record, indent=2)
        _private_file(path)
        return record
