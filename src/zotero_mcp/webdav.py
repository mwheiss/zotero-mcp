"""Helpers for accessing Zotero WebDAV attachment storage."""

from __future__ import annotations

import hashlib
import io
import os
import stat
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import requests

_PLACEHOLDER_PREFIX = "REPLACE_WITH_YOUR_"
DEFAULT_MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_COMPRESSION_RATIO = 200.0
DEFAULT_MAX_ARCHIVE_MEMBERS = 1024
DEFAULT_DOWNLOAD_DEADLINE_SECONDS = 300.0
HARD_MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024
HARD_MAX_UNCOMPRESSED_BYTES = 4 * 1024 * 1024 * 1024
HARD_MAX_COMPRESSION_RATIO = 1000.0
HARD_MAX_ARCHIVE_MEMBERS = 4096
HARD_DOWNLOAD_DEADLINE_SECONDS = 3600.0


class WebDAVNotConfiguredError(RuntimeError):
    """Raised when WebDAV environment variables are missing."""


def _get_env_value(name: str) -> str | None:
    """Return a non-placeholder environment value, if present."""
    value = os.getenv(name, "").strip()
    if not value or value.startswith(_PLACEHOLDER_PREFIX):
        return None
    return value


def get_webdav_config() -> tuple[str, str, str] | None:
    """Return configured WebDAV credentials, or None when incomplete."""
    base_url = _get_env_value("ZOTERO_WEBDAV_URL")
    username = _get_env_value("ZOTERO_WEBDAV_USERNAME")
    password = _get_env_value("ZOTERO_WEBDAV_PASSWORD")

    if not all((base_url, username, password)):
        return None

    return (base_url.rstrip("/") + "/", username, password)


def is_webdav_configured() -> bool:
    """Return True when direct WebDAV access is configured."""
    return get_webdav_config() is not None


def _bounded_int_setting(
    name: str,
    default: int,
    hard_max: int,
    *,
    fallback_name: str | None = None,
) -> int:
    raw = os.getenv(name, "").strip()
    if not raw and fallback_name:
        raw = os.getenv(fallback_name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(1, min(value, hard_max))


def _bounded_float_setting(
    name: str,
    default: float,
    hard_max: float,
    *,
    fallback_name: str | None = None,
) -> float:
    raw = os.getenv(name, "").strip()
    if not raw and fallback_name:
        raw = os.getenv(fallback_name, "").strip()
    try:
        value = float(raw) if raw else default
    except ValueError:
        value = default
    if not value > 0:
        value = default
    return min(value, hard_max)


def max_download_bytes() -> int:
    return _bounded_int_setting(
        "ZOTERO_MCP_WEBDAV_MAX_DOWNLOAD_BYTES",
        DEFAULT_MAX_DOWNLOAD_BYTES,
        HARD_MAX_DOWNLOAD_BYTES,
        fallback_name="ZOTERO_MCP_REMOTE_DOWNLOAD_MAX_BYTES",
    )


def max_uncompressed_bytes() -> int:
    return _bounded_int_setting(
        "ZOTERO_MCP_WEBDAV_MAX_UNCOMPRESSED_BYTES",
        DEFAULT_MAX_UNCOMPRESSED_BYTES,
        HARD_MAX_UNCOMPRESSED_BYTES,
    )


def max_compression_ratio() -> float:
    return _bounded_float_setting(
        "ZOTERO_MCP_WEBDAV_MAX_COMPRESSION_RATIO",
        DEFAULT_MAX_COMPRESSION_RATIO,
        HARD_MAX_COMPRESSION_RATIO,
    )


def max_archive_members() -> int:
    return _bounded_int_setting(
        "ZOTERO_MCP_WEBDAV_MAX_ARCHIVE_MEMBERS",
        DEFAULT_MAX_ARCHIVE_MEMBERS,
        HARD_MAX_ARCHIVE_MEMBERS,
    )


def download_deadline_seconds() -> float:
    return _bounded_float_setting(
        "ZOTERO_MCP_WEBDAV_DOWNLOAD_DEADLINE_SECONDS",
        DEFAULT_DOWNLOAD_DEADLINE_SECONDS,
        HARD_DOWNLOAD_DEADLINE_SECONDS,
        fallback_name="ZOTERO_MCP_REMOTE_DOWNLOAD_DEADLINE_SECONDS",
    )


def _remaining_timeout(deadline: float) -> tuple[float, float]:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("WebDAV download exceeded its total time limit")
    return min(10.0, remaining), min(30.0, remaining)


def _select_primary_member(
    members: list[zipfile.ZipInfo], expected_filename: str | None
) -> zipfile.ZipInfo:
    """Pick the most likely primary file from a Zotero WebDAV archive."""
    expected_basename = Path(expected_filename).name.lower() if expected_filename else ""
    expected_suffix = Path(expected_filename).suffix.lower() if expected_filename else ""

    def _score(info: zipfile.ZipInfo) -> tuple[bool, bool, int, int]:
        member_path = PurePosixPath(info.filename)
        basename = member_path.name.lower()
        suffix = member_path.suffix.lower()
        return (
            basename != expected_basename,
            bool(expected_suffix) and suffix != expected_suffix,
            len(member_path.parts),
            len(info.filename),
        )

    return min(members, key=_score)


def _extract_archive(
    archive_source: bytes | str | Path,
    destination_dir: str | Path,
    expected_filename: str | None,
    *,
    deadline: float | None = None,
) -> Path:
    """Extract a WebDAV attachment zip and return the primary file path."""
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)

    archive_file: io.BytesIO | str | Path
    if isinstance(archive_source, bytes):
        archive_file = io.BytesIO(archive_source)
    else:
        archive_file = archive_source

    with zipfile.ZipFile(archive_file) as zf:
        archive_members = zf.infolist()
        if len(archive_members) > max_archive_members():
            raise ValueError(
                "WebDAV archive exceeds the configured member-count limit"
            )
        members = [info for info in archive_members if not info.is_dir()]
        if not members:
            raise ValueError("WebDAV archive contained no files")

        total_declared = sum(info.file_size for info in members)
        if total_declared > max_uncompressed_bytes():
            raise ValueError(
                "WebDAV archive exceeds the configured uncompressed-size limit"
            )
        ratio_limit = max_compression_ratio()
        validated_members: list[tuple[zipfile.ZipInfo, Path]] = []
        destination_names: set[str] = set()
        for info in members:
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(
                    "WebDAV extraction exceeded its total time limit"
                )
            ratio = info.file_size / max(1, info.compress_size)
            if ratio > ratio_limit:
                raise ValueError(
                    "WebDAV archive member exceeds the configured compression-ratio limit"
                )
            unix_mode = info.external_attr >> 16
            if stat.S_ISLNK(unix_mode):
                raise ValueError(
                    f"Unsafe symbolic link in WebDAV archive: {info.filename}"
                )
            normalized_name = info.filename.replace("\\", "/")
            relative = Path(PurePosixPath(normalized_name))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe path in WebDAV archive: {info.filename}")
            normalized_destination = relative.as_posix()
            if normalized_destination in destination_names:
                raise ValueError(
                    f"Duplicate path in WebDAV archive: {info.filename}"
                )
            destination_names.add(normalized_destination)
            validated_members.append((info, relative))

        destination_parts = {tuple(path.parts) for _, path in validated_members}
        for parts in destination_parts:
            if any(parts[:index] in destination_parts for index in range(1, len(parts))):
                raise ValueError(
                    "WebDAV archive contains a file/directory path collision"
                )

        extracted_paths: dict[str, Path] = {}
        total_written = 0
        output_path: Path | None = None
        try:
            for info, relative in validated_members:
                output_path = destination / relative
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as source, open(output_path, "wb") as target:
                    while chunk := source.read(64 * 1024):
                        if deadline is not None and time.monotonic() > deadline:
                            raise TimeoutError(
                                "WebDAV extraction exceeded its total time limit"
                            )
                        total_written += len(chunk)
                        if total_written > max_uncompressed_bytes():
                            raise ValueError(
                                "WebDAV archive exceeded the configured "
                                "uncompressed-size limit while extracting"
                            )
                        target.write(chunk)
                extracted_paths[info.filename] = output_path
        except Exception:
            cleanup_paths = [*extracted_paths.values()]
            if output_path is not None:
                cleanup_paths.append(output_path)
            for cleanup_path in cleanup_paths:
                try:
                    if cleanup_path.is_file() or cleanup_path.is_symlink():
                        cleanup_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

        selected = _select_primary_member(members, expected_filename)
        return extracted_paths[selected.filename]


def _compute_file_md5(file_path: str | Path, chunk_size: int = 65536) -> str:
    """Return hex md5 of a file's contents, streamed in chunks."""
    digest = hashlib.md5()  # noqa: S324 — Zotero's WebDAV protocol mandates md5
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_zotero_zip(file_path: str | Path) -> bytes:
    """Return the bytes of a Zotero-formatted attachment zip.

    Zotero's WebDAV layout stores the attachment file inside a zip named
    <KEY>.zip, with a single member whose name is the original basename.
    """
    src = Path(file_path)
    buf = io.BytesIO()
    # ZIP_DEFLATED keeps the upload small for text-heavy PDFs; pyzotero
    # uses the same compression on its own ingest path.
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(src, arcname=src.name)
    return buf.getvalue()


def _build_prop_xml(md5_hex: str, mtime_ms: int) -> bytes:
    """Return the bytes of a Zotero-format <KEY>.prop sidecar file.

    Zotero stores a tiny XML doc next to each <KEY>.zip with mtime and
    md5 so other clients can detect changes without downloading the zip.
    """
    return (
        '<properties version="1">'
        f"<mtime>{int(mtime_ms)}</mtime>"
        f"<hash>{md5_hex}</hash>"
        "</properties>"
    ).encode()


def upload_attachment_to_webdav(
    attachment_key: str,
    file_path: str | Path,
    md5: str | None = None,
    mtime_ms: int | None = None,
    timeout: float | None = None,
) -> tuple[str, int]:
    """Upload an attachment to WebDAV in Zotero's expected layout.

    Writes two objects to the configured WebDAV server:
      - <KEY>.zip  (containing the original file)
      - <KEY>.prop (XML sidecar with mtime and md5)

    Returns ``(md5_hex, mtime_ms)`` so the caller can write the same
    values to the Zotero attachment item via the web API, keeping the
    two sides of the sync consistent.

    ``timeout`` is the read-side timeout passed to ``requests``. When
    ``None`` (the default), the value is read from the
    ``ZOTERO_WEBDAV_TIMEOUT`` environment variable; if that is unset,
    falls back to ``60.0``. An explicit float bypasses the env lookup.

    Raises ``WebDAVNotConfiguredError`` when the env vars are missing,
    or ``requests.HTTPError`` on a non-2xx response from the PUTs.
    """
    if timeout is None:
        env_timeout = _get_env_value("ZOTERO_WEBDAV_TIMEOUT")
        timeout = float(env_timeout) if env_timeout is not None else 60.0

    config = get_webdav_config()
    if not config:
        raise WebDAVNotConfiguredError(
            "Missing ZOTERO_WEBDAV_URL / ZOTERO_WEBDAV_USERNAME / ZOTERO_WEBDAV_PASSWORD"
        )

    base_url, username, password = config
    src = Path(file_path)
    if not src.is_file():
        raise FileNotFoundError(f"Attachment source not found: {src}")

    md5_hex = md5 if md5 else _compute_file_md5(src)
    if mtime_ms is None:
        mtime_ms = int(src.stat().st_mtime * 1000)

    zip_bytes = _build_zotero_zip(src)
    prop_bytes = _build_prop_xml(md5_hex, mtime_ms)

    zip_url = f"{base_url}{quote(attachment_key, safe='')}.zip"
    prop_url = f"{base_url}{quote(attachment_key, safe='')}.prop"

    session = requests.Session()
    session.auth = (username, password)
    session.trust_env = True
    try:
        # Order matters: Zotero treats a fresh .prop with no matching .zip
        # as evidence of corruption. PUT the zip first.
        zip_resp = session.put(
            zip_url,
            data=zip_bytes,
            headers={"Content-Type": "application/zip"},
            timeout=(10.0, timeout),
        )
        zip_resp.raise_for_status()
        prop_resp = session.put(
            prop_url,
            data=prop_bytes,
            headers={"Content-Type": "text/xml; charset=utf-8"},
            timeout=(10.0, timeout),
        )
        prop_resp.raise_for_status()
    finally:
        close = getattr(session, "close", None)
        if callable(close):
            close()

    return md5_hex, mtime_ms


def download_attachment_from_webdav(
    attachment_key: str,
    destination_dir: str | Path,
    expected_filename: str | None = None,
    timeout: float = 30.0,
) -> Path:
    """Download a WebDAV-backed Zotero attachment and return the primary file path."""
    config = get_webdav_config()
    if not config:
        raise WebDAVNotConfiguredError(
            "Missing ZOTERO_WEBDAV_URL / ZOTERO_WEBDAV_USERNAME / ZOTERO_WEBDAV_PASSWORD"
        )

    base_url, username, password = config
    url = f"{base_url}{quote(attachment_key, safe='')}.zip"
    deadline = time.monotonic() + download_deadline_seconds()

    session = requests.Session()
    session.auth = (username, password)
    session.trust_env = True

    temp_zip_path = None
    try:
        connect_timeout, read_timeout = _remaining_timeout(deadline)
        response = session.get(
            url,
            timeout=(
                min(connect_timeout, 10.0),
                min(read_timeout, timeout),
            ),
            stream=True,
        )
        if response.status_code == 404:
            raise FileNotFoundError(f"Attachment {attachment_key} was not found in WebDAV storage")
        response.raise_for_status()
        content_length = getattr(response, "headers", {}).get("Content-Length")
        if content_length:
            try:
                declared_size = int(content_length)
            except (TypeError, ValueError):
                declared_size = 0
            if declared_size > max_download_bytes():
                raise ValueError(
                    "WebDAV archive exceeds the configured download-size limit"
                )
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_zip:
            temp_zip_path = temp_zip.name
            received = 0
            for chunk in response.iter_content(chunk_size=1024 * 64):
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        "WebDAV download exceeded its total time limit"
                    )
                if chunk:
                    received += len(chunk)
                    if received > max_download_bytes():
                        raise ValueError(
                            "WebDAV archive exceeded the configured download-size limit"
                        )
                    temp_zip.write(chunk)
        return _extract_archive(
            temp_zip_path,
            destination_dir,
            expected_filename,
            deadline=deadline,
        )
    finally:
        if temp_zip_path and os.path.exists(temp_zip_path):
            os.unlink(temp_zip_path)
        close = getattr(session, "close", None)
        if callable(close):
            close()
