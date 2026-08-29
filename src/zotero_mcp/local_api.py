"""Authenticated Zotero Local API transport and file uploads."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import threading
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import httpx
from pyzotero import zotero
from pyzotero.zotero import build_url

from ._atomic_io import atomic_write_json

LOCAL_API_VERSION = "3"
LOCAL_API_APP_NAME = "Zotero MCP"
_WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


class LocalApiWriteUnavailableError(RuntimeError):
    """The running Zotero does not expose authenticated local writes."""


class LocalApiAuthorizationError(RuntimeError):
    """The user denied or Zotero rejected local write authorization."""


class LocalApiServerChangedError(RuntimeError):
    """The request reached a different Zotero database instance."""


def local_api_endpoint() -> str:
    """Return the configured local API endpoint without a trailing slash."""
    port = os.getenv("ZOTERO_LOCAL_PORT", "23119").strip() or "23119"
    return f"http://127.0.0.1:{port}/api"


def remote_local_api_endpoint() -> str | None:
    """Return a validated remote-local endpoint, when configured.

    Local API keys authorize broad writes. HTTPS and loopback tunnels are
    preferable, but an explicitly configured LAN HTTP endpoint is accepted.
    """
    configured = os.getenv("ZOTERO_REMOTE_LOCAL_URL", "").strip()
    if not configured:
        return None
    parsed = urlparse(configured)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "ZOTERO_REMOTE_LOCAL_URL must not contain credentials, a query, or a fragment"
        )
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(
            "ZOTERO_REMOTE_LOCAL_URL must be an absolute HTTP(S) URL"
        )
    path = parsed.path.rstrip("/")
    if not path:
        path = "/api"
    elif not path.endswith("/api"):
        raise ValueError("ZOTERO_REMOTE_LOCAL_URL must end in /api")
    return parsed._replace(path=path, params="", query="", fragment="").geturl()


def remote_local_host_header() -> str | None:
    """Return the Host header expected by the proxied Zotero HTTP server."""
    configured = os.getenv("ZOTERO_REMOTE_LOCAL_HOST_HEADER")
    if configured is None or not configured.strip():
        return "127.0.0.1:23119"
    value = configured.strip()
    if value.lower() in {"none", "preserve"}:
        return None
    if any(character.isspace() or ord(character) < 32 for character in value):
        raise ValueError(
            "ZOTERO_REMOTE_LOCAL_HOST_HEADER must be a single HTTP Host value"
        )
    return value


def writable_local_api_endpoints() -> list[tuple[str, str]]:
    """Return remote-first writable endpoints with stable role labels."""
    local_endpoint = local_api_endpoint()
    remote_endpoint = remote_local_api_endpoint()
    endpoints = []
    if remote_endpoint and remote_endpoint != local_endpoint:
        endpoints.append(("remote-local", remote_endpoint))
    endpoints.append(("server-local", local_endpoint))
    return endpoints


def local_auth_path() -> Path:
    """Return the private cache path for remembered local authorizations."""
    configured = os.getenv("ZOTERO_MCP_LOCAL_AUTH_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".config" / "zotero-mcp" / "local-api-auth.json"


def import_local_authorization_store(
    source_path: str | Path,
    *,
    destination_path: str | Path | None = None,
) -> list[str]:
    """Import remembered keys from BetterIssa or Zotero MCP without exposing them."""
    source = Path(source_path).expanduser()
    source_mode = source.stat().st_mode & 0o777
    if source_mode & 0o077:
        raise ValueError(
            f"Authorization source must not be group/world accessible (mode {source_mode:o})"
        )
    with source.open(encoding="utf-8") as source_file:
        payload = json.load(source_file)
    if not isinstance(payload, dict):
        raise ValueError("Authorization source must contain a JSON object")
    source_servers = payload.get("servers") if payload.get("version") else payload
    if not isinstance(source_servers, dict):
        raise ValueError("Authorization source has no server-key mapping")

    imported: dict[str, dict[str, str]] = {}
    for server_id, entry in source_servers.items():
        if not isinstance(server_id, str) or not server_id:
            continue
        if not isinstance(entry, dict) or entry.get("remember") is False:
            continue
        key = entry.get("key")
        if not isinstance(key, str) or len(key) != 32:
            continue
        imported[server_id] = {"key": key}
    if not imported:
        raise ValueError("Authorization source contains no remembered 32-character keys")

    destination = (
        Path(destination_path).expanduser()
        if destination_path is not None
        else local_auth_path()
    )
    try:
        with destination.open(encoding="utf-8") as destination_file:
            current = json.load(destination_file)
        if not isinstance(current, dict):
            current = {}
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        current = {}
    current["version"] = 1
    servers = current.setdefault("servers", {})
    if not isinstance(servers, dict):
        servers = {}
        current["servers"] = servers
    servers.update(imported)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination, current, indent=2)
    destination.chmod(0o600)
    return sorted(imported)


class LocalApiHttpClient(httpx.Client):
    """HTTP/1.1 client that adds Zotero's local write authorization contract."""

    def __init__(
        self,
        *,
        authorize_writes: bool = False,
        endpoint: str | None = None,
        host_header: str | None = None,
        api_key: str | None = None,
        auth_path: Path | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.local_endpoint = (endpoint or local_api_endpoint()).rstrip("/")
        self.authorize_writes = authorize_writes
        self.auth_path = auth_path or local_auth_path()
        self.server_id: str | None = None
        self.api_version: str | None = None
        self._api_key: str | None = api_key
        self._key_remembered = bool(api_key)
        self._identity_lock = threading.RLock()
        headers = {
            "User-Agent": "Zotero-MCP",
            "Zotero-API-Version": LOCAL_API_VERSION,
        }
        if host_header:
            headers["Host"] = host_header
        super().__init__(
            headers=headers,
            follow_redirects=True,
            timeout=30.0,
            transport=transport or httpx.HTTPTransport(http1=True, http2=False),
        )

    def _is_local_api_url(self, url: str | httpx.URL) -> bool:
        return str(url).startswith(self.local_endpoint)

    def _is_upload_url(self, url: str | httpx.URL) -> bool:
        return "/api/local/uploads/" in str(url)

    def _capture_identity(self, response: httpx.Response) -> None:
        if not self._is_local_api_url(response.request.url):
            return
        response_server_id = response.headers.get("Zotero-Server-ID")
        response_version = response.headers.get("Zotero-API-Version")
        if response_version:
            self.api_version = response_version
        if not response_server_id:
            return
        if self.server_id and self.server_id != response_server_id:
            old_server_id = self.server_id
            self.server_id = response_server_id
            self._api_key = None
            self._key_remembered = False
            raise LocalApiServerChangedError(
                "The Zotero Local API switched database instances "
                f"({old_server_id} -> {response_server_id}). Re-read the item "
                "and retry so local object versions cannot be mixed."
            )
        self.server_id = response_server_id

    def ensure_server_id(self) -> str:
        """Probe the API root and return the running Zotero database identity."""
        with self._identity_lock:
            if self.server_id:
                return self.server_id
            response = super().request("GET", f"{self.local_endpoint}/")
            response.raise_for_status()
            self._capture_identity(response)
            if self.api_version and self.api_version != LOCAL_API_VERSION:
                raise LocalApiWriteUnavailableError(
                    "Zotero MCP supports Local API version 3, but the running "
                    f"Zotero reports version {self.api_version}."
                )
            if not self.server_id:
                raise LocalApiWriteUnavailableError(
                    "The running Zotero does not advertise Zotero-Server-ID; "
                    "upgrade to a Zotero version with Local API write support."
                )
            return self.server_id

    def _read_auth_state(self) -> dict[str, Any]:
        try:
            with self.auth_path.open(encoding="utf-8") as state_file:
                value = json.load(state_file)
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _remembered_key(self) -> str | None:
        server_id = self.ensure_server_id()
        entry = self._read_auth_state().get("servers", {}).get(server_id, {})
        key = entry.get("key") if isinstance(entry, dict) else None
        return key if isinstance(key, str) and key else None

    def _save_remembered_key(self, key: str) -> None:
        server_id = self.ensure_server_id()
        state = self._read_auth_state()
        state["version"] = 1
        servers = state.setdefault("servers", {})
        servers[server_id] = {"key": key}
        self.auth_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.auth_path, state, indent=2)
        self.auth_path.chmod(0o600)

    def _forget_remembered_key(self) -> None:
        if not self.server_id:
            return
        state = self._read_auth_state()
        servers = state.get("servers")
        if not isinstance(servers, dict) or self.server_id not in servers:
            return
        servers.pop(self.server_id, None)
        self.auth_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.auth_path, state, indent=2)
        self.auth_path.chmod(0o600)

    def _authorize(self) -> None:
        server_id = self.ensure_server_id()
        response = super().request(
            "POST",
            f"{self.local_endpoint}/local/authorize",
            headers={
                "Content-Type": "application/json",
                "Zotero-Server-ID": server_id,
            },
            json={"appName": LOCAL_API_APP_NAME},
            timeout=120.0,
        )
        self._capture_identity(response)
        if response.status_code == 403:
            raise LocalApiAuthorizationError(
                "Zotero denied local write access. Retry the operation and "
                "choose Allow or Always Allow in Zotero's authorization dialog."
            )
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "shortly")
            raise LocalApiAuthorizationError(
                f"Zotero is rate-limiting authorization prompts; retry after {retry_after}."
            )
        response.raise_for_status()
        payload = response.json()
        key = payload.get("key") if isinstance(payload, dict) else None
        if not isinstance(key, str) or not key:
            raise LocalApiAuthorizationError("Zotero approved local writes but returned no authorization key.")
        self._api_key = key
        self._key_remembered = payload.get("remember") is True
        if self._key_remembered:
            self._save_remembered_key(key)

    def _ensure_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        remembered = self._remembered_key()
        if remembered:
            self._api_key = remembered
            self._key_remembered = True
            return remembered
        self._authorize()
        assert self._api_key is not None
        return self._api_key

    def _clear_key(self, *, forget_persistent: bool) -> None:
        if forget_persistent:
            self._forget_remembered_key()
        self._api_key = None
        self._key_remembered = False

    def ensure_write_authorization(self) -> dict[str, Any]:
        """Ensure a local key is available, prompting through Zotero if needed."""
        self._ensure_api_key()
        return {
            "server_id": self.ensure_server_id(),
            "remembered": self._key_remembered,
        }

    def request(self, method: str, url: str | httpx.URL, **kwargs) -> httpx.Response:
        """Send a request, authorizing and retrying local writes when needed."""
        normalized_method = method.upper()
        local_write = (
            self.authorize_writes
            and normalized_method in _WRITE_METHODS
            and self._is_local_api_url(url)
            and not self._is_upload_url(url)
        )
        if not local_write:
            request_kwargs = dict(kwargs)
            if self.server_id and self._is_local_api_url(url):
                headers = dict(request_kwargs.get("headers") or {})
                headers.setdefault("Zotero-Server-ID", self.server_id)
                request_kwargs["headers"] = headers
            response = super().request(method, url, **request_kwargs)
            self._capture_identity(response)
            return response

        for attempt in range(2):
            key = self._ensure_api_key()
            headers = dict(kwargs.get("headers") or {})
            headers["Zotero-Server-ID"] = self.ensure_server_id()
            headers["Zotero-API-Key"] = key
            request_kwargs = {**kwargs, "headers": headers}
            response = super().request(method, url, **request_kwargs)
            self._capture_identity(response)
            if response.status_code == 401 and attempt == 0:
                self._clear_key(forget_persistent=True)
                continue
            if not self._key_remembered:
                self._clear_key(forget_persistent=False)
            return response
        return response


class LocalWriteZotero(zotero.Zotero):
    """Pyzotero client with Zotero Local API's raw-byte upload protocol."""

    def _upload_local_file(
        self,
        attachment_key: str,
        file_path: str | Path,
        *,
        previous_md5: str | None = None,
    ) -> bool:
        source = Path(file_path)
        stat = source.stat()
        digest = hashlib.md5()  # noqa: S324 - mandated by Zotero's API
        with source.open("rb") as source_file:
            for chunk in iter(lambda: source_file.read(65536), b""):
                digest.update(chunk)
        headers = (
            {
                "If-Match": previous_md5,
            }
            if previous_md5
            else {"If-None-Match": "*"}
        )
        file_endpoint = build_url(
            self.endpoint,
            f"/{self.library_type}/{self.library_id}/items/{attachment_key}/file",
        )
        registration = self.client.post(
            file_endpoint,
            data={
                "md5": digest.hexdigest(),
                "filename": source.name,
                "filesize": stat.st_size,
                "mtime": int(stat.st_mtime * 1000),
            },
            headers=headers,
        )
        registration.raise_for_status()
        auth = registration.json()
        if auth.get("exists"):
            return False
        upload_url = (
            f"{self.endpoint}/local/uploads/"
            f"{quote(str(auth['uploadKey']), safe='')}"
        )
        upload = self.client.post(
            upload_url,
            content=source.read_bytes(),
            headers={
                "Content-Type": auth.get("contentType")
                or mimetypes.guess_type(source.name)[0]
                or "application/octet-stream"
            },
            timeout=self.upload_timeout,
        )
        upload.raise_for_status()
        completion = self.client.post(
            file_endpoint,
            data={"upload": auth["uploadKey"]},
            headers=headers,
        )
        completion.raise_for_status()
        return True

    def _create_and_upload_local_attachments(
        self,
        files: list[tuple[str, str]],
        parentid: str | None = None,
    ) -> dict[str, list]:
        templates = []
        source_paths: list[Path] = []
        for title, filename in files:
            source = Path(filename)
            template = self._attachment_template("imported_file")
            template["title"] = title
            template["filename"] = source.name
            template["contentType"] = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
            if parentid:
                template["parentItem"] = parentid
            templates.append(template)
            source_paths.append(source)

        created = self.create_items(templates)
        result: dict[str, list] = {"success": [], "failure": [], "unchanged": []}
        successes = created.get("success", {}) if isinstance(created, dict) else {}
        failures = created.get("failed", {}) if isinstance(created, dict) else {}
        for index, template in enumerate(templates):
            key = successes.get(str(index)) or successes.get(index)
            if not key:
                result["failure"].append({**template, "error": failures.get(str(index), "creation failed")})
                continue
            attached = {**template, "key": key}
            uploaded = self._upload_local_file(key, source_paths[index])
            result["success" if uploaded else "unchanged"].append(attached)
        return result

    def attachment_both(
        self,
        files: list[tuple[str, str]],
        parentid: str | None = None,
    ) -> dict[str, list]:
        return self._create_and_upload_local_attachments(files, parentid)

    def attachment_simple(
        self,
        files: list[str],
        parentid: str | None = None,
    ) -> dict[str, list]:
        return self._create_and_upload_local_attachments(
            [(Path(filename).name, filename) for filename in files],
            parentid,
        )

    def upload_attachments(
        self,
        attachments: Any,
        parentid: str | None = None,
        basedir: str | None = None,
    ) -> dict[str, list]:
        del parentid
        root = Path(basedir) if basedir else Path()
        result: dict[str, list] = {"success": [], "failure": [], "unchanged": []}
        for attachment in attachments:
            key = attachment.get("key")
            filename = attachment.get("filename")
            if not key or not filename:
                result["failure"].append(attachment)
                continue
            uploaded = self._upload_local_file(
                key,
                root / filename,
                previous_md5=attachment.get("md5"),
            )
            result["success" if uploaded else "unchanged"].append(attachment)
        return result
