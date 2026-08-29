"""Tests for authenticated Zotero Local API writes and uploads."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import httpx
import pytest

from zotero_mcp import cli
from zotero_mcp import client as zclient
from zotero_mcp.local_api import (
    LocalApiAuthorizationError,
    LocalApiHttpClient,
    LocalApiServerChangedError,
    LocalWriteZotero,
    remote_local_api_endpoint,
    writable_local_api_endpoints,
)
from zotero_mcp.tools import _helpers

ENDPOINT = "http://127.0.0.1:23119/api"
SERVER_ID = "server-database-one"


def _response(status=200, *, json_data=None, headers=None):
    return httpx.Response(
        status,
        json=json_data,
        headers={
            "Zotero-API-Version": "3",
            "Zotero-Server-ID": SERVER_ID,
            **(headers or {}),
        },
    )


def test_remembered_local_grant_is_persisted_and_reused(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.path == "/api/":
            return _response()
        if request.url.path == "/api/local/authorize":
            assert request.headers["Zotero-Server-ID"] == SERVER_ID
            assert json.loads(request.content) == {"appName": "Zotero MCP"}
            return _response(json_data={"key": "a" * 32, "remember": True})
        assert request.headers["Zotero-Server-ID"] == SERVER_ID
        assert request.headers["Zotero-API-Key"] == "a" * 32
        return _response(204)

    auth_path = tmp_path / "auth.json"
    first = LocalApiHttpClient(
        authorize_writes=True,
        endpoint=ENDPOINT,
        auth_path=auth_path,
        transport=httpx.MockTransport(handler),
    )
    first.patch(f"{ENDPOINT}/users/0/items/ABC", json={"title": "One"})
    assert auth_path.stat().st_mode & 0o777 == 0o600
    first.close()

    second = LocalApiHttpClient(
        authorize_writes=True,
        endpoint=ENDPOINT,
        auth_path=auth_path,
        transport=httpx.MockTransport(handler),
    )
    second.patch(f"{ENDPOINT}/users/0/items/ABC", json={"title": "Two"})
    second.close()

    assert [request.url.path for request in calls].count("/api/local/authorize") == 1


def test_single_use_grant_is_requested_for_each_write(tmp_path):
    authorizations = 0

    def handler(request):
        nonlocal authorizations
        if request.url.path == "/api/":
            return _response()
        if request.url.path == "/api/local/authorize":
            authorizations += 1
            return _response(json_data={"key": str(authorizations) * 32, "remember": False})
        assert request.headers["Zotero-API-Key"] in {"1" * 32, "2" * 32}
        return _response(204)

    client = LocalApiHttpClient(
        authorize_writes=True,
        endpoint=ENDPOINT,
        auth_path=tmp_path / "auth.json",
        transport=httpx.MockTransport(handler),
    )
    client.patch(f"{ENDPOINT}/users/0/items/A", json={})
    client.patch(f"{ENDPOINT}/users/0/items/B", json={})
    client.close()

    assert authorizations == 2
    assert not (tmp_path / "auth.json").exists()


def test_revoked_remembered_key_is_forgotten_and_reauthorized(tmp_path):
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps({"version": 1, "servers": {SERVER_ID: {"key": "x" * 32}}}))
    writes = []

    def handler(request):
        if request.url.path == "/api/":
            return _response()
        if request.url.path == "/api/local/authorize":
            return _response(json_data={"key": "y" * 32, "remember": True})
        writes.append(request.headers["Zotero-API-Key"])
        return _response(401 if len(writes) == 1 else 204)

    client = LocalApiHttpClient(
        authorize_writes=True,
        endpoint=ENDPOINT,
        auth_path=auth_path,
        transport=httpx.MockTransport(handler),
    )
    response = client.patch(f"{ENDPOINT}/users/0/items/A", json={})
    client.close()

    assert response.status_code == 204
    assert writes == ["x" * 32, "y" * 32]
    assert json.loads(auth_path.read_text())["servers"][SERVER_ID]["key"] == "y" * 32


def test_denied_authorization_is_actionable(tmp_path):
    def handler(request):
        if request.url.path == "/api/":
            return _response()
        return _response(403, json_data={"denied": True})

    client = LocalApiHttpClient(
        authorize_writes=True,
        endpoint=ENDPOINT,
        auth_path=tmp_path / "auth.json",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(LocalApiAuthorizationError, match="denied"):
        client.post(f"{ENDPOINT}/users/0/items", json=[])
    client.close()


def test_server_change_rejects_cached_version_namespace(tmp_path):
    responses = 0

    def handler(request):
        nonlocal responses
        responses += 1
        if responses == 2:
            assert request.headers["Zotero-Server-ID"] == SERVER_ID
        server_id = SERVER_ID if responses == 1 else "server-database-two"
        return _response(headers={"Zotero-Server-ID": server_id})

    client = LocalApiHttpClient(
        endpoint=ENDPOINT,
        auth_path=tmp_path / "auth.json",
        transport=httpx.MockTransport(handler),
    )
    assert client.ensure_server_id() == SERVER_ID
    with pytest.raises(LocalApiServerChangedError, match="switched database"):
        client.get(f"{ENDPOINT}/users/0/items")
    client.close()


def test_local_file_upload_uses_raw_bytes_and_two_authorized_stages(tmp_path):
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-local-upload")
    authorized_requests = []
    raw_upload = []
    authorization_count = 0

    def handler(request):
        nonlocal authorization_count
        if request.url.path == "/api/":
            return _response()
        if request.url.path == "/api/local/authorize":
            authorization_count += 1
            return _response(
                json_data={
                    "key": str(authorization_count) * 32,
                    "remember": False,
                }
            )
        if request.url.path == "/api/local/uploads/UPLOAD1":
            raw_upload.append(request)
            assert "Zotero-API-Key" not in request.headers
            return _response(201)
        authorized_requests.append(request)
        if len(authorized_requests) == 1:
            return _response(
                json_data={
                    "url": f"{ENDPOINT}/local/uploads/UPLOAD1",
                    "uploadKey": "UPLOAD1",
                    "contentType": "application/pdf",
                    "prefix": "",
                    "suffix": "",
                }
            )
        return _response(204)

    http_client = LocalApiHttpClient(
        authorize_writes=True,
        endpoint=ENDPOINT,
        auth_path=tmp_path / "auth.json",
        transport=httpx.MockTransport(handler),
    )
    zot = LocalWriteZotero(
        library_id="0",
        library_type="user",
        local=True,
        client=http_client,
    )
    zot.endpoint = ENDPOINT

    assert zot._upload_local_file("ATTACH01", source) is True
    zot.client.close()

    assert authorization_count == 2
    assert len(authorized_requests) == 2
    assert raw_upload[0].content == source.read_bytes()
    assert raw_upload[0].headers["Content-Type"] == "application/pdf"
    assert "upload=UPLOAD1" in authorized_requests[1].content.decode()


def test_remote_file_upload_ignores_backend_localhost_upload_url(tmp_path):
    endpoint = "http://paper-workstation:23119/api"
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-remote-upload")
    raw_upload_urls = []

    def handler(request):
        if request.url.path == "/api/":
            return httpx.Response(
                200,
                headers={
                    "Zotero-API-Version": "3",
                    "Zotero-Server-ID": "remote-database",
                },
            )
        if request.url.path == "/api/local/authorize":
            return httpx.Response(
                200,
                json={"key": "a" * 32, "remember": True},
                headers={
                    "Zotero-API-Version": "3",
                    "Zotero-Server-ID": "remote-database",
                },
            )
        if request.url.path == "/api/local/uploads/REMOTE1":
            raw_upload_urls.append(str(request.url))
            return httpx.Response(201)
        if not raw_upload_urls:
            return httpx.Response(
                200,
                json={
                    "url": "http://127.0.0.1:23119/api/local/uploads/REMOTE1",
                    "uploadKey": "REMOTE1",
                    "contentType": "application/pdf",
                },
                headers={
                    "Zotero-API-Version": "3",
                    "Zotero-Server-ID": "remote-database",
                },
            )
        return httpx.Response(
            204,
            headers={
                "Zotero-API-Version": "3",
                "Zotero-Server-ID": "remote-database",
            },
        )

    http_client = LocalApiHttpClient(
        authorize_writes=True,
        endpoint=endpoint,
        auth_path=tmp_path / "auth.json",
        transport=httpx.MockTransport(handler),
    )
    zot = LocalWriteZotero(
        library_id="0",
        library_type="user",
        local=True,
        client=http_client,
    )
    zot.endpoint = endpoint

    assert zot._upload_local_file("ATTACH01", source) is True
    zot.client.close()

    assert raw_upload_urls == [f"{endpoint}/local/uploads/REMOTE1"]


def test_authorize_local_writes_cli_reports_remembered_grant(
    monkeypatch,
    capsys,
):
    authorization_client = SimpleNamespace(
        ensure_write_authorization=lambda: {
            "server_id": SERVER_ID,
            "remembered": True,
        }
    )
    monkeypatch.setattr(sys, "argv", ["zotero-mcp", "authorize-local-writes"])
    monkeypatch.setattr(cli, "setup_zotero_environment", lambda: None)
    monkeypatch.setattr(
        cli,
        "_get_local_write_client_for_authorization",
        lambda: SimpleNamespace(client=authorization_client),
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert SERVER_ID in output
    assert "remembered until revoked" in output


def test_local_attachment_upload_never_duplicates_bytes_to_webdav(monkeypatch):
    monkeypatch.setattr(
        "zotero_mcp.webdav.is_webdav_configured",
        lambda: True,
    )
    monkeypatch.setattr(
        "zotero_mcp.webdav.upload_attachment_to_webdav",
        lambda **_kwargs: pytest.fail("local uploads must be left to Zotero"),
    )

    suffix = _helpers._maybe_upload_to_webdav(
        {"success": [{"key": "ATTACH01"}]},
        "/tmp/paper.pdf",
        None,
        write_zot=SimpleNamespace(local=True),
    )

    assert suffix == ""


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("http://paper-workstation:23119/api", "http://paper-workstation:23119/api"),
        ("https://zotero.lan/api/", "https://zotero.lan/api"),
        ("http://127.0.0.1:23120", "http://127.0.0.1:23120/api"),
    ],
)
def test_remote_local_endpoint_accepts_lan_https_and_tunnels(
    monkeypatch,
    configured,
    expected,
):
    monkeypatch.setenv("ZOTERO_REMOTE_LOCAL_URL", configured)

    assert remote_local_api_endpoint() == expected


@pytest.mark.parametrize(
    "configured",
    [
        "ftp://paper-workstation/api",
        "http://user:password@paper-workstation/api",
        "http://paper-workstation/not-api",
        "http://paper-workstation/api?key=secret",
    ],
)
def test_remote_local_endpoint_rejects_ambiguous_or_credentialed_urls(
    monkeypatch,
    configured,
):
    monkeypatch.setenv("ZOTERO_REMOTE_LOCAL_URL", configured)

    with pytest.raises(ValueError, match="ZOTERO_REMOTE_LOCAL_URL"):
        remote_local_api_endpoint()


def test_writable_endpoints_are_remote_first(monkeypatch):
    monkeypatch.setenv(
        "ZOTERO_REMOTE_LOCAL_URL",
        "http://paper-workstation:23119/api",
    )
    monkeypatch.setenv("ZOTERO_LOCAL_PORT", "23119")

    assert writable_local_api_endpoints() == [
        ("remote-local", "http://paper-workstation:23119/api"),
        ("server-local", "http://127.0.0.1:23119/api"),
    ]


def test_write_client_falls_back_to_server_local_when_remote_is_offline(
    monkeypatch,
    tmp_path,
):
    remote = "http://paper-workstation:23119/api"
    local = "http://127.0.0.1:23119/api"
    probes = []

    def make_client(*, authorize_writes=False, endpoint=None):
        def handler(request):
            probes.append(str(request.url))
            if endpoint == remote:
                raise httpx.ConnectError("workstation offline", request=request)
            return httpx.Response(
                200,
                headers={
                    "Zotero-API-Version": "3",
                    "Zotero-Server-ID": "server-local-database",
                },
            )

        return LocalApiHttpClient(
            authorize_writes=authorize_writes,
            endpoint=endpoint,
            auth_path=tmp_path / "auth.json",
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr(
        zclient,
        "writable_local_api_endpoints",
        lambda: [("remote-local", remote), ("server-local", local)],
    )
    monkeypatch.setattr(zclient, "_make_local_http_client", make_client)
    monkeypatch.setattr(
        zclient,
        "get_current_library",
        lambda: {"library_id": "0", "library_type": "user"},
    )

    write_zot = zclient.get_local_write_zotero_client()

    assert write_zot is not None
    assert write_zot.endpoint == local
    assert write_zot.local_endpoint_role == "server-local"
    assert len(probes) == 2
    write_zot.client.close()


def test_write_client_uses_remote_without_probing_server_local(
    monkeypatch,
    tmp_path,
):
    remote = "http://paper-workstation:23119/api"
    local = "http://127.0.0.1:23119/api"
    probes = []

    def make_client(*, authorize_writes=False, endpoint=None):
        def handler(request):
            probes.append(str(request.url))
            return httpx.Response(
                200,
                headers={
                    "Zotero-API-Version": "3",
                    "Zotero-Server-ID": "workstation-database",
                },
            )

        return LocalApiHttpClient(
            authorize_writes=authorize_writes,
            endpoint=endpoint,
            auth_path=tmp_path / "auth.json",
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr(
        zclient,
        "writable_local_api_endpoints",
        lambda: [("remote-local", remote), ("server-local", local)],
    )
    monkeypatch.setattr(zclient, "_make_local_http_client", make_client)
    monkeypatch.setattr(
        zclient,
        "get_current_library",
        lambda: {"library_id": "0", "library_type": "user"},
    )

    write_zot = zclient.get_local_write_zotero_client()

    assert write_zot is not None
    assert write_zot.endpoint == remote
    assert write_zot.local_endpoint_role == "remote-local"
    assert probes == [f"{remote}/"]
    write_zot.client.close()
