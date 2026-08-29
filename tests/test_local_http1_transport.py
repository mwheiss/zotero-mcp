"""Regression test for #160: force HTTP/1.1 for Zotero local API.

Zotero 8's local server (port 23119) only speaks HTTP/1.0. httpx defaults
to attempting HTTP/2 negotiation, which the local server rejects with
**502 Bad Gateway** — every tool call fails even though the MCP starts
cleanly. The fix pins the local pyzotero client's transport to HTTP/1.1.
"""

import httpx
from pyzotero import zotero

from zotero_mcp import client as zclient


def test_make_local_http_client_pins_http1():
    """The helper must return an httpx.Client whose transport is HTTP/1.1 only."""
    c = zclient._make_local_http_client()
    try:
        assert isinstance(c, httpx.Client)
        transport = c._transport
        assert isinstance(transport, httpx.HTTPTransport)
        # _pool is httpcore.ConnectionPool; http1/http2 are stored on it.
        pool = transport._pool
        assert pool._http1 is True
        assert pool._http2 is False
    finally:
        c.close()


def test_get_local_zotero_client_uses_http1_transport(monkeypatch):
    """get_local_zotero_client must pass the HTTP/1.1 client to pyzotero."""
    captured: dict = {}

    real_init = zotero.Zotero.__init__

    def spy_init(self, *args, **kwargs):
        captured["kwargs"] = kwargs
        # Make the constructor a no-op so we don't actually talk to Zotero.
        # Set attributes pyzotero would normally populate (including .client,
        # which pyzotero's __del__ references).
        self.library_id = kwargs.get("library_id", "0")
        self.library_type = kwargs.get("library_type", "users")
        self.api_key = kwargs.get("api_key")
        self.client = kwargs.get("client")

    monkeypatch.setattr(zotero.Zotero, "__init__", spy_init)
    # items() probe must not hit the network.
    monkeypatch.setattr(zotero.Zotero, "items", lambda self, **_kw: [])

    client = zclient.get_local_zotero_client()

    assert client is not None
    assert captured["kwargs"].get("local") is True
    passed = captured["kwargs"].get("client")
    assert isinstance(passed, httpx.Client)
    assert isinstance(passed._transport, httpx.HTTPTransport)
    assert passed._transport._pool._http2 is False

    # Restore for hygiene.
    monkeypatch.setattr(zotero.Zotero, "__init__", real_init)


def test_get_local_zotero_client_uses_active_library(monkeypatch):
    captured: dict = {}

    def spy_init(self, *args, **kwargs):
        captured["kwargs"] = kwargs
        self.library_id = kwargs.get("library_id")
        self.library_type = kwargs.get("library_type")
        self.api_key = kwargs.get("api_key")
        self.client = kwargs.get("client")

    monkeypatch.setattr(zotero.Zotero, "__init__", spy_init)
    monkeypatch.setattr(zotero.Zotero, "items", lambda self, **_kw: [])
    zclient.set_active_library("5910265", "group", session_id=None)

    try:
        client = zclient.get_local_zotero_client()
        assert client is not None
        assert captured["kwargs"]["library_id"] == "5910265"
        assert captured["kwargs"]["library_type"] == "group"
    finally:
        zclient.clear_active_library(session_id=None)


def test_get_zotero_client_uses_http1_only_when_local(monkeypatch):
    """The general get_zotero_client should pass an HTTP/1.1 client only
    when the underlying connection is local; the cloud Web API at
    api.zotero.org speaks HTTP/2 fine, so no transport override is needed
    in that case."""
    captured: dict = {}

    def spy_init(self, *args, **kwargs):
        captured["kwargs"] = kwargs
        self.library_id = kwargs.get("library_id", "0")
        self.library_type = kwargs.get("library_type", "users")
        self.api_key = kwargs.get("api_key")
        self.client = kwargs.get("client")

    monkeypatch.setattr(zotero.Zotero, "__init__", spy_init)
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "0")
    monkeypatch.delenv("ZOTERO_API_KEY", raising=False)

    zclient.get_zotero_client()
    local_client = captured["kwargs"].get("client")
    assert isinstance(local_client, httpx.Client)
    assert local_client._transport._pool._http2 is False

    captured.clear()
    monkeypatch.setenv("ZOTERO_LOCAL", "false")
    monkeypatch.setenv("ZOTERO_API_KEY", "fake")

    zclient.get_zotero_client()
    assert captured["kwargs"].get("client") is None, (
        "Cloud API path should not override the transport — only the local API needs HTTP/1.1."
    )


def test_local_client_never_sends_cloud_api_key(monkeypatch):
    captured = {}

    def spy_init(self, *args, **kwargs):
        captured.update(kwargs)
        self.library_id = kwargs.get("library_id")
        self.library_type = kwargs.get("library_type")
        self.api_key = kwargs.get("api_key")
        self.client = kwargs.get("client")
        self.endpoint = "http://localhost:23119/api"

    monkeypatch.setattr(zotero.Zotero, "__init__", spy_init)
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "0")
    monkeypatch.setenv("ZOTERO_API_KEY", "cloud-key-must-not-go-local")

    zclient.get_zotero_client()

    assert captured["local"] is True
    assert captured["api_key"] is None


def test_web_fallback_maps_local_user_zero_to_configured_account(monkeypatch):
    captured = {}

    def spy_init(self, *args, **kwargs):
        captured.update(kwargs)
        self.library_id = kwargs.get("library_id")
        self.library_type = kwargs.get("library_type")
        self.api_key = kwargs.get("api_key")
        self.client = kwargs.get("client")

    monkeypatch.setattr(zotero.Zotero, "__init__", spy_init)
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "20765677")
    monkeypatch.setenv("ZOTERO_API_KEY", "cloud-key")
    zclient.set_active_library("0", "user", session_id=None)
    try:
        client = zclient.get_web_zotero_client()
    finally:
        zclient.clear_active_library(session_id=None)

    assert client is not None
    assert captured["library_id"] == "20765677"
    assert captured["library_type"] == "user"


def test_web_fallback_rejects_feed_library(monkeypatch):
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "20765677")
    monkeypatch.setenv("ZOTERO_API_KEY", "cloud-key")
    zclient.set_active_library("12", "feed", session_id=None)
    try:
        assert zclient.get_web_zotero_client() is None
    finally:
        zclient.clear_active_library(session_id=None)
