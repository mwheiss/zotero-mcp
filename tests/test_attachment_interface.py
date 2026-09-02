"""Attachment interface, binary transfer, and destructive confirmation tests."""

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from conftest import DummyContext
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse
from starlette.routing import Route

from zotero_mcp import attachment_service as service
from zotero_mcp.attachment_http import download_attachment, upload_attachment
from zotero_mcp.client import AttachmentDownloadResult
from zotero_mcp.server import (
    list_attachments,
    mcp,
    prepare_attachment_change,
    put_attachment,
    set_attachment_trashed,
)


def _attachment(key="ATTACH01", version=7, md5="abc"):
    return {
        "key": key,
        "version": version,
        "data": {
            "key": key,
            "itemType": "attachment",
            "parentItem": "PARENT01",
            "title": "Paper PDF",
            "filename": "paper.pdf",
            "contentType": "application/pdf",
            "linkMode": "imported_file",
            "md5": md5,
            "dateModified": "2026-09-02T00:00:00Z",
        },
        "meta": {"fileSize": 1234},
    }


@pytest.fixture
def attachment_state(monkeypatch, tmp_path):
    monkeypatch.setenv("ZOTERO_MCP_ATTACHMENT_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("ZOTERO_MCP_ATTACHMENT_SECRET", raising=False)
    monkeypatch.delenv("ZOTERO_MCP_PUBLIC_BASE_URL", raising=False)
    return tmp_path


def test_attachment_descriptor_is_complete():
    descriptor = service.attachment_descriptor(_attachment())

    assert descriptor == {
        "attachment_key": "ATTACH01",
        "parent_item_key": "PARENT01",
        "title": "Paper PDF",
        "filename": "paper.pdf",
        "content_type": "application/pdf",
        "size": 1234,
        "md5": "abc",
        "version": 7,
        "date_modified": "2026-09-02T00:00:00Z",
        "link_mode": "imported_file",
        "url": "",
        "binary_readable": True,
        "binary_writable": True,
        "is_trashed": False,
    }


def test_binary_routes_are_registered_on_mcp_http_app():
    paths = {getattr(route, "path", None) for route in mcp.http_app().routes}

    assert "/mcp/attachments/download/{token}" in paths
    assert "/mcp/attachments/upload/{token}" in paths


def test_canonical_api_selector_matches_betterissa_priority():
    attachments = [
        {
            "key": "PDF00001",
            "data": {
                "itemType": "attachment",
                "title": "Publisher PDF",
                "filename": "paper.pdf",
                "contentType": "application/pdf",
            },
        },
        {
            "key": "READ0001",
            "data": {
                "itemType": "attachment",
                "title": "BetterIssa Reading View",
                "filename": "reading.html",
                "contentType": "text/html",
            },
        },
        {
            "key": "INDEX001",
            "data": {
                "itemType": "attachment",
                "title": "BetterIssa indexing text",
                "filename": "indexing.txt",
                "contentType": "text/plain",
            },
        },
        {
            "key": "STATE001",
            "data": {
                "itemType": "attachment",
                "title": "BetterIssa state",
                "filename": "state.json",
                "contentType": "application/json",
            },
        },
    ]

    selected, kind = service.select_best_attachment(attachments)

    assert selected["key"] == "INDEX001"
    assert kind == "betterissa-indexing"


def test_semantic_document_returns_plain_text_not_raw_json(tmp_path):
    path = tmp_path / "semantic.json"
    path.write_text(
        json.dumps(
            {
                "ok": True,
                "status": "complete",
                "structured": {"plain_text": "Canonical semantic body"},
                "machine_only": "must not leak",
            }
        ),
        encoding="utf-8",
    )

    text = service.document_text_from_file(path, "betterissa-semantic")

    assert text == "Canonical semantic body"
    assert "machine_only" not in text


def test_list_attachments_returns_all_children(monkeypatch):
    class Zotero:
        def item(self, key):
            return {"key": key, "data": {"key": key, "itemType": "journalArticle"}}

        def children(self, _key):
            return [_attachment(), {"key": "NOTE", "data": {"itemType": "note"}}]

    monkeypatch.setattr("zotero_mcp.tools.attachments._client.get_zotero_client", Zotero)

    result = json.loads(list_attachments(item_key="PARENT01", ctx=DummyContext()))

    assert result["count"] == 1
    assert result["attachments"][0]["attachment_key"] == "ATTACH01"


def test_signed_tokens_reject_tampering_and_wrong_action(attachment_state):
    token = service.make_token("download", {"attachment_key": "ATTACH01"})

    assert service.verify_token(token, "download")["attachment_key"] == "ATTACH01"
    with pytest.raises(ValueError, match="wrong action"):
        service.verify_token(token, "upload")
    with pytest.raises(ValueError, match="Invalid"):
        service.verify_token(token[:-1] + ("A" if token[-1] != "A" else "B"), "download")


@pytest.mark.asyncio
async def test_staged_upload_streams_and_verifies_bytes(attachment_state):
    payload = b"%PDF-1.4\nattachment bytes"
    digest = hashlib.sha256(payload).hexdigest()
    prepared = service.prepare_upload(
        "paper.pdf",
        "application/pdf",
        len(payload),
        digest,
        library={"library_id": "0", "library_type": "user"},
    )

    app = Starlette(
        routes=[Route("/mcp/attachments/upload/{token}", upload_attachment, methods=["PUT"])]
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.put(
            f"/mcp/attachments/upload/{prepared['upload_token']}",
            content=payload,
        )

    assert response.status_code == 201
    manifest, path = service.read_upload(prepared["upload_id"])
    assert manifest["status"] == "ready"
    assert path.read_bytes() == payload


@pytest.mark.asyncio
async def test_staged_upload_rejects_hash_mismatch(attachment_state):
    payload = b"actual bytes"
    prepared = service.prepare_upload(
        "paper.pdf",
        "application/pdf",
        len(payload),
        hashlib.sha256(b"different bytes").hexdigest(),
        library={"library_id": "0", "library_type": "user"},
    )

    app = Starlette(
        routes=[Route("/mcp/attachments/upload/{token}", upload_attachment, methods=["PUT"])]
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.put(
            f"/mcp/attachments/upload/{prepared['upload_token']}",
            content=payload,
        )

    assert response.status_code == 400
    with pytest.raises(ValueError, match="has not completed"):
        service.read_upload(prepared["upload_id"])


@pytest.mark.asyncio
async def test_signed_download_streams_exact_binary(monkeypatch, attachment_state):
    payload = b"exact binary attachment"
    token = service.make_token(
        "download",
        {
            "attachment_key": "ATTACH01",
            "filename": "paper.bin",
            "content_type": "application/octet-stream",
        },
    )

    def download(_key, destination, filename, **_kwargs):
        path = Path(destination) / filename
        path.write_bytes(payload)
        return AttachmentDownloadResult(path=path, source="Local Zotero", errors=[])

    monkeypatch.setattr(
        "zotero_mcp.attachment_http._client.download_attachment_file", download
    )
    monkeypatch.setattr(
        "zotero_mcp.attachment_http._client.get_local_zotero_client", lambda: object()
    )
    monkeypatch.setattr(
        "zotero_mcp.attachment_http._client.get_web_zotero_client", lambda: object()
    )
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": f"/mcp/attachments/download/{token}",
            "headers": [],
            "path_params": {"token": token},
        }
    )
    response = await download_attachment(request)

    assert isinstance(response, FileResponse)
    response_path = Path(response.path)
    assert response_path.read_bytes() == payload
    assert response.headers["content-disposition"].endswith('filename="paper.bin"')
    assert response.headers["cache-control"] == "private, no-store"
    shutil.rmtree(response_path.parent, ignore_errors=True)


def test_operation_confirmation_is_bound_and_one_use(attachment_state):
    details = {
        "action": "trash",
        "attachment_key": "ATTACH01",
        "expected_version": 7,
    }
    prepared = service.prepare_operation("trash", details)

    service.consume_operation(
        prepared["operation_id"],
        prepared["confirmation_token"],
        action="trash",
        details=details,
    )

    with pytest.raises(ValueError, match="already been used"):
        service.consume_operation(
            prepared["operation_id"],
            prepared["confirmation_token"],
            action="trash",
            details=details,
        )


def test_prepare_destructive_change_binds_live_version(monkeypatch, attachment_state):
    write_zot = SimpleNamespace(item=lambda _key: _attachment(version=11, md5="live-md5"))
    monkeypatch.setattr(
        "zotero_mcp.tools.attachments._helpers._get_write_client",
        lambda _ctx: (write_zot, write_zot),
    )

    result = json.loads(
        prepare_attachment_change(
            action="trash",
            attachment_key="ATTACH01",
            ctx=DummyContext(),
        )
    )

    assert result["preview"]["expected_version"] == 11
    assert result["preview"]["expected_md5"] == "live-md5"
    assert result["confirmation_token"]


def test_public_transfer_url_never_embeds_write_secret(monkeypatch, attachment_state):
    monkeypatch.setenv("ZOTERO_MCP_PUBLIC_BASE_URL", "https://example.test/private-mcp")
    monkeypatch.setenv("ZOTERO_MCP_WRITE_SECRET", "write-secret-value")

    prepared = service.prepare_upload(
        "paper.pdf",
        "application/pdf",
        0,
        hashlib.sha256(b"").hexdigest(),
        library={"library_id": "0", "library_type": "user"},
    )

    assert prepared["upload_url"].startswith("https://example.test/private-mcp/attachments/upload/")
    assert "write-secret-value" not in prepared["upload_url"]
    assert prepared["upload_token"] is None


def _ready_upload(payload: bytes, library: dict[str, str]):
    prepared = service.prepare_upload(
        "paper.pdf",
        "application/pdf",
        len(payload),
        hashlib.sha256(payload).hexdigest(),
        library=library,
    )
    manifest, path = service.read_upload(prepared["upload_id"], require_ready=False)
    path.write_bytes(payload)
    service.mark_upload_ready(
        prepared["upload_id"], size=len(payload), sha256=hashlib.sha256(payload).hexdigest()
    )
    return prepared, manifest


def test_replace_attachment_requires_bound_confirmation(monkeypatch, attachment_state):
    library = {"library_id": "0", "library_type": "user"}
    prepared_upload, _ = _ready_upload(b"replacement", library)
    uploads = []

    class WriteZotero:
        local_endpoint_role = "remote-local"

        def item(self, _key):
            return _attachment(version=7, md5="abc")

        def upload_attachments(self, attachments, basedir=None):
            uploads.append((attachments, basedir))
            return {"success": attachments, "failure": [], "unchanged": []}

    zot = WriteZotero()
    monkeypatch.setattr(
        "zotero_mcp.tools.attachments._helpers._get_write_client",
        lambda _ctx: (zot, zot),
    )
    monkeypatch.setattr(
        "zotero_mcp.tools.attachments._client.get_current_library", lambda: library
    )
    preview = json.loads(
        prepare_attachment_change(
            action="replace",
            attachment_key="ATTACH01",
            upload_id=prepared_upload["upload_id"],
            ctx=DummyContext(),
        )
    )

    blocked = put_attachment(
        parent_item_key="PARENT01",
        upload_id=prepared_upload["upload_id"],
        attachment_key="ATTACH01",
        ctx=DummyContext(),
    )
    assert "requires a prepared operation" in blocked
    assert uploads == []

    result = json.loads(
        put_attachment(
            parent_item_key="PARENT01",
            upload_id=prepared_upload["upload_id"],
            attachment_key="ATTACH01",
            operation_id=preview["operation_id"],
            confirmation_token=preview["confirmation_token"],
            ctx=DummyContext(),
        )
    )
    assert result["status"] == "replaced"
    assert result["write_target"] == "remote-local"
    assert uploads[0][0][0]["key"] == "ATTACH01"


def test_trash_requires_confirmation_but_restore_does_not(monkeypatch, attachment_state):
    patches = []

    class Response:
        status_code = 204
        text = ""

    class Client:
        def patch(self, **kwargs):
            patches.append(kwargs)
            return Response()

    zot = SimpleNamespace(
        item=lambda _key: _attachment(version=7, md5="abc"),
        endpoint="http://zotero.test/api",
        library_type="users",
        library_id="0",
        client=Client(),
    )
    monkeypatch.setattr(
        "zotero_mcp.tools.attachments._helpers._get_write_client",
        lambda _ctx: (zot, zot),
    )

    blocked = set_attachment_trashed(
        attachment_key="ATTACH01", trashed=True, ctx=DummyContext()
    )
    assert "requires a prepared operation" in blocked
    assert patches == []

    preview = json.loads(
        prepare_attachment_change(
            action="trash", attachment_key="ATTACH01", ctx=DummyContext()
        )
    )
    result = json.loads(
        set_attachment_trashed(
            attachment_key="ATTACH01",
            trashed=True,
            operation_id=preview["operation_id"],
            confirmation_token=preview["confirmation_token"],
            ctx=DummyContext(),
        )
    )
    assert result["status"] == "trashed"
    assert json.loads(patches[-1]["content"]) == {"deleted": 1}

    restored = json.loads(
        set_attachment_trashed(
            attachment_key="ATTACH01", trashed=False, ctx=DummyContext()
        )
    )
    assert restored["status"] == "restored"
    assert json.loads(patches[-1]["content"]) == {"deleted": 0}
