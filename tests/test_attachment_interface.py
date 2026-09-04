"""Attachment interface, binary transfer, and destructive confirmation tests."""

import hashlib
import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
from zotero_mcp import resources
from zotero_mcp.attachment_http import download_attachment, upload_attachment
from zotero_mcp.client import AttachmentDownloadResult
from zotero_mcp.server import (
    list_attachments,
    mcp,
    prepare_attachment_change,
    put_attachment,
    set_attachment_trashed,
    update_attachment,
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
    monkeypatch.delenv("ZOTERO_MCP_ATTACHMENT_INLINE_MAX_BYTES", raising=False)
    monkeypatch.delenv("ZOTERO_MCP_ATTACHMENT_RESOURCE_MAX_BYTES", raising=False)
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


def test_capability_secret_rejects_short_configuration(monkeypatch, attachment_state):
    monkeypatch.setenv("ZOTERO_MCP_ATTACHMENT_SECRET", "known-short-key")

    with pytest.raises(ValueError, match="at least 32 bytes"):
        service.make_token("download", {"attachment_key": "ATTACH01"})


def test_capability_secret_atomically_repairs_short_state(attachment_state):
    secret_path = service._state_root() / "secret"
    secret_path.parent.mkdir(parents=True)
    secret_path.write_bytes(b"")

    with ThreadPoolExecutor(max_workers=4) as executor:
        values = list(executor.map(lambda _index: service._secret(), range(4)))

    assert len(set(values)) == 1
    assert len(values[0]) >= service.MIN_SECRET_BYTES
    assert secret_path.read_bytes() == values[0]
    if os.name != "nt":
        assert secret_path.stat().st_mode & 0o777 == 0o600


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
    scoped_clients = []
    token = service.make_token(
        "download",
        {
            "attachment_key": "ATTACH01",
            "filename": "paper.bin",
            "content_type": "application/octet-stream",
            "library": {"library_id": "5910265", "library_type": "group"},
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
        "zotero_mcp.attachment_http._client.get_local_zotero_client",
        lambda **kwargs: scoped_clients.append(("local", kwargs["library"])) or object(),
    )
    monkeypatch.setattr(
        "zotero_mcp.attachment_http._client.get_web_zotero_client",
        lambda **kwargs: scoped_clients.append(("web", kwargs["library"])) or object(),
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
    assert scoped_clients == [
        ("local", {"library_id": "5910265", "library_type": "group"}),
        ("web", {"library_id": "5910265", "library_type": "group"}),
    ]
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


def test_multiple_operations_can_share_the_state_directory(attachment_state):
    first = service.prepare_operation("trash", {"attachment_key": "ATTACH01"})
    second = service.prepare_operation("trash", {"attachment_key": "ATTACH02"})

    assert first["operation_id"] != second["operation_id"]


def test_operation_confirmation_is_atomic_under_concurrency(attachment_state):
    details = {
        "action": "trash",
        "attachment_key": "ATTACH01",
        "expected_version": 7,
    }
    prepared = service.prepare_operation("trash", details)

    def consume():
        try:
            service.consume_operation(
                prepared["operation_id"],
                prepared["confirmation_token"],
                action="trash",
                details=details,
            )
            return "used"
        except ValueError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _index: consume(), range(2)))

    assert outcomes.count("used") == 1
    assert sum("already been used" in value for value in outcomes) == 1


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


def test_put_attachment_idempotency_is_atomic_under_concurrency(
    monkeypatch, attachment_state
):
    library = {"library_id": "0", "library_type": "user"}
    prepared, _ = _ready_upload(b"new attachment", library)

    class WriteZotero:
        local_endpoint_role = "server-local"

        def __init__(self):
            self.created = []
            self.lock = threading.Lock()

        def item(self, key):
            if key == "PARENT01":
                return {"key": key, "data": {"itemType": "journalArticle"}}
            if key in self.created:
                return _attachment(key=key)
            raise KeyError(key)

        def children(self, _parent_key, **_kwargs):
            return []

        def attachment_both(self, _attachments, parentid=None):
            with self.lock:
                key = f"NEW{len(self.created) + 1:05d}"
                time.sleep(0.05)
                self.created.append(key)
            return {"success": {"0": key}}

    zot = WriteZotero()
    monkeypatch.setattr(
        "zotero_mcp.tools.attachments._helpers._get_write_client",
        lambda _ctx: (zot, zot),
    )
    monkeypatch.setattr(
        "zotero_mcp.tools.attachments._client.get_current_library",
        lambda: library,
    )

    def create():
        return json.loads(
            put_attachment(
                parent_item_key="PARENT01",
                upload_id=prepared["upload_id"],
                idempotency_key="stable-request-key",
                ctx=DummyContext(),
            )
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: create(), range(2)))

    assert zot.created == ["NEW00001"]
    assert {result["status"] for result in results} == {
        "created",
        "already-created",
    }
    manifest, path = service.read_upload(
        prepared["upload_id"], allow_consumed=True
    )
    assert manifest["status"] == "consumed"
    assert not path.exists()


def test_failed_attachment_operation_preserves_staged_bytes(
    monkeypatch, attachment_state
):
    library = {"library_id": "0", "library_type": "user"}
    prepared, _ = _ready_upload(b"retry me", library)

    class WriteZotero:
        def item(self, key):
            if key == "PARENT01":
                return {"key": key, "data": {"itemType": "journalArticle"}}
            raise KeyError(key)

        def attachment_both(self, _attachments, parentid=None):
            return {"failure": [{"error": "temporary Zotero failure"}]}

        def children(self, _parent_key, **_kwargs):
            return []

    zot = WriteZotero()
    monkeypatch.setattr(
        "zotero_mcp.tools.attachments._helpers._get_write_client",
        lambda _ctx: (zot, zot),
    )
    monkeypatch.setattr(
        "zotero_mcp.tools.attachments._client.get_current_library",
        lambda: library,
    )

    result = put_attachment(
        parent_item_key="PARENT01",
        upload_id=prepared["upload_id"],
        idempotency_key="retryable-request",
        ctx=DummyContext(),
    )

    assert result.startswith("Error:")
    manifest, path = service.read_upload(prepared["upload_id"])
    assert manifest["status"] == "ready"
    assert path.read_bytes() == b"retry me"
    assert service.idempotency_record("retryable-request") is None


def test_attachment_creation_recovers_after_crash_before_completion_record(
    monkeypatch, attachment_state
):
    library = {"library_id": "0", "library_type": "user"}
    payload = b"crash-durable attachment"
    expected_md5 = hashlib.md5(payload).hexdigest()  # noqa: S324 - Zotero identity
    prepared, _ = _ready_upload(payload, library)

    class WriteZotero:
        local_endpoint_role = "server-local"

        def __init__(self):
            self.created = {}
            self.create_calls = 0

        def item(self, key):
            if key == "PARENT01":
                return {"key": key, "data": {"itemType": "journalArticle"}}
            return self.created[key]

        def children(self, _parent_key, **_kwargs):
            return list(self.created.values())

        def attachment_both(self, _attachments, parentid=None):
            self.create_calls += 1
            key = "RECOVER1"
            self.created[key] = _attachment(
                key=key, md5=expected_md5
            )
            return {"success": {"0": key}}

    zot = WriteZotero()
    monkeypatch.setattr(
        "zotero_mcp.tools.attachments._helpers._get_write_client",
        lambda _ctx: (zot, zot),
    )
    monkeypatch.setattr(
        "zotero_mcp.tools.attachments._client.get_current_library",
        lambda: library,
    )
    original_save = service.save_idempotency_record

    def crash_before_complete(key, value):
        if value.get("state") == "complete":
            raise RuntimeError("simulated process crash before completion record")
        return original_save(key, value)

    monkeypatch.setattr(service, "save_idempotency_record", crash_before_complete)
    first = put_attachment(
        parent_item_key="PARENT01",
        upload_id=prepared["upload_id"],
        idempotency_key="crash-recovery-key",
        ctx=DummyContext(),
    )
    assert first.startswith("Error:")
    assert service.idempotency_record("crash-recovery-key")["state"] == "pending"

    monkeypatch.setattr(service, "save_idempotency_record", original_save)
    recovered = json.loads(
        put_attachment(
            parent_item_key="PARENT01",
            upload_id=prepared["upload_id"],
            idempotency_key="crash-recovery-key",
            ctx=DummyContext(),
        )
    )

    assert recovered["status"] == "already-created"
    assert recovered["attachment"]["attachment_key"] == "RECOVER1"
    assert zot.create_calls == 1
    record = service.idempotency_record("crash-recovery-key")
    assert record["state"] == "complete"
    assert record["recovered"] is True


def test_indeterminate_attachment_creation_fails_closed_without_retrying_write(
    monkeypatch, attachment_state
):
    library = {"library_id": "0", "library_type": "user"}
    payload = b"uncertain attachment"
    prepared, _ = _ready_upload(payload, library)
    request = {
        "library": library,
        "parent_item_key": "PARENT01",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "filename": "paper.pdf",
        "title": "paper.pdf",
    }
    service.save_idempotency_record(
        "indeterminate-key",
        {
            "state": "pending",
            "request": request,
            "expected_md5": hashlib.md5(payload).hexdigest(),  # noqa: S324
            "baseline_attachment_keys": [],
        },
    )

    class WriteZotero:
        def children(self, _parent_key, **_kwargs):
            return []

        def attachment_both(self, *_args, **_kwargs):
            raise AssertionError("indeterminate retries must not issue a write")

    zot = WriteZotero()
    monkeypatch.setattr(
        "zotero_mcp.tools.attachments._helpers._get_write_client",
        lambda _ctx: (zot, zot),
    )
    monkeypatch.setattr(
        "zotero_mcp.tools.attachments._client.get_current_library",
        lambda: library,
    )

    result = put_attachment(
        parent_item_key="PARENT01",
        upload_id=prepared["upload_id"],
        idempotency_key="indeterminate-key",
        ctx=DummyContext(),
    )

    assert "indeterminate outcome" in result
    assert "No duplicate was created" in result


def test_expired_upload_gc_is_bounded_and_preserves_live_uploads(
    monkeypatch, attachment_state
):
    library = {"library_id": "0", "library_type": "user"}
    expired = []
    for index in range(3):
        prepared, _ = _ready_upload(f"expired-{index}".encode(), library)
        expired.append(prepared["upload_id"])
    live, _ = _ready_upload(b"live", library)
    for upload_id in expired:
        manifest_path = service.upload_directory(upload_id) / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["expires_at"] = 10
        manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setenv("ZOTERO_MCP_ATTACHMENT_GC_DELETE_LIMIT", "2")

    result = service.garbage_collect_uploads(now=20)

    assert result["deleted"] == 2
    remaining = [
        value for value in expired if service.upload_directory(value).exists()
    ]
    assert len(remaining) == 1
    for value in expired:
        if value not in remaining:
            assert not service.upload_lock_path(value).exists()
    assert service.upload_directory(live["upload_id"]).exists()


def test_expired_upload_gc_rotates_past_live_directories(
    monkeypatch, attachment_state
):
    uploads = service._state_root() / "uploads"
    for upload_id, expires_at in (("AAA", 100), ("BBB", 100), ("CCC", 10)):
        directory = uploads / upload_id
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "upload_id": upload_id,
                    "filename": "payload.bin",
                    "expires_at": expires_at,
                    "status": "ready",
                }
            )
        )
    monkeypatch.setenv("ZOTERO_MCP_ATTACHMENT_GC_SCAN_LIMIT", "1")
    monkeypatch.setenv("ZOTERO_MCP_ATTACHMENT_GC_DELETE_LIMIT", "1")

    assert service.garbage_collect_uploads(now=20)["deleted"] == 0
    assert service.garbage_collect_uploads(now=20)["deleted"] == 0
    assert service.garbage_collect_uploads(now=20)["deleted"] == 1
    assert not (uploads / "CCC").exists()


def test_upload_gc_skips_corrupt_expiry_without_blocking_new_uploads(
    monkeypatch, attachment_state
):
    directory = service._state_root() / "uploads" / "BROKEN"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text('{"expires_at": null}')
    monkeypatch.setenv("ZOTERO_MCP_ATTACHMENT_GC_SCAN_LIMIT", "1")

    assert service.garbage_collect_uploads(now=20) == {"scanned": 1, "deleted": 0}
    prepared = service.prepare_upload(
        "paper.pdf",
        "application/pdf",
        3,
        "0" * 64,
        library={"library_id": "0", "library_type": "user"},
    )
    assert prepared["status"] == "prepared"


def test_inline_and_resource_limits_are_configurable_and_bounded(
    monkeypatch, attachment_state
):
    assert service.max_inline_size() == 1024 * 1024
    assert service.max_resource_size() == 8 * 1024 * 1024

    monkeypatch.setenv("ZOTERO_MCP_ATTACHMENT_INLINE_MAX_BYTES", "2097152")
    monkeypatch.setenv("ZOTERO_MCP_ATTACHMENT_RESOURCE_MAX_BYTES", "16777216")
    assert service.max_inline_size() == 2 * 1024 * 1024
    assert service.max_resource_size() == 16 * 1024 * 1024

    monkeypatch.setenv("ZOTERO_MCP_ATTACHMENT_INLINE_MAX_BYTES", str(10**12))
    monkeypatch.setenv("ZOTERO_MCP_ATTACHMENT_RESOURCE_MAX_BYTES", str(10**12))
    assert service.max_inline_size() == service.HARD_MAX_INLINE_SIZE
    assert service.max_resource_size() == service.HARD_MAX_RESOURCE_SIZE


def test_binary_resource_rejects_large_in_memory_payload(
    monkeypatch, attachment_state
):
    monkeypatch.setenv("ZOTERO_MCP_ATTACHMENT_RESOURCE_MAX_BYTES", "4")
    monkeypatch.setattr(
        "zotero_mcp.resources._client.get_zotero_client",
        lambda: SimpleNamespace(item=lambda _key: _attachment()),
    )
    monkeypatch.setattr(
        "zotero_mcp.resources._client.get_local_zotero_client",
        lambda: object(),
    )
    monkeypatch.setattr(
        "zotero_mcp.resources._client.get_web_zotero_client",
        lambda: object(),
    )

    def download(_key, destination, filename, **_kwargs):
        path = Path(destination) / filename
        path.write_bytes(b"12345")
        return AttachmentDownloadResult(path=path, source="test", errors=[])

    monkeypatch.setattr(
        "zotero_mcp.resources._client.download_attachment_file",
        download,
    )

    with pytest.raises(ValueError, match="in-memory MCP resource limit"):
        resources.attachment_binary_resource("ATTACH01")


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


def test_reparent_confirmation_cannot_cover_unpreviewed_metadata_changes(
    monkeypatch, attachment_state
):
    updates = []
    zot = SimpleNamespace(
        item=lambda _key: _attachment(version=7, md5="abc"),
        update_item=lambda item: updates.append(item) or True,
    )
    monkeypatch.setattr(
        "zotero_mcp.tools.attachments._helpers._get_write_client",
        lambda _ctx: (zot, zot),
    )
    preview = json.loads(
        prepare_attachment_change(
            action="reparent",
            attachment_key="ATTACH01",
            parent_item_key="PARENT02",
            ctx=DummyContext(),
        )
    )

    result = update_attachment(
        attachment_key="ATTACH01",
        parent_item_key="PARENT02",
        title="Unpreviewed title",
        operation_id=preview["operation_id"],
        confirmation_token=preview["confirmation_token"],
        ctx=DummyContext(),
    )

    assert "must be committed separately" in result
    assert updates == []
