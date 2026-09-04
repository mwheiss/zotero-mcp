"""Coherent attachment discovery, text, binary, and mutation tools."""

from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import time
from typing import Literal

from pyzotero.zotero import build_url

from zotero_mcp import attachment_service as service
from zotero_mcp import client as _client
from zotero_mcp._app import mcp
from zotero_mcp._context import Context, context_error, context_info
from zotero_mcp.tools import _helpers
from zotero_mcp.tools.retrieval import get_item_fulltext

READ_ONLY = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}
WRITE = {
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}
DESTRUCTIVE = {
    "readOnlyHint": False,
    "destructiveHint": True,
    "idempotentHint": True,
    "openWorldHint": False,
}


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _attachment_item(zot, attachment_key: str) -> dict:
    item = zot.item(attachment_key)
    if not item or (item.get("data", {}) or {}).get("itemType") != "attachment":
        raise ValueError(f"No attachment found with key `{attachment_key}`")
    return item


def _created_attachment_key(result) -> str | None:
    if not isinstance(result, dict):
        return None
    for status in ("success", "unchanged"):
        value = result.get(status)
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, dict) and entry.get("key"):
                    return str(entry["key"])
        if isinstance(value, dict):
            for entry in value.values():
                if isinstance(entry, str):
                    return entry
                if isinstance(entry, dict) and entry.get("key"):
                    return str(entry["key"])
    for status in ("successful",):
        value = result.get(status)
        if isinstance(value, dict):
            for entry in value.values():
                if isinstance(entry, dict) and entry.get("key"):
                    return str(entry["key"])
    return None


def _consume_staged_upload(upload_id: str) -> str | None:
    """Best-effort cleanup after Zotero has durably accepted an upload."""
    try:
        service.consume_upload(upload_id, already_locked=True)
    except Exception as exc:
        return (
            "Zotero accepted the attachment, but staged-upload cleanup failed: "
            f"{exc}"
        )
    return None


def _file_md5(path) -> str:
    """Return Zotero's content identity for a staged attachment."""
    digest = hashlib.md5()  # noqa: S324 - Zotero exposes attachment MD5 values
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parent_attachment_inventory(zot, parent_item_key: str) -> dict[str, dict]:
    """Return every live direct attachment keyed by Zotero item key."""
    inventory: dict[str, dict] = {}
    for child in _helpers._paginate(zot.children, parent_item_key):
        data = child.get("data", {}) or {}
        if data.get("itemType") != "attachment":
            continue
        key = str(child.get("key") or data.get("key") or "")
        if key:
            inventory[key] = child
    return inventory


def _recover_pending_attachment(zot, record: dict) -> str | None:
    """Resolve a post-crash creation from Zotero without issuing another write."""
    request = record.get("request") or {}
    parent_key = str(request.get("parent_item_key") or "")
    filename = str(request.get("filename") or "")
    expected_md5 = str(record.get("expected_md5") or "").casefold()
    baseline = {str(key) for key in record.get("baseline_attachment_keys") or []}
    if not parent_key or not filename or not expected_md5:
        return None

    matches: list[str] = []
    for key, child in _parent_attachment_inventory(zot, parent_key).items():
        if key in baseline:
            continue
        data = child.get("data", {}) or {}
        if not data.get("filename") or not data.get("md5"):
            try:
                data = (_attachment_item(zot, key).get("data", {}) or {})
            except Exception:
                continue
        if (
            str(data.get("filename") or "") == filename
            and str(data.get("md5") or "").casefold() == expected_md5
        ):
            matches.append(key)
    return matches[0] if len(matches) == 1 else None


@mcp.tool(
    name="zotero_list_attachments",
    annotations=READ_ONLY,
    description=(
        "List every direct attachment of one Zotero item with stable keys, "
        "filename, MIME type, size, version, link mode, URL, trash state, and "
        "binary read/write capability. This never downloads or extracts a file. "
        "Pass an attachment key itself to inspect that one attachment."
    ),
)
def list_attachments(item_key: str, *, ctx: Context) -> str:
    try:
        zot = _client.get_zotero_client()
        item = zot.item(item_key)
        if not item:
            return f"Error: No item found with key `{item_key}`"
        if (item.get("data", {}) or {}).get("itemType") == "attachment":
            attachments = [item]
        else:
            attachments = [
                child
                for child in _helpers._paginate(zot.children, item_key)
                if (child.get("data", {}) or {}).get("itemType") == "attachment"
            ]
        return _json(
            {
                "item_key": item_key,
                "count": len(attachments),
                "attachments": [service.attachment_descriptor(value) for value in attachments],
            }
        )
    except Exception as exc:
        context_error(ctx, f"Could not list attachments: {exc}")
        return f"Error: Could not list attachments: {exc}"


@mcp.tool(
    name="zotero_get_document_text",
    annotations=READ_ONLY,
    description=(
        "Return the selected model-facing document text itself, never a path or "
        "binary file. With no attachment_key, use the same BetterIssa-first local "
        "selection policy as semantic indexing. With attachment_key, process only "
        "that exact child. The response identifies the selected source and includes "
        "the complete available text; PDFs may be page-capped."
    ),
)
def get_document_text(
    item_key: str,
    attachment_key: str | None = None,
    *,
    ctx: Context,
) -> str:
    return get_item_fulltext(
        item_key=item_key,
        attachment_key=attachment_key,
        ctx=ctx,
    )


@mcp.tool(
    name="zotero_get_attachment",
    annotations=READ_ONLY,
    description=(
        "Get an exact attachment's binary-delivery descriptor. Returns metadata, "
        "an MCP resource URI, and a short-lived signed streaming download URL when "
        "a public base URL is configured. inline=True embeds base64 only within "
        "the configured limit (1 MiB by default, adjustable up to 32 MiB). It "
        "never converts the binary to document text."
    ),
)
def get_attachment(
    attachment_key: str,
    inline: bool = False,
    *,
    ctx: Context,
) -> str:
    try:
        zot = _client.get_zotero_client()
        item = _attachment_item(zot, attachment_key)
        descriptor = service.attachment_descriptor(item)
        if not descriptor["binary_readable"]:
            return _json({"attachment": descriptor, "error": "Attachment has no stored binary"})
        token = service.make_token(
            "download",
            {
                "attachment_key": attachment_key,
                "filename": descriptor["filename"] or f"{attachment_key}.bin",
                "content_type": descriptor["content_type"],
                "library": _client.get_current_library(),
            },
            ttl=900,
        )
        result = {
            "attachment": descriptor,
            "resource_uri": f"zotero://attachments/{attachment_key}/content",
            "download_url": service.public_transfer_url("download", token),
            "expires_in_seconds": 900,
        }
        if inline:
            with tempfile.TemporaryDirectory() as temporary:
                downloaded = _client.download_attachment_file(
                    attachment_key,
                    temporary,
                    descriptor["filename"] or f"{attachment_key}.bin",
                    local_client=_client.get_local_zotero_client(),
                    web_client=_client.get_web_zotero_client(),
                )
                if not downloaded.path:
                    raise ValueError("; ".join(downloaded.errors) or "binary unavailable")
                inline_limit = service.max_inline_size()
                if downloaded.path.stat().st_size > inline_limit:
                    raise ValueError(
                        f"inline binary is limited to {inline_limit} bytes; "
                        "use download_url or raise "
                        "ZOTERO_MCP_ATTACHMENT_INLINE_MAX_BYTES"
                    )
                result["data_base64"] = base64.b64encode(downloaded.path.read_bytes()).decode("ascii")
        return _json(result)
    except Exception as exc:
        context_error(ctx, f"Could not retrieve attachment: {exc}")
        return f"Error: Could not retrieve attachment: {exc}"


@mcp.tool(
    name="zotero_prepare_attachment_upload",
    annotations=WRITE,
    description=(
        "Create a short-lived staged binary upload for a remote MCP client. "
        "Supply filename, MIME type, exact byte size, and SHA-256. The result "
        "contains a signed URL accepting one raw HTTP PUT. After upload, call "
        "zotero_put_attachment with the upload_id. Requires write-admin access."
    ),
)
def prepare_attachment_upload(
    filename: str,
    content_type: str,
    size: int,
    sha256: str,
    *,
    ctx: Context,
) -> str:
    try:
        return _json(
            service.prepare_upload(
                filename,
                content_type,
                size,
                sha256,
                library=_client.get_current_library(),
            )
        )
    except Exception as exc:
        return f"Error: Could not prepare attachment upload: {exc}"


@mcp.tool(
    name="zotero_prepare_attachment_change",
    annotations=WRITE,
    description=(
        "Prepare and preview a destructive attachment operation. action is "
        "'replace', 'trash', or 'reparent'. The returned short-lived operation ID "
        "and confirmation token are bound to the exact attachment version and "
        "arguments. Preparing does not mutate Zotero. Requires write-admin access."
    ),
)
def prepare_attachment_change(
    action: Literal["replace", "trash", "reparent"],
    attachment_key: str,
    upload_id: str | None = None,
    parent_item_key: str | None = None,
    *,
    ctx: Context,
) -> str:
    try:
        _, write_zot = _helpers._get_write_client(ctx)
        item = _attachment_item(write_zot, attachment_key)
        descriptor = service.attachment_descriptor(item)
        details = {
            "action": action,
            "attachment_key": attachment_key,
            "expected_version": descriptor["version"],
            "expected_md5": descriptor["md5"],
            "upload_id": upload_id or "",
            "parent_item_key": parent_item_key or "",
        }
        if action == "replace":
            if not upload_id:
                raise ValueError("replace requires upload_id")
            service.read_upload(upload_id)
        if action == "reparent" and not parent_item_key:
            raise ValueError("reparent requires parent_item_key")
        prepared = service.prepare_operation(action, details)
        return _json(
            {
                "operation_id": prepared["operation_id"],
                "confirmation_token": prepared["confirmation_token"],
                "expires_at": prepared["expires_at"],
                "preview": details,
                "attachment": descriptor,
            }
        )
    except Exception as exc:
        return f"Error: Could not prepare attachment change: {exc}"


@mcp.tool(
    name="zotero_put_attachment",
    annotations=DESTRUCTIVE,
    description=(
        "Commit a staged binary as a new or replacement Zotero attachment. "
        "For creation, provide parent_item_key and upload_id; idempotency_key is "
        "required. Creation records a durable pre-write intent and recovers a "
        "completed post-crash write by comparing the new attachment against its "
        "pre-write parent inventory; an ambiguous outcome fails closed rather than "
        "creating a duplicate. For replacement, also provide attachment_key plus the operation "
        "ID and confirmation token from zotero_prepare_attachment_change. Replacement "
        "preserves the attachment key and rechecks its version/hash. Requires "
        "write-admin access and client approval."
    ),
)
def put_attachment(
    parent_item_key: str,
    upload_id: str,
    attachment_key: str | None = None,
    title: str | None = None,
    idempotency_key: str | None = None,
    operation_id: str | None = None,
    confirmation_token: str | None = None,
    *,
    ctx: Context,
) -> str:
    try:
        with service.upload_lock(upload_id):
            _, write_zot = _helpers._get_write_client(ctx)
            manifest, path = service.read_upload(
                upload_id,
                allow_consumed=not attachment_key and bool(idempotency_key),
            )
            current_library = _client.get_current_library()
            if manifest.get("library") != current_library:
                raise ValueError("Upload belongs to a different Zotero library")
            if attachment_key:
                item = _attachment_item(write_zot, attachment_key)
                descriptor = service.attachment_descriptor(item)
                details = {
                    "action": "replace",
                    "attachment_key": attachment_key,
                    "expected_version": descriptor["version"],
                    "expected_md5": descriptor["md5"],
                    "upload_id": upload_id,
                    "parent_item_key": "",
                }
                if not operation_id or not confirmation_token:
                    raise ValueError(
                        "Replacement requires a prepared operation and confirmation token"
                    )
                service.consume_operation(
                    operation_id,
                    confirmation_token,
                    action="replace",
                    details=details,
                )
                result = write_zot.upload_attachments(
                    [
                        {
                            "key": attachment_key,
                            "filename": path.name,
                            "md5": descriptor["md5"] or None,
                        }
                    ],
                    basedir=str(path.parent),
                )
                if result.get("failure"):
                    raise RuntimeError(
                        f"Zotero rejected replacement: {result['failure']}"
                    )
                output_key = attachment_key
                action = "replaced"
            else:
                if not idempotency_key:
                    raise ValueError("Creating an attachment requires idempotency_key")
                with service.idempotency_lock(idempotency_key):
                    prior = service.idempotency_record(idempotency_key)
                    request_identity = {
                        "library": current_library,
                        "parent_item_key": parent_item_key,
                        "sha256": manifest["sha256"],
                        "filename": manifest["filename"],
                        "title": title or manifest["filename"],
                    }
                    if prior:
                        prior_request = prior.get("request") or {}
                        legacy_request = dict(request_identity)
                        legacy_request.pop("title", None)
                        if (
                            prior_request != request_identity
                            and prior_request != legacy_request
                        ):
                            raise ValueError(
                                "idempotency_key was already used for a different attachment"
                            )
                        prior_key = str(prior.get("attachment_key", ""))
                        if not prior_key:
                            prior_key = _recover_pending_attachment(
                                write_zot, prior
                            ) or ""
                            if not prior_key:
                                raise ValueError(
                                    "A previous attachment creation with this "
                                    "idempotency_key has an indeterminate outcome. "
                                    "No duplicate was created. Retry later with the "
                                    "same key; if it remains unresolved, inspect the "
                                    "parent attachments before choosing a new key."
                                )
                            service.save_idempotency_record(
                                idempotency_key,
                                {
                                    **prior,
                                    "state": "complete",
                                    "attachment_key": prior_key,
                                    "completed_at": int(time.time()),
                                    "recovered": True,
                                },
                            )
                        refreshed = _attachment_item(write_zot, prior_key)
                        cleanup_warning = _consume_staged_upload(upload_id)
                        response = {
                            "status": "already-created",
                            "attachment": service.attachment_descriptor(refreshed),
                            "sha256": manifest["sha256"],
                        }
                        if cleanup_warning:
                            response["warnings"] = [cleanup_warning]
                        return _json(response)
                    if manifest.get("status") == "consumed":
                        raise ValueError(
                            "Attachment upload was already consumed by another operation"
                        )
                    parent = write_zot.item(parent_item_key)
                    if not parent or (parent.get("data", {}) or {}).get(
                        "itemType"
                    ) == "attachment":
                        raise ValueError(
                            "parent_item_key must identify a bibliographic item"
                        )
                    expected_md5 = _file_md5(path)
                    baseline_attachment_keys = sorted(
                        _parent_attachment_inventory(
                            write_zot, parent_item_key
                        )
                    )
                    service.save_idempotency_record(
                        idempotency_key,
                        {
                            "state": "pending",
                            "request": request_identity,
                            "expected_md5": expected_md5,
                            "baseline_attachment_keys": baseline_attachment_keys,
                            "started_at": int(time.time()),
                        },
                    )
                    result = write_zot.attachment_both(
                        [(title or manifest["filename"], str(path))],
                        parentid=parent_item_key,
                    )
                    output_key = _created_attachment_key(result)
                    if not output_key:
                        if isinstance(result, dict) and result.get("failure"):
                            # A structured one-file failure is definitive. The
                            # local writer also compensates its empty shell.
                            service.clear_idempotency_record(idempotency_key)
                        raise RuntimeError(
                            f"Zotero did not create the attachment: {result}"
                        )
                    service.save_idempotency_record(
                        idempotency_key,
                        {
                            "state": "complete",
                            "request": request_identity,
                            "expected_md5": expected_md5,
                            "baseline_attachment_keys": baseline_attachment_keys,
                            "attachment_key": output_key,
                            "completed_at": int(time.time()),
                        },
                    )
                action = "created"
            refreshed = _attachment_item(write_zot, output_key)
            cleanup_warning = _consume_staged_upload(upload_id)
            response = {
                "status": action,
                "attachment": service.attachment_descriptor(refreshed),
                "sha256": manifest["sha256"],
                "write_target": getattr(
                    write_zot, "local_endpoint_role", "web-api"
                ),
            }
            if cleanup_warning:
                response["warnings"] = [cleanup_warning]
            return _json(response)
    except Exception as exc:
        context_error(ctx, f"Could not put attachment: {exc}")
        return f"Error: Could not put attachment: {exc}"


@mcp.tool(
    name="zotero_update_attachment",
    annotations=DESTRUCTIVE,
    description=(
        "Update attachment metadata without replacing bytes. Supports title, URL, "
        "content type, and parent item. Reparenting requires a matching prepared "
        "operation and confirmation token. Requires write-admin access."
    ),
)
def update_attachment(
    attachment_key: str,
    title: str | None = None,
    url: str | None = None,
    content_type: str | None = None,
    parent_item_key: str | None = None,
    operation_id: str | None = None,
    confirmation_token: str | None = None,
    *,
    ctx: Context,
) -> str:
    try:
        _, write_zot = _helpers._get_write_client(ctx)
        item = _attachment_item(write_zot, attachment_key)
        descriptor = service.attachment_descriptor(item)
        data = item.get("data", {})
        if parent_item_key is not None and parent_item_key != descriptor["parent_item_key"]:
            unconfirmed_metadata_changes = [
                field
                for field, value in {
                    "title": title,
                    "url": url,
                    "contentType": content_type,
                }.items()
                if value is not None and data.get(field) != value
            ]
            if unconfirmed_metadata_changes:
                raise ValueError(
                    "Reparenting must be committed separately from title, URL, or "
                    "content-type changes so the confirmation covers the complete mutation"
                )
            details = {
                "action": "reparent",
                "attachment_key": attachment_key,
                "expected_version": descriptor["version"],
                "expected_md5": descriptor["md5"],
                "upload_id": "",
                "parent_item_key": parent_item_key,
            }
            if not operation_id or not confirmation_token:
                raise ValueError("Reparenting requires a prepared operation and confirmation token")
            service.consume_operation(
                operation_id,
                confirmation_token,
                action="reparent",
                details=details,
            )
        changes = {
            "title": title,
            "url": url,
            "contentType": content_type,
            "parentItem": parent_item_key,
        }
        changed = {}
        for field, value in changes.items():
            if value is not None and data.get(field) != value:
                changed[field] = {"from": data.get(field), "to": value}
                data[field] = value
        if not changed:
            return _json({"status": "unchanged", "attachment": descriptor})
        response = write_zot.update_item(item)
        if not _helpers._handle_write_response(response, ctx):
            raise RuntimeError("Zotero rejected the metadata update")
        return _json(
            {
                "status": "updated",
                "changes": changed,
                "attachment": service.attachment_descriptor(_attachment_item(write_zot, attachment_key)),
            }
        )
    except Exception as exc:
        return f"Error: Could not update attachment: {exc}"


@mcp.tool(
    name="zotero_set_attachment_trashed",
    annotations=DESTRUCTIVE,
    description=(
        "Move one attachment to Zotero Trash or restore it. Trashing requires a "
        "matching operation ID and confirmation token from "
        "zotero_prepare_attachment_change(action='trash'). Restore is reversible "
        "and needs only write-admin access. Permanent deletion is not exposed."
    ),
)
def set_attachment_trashed(
    attachment_key: str,
    trashed: bool,
    operation_id: str | None = None,
    confirmation_token: str | None = None,
    *,
    ctx: Context,
) -> str:
    try:
        _, write_zot = _helpers._get_write_client(ctx)
        item = _attachment_item(write_zot, attachment_key)
        descriptor = service.attachment_descriptor(item)
        if trashed:
            details = {
                "action": "trash",
                "attachment_key": attachment_key,
                "expected_version": descriptor["version"],
                "expected_md5": descriptor["md5"],
                "upload_id": "",
                "parent_item_key": "",
            }
            if not operation_id or not confirmation_token:
                raise ValueError("Trashing requires a prepared operation and confirmation token")
            service.consume_operation(
                operation_id,
                confirmation_token,
                action="trash",
                details=details,
            )
        url = build_url(
            write_zot.endpoint,
            f"/{write_zot.library_type}/{write_zot.library_id}/items/{attachment_key}",
        )
        response = write_zot.client.patch(
            url=url,
            headers={"If-Unmodified-Since-Version": str(descriptor["version"])},
            content=json.dumps({"deleted": 1 if trashed else 0}),
        )
        if response.status_code not in (200, 204):
            raise RuntimeError(f"Zotero returned HTTP {response.status_code}: {response.text[:200]}")
        context_info(ctx, f"Attachment {attachment_key} trash state changed to {trashed}")
        return _json({"status": "trashed" if trashed else "restored", "attachment_key": attachment_key})
    except Exception as exc:
        return f"Error: Could not change attachment trash state: {exc}"
