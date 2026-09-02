"""Signed HTTP data plane for attachment downloads and staged uploads."""

from __future__ import annotations

import hashlib
import shutil

from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response

from zotero_mcp import attachment_service as service
from zotero_mcp import client as _client
from zotero_mcp._app import mcp


def _error(message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


@mcp.custom_route(
    "/mcp/attachments/download/{token}",
    methods=["GET", "HEAD"],
    include_in_schema=False,
)
async def download_attachment(request: Request) -> Response:
    try:
        claims = service.verify_token(request.path_params["token"], "download")
        attachment_key = str(claims["attachment_key"])
        filename = service.safe_filename(str(claims.get("filename") or f"{attachment_key}.bin"))
    except (ValueError, KeyError) as exc:
        return _error(str(exc), 403)

    temporary = service._state_root() / "downloads" / str(claims["jti"])
    temporary.mkdir(parents=True, mode=0o700)
    try:
        result = _client.download_attachment_file(
            attachment_key,
            temporary,
            filename,
            local_client=_client.get_local_zotero_client(),
            web_client=_client.get_web_zotero_client(),
        )
        if not result.path or not result.path.exists():
            shutil.rmtree(temporary, ignore_errors=True)
            return _error(
                "Attachment binary is unavailable: " + "; ".join(result.errors),
                404,
            )
        return FileResponse(
            result.path,
            filename=filename,
            media_type=str(claims.get("content_type") or "application/octet-stream"),
            headers={"Cache-Control": "private, no-store"},
            background=BackgroundTask(shutil.rmtree, temporary, ignore_errors=True),
        )
    except Exception as exc:
        shutil.rmtree(temporary, ignore_errors=True)
        return _error(f"Attachment download failed: {exc}", 502)


@mcp.custom_route(
    "/mcp/attachments/upload/{token}",
    methods=["PUT"],
    include_in_schema=False,
)
async def upload_attachment(request: Request) -> Response:
    try:
        claims = service.verify_token(request.path_params["token"], "upload")
        manifest, target = service.read_upload(
            str(claims["upload_id"]), require_ready=False
        )
    except (ValueError, KeyError) as exc:
        return _error(str(exc), 403)
    if manifest.get("status") == "ready":
        return _error("Attachment upload is already complete", 409)

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    digest = hashlib.sha256()
    received = 0
    try:
        with partial.open("wb") as handle:
            async for chunk in request.stream():
                received += len(chunk)
                if received > service.max_upload_size() or received > int(manifest["size"]):
                    raise ValueError("Attachment upload exceeds the declared size")
                digest.update(chunk)
                handle.write(chunk)
            handle.flush()
        partial.replace(target)
        try:
            service.mark_upload_ready(
                str(manifest["upload_id"]), size=received, sha256=digest.hexdigest()
            )
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return JSONResponse(
            {
                "upload_id": manifest["upload_id"],
                "status": "ready",
                "size": received,
                "sha256": digest.hexdigest(),
            },
            status_code=201,
        )
    except ValueError as exc:
        partial.unlink(missing_ok=True)
        return _error(str(exc), 400)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        return _error(f"Attachment upload failed: {exc}", 500)
