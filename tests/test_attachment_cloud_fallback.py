"""Regression tests for cloud attachment fallback while local mode is active."""

from pathlib import Path

from conftest import DummyContext

from zotero_mcp.client import AttachmentDetails, AttachmentDownloadResult
from zotero_mcp.tools import read_pdf, retrieval


class _MetadataClient:
    def item(self, item_key):
        return {
            "key": item_key,
            "data": {
                "key": item_key,
                "itemType": "journalArticle",
                "title": "Cloud fallback paper",
            },
        }

    def fulltext_item(self, _attachment_key):
        raise RuntimeError("not indexed")

    def children(self, _item_key):
        return [
            {
                "key": "ATTACH01",
                "data": {
                    "key": "ATTACH01",
                    "itemType": "attachment",
                    "parentItem": "PARENT01",
                    "title": "PDF",
                    "filename": "paper.pdf",
                    "contentType": "application/pdf",
                },
            }
        ]


def _unavailable_reader(**_kwargs):
    raise OSError("local snapshot unavailable")


def test_fulltext_local_mode_retains_web_api_file_fallback(monkeypatch):
    metadata_client = _MetadataClient()
    local_client = object()
    web_client = object()
    captured = {}

    def download(attachment_key, destination_dir, filename, **kwargs):
        captured.update(kwargs)
        path = Path(destination_dir) / filename
        path.write_text("cloud attachment", encoding="utf-8")
        return AttachmentDownloadResult(path=path, source="Web API", errors=[])

    monkeypatch.setattr(retrieval._client, "get_zotero_client", lambda: metadata_client)
    monkeypatch.setattr(retrieval._client, "get_local_zotero_client", lambda: local_client)
    monkeypatch.setattr(retrieval._client, "get_web_zotero_client", lambda: web_client)
    monkeypatch.setattr(retrieval._client, "download_attachment_file", download)
    monkeypatch.setattr(
        retrieval._attachments,
        "document_text_from_file",
        lambda _path, _kind: "Converted cloud text",
    )
    monkeypatch.setattr(retrieval._utils, "is_local_mode", lambda: True)
    monkeypatch.setattr("zotero_mcp.local_db.LocalZoteroReader", _unavailable_reader)

    result = retrieval.get_item_fulltext(item_key="PARENT01", ctx=DummyContext())

    assert "Converted cloud text" in result
    assert captured == {"local_client": local_client, "web_client": web_client}


def test_pdf_pages_local_mode_retains_web_api_file_fallback(monkeypatch):
    metadata_client = _MetadataClient()
    local_client = object()
    web_client = object()
    captured = {}

    def download(attachment_key, destination_dir, filename, **kwargs):
        captured.update(kwargs)
        path = Path(destination_dir) / filename
        path.write_bytes(b"%PDF-1.4 cloud attachment")
        return AttachmentDownloadResult(path=path, source="Web API", errors=[])

    monkeypatch.setattr(read_pdf._client, "get_zotero_client", lambda: metadata_client)
    monkeypatch.setattr(read_pdf._client, "get_local_zotero_client", lambda: local_client)
    monkeypatch.setattr(read_pdf._client, "get_web_zotero_client", lambda: web_client)
    monkeypatch.setattr(read_pdf._client, "download_attachment_file", download)
    monkeypatch.setattr(
        read_pdf._client,
        "get_pdf_attachment_details",
        lambda *_args: AttachmentDetails(
            key="ATTACH01",
            title="PDF",
            filename="paper.pdf",
            content_type="application/pdf",
        ),
    )
    monkeypatch.setattr(read_pdf._utils, "is_local_mode", lambda: True)
    monkeypatch.setattr("zotero_mcp.local_db.LocalZoteroReader", _unavailable_reader)

    resolved = read_pdf._get_pdf_path("PARENT01", DummyContext())

    assert resolved is not None
    assert resolved[2] == "ATTACH01"
    assert captured == {"local_client": local_client, "web_client": web_client}
    read_pdf._cleanup_path(resolved[0])
