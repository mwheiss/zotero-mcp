"""Tests for retrying fulltext extraction on previously-failed items.

A "failed" has_fulltext marker must clear when the item's attachment set
changes (e.g. a PDF is attached via zotero_attach_file to an item that was
indexed metadata-only). Attaching a file does NOT bump the parent's
dateModified, so the date check alone can never trigger the retry.
"""

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

from zotero_mcp import semantic_search

DATE_MODIFIED = "2026-07-02 01:01:48"


class FakeItem:
    def __init__(self):
        self.item_id = 1
        self.key = "ITEMKEY1"
        self.item_type = "journalArticle"
        self.title = "Calibration"
        self.abstract = ""
        self.extra = ""
        self.doi = None
        self.notes = None
        self.creators = "Kim, A."
        self.date_added = "2026-07-01 00:00:00"
        self.date_modified = DATE_MODIFIED
        self.fulltext = None
        self.fulltext_source = None


class FakeReader:
    """Minimal LocalZoteroReader stand-in for the extraction scan."""

    def __init__(
        self,
        *args,
        attachments=(),
        attachment_signature=None,
        extract_result=("extracted text", "pdf"),
        extraction_details=None,
        **kwargs,
    ):
        self._attachments = list(attachments)
        self._attachment_signature = (
            attachment_signature
            if attachment_signature is not None
            else ",".join(sorted(row[0] for row in self._attachments))
        )
        self._extract_result = extract_result
        self._extraction_details = extraction_details
        self.extract_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_all_item_keys(self):
        return {"ITEMKEY1"}

    def get_items_with_text(self, limit=None, include_fulltext=False, key_filter=None, collection_keys=None):
        return [FakeItem()]

    def get_fulltext_meta_for_item(self, item_id, allowed_attachment_keys=None):
        return [
            list(row)
            for row in self._attachments
            if allowed_attachment_keys is None or row[0] in allowed_attachment_keys
        ]

    def get_attachment_signature(self, item_id, allowed_attachment_keys=None):
        if allowed_attachment_keys is not None:
            return ",".join(sorted(allowed_attachment_keys))
        return self._attachment_signature

    def extract_fulltext_for_item(self, item_id, allowed_attachment_keys=None):
        self.extract_calls += 1
        if allowed_attachment_keys == set():
            return None
        self.last_extraction_details = self._extraction_details
        return self._extract_result


class FakeChromaClient:
    def __init__(self, stored_metadata):
        self.embedding_max_tokens = 8000
        self._stored = stored_metadata

    def get_document_metadata(self, key):
        return self._stored

    def get_existing_ids(self, ids):
        return set()

    def upsert_documents(self, documents, metadatas, ids):
        pass

    def reset_collection(self):
        pass


def _run_scan(
    monkeypatch,
    stored_metadata,
    attachments,
    *,
    attachment_signature=None,
    extract_result=("extracted text", "pdf"),
    extraction_details=None,
    retry_failed_fulltext=False,
    publication_date="1952",
    api_attachment_keys=None,
):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: True)

    reader = FakeReader(
        attachments=attachments,
        attachment_signature=attachment_signature,
        extract_result=extract_result,
        extraction_details=extraction_details,
    )
    monkeypatch.setattr(
        semantic_search, "LocalZoteroReader", lambda *a, **kw: reader
    )

    chroma = FakeChromaClient(stored_metadata)
    search = semantic_search.ZoteroSemanticSearch(chroma_client=chroma)
    def fake_api_items(*args, **kwargs):
        if api_attachment_keys is not None:
            search._last_api_attachment_keys_by_parent = {
                "ITEMKEY1": set(api_attachment_keys)
            }
        return [
            {
                "key": "ITEMKEY1",
                "version": 1,
                "data": {
                    "key": "ITEMKEY1",
                    "itemType": "journalArticle",
                    "title": "Calibration",
                    "date": publication_date,
                    "dateModified": DATE_MODIFIED,
                },
            }
        ]

    monkeypatch.setattr(search, "_get_items_from_api", fake_api_items)

    items = search._get_items_from_source(
        fulltext=True,
        chroma_client=chroma,
        force_rebuild=False,
        retry_failed_fulltext=retry_failed_fulltext,
    )
    return items, reader


def test_failed_item_skipped_when_nothing_changed(monkeypatch):
    """failed marker + same date + same attachment set -> no retry."""
    stored = {
        "has_fulltext": "failed",
        "date_modified": DATE_MODIFIED,
        "attachment_keys": "",
    }
    items, reader = _run_scan(monkeypatch, stored, attachments=[])
    assert items == []
    assert reader.extract_calls == 0


def test_retry_failed_fulltext_bypasses_cached_failure(monkeypatch):
    stored = {
        "has_fulltext": "failed",
        "date_modified": DATE_MODIFIED,
        "attachment_keys": "ATTKEY1",
        "attachment_signature": "same-signature",
    }
    attachments = [("ATTKEY1", "storage:paper.pdf", "application/pdf")]

    items, reader = _run_scan(
        monkeypatch,
        stored,
        attachments,
        attachment_signature="same-signature",
        retry_failed_fulltext=True,
    )

    assert len(items) == 1
    assert reader.extract_calls == 1
    assert items[0]["data"]["fulltext"] == "extracted text"


def test_cached_failure_summary_uses_publication_year_and_explains_retry(
    monkeypatch,
    capsys,
):
    stored = {
        "has_fulltext": "failed",
        "date_modified": DATE_MODIFIED,
        "attachment_keys": "",
        "fulltext_error": "The PDF is encrypted.",
    }

    _run_scan(monkeypatch, stored, attachments=[])

    output = capsys.readouterr().err
    assert "Kim (1952)" in output
    assert "Kim (2026)" not in output
    assert "Reason: The PDF is encrypted." in output
    assert "--retry-failed-fulltext" in output


def test_new_extraction_failure_is_saved_with_actionable_reason(monkeypatch):
    stored = {
        "has_fulltext": "failed",
        "date_modified": DATE_MODIFIED,
        "attachment_keys": "",
    }

    items, reader = _run_scan(
        monkeypatch,
        stored,
        attachments=[],
        extract_result=None,
        retry_failed_fulltext=True,
    )

    assert reader.extract_calls == 1
    assert items[0]["data"]["fulltextError"] == (
        "No candidate full-text attachment was available."
    )
    metadata = semantic_search.ZoteroSemanticSearch(
        chroma_client=FakeChromaClient({})
    )._create_metadata(items[0])
    assert metadata["has_fulltext"] == "failed"
    assert metadata["fulltext_error"] == (
        "No candidate full-text attachment was available."
    )


def test_failed_item_retried_when_attachment_added(monkeypatch):
    """failed marker + same date, but a PDF was attached since -> retry."""
    stored = {
        "has_fulltext": "failed",
        "date_modified": DATE_MODIFIED,
        "attachment_keys": "",
    }
    attachments = [("ATTKEY1", "storage:paper.pdf", "application/pdf")]
    items, reader = _run_scan(monkeypatch, stored, attachments=attachments)
    assert len(items) == 1
    assert reader.extract_calls == 1
    data = items[0]["data"]
    assert data["fulltext"] == "extracted text"
    assert data["attachmentKeys"] == "ATTKEY1"
    # And the resulting metadata records the new attachment set
    metadata = semantic_search.ZoteroSemanticSearch(
        chroma_client=FakeChromaClient({})
    )._create_metadata(items[0])
    assert metadata["has_fulltext"] is True
    assert metadata["attachment_keys"] == "ATTKEY1"


def test_failed_item_retried_when_date_modified_changed(monkeypatch):
    """failed marker + same attachment set, but the item was edited -> retry.

    Pins the pre-existing dateModified trigger ("user replaces a bad PDF and
    the parent gets re-saved") so it survives the new attachment-set check.
    """
    stored = {
        "has_fulltext": "failed",
        "date_modified": "2026-06-30 12:00:00",  # differs from the item's
        "attachment_keys": "",
    }
    items, reader = _run_scan(monkeypatch, stored, attachments=[])
    assert len(items) == 1
    assert reader.extract_calls == 1


def test_legacy_failed_record_without_attachment_keys_retries_once(monkeypatch):
    """Records written before attachment_keys existed retry once, then converge."""
    stored = {
        "has_fulltext": "failed",
        "date_modified": DATE_MODIFIED,
        # no attachment_keys field (legacy)
    }
    attachments = [("ATTKEY1", "storage:paper.pdf", "application/pdf")]
    items, reader = _run_scan(monkeypatch, stored, attachments=attachments)
    assert len(items) == 1
    assert reader.extract_calls == 1


def test_successful_item_skipped_when_metadata_and_attachment_unchanged(monkeypatch):
    attachments = [("ATTKEY1", "storage:paper.pdf", "application/pdf")]
    stored = {
        "has_fulltext": True,
        "date_modified": DATE_MODIFIED,
        "attachment_keys": "ATTKEY1",
        "attachment_signature": "same-signature",
    }

    items, reader = _run_scan(
        monkeypatch,
        stored,
        attachments,
        attachment_signature="same-signature",
    )

    assert items == []
    assert reader.extract_calls == 0


def test_successful_item_reindexed_when_metadata_changes(monkeypatch):
    attachments = [("ATTKEY1", "storage:paper.pdf", "application/pdf")]
    stored = {
        "has_fulltext": True,
        "date_modified": "2026-06-30 12:00:00",
        "attachment_keys": "ATTKEY1",
        "attachment_signature": "same-signature",
    }

    items, reader = _run_scan(
        monkeypatch,
        stored,
        attachments,
        attachment_signature="same-signature",
    )

    assert len(items) == 1
    assert reader.extract_calls == 1


def test_successful_item_reindexed_when_attachment_content_changes(monkeypatch):
    attachments = [("ATTKEY1", "storage:paper.pdf", "application/pdf")]
    stored = {
        "has_fulltext": True,
        "date_modified": DATE_MODIFIED,
        "attachment_keys": "ATTKEY1",
        "attachment_signature": "old-signature",
    }

    items, reader = _run_scan(
        monkeypatch,
        stored,
        attachments,
        attachment_signature="new-signature",
    )

    assert len(items) == 1
    assert reader.extract_calls == 1
    assert items[0]["data"]["attachmentSignature"] == "new-signature"


def test_successful_item_drops_stale_fulltext_when_attachment_removed(monkeypatch):
    stored = {
        "has_fulltext": True,
        "date_modified": DATE_MODIFIED,
        "attachment_keys": "ATTKEY1",
        "attachment_signature": "old-signature",
    }

    items, reader = _run_scan(
        monkeypatch,
        stored,
        [],
        attachment_signature="empty-signature",
        extract_result=None,
    )

    assert len(items) == 1
    assert reader.extract_calls == 1
    assert items[0]["data"]["fulltext"] == ""
    assert items[0]["data"]["fulltext_attempted"] is True


def test_api_deleted_attachment_overrides_stale_local_snapshot(monkeypatch, capsys):
    stored = {
        "has_fulltext": True,
        "date_modified": DATE_MODIFIED,
        "attachment_keys": "WEBPAGE1",
        "attachment_signature": "WEBPAGE1",
    }

    items, reader = _run_scan(
        monkeypatch,
        stored,
        [("WEBPAGE1", "storage:page.html", "text/html")],
        extract_result=None,
        api_attachment_keys=set(),
    )

    assert reader.extract_calls == 1
    assert len(items) == 1
    assert items[0]["data"]["attachmentKeys"] == ""
    assert items[0]["data"]["fulltext"] == ""
    assert items[0]["data"]["fulltext_attempted"] is True
    assert "No candidate full-text attachment was available." in capsys.readouterr().err


def test_api_attachment_missing_from_snapshot_defers_item(monkeypatch, capsys):
    stored = {
        "has_fulltext": True,
        "date_modified": DATE_MODIFIED,
        "attachment_keys": "PDF1",
        "attachment_signature": "PDF1",
    }

    items, reader = _run_scan(
        monkeypatch,
        stored,
        [],
        api_attachment_keys={"PDF1"},
    )

    assert items == []
    assert reader.extract_calls == 0
    output = capsys.readouterr().err
    assert "deferred 1 item(s)" in output
    assert "existing index records were left unchanged" in output


def test_pdf_over_page_cap_emits_summary_warning(monkeypatch, capsys):
    items, reader = _run_scan(
        monkeypatch,
        {},
        [("PDF1", "storage:paper.pdf", "application/pdf")],
        extract_result=("paper text", "pdf"),
        extraction_details={"page_count": 75, "page_cap": 50},
    )

    assert reader.extract_calls == 1
    assert len(items) == 1
    output = capsys.readouterr().err
    assert "PDF page cap truncated full-text extraction for 1 item(s)" in output
    assert "75 pages; indexed first 50" in output


def test_pdf_within_page_cap_does_not_emit_warning(monkeypatch, capsys):
    items, reader = _run_scan(
        monkeypatch,
        {},
        [("PDF1", "storage:paper.pdf", "application/pdf")],
        extract_result=("paper text", "pdf"),
        extraction_details={"page_count": 50, "page_cap": 50},
    )

    assert reader.extract_calls == 1
    assert len(items) == 1
    assert "PDF page cap" not in capsys.readouterr().err
