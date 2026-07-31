"""Tests for BetterIssa and legacy semantic full-text attachment selection."""

from pathlib import Path

import pytest

from zotero_mcp.local_db import LocalZoteroReader

TEI = """\
<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>Header title must be excluded</title>
        <author>Header author must be excluded</author>
      </titleStmt>
    </fileDesc>
    <profileDesc>
      <abstract><div><p>Semantic abstract text.</p></div></abstract>
    </profileDesc>
  </teiHeader>
  <text>
    <front><div><p>Front matter must be excluded.</p></div></front>
    <body>
      <div>
        <head>Methods</head>
        <p>Body text with an <ref>inline citation</ref>.</p>
      </div>
    </body>
    <back>
      <listBibl><biblStruct>Bibliography must be excluded.</biblStruct></listBibl>
    </back>
  </text>
</TEI>
"""


class _Reader(LocalZoteroReader):
    def __init__(self, attachments, paths, metadata=None, caches=None):
        self.db_path = "/dev/null"
        self._connection = None
        self.pdf_max_pages = 10
        self.pdf_timeout = 30
        self._attachments = attachments
        self._paths = paths
        self._metadata = metadata or {}
        self._caches = caches or {}

    def _iter_parent_attachments(self, _parent_item_id: int):
        yield from self._attachments

    def _get_attachment_selection_metadata(self, _parent_item_id: int):
        return self._metadata

    def _resolve_attachment_path(self, attachment_key: str, _zotero_path: str):
        return self._paths.get(attachment_key)

    def _read_zotero_ft_cache(self, attachment_key: str):
        return self._caches.get(attachment_key)

    def _extract_text_from_pdf(self, file_path: Path):
        return file_path.read_text()


def _attachment(key: str, filename: str, content_type: str):
    return key, f"storage:{filename}", content_type


@pytest.mark.parametrize(
    ("title", "path", "content_type", "expected"),
    [
        ("BetterIssa indexing text", "storage:BetterIssa-indexing.txt", "text/plain", True),
        (
            "BetterIssa semantic document",
            "storage:BetterIssa-semantic-document.json",
            "application/json",
            True,
        ),
        (
            "BetterIssa Advanced OCR Markdown",
            "storage:BetterIssa-Advanced-OCR.md",
            "text/markdown",
            True,
        ),
        (
            "BetterIssa Reading View",
            "storage:BetterIssa-Reading-View.html",
            "text/html",
            True,
        ),
        (
            "BetterIssa references",
            "storage:BetterIssa-references.json",
            "application/json",
            False,
        ),
        ("BetterIssa GROBID TEI", "storage:output.xml", "application/xml", True),
        ("", "storage:BetterIssa-fulltext.txt", "text/plain", True),
        ("BetterIssa OCR PDF", "storage:paper.pdf", "application/pdf", True),
        ("Full Text PDF", "storage:paper.pdf", "application/pdf", True),
        ("Supplement", "storage:supplement.pdf", "application/pdf", False),
        ("Extraction metadata", "storage:metadata.json", "application/json", False),
    ],
)
def test_named_attachment_precedence_classifier(
    title, path, content_type, expected
):
    assert (
        LocalZoteroReader._uses_named_attachment_precedence(
            title, path, content_type
        )
        is expected
    )


def test_grobid_tei_extracts_only_abstract_and_body(tmp_path):
    tei = tmp_path / "paper.xml"
    tei.write_text(TEI)
    reader = _Reader([], {})

    text = reader._extract_grobid_tei(tei)

    assert "Semantic abstract text." in text
    assert "Methods Body text with an inline citation." in text
    assert "Header title" not in text
    assert "Header author" not in text
    assert "Front matter" not in text
    assert "Bibliography" not in text


def test_betterissa_indexing_text_wins_over_every_other_artifact(tmp_path):
    files = {
        "indexing": (
            "BetterIssa-indexing.txt",
            "text/plain",
            "deliberately filtered indexing text",
        ),
        "semantic": (
            "BetterIssa-semantic-document.json",
            "application/json",
            '{"ok": true, "status": "complete", "structured": {"plain_text": "semantic text"}}',
        ),
        "ocr": (
            "BetterIssa-Advanced-OCR.md",
            "text/markdown",
            "advanced OCR text",
        ),
        "reading": (
            "BetterIssa-Reading-View.html",
            "text/html",
            "<html><body>reading view text</body></html>",
        ),
        "pdf": ("source-document.bin", "application/pdf", "direct PDF text"),
    }
    attachments = []
    paths = {}
    metadata = {}
    for key, (filename, content_type, content) in files.items():
        path = tmp_path / filename
        path.write_text(content)
        paths[key] = path
        attachments.append(_attachment(key, filename, content_type))
        metadata[key] = {
            "title": {
                "indexing": "BetterIssa indexing text",
                "semantic": "BetterIssa semantic document",
                "ocr": "BetterIssa Advanced OCR Markdown",
                "reading": "BetterIssa Reading View",
                "pdf": "Original source",
            }[key],
        }
    reader = _Reader(
        attachments,
        paths,
        metadata,
        caches={"pdf": "Zotero PDF cache"},
    )

    assert reader._extract_fulltext_for_item(1) == (
        "deliberately filtered indexing text",
        "betterissa-indexing",
    )


def test_betterissa_rehydration_advances_to_best_available_artifact(tmp_path):
    pdf = tmp_path / "source.pdf"
    pdf.write_text("original PDF body")
    ocr = tmp_path / "BetterIssa-Advanced-OCR.md"
    ocr.write_text(
        "<!-- BetterIssa page 1 -->\nOCR checkpoint body\n"
        "<!-- BetterIssa page 2 -->\nOCR second page"
    )
    semantic = tmp_path / "BetterIssa-semantic-document.json"
    semantic.write_text(
        '{"ok": true, "status": "complete", '
        '"structured": {"plain_text": "semantic body"}}'
    )
    indexing = tmp_path / "BetterIssa-indexing.txt"
    indexing.write_text("final indexing body")
    reading = tmp_path / "BetterIssa-Reading-View.html"
    reading.write_text("<html><body>reading view body</body></html>")
    references = tmp_path / "BetterIssa-references.json"
    references.write_text('[{"label": "[1]", "raw_reference": "Citation"}]')

    attachments = [_attachment("pdf", pdf.name, "application/pdf")]
    paths = {"pdf": pdf}
    metadata = {"pdf": {"title": "Original PDF"}}
    reader = _Reader(
        attachments,
        paths,
        metadata,
        caches={"pdf": "Zotero PDF cache"},
    )

    assert reader._extract_fulltext_for_item(1) == (
        "Zotero PDF cache",
        "zotero-cache",
    )

    attachments.append(_attachment("ocr", ocr.name, "text/markdown"))
    paths["ocr"] = ocr
    metadata["ocr"] = {"title": "BetterIssa Advanced OCR Markdown"}

    ocr_text, ocr_source = reader._extract_fulltext_for_item(1)
    assert ocr_source == "betterissa-ocr"
    assert ocr_text == "OCR checkpoint body\n\nOCR second page"
    assert "BetterIssa page" not in ocr_text

    attachments.extend(
        [
            _attachment("semantic", semantic.name, "application/json"),
            _attachment("indexing", indexing.name, "text/plain"),
            _attachment("reading", reading.name, "text/html"),
            _attachment("references", references.name, "application/json"),
        ]
    )
    paths.update(
        {
            "semantic": semantic,
            "indexing": indexing,
            "reading": reading,
            "references": references,
        }
    )
    metadata.update(
        {
            "semantic": {"title": "BetterIssa semantic document"},
            "indexing": {"title": "BetterIssa indexing text"},
            "reading": {"title": "BetterIssa Reading View"},
            "references": {"title": "BetterIssa references"},
        }
    )

    assert reader._extract_fulltext_for_item(1) == (
        "final indexing body",
        "betterissa-indexing",
    )

    indexing.write_text("final indexing body\n\nVision description")
    assert reader._extract_fulltext_for_item(1) == (
        "final indexing body\n\nVision description",
        "betterissa-indexing",
    )


def test_betterissa_fixed_artifact_fallback_order(tmp_path):
    files = {
        "semantic": (
            "BetterIssa-semantic-document.json",
            "application/json",
            '{"ok": true, "status": "complete", "structured": {"plain_text": "semantic text"}}',
            "BetterIssa semantic document",
        ),
        "ocr": (
            "BetterIssa-Advanced-OCR.md",
            "text/markdown",
            "advanced OCR text",
            "BetterIssa Advanced OCR Markdown",
        ),
        "reading": (
            "BetterIssa-Reading-View.html",
            "text/html",
            "<html><body>reading view text</body></html>",
            "BetterIssa Reading View",
        ),
        "pdf": (
            "opaque-name.bin",
            "application/pdf",
            "direct PDF text",
            "Scanned source",
        ),
    }
    expected = [
        ({"semantic", "ocr", "reading", "pdf"}, "semantic text", "betterissa-semantic"),
        ({"ocr", "reading", "pdf"}, "advanced OCR text", "betterissa-ocr"),
        ({"reading", "pdf"}, "reading view text", "betterissa-reading-view"),
        ({"pdf"}, "Zotero PDF cache", "zotero-cache"),
    ]
    for included, expected_text, expected_source in expected:
        attachments = []
        paths = {}
        metadata = {}
        for key in included:
            filename, content_type, content, title = files[key]
            path = tmp_path / f"{key}-{filename}"
            path.write_text(content)
            paths[key] = path
            attachments.append(_attachment(key, filename, content_type))
            metadata[key] = {"title": title}
        reader = _Reader(
            attachments,
            paths,
            metadata,
            caches={"pdf": "Zotero PDF cache"},
        )

        text, source = reader._extract_fulltext_for_item(1)

        assert expected_text in text
        assert source == expected_source


def test_invalid_betterissa_semantic_document_falls_back(tmp_path):
    semantic = tmp_path / "BetterIssa-semantic-document.json"
    semantic.write_text('{"ok": false, "structured": {"plain_text": "must not index"}}')
    ocr = tmp_path / "BetterIssa-Advanced-OCR.md"
    ocr.write_text("advanced OCR fallback")
    attachments = [
        _attachment("semantic", semantic.name, "application/json"),
        _attachment("ocr", ocr.name, "text/markdown"),
    ]
    metadata = {
        "semantic": {"title": "BetterIssa semantic document"},
        "ocr": {"title": "BetterIssa Advanced OCR Markdown"},
    }
    reader = _Reader(
        attachments,
        {"semantic": semantic, "ocr": ocr},
        metadata,
    )

    assert reader._extract_fulltext_for_item(1) == (
        "advanced OCR fallback",
        "betterissa-ocr",
    )


def test_betterissa_references_are_not_treated_as_fulltext(tmp_path):
    references = tmp_path / "BetterIssa-references.json"
    references.write_text('[{"label": "[1]", "raw": "A cited paper"}]')
    reader = _Reader(
        [_attachment("references", references.name, "application/json")],
        {"references": references},
        {"references": {"title": "BetterIssa references"}},
    )

    assert reader._extract_fulltext_for_item(1) is None


def test_betterissa_workflow_artifacts_never_reach_generic_fallback(tmp_path):
    state = tmp_path / "BetterIssa-state.json"
    state.write_text('{"workflow": "must not be embedded"}')
    summary = tmp_path / "BetterIssa-summary.md"
    summary.write_text("processing summary must not be embedded")
    pdf = tmp_path / "opaque-source.bin"
    pdf.write_text("paper body from PDF")

    auxiliary_only = _Reader(
        [
            _attachment("state", state.name, "application/json"),
            _attachment("summary", summary.name, "text/markdown"),
        ],
        {"state": state, "summary": summary},
        {
            "state": {"title": "BetterIssa state"},
            "summary": {"title": "BetterIssa summary"},
        },
    )
    assert auxiliary_only._extract_fulltext_for_item(1) is None

    reader = _Reader(
        [
            _attachment("state", state.name, "application/json"),
            _attachment("summary", summary.name, "text/markdown"),
            _attachment("pdf", pdf.name, "application/pdf"),
        ],
        {
            "state": state,
            "summary": summary,
            "pdf": pdf,
        },
        {
            "state": {"title": "BetterIssa state"},
            "summary": {"title": "BetterIssa summary"},
            "pdf": {"title": "Original source"},
        },
        caches={"pdf": "Zotero PDF cache"},
    )

    assert reader._extract_fulltext_for_item(1) == (
        "Zotero PDF cache",
        "zotero-cache",
    )


@pytest.mark.parametrize(
    ("included_keys", "expected_text", "expected_source"),
    [
        (
            {"tei", "fulltext", "ocr", "full_text_pdf", "pdf", "other"},
            "Semantic abstract text.",
            "grobid-tei",
        ),
        (
            {"fulltext", "ocr", "full_text_pdf", "pdf", "other"},
            "named fulltext",
            "fulltext",
        ),
        (
            {"ocr", "full_text_pdf", "pdf", "other"},
            "ocr pdf",
            "ocr-pdf",
        ),
        (
            {"full_text_pdf", "pdf", "other"},
            "full text pdf",
            "pdf",
        ),
        ({"pdf", "other"}, "generic pdf", "pdf"),
        ({"other"}, "generic text", "file"),
    ],
)
def test_requested_attachment_precedence(
    tmp_path, included_keys, expected_text, expected_source
):
    files = {
        "tei": ("BetterIssa-GROBID-TEI.xml", "application/xml", TEI),
        "fulltext": ("BetterIssa-fulltext.txt", "text/plain", "named fulltext"),
        "ocr": ("BetterIssa-OCR.pdf", "application/pdf", "ocr pdf"),
        "full_text_pdf": ("Full Text PDF.pdf", "application/pdf", "full text pdf"),
        "pdf": ("supplement.pdf", "application/pdf", "generic pdf"),
        "other": ("transcript.txt", "text/plain", "generic text"),
    }
    attachments = []
    paths = {}
    for key, (filename, content_type, content) in files.items():
        if key not in included_keys:
            continue
        path = tmp_path / filename
        path.write_text(content)
        paths[key] = path
        attachments.append(_attachment(key, filename, content_type))

    reader = _Reader(attachments, paths)
    text, source = reader._extract_fulltext_for_item(1)

    assert expected_text in text
    assert source == expected_source


def test_malformed_or_bodyless_tei_falls_back_to_fulltext(tmp_path):
    broken_tei = tmp_path / "BetterIssa-GROBID-TEI.xml"
    broken_tei.write_text("<TEI><text><body>")
    fulltext = tmp_path / "BetterIssa-fulltext.txt"
    fulltext.write_text("fallback fulltext")
    attachments = [
        _attachment("tei", broken_tei.name, "application/xml"),
        _attachment("fulltext", fulltext.name, "text/plain"),
    ]
    reader = _Reader(
        attachments,
        {"tei": broken_tei, "fulltext": fulltext},
    )

    assert reader._extract_fulltext_for_item(1) == (
        "fallback fulltext",
        "fulltext",
    )


def test_pdf_cache_does_not_preempt_grobid_tei(tmp_path):
    tei = tmp_path / "BetterIssa-GROBID-TEI.xml"
    tei.write_text(TEI)
    pdf = tmp_path / "Full Text PDF.pdf"
    pdf.write_text("direct PDF text")
    attachments = [
        _attachment("pdf", pdf.name, "application/pdf"),
        _attachment("tei", tei.name, "application/xml"),
    ]
    reader = _Reader(
        attachments,
        {"pdf": pdf, "tei": tei},
        caches={"pdf": "older Zotero PDF cache"},
    )

    text, source = reader._extract_fulltext_for_item(1)

    assert "Semantic abstract text." in text
    assert "older Zotero PDF cache" not in text
    assert source == "grobid-tei"


def test_newest_ocr_pdf_wins_within_its_precedence_group(tmp_path):
    old = tmp_path / "BetterIssa-OCR-old.pdf"
    old.write_text("old OCR")
    new = tmp_path / "BetterIssa-OCR-new.pdf"
    new.write_text("new OCR")
    attachments = [
        _attachment("new", new.name, "application/pdf"),
        _attachment("old", old.name, "application/pdf"),
    ]
    metadata = {
        "old": {
            "title": "BetterIssa OCR PDF",
            "date_modified": "2026-07-20 10:00:00",
            "storage_mod_time": 1,
            "item_id": 10,
        },
        "new": {
            "title": "BetterIssa OCR PDF",
            "date_modified": "2026-07-27 10:00:00",
            "storage_mod_time": 2,
            "item_id": 11,
        },
    }
    reader = _Reader(attachments, {"old": old, "new": new}, metadata)

    assert reader._extract_fulltext_for_item(1) == ("new OCR", "ocr-pdf")
