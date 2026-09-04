"""Tests for get_annotations, especially descending through attachments
to find annotations (which in Zotero's data model are children of the
PDF attachment, not of the parent paper)."""

import json

from zotero_mcp import server
from zotero_mcp.tools.annotations import _annotation_to_record


class DummyContext:
    def info(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None


class FakeZoteroForAnnotations:
    """Fake Zotero client that models the real parent/child hierarchy.

    Papers contain notes and attachments; annotations live under
    attachments, not directly under the paper.
    """

    def __init__(self, parents, children_by_key):
        self._parents = parents
        self._children = children_by_key

    def item(self, key):
        return self._parents.get(key, {"data": {"title": "Unknown"}})

    def children(self, item_key, start=0, limit=100, itemType=None, **_kwargs):
        all_children = self._children.get(item_key, [])
        if itemType:
            all_children = [
                c for c in all_children
                if c.get("data", {}).get("itemType") == itemType
            ]
        return all_children[start:start + limit]

    def items(self, itemKey=None, **_kwargs):
        keys = (itemKey or "").split(",")
        return [self._parents[key] for key in keys if key in self._parents]


def _annotation(key, parent, text):
    return {
        "key": key,
        "data": {
            "itemType": "annotation",
            "annotationType": "highlight",
            "annotationText": text,
            "annotationComment": "",
            "parentItem": parent,
            "tags": [],
        },
    }


def test_get_annotations_descends_through_attachment(monkeypatch):
    """Paper key → descend through its PDF attachment to find annotations."""
    parents = {
        "PAPER001": {"data": {"title": "A Paper", "itemType": "journalArticle"}},
    }
    # Paper's direct children: one attachment (PDF) and one note.
    # The Zotero web API does NOT return annotations as children of the paper.
    children = {
        "PAPER001": [
            {"key": "ATTACH01", "data": {"itemType": "attachment", "contentType": "application/pdf"}},
            {"key": "NOTE0001", "data": {"itemType": "note", "note": "<p>n</p>"}},
        ],
        "ATTACH01": [
            _annotation("ANNO0001", "ATTACH01", "first highlight"),
            _annotation("ANNO0002", "ATTACH01", "second highlight"),
            _annotation("ANNO0003", "ATTACH01", "third highlight"),
        ],
    }
    fake = FakeZoteroForAnnotations(parents, children)
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setenv("ZOTERO_LOCAL", "")

    result = server.get_annotations(item_key="PAPER001", ctx=DummyContext())

    assert "ANNO0001" in result
    assert "ANNO0002" in result
    assert "ANNO0003" in result
    assert "first highlight" in result
    assert "No annotations found" not in result


def test_get_annotations_json_returns_stable_structured_records(monkeypatch):
    parents = {
        "PAPER001": {
            "key": "PAPER001",
            "data": {"title": "A Paper", "itemType": "journalArticle"},
        },
        "ATTACH01": {
            "key": "ATTACH01",
            "data": {
                "title": "Full Text PDF",
                "itemType": "attachment",
                "parentItem": "PAPER001",
            },
        },
    }
    annotation = _annotation("ANNO0001", "ATTACH01", "structured highlight")
    annotation["data"].update(
        {
            "annotationComment": "important claim",
            "annotationColor": "#ffd400",
            "annotationPageLabel": "7",
            "annotationPosition": json.dumps({"pageIndex": 6, "rects": []}),
            "dateAdded": "2026-04-10T12:00:00Z",
            "dateModified": "2026-04-11T12:00:00Z",
            "tags": [{"tag": "method", "type": 1}, {"tag": "to-read"}],
        }
    )
    children = {
        "PAPER001": [
            {
                "key": "ATTACH01",
                "data": {
                    "itemType": "attachment",
                    "contentType": "application/pdf",
                },
            }
        ],
        "ATTACH01": [annotation],
    }
    fake = FakeZoteroForAnnotations(parents, children)
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setenv("ZOTERO_LOCAL", "")

    result = server.get_annotations(
        item_key="PAPER001", format="json", ctx=DummyContext()
    )

    assert json.loads(result) == [
        {
            "annotation_key": "ANNO0001",
            "item_key": "PAPER001",
            "attachment_key": "ATTACH01",
            "parent_title": "A Paper",
            "attachment_title": "Full Text PDF",
            "type": "highlight",
            "page": "7",
            "page_index": 6,
            "text": "structured highlight",
            "comment": "important claim",
            "color": "#ffd400",
            "color_category": "Yellow",
            "tags": ["method", "to-read"],
            "created": "2026-04-10T12:00:00Z",
            "modified": "2026-04-11T12:00:00Z",
            "source": "zotero",
        }
    ]


def test_get_annotations_json_attachment_key_resolves_to_paper(monkeypatch):
    parents = {
        "PAPER001": {
            "key": "PAPER001",
            "data": {"title": "Actual Paper", "itemType": "journalArticle"},
        },
        "ATTACH01": {
            "key": "ATTACH01",
            "data": {
                "title": "Full Text PDF",
                "itemType": "attachment",
                "parentItem": "PAPER001",
            },
        },
    }
    fake = FakeZoteroForAnnotations(
        parents,
        {"ATTACH01": [_annotation("ANNO0001", "ATTACH01", "highlight")]},
    )
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setenv("ZOTERO_LOCAL", "")

    record = json.loads(
        server.get_annotations(
            item_key="ATTACH01", format="json", ctx=DummyContext()
        )
    )[0]

    assert record["item_key"] == "PAPER001"
    assert record["attachment_key"] == "ATTACH01"
    assert record["parent_title"] == "Actual Paper"
    assert record["attachment_title"] == "Full Text PDF"


def test_get_annotations_json_library_wide_resolves_paper_context(monkeypatch):
    annotation = _annotation("ANNO0001", "ATTACH01", "library highlight")
    parents = {
        "PAPER001": {
            "key": "PAPER001",
            "data": {"title": "A Paper", "itemType": "journalArticle"},
        },
        "ATTACH01": {
            "key": "ATTACH01",
            "data": {
                "title": "Full Text PDF",
                "itemType": "attachment",
                "parentItem": "PAPER001",
            },
        },
    }

    class LibraryFake(FakeZoteroForAnnotations):
        def items(
            self, start=0, limit=100, itemType=None, itemKey=None, **_kwargs
        ):
            if itemKey:
                keys = itemKey.split(",")
                return [self._parents[key] for key in keys if key in self._parents]
            if itemType == "annotation":
                return [annotation][start:start + limit]
            return []

    fake = LibraryFake(parents, {})
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setenv("ZOTERO_LOCAL", "")

    record = json.loads(
        server.get_annotations(format="json", ctx=DummyContext())
    )[0]

    assert record["item_key"] == "PAPER001"
    assert record["attachment_key"] == "ATTACH01"
    assert record["parent_title"] == "A Paper"
    assert record["attachment_title"] == "Full Text PDF"


def test_get_annotations_json_does_not_present_attachment_as_paper(monkeypatch):
    """An unresolved paper title remains null instead of using the PDF title."""
    parents = {
        "ATTACH01": {
            "key": "ATTACH01",
            "data": {
                "title": "Full Text PDF",
                "itemType": "attachment",
                "parentItem": "PAPER001",
            },
        },
    }

    class MissingPaper(FakeZoteroForAnnotations):
        def item(self, key):
            if key == "PAPER001":
                raise RuntimeError("paper not visible")
            return super().item(key)

    fake = MissingPaper(
        parents,
        {"ATTACH01": [_annotation("ANNO0001", "ATTACH01", "highlight")]},
    )
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setenv("ZOTERO_LOCAL", "")

    record = json.loads(
        server.get_annotations(
            item_key="ATTACH01", format="json", ctx=DummyContext()
        )
    )[0]

    assert record["item_key"] == "PAPER001"
    assert record["parent_title"] is None
    assert record["attachment_title"] == "Full Text PDF"


def test_annotation_record_normalizes_all_sources():
    context = {
        "item_key": "PAPER001",
        "attachment_key": "ATTACH01",
        "parent_title": "A Paper",
        "attachment_title": "Full Text PDF",
    }
    common = {
        "annotationType": "highlight",
        "annotationText": "same highlight",
        "annotationComment": "same comment",
        "annotationColor": "#ffd400",
        "tags": [{"tag": "method"}, "to-read", {"tag": "method"}],
        "dateAdded": "2026-04-10T12:00:00Z",
        "dateModified": "2026-04-11T12:00:00Z",
    }
    source_data = {
        "zotero": {
            **common,
            "annotationPageLabel": "7",
            "annotationPosition": {"pageIndex": 6},
        },
        "better_bibtex": {
            **common,
            "_pdf_page": 7,
            "_pageLabel": "7",
            "_from_better_bibtex": True,
        },
        "pdf_extraction": {
            **common,
            "_pdf_page": 7,
            "_from_pdf_extraction": True,
        },
    }

    records = {
        source: _annotation_to_record(
            {"key": "ANNO0001", "data": data}, context
        )
        for source, data in source_data.items()
    }

    for source, record in records.items():
        assert record["source"] == source
        assert record["page"] == "7"
        assert record["page_index"] == 6
        assert record["color_category"] == "Yellow"
        assert record["tags"] == ["method", "to-read"]
        assert record["created"] == "2026-04-10T12:00:00Z"
        assert record["modified"] == "2026-04-11T12:00:00Z"
        assert record["item_key"] == "PAPER001"
        assert record["attachment_key"] == "ATTACH01"
        assert record["parent_title"] == "A Paper"


def test_get_annotations_json_empty_result_is_an_array(monkeypatch):
    parents = {
        "PAPER001": {
            "key": "PAPER001",
            "data": {"title": "Bare Paper", "itemType": "journalArticle"},
        },
    }
    fake = FakeZoteroForAnnotations(parents, {"PAPER001": []})
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setenv("ZOTERO_LOCAL", "")

    result = server.get_annotations(
        item_key="PAPER001", format="json", ctx=DummyContext()
    )

    assert json.loads(result) == []


def test_get_annotations_accepts_attachment_key(monkeypatch):
    """Attachment key → annotations are returned as direct children."""
    parents = {
        "ATTACH01": {"data": {"title": "PDF", "itemType": "attachment"}},
    }
    children = {
        "ATTACH01": [
            _annotation("ANNO0001", "ATTACH01", "highlight text"),
            _annotation("ANNO0002", "ATTACH01", "another one"),
        ],
    }
    fake = FakeZoteroForAnnotations(parents, children)
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setenv("ZOTERO_LOCAL", "")

    result = server.get_annotations(item_key="ATTACH01", ctx=DummyContext())

    assert "ANNO0001" in result
    assert "ANNO0002" in result
    assert "highlight text" in result


def test_get_annotations_dedupes_across_paths(monkeypatch):
    """If the API returns the same annotation as both a direct child and
    a nested attachment child, it should appear only once."""
    parents = {
        "PAPER001": {"data": {"title": "Paper", "itemType": "journalArticle"}},
    }
    anno = _annotation("ANNO0001", "ATTACH01", "only once")
    children = {
        # Local API quirk: paper+itemType=annotation returns grandchildren
        "PAPER001": [
            anno,
            {"key": "ATTACH01", "data": {"itemType": "attachment", "contentType": "application/pdf"}},
        ],
        "ATTACH01": [anno],
    }
    fake = FakeZoteroForAnnotations(parents, children)
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setenv("ZOTERO_LOCAL", "")

    result = server.get_annotations(item_key="PAPER001", ctx=DummyContext())

    assert result.count("ANNO0001") == 1
    assert result.count("only once") == 1


def test_get_annotations_handles_multiple_attachments(monkeypatch):
    """Paper with two PDF attachments → annotations from both are returned."""
    parents = {
        "PAPER001": {"data": {"title": "Paper", "itemType": "journalArticle"}},
    }
    children = {
        "PAPER001": [
            {"key": "ATTACH01", "data": {"itemType": "attachment", "contentType": "application/pdf"}},
            {"key": "ATTACH02", "data": {"itemType": "attachment", "contentType": "application/pdf"}},
        ],
        "ATTACH01": [_annotation("ANNO0001", "ATTACH01", "from pdf one")],
        "ATTACH02": [_annotation("ANNO0002", "ATTACH02", "from pdf two")],
    }
    fake = FakeZoteroForAnnotations(parents, children)
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setenv("ZOTERO_LOCAL", "")

    result = server.get_annotations(item_key="PAPER001", ctx=DummyContext())

    assert "ANNO0001" in result
    assert "ANNO0002" in result


def test_get_annotations_reports_none_when_empty(monkeypatch):
    """Paper with attachments but zero annotations → friendly empty message."""
    parents = {
        "PAPER001": {"data": {"title": "Bare Paper", "itemType": "journalArticle"}},
    }
    children = {
        "PAPER001": [{"key": "ATTACH01", "data": {"itemType": "attachment", "contentType": "application/pdf"}}],
        "ATTACH01": [],
    }
    fake = FakeZoteroForAnnotations(parents, children)
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setenv("ZOTERO_LOCAL", "")

    result = server.get_annotations(item_key="PAPER001", ctx=DummyContext())

    assert "No annotations found" in result
    assert "Bare Paper" in result


def test_get_annotations_paginates_through_many_annotations(monkeypatch):
    """Works correctly when attachment has >100 annotations (page boundary)."""
    parents = {
        "PAPER001": {"data": {"title": "Paper", "itemType": "journalArticle"}},
    }
    many = [_annotation(f"A{i:05d}", "ATTACH01", f"hl {i}") for i in range(150)]
    children = {
        "PAPER001": [{"key": "ATTACH01", "data": {"itemType": "attachment", "contentType": "application/pdf"}}],
        "ATTACH01": many,
    }
    fake = FakeZoteroForAnnotations(parents, children)
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setenv("ZOTERO_LOCAL", "")

    result = server.get_annotations(item_key="PAPER001", ctx=DummyContext())

    assert "A00000" in result
    assert "A00099" in result
    assert "A00149" in result  # past first page


def test_get_annotations_reports_api_failure_instead_of_empty(monkeypatch):
    class BrokenZotero(FakeZoteroForAnnotations):
        def children(self, *_args, **_kwargs):
            raise TimeoutError("annotation backend down")

    parents = {
        "PAPER001": {"data": {"title": "A Paper", "itemType": "journalArticle"}},
    }
    fake = BrokenZotero(parents, {})
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setenv("ZOTERO_LOCAL", "")

    result = server.get_annotations(item_key="PAPER001", ctx=DummyContext())

    assert result.startswith("Error: Annotation retrieval failed:")
    assert "annotation backend down" in result
