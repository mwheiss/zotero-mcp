from zotero_mcp import server


class DummyContext:
    def info(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None


class FakeZotero:
    def __init__(self, items):
        self._items = items

    def items(self, start=0, limit=100, **_kwargs):
        return self._items[start : start + limit]


def test_advanced_search_filters_items(monkeypatch):
    fake_items = [
        {
            "key": "AAA11111",
            "data": {
                "itemType": "journalArticle",
                "title": "Quantum Networks and Learning",
                "date": "2024",
                "creators": [{"firstName": "Jane", "lastName": "Doe"}],
                "tags": [{"tag": "physics"}],
            },
        },
        {
            "key": "BBB22222",
            "data": {
                "itemType": "journalArticle",
                "title": "Classical Literature Review",
                "date": "2018",
                "creators": [{"firstName": "Alex", "lastName": "Smith"}],
                "tags": [{"tag": "history"}],
            },
        },
        {
            "key": "CCC33333",
            "data": {
                "itemType": "attachment",
                "title": "Ignored Attachment",
                "date": "2024",
                "creators": [],
                "tags": [],
            },
        },
    ]
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: FakeZotero(fake_items))

    result = server.advanced_search(
        conditions=[
            {"field": "title", "operation": "contains", "value": "quantum"},
            {"field": "year", "operation": "isGreaterThan", "value": "2020"},
        ],
        join_mode="all",
        limit=10,
        ctx=DummyContext(),
    )

    assert "Quantum Networks and Learning" in result
    assert "Classical Literature Review" not in result
    assert "Ignored Attachment" not in result


def test_advanced_search_rejects_unknown_operation(monkeypatch):
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: FakeZotero([]))

    result = server.advanced_search(
        conditions=[{"field": "title", "operation": "regex", "value": ".*"}],
        ctx=DummyContext(),
    )

    assert "Unsupported operation" in result


def test_advanced_search_exposes_bulk_inventory_fields(monkeypatch):
    fake_items = [
        {
            "key": "AAA11111",
            "data": {
                "itemType": "journalArticle",
                "title": "Complete Metadata",
                "date": "2024-03-01",
                "DOI": "10.1000/complete",
                "language": "en",
                "dateAdded": "2026-07-01T10:00:00Z",
                "dateModified": "2026-08-06T11:30:00Z",
                "creators": [],
                "tags": [],
            },
        }
    ]
    monkeypatch.setattr(
        "zotero_mcp.client.get_zotero_client",
        lambda: FakeZotero(fake_items),
    )

    result = server.advanced_search(
        conditions=[{"field": "itemType", "operation": "is", "value": "journalArticle"}],
        ctx=DummyContext(),
    )

    assert "**DOI:** 10.1000/complete" in result
    assert "**Language:** en" in result
    assert "**Date Added:** 2026-07-01T10:00:00Z" in result
    assert "**Date Modified:** 2026-08-06T11:30:00Z" in result


def test_advanced_search_caps_bulk_results_at_2000(monkeypatch):
    fake_items = [
        {
            "key": f"K{i:07d}",
            "data": {
                "itemType": "journalArticle",
                "title": f"Paper {i}",
                "date": "2026",
                "creators": [],
                "tags": [],
            },
        }
        for i in range(2100)
    ]
    monkeypatch.setattr(
        "zotero_mcp.client.get_zotero_client",
        lambda: FakeZotero(fake_items),
    )

    result = server.advanced_search(
        conditions=[{"field": "itemType", "operation": "is", "value": "journalArticle"}],
        limit=2500,
        ctx=DummyContext(),
    )

    assert "Found 2000 items matching the search criteria" in result
    assert "Paper 1999" in result
    assert "Paper 2000" not in result
