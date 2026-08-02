"""Tests for API metadata ingest and since-based incremental sync."""

import json
import sys
from types import SimpleNamespace

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

from zotero_mcp import semantic_search


class FakeChromaClient:
    """Minimal ChromaClient stand-in that records operations for assertions."""

    def __init__(self, preloaded_ids=None, preloaded_metadata=None):
        self.embedding_max_tokens = 8000
        self._ids = set(preloaded_ids or [])
        self.added = []  # list of (docs, metas, ids)
        self.deleted = []  # list of ids deleted
        self.reset_calls = 0
        self.staged_rebuild_calls = 0
        self.staged_commit_calls = 0
        self.staged_abort_calls = 0
        self.operation_events = []
        self.metadata_by_key = dict(preloaded_metadata or {})
        self.metadata_updates = []
        self._pre_staging_ids = None

    def truncate_text(self, text, max_tokens=None):
        return text[:4000]

    def get_existing_ids(self, ids):
        return {i for i in ids if i in self._ids}

    def get_all_ids(self):
        return set(self._ids)

    def get_collection_info(self):
        return {"count": len(self._ids)}

    def get_document_metadata(self, doc_id):
        return self.metadata_by_key.get(doc_id)

    def update_item_metadata(self, item_key, updates):
        if item_key not in self._ids:
            return 0
        self.operation_events.append("metadata")
        existing = dict(self.metadata_by_key.get(item_key, {}))
        existing.update(updates)
        self.metadata_by_key[item_key] = existing
        self.metadata_updates.append((item_key, dict(updates)))
        return 1

    def upsert_documents(self, documents, metadatas, ids):
        self.operation_events.append("upsert")
        self.added.append((list(documents), list(metadatas), list(ids)))
        for i in ids:
            self._ids.add(i)

    def add_documents(self, documents, metadatas, ids):
        self.upsert_documents(documents, metadatas, ids)

    def delete_documents(self, ids):
        self.operation_events.append("delete")
        self.deleted.extend(list(ids))
        for i in ids:
            self._ids.discard(i)

    def reset_collection(self):
        self.reset_calls += 1
        self._ids = set()

    def begin_staged_rebuild(self):
        self.operation_events.append("begin_staging")
        self.staged_rebuild_calls += 1
        self._pre_staging_ids = set(self._ids)
        self._ids = set()

    def commit_staged_rebuild(self):
        self.operation_events.append("commit_staging")
        self.staged_commit_calls += 1
        self._pre_staging_ids = None

    def abort_staged_rebuild(self):
        self.operation_events.append("abort_staging")
        self.staged_abort_calls += 1
        if self._pre_staging_ids is not None:
            self._ids = self._pre_staging_ids
            self._pre_staging_ids = None


class FakeZoteroClient:
    """pyzotero client double with scripted responses."""

    def __init__(self):
        self.items_by_key = {}
        self.fulltext_by_key = {}   # key -> dict (content, indexedChars, ...)
        self.fulltext_versions_state = {}  # key -> library version
        self.children_by_parent = {}  # parent_key -> list[item]
        self.versions_state = {}    # key -> library_version (current state)
        self.version_history = []   # (since_version, changed_dict) pairs
        self.current_library_version = 0
        self.current_versions_error = None
        self.new_fulltext_error = None
        self.deleted_state = {"items": []}
        # Pagination helper
        self.items_order = []
        # Recording calls for assertions
        self.calls = []

    # ---- setup helpers for tests ----

    def load_scenario(self, items, fulltext=None, children=None, library_version=0):
        """items: list of pyzotero-shaped dicts with top-level 'key' and 'data'."""
        for it in items:
            self.items_by_key[it["key"]] = it
            self.items_order.append(it["key"])
            self.versions_state[it["key"]] = library_version
        for k, v in (fulltext or {}).items():
            self.fulltext_by_key[k] = v
            self.fulltext_versions_state[k] = library_version
        for k, v in (children or {}).items():
            self.children_by_parent[k] = v
        self.current_library_version = library_version

    # ---- pyzotero surface used by the code under test ----

    def items(self, start=0, limit=100, **kwargs):
        self.calls.append(("items", start, limit))
        chunk = self.items_order[start:start + limit]
        return [self.items_by_key[k] for k in chunk]

    def item(self, key):
        self.calls.append(("item", key))
        if key not in self.items_by_key:
            raise LookupError(f"item {key} not found")
        return self.items_by_key[key]

    def children(self, key):
        self.calls.append(("children", key))
        return list(self.children_by_parent.get(key, []))

    def fulltext_item(self, key):
        self.calls.append(("fulltext_item", key))
        if key not in self.fulltext_by_key:
            raise RuntimeError(f"404 fulltext not found for {key}")
        return self.fulltext_by_key[key]

    def item_versions(self, since=None, **kwargs):
        self.calls.append(("item_versions", since))
        if since is None:
            if self.current_versions_error:
                raise self.current_versions_error
            return dict(self.versions_state)
        return {k: v for k, v in self.versions_state.items() if v > since}

    def new_fulltext(self, since):
        self.calls.append(("new_fulltext", since))
        if self.new_fulltext_error:
            raise self.new_fulltext_error
        return {k: v for k, v in self.fulltext_versions_state.items() if v > since}

    def deleted(self, since):
        self.calls.append(("deleted", since))
        return self.deleted_state

    def last_modified_version(self, **kwargs):
        self.calls.append(("last_modified_version",))
        return self.current_library_version


def _paper(key, title="Paper", item_type="conferencePaper", version=1):
    return {
        "key": key,
        "version": version,
        "data": {
            "key": key,
            "itemType": item_type,
            "title": title,
            "abstractNote": f"abstract of {title}",
            "creators": [{"creatorType": "author", "firstName": "A", "lastName": "Author"}],
            "dateAdded": "2024-01-01T00:00:00Z",
            "dateModified": "2024-01-01T00:00:00Z",
        },
    }


def _build_search(monkeypatch, zot: FakeZoteroClient, chroma: FakeChromaClient,
                  config_path: str | None = None) -> "semantic_search.ZoteroSemanticSearch":
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: zot)
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: False)
    return semantic_search.ZoteroSemanticSearch(
        chroma_client=chroma,
        config_path=config_path,
    )


# --------- Integration tests: _get_items_from_api ----------

def test_get_items_from_api_never_fetches_cached_fulltext(monkeypatch):
    zot = FakeZoteroClient()
    zot.load_scenario([_paper("A"), _paper("B")], fulltext={"A": {"content": "Abody"}, "B": {"content": "Bbody"}})
    search = _build_search(monkeypatch, zot, FakeChromaClient())
    items = search._get_items_from_api()
    assert len(items) == 2
    for it in items:
        assert "fulltext" not in it["data"]
    assert not any(c[0] == "fulltext_item" for c in zot.calls)


def test_get_items_from_api_excludes_all_child_artifact_types(monkeypatch):
    zot = FakeZoteroClient()
    zot.load_scenario(
        [
            _paper("P1"),
            _paper("ATT1", item_type="attachment"),
            _paper("NOTE1", item_type="note"),
            _paper("ANN1", item_type="annotation"),
        ]
    )
    search = _build_search(monkeypatch, zot, FakeChromaClient())

    items = search._get_items_from_api()

    assert [item["key"] for item in items] == ["P1"]


def test_get_items_from_api_tracks_active_attachments_by_parent(monkeypatch):
    zot = FakeZoteroClient()
    active = _paper("ATT1", item_type="attachment")
    active["data"]["parentItem"] = "P1"
    deleted = _paper("ATT2", item_type="attachment")
    deleted["data"]["parentItem"] = "P1"
    deleted["data"]["deleted"] = True
    zot.load_scenario([_paper("P1"), active, deleted])
    search = _build_search(monkeypatch, zot, FakeChromaClient())

    search._get_items_from_api()

    assert search._last_api_attachment_keys_by_parent == {"P1": {"ATT1"}}


# --------- Integration tests: incremental fetch ----------

def test_get_changed_items_from_api_returns_only_changed_keys(monkeypatch):
    zot = FakeZoteroClient()
    zot.load_scenario(
        [_paper("OLD"), _paper("NEW")],
        library_version=5,
    )
    # Mark OLD as unchanged since v3, NEW as changed at v5
    zot.versions_state = {"OLD": 3, "NEW": 5}
    search = _build_search(monkeypatch, zot, FakeChromaClient())
    changed, current_keys = search._get_changed_items_from_api(since_version=3)
    assert [c["key"] for c in changed] == ["NEW"]
    assert current_keys == {"OLD", "NEW"}


def test_get_changed_items_filters_out_attachments_and_notes(monkeypatch):
    zot = FakeZoteroClient()
    note = _paper("N1", item_type="note")
    ann = _paper("A1", item_type="annotation")
    paper = _paper("P1", item_type="conferencePaper")
    zot.load_scenario([note, ann, paper], library_version=10)
    zot.versions_state = {"N1": 10, "A1": 10, "P1": 10}
    search = _build_search(monkeypatch, zot, FakeChromaClient())
    changed, _ = search._get_changed_items_from_api(since_version=0)
    assert [c["key"] for c in changed] == ["P1"]


def test_attachment_only_change_reconciles_parent(monkeypatch):
    parent = _paper("P1")
    attachment = _paper("ATT1", item_type="attachment")
    attachment["data"]["parentItem"] = "P1"
    zot = FakeZoteroClient()
    zot.load_scenario([parent, attachment], library_version=8)
    zot.versions_state = {"P1": 3, "ATT1": 8}
    search = _build_search(monkeypatch, zot, FakeChromaClient())

    changed, _ = search._get_changed_items_from_api(since_version=5)

    assert [item["key"] for item in changed] == ["P1"]


def test_deleted_attachment_reconciles_parent(monkeypatch):
    parent = _paper("P1")
    attachment = _paper("ATT1", item_type="attachment")
    attachment["data"].update({"parentItem": "P1", "deleted": True})
    zot = FakeZoteroClient()
    zot.load_scenario([parent, attachment], library_version=8)
    zot.versions_state = {"P1": 3}
    zot.deleted_state = {"items": ["ATT1"]}
    search = _build_search(monkeypatch, zot, FakeChromaClient())

    changed, _ = search._get_changed_items_from_api(since_version=5)

    assert [item["key"] for item in changed] == ["P1"]


# --------- Integration tests: update_database orchestration ----------

def _write_config(tmp_path, extra: dict | None = None):
    cfg = {
        "semantic_search": {
            "embedding_model": "default",
            "update_config": {"auto_update": False, "update_frequency": "manual"},
            "extraction": {"pdf_max_pages": 10},
            "fulltext": False,
            "indexed_fulltext": False,
        }
    }
    if extra:
        cfg["semantic_search"].update(extra)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))
    return str(config_path)


def test_update_database_bootstrap_scans_full_library(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path)
    zot = FakeZoteroClient()
    zot.load_scenario(
        [_paper("P1"), _paper("P2")],
        fulltext={"P1": {"content": "Paper one body"}, "P2": {"content": "Paper two body"}},
        library_version=7,
    )
    chroma = FakeChromaClient()
    search = _build_search(monkeypatch, zot, chroma, config_path=config_path)

    stats = search.update_database()

    assert stats["total_items"] == 2
    assert stats["processed_items"] == 2
    # last_sync_version should be persisted to config
    saved = json.loads(open(config_path).read())
    assert saved["semantic_search"]["last_sync_version"] == 7
    assert saved["semantic_search"]["indexed_fulltext"] is False


def test_update_database_incremental_only_fetches_changed(monkeypatch, tmp_path):
    """Second run with last_sync_version set should only reindex changed items."""
    config_path = _write_config(tmp_path, extra={"last_sync_version": 5})
    zot = FakeZoteroClient()
    zot.load_scenario(
        [_paper("OLD"), _paper("NEW")],
        fulltext={"NEW": {"content": "New body"}},
        library_version=8,
    )
    zot.versions_state = {"OLD": 3, "NEW": 8}
    chroma = FakeChromaClient(preloaded_ids=["OLD"])  # OLD was already indexed
    search = _build_search(monkeypatch, zot, chroma, config_path=config_path)

    stats = search.update_database()

    # Only NEW should be processed, OLD untouched
    assert stats["processed_items"] == 1
    added_ids = [i for batch in chroma.added for i in batch[2]]
    assert added_ids == ["NEW"]
    saved = json.loads(open(config_path).read())
    assert saved["semantic_search"]["last_sync_version"] == 8


def test_update_database_incremental_deletes_removed_items(monkeypatch, tmp_path):
    """Items present in ChromaDB but removed from the library should be deleted."""
    config_path = _write_config(tmp_path, extra={"last_sync_version": 5})
    zot = FakeZoteroClient()
    # Only NEW is still in the library; DELETED_ME was removed
    zot.load_scenario([_paper("NEW")], fulltext={"NEW": {"content": "body"}}, library_version=9)
    zot.versions_state = {"NEW": 9}
    chroma = FakeChromaClient(preloaded_ids=["NEW", "DELETED_ME"])
    search = _build_search(monkeypatch, zot, chroma, config_path=config_path)

    stats = search.update_database()

    assert "DELETED_ME" in chroma.deleted
    assert stats["deleted_items"] == 1


def test_update_database_skips_deletion_when_current_keys_fail(monkeypatch, tmp_path):
    """A failed current-key enumeration must not look like an empty library."""
    config_path = _write_config(tmp_path, extra={"last_sync_version": 5})
    zot = FakeZoteroClient()
    zot.load_scenario([_paper("NEW")], library_version=9)
    zot.current_versions_error = RuntimeError("temporary API failure")
    chroma = FakeChromaClient(preloaded_ids=["NEW", "KEEP_ME"])
    search = _build_search(monkeypatch, zot, chroma, config_path=config_path)

    stats = search.update_database()

    assert chroma.deleted == []
    assert stats["deleted_items"] == 0
    saved = json.loads(open(config_path).read())
    assert saved["semantic_search"]["last_sync_version"] == 5


def test_update_database_incremental_noop_when_version_unchanged(monkeypatch, tmp_path):
    """If last_modified_version == last_sync_version, skip ingest entirely."""
    config_path = _write_config(tmp_path, extra={"last_sync_version": 42})
    zot = FakeZoteroClient()
    zot.load_scenario([_paper("X")], library_version=42)
    zot.versions_state = {"X": 42}
    chroma = FakeChromaClient(preloaded_ids=["X"])
    search = _build_search(monkeypatch, zot, chroma, config_path=config_path)

    stats = search.update_database()

    assert stats["processed_items"] == 0
    assert stats["added_items"] == 0
    assert stats["deleted_items"] == 0
    # No ingest calls (no items() fetch, no item_versions(since=...) etc.)
    # Must have called last_modified_version to compare
    assert ("last_modified_version",) in zot.calls


def test_update_database_default_mode_skips_api_fulltext(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path)
    zot = FakeZoteroClient()
    zot.load_scenario(
        [_paper("A")],
        fulltext={"A": {"content": "this should NOT appear"}},
        library_version=3,
    )
    chroma = FakeChromaClient()
    search = _build_search(monkeypatch, zot, chroma, config_path=config_path)

    search.update_database()

    # fulltext_item should never be called
    assert not any(c[0] == "fulltext_item" for c in zot.calls)
    indexed_documents = [
        document
        for documents, _metadatas, _ids in chroma.added
        for document in documents
    ]
    assert indexed_documents
    assert all("this should NOT appear" not in document for document in indexed_documents)


def test_metadata_only_noop_keeps_fulltext_index_state(
    monkeypatch, tmp_path, capsys
):
    config_path = _write_config(
        tmp_path,
        extra={
            "last_sync_version": 5,
            "fulltext": False,
            "indexed_fulltext": True,
        },
    )
    zot = FakeZoteroClient()
    zot.load_scenario(
        [_paper("A")],
        fulltext={"A": {"content": "must not be indexed"}},
        library_version=5,
    )
    chroma = FakeChromaClient(preloaded_ids=["A"])
    search = _build_search(monkeypatch, zot, chroma, config_path=config_path)

    stats = search.update_database()

    assert stats["processed_items"] == 0
    assert chroma.reset_calls == 0
    assert not any(c[0] == "fulltext_item" for c in zot.calls)
    assert chroma.added == []
    assert "full-text mode" not in capsys.readouterr().err
    saved = json.loads(open(config_path).read())
    assert saved["semantic_search"]["indexed_fulltext"] is True


def test_metadata_change_preserves_existing_fulltext_vectors(monkeypatch, tmp_path):
    config_path = _write_config(
        tmp_path,
        extra={
            "last_sync_version": 5,
            "fulltext": False,
            "indexed_fulltext": True,
        },
    )
    changed = _paper("A", title="Revised title", version=8)
    changed["data"]["dateModified"] = "2026-08-01T00:00:00Z"
    zot = FakeZoteroClient()
    zot.load_scenario([changed], library_version=8)
    zot.versions_state = {"A": 8}
    chroma = FakeChromaClient(
        preloaded_ids=["A"],
        preloaded_metadata={
            "A": {
                "has_fulltext": True,
                "fulltext_source": "betterissa-indexing",
                "attachment_signature": "same",
            }
        },
    )
    search = _build_search(monkeypatch, zot, chroma, config_path=config_path)

    stats = search.update_database()

    assert chroma.added == []
    assert chroma.deleted == []
    assert chroma.metadata_updates[0][0] == "A"
    assert chroma.metadata_by_key["A"]["title"] == "Revised title"
    assert chroma.metadata_by_key["A"]["has_fulltext"] is True
    assert stats["preserved_fulltext_items"] == 1


def test_complete_fulltext_enrichment_promotes_index_state(monkeypatch, tmp_path, capsys):
    config_path = _write_config(
        tmp_path,
        extra={
            "last_sync_version": 5,
            "fulltext": False,
            "indexed_fulltext": False,
        },
    )
    zot = FakeZoteroClient()
    zot.load_scenario([_paper("A")], library_version=8)
    chroma = FakeChromaClient(preloaded_ids=["A"])
    search = _build_search(monkeypatch, zot, chroma, config_path=config_path)
    monkeypatch.setattr(search, "_get_items_from_source", lambda **kwargs: [])
    monkeypatch.setattr(
        search, "_verify_local_snapshot_version", lambda version: version
    )

    search.update_database(fulltext=True)

    assert "full-text mode" not in capsys.readouterr().err
    saved = json.loads(open(config_path).read())
    assert saved["semantic_search"]["indexed_fulltext"] is True


def test_metadata_first_update_downgrades_when_source_was_deleted(
    monkeypatch, tmp_path
):
    class Reader:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get_item_by_key(self, key):
            return SimpleNamespace(item_id=1)

        def get_fulltext_meta_for_item(self, item_id):
            return [["STALE", "storage:page.html", "text/html"]]

        def get_attachment_signature(self, item_id, allowed_attachment_keys):
            assert allowed_attachment_keys == set()
            return "empty-signature"

        def extract_fulltext_for_item(self, item_id, allowed_attachment_keys):
            assert allowed_attachment_keys == set()
            return None

    config_path = _write_config(tmp_path)
    zot = FakeZoteroClient()
    item = _paper("A")
    zot.load_scenario([item], children={"A": []}, library_version=8)
    chroma = FakeChromaClient(
        preloaded_ids=["A"],
        preloaded_metadata={
            "A": {
                "has_fulltext": True,
                "attachment_signature": "old-signature",
            }
        },
    )
    search = _build_search(monkeypatch, zot, chroma, config_path=config_path)
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: True)
    monkeypatch.setattr(
        semantic_search, "LocalZoteroReader", lambda **kwargs: Reader()
    )

    remaining, preserved_items, _records, complete = (
        search._maintain_existing_fulltext([item])
    )

    assert complete is True
    assert preserved_items == 0
    assert len(remaining) == 1
    assert remaining[0]["data"]["fulltext"] == ""
    assert remaining[0]["data"]["fulltext_attempted"] is True
    assert (
        remaining[0]["data"]["fulltextError"]
        == "No candidate full-text attachment was available."
    )


def test_metadata_first_update_keeps_unchanged_fulltext(monkeypatch, tmp_path):
    class Reader:
        extract_calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get_item_by_key(self, key):
            return SimpleNamespace(item_id=1)

        def get_fulltext_meta_for_item(self, item_id):
            return [["ATT1", "storage:paper.pdf", "application/pdf"]]

        def get_attachment_signature(self, item_id, allowed_attachment_keys):
            return "same-signature"

        def extract_fulltext_for_item(self, item_id, allowed_attachment_keys):
            self.extract_calls += 1
            return ("should not be extracted", "pdf")

    attachment = _paper("ATT1", item_type="attachment")
    attachment["data"]["parentItem"] = "A"
    zot = FakeZoteroClient()
    item = _paper("A", title="Updated metadata title")
    zot.load_scenario([item], children={"A": [attachment]}, library_version=8)
    chroma = FakeChromaClient(
        preloaded_ids=["A"],
        preloaded_metadata={
            "A": {
                "has_fulltext": True,
                "attachment_signature": "same-signature",
            }
        },
    )
    search = _build_search(
        monkeypatch,
        zot,
        chroma,
        config_path=_write_config(tmp_path),
    )
    reader = Reader()
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: True)
    monkeypatch.setattr(
        semantic_search, "LocalZoteroReader", lambda **kwargs: reader
    )

    remaining, preserved_items, preserved_records, complete = (
        search._maintain_existing_fulltext([item])
    )

    assert remaining == []
    assert complete is True
    assert preserved_items == preserved_records == 1
    assert reader.extract_calls == 0
    assert chroma.metadata_by_key["A"]["title"] == "Updated metadata title"


def test_content_contract_change_warns_without_automatic_rebuild(
    monkeypatch, tmp_path, capsys
):
    config_path = _write_config(
        tmp_path,
        extra={
            "last_sync_version": 5,
            "fulltext": False,
            "indexed_fulltext": False,
            "indexed_content_signature": "legacy-v1",
        },
    )
    zot = FakeZoteroClient()
    zot.load_scenario([_paper("A")], library_version=5)
    chroma = FakeChromaClient(preloaded_ids=["A"])
    search = _build_search(monkeypatch, zot, chroma, config_path=config_path)

    stats = search.update_database()

    assert stats["processed_items"] == 0
    assert chroma.reset_calls == 0
    assert chroma.added == []
    assert "will not re-embed unchanged items" in capsys.readouterr().err
    saved = json.loads(open(config_path).read())
    assert (
        saved["semantic_search"]["indexed_content_signature"]
        == "legacy-v1"
    )


def test_update_database_force_rebuild_stages_replacement_after_pruning(
    monkeypatch, tmp_path
):
    """A force rebuild prunes first, then swaps in a staged full scan."""
    config_path = _write_config(tmp_path, extra={"last_sync_version": 100})
    zot = FakeZoteroClient()
    zot.load_scenario([_paper("A"), _paper("B")], library_version=120)
    zot.versions_state = {"A": 120, "B": 120}
    chroma = FakeChromaClient(
        preloaded_ids=["A", "STALE"],
        preloaded_metadata={"A": {"title": "Old title"}},
    )
    search = _build_search(monkeypatch, zot, chroma, config_path=config_path)

    search.update_database(force_full_rebuild=True)

    assert chroma.reset_calls == 0
    assert chroma.staged_rebuild_calls == 1
    assert chroma.staged_commit_calls == 1
    assert chroma.operation_events[:3] == [
        "delete",
        "metadata",
        "begin_staging",
    ]
    # No since-based fetch: full scan used items() not item_versions(since=...)
    assert not any(c[0] == "item_versions" and c[1] is not None for c in zot.calls)


def test_force_clear_preserves_destructive_rebuild_escape_hatch(
    monkeypatch, tmp_path
):
    config_path = _write_config(tmp_path, extra={"last_sync_version": 100})
    zot = FakeZoteroClient()
    zot.load_scenario([_paper("A")], library_version=120)
    chroma = FakeChromaClient(preloaded_ids=["STALE"])
    search = _build_search(monkeypatch, zot, chroma, config_path=config_path)

    search.update_database(force_full_rebuild=True, force_clear=True)

    assert chroma.reset_calls == 1
    assert chroma.staged_rebuild_calls == 0


def test_force_clear_requires_force_rebuild(monkeypatch, tmp_path):
    search = _build_search(
        monkeypatch,
        FakeZoteroClient(),
        FakeChromaClient(),
        config_path=_write_config(tmp_path),
    )

    stats = search.update_database(force_clear=True)

    assert stats["error"] == "force_clear requires force_full_rebuild"


def test_force_clear_rejects_openai_batch_mode(monkeypatch, tmp_path):
    chroma = FakeChromaClient()
    chroma.embedding_model = "openai"
    search = _build_search(
        monkeypatch,
        FakeZoteroClient(),
        chroma,
        config_path=_write_config(tmp_path),
    )

    stats = search.update_database(
        force_full_rebuild=True,
        force_clear=True,
        use_openai_batch=True,
    )

    assert "cannot be combined with OpenAI Batch" in stats["error"]


def test_failed_staged_rebuild_keeps_previous_records(monkeypatch, tmp_path):
    class FailingChroma(FakeChromaClient):
        def upsert_documents(self, documents, metadatas, ids):
            self.operation_events.append("failed_upsert")
            raise RuntimeError("encoder unavailable")

    config_path = _write_config(tmp_path, extra={"last_sync_version": 100})
    zot = FakeZoteroClient()
    zot.load_scenario([_paper("A")], library_version=120)
    chroma = FailingChroma(
        preloaded_ids=["A"],
        preloaded_metadata={"A": {"title": "Old title"}},
    )
    search = _build_search(monkeypatch, zot, chroma, config_path=config_path)

    stats = search.update_database(force_full_rebuild=True)

    assert stats["errors"] == 1
    assert chroma.staged_commit_calls == 0
    assert chroma.staged_abort_calls == 1
    assert chroma._ids == {"A"}
    assert chroma.metadata_by_key["A"]["title"] == "Paper"


def test_unverified_local_snapshot_never_replaces_live_index(
    monkeypatch, tmp_path
):
    config_path = _write_config(tmp_path, extra={"last_sync_version": 100})
    zot = FakeZoteroClient()
    zot.load_scenario([_paper("A")], library_version=120)
    chroma = FakeChromaClient(preloaded_ids=["A"])
    search = _build_search(monkeypatch, zot, chroma, config_path=config_path)
    monkeypatch.setattr(
        search,
        "_get_items_from_source",
        lambda **_kwargs: [_paper("A")],
    )
    monkeypatch.setattr(
        search,
        "_verify_local_snapshot_version",
        lambda _version: None,
    )

    stats = search.update_database(
        force_full_rebuild=True,
        fulltext=True,
    )

    assert "could not be verified as complete" in stats["error"]
    assert chroma.staged_rebuild_calls == 0
    assert chroma._ids == {"A"}


def test_api_full_scan_prunes_before_upserting(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path, extra={"last_sync_version": 0})
    zot = FakeZoteroClient()
    zot.load_scenario([_paper("A")], library_version=120)
    chroma = FakeChromaClient(
        preloaded_ids=["A", "REMOVED"],
        preloaded_metadata={"A": {"title": "Old title"}},
    )
    search = _build_search(monkeypatch, zot, chroma, config_path=config_path)

    stats = search.update_database()

    assert stats["deleted_items"] == 1
    assert chroma.deleted == ["REMOVED"]
    assert chroma.operation_events.index("delete") < chroma.operation_events.index(
        "upsert"
    )
    assert chroma.operation_events.index("metadata") < chroma.operation_events.index(
        "upsert"
    )


def test_update_database_force_rebuild_updates_last_sync_version(monkeypatch, tmp_path):
    """After force_full_rebuild, last_sync_version must advance to the library's
    current version. Otherwise the next incremental run would use a stale
    watermark and permanently lose items not modified since that older version.
    """
    config_path = _write_config(tmp_path, extra={"last_sync_version": 50})
    zot = FakeZoteroClient()
    zot.load_scenario([_paper("A")], library_version=200)
    zot.versions_state = {"A": 200}
    chroma = FakeChromaClient()
    search = _build_search(monkeypatch, zot, chroma, config_path=config_path)

    search.update_database(force_full_rebuild=True)

    saved = json.loads(open(config_path).read())
    assert saved["semantic_search"]["last_sync_version"] == 200


# --------- Config loaders ----------

def test_load_fulltext_defaults_to_false(monkeypatch, tmp_path):
    # No config file exists
    search = _build_search(monkeypatch, FakeZoteroClient(), FakeChromaClient(),
                           config_path=str(tmp_path / "missing.json"))
    assert search._load_fulltext_setting() is False


def test_load_fulltext_respects_true(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path, extra={"fulltext": True})
    search = _build_search(monkeypatch, FakeZoteroClient(), FakeChromaClient(),
                           config_path=config_path)
    assert search._load_fulltext_setting() is True


def test_load_last_sync_version_defaults_zero(monkeypatch, tmp_path):
    search = _build_search(monkeypatch, FakeZoteroClient(), FakeChromaClient(),
                           config_path=str(tmp_path / "missing.json"))
    assert search._load_last_sync_version() == 0


def test_load_last_sync_version_reads_int(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path, extra={"last_sync_version": 123})
    search = _build_search(monkeypatch, FakeZoteroClient(), FakeChromaClient(),
                           config_path=config_path)
    assert search._load_last_sync_version() == 123
