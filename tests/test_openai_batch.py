import builtins
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from zotero_mcp import openai_batch, setup_helper

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

pytest.importorskip("chromadb")

from zotero_mcp import semantic_search  # noqa: E402
from zotero_mcp.chroma_client import ChromaClient  # noqa: E402


def test_build_embedding_request_uses_batch_embeddings_shape():
    record = {"id": "ABC123", "document": "paper text", "metadata": {"title": "A"}}

    request = openai_batch.build_embedding_request(record, "text-embedding-3-small")

    assert request == {
        "custom_id": "ABC123",
        "method": "POST",
        "url": "/v1/embeddings",
        "body": {
            "model": "text-embedding-3-small",
            "input": "paper text",
            "encoding_format": "float",
        },
    }


def test_split_embedding_records_respects_request_limit():
    records = [
        {"id": f"ID{i}", "document": f"text {i}", "metadata": {}}
        for i in range(3)
    ]

    chunks = openai_batch.split_embedding_records(
        records,
        "text-embedding-3-small",
        max_requests=2,
        max_file_bytes=10_000,
    )

    assert [len(chunk_records) for chunk_records, _ in chunks] == [2, 1]
    assert chunks[0][1][0]["custom_id"] == "ID0"
    assert chunks[1][1][0]["custom_id"] == "ID2"


def test_parse_embedding_output_uses_custom_ids_and_keeps_failures():
    output = "\n".join(
        [
            json.dumps({
                "custom_id": "B",
                "response": {"status_code": 200, "body": {"data": [{"embedding": [0.2, 0.3]}]}},
            }),
            json.dumps({
                "custom_id": "A",
                "response": {"status_code": 200, "body": {"data": [{"embedding": [0.1, 0.2]}]}},
            }),
            json.dumps({
                "custom_id": "C",
                "response": {"status_code": 429, "body": {}},
                "error": {"message": "rate limited"},
            }),
        ]
    )

    embeddings, failures = openai_batch.parse_embedding_output(output)

    assert embeddings == {"B": [0.2, 0.3], "A": [0.1, 0.2]}
    assert failures == [{"custom_id": "C", "error": {"message": "rate limited"}, "status_code": 429}]


def test_submit_embedding_batches_writes_manifest_and_jsonl(tmp_path):
    class FakeFiles:
        def create(self, file, purpose):
            assert purpose == "batch"
            assert Path(file.name).read_text(encoding="utf-8")
            return SimpleNamespace(id="file-1")

    class FakeBatches:
        def create(self, **kwargs):
            assert kwargs["endpoint"] == "/v1/embeddings"
            assert kwargs["completion_window"] == "24h"
            return SimpleNamespace(
                id="batch-1",
                status="validating",
                output_file_id=None,
                error_file_id=None,
                request_counts={"total": 1, "completed": 0, "failed": 0},
            )

    records = [{"id": "ABC123", "document": "paper text", "metadata": {"title": "A"}}]

    manifest = openai_batch.submit_embedding_batches(
        records=records,
        model_name="text-embedding-3-small",
        embedding_config={"api_key": "test"},
        config_path=str(tmp_path / "config.json"),
        source_server_id="local-database",
        metadata_only_records=[
            {"id": "UNCHANGED", "metadata": {"title": "Current title"}},
        ],
        client=SimpleNamespace(files=FakeFiles(), batches=FakeBatches()),
    )

    assert manifest["batches"][0]["batch_id"] == "batch-1"
    assert manifest["source_server_id"] == "local-database"
    assert manifest["source_guard_version"] == 2
    assert Path(manifest["manifest_path"]).exists()
    input_rows = openai_batch.read_jsonl(Path(manifest["batches"][0]["input_path"]))
    assert input_rows[0]["url"] == "/v1/embeddings"
    assert openai_batch.read_jsonl(
        Path(manifest["metadata_only_records_path"])
    ) == [{"id": "UNCHANGED", "metadata": {"title": "Current title"}}]


def test_status_refresh_preserves_newer_import_bookkeeping(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    latest = {
        "run_id": "run-1",
        "manifest_path": str(manifest_path),
        "batches": [{
            "batch_id": "batch-1",
            "status": "in_progress",
            "imported_at": "2026-08-06T12:00:00",
            "imported_count": 7,
        }],
    }
    openai_batch.save_manifest(latest)
    stale = {
        "run_id": "run-1",
        "manifest_path": str(manifest_path),
        "batches": [{
            "batch_id": "batch-1",
            "status": "in_progress",
            "imported_at": None,
            "imported_count": 0,
        }],
    }
    client = SimpleNamespace(
        batches=SimpleNamespace(
            retrieve=lambda _batch_id: SimpleNamespace(
                status="completed",
                output_file_id="output-1",
                error_file_id=None,
                request_counts={"completed": 7},
            )
        )
    )

    refreshed = openai_batch.refresh_manifest_status(
        stale,
        embedding_config=None,
        client=client,
    )

    batch = refreshed["batches"][0]
    assert batch["status"] == "completed"
    assert batch["output_file_id"] == "output-1"
    assert batch["imported_at"] == "2026-08-06T12:00:00"
    assert batch["imported_count"] == 7


def test_download_file_text_uses_atomic_persistence(tmp_path, monkeypatch):
    output_path = tmp_path / "output.jsonl"
    writes = []
    monkeypatch.setattr(
        openai_batch,
        "atomic_write_text",
        lambda path, content: writes.append((path, content)),
    )
    client = SimpleNamespace(
        files=SimpleNamespace(content=lambda _file_id: b"complete output")
    )

    text = openai_batch.download_file_text(client, "file-1", output_path)

    assert text == "complete output"
    assert writes == [(output_path, "complete output")]


def test_import_locks_before_reading_manifest(monkeypatch):
    events = []

    class Lock:
        def __enter__(self):
            events.append("lock-enter")
            return True

        def __exit__(self, *_args):
            events.append("lock-exit")

    def find_manifest(**_kwargs):
        events.append("manifest-read")
        raise RuntimeError("stop after manifest read")

    monkeypatch.setattr(semantic_search, "_acquire_update_lock", lambda _path: Lock())
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(semantic_search.openai_batch, "find_manifest", find_manifest)
    search = semantic_search.ZoteroSemanticSearch(chroma_client=FakeChromaClient())

    with pytest.raises(RuntimeError, match="stop after manifest read"):
        search.import_openai_batch()

    assert events == ["lock-enter", "manifest-read", "lock-exit"]


def test_setup_openai_new_config_defaults_to_batch(monkeypatch):
    answers = iter([
        "2",  # OpenAI
        "1",  # text-embedding-3-small
        "",  # default base URL
        "",  # default batch choice: yes for new configs
        "1",  # manual updates
        "",  # default PDF max pages
        "",  # auto-detect DB path
    ])
    monkeypatch.setattr(builtins, "input", lambda *args: next(answers))
    monkeypatch.setattr(setup_helper.getpass, "getpass", lambda *args: "sk-test")

    config = setup_helper.setup_semantic_search()

    assert config["embedding_model"] == "openai"
    assert config["openai_batch"] == {"enabled": True}
    assert config["chunking"]["max_chunks_per_item"] == 768


class FakeChromaClient:
    def __init__(self, embedding_model="openai"):
        self.embedding_model = embedding_model
        self.embedding_config = {"model_name": "text-embedding-3-small", "api_key": "test"}
        self.embedding_max_tokens = 8000

    def truncate_text(self, text, max_tokens=None):
        return text

    def get_existing_ids(self, ids):
        return {"EXISTING"} & set(ids)


def test_update_db_batch_flag_resolution_reads_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"semantic_search": {"openai_batch": {"enabled": True}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    search = semantic_search.ZoteroSemanticSearch(
        chroma_client=FakeChromaClient(),
        config_path=str(config_path),
    )

    assert search._resolve_openai_batch_enabled(None) is True
    assert search._resolve_openai_batch_enabled(False) is False
    assert search._resolve_openai_batch_enabled(True) is True

    non_openai = semantic_search.ZoteroSemanticSearch(
        chroma_client=FakeChromaClient(embedding_model="gemini"),
        config_path=str(config_path),
    )
    assert non_openai._resolve_openai_batch_enabled(True) is False


def test_empty_force_rebuild_batch_commits_an_empty_staged_collection(
    monkeypatch,
):
    class EmptyRebuildClient(FakeChromaClient):
        def __init__(self):
            super().__init__()
            self.events = []
            self.reconciled = []

        def begin_staged_rebuild(self):
            self.events.append("begin")

        def commit_staged_rebuild(self):
            self.events.append("commit")

        def abort_staged_rebuild(self):
            self.events.append("abort")

        def reconcile_item_records(self, parent, expected_ids):
            self.reconciled.append((parent, set(expected_ids)))

    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(
        semantic_search.openai_batch,
        "submit_embedding_batches",
        lambda **_kwargs: pytest.fail("an empty rebuild must not submit a batch"),
    )
    client = EmptyRebuildClient()
    search = semantic_search.ZoteroSemanticSearch(chroma_client=client)
    monkeypatch.setattr(search, "_save_update_config", lambda **_kwargs: None)

    stats = search._submit_openai_batch_index(
        [
            {
                "key": "EMPTY",
                "data": {
                    "title": "",
                    "abstractNote": "",
                    "itemType": "journalArticle",
                    "creators": [],
                },
            }
        ],
        force_full_rebuild=True,
        target_sync_version=1,
        stats={
            "processed_items": 0,
            "skipped_items": 0,
            "errors": 0,
        },
    )

    assert stats["batch_submitted"] is False
    assert stats["skipped_items"] == 1
    assert client.events == ["begin", "commit"]
    assert client.reconciled == [("EMPTY", set())]


def test_failed_batch_submit_does_not_report_added_or_updated(monkeypatch):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    search = semantic_search.ZoteroSemanticSearch(chroma_client=FakeChromaClient())

    def fail_submit(**kwargs):
        raise RuntimeError("missing files.write")

    monkeypatch.setattr(semantic_search.openai_batch, "submit_embedding_batches", fail_submit)
    stats = {
        "processed_items": 0,
        "added_items": 0,
        "updated_items": 0,
        "skipped_items": 0,
        "errors": 0,
    }

    with pytest.raises(RuntimeError, match="missing files.write"):
        search._submit_openai_batch_index(
            [
                {
                    "key": "EXISTING",
                    "data": {
                        "title": "Existing item",
                        "itemType": "journalArticle",
                        "abstractNote": "A",
                        "creators": [],
                    },
                }
            ],
            force_full_rebuild=False,
            target_sync_version=1,
            stats=stats,
        )

    assert stats["processed_items"] == 1
    assert stats["added_items"] == 0
    assert stats["updated_items"] == 0
    assert "estimated_added_items" not in stats
    assert "estimated_updated_items" not in stats


def test_batch_and_realtime_indexing_share_prepared_payload(monkeypatch):
    class RecordingChromaClient(FakeChromaClient):
        def __init__(self):
            super().__init__()
            self.upserts = []

        def upsert_documents(self, documents, metadatas, ids):
            self.upserts.append({
                "documents": list(documents),
                "metadatas": list(metadatas),
                "ids": list(ids),
            })

    item = {
        "key": "ITEM1",
        "data": {
            "itemType": "journalArticle",
            "title": "Semantic Batch Paper",
            "abstractNote": "An abstract that should be embedded.",
            "publicationTitle": "Journal of Tests",
            "creators": [{"firstName": "Ada", "lastName": "Lovelace", "creatorType": "author"}],
            "tags": [{"tag": "embeddings"}, {"tag": "batch"}],
            "fulltext": "Full text must be included in both indexing paths.",
            "fulltextSource": "zotero_web_api",
            "date": "2026",
            "DOI": "10.1234/example",
        },
    }

    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())

    realtime_client = RecordingChromaClient()
    realtime_search = semantic_search.ZoteroSemanticSearch(chroma_client=realtime_client)
    realtime_search._process_item_batch([item])
    assert realtime_client.upserts

    captured = {}

    def capture_batch_submit(**kwargs):
        captured.update(kwargs)
        return {
            "run_id": "run-1",
            "manifest_path": "/tmp/manifest.json",
            "batches": [{"batch_id": "batch-1"}],
        }

    monkeypatch.setattr(semantic_search.openai_batch, "submit_embedding_batches", capture_batch_submit)
    batch_client = RecordingChromaClient()
    batch_search = semantic_search.ZoteroSemanticSearch(chroma_client=batch_client)
    batch_search._submit_openai_batch_index(
        [item],
        force_full_rebuild=False,
        target_sync_version=123,
        stats={
            "processed_items": 0,
            "skipped_items": 0,
            "errors": 0,
        },
    )

    realtime_payload = realtime_client.upserts[0]
    assert captured["records"] == [{
        "id": realtime_payload["ids"][0],
        "document": realtime_payload["documents"][0],
        "metadata": realtime_payload["metadatas"][0],
    }]
    assert "Full text must be included" in captured["records"][0]["document"]
    assert captured["records"][0]["metadata"]["has_fulltext"] is True


def test_batch_reuses_unchanged_vectors_without_submission(monkeypatch):
    class ReusingClient(FakeChromaClient):
        def __init__(self):
            super().__init__()
            self.metadata_updates = []
            self.reconciled = []

        def get_records(self, ids):
            return {
                doc_id: {
                    "document": "Existing item\n\nA",
                    "metadata": {},
                }
                for doc_id in ids
            }

        def update_metadatas(self, ids, metadatas):
            self.metadata_updates.extend(zip(ids, metadatas, strict=True))

        def reconcile_item_records(self, parent, expected_ids):
            self.reconciled.append((parent, set(expected_ids)))

    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(
        semantic_search.openai_batch,
        "submit_embedding_batches",
        lambda **kwargs: pytest.fail("unchanged text must not be submitted"),
    )
    client = ReusingClient()
    search = semantic_search.ZoteroSemanticSearch(chroma_client=client)
    saved = []
    monkeypatch.setattr(search, "_save_update_config", lambda **kwargs: saved.append(kwargs))

    stats = search._submit_openai_batch_index(
        [{
            "key": "EXISTING",
            "data": {
                "title": "Existing item",
                "abstractNote": "A",
                "itemType": "journalArticle",
                "creators": [],
            },
        }],
        force_full_rebuild=False,
        target_sync_version=42,
        stats={
            "processed_items": 0,
            "skipped_items": 0,
            "errors": 0,
        },
    )

    assert stats["batch_submitted"] is False
    assert stats["reused_embeddings"] == 1
    assert len(client.metadata_updates) == 1
    assert client.reconciled == [("EXISTING", {"EXISTING"})]
    assert saved == [{
        "last_sync_version": 42,
        "indexed_fulltext": False,
        "indexed_content_signature": "lean-paper-content-v2",
    }]


def test_batch_manifest_keeps_reused_chunks_in_expected_layout(monkeypatch):
    class ReusingClient(FakeChromaClient):
        def __init__(self):
            super().__init__()
            self.records = {}
            self.metadata_updates = []

        def get_records(self, ids):
            return {
                doc_id: self.records[doc_id]
                for doc_id in ids
                if doc_id in self.records
            }

        def update_metadatas(self, ids, metadatas):
            self.metadata_updates.extend(zip(ids, metadatas, strict=True))

    item = {
        "key": "MIXED",
        "data": {
            "title": "Mixed item",
            "abstractNote": "Abstract",
            "itemType": "journalArticle",
            "creators": [],
            "fulltext": "First passage. " * 80,
            "fulltextSource": "zotero_web_api",
        },
    }
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    client = ReusingClient()
    search = semantic_search.ZoteroSemanticSearch(chroma_client=client)
    search._chunking_config = {
        "enabled": True,
        "chunk_size": 120,
        "overlap": 20,
        "max_chunks_per_item": 10,
    }
    prepared = search._prepare_item_batch([item])
    reused_id = prepared.ids[0]
    client.records[reused_id] = {
        "document": prepared.documents[0],
        "metadata": {},
    }
    captured = {}

    def capture_submit(**kwargs):
        captured.update(kwargs)
        return {
            "run_id": "run-mixed",
            "manifest_path": "/tmp/manifest.json",
            "batches": [{"batch_id": "batch-mixed"}],
        }

    monkeypatch.setattr(
        semantic_search.openai_batch,
        "submit_embedding_batches",
        capture_submit,
    )

    search._submit_openai_batch_index(
        [item],
        force_full_rebuild=False,
        target_sync_version=42,
        stats={
            "processed_items": 0,
            "skipped_items": 0,
            "errors": 0,
        },
    )

    submitted_ids = {record["id"] for record in captured["records"]}
    assert reused_id not in submitted_ids
    assert captured["expected_ids_by_item"] == {
        "MIXED": sorted(prepared.expected_ids),
    }
    assert [
        record["id"] for record in captured["metadata_only_records"]
    ] == [reused_id]
    assert client.metadata_updates == []


def test_batch_submission_counts_parent_items_not_passage_records(monkeypatch):
    class Client(FakeChromaClient):
        embedding_model = "openai"
        embedding_config = {"model_name": "text-embedding-3-small"}

        def get_item_embedding_hashes(self, item_key):
            return {f"{item_key}#0": "old"} if item_key == "EXISTING" else {}

        def get_item_record_hashes(self, _item_key):
            return {}

    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    search = semantic_search.ZoteroSemanticSearch(chroma_client=Client())
    prepared = semantic_search._PreparedIndexBatch(
        documents=["a", "b", "c"],
        metadatas=[
            {"parent_item_key": "EXISTING"},
            {"parent_item_key": "EXISTING"},
            {"parent_item_key": "NEWITEM"},
        ],
        ids=["EXISTING#0", "EXISTING#1", "NEWITEM#0"],
        expected_ids=["EXISTING#0", "EXISTING#1", "NEWITEM#0"],
        metadata_only_ids=[],
        metadata_only_metadatas=[],
        item_keys=["EXISTING", "NEWITEM"],
        stats={
            "processed": 2,
            "added": 0,
            "updated": 0,
            "skipped": 0,
            "errors": 0,
            "reused_embeddings": 0,
        },
    )
    monkeypatch.setattr(
        search,
        "_prepare_index_records",
        lambda items, force_rebuild: (
            [
                {"id": doc_id, "document": document, "metadata": metadata}
                for doc_id, document, metadata in zip(
                    prepared.ids,
                    prepared.documents,
                    prepared.metadatas,
                    strict=True,
                )
            ],
            prepared.stats,
            prepared,
        ),
    )
    monkeypatch.setattr(
        semantic_search.openai_batch,
        "submit_embedding_batches",
        lambda **_kwargs: {
            "run_id": "run-1",
            "manifest_path": "/tmp/manifest.json",
            "batches": [{"batch_id": "batch-1"}],
        },
    )
    stats = {
        "processed_items": 0,
        "skipped_items": 0,
        "errors": 0,
        "reused_embeddings": 0,
    }

    search._submit_openai_batch_index(
        [{}, {}],
        force_full_rebuild=False,
        target_sync_version=10,
        stats=stats,
    )

    assert stats["submitted_items"] == 2
    assert stats["submitted_records"] == 3
    assert stats["estimated_updated_items"] == 1
    assert stats["estimated_added_items"] == 1


def test_chroma_client_upsert_embeddings_passes_precomputed_vectors():
    class FakeCollection:
        def __init__(self):
            self.kwargs = None

        def upsert(self, **kwargs):
            self.kwargs = kwargs

    client = ChromaClient.__new__(ChromaClient)
    client.collection = FakeCollection()

    client.upsert_embeddings(
        documents=["doc"],
        metadatas=[{"title": "Title"}],
        ids=["ID1"],
        embeddings=[[0.1, 0.2]],
    )

    assert client.collection.kwargs == {
        "documents": ["doc"],
        "metadatas": [{"title": "Title"}],
        "ids": ["ID1"],
        "embeddings": [[0.1, 0.2]],
    }


def test_import_openai_batch_reports_records_missing_from_output(tmp_path, monkeypatch):
    class ImportChromaClient(FakeChromaClient):
        def __init__(self):
            super().__init__()
            self.upserted = None
            self.reset_count = 0

        def upsert_embeddings(self, documents, metadatas, ids, embeddings):
            self.upserted = {
                "documents": list(documents),
                "metadatas": list(metadatas),
                "ids": list(ids),
                "embeddings": list(embeddings),
            }

        def reset_collection(self):
            self.reset_count += 1

    records_path = tmp_path / "batch-001-records.jsonl"
    openai_batch.write_jsonl(
        records_path,
        [
            {"id": "A", "document": "doc A", "metadata": {"title": "A"}},
            {"id": "B", "document": "doc B", "metadata": {"title": "B"}},
        ],
    )
    output_path = tmp_path / "batch-001-records-output.jsonl"
    output_path.write_text(
        json.dumps({
            "custom_id": "A",
            "response": {"status_code": 200, "body": {"data": [{"embedding": [0.1, 0.2]}]}},
        }) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "run_id": "run-1",
        "manifest_path": str(tmp_path / "manifest.json"),
        "force_full_rebuild": False,
        "batches": [
            {
                "batch_id": "batch-1",
                "status": "completed",
                "output_file_id": "file-1",
                "records_path": str(records_path),
                "imported_at": None,
            }
        ],
    }

    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(semantic_search.openai_batch, "find_manifest", lambda **kwargs: manifest)
    monkeypatch.setattr(
        semantic_search.openai_batch,
        "refresh_manifest_status",
        lambda manifest, **kwargs: manifest,
    )
    monkeypatch.setattr(semantic_search.openai_batch, "create_openai_client", lambda config: object())

    chroma_client = ImportChromaClient()
    search = semantic_search.ZoteroSemanticSearch(chroma_client=chroma_client)

    stats = search.import_openai_batch()

    assert chroma_client.upserted is None
    assert stats["imported_items"] == 0
    assert stats["missing_items"] == 1
    assert {
        "custom_id": "B",
        "error": "No embedding or error row returned for batch record",
    } in stats["errors"]
    assert manifest["batches"][0]["imported_at"] is None
    assert manifest["batches"][0]["import_error_count"] == 1


def test_force_batch_partial_output_does_not_reset_existing_index(
    tmp_path,
    monkeypatch,
):
    records_path = tmp_path / "records.jsonl"
    openai_batch.write_jsonl(
        records_path,
        [
            {"id": "A#0", "document": "a", "metadata": {"parent_item_key": "A"}},
            {"id": "A#1", "document": "b", "metadata": {"parent_item_key": "A"}},
        ],
    )
    records_path.with_name("records-output.jsonl").write_text(
        json.dumps({
            "custom_id": "A#0",
            "response": {"status_code": 200, "body": {"data": [{"embedding": [0.1]}]}},
        }) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "run_id": "force-partial",
        "manifest_path": str(tmp_path / "manifest.json"),
        "force_full_rebuild": True,
        "batches": [{
            "batch_id": "batch-1",
            "status": "completed",
            "output_file_id": "file-1",
            "records_path": str(records_path),
            "imported_at": None,
        }],
    }

    class Client(FakeChromaClient):
        def __init__(self):
            super().__init__()
            self.reset_count = 0

        def reset_collection(self):
            self.reset_count += 1

    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(semantic_search.openai_batch, "find_manifest", lambda **kwargs: manifest)
    monkeypatch.setattr(semantic_search.openai_batch, "refresh_manifest_status", lambda value, **kwargs: value)
    monkeypatch.setattr(semantic_search.openai_batch, "create_openai_client", lambda config: object())
    client = Client()

    stats = semantic_search.ZoteroSemanticSearch(
        chroma_client=client
    ).import_openai_batch()

    assert stats["missing_items"] == 1
    assert client.reset_count == 0
    assert manifest["batches"][0]["imported_at"] is None


def test_complete_batch_reconciles_before_advancing_watermark(tmp_path, monkeypatch):
    records_path = tmp_path / "records.jsonl"
    records = [
        {"id": "A#0", "document": "a", "metadata": {"parent_item_key": "A"}},
        {"id": "A#1", "document": "b", "metadata": {"parent_item_key": "A"}},
    ]
    openai_batch.write_jsonl(records_path, records)
    records_path.with_name("records-output.jsonl").write_text(
        "\n".join(
            json.dumps({
                "custom_id": record["id"],
                "response": {"status_code": 200, "body": {"data": [{"embedding": [0.1]}]}},
            })
            for record in records
        ) + "\n",
        encoding="utf-8",
    )
    metadata_only_path = tmp_path / "metadata-only-records.jsonl"
    openai_batch.write_jsonl(
        metadata_only_path,
        [{"id": "A#2", "metadata": {"title": "Refreshed"}}],
    )
    manifest = {
        "run_id": "complete",
        "manifest_path": str(tmp_path / "manifest.json"),
        "force_full_rebuild": False,
        "target_sync_version": 42,
        "fulltext": True,
        "content_signature": "contract-v2",
        "expected_ids_by_item": {"A": ["A#0", "A#1", "A#2"]},
        "metadata_only_records_path": str(metadata_only_path),
        "batches": [{
            "batch_id": "batch-1",
            "status": "completed",
            "output_file_id": "file-1",
            "records_path": str(records_path),
            "imported_at": None,
        }],
    }

    class Client(FakeChromaClient):
        def __init__(self):
            super().__init__()
            self.reconciled = []
            self.metadata_updates = []

        def upsert_embeddings(self, documents, metadatas, ids, embeddings):
            pass

        def reconcile_item_records(self, parent, expected_ids):
            self.reconciled.append((parent, set(expected_ids)))

        def update_metadatas(self, ids, metadatas):
            self.metadata_updates.append((list(ids), list(metadatas)))

    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(semantic_search.openai_batch, "find_manifest", lambda **kwargs: manifest)
    monkeypatch.setattr(semantic_search.openai_batch, "refresh_manifest_status", lambda value, **kwargs: value)
    monkeypatch.setattr(semantic_search.openai_batch, "create_openai_client", lambda config: object())
    client = Client()
    search = semantic_search.ZoteroSemanticSearch(chroma_client=client)
    saved = []
    monkeypatch.setattr(search, "_save_update_config", lambda **kwargs: saved.append(kwargs))

    search.import_openai_batch()

    assert client.metadata_updates == [
        (["A#2"], [{"title": "Refreshed"}]),
    ]
    assert client.reconciled == [("A", {"A#0", "A#1", "A#2"})]
    assert saved == [{
        "last_sync_version": 42,
        "indexed_fulltext": True,
        "indexed_content_signature": "contract-v2",
    }]


def test_batch_baseline_rejects_item_changed_after_submission(
    tmp_path, monkeypatch
):
    records_path = tmp_path / "records.jsonl"
    openai_batch.write_jsonl(
        records_path,
        [{
            "id": "A#0",
            "document": "submitted",
            "metadata": {
                "parent_item_key": "A",
                "embedding_content_sha256": "submitted-hash",
            },
        }],
    )
    manifest = {
        "baseline_embedding_hashes_by_item": {"A": {"A#0": "old-hash"}},
        "expected_ids_by_item": {"A": ["A#0"]},
        "batches": [{"records_path": str(records_path)}],
    }

    class Client(FakeChromaClient):
        def get_item_embedding_hashes(self, item_key):
            assert item_key == "A"
            return {"A#0": "newer-hash"}

    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    search = semantic_search.ZoteroSemanticSearch(chroma_client=Client())

    with pytest.raises(RuntimeError, match="changed after submission"):
        search._validate_openai_batch_baseline(manifest)


def test_batch_baseline_rejects_metadata_changed_after_submission(
    tmp_path, monkeypatch
):
    submitted_metadata = {
        "parent_item_key": "A",
        "embedding_content_sha256": "same-content",
        "title": "Submitted title",
    }
    records_path = tmp_path / "records.jsonl"
    openai_batch.write_jsonl(
        records_path,
        [{
            "id": "A#0",
            "document": "same document",
            "metadata": submitted_metadata,
        }],
    )
    baseline_metadata = dict(submitted_metadata, title="Old title")
    current_metadata = dict(submitted_metadata, title="Newer title")
    manifest = {
        "baseline_embedding_hashes_by_item": {
            "A": {"A#0": "same-content"}
        },
        "baseline_record_hashes_by_item": {
            "A": {
                "A#0": semantic_search.record_state_hash(
                    "same document", baseline_metadata
                )
            }
        },
        "expected_ids_by_item": {"A": ["A#0"]},
        "batches": [{"records_path": str(records_path)}],
    }

    class Client(FakeChromaClient):
        def get_item_embedding_hashes(self, item_key):
            return {"A#0": "same-content"}

        def get_item_record_hashes(self, item_key):
            return {
                "A#0": semantic_search.record_state_hash(
                    "same document", current_metadata
                )
            }

    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    search = semantic_search.ZoteroSemanticSearch(chroma_client=Client())

    with pytest.raises(RuntimeError, match="metadata changed after submission"):
        search._validate_openai_batch_baseline(manifest)


def test_force_batch_write_failure_discards_staging(tmp_path, monkeypatch):
    records_path = tmp_path / "records.jsonl"
    record = {
        "id": "A#0",
        "document": "replacement",
        "metadata": {"parent_item_key": "A"},
    }
    openai_batch.write_jsonl(records_path, [record])
    records_path.with_name("records-output.jsonl").write_text(
        json.dumps({
            "custom_id": "A#0",
            "response": {
                "status_code": 200,
                "body": {"data": [{"embedding": [0.1]}]},
            },
        })
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "run_id": "force-failure",
        "manifest_path": str(tmp_path / "manifest.json"),
        "force_full_rebuild": True,
        "batches": [{
            "batch_id": "batch-1",
            "status": "completed",
            "output_file_id": "file-1",
            "records_path": str(records_path),
            "imported_at": None,
        }],
    }

    class Client(FakeChromaClient):
        def __init__(self):
            super().__init__()
            self.live = {"OLD"}
            self.staging = None
            self.commits = 0
            self.aborts = 0

        def begin_staged_rebuild(self):
            self.staging = set()

        def upsert_embeddings(self, documents, metadatas, ids, embeddings):
            raise RuntimeError("disk write failed")

        def abort_staged_rebuild(self):
            self.staging = None
            self.aborts += 1

        def commit_staged_rebuild(self):
            self.commits += 1

    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(semantic_search.openai_batch, "find_manifest", lambda **kwargs: manifest)
    monkeypatch.setattr(semantic_search.openai_batch, "refresh_manifest_status", lambda value, **kwargs: value)
    monkeypatch.setattr(semantic_search.openai_batch, "create_openai_client", lambda config: object())
    client = Client()
    search = semantic_search.ZoteroSemanticSearch(chroma_client=client)

    with pytest.raises(RuntimeError, match="disk write failed"):
        search.import_openai_batch()

    assert client.live == {"OLD"}
    assert client.staging is None
    assert client.commits == 0
    assert client.aborts == 1
    assert manifest["batches"][0]["imported_at"] is None


def test_complete_force_batch_activates_staged_replacement(tmp_path, monkeypatch):
    records_path = tmp_path / "records.jsonl"
    record = {
        "id": "A#0",
        "document": "replacement",
        "metadata": {"parent_item_key": "A"},
    }
    openai_batch.write_jsonl(records_path, [record])
    records_path.with_name("records-output.jsonl").write_text(
        json.dumps({
            "custom_id": "A#0",
            "response": {
                "status_code": 200,
                "body": {"data": [{"embedding": [0.1]}]},
            },
        })
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "run_id": "force-complete",
        "manifest_path": str(tmp_path / "manifest.json"),
        "force_full_rebuild": True,
        "batches": [{
            "batch_id": "batch-1",
            "status": "completed",
            "output_file_id": "file-1",
            "records_path": str(records_path),
            "imported_at": None,
        }],
    }

    class Client(FakeChromaClient):
        def __init__(self):
            super().__init__()
            self.live = {"OLD"}
            self.staging = None

        def begin_staged_rebuild(self):
            self.staging = set()

        def upsert_embeddings(self, documents, metadatas, ids, embeddings):
            self.staging.update(ids)

        def commit_staged_rebuild(self):
            self.live = self.staging
            self.staging = None

        def abort_staged_rebuild(self):
            self.staging = None

    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(semantic_search.openai_batch, "find_manifest", lambda **kwargs: manifest)
    monkeypatch.setattr(semantic_search.openai_batch, "refresh_manifest_status", lambda value, **kwargs: value)
    monkeypatch.setattr(semantic_search.openai_batch, "create_openai_client", lambda config: object())
    client = Client()
    search = semantic_search.ZoteroSemanticSearch(chroma_client=client)

    stats = search.import_openai_batch()

    assert stats["batches_imported"] == 1
    assert client.live == {"A#0"}
    assert client.staging is None


def test_batch_source_guard_rejects_changed_target_item(monkeypatch):
    class Zotero:
        def last_modified_version(self):
            return 6

        def item_versions(self, **kwargs):
            if kwargs.get("since") is not None:
                return {"A": 6}
            return {"A": 6}

        def item(self, key):
            return {
                "key": key,
                "data": {"itemType": "journalArticle", "title": "Changed"},
            }

        def deleted(self, **kwargs):
            return {"items": []}

    zotero = Zotero()
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: zotero)
    search = semantic_search.ZoteroSemanticSearch(chroma_client=FakeChromaClient())
    manifest = {
        "source_guard_version": 1,
        "target_sync_version": 5,
        "force_full_rebuild": False,
        "expected_ids_by_item": {"A": ["A#0"]},
    }

    with pytest.raises(RuntimeError, match="changed after OpenAI batch"):
        search._validate_openai_batch_source(manifest)


def test_batch_source_guard_allows_unrelated_item_change(monkeypatch):
    class Zotero:
        def last_modified_version(self):
            return 6

        def item_versions(self, **kwargs):
            if kwargs.get("since") is not None:
                return {"B": 6}
            return {"A": 5, "B": 6}

        def item(self, key):
            return {
                "key": key,
                "data": {"itemType": "journalArticle", "title": "Unrelated"},
            }

        def deleted(self, **kwargs):
            return {"items": []}

    zotero = Zotero()
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: zotero)
    search = semantic_search.ZoteroSemanticSearch(chroma_client=FakeChromaClient())
    manifest = {
        "source_guard_version": 1,
        "target_sync_version": 5,
        "force_full_rebuild": False,
        "expected_ids_by_item": {"A": ["A#0"]},
    }

    search._validate_openai_batch_source(manifest)


def test_batch_source_guard_rejects_different_local_database(monkeypatch):
    zotero = object()
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: zotero)
    monkeypatch.setattr(
        semantic_search,
        "get_zotero_server_id",
        lambda _client: "current-database",
    )
    search = semantic_search.ZoteroSemanticSearch(
        chroma_client=FakeChromaClient()
    )

    with pytest.raises(RuntimeError, match="database changed"):
        search._validate_openai_batch_source(
            {
                "source_server_id": "submitted-database",
                "source_guard_version": 2,
                "target_sync_version": 5,
            }
        )


def test_batch_source_guard_rejects_legacy_manifest_in_local_mode(monkeypatch):
    zotero = object()
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: zotero)
    monkeypatch.setattr(
        semantic_search,
        "get_zotero_server_id",
        lambda _client: "current-database",
    )
    search = semantic_search.ZoteroSemanticSearch(
        chroma_client=FakeChromaClient()
    )

    with pytest.raises(RuntimeError, match="predates Local API"):
        search._validate_openai_batch_source(
            {
                "source_guard_version": 1,
                "target_sync_version": 5,
            }
        )
