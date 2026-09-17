import hashlib
import json
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from zotero_mcp import reembed_chroma
from zotero_mcp.chroma_client import OpenAIEmbeddingFunction, Settings, chromadb


def _metadata(document: str, index: int) -> dict:
    return {
        "item_key": f"ITEM{index}",
        "parent_item_key": f"ITEM{index}",
        "embedding_content_sha256": hashlib.sha256(document.encode()).hexdigest(),
        "embedding_metadata_sha256": hashlib.sha256(f"metadata-{index}".encode()).hexdigest(),
    }


def _make_target(tmp_path: Path, record_count: int = 4):
    root = tmp_path / "zotero-mcp-nemotron-test"
    chroma_path = root / "chroma_db_nemotron"
    chroma_path.mkdir(parents=True)
    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )
    collection = client.create_collection(
        "zotero_library",
        metadata={"hnsw:space": "cosine"},
    )
    documents = [f"stored document {index}" for index in range(record_count)]
    ids = [f"ITEM{index}#0" for index in range(record_count)]
    metadatas = [_metadata(document, index) for index, document in enumerate(documents)]
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=[[1.0, 0.0, 0.0] for _ in ids],
    )
    client.create_collection("zotero_library__empty")
    config_path = root / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "semantic_search": {
                    "persist_directory": str(chroma_path),
                    "embedding_model": "openai",
                    "embedding_config": {
                        "api_key": "test-key",
                        "base_url": "http://127.0.0.1:8032/v1",
                        "model_name": "nemotron-test",
                        "model_identity": "Nemotron-test",
                        "query_prefix": "query: ",
                        "document_prefix": "passage: ",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return root, chroma_path, config_path, collection, ids, documents, metadatas


class _RecordingEndpoint:
    def __init__(self):
        self.embeddings = self
        self.inputs = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def create(self, *, model, input, encoding_format):
        assert model == "nemotron-test"
        assert encoding_format == "float"
        assert len(input) == 1
        assert input[0].startswith("passage: stored document ")
        with self.lock:
            self.inputs.append(input[0])
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        with self.lock:
            self.active -= 1
        suffix = int(hashlib.sha256(input[0].encode()).hexdigest()[:4], 16) / 65535
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.0, 1.0, suffix + 1.0])]
        )


def _embedding_function(endpoint: _RecordingEndpoint) -> OpenAIEmbeddingFunction:
    function = OpenAIEmbeddingFunction.__new__(OpenAIEmbeddingFunction)
    function.model_name = "nemotron-test"
    function.query_instruction = None
    function.query_prefix = "query: "
    function.document_prefix = "passage: "
    function.request_batch_size = 1
    function.rate_limit_rps = None
    function.client = endpoint
    return function


def test_embedding_only_migration_preserves_records_and_resumes(tmp_path):
    (
        root,
        chroma_path,
        config_path,
        collection,
        ids,
        documents,
        metadatas,
    ) = _make_target(tmp_path)
    reference = tmp_path / "reference-nemotron-chroma"
    shutil.copytree(chroma_path, reference)
    before = collection.get(
        ids=ids,
        include=["documents", "metadatas", "embeddings"],
    )
    spec = reembed_chroma.resolve_target(config_path, chroma_path)
    inventory = reembed_chroma.read_inventory(chroma_path)
    endpoint = _RecordingEndpoint()
    function = _embedding_function(endpoint)

    first = reembed_chroma.execute_migration(
        spec,
        inventory,
        embedding_function=function,
    )

    assert first == {"selected": 4, "updated": 4, "skipped": 0}
    assert endpoint.max_active == 2
    assert len(endpoint.inputs) == 4
    after = collection.get(
        ids=ids,
        include=["documents", "metadatas", "embeddings"],
    )
    assert after["ids"] == before["ids"]
    assert after["documents"] == before["documents"] == documents
    assert after["metadatas"] == before["metadatas"] == metadatas
    assert [list(vector) for vector in after["embeddings"]] != [
        list(vector) for vector in before["embeddings"]
    ]
    assert all(len(vector) == 3 for vector in after["embeddings"])
    assert {item.name: item.count for item in inventory} == {
        "zotero_library": 4,
        "zotero_library__empty": 0,
    }
    assert reembed_chroma.validate_against_reference(chroma_path, reference)["ok"]

    second = reembed_chroma.execute_migration(
        spec,
        inventory,
        embedding_function=function,
    )

    assert second == {"selected": 4, "updated": 0, "skipped": 4}
    assert len(endpoint.inputs) == 4
    with reembed_chroma.sqlite3.connect(root / "reembed-progress.sqlite3") as progress:
        assert progress.execute("SELECT count(*) FROM completed").fetchone()[0] == 4


def test_changed_document_hash_invalidates_progress(tmp_path):
    _, chroma_path, config_path, collection, ids, _, _ = _make_target(
        tmp_path, record_count=1
    )
    spec = reembed_chroma.resolve_target(config_path, chroma_path)
    inventory = reembed_chroma.read_inventory(chroma_path)
    endpoint = _RecordingEndpoint()
    function = _embedding_function(endpoint)
    reembed_chroma.execute_migration(spec, inventory, embedding_function=function)
    collection.update(
        ids=ids,
        documents=["stored document changed"],
        embeddings=[[0.0, 1.0, 0.0]],
    )

    result = reembed_chroma.execute_migration(
        spec,
        inventory,
        embedding_function=function,
    )

    assert result == {"selected": 1, "updated": 1, "skipped": 0}
    assert endpoint.inputs == [
        "passage: stored document 0",
        "passage: stored document changed",
    ]


def test_resolve_target_rejects_unsafe_paths_and_prefixes(tmp_path):
    _, chroma_path, config_path, *_ = _make_target(tmp_path, record_count=1)
    config = json.loads(config_path.read_text())
    config["semantic_search"]["embedding_config"].pop("document_prefix")
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="document_prefix"):
        reembed_chroma.resolve_target(config_path, chroma_path)

    with pytest.raises(ValueError, match="must contain 'nemotron'"):
        reembed_chroma.resolve_target(config_path, tmp_path)


def test_dry_run_does_not_create_progress_database(tmp_path, capsys):
    root, chroma_path, config_path, *_ = _make_target(tmp_path, record_count=1)

    result = reembed_chroma.main(
        [
            "--config-path",
            str(config_path),
            "--chroma-path",
            str(chroma_path),
        ]
    )

    assert result == 0
    assert not (root / "reembed-progress.sqlite3").exists()
    assert "DRY RUN ONLY" in capsys.readouterr().out


def test_embedding_commit_never_supplies_documents_or_metadata():
    class Collection:
        def __init__(self):
            self.calls = []

        def update(self, **kwargs):
            self.calls.append(kwargs)

    collection = Collection()

    reembed_chroma._update_embedding_only(collection, "ITEM#0", [0.1, 0.2])

    assert collection.calls == [
        {"ids": ["ITEM#0"], "embeddings": [[0.1, 0.2]]}
    ]
