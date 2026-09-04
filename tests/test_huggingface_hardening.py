"""Hugging Face model code is disabled by default and revision-pinned."""

import sys
from types import SimpleNamespace

import pytest

from zotero_mcp.chroma_client import HuggingFaceEmbeddingFunction


class _Model:
    max_seq_length = 512

    def encode(self, inputs, convert_to_numpy=True):
        return SimpleNamespace(tolist=lambda: [[0.0] for _ in inputs])


def _fake_sentence_transformers(monkeypatch, calls):
    def construct(model_name, **kwargs):
        calls.append((model_name, kwargs))
        return _Model()

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=construct),
    )


def test_huggingface_remote_code_is_disabled_by_default(monkeypatch):
    calls = []
    _fake_sentence_transformers(monkeypatch, calls)

    function = HuggingFaceEmbeddingFunction("sentence-transformers/test-model")

    assert calls == [
        ("sentence-transformers/test-model", {"trust_remote_code": False})
    ]
    assert function.get_config()["trust_remote_code"] is False


def test_huggingface_remote_code_requires_exact_commit_revision(monkeypatch):
    calls = []
    _fake_sentence_transformers(monkeypatch, calls)

    with pytest.raises(ValueError, match="40-character commit hash"):
        HuggingFaceEmbeddingFunction(
            "organization/custom-model", trust_remote_code=True
        )
    with pytest.raises(ValueError, match="40-character commit hash"):
        HuggingFaceEmbeddingFunction(
            "organization/custom-model",
            trust_remote_code=True,
            revision="main",
        )
    assert calls == []


def test_huggingface_pinned_remote_code_round_trips_config(monkeypatch):
    calls = []
    _fake_sentence_transformers(monkeypatch, calls)
    revision = "a" * 40

    function = HuggingFaceEmbeddingFunction.build_from_config(
        {
            "model_name": "organization/custom-model",
            "query_instruction": "Retrieve matching passages",
            "trust_remote_code": True,
            "revision": revision,
        }
    )

    assert calls == [
        (
            "organization/custom-model",
            {"trust_remote_code": True, "revision": revision},
        )
    ]
    assert function.get_config() == {
        "model_name": "organization/custom-model",
        "query_instruction": "Retrieve matching passages",
        "trust_remote_code": True,
        "revision": revision,
    }
