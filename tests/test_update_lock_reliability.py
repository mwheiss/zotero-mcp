"""Tests for update-lock diagnostics and the cross-encoder score API."""

import os
import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb relies on pydantic v1 paths incompatible with Python 3.14+",
        allow_module_level=True,
    )

from conftest import skip_on_ci

from zotero_mcp import semantic_search


def test_update_lock_path_honors_environment(monkeypatch, tmp_path):
    configured = tmp_path / "custom-update.lock"
    monkeypatch.setenv("ZOTERO_MCP_UPDATE_LOCK_PATH", str(configured))

    assert semantic_search.update_lock_path() == configured


@skip_on_ci
def test_read_lock_holder_missing_file(tmp_path):
    pid, alive = semantic_search.read_lock_holder(tmp_path / "nope.lock")
    assert pid is None
    assert alive is False


@skip_on_ci
def test_read_lock_holder_live_pid(tmp_path):
    lock = tmp_path / "update.lock"
    lock.write_text(str(os.getpid()))
    pid, alive = semantic_search.read_lock_holder(lock)
    assert pid == os.getpid()
    assert alive is True


@skip_on_ci
def test_read_lock_holder_dead_pid(tmp_path):
    lock = tmp_path / "update.lock"
    # A pid that is essentially certain not to exist.
    lock.write_text("999999")
    pid, alive = semantic_search.read_lock_holder(lock)
    assert pid == 999999
    assert alive is False


@skip_on_ci
def test_force_update_env_does_not_bypass_live_lock(tmp_path, monkeypatch):
    lock = tmp_path / "update.lock"
    with semantic_search._acquire_update_lock(lock) as first_acquired:
        assert first_acquired is True
        monkeypatch.setenv("ZOTERO_MCP_FORCE_UPDATE", "1")
        with semantic_search._acquire_update_lock(lock) as second_acquired:
            assert second_acquired is False


@skip_on_ci
def test_acquire_lock_writes_pid(tmp_path, monkeypatch):
    lock = tmp_path / "update.lock"
    monkeypatch.delenv("ZOTERO_MCP_FORCE_UPDATE", raising=False)
    with semantic_search._acquire_update_lock(lock) as acquired:
        assert acquired is True
        # While held, the file records our pid (POSIX/flock platforms).
        if lock.exists() and lock.read_text().strip():
            assert lock.read_text().strip() == str(os.getpid())


@skip_on_ci
def test_lock_contender_preserves_holder_pid(tmp_path, monkeypatch):
    lock = tmp_path / "update.lock"
    monkeypatch.delenv("ZOTERO_MCP_FORCE_UPDATE", raising=False)

    with semantic_search._acquire_update_lock(lock) as first_acquired:
        assert first_acquired is True
        holder_pid = lock.read_text().strip()
        with semantic_search._acquire_update_lock(lock) as second_acquired:
            assert second_acquired is False
        assert lock.read_text().strip() == holder_pid == str(os.getpid())


def test_rerank_with_scores_orders_and_scores(monkeypatch):
    rr = semantic_search.CrossEncoderReranker.__new__(semantic_search.CrossEncoderReranker)

    class _Model:
        def predict(self, pairs):
            # Higher score for the doc containing "match".
            return [9.0 if "match" in d else 1.0 for _, d in pairs]

    rr.model = _Model()
    ranked = rr.rerank_with_scores("q", ["no", "the match here", "no"], top_k=2)
    assert ranked[0][0] == 1  # index of the matching doc first
    assert ranked[0][1] == 9.0
    assert len(ranked) == 2
    # rerank() delegates to rerank_with_scores and returns indices only.
    assert rr.rerank("q", ["no", "the match here", "no"], top_k=1) == [1]
