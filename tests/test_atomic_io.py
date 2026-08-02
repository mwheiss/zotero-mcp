"""Failure-injection tests for crash-resistant state-file writes."""

import json

import pytest

from zotero_mcp import _atomic_io


def test_atomic_write_failure_preserves_existing_file(monkeypatch, tmp_path):
    destination = tmp_path / "state.json"
    destination.write_text('{"state": "old"}')

    def fail_replace(*_args):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(_atomic_io.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        _atomic_io.atomic_write_json(destination, {"state": "new"})

    assert json.loads(destination.read_text()) == {"state": "old"}
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_atomic_write_replaces_complete_json(tmp_path):
    destination = tmp_path / "state.json"
    destination.write_text('{"state": "old"}')

    _atomic_io.atomic_write_json(destination, {"state": "new"}, indent=2)

    assert json.loads(destination.read_text()) == {"state": "new"}


def test_atomic_write_lines_failure_preserves_existing_file(monkeypatch, tmp_path):
    destination = tmp_path / "records.jsonl"
    destination.write_text("old\n")

    monkeypatch.setattr(
        _atomic_io.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(OSError, match="replace failed"):
        _atomic_io.atomic_write_lines(destination, iter(["new-1\n", "new-2\n"]))

    assert destination.read_text() == "old\n"
    assert list(tmp_path.glob(".records.jsonl.*.tmp")) == []


def test_durable_unlink_is_idempotent(tmp_path):
    destination = tmp_path / "marker.json"
    destination.write_text("marker")

    _atomic_io.durable_unlink(destination)
    _atomic_io.durable_unlink(destination)

    assert not destination.exists()
