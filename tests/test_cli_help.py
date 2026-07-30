from unittest.mock import patch

import pytest

from zotero_mcp.cli import main


def test_zotero_mcp_help_mentions_openai_batch(capsys):
    with patch("sys.argv", ["zotero-mcp", "help"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "OpenAI Batch API indexing" in out
    assert "zotero-mcp update-db --openai-batch" in out
    assert "openai-batch-status" in out
    assert "openai-batch-import" in out


def test_update_db_help_mentions_embedding_concurrency(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["zotero-mcp", "help", "update-db"],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 0
    assert "--embedding-concurrency N" in capsys.readouterr().out


def test_update_db_fulltext_requires_source_value(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["zotero-mcp", "update-db", "--fulltext"],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 2
    assert "expected one argument" in capsys.readouterr().err


def test_update_db_help_documents_fulltext_sources(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["zotero-mcp", "help", "update-db"],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "--fulltext {api,local,none}" in output
    assert "default: api" in output


def test_zotero_mcp_help_update_db_shows_batch_flags(capsys):
    with patch("sys.argv", ["zotero-mcp", "help", "update-db"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--openai-batch" in out
    assert "--no-openai-batch" in out
