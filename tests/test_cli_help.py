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


def test_update_db_help_documents_fulltext_switch(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["zotero-mcp", "help", "update-db"],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "--fulltext" in output
    assert "{api,local,none}" not in output
    assert "title and abstract only" in output
    assert "--retry-failed-fulltext" in output
    assert "does not rebuild unaffected items" in " ".join(output.split())


def test_zotero_mcp_help_update_db_shows_batch_flags(capsys):
    with patch("sys.argv", ["zotero-mcp", "help", "update-db"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--openai-batch" in out
    assert "--no-openai-batch" in out
