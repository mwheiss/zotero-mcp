"""Regression tests for expensive force-rebuild confirmation safeguards."""

import builtins
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import zotero_mcp.cli as cli
import zotero_mcp.cli_standalone as cli_standalone
import zotero_mcp.semantic_search as semantic_search
from zotero_mcp.tools import search as search_tools


class _Context:
    def __init__(self):
        self.warnings = []

    def info(self, _message):
        pass

    def warning(self, message):
        self.warnings.append(message)

    def error(self, _message):
        pass


@pytest.mark.parametrize("response", ["", "n", "N", "no", "anything else"])
def test_cli_force_rebuild_confirmation_defaults_to_no(
    monkeypatch, response
):
    monkeypatch.setattr(builtins, "input", lambda _prompt: response)

    assert cli._confirm_force_rebuild() is False


@pytest.mark.parametrize("response", ["y", "Y", "yes", "YES", " yes "])
def test_cli_force_rebuild_confirmation_accepts_explicit_yes(
    monkeypatch, response
):
    monkeypatch.setattr(builtins, "input", lambda _prompt: response)

    assert cli._confirm_force_rebuild() is True


def test_cli_force_rebuild_confirmation_denies_eof(monkeypatch):
    def raise_eof(_prompt):
        raise EOFError

    monkeypatch.setattr(builtins, "input", raise_eof)

    assert cli._confirm_force_rebuild() is False


def test_zotero_mcp_cli_cancellation_happens_before_setup(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        sys, "argv", ["zotero-mcp", "update-db", "--force-rebuild"]
    )
    monkeypatch.setattr(cli, "_confirm_force_rebuild", lambda: False)

    def unexpected_setup():
        raise AssertionError("backend setup must not run after cancellation")

    monkeypatch.setattr(cli, "setup_zotero_environment", unexpected_setup)

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    assert "Force rebuild cancelled." in capsys.readouterr().out


def test_standalone_cli_cancellation_happens_before_setup(
    monkeypatch, capsys
):
    args = SimpleNamespace(subcommand="update", force_rebuild=True)
    monkeypatch.setattr(
        cli_standalone, "_confirm_force_rebuild", lambda: False
    )

    def unexpected_setup():
        raise AssertionError("backend setup must not run after cancellation")

    monkeypatch.setattr(
        cli_standalone, "setup_zotero_environment", unexpected_setup
    )

    cli_standalone.cmd_db(args)

    assert "Force rebuild cancelled." in capsys.readouterr().out


@pytest.mark.parametrize("confirmation", [None, "", "yes", "REBUILD"])
def test_mcp_force_rebuild_rejects_missing_or_wrong_confirmation(
    monkeypatch, confirmation
):
    def unexpected_create(*_args, **_kwargs):
        raise AssertionError("search backend must not be created")

    monkeypatch.setattr(
        semantic_search, "create_semantic_search", unexpected_create
    )
    ctx = _Context()

    result = search_tools.update_search_database(
        force_rebuild=True,
        confirm_force_rebuild=confirmation,
        ctx=ctx,
    )

    assert "Force Rebuild Not Started" in result
    assert "REBUILD ALL ITEMS" not in result
    assert "intentionally not disclosed" in result
    assert ctx.warnings == ["Blocked unconfirmed force rebuild."]


def test_mcp_tool_description_does_not_disclose_confirmation_phrase():
    source = Path(search_tools.__file__).read_text()
    start = source.index(
        '@mcp.tool(\n    name="zotero_update_search_database"'
    )
    end = source.index("def update_search_database(", start)

    assert "REBUILD ALL ITEMS" not in source[start:end]


def test_mcp_force_rebuild_accepts_exact_confirmation(monkeypatch):
    captured = {}
    factory_args = {}

    class FakeSearch:
        def update_database(self, **kwargs):
            captured.update(kwargs)
            return {
                "total_items": 3,
                "processed_items": 3,
                "added_items": 0,
                "updated_items": 3,
                "skipped_items": 0,
                "errors": 0,
                "duration": "1s",
            }

    monkeypatch.setattr(
        semantic_search,
        "create_semantic_search",
        lambda _config_path, **kwargs: factory_args.update(kwargs) or FakeSearch(),
    )
    monkeypatch.setattr(search_tools._utils, "is_local_mode", lambda: True)

    result = search_tools.update_search_database(
        force_rebuild=True,
        confirm_force_rebuild="REBUILD ALL ITEMS",
        ctx=_Context(),
    )

    assert captured["force_full_rebuild"] is True
    assert captured["fulltext"] is False
    assert factory_args["allow_embedding_mismatch"] is True
    assert "# Database Update Results" in result


def test_mcp_incremental_update_needs_no_confirmation(monkeypatch):
    captured = {}
    factory_args = {}

    class FakeSearch:
        def update_database(self, **kwargs):
            captured.update(kwargs)
            return {"total_items": 0, "errors": 0}

    monkeypatch.setattr(
        semantic_search,
        "create_semantic_search",
        lambda _config_path, **kwargs: factory_args.update(kwargs) or FakeSearch(),
    )
    monkeypatch.setattr(search_tools._utils, "is_local_mode", lambda: True)

    result = search_tools.update_search_database(ctx=_Context())

    assert captured["force_full_rebuild"] is False
    assert captured["fulltext"] is False
    assert factory_args["allow_embedding_mismatch"] is False
    assert "# Database Update Results" in result
