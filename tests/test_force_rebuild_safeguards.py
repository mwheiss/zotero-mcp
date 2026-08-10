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


@pytest.mark.parametrize("saved_fulltext", [False, True])
def test_cli_uses_saved_fulltext_mode_when_resuming(saved_fulltext):
    search = SimpleNamespace(
        get_database_status=lambda: {
            "rebuild_in_progress": {"fulltext": saved_fulltext}
        }
    )

    effective, resuming = cli._resolve_update_fulltext_mode(
        search,
        requested=not saved_fulltext,
        force_rebuild=False,
    )

    assert effective is saved_fulltext
    assert resuming is True


def test_cli_new_force_rebuild_uses_explicit_fulltext_mode():
    search = SimpleNamespace(
        get_database_status=lambda: {
            "rebuild_in_progress": {"fulltext": True}
        }
    )

    effective, resuming = cli._resolve_update_fulltext_mode(
        search,
        requested=False,
        force_rebuild=True,
    )

    assert effective is False
    assert resuming is False


def test_cli_resume_reports_and_passes_saved_fulltext_mode(
    monkeypatch,
    capsys,
):
    captured = {}

    class ResumingSearch:
        chroma_client = SimpleNamespace(embedding_model="openai")

        def get_database_status(self):
            return {"rebuild_in_progress": {"fulltext": True}}

        def update_database(self, **kwargs):
            captured.update(kwargs)
            return {"errors": 0}

    monkeypatch.setattr(sys, "argv", ["zotero-mcp", "update-db"])
    monkeypatch.setattr(cli, "setup_zotero_environment", lambda: None)
    monkeypatch.setattr(cli, "_print_update_stats", lambda _stats: None)
    monkeypatch.setattr(
        semantic_search,
        "create_semantic_search",
        lambda *_args, **_kwargs: ResumingSearch(),
    )
    monkeypatch.setenv("ZOTERO_LOCAL", "true")

    cli.main()

    output = capsys.readouterr().out
    assert "Resuming the saved semantic rebuild contract" in output
    assert "Extracting full-text content" in output
    assert "full text disabled" not in output
    assert captured["fulltext"] is True


def test_cli_resume_allows_retry_failed_fulltext_from_saved_mode(
    monkeypatch,
):
    captured = {}

    class ResumingSearch:
        chroma_client = SimpleNamespace(embedding_model="openai")

        def get_database_status(self):
            return {"rebuild_in_progress": {"fulltext": True}}

        def update_database(self, **kwargs):
            captured.update(kwargs)
            return {"errors": 0}

    monkeypatch.setattr(
        sys,
        "argv",
        ["zotero-mcp", "update-db", "--retry-failed-fulltext"],
    )
    monkeypatch.setattr(cli, "setup_zotero_environment", lambda: None)
    monkeypatch.setattr(cli, "_print_update_stats", lambda _stats: None)
    monkeypatch.setattr(
        semantic_search,
        "create_semantic_search",
        lambda *_args, **_kwargs: ResumingSearch(),
    )
    monkeypatch.setenv("ZOTERO_LOCAL", "true")

    cli.main()

    assert captured["fulltext"] is True
    assert captured["retry_failed_fulltext"] is True


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


def test_zotero_mcp_cli_reports_graceful_update_interrupt(
    monkeypatch,
    capsys,
):
    class InterruptedSearch:
        chroma_client = SimpleNamespace(embedding_model="openai")

        def update_database(self, **_kwargs):
            raise KeyboardInterrupt

    monkeypatch.setattr(sys, "argv", ["zotero-mcp", "update-db"])
    monkeypatch.setattr(cli, "setup_zotero_environment", lambda: None)
    monkeypatch.setattr(
        semantic_search,
        "create_semantic_search",
        lambda *_args, **_kwargs: InterruptedSearch(),
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 130
    assert "Database update stopped cleanly" in capsys.readouterr().err


def test_cli_force_clear_requires_force_rebuild(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["zotero-mcp", "update-db", "--force-clear"])

    def unexpected_setup():
        raise AssertionError("backend setup must not run for invalid flags")

    monkeypatch.setattr(cli, "setup_zotero_environment", unexpected_setup)

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 2
    assert "requires --force-rebuild" in capsys.readouterr().err


def test_cli_rejects_force_rebuild_with_limit_before_confirmation(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["zotero-mcp", "update-db", "--force-rebuild", "--limit", "1"],
    )
    monkeypatch.setattr(
        cli,
        "_confirm_force_rebuild",
        lambda: pytest.fail("invalid flags must not prompt"),
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 2
    assert "cannot be combined" in capsys.readouterr().err


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
        force_clear=True,
        confirm_force_rebuild="REBUILD ALL ITEMS",
        ctx=_Context(),
    )

    assert captured["force_full_rebuild"] is True
    assert captured["force_clear"] is True
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
    assert captured["force_clear"] is False
    assert captured["fulltext"] is False
    assert factory_args["allow_embedding_mismatch"] is False
    assert "# Database Update Results" in result


def test_mcp_force_clear_requires_force_rebuild(monkeypatch):
    def unexpected_create(*_args, **_kwargs):
        raise AssertionError("search backend must not be created")

    monkeypatch.setattr(
        semantic_search, "create_semantic_search", unexpected_create
    )

    result = search_tools.update_search_database(
        force_clear=True,
        ctx=_Context(),
    )

    assert "force_clear requires force_rebuild=True" in result


def test_mcp_rejects_force_rebuild_with_limit_before_backend(monkeypatch):
    monkeypatch.setattr(
        semantic_search,
        "create_semantic_search",
        lambda *_args, **_kwargs: pytest.fail("backend must not be created"),
    )

    result = search_tools.update_search_database(
        force_rebuild=True,
        limit=1,
        confirm_force_rebuild="REBUILD ALL ITEMS",
        ctx=_Context(),
    )

    assert "limit cannot be combined" in result
