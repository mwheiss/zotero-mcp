"""Automatic semantic updates preserve full text and search capacity."""

import json
from types import SimpleNamespace

from zotero_mcp import _app
from zotero_mcp.config_light import auto_update_embedding_concurrency
from zotero_mcp.tools import search as search_tools


def test_auto_update_embedding_concurrency_is_bounded():
    assert auto_update_embedding_concurrency({}) == 1
    assert auto_update_embedding_concurrency({"embedding_concurrency": "bad"}) == 1
    assert auto_update_embedding_concurrency({"embedding_concurrency": 0}) == 1
    assert auto_update_embedding_concurrency({"embedding_concurrency": 100}) == 32


def test_startup_auto_update_passes_saved_fulltext_and_concurrency(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "semantic_search": {
                    "fulltext": True,
                    "update_config": {
                        "auto_update": True,
                        "update_frequency": "startup",
                        "embedding_concurrency": 1,
                    },
                }
            }
        )
    )
    calls = []
    search = SimpleNamespace(
        should_update_database=lambda: True,
        update_database=lambda **kwargs: calls.append(kwargs)
        or {"processed_items": 0},
    )
    monkeypatch.setattr(
        "zotero_mcp.semantic_search.create_semantic_search",
        lambda _path: search,
    )

    _app._sync_semantic_update(config_path)

    assert calls == [{"fulltext": True, "embedding_concurrency": 1}]


def test_automatic_schema_refresh_is_opt_in(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"schema_refresh": {}}))

    def unexpected_refresh(**_kwargs):
        raise AssertionError("disabled schema refresh must not run")

    monkeypatch.setattr("zotero_mcp.schema.refresh_if_due", unexpected_refresh)

    _app._sync_schema_refresh(config_path)


def test_automatic_schema_refresh_passes_bounded_schedule(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_refresh": {
                    "auto_refresh": True,
                    "source": "local",
                    "interval_days": 7,
                    "failure_backoff_hours": 24,
                }
            }
        )
    )
    calls = []
    monkeypatch.setattr(
        "zotero_mcp.schema.refresh_if_due",
        lambda **kwargs: calls.append(kwargs)
        or {
            "status": "not_due",
            "source": "local",
            "schema_version": 44,
        },
    )

    _app._sync_schema_refresh(config_path)

    assert calls == [
        {
            "source": "local",
            "interval_seconds": 7 * 24 * 60 * 60,
            "failure_backoff_seconds": 24 * 60 * 60,
        }
    ]


def test_schema_refresh_schedule_rejects_invalid_numbers():
    assert _app._bounded_number(
        "NaN", default=7, minimum=1 / 24, maximum=365
    ) == 7
    assert _app._bounded_number(
        True, default=24, minimum=1, maximum=168
    ) == 24
    assert _app._bounded_number(
        1000, default=24, minimum=1, maximum=168
    ) == 168


def test_startup_policy_is_not_relaunched_before_each_search():
    search = SimpleNamespace(
        update_config={
            "auto_update": True,
            "update_frequency": "startup",
            "embedding_concurrency": 1,
        },
        should_update_database=lambda: (_ for _ in ()).throw(
            AssertionError("startup policy must stop before the due check")
        ),
    )

    search_tools._maybe_fire_presearch_sync(search)


def test_due_periodic_presearch_update_uses_saved_options(monkeypatch):
    calls = []
    search = SimpleNamespace(
        update_config={
            "auto_update": True,
            "update_frequency": "daily",
            "embedding_concurrency": 1,
        },
        _active_fulltext=True,
        should_update_database=lambda: True,
        update_database=lambda **kwargs: calls.append(kwargs),
    )

    class ImmediateThread:
        def __init__(self, target, **_kwargs):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(search_tools._threading, "Thread", ImmediateThread)
    monkeypatch.setattr(search_tools, "_last_presearch_sync_ts", 0.0)

    search_tools._maybe_fire_presearch_sync(search)

    assert calls == [{"fulltext": True, "embedding_concurrency": 1}]
