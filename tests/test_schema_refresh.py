"""Manual local/web Zotero metadata-schema refresh tests."""

import json
import sys

import pytest

from zotero_mcp import cli, schema


def _schema_document(*, version=44, extra_type=True):
    item_types = []
    for item_type, aliases in schema._ALIASES.items():
        item_types.append(
            {
                "itemType": item_type,
                "fields": [
                    {"field": actual, "baseField": base}
                    for base, actual in aliases.items()
                ],
                "creatorTypes": [],
            }
        )
    if extra_type:
        item_types.append(
            {
                "itemType": "futureType",
                "fields": [
                    {"field": "futureTitle", "baseField": "title"}
                ],
                "creatorTypes": [],
            }
        )
    return {"version": version, "itemTypes": item_types}


class _Response:
    def __init__(self, document, *, status=200, headers=None):
        self.status_code = status
        self._document = document
        self.headers = headers or {}

    def json(self):
        return self._document

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture(autouse=True)
def isolated_schema_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("ZOTERO_MCP_SCHEMA_CACHE_DIR", str(tmp_path))
    schema._table_cache.clear()
    yield
    schema._table_cache.clear()


def test_auto_refresh_prefers_installed_local_zotero(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    seen = {}

    def request(url, headers):
        seen.update(url=url, headers=headers)
        return _Response(
            _schema_document(),
            headers={
                "X-Zotero-Version": "10.0.1",
                "Zotero-Schema-Version": "44",
            },
        )

    monkeypatch.setattr(schema, "_http_get", request)
    result = schema.refresh()

    assert result["status"] == "refreshed"
    assert result["source"] == "local"
    assert result["zotero_version"] == "10.0.1"
    assert seen["url"] == "http://127.0.0.1:23119/api/schema"
    assert schema.cache_path("local").name == "schema-local.json"
    assert schema.resolve_field("futureType", "title") == "futureTitle"


def test_web_refresh_uses_separate_cache_and_etag(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setattr(
        schema,
        "_http_get",
        lambda _url, _headers: _Response(_schema_document(version=44)),
    )
    schema.refresh(source="local")
    seen = {}

    def request(url, headers):
        seen.update(url=url, headers=headers)
        return _Response(
            _schema_document(version=45), headers={"ETag": '"schema-45"'}
        )

    monkeypatch.setattr(schema, "_http_get", request)
    result = schema.refresh(source="web")

    assert result["source"] == "web"
    assert result["schema_version"] == 45
    assert seen["url"] == schema.WEB_SCHEMA_URL
    assert schema.cache_path("web") != schema.cache_path("local")
    assert schema.cache_path("local").exists()
    assert schema.cache_path("web").exists()


def test_invalid_refresh_never_replaces_last_good_cache(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setattr(
        schema,
        "_http_get",
        lambda _url, _headers: _Response(_schema_document()),
    )
    schema.refresh(source="local")
    cache = schema.cache_path("local")
    original = cache.read_bytes()
    monkeypatch.setattr(
        schema,
        "_http_get",
        lambda _url, _headers: _Response({"version": 45, "itemTypes": []}),
    )

    with pytest.raises(ValueError, match="itemTypes"):
        schema.refresh(source="local")

    assert cache.read_bytes() == original


def test_automatic_refresh_skips_a_fresh_cache(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setattr(schema.time, "time", lambda: 1000.0)
    monkeypatch.setattr(
        schema,
        "_http_get",
        lambda _url, _headers: _Response(_schema_document()),
    )
    schema.refresh(source="local")

    def unexpected_request(_url, _headers):
        raise AssertionError("fresh schemas must not be requested again")

    monkeypatch.setattr(schema, "_http_get", unexpected_request)
    result = schema.refresh_if_due(
        source="local",
        interval_seconds=60,
    )

    assert result["status"] == "not_due"
    assert result["schema_version"] == 44


def test_automatic_refresh_updates_a_stale_cache(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    now = {"value": 1000.0}
    monkeypatch.setattr(schema.time, "time", lambda: now["value"])
    monkeypatch.setattr(
        schema,
        "_http_get",
        lambda _url, _headers: _Response(_schema_document()),
    )
    schema.refresh(source="local")
    now["value"] = 1061.0
    monkeypatch.setattr(
        schema,
        "_http_get",
        lambda _url, _headers: _Response(_schema_document(version=45)),
    )

    result = schema.refresh_if_due(
        source="local",
        interval_seconds=60,
    )

    assert result["status"] == "refreshed"
    assert result["schema_version"] == 45


def test_automatic_refresh_failure_backs_off_without_replacing_cache(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    now = {"value": 1000.0}
    monkeypatch.setattr(schema.time, "time", lambda: now["value"])
    monkeypatch.setattr(
        schema,
        "_http_get",
        lambda _url, _headers: _Response(_schema_document()),
    )
    schema.refresh(source="local")
    cache = schema.cache_path("local")
    original = cache.read_bytes()
    now["value"] = 1061.0
    calls = {"count": 0}

    def failed_request(_url, _headers):
        calls["count"] += 1
        raise RuntimeError("Zotero is offline")

    monkeypatch.setattr(schema, "_http_get", failed_request)
    failed = schema.refresh_if_due(
        source="local",
        interval_seconds=60,
        failure_backoff_seconds=300,
    )
    now["value"] = 1100.0
    backed_off = schema.refresh_if_due(
        source="local",
        interval_seconds=60,
        failure_backoff_seconds=300,
    )

    assert failed["status"] == "error"
    assert failed["schema_version"] == 44
    assert backed_off["status"] == "backoff"
    assert calls["count"] == 1
    assert cache.read_bytes() == original
    assert schema.failed_attempt_path("local").exists()


def test_manual_refresh_bypasses_and_clears_automatic_failure_backoff(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setattr(schema.time, "time", lambda: 1000.0)

    def failed_request(_url, _headers):
        raise RuntimeError("Zotero is offline")

    monkeypatch.setattr(schema, "_http_get", failed_request)
    failed = schema.refresh_if_due(source="local")
    assert failed["status"] == "error"
    assert schema.failed_attempt_path("local").exists()

    monkeypatch.setattr(
        schema,
        "_http_get",
        lambda _url, _headers: _Response(_schema_document()),
    )
    refreshed = schema.refresh(source="local")

    assert refreshed["status"] == "refreshed"
    assert not schema.failed_attempt_path("local").exists()


def test_corrupt_source_cache_falls_back_to_static_floor(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    schema.cache_path("local").write_text('{"source":"local"}')

    assert schema.resolve_field("case", "title") == "caseName"


def test_schema_refresh_cli_reports_machine_readable_result(monkeypatch, capsys):
    monkeypatch.setattr(cli, "setup_zotero_environment", lambda: None)
    monkeypatch.setattr(
        schema,
        "refresh",
        lambda source="auto": {
            "status": "refreshed",
            "source": "local",
            "schema_version": 44,
            "zotero_version": "10.0.1",
            "cache_path": "/tmp/schema-local.json",
        },
    )
    monkeypatch.setattr(
        sys, "argv", ["zotero-mcp", "schema-refresh", "--json"]
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["source"] == "local"
    assert output["schema_version"] == 44
