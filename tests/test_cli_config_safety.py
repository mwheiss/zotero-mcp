"""Crash-safe CLI configuration update tests."""

import json

from zotero_mcp.cli import _save_zotero_db_path_to_config


def test_db_path_update_preserves_unrelated_configuration(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"client_env": {"ZOTERO_API_KEY": "secret"}, "other": 7})
    )

    _save_zotero_db_path_to_config(config_path, "/data/zotero.sqlite")

    saved = json.loads(config_path.read_text())
    assert saved["client_env"] == {"ZOTERO_API_KEY": "secret"}
    assert saved["other"] == 7
    assert saved["semantic_search"]["zotero_db_path"] == "/data/zotero.sqlite"


def test_db_path_update_refuses_to_replace_malformed_configuration(
    tmp_path, capsys
):
    config_path = tmp_path / "config.json"
    original = b'{"client_env": '
    config_path.write_bytes(original)

    _save_zotero_db_path_to_config(config_path, "/data/zotero.sqlite")

    assert config_path.read_bytes() == original
    assert "Refusing to replace" in capsys.readouterr().out
