"""Cheap configuration reads used before importing semantic dependencies."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_UPDATE_CONFIG: dict[str, Any] = {
    "auto_update": False,
    "update_frequency": "manual",
    "last_update": None,
    "update_days": 7,
}

_DEFAULT_RERANKER_CONFIG: dict[str, Any] = {
    "enabled": False,
    "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "candidate_multiplier": 3,
}


def _read_config(config_path: str | None) -> dict[str, Any]:
    if not config_path or not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except Exception as exc:
        logger.warning("Could not read Zotero MCP config: %s", exc)
        return {}


def load_update_config(config_path: str | None) -> dict[str, Any]:
    config = dict(_DEFAULT_UPDATE_CONFIG)
    config.update(
        (_read_config(config_path).get("semantic_search") or {}).get(
            "update_config", {}
        )
    )
    return config


def should_update(update_config: dict[str, Any]) -> bool:
    if not update_config.get("auto_update", False):
        return False
    frequency = update_config.get("update_frequency", "manual")
    if frequency == "manual":
        return False
    if frequency == "startup":
        return True
    if frequency == "daily":
        days = 1
    elif isinstance(frequency, str) and frequency.startswith("every_"):
        try:
            days = int(frequency.split("_", 1)[1])
        except (ValueError, IndexError):
            return False
    else:
        return False
    last_update = update_config.get("last_update")
    if not last_update:
        return True
    try:
        return datetime.now() - datetime.fromisoformat(last_update) >= timedelta(
            days=days
        )
    except (TypeError, ValueError):
        return False


def load_reranker_config(config_path: str | None) -> dict[str, Any]:
    config = dict(_DEFAULT_RERANKER_CONFIG)
    config.update(
        (_read_config(config_path).get("semantic_search") or {}).get(
            "reranker", {}
        )
    )
    return config


def reranker_enabled(config_path: str | None) -> bool:
    return bool(load_reranker_config(config_path).get("enabled", False))


def semantic_search_configured(config_path: str | None) -> bool:
    return bool(_read_config(config_path).get("semantic_search"))
