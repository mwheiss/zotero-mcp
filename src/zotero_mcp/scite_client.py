"""Thin HTTP client for the Scite public API (api.scite.ai).

Provides citation tallies and paper metadata for Zotero library items.
No API key required — uses Scite's public endpoints.

This is the MCP counterpart of the `Scite Zotero Plugin`_ which adds
tally columns to Zotero's desktop interface.  Instead of columns, Scite
data appears inline in MCP search results and tool outputs.

.. _Scite Zotero Plugin: https://github.com/scitedotai/scite-zotero-plugin
"""

from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)

_BASE = "https://api.scite.ai"
_TIMEOUT = 15  # seconds
_HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}


class SciteAPIError(RuntimeError):
    """The Scite service failed to answer a request."""


def _json(response, endpoint: str) -> dict:
    try:
        payload = response.json()
    except Exception as exc:
        raise SciteAPIError(f"Scite {endpoint} returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SciteAPIError(f"Scite {endpoint} returned a non-object response")
    return payload


# ---------------------------------------------------------------------------
# Tallies
# ---------------------------------------------------------------------------


def get_tally(doi: str) -> dict | None:
    """Fetch citation tally for a single DOI.

    Returns dict with keys: ``doi``, ``total``, ``supporting``,
    ``contradicting``, ``mentioning``, ``unclassified``,
    ``citingPublications``. Returns ``None`` only when Scite has no record.
    """
    try:
        resp = requests.get(
            f"{_BASE}/tallies/{doi}",
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        if resp.status_code == 200:
            return _json(resp, "tally endpoint")
        if resp.status_code == 404:
            return None
        raise SciteAPIError(f"Scite tally endpoint returned HTTP {resp.status_code}")
    except requests.RequestException as exc:
        logger.debug("Scite tally request failed for %s: %s", doi, exc)
        raise SciteAPIError(f"Scite tally request failed: {exc}") from exc


def get_tallies_batch(dois: list[str]) -> dict[str, dict]:
    """Fetch tallies for up to 500 DOIs.

    Returns ``{doi: tally_dict}``; an empty dict means no DOI had tally data.
    """
    if not dois:
        return {}
    try:
        resp = requests.post(
            f"{_BASE}/tallies",
            json=dois[:500],
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        if resp.status_code == 200:
            return _json(resp, "batch tally endpoint").get("tallies", {})
        raise SciteAPIError(
            f"Scite batch tally endpoint returned HTTP {resp.status_code}"
        )
    except requests.RequestException as exc:
        logger.debug("Scite batch tallies request failed: %s", exc)
        raise SciteAPIError(f"Scite batch tally request failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Paper metadata (includes editorialNotices / retraction info)
# ---------------------------------------------------------------------------


def get_paper(doi: str) -> dict | None:
    """Fetch paper metadata including ``editorialNotices``.

    Returns the full paper dict or ``None`` when Scite has no record.
    """
    try:
        resp = requests.get(
            f"{_BASE}/papers/{doi}",
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        if resp.status_code == 200:
            return _json(resp, "paper endpoint")
        if resp.status_code == 404:
            return None
        raise SciteAPIError(f"Scite paper endpoint returned HTTP {resp.status_code}")
    except requests.RequestException as exc:
        logger.debug("Scite paper request failed for %s: %s", doi, exc)
        raise SciteAPIError(f"Scite paper request failed: {exc}") from exc


def get_papers_batch(dois: list[str]) -> dict[str, dict]:
    """Fetch paper metadata for up to 500 DOIs.

    Returns ``{doi: paper_dict}``; an empty dict means no DOI had paper data.
    """
    if not dois:
        return {}
    try:
        resp = requests.post(
            f"{_BASE}/papers",
            json=dois[:500],
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        if resp.status_code == 200:
            return _json(resp, "batch paper endpoint").get("papers", {})
        raise SciteAPIError(
            f"Scite batch paper endpoint returned HTTP {resp.status_code}"
        )
    except requests.RequestException as exc:
        logger.debug("Scite batch papers request failed: %s", exc)
        raise SciteAPIError(f"Scite batch paper request failed: {exc}") from exc
