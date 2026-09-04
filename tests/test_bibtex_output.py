"""Truthful BibTeX fallback and trash-status regression tests."""

from unittest.mock import patch

import pytest

from zotero_mcp.better_bibtex_client import BetterBibTexError
from zotero_mcp.client import generate_bibtex

ITEM = {
    "key": "ITEM0001",
    "data": {
        "key": "ITEM0001",
        "itemType": "journalArticle",
        "title": "Useful result",
        "publicationTitle": "Journal of Tests",
        "date": "2024",
        "creators": [{"creatorType": "author", "firstName": "Ada", "lastName": "Lovelace"}],
    },
}


def _bbt(*, result=None, error=None):
    running = patch(
        "zotero_mcp.better_bibtex_client.ZoteroBetterBibTexAPI.is_zotero_running",
        return_value=True,
    )
    kwargs = {"side_effect": error} if error else {"return_value": result}
    export = patch(
        "zotero_mcp.better_bibtex_client.ZoteroBetterBibTexAPI.export_bibtex",
        **kwargs,
    )
    return running, export


@pytest.mark.parametrize(
    "result,error",
    [
        ("", None),
        ("  \n", None),
        (None, BetterBibTexError("no citekey")),
    ],
)
def test_bbt_failure_falls_back_to_nonempty_local_bibtex(result, error):
    running, export = _bbt(result=result, error=error)
    with running, export:
        output = generate_bibtex(ITEM)
    assert output.startswith("@article{")
    assert "Useful result" in output


def test_trashed_item_is_marked_for_bbt_and_fallback():
    item = {"key": ITEM["key"], "data": {**ITEM["data"], "deleted": True}}
    running, export = _bbt(result="@article{realKey,\n  title={T}\n}")
    with running, export:
        assert generate_bibtex(item).startswith("% Status: in trash\n@article{realKey")

    running, export = _bbt(error=BetterBibTexError("trashed"))
    with running, export:
        assert generate_bibtex(item).startswith("% Status: in trash\n@article{")


def test_item_without_key_never_renders_none_in_citekey():
    item = {"data": {key: value for key, value in ITEM["data"].items() if key != "key"}}
    with patch(
        "zotero_mcp.better_bibtex_client.ZoteroBetterBibTexAPI.is_zotero_running",
        return_value=False,
    ):
        output = generate_bibtex(item)
    assert "None" not in output.splitlines()[0]
