"""Contract checks for guidance prompts and attachable Zotero resources."""

import asyncio

from zotero_mcp.prompts import (
    expand_from_paper,
    find_contradicting_evidence,
    literature_review,
)
from zotero_mcp.server import mcp


def test_research_prompts_verify_semantic_passages_before_claims():
    review = literature_review("muon capture")
    contradictions = find_contradicting_evidence("capture favors heavy atoms")

    assert "zotero_get_semantic_context" in review
    assert "Chunk Hash" in review
    assert "zotero_get_semantic_context" in contradictions


def test_expansion_prompt_requires_convergent_confirmed_writes():
    prompt = expand_from_paper("ABCD1234")

    assert "ask before writing" in prompt
    assert "if_exists='reuse'" in prompt
    assert "duplicate mode" in prompt


def test_resource_descriptions_state_scope_and_fulltext_limits(monkeypatch):
    monkeypatch.setenv("ZOTERO_MCP_TOOL_PROFILE", "all")

    async def listed():
        resources = await mcp.list_resources()
        templates = await mcp.list_resource_templates()
        return resources, templates

    resources, templates = asyncio.run(listed())
    descriptions = {
        str(getattr(resource, "uri", None) or resource.uri_template): resource.description
        for resource in [*resources, *templates]
    }

    assert "parent key" in descriptions["zotero://collections"]
    assert "does not include attachment full text" in descriptions[
        "zotero://items/{item_key}"
    ]
    assert "up to 200 items" in descriptions[
        "zotero://collections/{collection_key}/items"
    ]
