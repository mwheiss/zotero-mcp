# Encoder-only passage selection

For a passage-indexed collection with the optional cross-encoder reranker
**disabled**, `limit=K` is a parent-paper limit, not a passage limit.

1. Retrieve chunks ranked by Chroma, widening until K+1 distinct papers are
   visible, the filtered candidate set is exhausted, or a protective candidate
   ceiling is reached when the collection count is unavailable.
2. If a K+1-th paper exists, retain the ranked prefix immediately before its
   first chunk. Discard that chunk and every following chunk, including later
   chunks from otherwise selected papers.
3. Without an excluded paper, retain the entire retrieved prefix. Confirmed
   exhaustion supplies the natural end of the ranking; do not switch to a
   different three-passage/overlap/score-bonus heuristic.
4. Group by paper, preserving first-appearance order. Its first chunk is primary;
   its remaining retained chunks are supporting passages. Report the primary
   similarity without a supporting-passage bonus.

Thus a single-paper filter can expose **every indexed chunk of that paper**,
including low or negative similarity scores. The query orders the passages; it
no longer determines which of that paper's chunks survive an implicit cap.
This is deliberate audit-oriented behavior, not a declaration that every chunk
is relevant to every claim. `limit=1` is sufficient for an exact single-paper
filter; increasing it cannot increase the number of matching papers.

## Exhaustion edge case

For the fixed ranking `A0 A1 A2 A3 A4 A5 A6 A7 B0`:

- K=1: B0 establishes the boundary; all eight A chunks are retained.
- K=2: there is no excluded paper; all eight A chunks and B0 are retained.
- K>2: the same exhausted set is retained.

Increasing the paper limit cannot remove a previously surfaced passage under
this selection policy **for a fixed ranking and fixed filtered candidate set**.
Chroma's approximate search, unstable tie order, and concurrent index updates
are outside that guarantee; this is not a new exact-nearest-neighbor algorithm.

## Completeness and the visible response

The search engine adds `passage_selection` with `policy`, `stop_reason`,
`selection_complete`, `candidate_chunks`, and `returned_chunks`.
`stop_reason` is `excluded_parent`, `exhausted`, or `candidate_limit`.
Completeness describes selection within the indexed ranking, not completeness
of an original PDF or recall against all potentially relevant literature.

When the count is unavailable, the configured maximum chunk count is only a
protective fetch bound: existing documents may have been indexed with another
configuration. Hitting that bound with a full response and no excluded paper
sets `selection_complete=false`. All retrieved chunks are still exposed, and
an explicit partial-coverage warning reaches the MCP text, including global
multi-library searches. No silent cap is introduced.

The engine field is additive Python-level metadata. The existing public MCP
text interface is retained; the partial-coverage warning is rendered there.
Primary and supporting IDs/hashes remain available even if live Zotero parent
metadata cannot be fetched or a supporting preview is empty. The legacy
structured fields still do not enumerate every supporting passage; consumers
must parse the supporting IDs in the returned text as before.

## Costs and boundaries

Search returns previews/IDs, not all complete 6k texts inline. Expand selected
chunks with `zotero_get_semantic_context`, use their exact hashes, and reuse
completed comparisons from a provenance-aware cache. Long books can produce
large preview responses and many expansions: there is intentionally no hidden
relevance threshold or passage cap. A future bounded quick-search mode should
make truncation/pagination explicit instead of overloading the paper limit.

The adaptive fetch algorithm and its database/query count are unchanged. The
patch changes what is retained after those fetches; retaining more previews
increases formatting, response-size, and downstream context costs.

Optional cross-encoder reranking and non-passage indexes keep their existing
behavior. Cross-encoder candidate-pool policy is a separate retrieval contract;
this patch does not silently broaden or redefine it. Multi-library searches
still apply the paper-prefix policy within each library and merge the resulting
paper rankings; this patch does not introduce a globally merged passage cutoff.

No index schema, embeddings, stored chunk IDs, or content hashes change.
Deploying the patched code requires the usual service restart, not a rebuild or
re-embedding. Original-attachment fidelity and missing/unindexed content remain
separate concerns.
