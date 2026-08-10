# Implementation plan: Qwen3 reranking with `ik_llama.cpp` + `zotero-mcp`

One useful correction from the earlier inspection: the current `zotero-mcp` branch defaults passage chunking to **1500 characters with 200-character overlap**, not 6000 characters. That makes CPU reranking considerably more attractive because each candidate is relatively small.

The existing `zotero-mcp` architecture is already close to ideal: it over-fetches Chroma candidates, optionally reranks passages, and only then aggregates passages back into Zotero items. The implementation should therefore mostly be **a new reranker backend**, not a retrieval rewrite.

On the `ik_llama.cpp` side, there is also less work than expected: the server already has rerank-related task plumbing, while raw logits are available internally. The main missing pieces are a public rerank endpoint and the Qwen3-specific yes/no-logit scoring.

---

## 1. Target architecture

```text
                    zotero-mcp
                       │
                       │ semantic_search(query)
                       ▼
               Qwen3 embedding model
                       │
                       ▼
                  Chroma index
                       │
             dense top-N passages
                       │
        candidate cap + paper diversity
                       │
                       ▼
          POST http://127.0.0.1:8182
                  /v1/rerank
                       │
                       ▼
        Qwen3-Reranker-0.6B GGUF
                 ik_llama.cpp
                       │
             yes/no final logits
                       │
      score = logit_yes - logit_no
                       │
                       ▼
              reranked passages
                       │
         aggregate passages by paper
                       │
                       ▼
              top Zotero results
```

### Initial model

Start with:

```text
Qwen3-Reranker-0.6B
CPU only
Q6/Q8-quality GGUF
~40 candidate passages maximum
```

Do **not** make 4B part of the initial implementation. Once the entire pipeline is benchmarkable, testing 4B becomes trivial.

---

# 2. Scope decision for the `ik_llama.cpp` patch

## Do this

Implement a **small Qwen3-compatible `/v1/rerank` API** which evaluates query/document prompts and directly reads the `yes` and `no` logits.

Qwen3 reranking fundamentally scores relevance from the difference between those two logits:

```text
raw_score = logit_yes - logit_no

probability = sigmoid(raw_score)
```

Use `raw_score` for sorting.

## Do not do this initially

Do **not**:

- port llama.cpp's complete rank-pooling implementation;
- change the GGML computation graph;
- create a generic classification framework;
- implement arbitrary reranker architectures;
- generate an actual `yes` or `no` token;
- use completion logprobs as a workaround.

The external API can nevertheless be generic enough to support additional reranker implementations later.

---

# 3. Phase A — establish a reference before touching `ik_llama.cpp`

Before implementing the server patch, create a tiny reference test using the official Qwen model.

### Fixture

Use perhaps 10 fixed examples:

```text
Query:
"negative muon X-ray measurements of archaeological copper"

Documents:
1. Actual MIXE archaeological copper passage
2. Generic muonic atom theory passage
3. Copper crystallography paper
4. Positive muon spectroscopy passage
...
```

For every pair save:

```json
{
  "query": "...",
  "document": "...",
  "raw_score": 7.8213,
  "rank": 1
}
```

The purpose is **not exact floating-point reproduction** after GGUF quantization.

We want:

- same broad ordering;
- relevant passages clearly above irrelevant ones;
- high rank correlation;
- no pathological GGUF conversion/model-template problem.

This becomes the regression fixture for the `ik_llama.cpp` patch.

---

# 4. Phase B — `ik_llama.cpp` patch

## B1. Add a dedicated server mode

Expose something like:

```bash
llama-server \
    -m Qwen3-Reranker-0.6B-Q6.gguf \
    --reranking \
    --port 8182 \
    -c 4096
```

`--reranking` should:

- enable the reranker route;
- resolve the positive and negative score tokens once at startup;
- validate that the loaded tokenizer/model is compatible;
- avoid exposing a silently nonsensical endpoint for arbitrary causal models.

### Important

Do **not hardcode token IDs**.

Resolve:

```text
"yes"
"no"
```

using the loaded tokenizer and verify that each corresponds to exactly one scoring token in the required context.

That keeps the implementation independent of a particular GGUF conversion.

---

## B2. HTTP API

Support at minimum:

```text
POST /v1/rerank
```

Optionally alias:

```text
/rerank
/reranking
```

### Request

```json
{
  "query": "Which papers measured oxygen content in archaeological copper using MIXE?",
  "documents": [
    "Passage one...",
    "Passage two...",
    "Passage three..."
  ],
  "instruction": "Retrieve scientific-literature passages that are directly relevant to the research query.",
  "top_n": 10,
  "return_documents": false
}
```

### Response

Return **both raw and normalized scores**:

```json
{
  "model": "Qwen3-Reranker-0.6B",
  "results": [
    {
      "index": 2,
      "score": 8.731,
      "probability": 0.999839
    },
    {
      "index": 0,
      "score": 3.214,
      "probability": 0.961358
    },
    {
      "index": 1,
      "score": -2.145,
      "probability": 0.104799
    }
  ],
  "usage": {
    "prompt_tokens": 1273
  }
}
```

`score` is the canonical reranking quantity.

`probability` is useful for humans but should **not** be interpreted as a calibrated probability that a paper is relevant.

---

# 5. Qwen prompt construction

Use Qwen's reranker chat template faithfully rather than inventing a prompt.

Conceptually it is:

```text
SYSTEM
Judge Query/Instruction/Document relevance;
the decision is yes or no.

USER
Instruction: <scientific retrieval instruction>
Query: <query>
Document: <candidate passage>

ASSISTANT
<think>

</think>

<score position>
```

The implementation should reproduce the model's official reranker template exactly from the model metadata/reference implementation rather than baking a paraphrased version like the above into tests.

For `zotero-mcp`, make the instruction configurable.

A good initial one is essentially:

```text
Given a scientific literature search query, retrieve relevant passages
that identify papers addressing the query.
```

That keeps reranking semantically aligned with the Qwen embedding query instruction already used by `zotero-mcp`.

---

# 6. B3 — evaluate without generating

This is the important implementation detail.

For every candidate:

1. Construct the complete reranker prompt.
2. Tokenize it.
3. Feed it through the normal model.
4. Request logits for the **final prompt position**.
5. Read:

   ```cpp
   logits[yes_token]
   logits[no_token]
   ```

6. Calculate:

   ```cpp
   score = logits[yes_token] - logits[no_token];
   ```

7. Finish the task immediately.

No sampling.

No decoded token.

No temperature/top-k/etc.

`ik_llama.cpp` already exposes the required raw-logit primitives, so no new model API should be necessary.

---

# 7. B4 — use the existing server scheduler

Don't initially implement an elaborate custom batch executor.

Submit each document as a rerank task and let the existing server scheduler continuously batch them.

The server handler would conceptually do:

```text
HTTP request
    │
    ├── rerank task document 0
    ├── rerank task document 1
    ├── rerank task document 2
    ├── ...
    │
    ▼
continuous batching
    │
    ▼
collect scores
    │
    ▼
sort
    │
    ▼
HTTP response
```

### One low-level thing to be careful about

When several sequences are decoded together, don't assume:

```cpp
llama_get_logits_ith(ctx, -1)
```

belongs to the desired rerank task.

Track which output row corresponds to each sequence's final prompt token and retrieve that specific logits row.

That deserves an explicit test because it is exactly the sort of bug that can produce plausible-looking but completely wrong scores.

---

# 8. Files likely touched in `ik_llama.cpp`

Approximate patch surface:

```text
examples/server/server.cpp
    add /v1/rerank route
    validate JSON request
    queue document tasks
    gather/sort results

examples/server/server-task.h
    extend existing RERANK task/result data

examples/server/server-context.h
    reranker state / token IDs
    send_rerank declaration

examples/server/server-context.cpp
    detect completion of rerank prompt
    retrieve final-position logits
    calculate score
    return task result

common/arg.*
    --reranking option
```

Potentially also:

```text
examples/server/README.md
tests/...
```

Crucially, ideally **nothing under the model graph implementation** needs modification.

---

# 9. `ik_llama.cpp` tests

Before touching Zotero, require:

### API tests

```text
✓ query required
✓ documents required
✓ empty documents rejected
✓ invalid top_n rejected
✓ original document indexes preserved
✓ top_n respected
```

### Scoring tests

Given mocked logits:

```text
yes =  5
no  =  2  → score =  3

yes = -1
no  =  4  → score = -5
```

Verify exact output.

### Scheduler test

Have multiple reranking tasks finish/interleave in different orders and verify each score stays associated with the correct document index.

### Real-model integration test

Compare the GGUF server against the Phase-A Qwen reference fixture:

```text
HF Qwen3-Reranker
        vs.
ik_llama GGUF
```

Acceptance should be based primarily on **ranking**, not exact score equality.

---

# 10. Phase C — refactor `zotero-mcp` rerankers behind an interface

The existing reranker already exposes operations equivalent to:

```python
rerank()
rerank_with_scores()
```

Turn that into a small backend abstraction:

```python
class Reranker(Protocol):
    def rerank_with_scores(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        ...
```

Implement:

```text
SentenceTransformersReranker
IkLlamaReranker
```

Keep the old class name as an alias if useful for compatibility.

---

# 11. Configuration

Extend the reranker config to something along these lines:

```json
"reranker": {
  "enabled": true,
  "backend": "ik_llama",
  "model": "Qwen3-Reranker-0.6B",
  "base_url": "http://127.0.0.1:8182",

  "instruction": "Given a scientific literature search query, retrieve relevant passages that identify papers addressing the query.",

  "candidate_multiplier": 4,
  "max_candidates": 40,
  "max_chunks_per_item": 3,

  "timeout_seconds": 20,
  "fail_open": true
}
```

Keep backward compatibility:

```text
backend absent
    ↓
sentence_transformers
```

so existing configurations continue working.

---

# 12. Candidate selection needs one small improvement

This is probably the most important change inside the existing Zotero search algorithm.

Passage retrieval can widen the Chroma candidate set to obtain enough distinct parent items. That's fine for a tiny MiniLM reranker, but undesirable for a Qwen CPU reranker.

Instead:

```text
Chroma candidates
      │
      ▼
sort by dense similarity
      │
      ▼
diversity-aware candidate selection
      │
      ├─ max 3 chunks/paper
      └─ max 40 chunks total
      │
      ▼
Qwen reranker
```

For example:

```python
selected = []

for candidate in dense_candidates:
    if chunks_from(candidate.item) >= 3:
        continue

    selected.append(candidate)

    if len(selected) >= 40:
        break
```

This prevents one 300-page book with many similar chunks from consuming almost the entire CPU reranking budget.

---

# 13. Preserve both scores

While changing this code, fix the current score ambiguity.

Reranking should determine the **ordering**, while vector similarity and reranker relevance remain separate values.

Return something conceptually like:

```json
{
  "embedding_similarity": 0.8124,
  "reranker_score": 6.381,
  "reranker_probability": 0.9983
}
```

Don't overwrite one with the other.

The scales mean different things.

For MCP output, something like:

```text
Relevance: 0.998
Embedding similarity: 0.812
```

is reasonable, provided `Relevance` is clearly the sigmoid-normalized reranker score.

Internally, retain the raw score too.

---

# 14. Failure behaviour

This should be **fail-open**.

```text
ik_llama available
      │
      └── rerank normally

ik_llama unavailable / timeout / malformed response
      │
      └── warning + retain dense-vector ranking
```

A local auxiliary model going down should not break:

```text
zotero_semantic_search
```

entirely.

It should merely reduce ranking quality temporarily.

---

# 15. Startup/warmup behaviour

The SentenceTransformers backend needs process-wide model caching because the model is loaded in Python.

For `ik_llama`:

```text
no Python model cache needed
```

Instead, startup warmup can become:

```text
GET /health
```

or optionally one tiny reranker request.

That establishes immediately whether the configured local reranking service is usable.

---

# 16. `zotero-mcp` tests

Extend the semantic-search test suite with:

```text
✓ old SentenceTransformers configuration still works

✓ ik_llama request JSON is correct

✓ returned indexes correctly reorder candidates

✓ raw reranker scores survive result construction

✓ embedding score remains separately available

✓ timeout falls back to dense ordering

✓ HTTP 500 falls back to dense ordering

✓ malformed server response falls back safely

✓ max_candidates is enforced

✓ max_chunks_per_item is enforced

✓ reranking occurs before parent-paper aggregation

✓ parent grouping preserves reranked passage ordering
```

---

# 17. Phase D — benchmark before choosing defaults

Create a real Zotero retrieval benchmark rather than judging a couple of hand-picked searches.

Aim for **50–100 actual literature-search queries**.

Compare:

```text
A. Dense retrieval only

B. Dense
   + current MiniLM reranker

C. Dense
   + Qwen3-Reranker-0.6B ik_llama

D. optional:
   Dense
   + Qwen3-Reranker-4B ik_llama
```

Test candidate sets:

```text
20
40
60
```

and perhaps:

```text
Q8
Q6
Q5
```

## Measure quality

```text
MRR@10
nDCG@10
Hit@5 / Hit@10
known-relevant-paper rank
```

Also separately measure:

```text
candidate recall before reranking
```

because reranking cannot recover a paper Chroma never supplied.

## Measure performance

```text
median search latency
p95 search latency
reranker-only latency
prompt tokens/second
peak RAM
CPU utilisation
```

That gives a rational answer to:

> 0.6B or 4B?

rather than guessing from model sizes.

---

# 18. Phase E — only then optimize `ik_llama`

If profiling says inference overhead is important, the next optimization is **not a bigger quantization adventure**.

It is batching.

Potential Phase 2:

```text
same instruction + same query
           │
           ├──── document A
shared ────┼──── document B
prefix     ├──── document C
           └──── document D
```

Possible optimizations:

1. multi-sequence batched prompt evaluation;
2. reuse/share the instruction + query prefix;
3. batch score extraction;
4. tune parallel slots/core allocation.

But leave these **out of the first patch**.

The existing server scheduler gives a working baseline first.

---

# 19. Suggested Git history

## `ik_llama.cpp`

Branch:

```text
feat/qwen3-reranker-server
```

Commits:

```text
1. server: expose existing rerank task through HTTP API

2. server: score Qwen3 rerank tasks from yes/no logits

3. server: add reranker mode and model validation

4. tests: add Qwen3 reranker API and scoring coverage

5. docs: document /v1/rerank
```

Keeping the raw-logit machinery separate from the HTTP API makes the patch easier to review.

## `zotero-mcp`

From:

```text
fix/semantic-index-reconciliation
```

create:

```text
feat/ik-llama-reranker
```

Commits:

```text
1. semantic: abstract reranker backend

2. semantic: add ik_llama HTTP reranker

3. semantic: cap and diversify reranker candidates

4. semantic: propagate reranker and embedding scores separately

5. semantic: add fail-open reranker handling

6. tests: cover remote reranker integration

7. docs: document local Qwen reranking
```

---

# 20. Definition of done

```text
[ ] Qwen3-Reranker-0.6B GGUF loads correctly in ik_llama

[ ] /v1/rerank performs zero generated tokens

[ ] scores come directly from final-position yes/no logits

[ ] GGUF ordering agrees closely with the HF reference fixture

[ ] multiple simultaneous documents cannot exchange/misassociate logits

[ ] zotero-mcp can select ik_llama through configuration

[ ] old SentenceTransformers backend still works

[ ] maximum CPU reranking workload is explicitly bounded

[ ] candidate selection preserves diversity across parent Zotero items

[ ] reranking happens before passage→paper aggregation

[ ] embedding and reranker scores remain separately visible

[ ] failure of ik_llama falls back to dense retrieval

[ ] semantic-search tests all pass

[ ] real-corpus benchmark compares dense / MiniLM / Qwen-0.6B

[ ] 4B is considered only after measured 0.6B results
```

---

# 21. Recommended implementation strategy

Keep **both patches surprisingly small**.

`ik_llama.cpp`:

```text
existing causal model inference
+ existing rerank task plumbing
+ one logits-based scorer
+ one HTTP endpoint
```

`zotero-mcp`:

```text
existing reranking abstraction
+ HTTP implementation
+ bounded candidate selection
+ proper score propagation
```

Avoid turning the first iteration into a port of llama.cpp's general reranking subsystem. **Qwen3 is the target model, its scoring rule is simple, and `ik_llama` already contains most of the server machinery needed to execute the tasks.**

If the baseline works well, the interesting later optimization is shared-prefix/batched CPU execution—not more retrieval architecture.
