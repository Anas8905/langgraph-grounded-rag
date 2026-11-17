# Grounded RAG agent with bounded self-correction (LangGraph)

A retrieval agent that does not let a confident wrong answer through. It retrieves, drafts
an answer, then checks the draft against the retrieved text **in plain code**. If any error
code, version, number or citation can't be found verbatim in what was retrieved, the graph
rewrites the search and tries again, at most twice. Then it abstains with
*"Not supported by available sources"* instead of guessing.

Python · LangGraph · LangChain · OpenAI · ~285 lines of application code.

```bash
cp .env.example .env        # add OPENAI_API_KEY
uv run run.py               # runs the three example questions below
```

`uv run run.py "your question"` asks anything. `uv run pytest` runs the offline test suite,
which needs no API key.

## Graph

```mermaid
graph TD;
    __start__([start]) --> retrieve
    retrieve --> generate
    generate --> verify
    verify -. grounded .-> __end__([end])
    verify -. "ungrounded, attempts < 2" .-> reformulate
    verify -. "ungrounded, attempts = 2" .-> abstain
    reformulate --> retrieve
    abstain --> __end__
```

| Node | What it does | LLM? |
|---|---|---|
| `retrieve` | Embeds the current search (reformulated query if there is one, else the question) and takes the top 3 chunks from a local vector store | embeddings only |
| `generate` | Answers the *original question* strictly from those chunks, citing `[kb-NN]` after each sentence | yes |
| `verify` | Deterministic grounding check; returns `grounded`, `ungrounded` or `fabricated` plus the reasons | **no** |
| `reformulate` | Rewrites the search using what the failed attempt revealed: which chunks came back and what the verifier couldn't find. Increments `attempts` | yes |
| `abstain` | Returns "Not supported by available sources", with the last search and the reasons | no |

The only branch is a real conditional edge on `verify` (`route_after_verify` in
[grounded_rag/graph.py](grounded_rag/graph.py)). No node contains routing logic. `fabricated` routes the
same way as `ungrounded`; the label is kept separately so you can tell the failures apart in logs.

**State** ([grounded_rag/state.py](grounded_rag/state.py)) is a typed `TypedDict`:
`query`, `reformulated_query`, `retrieved_chunks`, `draft_answer`, `verdict`, `unsupported_claims`, `attempts`.

**Checkpointer.** Every step is persisted to SQLite (`checkpoints.sqlite`), keyed by run ID:

```bash
uv run run.py --history run-1a2b3c4d   # every checkpoint: step, next node, verdict, search, drafts
uv run run.py --resume  run-1a2b3c4d   # continue a run that died mid-graph (e.g. API outage)
```

Rejected drafts aren't lost when the graph abstains. They stay in the checkpoint history,
so you can see exactly what the model tried to say.

## Why the verifier is deterministic

The verifier is the component that has to be right when the model is wrong. If it were an
LLM call, it would share the failure modes it's supposed to catch. A model that confuses
`E-4021` with `E-4012` while answering can make the same confusion while judging. Code
can't. [grounded_rag/verify.py](grounded_rag/verify.py) is 46 lines and enforces three rules:

1. Every citation `[kb-NN]` must point at a chunk that was actually retrieved. Otherwise the verdict is **fabricated**.
2. Every identifier (`E-4021`, `v2.4`) and number (`72`, `5,000`, `02:00`) in the answer must
   appear verbatim in the retrieved text. Otherwise the verdict is **ungrounded**.
3. The answer must cite at least one retrieved chunk. "I couldn't find it" with no citation is
   **ungrounded**, so it leads to a retry and then abstention, never to a silent pass.

Matching is boundary-aware: `E-402` doesn't match inside `E-4021`, and `15` doesn't match inside `150`.
In a corpus full of adjacent identifiers, substring matching would wave exactly the wrong answers through.

It's also deterministic in the testing sense: same input, same verdict, every run. You can
unit-test it ([tests/test_graph.py](tests/test_graph.py) has 9 verifier cases) and explain a rejection
by pointing at a line. The trade-off is strictness. A correct answer that rephrases a number
("a quarter of an hour" for 15 minutes) fails, so the answer prompt tells the model to copy
figures exactly. It's the right trade for support, finance and ops content, where a wrong
number costs more than a retry.

## The corpus is built to make retrieval fail

[grounded_rag/corpus.md](grounded_rag/corpus.md) is a 12-chunk error runbook for a fictional payments API.
It deliberately contains adjacent identifiers: `E-4012`, `E-4021`, `E-4102`, `E-4120`, `E-4201`,
`E-4202`, `E-4210`, plus API `v2.3` and `v2.4`. Embedding search ranks by meaning, and to an
embedding model these codes are nearly the same token. Ask about `E-4021` while describing a
billing-sync problem, and the vector store returns the three *billing sync* chunks (E-4012,
E-4102, E-4120), not E-4021. A plain RAG pipeline would then confidently tell you how to fix
the wrong error. That's the case the retry path exists for.

## Example runs

The three questions `uv run run.py` asks, one per path through the graph:

| # | Question | Expected path |
|---|---|---|
| 1 | What does error E-4210 mean, and should I issue the refund again? | grounded on the first try |
| 2 | Our billing sync keeps failing with E-4021. How do we fix it? | first search returns the adjacent billing-sync codes → verifier rejects the draft → reformulated search finds E-4021 → grounded |
| 3 | How do I resolve error E-4022 when issuing a refund? | E-4022 doesn't exist (E-4021, E-4202 do) → two reformulations → abstain |

<!-- EXAMPLE-RUNS: replace the three blocks below with real output from `uv run run.py` -->

### 1. Grounded on the first try

```text
(paste output of: uv run run.py "What does error E-4210 mean, and should I issue the refund again?")
```

### 2. Grounded after reformulation

```text
(paste output of: uv run run.py "Our billing sync keeps failing with E-4021. How do we fix it?")
```

### 3. Abstention

```text
(paste output of: uv run run.py "How do I resolve error E-4022 when issuing a refund?")
```

## Tests

`uv run pytest` runs 23 tests offline, with no API key. They use a scripted LLM and
bag-of-words embeddings, which fall into the same adjacent-identifier trap as real
embeddings (a test asserts the first search misses kb-02). Coverage:

- all three paths above, asserting the exact node sequence;
- every routing case of `route_after_verify`;
- verifier edge cases: adjacent codes, `6` vs `60`, version `v2.4` vs `v2.5`, fabricated citation, no citation, numbered lists;
- a transient API error retried by the node's `RetryPolicy`;
- a run that crashes mid-graph, then resumes from the SQLite checkpoint without re-running retrieval;
- rejected drafts still readable from checkpoint history after abstention.

## Layout

```
grounded_rag/state.py    typed state
grounded_rag/verify.py   deterministic grounding check
grounded_rag/graph.py    nodes, conditional edge, compile with checkpointer
grounded_rag/corpus.md   12-chunk demo corpus
run.py                   CLI: prints every state transition; --history, --resume
tests/test_graph.py      offline tests
```

Swapping in a production store is one function. The graph only calls
`similarity_search_with_score`, so FAISS, Chroma or pgvector drop into `build_vector_store`.
