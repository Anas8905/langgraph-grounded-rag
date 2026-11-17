"""Offline tests: every branch of the graph, the verifier's edge cases, and checkpoint resume.

Uses a scripted LLM and bag-of-words embeddings instead of OpenAI. Bag-of-words embeddings
fall into the same trap as real ones here: "billing sync ... E-4021" pulls in the three
billing-sync chunks (E-4012, E-4102, E-4120), not the E-4021 chunk.
"""
import re
import sqlite3
import zlib

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from grounded_rag.graph import build_graph, build_vector_store, load_corpus, route_after_verify
from grounded_rag.state import initial_state
from grounded_rag.verify import verify_answer

STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "is", "it", "do", "we", "i", "how",
        "what", "does", "our", "with", "when", "should", "be", "for", "was", "that", "this"}


class KeywordEmbeddings(Embeddings):
    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * 4096
        for tok in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower()):
            if tok not in STOP:
                vec[zlib.crc32(tok.encode()) % 4096] += 1.0
        return vec

    def embed_documents(self, texts): return [self._vec(t) for t in texts]
    def embed_query(self, text): return self._vec(text)


class ScriptedLLM:
    """Returns canned replies in order; an Exception in the script is raised instead."""
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), 0

    def invoke(self, messages):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return AIMessage(content=reply)


class CountingStore:
    def __init__(self, inner):
        self.inner, self.searches = inner, []

    def similarity_search_with_score(self, query, k):
        self.searches.append(query)
        return self.inner.similarity_search_with_score(query, k=k)


@pytest.fixture(scope="module")
def store():
    return build_vector_store(KeywordEmbeddings())


def run(graph, question, thread="t1"):
    config = {"configurable": {"thread_id": thread}}
    path = [node for step in graph.stream(initial_state(question), config, stream_mode="updates")
            for node in step]
    return path, graph.get_state(config).values


def sqlite_saver():
    return SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))


# ── the three README paths ────────────────────────────────────────────────────────────

def test_grounded_first_try(store):
    llm = ScriptedLLM("E-4210 means the card network did not confirm the refund in time [kb-07]. "
                      "Do not refund again: Acme Pay retries for up to 72 hours [kb-07].")
    path, final = run(build_graph(llm, store), "What does error E-4210 mean, and should I issue the refund again?")
    assert path == ["retrieve", "generate", "verify"]
    assert final["verdict"] == "grounded" and final["attempts"] == 0


def test_grounded_after_reformulation(store):
    llm = ScriptedLLM(
        # first draft borrows the fix for the adjacent code E-4012: the dangerous failure
        "To fix error 4021, change the invoice currency and re-run the billing sync [kb-01].",
        "E-4021 billing sync error",   # code rewritten in the format the returned passages use
        "Error 4021 (E-4021) means the ledger export reused a journal reference [kb-02]. Void the "
        "duplicate entry and start a manual export; one export is capped at 5,000 journal lines [kb-02].",
    )
    counting = CountingStore(store)
    path, final = run(build_graph(llm, counting), "Our billing sync keeps failing with error 4021. How do we fix it?")
    assert path == ["retrieve", "generate", "verify", "reformulate", "retrieve", "generate", "verify"]
    assert final["verdict"] == "grounded" and final["attempts"] == 1
    assert counting.searches[1] == "E-4021 billing sync error"
    assert final["retrieved_chunks"][0] == {**final["retrieved_chunks"][0], "id": "kb-02", "match": "exact"}


def test_vector_search_alone_misses_the_adjacent_identifier(store):
    hits = store.similarity_search_with_score("Our billing sync keeps failing with E-4021. How do we fix it?", k=3)
    assert {d.metadata["id"] for d, _ in hits} == {"kb-01", "kb-03", "kb-04"}  # E-4012, E-4102, E-4120


def test_exact_identifier_is_pinned_ahead_of_vector_results(store):
    llm = ScriptedLLM("E-4021 means the ledger export reused a journal reference [kb-02].")
    _, final = run(build_graph(llm, store), "Our billing sync keeps failing with E-4021. How do we fix it?")
    assert [(c["id"], c["match"]) for c in final["retrieved_chunks"]] == \
           [("kb-02", "exact"), ("kb-01", "vector"), ("kb-03", "vector")]


def test_abstains_on_identifier_missing_from_corpus(store):
    not_covered = "The context does not cover E-4022 [kb-06]."
    llm = ScriptedLLM(not_covered, "E-4022 error code", not_covered, "E-4022 refund error", not_covered)
    path, final = run(build_graph(llm, store), "How do I resolve error E-4022 when issuing a refund?")
    assert path == ["retrieve", "generate", "verify", "reformulate", "retrieve", "generate", "verify",
                    "reformulate", "retrieve", "generate", "verify", "abstain"]
    assert final["attempts"] == 2 and final["verdict"] == "ungrounded"
    assert final["draft_answer"].startswith("Not supported by available sources.")
    assert "E-4022" in final["draft_answer"]


# ── routing ───────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("verdict,attempts,expected", [
    ("grounded", 0, "grounded"), ("grounded", 2, "grounded"),
    ("ungrounded", 0, "reformulate"), ("ungrounded", 1, "reformulate"), ("ungrounded", 2, "abstain"),
    ("fabricated", 1, "reformulate"), ("fabricated", 2, "abstain"),
])
def test_route_after_verify(verdict, attempts, expected):
    assert route_after_verify({"verdict": verdict, "attempts": attempts}) == expected


# ── deterministic verifier ────────────────────────────────────────────────────────────

CHUNKS = [{"id": c.metadata["id"], "text": c.page_content, "score": 1.0} for c in load_corpus()]
KB01, KB02, KB04, KB07, KB08 = (next(c for c in CHUNKS if c["id"] == i)
                                for i in ("kb-01", "kb-02", "kb-04", "kb-07", "kb-08"))


@pytest.mark.parametrize("answer,chunks,verdict,claim", [
    ("E-4012 retries every 10 minutes [kb-01].", [KB01], "grounded", None),
    ("E-4021 retries every 10 minutes [kb-01].", [KB01], "ungrounded", "identifier E-4021"),   # adjacent code
    ("E-4012 retries every 15 minutes [kb-01].", [KB01], "ungrounded", "number 15"),
    ("Throttling clears after 6 seconds [kb-04].", [KB04], "ungrounded", "number 6"),          # not inside "60"
    ("Retries last up to 72 hours [kb-09].", [KB07], "fabricated", "cites [kb-09]"),
    ("Retries last up to 72 hours.", [KB07], "ungrounded", "cites no retrieved chunk"),
    ("API v2.4 rejects events older than 300 seconds [kb-08].", [KB08], "grounded", None),
    ("API v2.5 rejects events older than 300 seconds [kb-08].", [KB08], "ungrounded", "identifier v2.5"),
    ("Steps:\n1. Wait up to 72 hours [kb-07].\n2. Do not refund twice [kb-07].", [KB07], "grounded", None),
    ("Error 4021 (E-4021) is a ledger export error [kb-02].", [KB02], "grounded", None),     # number inside a code
    ("Error 4021 is a billing sync error [kb-01].", [KB01], "ungrounded", "number 4021"),  # not inside E-4012
    ("E-402 is a ledger export error [kb-02].", [KB02], "ungrounded", "identifier E-402"),   # not inside E-4021
])
def test_verify_answer(answer, chunks, verdict, claim):
    got_verdict, claims = verify_answer(answer, chunks)
    assert got_verdict == verdict
    assert (claim is None and claims == []) or any(claim in c for c in claims)


# ── failure handling: retry and checkpoint resume ─────────────────────────────────────

def test_transient_llm_error_is_retried(store):
    llm = ScriptedLLM(ConnectionError("API reset"), "Retries last up to 72 hours [kb-07].")
    _, final = run(build_graph(llm, store), "What does error E-4210 mean?")
    assert final["verdict"] == "grounded" and llm.calls == 2


def test_crashed_run_resumes_from_checkpoint_without_redoing_retrieval(store):
    saver, config = sqlite_saver(), {"configurable": {"thread_id": "crash"}}
    counting = CountingStore(store)
    broken = build_graph(ScriptedLLM(RuntimeError("model unavailable")), counting, saver)
    with pytest.raises(RuntimeError):
        broken.invoke(initial_state("What does error E-4210 mean?"), config)
    assert broken.get_state(config).next == ("generate",)  # retrieval result was checkpointed

    healthy = build_graph(ScriptedLLM("Retries last up to 72 hours [kb-07]."), counting, saver)
    final = healthy.invoke(None, config)  # None = resume from the last checkpoint
    assert final["verdict"] == "grounded"
    assert len(counting.searches) == 1  # retrieve did not run again


def test_rejected_drafts_stay_inspectable_in_checkpoint_history(store):
    not_covered = "The context does not cover E-4022 [kb-06]."
    saver = sqlite_saver()
    llm = ScriptedLLM(not_covered, "E-4022", not_covered, "E-4022 refund", not_covered)
    graph = build_graph(llm, store, saver)
    run(graph, "How do I resolve error E-4022?", thread="h1")
    drafts = {s.values["draft_answer"] for s in graph.get_state_history({"configurable": {"thread_id": "h1"}})
              if s.values}  # the first snapshot (before START) is empty
    assert not_covered in drafts
    assert any(d.startswith("Not supported by available sources.") for d in drafts)
