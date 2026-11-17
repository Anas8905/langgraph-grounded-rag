"""Typed state shared by every node in the graph."""
from typing import Literal, TypedDict

Verdict = Literal["grounded", "ungrounded", "fabricated"]


class Chunk(TypedDict):
    id: str        # e.g. "kb-07"; answers cite chunks as [kb-07]
    text: str
    score: float   # cosine similarity to the search query


class RAGState(TypedDict):
    query: str                      # the user's question, never modified
    reformulated_query: str | None  # latest rewritten search query (None on the first pass)
    retrieved_chunks: list[Chunk]   # top-k chunks for the current search
    draft_answer: str               # model draft, or the abstention message
    verdict: Verdict | None         # set by the deterministic verifier
    unsupported_claims: list[str]   # why the verifier rejected the draft
    attempts: int                   # reformulations performed so far


def initial_state(query: str) -> RAGState:
    return RAGState(query=query, reformulated_query=None, retrieved_chunks=[], draft_answer="",
                    verdict=None, unsupported_claims=[], attempts=0)
