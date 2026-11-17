"""Grounded retrieval with bounded self-correction.

    retrieve -> generate -> verify --grounded--------------------------> END
                               |---ungrounded, attempts < 2--> reformulate -> retrieve
                               |---ungrounded, attempts = 2--> abstain -----> END

The LLM writes answers and search queries. It never decides whether an answer is grounded:
that is `verify_answer`, plain code.
"""
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Checkpointer, RetryPolicy

from .state import RAGState
from .verify import IDENTIFIER, appears_verbatim, verify_answer

TOP_K = 3
MAX_ATTEMPTS = 2
CORPUS = Path(__file__).with_name("corpus.md")

ANSWER_PROMPT = """You answer questions about the Acme Pay API using only the context passages.
- Use only facts stated in the passages. Do not use outside knowledge.
- After each sentence, cite the passage it came from in square brackets, e.g. [kb-03].
- Copy error codes, version numbers and figures exactly as they are written.
- If the passages do not answer the question, say the context does not cover it and name
  what is missing. Do not guess and do not substitute a similar error code."""

REFORMULATE_PROMPT = """A search over the Acme Pay error runbook returned passages that did not
support an answer. Write one new search query that will find the passage that does.
- Keep the identifiers from the question, written in the exact format the passages use
  (e.g. E-4012, not 4012): identifiers in that format are matched verbatim.
- Drop wording that pulled in the wrong passages; the verifier findings say what was missing.
Return only the query text."""


def load_corpus(path: Path = CORPUS) -> list[Document]:
    """Each '## kb-NN' section of the markdown file is one chunk."""
    sections = re.split(r"(?m)^## ", path.read_text(encoding="utf-8"))[1:]
    return [Document(page_content=" ".join(body.split()), metadata={"id": head.strip()})
            for head, _, body in (s.partition("\n") for s in sections)]


def build_vector_store(embeddings: Embeddings, path: Path = CORPUS) -> VectorStore:
    """Local in-process vector store (cosine similarity). Any LangChain VectorStore works here:
    the graph only calls similarity_search_with_score, so FAISS, Chroma or pgvector drop in."""
    return InMemoryVectorStore.from_documents(load_corpus(path), embeddings)


def route_after_verify(state: RAGState) -> str:
    if state["verdict"] == "grounded":
        return "grounded"
    return "reformulate" if state["attempts"] < MAX_ATTEMPTS else "abstain"


def build_graph(llm: BaseChatModel, store: VectorStore, checkpointer: Checkpointer = None,
                corpus: list[Document] | None = None):
    """Compile the graph. Always checkpointed: SQLite from the CLI, in-memory by default."""
    corpus = load_corpus() if corpus is None else corpus

    def retrieve(state: RAGState) -> dict:
        # Hybrid. Embeddings rank E-4021's chunk 4th even for the query "E-4021", so chunks that
        # contain an identifier from the search verbatim go first; vector results fill the rest.
        search = state["reformulated_query"] or state["query"]
        wanted = IDENTIFIER.findall(search)
        chunks = [{"id": d.metadata["id"], "text": d.page_content, "score": 1.0, "match": "exact"}
                  for d in corpus if any(appears_verbatim(i, d.page_content) for i in wanted)]
        for doc, score in store.similarity_search_with_score(search, k=TOP_K):
            if doc.metadata["id"] not in {c["id"] for c in chunks}:
                chunks.append({"id": doc.metadata["id"], "text": doc.page_content,
                               "score": round(float(score), 3), "match": "vector"})
        return {"retrieved_chunks": chunks[:TOP_K]}

    def generate(state: RAGState) -> dict:
        context = "\n".join(f"[{c['id']}] {c['text']}" for c in state["retrieved_chunks"])
        reply = llm.invoke([("system", ANSWER_PROMPT),
                            ("human", f"Context:\n{context}\n\nQuestion: {state['query']}")])
        return {"draft_answer": reply.content.strip()}

    def verify(state: RAGState) -> dict:
        verdict, unsupported = verify_answer(state["draft_answer"], state["retrieved_chunks"])
        return {"verdict": verdict, "unsupported_claims": unsupported}

    def reformulate(state: RAGState) -> dict:
        returned = "\n".join(f"- [{c['id']}] {c['text'][:80]}..." for c in state["retrieved_chunks"])
        findings = "\n".join(f"- {claim}" for claim in state["unsupported_claims"])
        reply = llm.invoke([("system", REFORMULATE_PROMPT), ("human",
            f"Question: {state['query']}\n"
            f"Last search: {state['reformulated_query'] or state['query']}\n"
            f"Passages that search returned:\n{returned}\n"
            f"Verifier findings:\n{findings}")])
        return {"reformulated_query": reply.content.strip().strip('"'),
                "attempts": state["attempts"] + 1}

    def abstain(state: RAGState) -> dict:
        reasons = "; ".join(state["unsupported_claims"][:3])
        return {"draft_answer": (
            "Not supported by available sources. "
            f"After {state['attempts']} reformulations (last search: \"{state['reformulated_query']}\"), "
            f"no retrieved passage supports an answer: {reasons}.")}

    llm_retry = RetryPolicy(max_attempts=3)  # transient API errors; the checkpointer covers the rest
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve, retry_policy=llm_retry)
    graph.add_node("generate", generate, retry_policy=llm_retry)
    graph.add_node("verify", verify)
    graph.add_node("reformulate", reformulate, retry_policy=llm_retry)
    graph.add_node("abstain", abstain)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", route_after_verify,
                                {"grounded": END, "reformulate": "reformulate", "abstain": "abstain"})
    graph.add_edge("reformulate", "retrieve")
    graph.add_edge("abstain", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
