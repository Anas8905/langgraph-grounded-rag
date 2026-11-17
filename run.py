"""Run the grounded-RAG graph and print every state transition.

    uv run run.py                      # the three README examples
    uv run run.py "your question"      # any question
    uv run run.py --history RUN_ID     # every checkpoint of a past run, from SQLite
    uv run run.py --resume RUN_ID      # continue a run that stopped mid-graph
"""
import argparse
import os
import re
import sqlite3
import sys
import textwrap
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver

from grounded_rag.graph import build_graph, build_vector_store
from grounded_rag.state import initial_state

EXAMPLES = [
    "What does error E-4210 mean, and should I issue the refund again?",  # grounded first try
    "Our billing sync keeps failing with error 4021. How do we fix it?",   # grounded after reformulation
    "How do I resolve error E-4022 when issuing a refund?",                # E-4022 does not exist: abstain
]


def label(chunk: dict) -> str:
    """'kb-02 E-4021 (exact)' or 'kb-09 Rate limits (0.62)'."""
    title = re.split(r" · |\. ", chunk["text"], maxsplit=1)[0]
    return f"{chunk['id']} {title} ({'exact' if chunk['match'] == 'exact' else format(chunk['score'], '.2f')})"


def show(node: str, update: dict) -> None:
    print(f"\n── {node} {'─' * (64 - len(node))}")
    for key, value in update.items():
        if key == "retrieved_chunks":
            value = "  |  ".join(label(c) for c in value)
        elif key == "unsupported_claims":
            value = "; ".join(value) or "none"
        elif key == "verdict":
            value = str(value).upper()
        print(textwrap.fill(str(value), width=100, initial_indent=f"   {key + ':':<20} ",
                            subsequent_indent=" " * 24))


def run(graph, run_id: str, inputs) -> None:
    config = {"configurable": {"thread_id": run_id}}
    for step in graph.stream(inputs, config, stream_mode="updates"):
        for node, update in step.items():
            show(node, update)
    final = graph.get_state(config).values
    if not final:
        sys.exit(f"No checkpoints for run {run_id}.")
    print(f"\n══ result [{run_id}] verdict={final['verdict']} attempts={final['attempts']}")
    print(textwrap.fill(final["draft_answer"], width=96))
    print(f"   (inspect: uv run run.py --history {run_id})\n")


def history(graph, run_id: str) -> None:
    snapshots = [s for s in graph.get_state_history({"configurable": {"thread_id": run_id}}) if s.values]
    if not snapshots:
        sys.exit(f"No checkpoints for run {run_id}.")
    last_draft = ""
    for snap in reversed(snapshots):
        v = snap.values
        print(f"step {snap.metadata.get('step'):>2}  next={','.join(snap.next) or 'END':<12} "
              f"attempts={v['attempts']}  verdict={str(v['verdict']):<10}  "
              f"search={v['reformulated_query'] or v['query']!r}")
        if v["draft_answer"] != last_draft:  # print each draft once, where it first appears
            print(f"          draft={v['draft_answer'][:110]!r}")
            last_draft = v["draft_answer"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--history", metavar="RUN_ID")
    parser.add_argument("--resume", metavar="RUN_ID")
    args = parser.parse_args()

    load_dotenv()
    checkpointer = SqliteSaver(sqlite3.connect("checkpoints.sqlite", check_same_thread=False))
    if args.history:  # read-only: no model or index needed
        return history(build_graph(None, None, checkpointer), args.history)
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), temperature=0)
    store = build_vector_store(OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")))
    graph = build_graph(llm, store, checkpointer)

    if args.resume:
        return run(graph, args.resume, None)  # None = continue from the last checkpoint
    for question in [args.question] if args.question else EXAMPLES:
        print(f"\n{'═' * 64}\nQ: {question}")
        run(graph, f"run-{uuid.uuid4().hex[:8]}", initial_state(question))


if __name__ == "__main__":
    main()
