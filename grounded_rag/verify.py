"""Deterministic grounding check. No LLM call, on purpose. Rules:
  * every citation [kb-NN] must point at a chunk that was actually retrieved -> else "fabricated"
  * every identifier (E-4021, v2.4) and number (72, 5,000, 02:00) in the answer must appear
    verbatim in the retrieved text -> else "ungrounded"
  * the answer must cite at least one retrieved chunk -> else "ungrounded"
Matching is boundary-aware: "E-402" does not match inside "E-4021", "15" not inside "150".
With adjacent identifiers in the corpus, substring matching would pass exactly the wrong answers.
"""
import re

from .state import Chunk, Verdict

CITATION = re.compile(r"\bkb-\d+\b")
IDENTIFIER = re.compile(r"\b[A-Z]{1,5}-\d{2,6}\b|\bv\d+(?:\.\d+)+\b")
NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,:]\d+)*")
LIST_MARKER = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*•])\s+")


def appears_verbatim(token: str, text: str) -> bool:
    # a bare number may sit inside a code ("4021" in "E-4021"); an identifier may not ("E-402")
    before = r"(?<![\w.])" if token[0].isdigit() else r"(?<![\w.-])"
    return re.search(rf"{before}{re.escape(token)}(?![\w-]|[.,:]\d)", text) is not None


def verify_answer(answer: str, chunks: list[Chunk]) -> tuple[Verdict, list[str]]:
    context = "\n".join(chunk["text"] for chunk in chunks)
    retrieved_ids = {chunk["id"] for chunk in chunks}

    citations = list(dict.fromkeys(CITATION.findall(answer)))
    fabricated = [f"cites [{c}], which was not retrieved" for c in citations if c not in retrieved_ids]

    body = LIST_MARKER.sub("", CITATION.sub("", answer))
    identifiers = list(dict.fromkeys(IDENTIFIER.findall(body)))
    numbers = list(dict.fromkeys(NUMBER.findall(IDENTIFIER.sub("", body))))
    unsupported = [f"identifier {t} not in retrieved context" for t in identifiers
                   if not appears_verbatim(t, context)]
    unsupported += [f"number {n} not in retrieved context" for n in numbers
                    if not appears_verbatim(n, context)]
    if not citations:
        unsupported.append("answer cites no retrieved chunk")

    if fabricated:
        return "fabricated", fabricated + unsupported
    return ("ungrounded" if unsupported else "grounded"), unsupported
