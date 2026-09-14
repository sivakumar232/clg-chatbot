# src/chat_bot/bm25_search.py
"""
BM25 keyword search over a pre-fetched candidate pool.

Why apply BM25 over the dense candidate pool instead of all chunks?
───────────────────────────────────────────────────────────────────
- No separate index to build or maintain.
- Works with the existing Qdrant collection — zero re-indexing needed.
- Dense search provides the semantic candidate pool; BM25 gives a
  second opinion using exact token matching on the same pool.
- This rescues results that dense search buries:
    • Academic subject codes  (CS3201, AIDS304)
    • Regulation numbers      (R20, R23, R24)
    • Faculty/HOD names       (exact string matches)
"""

import re
from typing import List
from langsmith import traceable

from rank_bm25 import BM25Okapi

from app.models import RetrievedChunk


# ── Tokenizer ──────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"\b[a-z0-9]+\b")


def tokenize(text: str) -> List[str]:
    """
    Lowercase alphanumeric tokenizer.

    Examples:
        "Machine Learning CS3201 R23" → ["machine", "learning", "cs3201", "r23"]
        "Dr. Mahesh (HOD-CSE)"        → ["dr", "mahesh", "hod", "cse"]

    Preserves exact subject codes and regulation numbers since they are
    purely alphanumeric. Strips punctuation that BM25 doesn't need.
    """
    return _TOKEN_RE.findall(text.lower())


# ── BM25 Ranking ───────────────────────────────────────────────────────────

@traceable(name="BM25 Keyword Ranking", run_type="retriever")
def bm25_rank(query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """
    Builds a BM25Okapi index in-memory from the given chunks and scores
    every chunk against the query. Returns chunks sorted by BM25 score
    descending (highest relevance first).

    Args:
        query  : The original user query string.
        chunks : Candidate chunks — typically the output of dense vector search.

    Returns:
        Same list of chunks, re-ordered by BM25 term-frequency relevance.

    BM25Okapi formula (Robertson & Zaragoza 2009):
        score(D, Q) = Σ_i IDF(qi) * (tf(qi,D) * (k1+1)) / (tf(qi,D) + k1*(1-b+b*|D|/avgdl))
    """
    if not chunks:
        return []

    corpus_tokens = [tokenize(chunk.text) for chunk in chunks]
    bm25          = BM25Okapi(corpus_tokens)
    query_tokens  = tokenize(query)
    scores        = bm25.get_scores(query_tokens)

    # Pair each chunk with its BM25 score and sort descending
    ranked_pairs = sorted(
        zip(scores, chunks),
        key=lambda pair: pair[0],
        reverse=True,
    )

    return [chunk for _, chunk in ranked_pairs]
