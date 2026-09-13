"""
agent/nodes/reranker.py
───────────────────────
Cross-Encoder Reranker Node for Agentic RAG.

Responsibilities:
1. Takes the fused candidate chunks (up to 25) from parallel_executor.
2. Evaluates query-document pairs using Jina Reranker v2 (deep cross-attention).
3. Selects the top_n=5 precision chunks with normalized relevance scores (0.0 to 1.0).
4. Emits reranked_chunks for the evidence_validator gate.
"""

import time
from typing import List
import logfire

from agent.state import AgentState
from app.models import RetrievedChunk
from app.services.retrieval.reranker import JinaReranker

# Cached reranker instance to reuse HTTP sessions & key rotation state
_RERANKER_INSTANCE: JinaReranker | None = None


def _get_reranker() -> JinaReranker:
    global _RERANKER_INSTANCE
    if _RERANKER_INSTANCE is None:
        _RERANKER_INSTANCE = JinaReranker()
    return _RERANKER_INSTANCE


def _compute_top_n(state: AgentState) -> int:
    """
    Dynamically compute top_n rerank chunks based on query scope.
    Broader queries need more evidence; narrow queries stay efficient.
    """
    intent = state.get("intent", {})
    query_type = str(state.get("query_type", "")).lower()

    if intent.get("is_aggregate"):
        return 12  # "list all HODs / all branches" needs many chunks
    if "multi_hop" in query_type:
        return 10
    if "sub" in query_type:
        return 7
    return 5  # single focused query — keep it precise


def reranker_node(state: AgentState) -> AgentState:
    """
    LangGraph Reranker node:
    Reads:  candidate_chunks, rewritten_query, query, query_type, intent
    Writes: reranked_chunks
    """
    candidate_chunks = state.get("candidate_chunks", [])
    scoring_query = state.get("rewritten_query") or state.get("query", "")

    if not candidate_chunks:
        return {"reranked_chunks": []}

    reranker = _get_reranker()
    top_n = _compute_top_n(state)

    with logfire.span(
        "Reranker Node",
        input_chunks=len(candidate_chunks),
        query=scoring_query,
        top_n=top_n,
    ) as span:
        t0 = time.time()

        # Score top candidates down to dynamic top_n precision chunks
        reranked = reranker.rerank(
            query=scoring_query,
            chunks=candidate_chunks,
            top_n=top_n,
        )

        elapsed = time.time() - t0
        top_score = reranked[0].score if reranked else 0.0

        span.set_attribute("reranked_count", len(reranked))
        span.set_attribute("top_score", round(top_score, 4))
        span.set_attribute("latency_seconds", round(elapsed, 3))

        print("=" * 60)
        print(f"  RERANKER COMPLETE: {len(candidate_chunks)} -> Top {len(reranked)} (top_n={top_n}) in {elapsed:.2f}s")
        print("=" * 60)
        for i, chunk in enumerate(reranked, 1):
            snippet = chunk.text.replace("\n", " ").strip()[:100]
            print(f"  [{i}] Score: {chunk.score:.4f} | {chunk.source}")
            print(f"      \"{snippet}...\"")
        print("=" * 60 + "\n")

        return {
            "reranked_chunks": reranked,
        }

