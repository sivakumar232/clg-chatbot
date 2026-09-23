"""
agent/nodes/executor.py
───────────────────────
Parallel Retrieval Executor Node for Agentic RAG.

Responsibilities:
1. Reads sub_queries from planner (or reformulated_query from reformulator).
2. Executes Hybrid Retrieval (Dense + BM25 + RRF) concurrently via ThreadPoolExecutor.
3. Performs Cross-Query RRF Fusion across all sub-queries.
4. Unions current pass results with accumulated_chunks (Pass 1 ∪ Pass 2 on retries).
5. Emits top candidate chunks for downstream Cross-Encoder reranking.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
import threading
import time
from typing import Any, Dict, List
from langsmith import traceable

from agent.state import AgentState
from app.models import RetrievedChunk
from app.services.retrieval.retriever import HybridRetriever
from app.services.retrieval.rrf import rrf_fuse

# Cached retriever instance to reuse JinaEmbedder and Qdrant connections
_RETRIEVER_INSTANCE: HybridRetriever | None = None
_RETRIEVER_LOCK = threading.Lock()


def _get_retriever() -> HybridRetriever:
    """Thread-safe singleton getter with double-checked locking."""
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None:
        with _RETRIEVER_LOCK:
            if _RETRIEVER_INSTANCE is None:   # second check inside lock
                _RETRIEVER_INSTANCE = HybridRetriever()
    return _RETRIEVER_INSTANCE


@traceable(name="Single Sub-Query Retrieval", run_type="retriever")
def _execute_single_subquery(sub_q: Dict[str, Any], retriever: HybridRetriever) -> List[RetrievedChunk]:
    """
    Executes a single hybrid retrieval pass for one sub-query.
    Fetches top 40 dense candidates, applies BM25, and fuses to top 20.
    Passes any metadata_filter from the planner to Qdrant for filtered search.
    """
    query_text = sub_q.get("query", "").strip()
    if not query_text:
        return []

    # Build Qdrant filter from planner metadata_filter if present
    qdrant_filter = None
    metadata_filter = sub_q.get("metadata_filter")
    if metadata_filter and isinstance(metadata_filter, dict):
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            conditions = []
            for k, v in metadata_filter.items():
                if v is None:
                    continue
                # Map department to category in Qdrant payloads if needed
                field_key = "category" if str(k).lower() == "department" else str(k)
                val = str(v).lower() if field_key == "category" else v
                conditions.append(FieldCondition(key=field_key, match=MatchValue(value=val)))
            if conditions:
                qdrant_filter = Filter(must=conditions)
        except Exception:
            qdrant_filter = None  # Silently skip filter if qdrant_client models unavailable

    try:
        # Retrieve top 40 dense candidates, fuse to top 20 hybrid chunks
        chunks = retriever.retrieve(
            query=query_text,
            top_k_dense=40,
            top_k_final=20,
            qdrant_filter=qdrant_filter,
        )
        return chunks
    except Exception as e:
        if qdrant_filter is not None:
            print(f"Filtered retrieval failed for '{query_text}' ({e}). Retrying without filter...")
            try:
                return retriever.retrieve(
                    query=query_text,
                    top_k_dense=40,
                    top_k_final=20,
                    qdrant_filter=None,
                )
            except Exception as e2:
                print(f"Fallback retrieval failed for '{query_text}': {e2}")
                return []
        print(f"Failed sub-query retrieval for '{query_text}': {e}")
        return []


def _merge_cumulative_chunks(
    new_chunks: List[RetrievedChunk],
    accumulated_chunks: List[RetrievedChunk],
    max_keep: int = 25,
) -> List[RetrievedChunk]:
    """
    Unions new chunks with previously accumulated chunks (Pass 1 ∪ Pass 2).
    Deduplicates by chunk_id, preserving the higher score.
    """
    chunk_map: Dict[str, RetrievedChunk] = {}

    # Add previously accumulated chunks first
    for c in accumulated_chunks:
        chunk_map[c.chunk_id] = c

    # Add or update with new chunks (keep max score)
    for c in new_chunks:
        if c.chunk_id in chunk_map:
            existing = chunk_map[c.chunk_id]
            if c.score > existing.score:
                chunk_map[c.chunk_id] = c
        else:
            chunk_map[c.chunk_id] = c

    # Sort descending by score
    merged = sorted(chunk_map.values(), key=lambda x: x.score, reverse=True)
    return merged[:max_keep]


@traceable(name="Parallel Executor Node", run_type="chain")
def executor_node(state: AgentState) -> AgentState:
    """
    LangGraph Executor node:
    Reads:  sub_queries, reformulated_query, accumulated_chunks
    Writes: candidate_chunks, accumulated_chunks
    """
    retriever = _get_retriever()

    # ── 1. Determine Queries to Run ──────────────────────────────────────────
    queries_to_run: List[Dict[str, Any]] = []

    # If on retry path, prioritize reformulated_query
    reformulated = state.get("reformulated_query")
    if reformulated:
        queries_to_run = [{"query": reformulated, "metadata_filter": None}]
    else:
        queries_to_run = state.get("sub_queries", [])

    # Fallback to rewritten_query or raw query if empty
    if not queries_to_run:
        fallback_q = state.get("rewritten_query") or state.get("query", "")
        queries_to_run = [{"query": fallback_q, "metadata_filter": None}]

    # Cap to max 4 sub-queries for latency control
    queries_to_run = queries_to_run[:4]

    t0 = time.time()
    print("=" * 60)
    print(f"  PARALLEL EXECUTOR: Running {len(queries_to_run)} sub-queries")
    print("=" * 60)
    for i, q in enumerate(queries_to_run, 1):
        print(f"  • Sub-query [{i}]: \"{q.get('query')}\"")
    print("=" * 60 + "\n")

    # ── 2. Concurrent Retrieval via ThreadPoolExecutor ───────────────────
    subquery_results: List[List[RetrievedChunk]] = []

    if len(queries_to_run) == 1:
        # Single query — run in current thread
        res = _execute_single_subquery(queries_to_run[0], retriever)
        if res:
            subquery_results.append(res)
    else:
        # Multiple queries — run in parallel
        max_workers = min(4, len(queries_to_run))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_execute_single_subquery, sq, retriever): sq.get("query")
                for sq in queries_to_run
            }
            try:
                for fut in as_completed(futures, timeout=15):  # hard cap: 15s per sub-query batch
                    q_text = futures[fut]
                    try:
                        res = fut.result()
                        if res:
                            subquery_results.append(res)
                    except Exception as e:
                        print(f"Thread retrieval error for '{q_text}': {e}")
            except FuturesTimeoutError:
                print("  ⚠️  ThreadPoolExecutor sub-query batch timed out after 15s. Continuing with partial results.")

    # ── 3. Cross-Query RRF Fusion ────────────────────────────────────────
    if not subquery_results:
        current_pass_chunks: List[RetrievedChunk] = []
    elif len(subquery_results) == 1:
        current_pass_chunks = subquery_results[0]
    else:
        # Fuse multiple sub-query result lists using RRF
        current_pass_chunks = rrf_fuse(subquery_results, k=60)

    # ── 4. Cumulative Union with Previous Passes ─────────────────────────
    prev_accumulated = state.get("accumulated_chunks", [])
    final_candidates = _merge_cumulative_chunks(
        new_chunks=current_pass_chunks,
        accumulated_chunks=prev_accumulated,
        max_keep=25,
    )

    elapsed = time.time() - t0

    print(f"  ✓ Parallel Executor complete in {elapsed:.2f}s.")
    print(f"  • Total unique candidates: {len(final_candidates)} (fused & deduplicated)\n")

    return {
        "candidate_chunks":   final_candidates,
        "accumulated_chunks": final_candidates,
    }
