# src/chat_bot/rrf.py
"""
Reciprocal Rank Fusion (RRF) — merges multiple ranked lists into one.

Reference: Cormack, Clarke & Buettcher (2009)
    "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"

Formula:
    RRF_score(d) = Σ_m   1 / (k + rank_m(d))

Where:
    d       → a document/chunk
    m       → one ranking system (e.g. dense search, BM25 search)
    rank_m  → rank position of d in system m  (1-indexed, lower = better)
    k       → smoothing constant (default 60, standard from the original paper)

The constant k=60:
    • Reduces the impact of very high rankings, making the fusion more stable.
    • A chunk at rank 1 in both systems scores: 1/61 + 1/61 ≈ 0.0328
    • A chunk at rank 1 in one and rank 30 in the other: 1/61 + 1/90 ≈ 0.0274
    • Lower k → amplifies top-rank advantage. Higher k → flattens differences.
"""

from typing import Dict, List

from .models import RetrievedChunk


def rrf_fuse(
    ranked_lists: List[List[RetrievedChunk]],
    k:            int = 60,
) -> List[RetrievedChunk]:
    """
    Fuses any number of ranked lists of RetrievedChunks using RRF.

    Accepts a variable number of ranked lists so future search methods
    (e.g. metadata search, MMR) can be added without changing this function.

    Args:
        ranked_lists : List of ranked chunk lists. Each list is one ranking
                       system (e.g. [dense_ranked, bm25_ranked]).
        k            : RRF smoothing constant. Default 60.

    Returns:
        Deduplicated, RRF-fused list sorted by RRF score descending.
        Each chunk's .score is its final RRF score.
        The original dense cosine score is preserved in chunk.metadata["dense_score"].

    Example:
        fused = rrf_fuse([dense_results, bm25_results], k=60)
    """
    rrf_scores: Dict[str, float]         = {}
    chunk_map:  Dict[str, RetrievedChunk] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            cid = chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            # First list to mention the chunk wins the slot in chunk_map.
            # This means the dense search's metadata is preserved (includes cosine score).
            chunk_map.setdefault(cid, chunk)

    # Sort all chunk IDs by their cumulative RRF score
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    # Rebuild the final list, attaching the RRF score as the canonical .score
    fused: List[RetrievedChunk] = []
    for cid in sorted_ids:
        original = chunk_map[cid]
        fused.append(RetrievedChunk(
            chunk_id=original.chunk_id,
            score=rrf_scores[cid],                              # RRF score
            text=original.text,
            metadata={**original.metadata, "dense_score": original.score},  # preserve cosine
            source=original.source,
        ))

    return fused
