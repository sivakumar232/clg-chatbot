# src/chat_bot/retriever.py
"""
HybridRetriever — orchestrates the tri-brid retrieval pipeline.

Responsibilities:
    1. Dense vector search via Qdrant (semantic meaning).
    2. Calls bm25_search.bm25_rank()  for keyword/exact-match re-ranking.
    3. Calls rrf.rrf_fuse()           to merge both ranked lists.

Each concern lives in its own module:
    models.py      → RetrievedChunk data class
    bm25_search.py → BM25 tokenization + ranking
    rrf.py         → Reciprocal Rank Fusion
    retriever.py   → Qdrant dense search + pipeline orchestration  (this file)
"""

import sys
import time
from pathlib import Path
from typing import Any, List, Optional

# Ensure project root is importable (for config and Ingestion)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from app.services.retrieval.embedding import JinaEmbedder
from app.ingestion.pipeline import get_qdrant_client, get_production_collection_name

from app.models import RetrievedChunk
from .bm25_search import bm25_rank
from .rrf import rrf_fuse


class HybridRetriever:
    """
    Tri-brid retrieval pipeline:

        Dense Vector Search  (Jina v5 1024-d, Qdrant cosine)
                 +
        BM25 Keyword Search  (rank_bm25, in-memory over dense candidates)
                 ↓
        RRF Fusion           (Reciprocal Rank Fusion, k=60)
                 ↓
        Top-K hybrid candidates → passed to Jina Reranker v2
    """

    def __init__(self, collection_name: Optional[str] = None):
        self.collection_name = collection_name or get_production_collection_name()
        self.client   = get_qdrant_client()
        self.embedder = JinaEmbedder()

    # ──────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query:          str,
        top_k_dense:    int = 40,
        top_k_final:    int = 20,
        qdrant_filter:  Optional[Any] = None,
    ) -> List[RetrievedChunk]:
        """
        Run the full hybrid retrieval pipeline for a given query.

        Args:
            query          : User question.
            top_k_dense    : Candidate pool size fetched from Qdrant.
                             Larger pool → BM25 has more candidates to re-rank.
            top_k_final    : Chunks returned after RRF fusion (input to reranker).
            qdrant_filter  : Optional Qdrant Filter object for metadata-based
                             pre-filtering (e.g. {regulation: R23, department: CSE}).

        Returns:
            List[RetrievedChunk] sorted by RRF score descending.
        """
        t0 = time.time()
        self._print_header(query, top_k_dense, top_k_final, qdrant_filter)

        # Step 1 — Embed the query
        t_emb = time.time()
        print(f"  [STEP 1] Embedding query → {settings.JINA_EMB_MODEL} ...")
        query_vector = self.embedder.embed_query(query)
        print(f"  ✓ {len(query_vector)}-d vector embedded in {time.time() - t_emb:.2f}s\n")

        # Step 2 — Dense vector search (Qdrant)
        t_dense = time.time()
        filter_label = f" [filter: {qdrant_filter}]" if qdrant_filter else ""
        print(f"  [STEP 2] Dense search in Qdrant '{self.collection_name}' (top {top_k_dense}){filter_label} ...")
        dense_candidates = self._dense_search(query_vector, top_k_dense, qdrant_filter=qdrant_filter)
        print(f"  ✓ {len(dense_candidates)} candidates retrieved in {time.time() - t_dense:.2f}s")
        self._print_top(dense_candidates, label="Dense", n=3)

        if not dense_candidates:
            print("  ⚠️  No dense results found. Returning empty.")
            return []

        # Step 3 — BM25 re-ranking over the same candidates
        t_bm25 = time.time()
        print(f"\n  [STEP 3] BM25 keyword rank (rank_bm25) over {len(dense_candidates)} candidates ...")
        bm25_candidates = bm25_rank(query, dense_candidates)
        print(f"  ✓ BM25 scored in {time.time() - t_bm25:.4f}s")
        self._print_top(bm25_candidates, label="BM25", n=3)

        # Step 4 — RRF Fusion: merge dense + BM25 ranked lists
        t_rrf = time.time()
        print(f"\n  [STEP 4] RRF Fusion (k=60) — merging dense + BM25 ...")
        fused  = rrf_fuse([dense_candidates, bm25_candidates], k=60)
        final  = fused[:top_k_final]
        print(f"  ✓ Fused {len(fused)} chunks → kept top {len(final)} in {time.time() - t_rrf:.4f}s")

        self._print_summary(final, elapsed=time.time() - t0)
        return final

    # ──────────────────────────────────────────────────────────────────────
    #  Qdrant dense search
    # ──────────────────────────────────────────────────────────────────────

    def _dense_search(
        self,
        query_vector:  List[float],
        top_k:         int,
        qdrant_filter: Optional[Any] = None,
    ) -> List[RetrievedChunk]:
        """
        Queries Qdrant with the embedded query vector using cosine similarity.
        Returns the top_k nearest chunks with their full payloads.
        Optionally applies a Qdrant metadata filter for pre-filtering results.
        """
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            query_filter=qdrant_filter,
        )

        chunks: List[RetrievedChunk] = []
        for point in results.points:
            payload = point.payload or {}
            text    = payload.get("text") or payload.get("page_content") or ""
            source  = (
                payload.get("source_url")
                or payload.get("source")
                or "Unknown Source"
            )
            chunks.append(RetrievedChunk(
                chunk_id=str(point.id),
                score=float(point.score),
                text=text,
                metadata=payload,
                source=str(source),
            ))
        return chunks

    # ──────────────────────────────────────────────────────────────────────
    #  Print helpers
    # ──────────────────────────────────────────────────────────────────────

    def _print_header(self, query: str, top_k_dense: int, top_k_final: int, qdrant_filter: Optional[Any] = None) -> None:
        print("\n" + "=" * 65)
        print("  HYBRID RETRIEVAL — Dense + BM25 + RRF Fusion")
        print("=" * 65)
        print(f"  • Query        : \"{query}\"")
        print(f"  • Dense pool   : top {top_k_dense} from Qdrant")
        print(f"  • Final output : top {top_k_final} after RRF fusion")
        if qdrant_filter:
            print(f"  • Qdrant Filter: {qdrant_filter}")
        print()

    def _print_top(self, chunks: List[RetrievedChunk], label: str, n: int = 3) -> None:
        """Prints a compact preview of the top-n chunks in a ranked list."""
        print(f"\n  Top {min(n, len(chunks))} by {label}:")
        for rank, chunk in enumerate(chunks[:n], start=1):
            page = (
                f" (Page {chunk.metadata.get('page') + 1})"
                if isinstance(chunk.metadata.get("page"), int)
                else ""
            )
            snippet = chunk.text.replace("\n", " ").strip()[:140]
            print(f"    [{rank}] Score: {chunk.score:.4f} | {chunk.source}{page}")
            print(f"        \"{snippet}...\"\n")

    def _print_summary(self, final: List[RetrievedChunk], elapsed: float) -> None:
        """Prints the final ranked table after RRF fusion."""
        print(f"\n  {'─'*62}")
        print(f"  HYBRID RETRIEVAL COMPLETE in {elapsed:.2f}s")
        print(f"  {'─'*62}")
        print(f"  {'Rank':<5} {'RRF Score':<12} {'Dense Score':<14} Source")
        print(f"  {'─'*62}")
        for rank, chunk in enumerate(final, start=1):
            dense_score = chunk.metadata.get("dense_score", 0.0)
            src         = chunk.source[-52:] if len(chunk.source) > 52 else chunk.source
            print(f"  [{rank:<3}] {chunk.score:<12.6f} {dense_score:<14.4f} {src}")
        print(f"  {'─'*62}\n")

    # ──────────────────────────────────────────────────────────────────────
    #  Cleanup
    # ──────────────────────────────────────────────────────────────────────

    def close(self):
        """Close persistent Jina embedder HTTP session."""
        self.embedder.close()


# Backward-compatibility alias so rag_chain.py and reranker.py still work
QdrantRetriever = HybridRetriever
