import time
from typing import Any, Dict, List, Optional

from app.services.retrieval.retriever import HybridRetriever
from app.services.retrieval.reranker import JinaReranker
from app.services.generation.generator import LLMGenerator
from app.models import RetrievedChunk
from langsmith import traceable


class RAGPipeline:
    """
    Three-Pass Hybrid RAG Pipeline:

    Pass 1 — Dense Vector Search (Qdrant + Jina v5 1024-d)
        Fetches a large candidate pool (top_k_dense=40) by semantic cosine similarity.

    Pass 2 — BM25 Keyword Fusion (rank_bm25 + Reciprocal Rank Fusion)
        Re-orders the dense candidate pool using BM25 term-frequency scoring.
        Fuses both ranked lists via RRF → top_k_final=20 hybrid chunks.
        This rescues exact-match results that dense search may bury:
        course codes (CS3201), regulation numbers (R23), faculty names.

    Pass 3 — Cross-Encoder Re-Ranking (Jina Reranker v2)
        Precision-scores the top 20 hybrid chunks with deep joint attention.
        Returns top_n=5 precision chunks for the final LLM generation.

    Generation — LLM (Groq primary / Gemini fallback)
    """

    def __init__(self, collection_name: Optional[str] = None):
        self.retriever = HybridRetriever(collection_name=collection_name)
        self.reranker  = JinaReranker()
        self.generator = LLMGenerator()

    @traceable(name="RAG Pipeline Execution", run_type="chain")
    def run(
        self,
        query:        str,
        top_k:        int = 20,   # kept for CLI backward-compat; maps to top_k_final
        top_n:        int = 5,
        top_k_dense:  int = 40,   # dense candidate pool size
    ) -> Dict[str, Any]:
        """
        Executes the three-pass hybrid RAG workflow.

        Args:
            query:       User question.
            top_k:       Final chunks after RRF (passed to reranker). Default 20.
            top_n:       Precision chunks kept after reranking (for generation). Default 5.
            top_k_dense: Dense candidate pool fetched from Qdrant. Default 40.
        """
        t0 = time.time()
        print("\n" + "█" * 65)
        print(f"  HYBRID RAG PIPELINE: \"{query}\"")
        print(f"  Dense pool: {top_k_dense} → RRF fusion → {top_k} → Rerank → {top_n}")
        print("█" * 65)

        # ── Pass 1+2: Dense Search + BM25 + RRF Fusion ──────────
        candidate_chunks = self.retriever.retrieve(
            query=query,
            top_k_dense=top_k_dense,
            top_k_final=top_k,
        )

        if not candidate_chunks:
            print("  ⚠️ No matching chunks found in the vector database.")
            return {
                "query": query,
                "answer": "No relevant documentation found in the knowledge base.",
                "chunks": [],
                "provider": "None",
                "sources": [],
                "elapsed_seconds": time.time() - t0,
            }

        # ── Pass 3: Cross-Encoder Re-Ranking ────────────────────
        t_rerank = time.time()
        precision_chunks = self.reranker.rerank(
            query=query,
            chunks=candidate_chunks,
            top_n=top_n,
        )
        print(f"  ✓ Re-ranking finished in {time.time() - t_rerank:.2f}s.")

        # ── Generation ───────────────────────────────────────────
        answer, provider = self.generator.generate(query=query, chunks=precision_chunks)

        elapsed = time.time() - t0
        sources = list(dict.fromkeys(chunk.source for chunk in precision_chunks))

        print("=" * 60)
        print("  FINAL: Grounded Answer & Source Attribution")
        print("=" * 60)
        print(f"  • Generated in: {elapsed:.2f}s | Provider: {provider}\n")
        print("──────────────────────── ANSWER ────────────────────────\n")
        print(answer.strip())
        print("\n────────────────────────────────────────────────────────")
        print("  Referenced Sources (After Re-Ranking):")
        for s in sources:
            print(f"    • {s}")
        print("█" * 65 + "\n")

        return {
            "query":            query,
            "answer":           answer,
            "chunks":           precision_chunks,
            "candidate_chunks": candidate_chunks,
            "provider":         provider,
            "sources":          sources,
            "elapsed_seconds":  elapsed,
        }

    def close(self):
        """Cleanup network connections."""
        self.retriever.close()
        self.reranker.close()
