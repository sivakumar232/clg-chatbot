import time
from typing import Any, Dict, List, Optional
from .retriever import QdrantRetriever, RetrievedChunk
from .reranker import JinaReranker
from .generator import LLMGenerator


class RAGPipeline:
    """
    End-to-end Two-Pass RAG Pipeline:
    - Pass 1: Broad Dense Vector Search via Qdrant (top_k candidates)
    - Pass 2: Precision Cross-Encoder Re-Ranking via Jina Reranker v2 (top_n precision chunks)
    - Generation: LLM Generation (Groq / Gemini fallback)
    """

    def __init__(self, collection_name: Optional[str] = None):
        self.retriever = QdrantRetriever(collection_name=collection_name)
        self.reranker = JinaReranker()
        self.generator = LLMGenerator()

    def run(
        self,
        query: str,
        top_k: int = 20,
        top_n: int = 5,
    ) -> Dict[str, Any]:
        """
        Executes the two-pass RAG workflow with detailed step-by-step terminal prints.
        """
        t0 = time.time()
        print("\n" + "█" * 65)
        print(f"  TWO-PASS RAG PIPELINE EXECUTION: \"{query}\"")
        print("█" * 65)

        # Pass 1 (Step 1 & 2): Embed Query & Retrieve Candidate Chunks from Qdrant
        candidate_chunks = self.retriever.retrieve(query=query, top_k=top_k)

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

        # Pass 2 (Step 3): Cross-Encoder Re-Ranking using Jina Reranker v2
        t_rerank = time.time()
        precision_chunks = self.reranker.rerank(
            query=query,
            chunks=candidate_chunks,
            top_n=top_n,
        )
        print(f"  ✓ Re-ranking finished in {time.time() - t_rerank:.2f}s.")

        # Step 4 & 5: Assemble Context & Generate Answer
        answer, provider = self.generator.generate(query=query, chunks=precision_chunks)

        # Step 6: Format and Display Result
        elapsed = time.time() - t0
        sources = list(dict.fromkeys(chunk.source for chunk in precision_chunks))

        print("=" * 60)
        print("  STEP 6: Grounded Answer & Source Attribution")
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
            "query": query,
            "answer": answer,
            "chunks": precision_chunks,
            "candidate_chunks": candidate_chunks,
            "provider": provider,
            "sources": sources,
            "elapsed_seconds": elapsed,
        }

    def close(self):
        """Cleanup network connections."""
        self.retriever.close()
        self.reranker.close()
