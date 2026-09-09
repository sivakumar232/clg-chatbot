
import time
from typing import Any, Dict, List, Optional
from .retriever import QdrantRetriever, RetrievedChunk
from .generator import LLMGenerator


class RAGPipeline:
    """
    End-to-end RAG Pipeline connecting Dense Vector Retrieval
    and LLM Generation with Fallback.
    """

    def __init__(self, collection_name: Optional[str] = None):
        self.retriever = QdrantRetriever(collection_name=collection_name)
        self.generator = LLMGenerator()

    def run(self, query: str, top_k: int = 20) -> Dict[str, Any]:
        """
        Executes the full RAG workflow with detailed step-by-step terminal prints.
        """
        t0 = time.time()
        print("\n" + "█" * 65)
        print(f"  RAG PIPELINE EXECUTION: \"{query}\"")
        print("█" * 65)

        # Step 1 & 2: Embed Query & Retrieve Chunks
        chunks = self.retriever.retrieve(query=query, top_k=top_k)

        if not chunks:
            print("  ⚠️ No matching chunks found in the vector database.")
            return {
                "query": query,
                "answer": "No relevant documentation found in the knowledge base.",
                "chunks": [],
                "provider": "None",
                "sources": [],
                "elapsed_seconds": time.time() - t0,
            }

        # Step 3 & 4: Assemble Context & Generate Answer
        answer, provider = self.generator.generate(query=query, chunks=chunks)

        # Step 5: Format and Display Result
        elapsed = time.time() - t0
        sources = list(dict.fromkeys(chunk.source for chunk in chunks))

        print("=" * 60)
        print("  STEP 5: Grounded Answer & Source Attribution")
        print("=" * 60)
        print(f"  • Generated in: {elapsed:.2f}s | Provider: {provider}\n")
        print("──────────────────────── ANSWER ────────────────────────\n")
        print(answer.strip())
        print("\n────────────────────────────────────────────────────────")
        print("  Referenced Sources:")
        for s in sources:
            print(f"    • {s}")
        print("█" * 65 + "\n")

        return {
            "query": query,
            "answer": answer,
            "chunks": chunks,
            "provider": provider,
            "sources": sources,
            "elapsed_seconds": elapsed,
        }

    def close(self):
        """Cleanup network connections."""
        self.retriever.close()
