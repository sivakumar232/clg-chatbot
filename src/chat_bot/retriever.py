import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Ensure project root is importable (for config and Ingestion)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from Ingestion.embedder import JinaEmbedder
from Ingestion.pipeline import get_qdrant_client, get_production_collection_name


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any]
    source: str


class QdrantRetriever:
    """
    Handles dense vector retrieval using Jina Embeddings (1024-d)
    and Qdrant vector database.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
    ):
        self.collection_name = collection_name or get_production_collection_name()
        self.client = get_qdrant_client()
        self.embedder = JinaEmbedder()

    def retrieve(self, query: str, top_k: int = 20) -> List[RetrievedChunk]:
        """
        Embeds the query and fetches the top_k most similar chunks from Qdrant.
        Prints detailed information for each step.
        """
        print("\n" + "=" * 60)
        print("  STEP 1: Embed Query (Jina AI)")
        print("=" * 60)
        print(f"  • Query: \"{query}\"")
        print(f"  • Model: {settings.JINA_EMB_MODEL} (task='retrieval.query')")

        query_vector = self.embedder.embed_query(query)
        print(f"  ✓ Successfully generated {len(query_vector)}-d dense vector.\n")

        print("=" * 60)
        print(f"  STEP 2: Dense Vector Search (Qdrant: {self.collection_name})")
        print("=" * 60)
        print(f"  • Searching top {top_k} nearest chunks by Cosine similarity...\n")

        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        retrieved: List[RetrievedChunk] = []

        print(f"  ✓ Retrieved {len(search_results.points)} chunks:\n")

        for rank, point in enumerate(search_results.points, start=1):
            payload = point.payload or {}
            text = payload.get("text") or payload.get("page_content") or ""
            source = (
                payload.get("source_url")
                or payload.get("source")
                or "Unknown Source"
            )

            chunk = RetrievedChunk(
                chunk_id=str(point.id),
                score=float(point.score),
                text=text,
                metadata=payload,
                source=str(source),
            )
            retrieved.append(chunk)

            # Terminal print for each chunk
            page_info = f" (Page {payload.get('page') + 1})" if isinstance(payload.get("page"), int) else ""
            preview_snippet = text.replace("\n", " ").strip()[:160]

            print(f"  [{rank}] Score: {chunk.score:.4f} | Source: {chunk.source}{page_info}")
            print(f"      Snippet: \"{preview_snippet}...\"\n")

        return retrieved

    def close(self):
        """Closes the embedder session."""
        self.embedder.close()
