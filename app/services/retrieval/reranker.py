import os
import sys
import time
from pathlib import Path
from typing import List, Optional
import requests
import logfire

# Ensure project root is importable (for config)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from app.models import RetrievedChunk

RERANK_URL = "https://api.jina.ai/v1/rerank"
DEFAULT_MODEL = "jina-reranker-v2-base-multilingual"
MAX_RETRIES = 3


class JinaReranker:
    """
    Cross-Encoder Re-Ranker using Jina Reranker v2.
    Performs second-pass precision scoring on retrieved vector candidates.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = os.getenv("JINA_RERANK_MODEL", model)

        keys = (
            os.getenv("JINA_API_KEYS")
            or os.getenv("JINA_API_KEY")
            or getattr(settings, "JINA_API_KEY", "")
            or ""
        )
        self.api_keys = [k.strip() for k in keys.split(",") if k.strip()]

        if not self.api_keys:
            raise ValueError("Jina API key not found for JinaReranker.")

        self.current_key = 0
        self.session = requests.Session()

    def _get_key(self) -> str:
        return self.api_keys[self.current_key]

    def _rotate_key(self):
        if len(self.api_keys) > 1:
            self.current_key = (self.current_key + 1) % len(self.api_keys)
            print(f"  Rotating to Jina API key {self.current_key + 1}/{len(self.api_keys)}")

    @logfire.instrument("Pass 3: Jina Cross-Encoder Reranker", extract_args=False)
    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_n: int = 5,
    ) -> List[RetrievedChunk]:
        """
        Takes candidate chunks from Pass 1 vector search, computes cross-encoder
        relevance scores using Jina Reranker v2, and returns the top_n precision chunks.
        """
        if not chunks:
            return []

        top_n = min(top_n, len(chunks))

        print("\n" + "=" * 60)
        print("  STEP 3: Second-Pass Re-Ranking (Jina Reranker v2)")
        print("=" * 60)
        print(f"  • Model: {self.model}")
        print(f"  • Input: Re-ranking {len(chunks)} candidate chunks ──► selecting top {top_n}...\n")

        documents = [chunk.text for chunk in chunks]

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }

        reranked_results = None

        for attempt in range(MAX_RETRIES):
            headers = {
                "Authorization": f"Bearer {self._get_key()}",
                "Content-Type": "application/json",
            }

            try:
                response = self.session.post(
                    RERANK_URL,
                    headers=headers,
                    json=payload,
                    timeout=30,
                )

                if response.ok:
                    data = response.json()
                    reranked_results = data.get("results", [])
                    break

                if response.status_code in (401, 403, 429):
                    print(f"  ⚠️ Jina Reranker key error/rate-limit ({response.status_code}). Retrying...")
                    self._rotate_key()
                    time.sleep(1.5)
                    continue

                print(f"  ⚠️ Jina Reranker API error ({response.status_code}): {response.text}")
                time.sleep(2)

            except requests.exceptions.RequestException as e:
                print(f"  ⚠️ Network error calling Jina Reranker (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                time.sleep(2)

        if not reranked_results:
            print("  ⚠️ Reranking could not complete. Falling back to Pass 1 vector ranking.")
            return chunks[:top_n]

        reranked_chunks: List[RetrievedChunk] = []

        print(f"  ✓ Top {len(reranked_results)} Re-Ranked Chunks:\n")

        for new_rank, item in enumerate(reranked_results, start=1):
            orig_idx = item["index"]
            rerank_score = float(item["relevance_score"])
            orig_chunk = chunks[orig_idx]

            # Create new chunk instance preserving original vector score in metadata
            meta = orig_chunk.metadata.copy()
            meta["vector_score"] = orig_chunk.score
            meta["rerank_score"] = rerank_score

            chunk = RetrievedChunk(
                chunk_id=orig_chunk.chunk_id,
                score=rerank_score,
                text=orig_chunk.text,
                metadata=meta,
                source=orig_chunk.source,
            )
            reranked_chunks.append(chunk)

            page_info = f" (Page {meta.get('page') + 1})" if isinstance(meta.get("page"), int) else ""
            preview_snippet = chunk.text.replace("\n", " ").strip()[:140]

            print(
                f"  [{new_rank}] Relevance: {rerank_score:.4f} (Vector Score: {orig_chunk.score:.4f}) | "
                f"Source: {chunk.source}{page_info}"
            )
            print(f"      Snippet: \"{preview_snippet}...\"\n")

        return reranked_chunks

    def close(self):
        self.session.close()
