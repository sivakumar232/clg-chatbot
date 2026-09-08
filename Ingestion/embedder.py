# Ingestion/embedder.py

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

try:
    from config import settings
    MODEL = getattr(settings, "JINA_EMB_MODEL", "jina-embeddings-v5-text-small")
    EMBEDDING_DIM = getattr(settings, "EMBEDDING_DIM", 1024)
except ImportError:
    MODEL = "jina-embeddings-v5-text-small"
    EMBEDDING_DIM = 1024

JINA_URL = "https://api.jina.ai/v1/embeddings"

BATCH_SIZE = 32

MAX_RETRIES = 5


class JinaEmbedder:

    def __init__(self):
        keys = (
            os.getenv("JINA_API_KEYS")
            or os.getenv("JINA_API_KEY")
            or ""
        )

        self.api_keys = [
            key.strip()
            for key in keys.split(",")
            if key.strip()
        ]

        if not self.api_keys:
            raise ValueError(
                "Neither JINA_API_KEY nor JINA_API_KEYS is configured."
            )

        self.current_key = 0

        self.session = requests.Session()

    # --------------------------------------------------
    # API key
    # --------------------------------------------------

    def _get_key(self):
        return self.api_keys[self.current_key]

    def _rotate_key(self):
        if len(self.api_keys) == 1:
            return

        self.current_key = (
            self.current_key + 1
        ) % len(self.api_keys)

        print(
            f"Rotating to Jina API key "
            f"{self.current_key + 1}/{len(self.api_keys)}"
        )

    # --------------------------------------------------
    # Single API request
    # --------------------------------------------------

    def _request(
        self,
        texts: list[str],
        task: str,
    ) -> list[list[float]]:

        for attempt in range(MAX_RETRIES):

            headers = {
                "Authorization": f"Bearer {self._get_key()}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": MODEL,
                "task": task,
                "input": texts,
                "dimensions": EMBEDDING_DIM,
                "normalized": True,
            }

            try:
                response = self.session.post(
                    JINA_URL,
                    headers=headers,
                    json=payload,
                    timeout=120,
                )
            except requests.exceptions.RequestException as exc:
                wait = min(30, 2 ** (attempt + 1))
                print(
                    f"Jina network error (attempt {attempt + 1}/{MAX_RETRIES}): {exc}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
                continue

            # ------------------------------------------
            # Success
            # ------------------------------------------

            if response.ok:

                data = response.json()["data"]

                # Jina normally returns indexes.
                # Sort so output matches input order.
                data.sort(key=lambda x: x["index"])

                vectors = [
                    item["embedding"]
                    for item in data
                ]

                if len(vectors) != len(texts):
                    raise RuntimeError(
                        "Jina returned an incorrect number "
                        "of embeddings."
                    )

                return vectors

            # ------------------------------------------
            # Invalid / expired API key
            # ------------------------------------------

            if response.status_code in (401, 403):

                print(
                    f"Jina key {self.current_key + 1} "
                    f"was rejected."
                )

                self._rotate_key()
                continue

            # ------------------------------------------
            # Rate limit / quota
            # ------------------------------------------

            if response.status_code == 429:

                print(
                    f"Jina rate limit/quota hit "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})"
                )

                self._rotate_key()

                retry_after = response.headers.get(
                    "Retry-After"
                )

                if retry_after:
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        wait = 5
                else:
                    wait = 2 ** attempt

                time.sleep(wait)
                continue

            # ------------------------------------------
            # Temporary server error
            # ------------------------------------------

            if response.status_code >= 500:

                wait = 2 ** attempt

                print(
                    f"Jina server error "
                    f"{response.status_code}. "
                    f"Retrying in {wait}s..."
                )

                time.sleep(wait)
                continue

            # ------------------------------------------
            # Permanent error
            # ------------------------------------------

            raise RuntimeError(
                f"Jina API error "
                f"{response.status_code}: "
                f"{response.text}"
            )

        raise RuntimeError(
            "Jina embedding failed after retries. "
            "Pipeline stopped safely."
        )

    # --------------------------------------------------
    # Documents
    # --------------------------------------------------

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        all_vectors = []

        for start in range(
            0,
            len(texts),
            BATCH_SIZE,
        ):

            batch = texts[
                start:start + BATCH_SIZE
            ]

            print(
                f"Embedding "
                f"{start + 1}-{start + len(batch)} "
                f"/ {len(texts)}"
            )

            vectors = self._request(
                batch,
                task="retrieval.passage",
            )

            all_vectors.extend(vectors)

        return all_vectors

    # --------------------------------------------------
    # Query
    # --------------------------------------------------

    def embed_query(
        self,
        query: str,
    ) -> list[float]:

        return self._request(
            [query],
            task="retrieval.query",
        )[0]

    def close(self):
        self.session.close()