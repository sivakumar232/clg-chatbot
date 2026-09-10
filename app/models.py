# src/chat_bot/models.py
"""
Shared data models for the retrieval pipeline.
Kept in a dedicated file so every module imports from one place
instead of creating circular dependencies.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RetrievedChunk:
    """
    A single chunk returned from the retrieval pipeline.

    Attributes:
        chunk_id  : Unique identifier (Qdrant point UUID).
        score     : Relevance score (cosine similarity, BM25, or RRF — depends on stage).
        text      : Raw chunk text used for context assembly.
        metadata  : Full Qdrant payload (source_url, category, page, etc.).
        source    : Human-readable source label (file path or URL).
    """
    chunk_id: str
    score:    float
    text:     str
    metadata: Dict[str, Any]
    source:   str
