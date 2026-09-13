"""
agent/state.py
──────────────
Defines the AgentState TypedDict and Enums for the Agentic RAG graph.

Every field is tracked across the graph lifecycle:
Cache -> Planner -> Executor -> Reranker -> Validator -> Reformulator
-> Generator -> Guard -> Responder -> CacheWrite
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

from app.models import RetrievedChunk


class RouteType(str, Enum):
    """Set by planner node."""
    DIRECT          = "direct"          # Conversational / Chit-chat / Out-of-scope (no retrieval)
    NEEDS_RETRIEVAL = "needs_retrieval" # Academic / Syllabus / Faculty / Specific queries


class QueryType(str, Enum):
    """Set by planner node."""
    SINGLE    = "single_query"    # Single direct query
    SUB       = "sub_query"       # 2+ sub-queries for same topic/regulations
    MULTI_HOP = "multi_hop_query" # Parallel queries across multiple departments/entities


class EvidenceStatus(str, Enum):
    """Set by validator node."""
    SUFFICIENT             = "sufficient"
    INSUFFICIENT_RETRY     = "insufficient_retry"
    INSUFFICIENT_EXHAUSTED = "insufficient_exhausted"


class GuardStatus(str, Enum):
    """Set by guard node."""
    GROUNDED             = "grounded"
    UNGROUNDED_RETRY     = "ungrounded_retry"
    UNGROUNDED_EXHAUSTED = "ungrounded_exhausted"


class AgentState(TypedDict, total=False):
    """
    Mutable state object passed between all LangGraph nodes.
    """

    # ── INPUT & CACHE ────────────────────────────────────────────────────────
    query: str
    chat_history: List[Dict[str, str]]
    cache_hit: bool
    cached_response: Optional[Dict[str, Any]]

    # ── PLANNER ──────────────────────────────────────────────────────────────
    rewritten_query: str
    route: RouteType
    query_type: QueryType
    intent: Dict[str, Any]
    sub_queries: List[Dict[str, Any]]

    # ── RETRIEVAL & FUSION ───────────────────────────────────────────────────
    candidate_chunks: List[RetrievedChunk]
    accumulated_chunks: List[RetrievedChunk]  # Preserved across retries
    reranked_chunks: List[RetrievedChunk]

    # ── VALIDATION & REFORMULATION ───────────────────────────────────────────
    evidence_status: EvidenceStatus
    retrieval_retry_count: int
    max_retrieval_retries: int
    reformulated_query: Optional[str]
    degraded_reason: Optional[str]
    missing_sub_queries: List[str]       # Sub-queries that lacked supporting evidence
    contradictions: List[str]            # Non-blocking detected factual contradictions
    pruned_chunks: List[RetrievedChunk]  # Chunks after redundancy / diversity pruning


    # ── GENERATION & GUARD ───────────────────────────────────────────────────
    draft_answer: str
    guard_status: GuardStatus
    guard_feedback: Optional[str]
    guard_retry_count: int
    max_guard_retries: int

    # ── RESPONDER & OUTPUT ───────────────────────────────────────────────────
    answer: str
    sources: List[str]
    provider: str
    degraded: bool
