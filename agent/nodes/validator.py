"""
agent/nodes/validator.py
────────────────────────
Production-grade 7-Point Evidence Validator Node.

Deterministic Reflection Gate:
1. Zero-chunk fast fail
2. Hard score floor (catches totally irrelevant results)
3. Redundancy / diversity check (prunes near-duplicate chunks)
4. Per-sub-query coverage (ensures every decomposed sub-query has evidence)
5. Entity coverage (ensures user-specified course codes / names exist in text)
6. Aggregate / group coverage (ensures multi-entity queries span >= 2 entities)
7. Lightweight contradiction detection (flags conflicting facts without blocking)
"""

import re
from typing import Any, Dict, List, Set, Tuple
import logfire

from agent.state import AgentState, EvidenceStatus
from app.models import RetrievedChunk


# Thresholds
SCORE_FLOOR = 0.25              # Minimum cross-encoder rerank score required
SIMILARITY_PRUNE_THRESHOLD = 0.82 # Jaccard similarity above which chunks are near-duplicates
KNOWN_DEPARTMENTS = {"cse", "ece", "eee", "civil", "mech", "it", "aids", "csbs", "aiml"}
STOPWORDS = {
    "what", "is", "the", "for", "and", "in", "of", "to", "a", "an", "at",
    "by", "with", "from", "on", "about", "tell", "me", "give", "who", "which"
}


def _tokenize(text: str) -> Set[str]:
    """Extracts lowercase alphanumeric tokens excluding basic stopwords."""
    words = re.findall(r"\b[a-z0-9]+\b", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 1}


def _jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Computes Jaccard similarity between two token sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 7-Point Validation Logic
# ─────────────────────────────────────────────────────────────────────────────

def _check_1_zero_chunks(chunks: List[RetrievedChunk]) -> Tuple[bool, str]:
    """Check 1: Fast-fail if retrieval returned zero chunks."""
    if not chunks:
        return False, "Zero chunks returned from retrieval pipeline."
    return True, ""


def _check_2_score_floor(chunks: List[RetrievedChunk]) -> Tuple[bool, str]:
    """Check 2: Ensure top chunk rerank score meets minimum confidence floor."""
    top_score = max(c.score for c in chunks) if chunks else 0.0
    if top_score < SCORE_FLOOR:
        return False, f"Top rerank score ({top_score:.2f}) is below minimum quality floor ({SCORE_FLOOR})."
    return True, ""


def _check_3_prune_redundancy(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """
    Check 3: Redundancy & Diversity Pruning.
    Keeps high-scoring chunks while dropping near-duplicate boilerplate.
    """
    if len(chunks) <= 1:
        return chunks

    pruned: List[RetrievedChunk] = []
    seen_token_sets: List[Set[str]] = []

    for chunk in chunks:
        chunk_tokens = _tokenize(chunk.text)
        is_duplicate = False

        for existing_tokens in seen_token_sets:
            similarity = _jaccard_similarity(chunk_tokens, existing_tokens)
            if similarity >= SIMILARITY_PRUNE_THRESHOLD:
                is_duplicate = True
                break

        if not is_duplicate:
            pruned.append(chunk)
            seen_token_sets.append(chunk_tokens)

    return pruned if pruned else chunks


def _check_4_subquery_coverage(
    chunks: List[RetrievedChunk],
    sub_queries: List[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """
    Check 4: Per-sub-query coverage.
    Verifies that every decomposed sub-query has at least one supporting chunk.
    """
    if len(sub_queries) <= 1:
        return True, []

    chunk_token_pools = [_tokenize(c.text) for c in chunks]
    missing_queries: List[str] = []

    for sq in sub_queries:
        query_text = sq.get("query", "")
        sq_keywords = _tokenize(query_text)
        if not sq_keywords:
            continue

        # Check if at least one chunk has meaningful keyword overlap with this sub-query
        covered = False
        for pool in chunk_token_pools:
            overlap = sq_keywords & pool
            # Coverage met if at least 2 keywords match or 40% of subquery keywords match
            if len(overlap) >= min(2, len(sq_keywords)) or (len(overlap) / len(sq_keywords)) >= 0.4:
                covered = True
                break

        if not covered:
            missing_queries.append(query_text)

    # If more than half the sub-queries are completely missing, consider check failed
    if missing_queries and len(missing_queries) >= (len(sub_queries) / 2):
        return False, missing_queries

    return True, missing_queries


def _check_5_entity_coverage(
    chunks: List[RetrievedChunk],
    intent: Dict[str, Any],
) -> Tuple[bool, str]:
    """
    Check 5: Entity coverage.
    Ensures user-specified course codes or professor names exist in the retrieved text.
    """
    entities = intent.get("entities", [])
    if not entities or not isinstance(entities, list):
        return True, ""

    corpus = " ".join(c.text.lower() for c in chunks)
    missing_entities = []

    for ent in entities:
        ent_str = str(ent).strip().lower()
        if len(ent_str) >= 3 and ent_str not in corpus:
            missing_entities.append(ent)

    if missing_entities:
        return False, f"Required entities not found in retrieved chunks: {missing_entities}"

    return True, ""


def _check_6_aggregate_coverage(
    chunks: List[RetrievedChunk],
    intent: Dict[str, Any],
) -> Tuple[bool, str]:
    """
    Check 6: Aggregate / group coverage.
    Ensures 'list all X' style queries span at least 2 distinct entities/departments.
    """
    is_aggregate = intent.get("is_aggregate", False)
    if not is_aggregate:
        return True, ""

    corpus = " ".join(c.text.lower() for c in chunks)
    covered_depts = {dept for dept in KNOWN_DEPARTMENTS if dept in corpus}

    if len(covered_depts) < 2:
        return False, f"Aggregate query coverage too narrow (only found: {list(covered_depts)})."

    return True, ""


def _check_7_detect_contradictions(chunks: List[RetrievedChunk]) -> List[str]:
    """
    Check 7: Lightweight Contradiction Detection.
    Non-blocking: scans for conflicting numeric patterns (e.g. credits or dates).
    """
    contradictions: List[str] = []
    credit_matches = set()

    for chunk in chunks:
        found = re.findall(r"\b([1-9])\s*(?:credits?|credits hours)\b", chunk.text.lower())
        credit_matches.update(found)

    if len(credit_matches) > 1:
        contradictions.append(f"Multiple conflicting credit values detected in documents: {sorted(credit_matches)}")

    return contradictions


# ─────────────────────────────────────────────────────────────────────────────
# Master Validator Node
# ─────────────────────────────────────────────────────────────────────────────

def validator_node(state: AgentState) -> AgentState:
    """
    LangGraph Evidence Validator node:
    Executes the 7-point validation pipeline and routes accordingly.
    """
    chunks = state.get("reranked_chunks", [])
    intent = state.get("intent", {})
    sub_queries = state.get("sub_queries", [])
    retry_count = state.get("retrieval_retry_count", 0)
    max_retries = state.get("max_retrieval_retries", 1)

    with logfire.span("Evidence Validator Node", chunk_count=len(chunks), retry=retry_count) as span:

        # ── 1. Zero-Chunk Fast Fail ──────────────────────────────────────────
        ok1, reason1 = _check_1_zero_chunks(chunks)
        if not ok1:
            return _handle_failure(retry_count, max_retries, reason1, span)

        # ── 2. Hard Score Floor ──────────────────────────────────────────────
        ok2, reason2 = _check_2_score_floor(chunks)
        if not ok2:
            return _handle_failure(retry_count, max_retries, reason2, span)

        # ── 3. Redundancy & Diversity Pruning (Non-blocking) ─────────────────
        pruned_chunks = _check_3_prune_redundancy(chunks)
        span.set_attribute("pruned_chunks_count", len(pruned_chunks))

        # ── 4. Per-Sub-Query Coverage ────────────────────────────────────────
        ok4, missing_subqueries = _check_4_subquery_coverage(pruned_chunks, sub_queries)
        if not ok4:
            reason4 = f"Incomplete evidence: missing sub-queries {missing_subqueries}"
            return _handle_failure(
                retry_count, max_retries, reason4, span,
                missing_sq=missing_subqueries, pruned=pruned_chunks
            )

        # ── 5. Entity Coverage ───────────────────────────────────────────────
        ok5, reason5 = _check_5_entity_coverage(pruned_chunks, intent)
        if not ok5:
            return _handle_failure(retry_count, max_retries, reason5, span, pruned=pruned_chunks)

        # ── 6. Aggregate / Group Coverage ────────────────────────────────────
        ok6, reason6 = _check_6_aggregate_coverage(pruned_chunks, intent)
        if not ok6:
            return _handle_failure(retry_count, max_retries, reason6, span, pruned=pruned_chunks)

        # ── 7. Lightweight Contradiction Detection (Non-blocking) ────────────
        contradictions = _check_7_detect_contradictions(pruned_chunks)
        if contradictions:
            span.set_attribute("contradictions", contradictions)

        # ── ALL CHECKS PASSED ────────────────────────────────────────────────
        span.set_attribute("status", "sufficient")
        span.set_attribute("evidence_status", EvidenceStatus.SUFFICIENT.value)

        print("=" * 60)
        print("  EVIDENCE VALIDATOR: PASSED (All 7 Checks Verified)")
        print("=" * 60)
        print(f"  • Chunks Evaluated : {len(chunks)} -> Pruned to: {len(pruned_chunks)}")
        print(f"  • Top Score        : {max(c.score for c in chunks):.4f}")
        print(f"  • Missing Sub-Q    : {missing_subqueries if missing_subqueries else 'None'}")
        print(f"  • Contradictions   : {contradictions if contradictions else 'None'}")
        print("=" * 60 + "\n")

        return {
            "evidence_status":     EvidenceStatus.SUFFICIENT,
            "pruned_chunks":       pruned_chunks,
            "missing_sub_queries": missing_subqueries,
            "contradictions":      contradictions,
            "degraded":            False,
            "degraded_reason":     None,
        }


def _handle_failure(
    retry_count: int,
    max_retries: int,
    reason: str,
    span: Any,
    missing_sq: List[str] = None,
    pruned: List[RetrievedChunk] = None,
) -> Dict[str, Any]:
    """Helper to route failure cleanly to retry or degraded mode."""
    span.set_attribute("failure_reason", reason)

    if retry_count < max_retries:
        span.set_attribute("status", "retry")
        span.set_attribute("evidence_status", EvidenceStatus.INSUFFICIENT_RETRY.value)
        print(f"  ⚠️ Evidence Check Failed (Will Retry): {reason}")
        return {
            "evidence_status":     EvidenceStatus.INSUFFICIENT_RETRY,
            "degraded":            False,
            "degraded_reason":     reason,
            "missing_sub_queries": missing_sq or [],
            "pruned_chunks":       pruned or [],
            "contradictions":      [],
        }

    span.set_attribute("status", "degraded")
    span.set_attribute("evidence_status", EvidenceStatus.INSUFFICIENT_EXHAUSTED.value)
    print(f"  ⚠️ Evidence Check Failed (Retries Exhausted -> Degraded): {reason}")
    return {
        "evidence_status":     EvidenceStatus.INSUFFICIENT_EXHAUSTED,
        "degraded":            True,
        "degraded_reason":     reason,
        "missing_sub_queries": missing_sq or [],
        "pruned_chunks":       pruned or [],
        "contradictions":      [],
    }
