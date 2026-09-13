"""
agent/nodes/guard.py
────────────────────
Answer Grounding and Faithfulness Guard Node for Agentic RAG.

Responsibilities:
1. Compares draft_answer against retrieved context chunks.
2. Checks for:
   - Hallucinated course codes or entity names not present in evidence.
   - Unsupported factual claims.
3. Decision:
   - GROUNDED: Passes draft directly to responder.
   - UNGROUNDED_RETRY (guard_retry_count < max_guard_retries):
     Loops back to generator with diagnostic feedback.
   - UNGROUNDED_EXHAUSTED:
     Proceeds to responder with unverified claims flagged or sanitized.
"""

import json
import re
import time
from typing import Any, Dict, List, Tuple
import logfire

from config import settings
from agent.state import AgentState, GuardStatus
from app.models import RetrievedChunk


GUARD_SYSTEM_PROMPT = """You are a strict academic verification guard for an educational institution.
Your job is to determine if a generated answer is strictly grounded in and faithful to the provided context blocks.

EVALUATION CRITERIA:
1. Every factual statement (course code, credit number, faculty designation, policy rule) MUST be supported by the context.
2. No hallucinated course codes or extrapolated numbers that do not appear in the context.
3. If the answer accurately reflects the context or states that certain facts could not be found, mark it as grounded.

Output MUST be a JSON object:
{
  "is_grounded": true | false,
  "reason": "Clear explanation of what claim was unsupported, or null if grounded"
}
"""


def _fast_deterministic_entity_check(draft_answer: str, chunks: List[RetrievedChunk]) -> Tuple[bool, str | None]:
    """
    Fast regex check: scans draft_answer for academic course codes (e.g. CS3201, B23HS1201).
    Ensures every mentioned course code actually exists in the retrieved text.
    """
    if not chunks:
        return True, None

    # Matches course codes like CS3201, B23CS1201, IT201, etc.
    mentioned_codes = set(re.findall(r"\b[A-Z]{1,4}\d{3,4}[A-Z]?\b", draft_answer))
    if not mentioned_codes:
        return True, None

    corpus = " ".join(c.text for c in chunks)
    for code in mentioned_codes:
        # Ignore common non-course uppercase codes like IEEE, NBA, NAAC, R20, R23, R24
        if code in {"IEEE", "NAAC", "NIRF", "AICTE", "JNTUK", "SRKR", "R19", "R20", "R23", "R24"}:
            continue
        if code not in corpus:
            return False, f"Course code '{code}' was mentioned in answer but not found in official context documents."

    return True, None


def _call_llm_faithfulness_check(draft_answer: str, chunks: List[RetrievedChunk]) -> Tuple[bool, str | None]:
    """Uses fast LLM inference to verify factual faithfulness."""
    if not settings.GROQ_API_KEY:
        return True, None

    context_text = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks[:5], 1))
    prompt = (
        f"Context Blocks:\n{context_text}\n\n"
        f"Generated Draft Answer:\n{draft_answer}\n\n"
        f"Is the draft answer strictly faithful to the context blocks?"
    )

    try:
        from groq import Groq
        client = Groq(api_key=settings.GROQ_API_KEY)
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": GUARD_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=150,
        )
        content = response.choices[0].message.content or ""
        data = json.loads(content)
        is_grounded = bool(data.get("is_grounded", True))
        reason = data.get("reason")
        return is_grounded, reason
    except Exception as e:
        logfire.warn(f"Guard LLM check failed: {e}", exc_info=True)
        return True, None


def guard_node(state: AgentState) -> AgentState:
    """
    LangGraph Guard node:
    Reads:  draft_answer, reranked_chunks, pruned_chunks, guard_retry_count, max_guard_retries
    Writes: guard_status, guard_feedback, guard_retry_count
    """
    draft = state.get("draft_answer", "").strip()
    chunks = state.get("pruned_chunks") or state.get("reranked_chunks", [])
    retry_count = state.get("guard_retry_count", 0)
    max_retries = state.get("max_guard_retries", 1)

    with logfire.span("Answer Guard Node", retry=retry_count) as span:
        t0 = time.time()

        # 1. Deterministic code check (e.g. fabricated course codes)
        det_ok, det_reason = _fast_deterministic_entity_check(draft, chunks)
        if not det_ok:
            is_grounded = False
            feedback = det_reason
        else:
            # 2. LLM semantic faithfulness check
            is_grounded, feedback = _call_llm_faithfulness_check(draft, chunks)

        elapsed = time.time() - t0
        span.set_attribute("is_grounded", is_grounded)
        span.set_attribute("latency_seconds", round(elapsed, 3))

        # Decision routing
        if is_grounded:
            span.set_attribute("status", "grounded")
            print("=" * 60)
            print(f"  ANSWER GUARD: PASSED (Grounded in {elapsed:.2f}s)")
            print("=" * 60 + "\n")
            return {
                "guard_status":   GuardStatus.GROUNDED,
                "guard_feedback": None,
            }

        # Handle ungrounded
        if retry_count < max_retries:
            new_retries = retry_count + 1
            span.set_attribute("status", "retry")
            span.set_attribute("feedback", feedback)
            print("=" * 60)
            print(f"  ⚠️ ANSWER GUARD: REJECTED (Retry {new_retries}/{max_retries})")
            print(f"  • Reason: {feedback}")
            print("=" * 60 + "\n")
            return {
                "guard_status":      GuardStatus.UNGROUNDED_RETRY,
                "guard_feedback":    feedback,
                "guard_retry_count": new_retries,
            }

        # Retries exhausted -> proceed to responder
        span.set_attribute("status", "exhausted")
        print("=" * 60)
        print(f"  ⚠️ ANSWER GUARD: Retries Exhausted -> Proceeding with Notice")
        print(f"  • Reason: {feedback}")
        print("=" * 60 + "\n")
        return {
            "guard_status":      GuardStatus.UNGROUNDED_EXHAUSTED,
            "guard_feedback":    feedback,
        }
