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
from langsmith import traceable

from config import settings
from agent.state import AgentState, GuardStatus
from app.models import RetrievedChunk
from guardrails.actions import scrub_pii_text


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


@traceable(name="Guard LLM Faithfulness Check", run_type="llm")
def _call_llm_faithfulness_check(draft_answer: str, chunks: List[RetrievedChunk]) -> Tuple[bool, str | None]:
    """Uses fast LLM inference to verify factual faithfulness with key failover."""
    groq_keys = settings.GROQ_API_KEYS or ([settings.GROQ_API_KEY] if settings.GROQ_API_KEY else [])
    gemini_keys = settings.GEMINI_API_KEYS or ([settings.GEMINI_API_KEY] if settings.GEMINI_API_KEY else [])

    if not groq_keys and not gemini_keys:
        return True, None

    context_text = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks[:5], 1))
    prompt = (
        f"Context Blocks:\n{context_text}\n\n"
        f"Generated Draft Answer:\n{draft_answer}\n\n"
        f"Is the draft answer strictly faithful to the context blocks?"
    )

    # 1. Try Groq keys
    for key in groq_keys:
        try:
            from groq import Groq
            client = Groq(api_key=key)
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
            print(f"Guard Groq check failed on key: {e}")
            continue

    # 2. Fallback to Gemini keys
    for key in gemini_keys:
        try:
            from google import genai
            client = genai.Client(api_key=key)
            combined = f"{GUARD_SYSTEM_PROMPT}\n\n{prompt}\n\nRespond with ONLY JSON: {{\"is_grounded\": true/false, \"reason\": \"...\"}}"
            resp = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=combined,
            )
            raw = (resp.text or "").strip()
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.startswith("```"):
                raw = raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            data = json.loads(raw.strip())
            return bool(data.get("is_grounded", True)), data.get("reason")
        except Exception as e:
            print(f"Guard Gemini fallback failed on key: {e}")
            continue

    return True, None


@traceable(name="Answer Guard Node", run_type="chain")
def guard_node(state: AgentState) -> AgentState:
    """
    LangGraph Guard node:
    Reads:  draft_answer, reranked_chunks, pruned_chunks, guard_retry_count, max_guard_retries
    Writes: guard_status, guard_feedback, guard_retry_count

    Strategy: fast deterministic course-code check only.
    Semantic faithfulness is enforced upstream via the generator's
    self-verification system prompt (saves 500-1200ms per query).
    """
    draft = state.get("draft_answer", "").strip()
    chunks = state.get("pruned_chunks") or state.get("reranked_chunks", [])
    retry_count = state.get("guard_retry_count", 0)
    max_retries = state.get("max_guard_retries", 1)

    t0 = time.time()

    # NeMo Guardrails Output Rail execution (includes PII scrubbing + grounding check)
    try:
        from guardrails import get_guardrails_service
        out_guard = get_guardrails_service().check_output(
            query=state.get("query", ""),
            draft_answer=draft,
            context_chunks=chunks,
        )
        is_grounded = out_guard.allowed
        feedback = out_guard.reason
        sanitized_draft = out_guard.sanitized_text or draft
    except Exception as e:
        print(f"NeMo Guardrails output check warning, using local fallback: {e}")
        is_grounded, feedback = _fast_deterministic_entity_check(draft, chunks)
        sanitized_draft, _ = scrub_pii_text(draft)

    elapsed = time.time() - t0

    # Decision routing
    if is_grounded:
        print("=" * 60)
        print(f"  NEMO OUTPUT GUARD: PASSED in {elapsed:.2f}s")
        print("=" * 60 + "\n")
        return {
            "draft_answer":   sanitized_draft,
            "guard_status":   GuardStatus.GROUNDED,
            "guard_feedback": None,
        }

    # Handle ungrounded (fabricated course code detected)
    if retry_count < max_retries:
        new_retries = retry_count + 1
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
    print("=" * 60)
    print(f"  ⚠️ ANSWER GUARD: Retries Exhausted -> Proceeding with Notice")
    print(f"  • Reason: {feedback}")
    print("=" * 60 + "\n")
    return {
        "draft_answer":      sanitized_draft,
        "guard_status":      GuardStatus.UNGROUNDED_EXHAUSTED,
        "guard_feedback":    feedback,
    }

