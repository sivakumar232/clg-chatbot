"""
guardrails/actions.py
─────────────────────
Production-grade custom actions for NeMo Guardrails with LangSmith observability.
Includes:
- PII and phone number scrubbing (deterministic regex filter)
- Course code and factual grounding verification
- LangSmith tracing integration for guardrail intervention events
"""

import re
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from langsmith import traceable
from nemoguardrails.actions import action

from config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic PII Scrubbing
# ─────────────────────────────────────────────────────────────────────────────

def scrub_pii_text(text: str) -> Tuple[str, bool]:
    """
    Deterministically scrubs mobile numbers, phone numbers, and sensitive contact digits
    from text to prevent privacy leaks.
    Returns (cleaned_text, was_scrubbed).
    """
    if not text:
        return text, False

    original = text
    replacement = "[Contact number withheld for privacy]"

    # Indian mobile numbers (+91-..., 9848..., 9493671967, etc.)
    text = re.sub(r'(?:\+91[\s-]?)?[6-9]\d{9}\b', replacement, text)
    # Formats like 98484-66678 or 98484 66678
    text = re.sub(r'\b[6-9]\d{4}[\s-]\d{5}\b', replacement, text)
    # Formats like 08816-223344 (landlines with STD code)
    text = re.sub(r'\b0\d{3,5}[-\s]?\d{6,8}\b', replacement, text)

    was_scrubbed = text != original
    return text, was_scrubbed


# ─────────────────────────────────────────────────────────────────────────────
# Course Code / Entity Grounding Check
# ─────────────────────────────────────────────────────────────────────────────

def check_course_code_grounding(response_text: str, context_text: str) -> Tuple[bool, Optional[str]]:
    """
    Verifies that any academic course code mentioned in the answer
    (e.g., CS3201, B23CS1201, IT201) actually exists in the retrieved context.
    """
    if not response_text or not context_text:
        return True, None

    # Matches uppercase course codes like CS3201, B23CS1201, IT201, etc.
    mentioned_codes = set(re.findall(r"\b[A-Z]{1,4}\d{3,4}[A-Z]?\b", response_text))
    if not mentioned_codes:
        return True, None

    # Common institutional uppercase tokens that are not course codes
    institutional_tokens = {
        "IEEE", "NAAC", "NIRF", "AICTE", "JNTUK", "SRKR",
        "R19", "R20", "R23", "R24", "BTECH", "MTECH", "MCA", "MBA", "CSE", "ECE", "AIDS", "MECH", "CIVIL", "IT", "CSBS", "EEE"
    }
    course_codes = [c for c in mentioned_codes if c not in institutional_tokens]

    for code in course_codes:
        if code not in context_text:
            return False, f"Course code '{code}' mentioned in response was not found in official context."

    # Dynamic alias / identity fabrication check
    alias_match = re.search(r"\b(commonly known as|also known as|known as|referred to as|aka|a\.k\.a\.)\b", response_text, re.IGNORECASE)
    if alias_match:
        matched_phrase = alias_match.group(0)
        if matched_phrase.lower() not in context_text.lower():
            return False, f"Response asserted an unverified alias or identity equivalence ('{matched_phrase}') not present in official context."

    return True, None


# ─────────────────────────────────────────────────────────────────────────────
# NeMo Guardrails Registered Actions (with LangSmith Tracing)
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="NeMo Guardrails: Scrub PII Action", run_type="chain")
@action(name="scrub_pii_action")
async def scrub_pii_action(
    context: Optional[dict] = None,
    bot_message: Optional[str] = None,
    **kwargs: Any,
) -> Optional[str]:
    """
    NeMo Guardrails action to scrub PII from generated bot messages.
    """
    context = context or {}
    message = bot_message or context.get("last_bot_message") or context.get("bot_message") or ""

    cleaned, was_scrubbed = scrub_pii_text(message)
    if was_scrubbed:
        logger.info("[NeMo Guardrails] PII / Phone number detected and redacted from response.")
        return cleaned

    return None


@traceable(name="NeMo Guardrails: Grounding Check Action", run_type="chain")
@action(name="check_grounding_action")
async def check_grounding_action(
    context: Optional[dict] = None,
    bot_message: Optional[str] = None,
    relevant_chunks: Optional[Any] = None,
    **kwargs: Any,
) -> bool:
    """
    NeMo Guardrails action to verify that the bot output is grounded
    in the retrieved evidence and contains no fabricated course codes.
    """
    context = context or {}
    message = bot_message or context.get("last_bot_message") or context.get("bot_message") or ""
    evidence_chunks = relevant_chunks if relevant_chunks is not None else context.get("relevant_chunks", [])

    if not evidence_chunks:
        return True

    # Combine evidence chunks into text
    if isinstance(evidence_chunks, list):
        evidence_text = " ".join(
            chunk.get("text", "") if isinstance(chunk, dict) else getattr(chunk, "text", str(chunk))
            for chunk in evidence_chunks
        )
    else:
        evidence_text = str(evidence_chunks)

    is_grounded, reason = check_course_code_grounding(message, evidence_text)
    if not is_grounded:
        logger.warning(f"[NeMo Guardrails] Grounding check failed: {reason}")
        return False

    return True
