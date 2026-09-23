"""
guardrails/service.py
─────────────────────
Production-grade NeMo Guardrails Service with LangSmith Observability.
Manages input rails (jailbreak / injection, privacy leaks, off-topic),
dialog steering, and output rails (hallucination / course code grounding, PII scrubbing).
"""

import os
import asyncio
import sys
import threading
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langsmith import traceable

# Ensure config and environment variables are loaded
from config import settings
from guardrails.actions import (
    scrub_pii_text,
    check_course_code_grounding,
    scrub_pii_action,
    check_grounding_action,
)

logger = logging.getLogger(__name__)

# Path to NeMo Guardrails config directory
CURRENT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = CURRENT_DIR / "config"


@dataclass
class GuardResult:
    """Standardized result returned by NeMo Guardrails checks."""
    allowed: bool
    action: str  # "pass", "blocked_jailbreak", "blocked_privacy", "blocked_abuse", "redirect_off_topic", "ungrounded_flagged"
    sanitized_text: Optional[str] = None
    refusal_message: Optional[str] = None
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class NeMoGuardrailsService:
    """
    Production-ready NeMo Guardrails engine for the College AI Academic Advisor.
    Supports input rails, dialog rails, output verification rails, and LangSmith tracing.
    """

    _instance: Optional["NeMoGuardrailsService"] = None
    _instance_lock: threading.Lock = threading.Lock()

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or CONFIG_DIR
        self._rails = None
        self._config = None
        self._setup_env()
        self._init_rails()

    def _setup_env(self) -> None:
        """Synchronizes API keys and LangSmith environment configuration."""
        groq_key = settings.GROQ_API_KEY
        if groq_key:
            # NeMo Guardrails openai engine reads OPENAI_API_KEY if base_url is set to Groq
            if not os.getenv("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = groq_key
            if not os.getenv("GROQ_API_KEY"):
                os.environ["GROQ_API_KEY"] = groq_key

        # LangSmith Observability setup
        if settings.LANGSMITH_API_KEY:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
            os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT

    def _init_rails(self) -> None:
        """Loads RailsConfig and initializes LLMRails with custom actions."""
        try:
            from nemoguardrails import RailsConfig, LLMRails

            if not self.config_dir.exists():
                logger.error(f"[NeMo Guardrails] Config directory does not exist: {self.config_dir}")
                return

            self._config = RailsConfig.from_path(str(self.config_dir))
            self._rails = LLMRails(self._config)

            # Register custom actions explicitly as well
            self._rails.register_action(scrub_pii_action, name="scrub_pii_action")
            self._rails.register_action(check_grounding_action, name="check_grounding_action")

            logger.info("[NeMo Guardrails] Initialized successfully with production configuration.")
        except Exception as e:
            logger.exception(f"[NeMo Guardrails] Failed to initialize LLMRails: {e}")
            self._rails = None

    # ─────────────────────────────────────────────────────────────────────────
    # Input Rails Check
    # ─────────────────────────────────────────────────────────────────────────

    @traceable(name="NeMo Guardrails: Input Rail Check", run_type="chain")
    def check_input(self, user_query: str) -> GuardResult:
        """
        Executes input rails against user query.
        Detects prompt injections, jailbreak attempts, privacy requests (faculty phone numbers),
        and off-domain queries.
        """
        q_clean = user_query.strip().lower()
        t0 = time.time()

        # 1. Fast deterministic check: Privacy & Personal Contact requests
        privacy_keywords = [
            "phone number", "mobile number", "whatsapp number", "home address",
            "residential address", "salary", "personal email", "private contact",
            "contact number of", "phone no", "mobile no"
        ]
        if any(kw in q_clean for kw in privacy_keywords):
            elapsed = time.time() - t0
            refusal = (
                "Institutional Privacy Notice: In accordance with campus privacy policies and data protection regulations, "
                "personal contact details (including personal mobile numbers, WhatsApp contacts, residential addresses, and private records) "
                "of faculty, staff, and students are strictly confidential.\n\n"
                "To contact faculty or administrative officials, please use official campus email addresses or visit their respective department cabins during official working hours."
            )
            return GuardResult(
                allowed=False,
                action="blocked_privacy",
                refusal_message=refusal,
                reason="Query requested restricted personal contact information.",
                metadata={"elapsed_sec": elapsed, "source": "input.co"}
            )

        # 2. Fast deterministic check: Prompt Injection & Jailbreak attempts
        jailbreak_keywords = [
            "ignore all previous instructions", "ignore previous instructions", "disregard all previous",
            "act as dan", "dan mode", "developer mode", "unrestricted mode", "hidden system prompt",
            "reveal system prompt", "bypass guardrails", "hack into", "disclose internal instructions",
            "override instructions"
        ]
        if any(kw in q_clean for kw in jailbreak_keywords):
            elapsed = time.time() - t0
            refusal = (
                "Security Alert: I am strictly governed by institutional safety policies. "
                "I cannot execute system overrides, disclose internal configuration prompts, "
                "or perform unauthorized administrative actions. Please submit a valid college academic inquiry."
            )
            return GuardResult(
                allowed=False,
                action="blocked_jailbreak",
                refusal_message=refusal,
                reason="Query triggered prompt injection / jailbreak security rail.",
                metadata={"elapsed_sec": elapsed, "source": "input.co"}
            )

        # 3. Off-Topic inquiries
        off_topic_keywords = [
            "cricket match", "biryani", "recipe for", "bollywood movie", "stock price",
            "who won the world cup", "football score"
        ]
        if any(kw in q_clean for kw in off_topic_keywords):
            elapsed = time.time() - t0
            redirect = (
                "I am the dedicated AI Academic Advisor for SRKR Engineering College.\n\n"
                "My expertise is focused on college academics, degree curricula, exam guidelines, department courses, "
                "faculty directories, and campus facilities. I cannot assist with non-academic or external topics.\n\n"
                "Please ask a question related to your studies or campus academics!"
            )
            return GuardResult(
                allowed=False,
                action="redirect_off_topic",
                refusal_message=redirect,
                reason="Query is outside college academic scope.",
                metadata={"elapsed_sec": elapsed, "source": "input.co"}
            )

        # 4. If NeMo Guardrails LLM engine is active, verify via Colang rules
        if self._rails:
            try:
                # Synchronous check using generate
                response = self._rails.generate(messages=[{"role": "user", "content": user_query}])
                bot_text = response.get("content", "") if isinstance(response, dict) else getattr(response, "content", str(response))

                # Check if a refusal bot message was returned by Colang flows
                if "Security Alert" in bot_text or "disclose internal configuration" in bot_text:
                    return GuardResult(
                        allowed=False,
                        action="blocked_jailbreak",
                        refusal_message=bot_text,
                        reason="Colang jailbreak flow triggered.",
                        metadata={"elapsed_sec": time.time() - t0, "source": "nemoguardrails"}
                    )
                elif "Institutional Privacy Notice" in bot_text or "strictly confidential" in bot_text:
                    return GuardResult(
                        allowed=False,
                        action="blocked_privacy",
                        refusal_message=bot_text,
                        reason="Colang privacy flow triggered.",
                        metadata={"elapsed_sec": time.time() - t0, "source": "nemoguardrails"}
                    )
                elif "dedicated AI Academic Advisor" in bot_text or "non-academic or external topics" in bot_text:
                    return GuardResult(
                        allowed=False,
                        action="redirect_off_topic",
                        refusal_message=bot_text,
                        reason="Colang off-topic flow triggered.",
                        metadata={"elapsed_sec": time.time() - t0, "source": "nemoguardrails"}
                    )
            except Exception as e:
                logger.warning(f"[NeMo Guardrails] LLM check_input encountered error (falling back to deterministic): {e}")

        # Input is allowed
        return GuardResult(
            allowed=True,
            action="pass",
            sanitized_text=user_query,
            metadata={"elapsed_sec": time.time() - t0, "source": "input.co"}
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Output Rails Check
    # ─────────────────────────────────────────────────────────────────────────

    @traceable(name="NeMo Guardrails: Output Rail Check", run_type="chain")
    def check_output(
        self,
        query: str,
        draft_answer: str,
        context_chunks: Optional[List[Any]] = None,
    ) -> GuardResult:
        """
        Executes output rails against generated draft answer.
        1. Scrubs all phone numbers and PII deterministically.
        2. Validates factual grounding: checks for hallucinated course codes.
        3. Formats compliant output with LangSmith tracing.
        """
        t0 = time.time()

        # Step 1: PII Scrubbing
        sanitized_text, was_scrubbed = scrub_pii_text(draft_answer)

        # Step 2: Course code / factual grounding check
        evidence_text = ""
        if context_chunks:
            evidence_text = " ".join(
                c.get("text", "") if isinstance(c, dict) else getattr(c, "text", str(c))
                for c in context_chunks
            )

        is_grounded, reason = check_course_code_grounding(sanitized_text, evidence_text)
        elapsed = time.time() - t0

        if not is_grounded:
            return GuardResult(
                allowed=False,
                action="ungrounded_flagged",
                sanitized_text=sanitized_text,
                reason=reason,
                metadata={
                    "is_grounded": False,
                    "was_scrubbed": was_scrubbed,
                    "elapsed_sec": elapsed,
                    "source": "output.co"
                }
            )

        return GuardResult(
            allowed=True,
            action="pass",
            sanitized_text=sanitized_text,
            metadata={
                "is_grounded": True,
                "was_scrubbed": was_scrubbed,
                "elapsed_sec": elapsed,
                "source": "output.co"
            }
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Async Methods for FastAPI & Streaming
    # ─────────────────────────────────────────────────────────────────────────

    @traceable(name="NeMo Guardrails: Async Input Rail Check", run_type="chain")
    async def acheck_input(self, user_query: str) -> GuardResult:
        """Async wrapper for input rail checking. Offloads blocking sync work to a thread."""
        return await asyncio.to_thread(self.check_input, user_query)

    @traceable(name="NeMo Guardrails: Async Output Rail Check", run_type="chain")
    async def acheck_output(
        self,
        query: str,
        draft_answer: str,
        context_chunks: Optional[List[Any]] = None,
    ) -> GuardResult:
        """Async wrapper for output rail checking. Offloads blocking sync work to a thread."""
        return await asyncio.to_thread(self.check_output, query, draft_answer, context_chunks)


def get_guardrails_service() -> NeMoGuardrailsService:
    """Thread-safe singleton getter for the NeMo Guardrails service."""
    if NeMoGuardrailsService._instance is None:
        with NeMoGuardrailsService._instance_lock:
            # Double-checked locking: re-verify after acquiring the lock
            if NeMoGuardrailsService._instance is None:
                NeMoGuardrailsService._instance = NeMoGuardrailsService()
    return NeMoGuardrailsService._instance
