"""
guardrails package
──────────────────
Production-grade NeMo Guardrails integration for College AI Academic Advisor.
"""

from guardrails.service import NeMoGuardrailsService, get_guardrails_service, GuardResult
from guardrails.actions import scrub_pii_text, check_course_code_grounding

__all__ = [
    "NeMoGuardrailsService",
    "get_guardrails_service",
    "GuardResult",
    "scrub_pii_text",
    "check_course_code_grounding",
]
