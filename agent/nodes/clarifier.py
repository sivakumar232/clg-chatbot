"""
agent/nodes/clarifier.py
────────────────────────
Clarifier Node for Agentic RAG.

Responsibilities:
1. Production-grade dedicated sub-agent for conversational disambiguation.
2. Invokes an LLM with a specialized Clarification System Prompt when the planner
   identifies an under-specified or ambiguous query.
3. Dynamically generates:
   - A friendly, context-aware clarification question tailored to the user's exact phrasing.
   - 3 to 6 highly relevant, dynamic quick-reply option chips (e.g. departments, regulations, fee types)
     without rigid hardcoded templates.
4. Returns the clarification payload directly to the responder node, bypassing expensive retrieval/reranking.
"""

import json
import time
from typing import Any, Dict, List, Optional
from langsmith import traceable

from config import settings
from agent.state import AgentState, ClarificationPayload
from agent.prompts import CLARIFIER_SYSTEM_PROMPT



@traceable(name="Groq Clarifier LLM Inference", run_type="llm")
def _call_groq_clarifier(prompt_text: str) -> Optional[Dict[str, Any]]:
    """Calls Groq with strict JSON output format for clarification generation."""
    keys = settings.GROQ_API_KEYS or ([settings.GROQ_API_KEY] if settings.GROQ_API_KEY else [])
    if not keys:
        return None

    from groq import Groq
    for key in keys:
        try:
            client = Groq(api_key=key)
            response = client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": CLARIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=256,
            )
            content = response.choices[0].message.content or ""
            parsed = json.loads(content)
            return parsed
        except Exception as e:
            print(f"Groq clarifier key failed: {e}")
            continue
    return None


@traceable(name="Gemini Clarifier Fallback Inference", run_type="llm")
def _call_gemini_clarifier(prompt_text: str) -> Optional[Dict[str, Any]]:
    """Fallback to Google Gemini for JSON clarification generation."""
    keys = settings.GEMINI_API_KEYS or ([settings.GEMINI_API_KEY] if settings.GEMINI_API_KEY else [])
    if not keys:
        return None

    from google import genai
    for key in keys:
        try:
            client = genai.Client(api_key=key)
            combined_prompt = f"{CLARIFIER_SYSTEM_PROMPT}\n\nUser Input:\n{prompt_text}\n\nRespond with ONLY valid JSON."
            resp = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=combined_prompt,
            )
            raw_text = (resp.text or "").strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]

            parsed = json.loads(raw_text.strip())
            return parsed
        except Exception as e:
            print(f"Gemini clarifier key failed: {e}")
            continue
    return None


@traceable(name="Clarifier Node", run_type="chain")
def clarifier_node(state: AgentState) -> AgentState:
    """
    LangGraph Clarifier node:
    Reads:  query, rewritten_query, intent, chat_history
    Writes: answer, clarification
    """
    raw_query = state.get("query", "").strip()
    rewritten_query = state.get("rewritten_query") or raw_query
    intent = state.get("intent") or {}
    slot_needed = intent.get("slot_needed") or "missing_detail"
    category = intent.get("category") or "general"

    # Build prompt for Clarifier LLM
    prompt_payload = (
        f"User Query: \"{raw_query}\"\n"
        f"Resolved Query: \"{rewritten_query}\"\n"
        f"Intent Category: {category}\n"
        f"Missing Parameter / Slot Needed: {slot_needed}\n\n"
        f"Generate the clarifying question and quick-reply options."
    )

    t0 = time.time()
    result_dict = _call_groq_clarifier(prompt_payload)
    provider_used = "Groq"

    if result_dict is None:
        result_dict = _call_gemini_clarifier(prompt_payload)
        provider_used = "Gemini"

    elapsed = time.time() - t0

    # Resiliency fallback if LLM call fails
    if not result_dict or not isinstance(result_dict, dict):
        question = f"Could you please specify which {slot_needed.replace('_', ' ')} you are inquiring about?"
        options = []
    else:
        question = result_dict.get("question") or f"Could you please clarify your question regarding '{raw_query}'?"
        raw_options = result_dict.get("options") or []
        options = [str(opt).strip() for opt in raw_options if str(opt).strip()]

    payload: ClarificationPayload = {
        "question": question,
        "slot_needed": slot_needed,
        "options": options,
    }

    print("=" * 60)
    print(f"  CLARIFIER NODE ({provider_used} in {elapsed:.2f}s)")
    print("=" * 60)
    print(f"  • Slot Needed : {slot_needed}")
    print(f"  • Question    : {question}")
    print(f"  • Dynamic Options ({len(options)}): {options}\n")
    print("=" * 60 + "\n")

    return {
        "answer": question,
        "clarification": payload,
    }
