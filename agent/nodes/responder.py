"""
agent/nodes/responder.py
────────────────────────
Final Responder Node for Agentic RAG.

Responsibilities:
1. Returns cached response immediately if cache_hit is True.
2. Handles DIRECT route:
   - Polite conversational greetings and introductions.
   - Polite out-of-scope rejections.
3. Handles NEEDS_RETRIEVAL route:
   - Formats the verified draft answer.
   - Appends deduplicated, clickable source attributions (### Sources:).
   - Emits final answer, sources, and provider for delivery and cache writing.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
from langsmith import traceable

from config import settings
from agent.state import AgentState, RouteType
from agent.prompts import DIRECT_RESPONDER_SYSTEM_PROMPT


def _format_history_context(chat_history: List[Dict[str, str]], limit: int = 10) -> str:
    """Formats recent conversation history for conversational context and meta-queries."""
    if not chat_history:
        return "None"
    recent = chat_history[-limit:]
    parts = []
    for idx, msg in enumerate(recent, 1):
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "").strip()
        parts.append(f"Turn {idx} [{role}]: {content}")
    return "\n".join(parts)


@traceable(name="Groq Direct Responder LLM", run_type="llm")
def _call_groq_direct(prompt_text: str) -> Optional[str]:
    """Calls Groq for fast, natural conversational response generation."""
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
                    {"role": "system", "content": DIRECT_RESPONDER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                temperature=0.3,
                max_tokens=220,
            )
            content = response.choices[0].message.content or ""
            return content.strip()
        except Exception as e:
            print(f"Groq direct responder key failed: {e}")
            continue
    return None


@traceable(name="Gemini Direct Responder Fallback", run_type="llm")
def _call_gemini_direct(prompt_text: str) -> Optional[str]:
    """Fallback to Google Gemini for conversational direct response generation."""
    keys = settings.GEMINI_API_KEYS or ([settings.GEMINI_API_KEY] if settings.GEMINI_API_KEY else [])
    if not keys:
        return None

    from google import genai
    for key in keys:
        try:
            client = genai.Client(api_key=key)
            combined_prompt = f"{DIRECT_RESPONDER_SYSTEM_PROMPT}\n\nUser Context:\n{prompt_text}"
            resp = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=combined_prompt,
            )
            return (resp.text or "").strip()
        except Exception as e:
            print(f"Gemini direct responder key failed: {e}")
            continue
    return None


def _fallback_direct_response(query: str, intent: Dict[str, Any]) -> str:
    """Safe static fallback in case all LLM APIs are unreachable."""
    category = str(intent.get("category", "")).lower()
    q_lower = query.lower().strip()

    if "privacy" in category or any(w in q_lower for w in ["phone number", "mobile number", "whatsapp", "home address", "residential address", "salary"]):
        return (
            "For privacy and institutional policy reasons, personal contact information (such as personal phone numbers, mobile numbers, and residential addresses) of faculty, staff, and students is confidential.\n\n"
            "To connect with faculty or departments, please reach out via official campus email or visit their department cabin during college working hours."
        )

    if any(k in category for k in ["illogical", "unclear", "incomplete", "nonsense"]):
        return (
            "Your question appears to have an unclear premise or does not match academic college departments.\n\n"
            "Academic departments are educational divisions rather than individuals. Please feel free to ask about our academic programs, faculty members, syllabi, or campus facilities!"
        )

    if "out_of_scope" in category:
        return (
            "I am dedicated exclusively to college academics and campus resources. "
            "Please let me know if you have questions regarding academic regulations, syllabi, faculty directories, exams, or admissions!"
        )

    return (
        "Hello! I am your AI Academic Advisor and Campus Assistant. "
        "How can I assist you with your academic curriculum, syllabus details, faculty contacts, or regulations today?"
    )


def _build_direct_response(
    query: str,
    intent: Dict[str, Any],
    chat_history: List[Dict[str, str]],
) -> Tuple[str, str]:
    """Generates polite, context-aware, professional responses via fast LLM inference."""
    # If explicitly blocked by Input Rail with a safety message, respect that immediately
    if intent.get("refusal_message") and str(intent.get("category", "")).startswith("blocked_"):
        return intent["refusal_message"], "SafetyRail"

    category = intent.get("category", "general")
    reason = intent.get("reason") or intent.get("slot_needed") or ""
    history_str = _format_history_context(chat_history)

    prompt_payload = (
        f"Conversation History:\n{history_str}\n\n"
        f"User Query: \"{query}\"\n"
        f"Diagnosed Category: {category}\n"
    )
    if reason:
        prompt_payload += f"Diagnostic Note / Premise Issue: {reason}\n"
    prompt_payload += "\nProvide a natural, helpful, concise response strictly following your advisor persona."

    t0 = time.time()
    response_text = _call_groq_direct(prompt_payload)
    provider_used = f"Groq ({settings.GROQ_MODEL})"

    if not response_text:
        response_text = _call_gemini_direct(prompt_payload)
        provider_used = f"Gemini ({settings.GEMINI_MODEL})"

    if not response_text:
        response_text = _fallback_direct_response(query, intent)
        provider_used = "Direct (Fallback)"

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"  DIRECT RESPONDER COMPLETE ({provider_used} in {elapsed:.2f}s)")
    print(f"  • Category : {category}")
    print(f"  • Response : {response_text[:120]}...")
    print("=" * 60 + "\n")

    return response_text, provider_used


@traceable(name="Responder Node", run_type="chain")
def responder_node(state: AgentState) -> AgentState:
    """
    LangGraph Responder node:
    Reads:  cache_hit, cached_response, route, intent, chat_history, draft_answer, reranked_chunks, pruned_chunks, provider, degraded
    Writes: answer, sources, provider
    """
    cached = state.get("cached_response")
    if state.get("cache_hit") and cached:
        print("  ✓ Returning cached response (< 1ms).")
        return {
            "answer":   cached.get("answer", ""),
            "sources":  cached.get("sources", []),
            "provider": "Cache",
        }

    route = state.get("route", RouteType.NEEDS_RETRIEVAL)
    query = state.get("query", "")
    intent = state.get("intent", {})

    # ── 2. Direct Route (No Retrieval Needed) ────────────────────────────────
    if route == RouteType.DIRECT:
        chat_history = state.get("chat_history", [])
        answer, provider_used = _build_direct_response(query, intent, chat_history)
        return {
            "answer":   answer,
            "sources":  [],
            "provider": provider_used,
        }

    # ── 3. Clarification Route ───────────────────────────────────────────────
    if route == RouteType.CLARIFY:
        clarification = state.get("clarification")
        answer = state.get("answer") or (clarification.get("question") if clarification else "Could you please clarify your question?")
        return {
            "answer":        answer,
            "sources":       [],
            "provider":      "Clarifier",
            "clarification": clarification,
        }

    # ── 3. RAG Route (Format Final Answer with Sources) ──────────────────────
    draft = state.get("draft_answer", "I could not find sufficient documentation in college records.")
    chunks = state.get("pruned_chunks") or state.get("reranked_chunks", [])

    # Extract clean, deduplicated source labels
    sources: List[str] = []
    seen = set()
    for c in chunks:
        src = getattr(c, "source", None)
        if src and src not in seen:
            seen.add(src)
            page = c.metadata.get("page")
            if isinstance(page, int):
                sources.append(f"{src} (Page {page + 1})")
            else:
                sources.append(src)

    # Clean final answer (sources are handled by UI components via sources array)
    final_answer = draft.strip()
    provider = state.get("provider", "Groq")

    return {
        "answer":   final_answer,
        "sources":  sources,
        "provider": provider,
    }
