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

from typing import Any, Dict, List
from langsmith import traceable

from agent.state import AgentState, RouteType


def _build_direct_response(query: str, intent: Dict[str, Any]) -> str:
    """Generates polite, professional responses for conversational or out-of-scope inputs."""
    if intent.get("refusal_message"):
        return intent["refusal_message"]

    category = str(intent.get("category", "")).lower()
    q_lower = query.lower().strip()

    # Privacy restriction refusal
    if "privacy" in category or any(w in q_lower for w in ["phone number", "mobile number", "whatsapp", "home address", "residential address", "salary"]):
        return (
            "For privacy and security policies, personal contact information (such as personal phone numbers, mobile numbers, residential addresses, and private records) of faculty, staff, and students is not disclosed.\n\n"
            "If you need to contact a faculty member or department, please reach out through official campus email or visit the respective department office during working hours."
        )

    # Out-of-scope rejection
    if "out_of_scope" in category or any(w in q_lower for w in ["biryani", "recipe", "cricket", "movie", "weather", "song", "essay"]):
        return (
            "I am the official AI Academic Advisor and Campus Assistant for the college.\n\n"
            "I can assist with college-related queries, such as:\n"
            "- Academic regulations and course syllabi\n"
            "- Departmental curriculum, labs, and elective subjects\n"
            "- Faculty directories and HOD contact information\n"
            "- College admissions, examinations, and placement statistics\n\n"
            "Please ask a question related to college academics or campus facilities!"
        )

    # Conversational greeting
    return (
        "Hello! I am your AI Academic Advisor and Campus Assistant.\n\n"
        "How can I assist you with your academic curriculum, syllabus details, faculty contacts, or regulations today?"
    )


@traceable(name="Responder Node", run_type="chain")
def responder_node(state: AgentState) -> AgentState:
    """
    LangGraph Responder node:
    Reads:  cache_hit, cached_response, route, intent, draft_answer, reranked_chunks, pruned_chunks, provider, degraded
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
        answer = _build_direct_response(query, intent)
        return {
            "answer":   answer,
            "sources":  [],
            "provider": "Direct",
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
