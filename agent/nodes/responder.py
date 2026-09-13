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
import logfire

from agent.state import AgentState, RouteType


def _build_direct_response(query: str, intent: Dict[str, Any]) -> str:
    """Generates polite, professional responses for conversational or out-of-scope inputs."""
    category = str(intent.get("category", "")).lower()
    q_lower = query.lower().strip()

    # Out-of-scope rejection
    if "out_of_scope" in category or any(w in q_lower for w in ["biryani", "recipe", "cricket", "movie", "weather", "song", "essay"]):
        return (
            "I am the official AI Academic Advisor for **SRKR Engineering College (Autonomous), Bhimavaram**.\n\n"
            "I can only assist with college-related queries, such as:\n"
            "- Academic regulations (R20, R23, R24) and course syllabi\n"
            "- Departmental curriculum, labs, and elective subjects\n"
            "- Faculty directories and HOD contact information\n"
            "- College admissions, examinations, and placement statistics\n\n"
            "Please ask a question related to SRKR Engineering College!"
        )

    # Conversational greeting
    return (
        "Hello! I am your AI Academic Advisor for **SRKR Engineering College (Autonomous)**.\n\n"
        "How can I assist you with your academic curriculum, syllabus details, faculty contacts, or college regulations today?"
    )


def responder_node(state: AgentState) -> AgentState:
    """
    LangGraph Responder node:
    Reads:  cache_hit, cached_response, route, intent, draft_answer, reranked_chunks, pruned_chunks, provider, degraded
    Writes: answer, sources, provider
    """
    # ── 1. Cache Hit Path ────────────────────────────────────────────────────
    if state.get("cache_hit") and state.get("cached_response"):
        cached = state["cached_response"]
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

    # Format sources footer if sources exist and aren't already formatted in the draft
    final_answer = draft.strip()
    if sources and "### Sources" not in final_answer:
        source_bullets = "\n".join(f"- {s}" for s in sources)
        final_answer = f"{final_answer}\n\n### Sources:\n{source_bullets}"

    provider = state.get("provider", "Groq")

    return {
        "answer":   final_answer,
        "sources":  sources,
        "provider": provider,
    }
