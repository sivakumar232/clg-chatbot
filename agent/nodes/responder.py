"""agent/nodes/responder.py — prepares final response (grounded, hedged, conversational, or cached)."""

from agent.state import AgentState, RouteType


def responder_node(state: AgentState) -> AgentState:
    if state.get("cache_hit") and state.get("cached_response"):
        cached = state["cached_response"]
        return {
            "answer": cached.get("answer", ""),
            "sources": cached.get("sources", []),
            "provider": "Cache",
        }

    route = state.get("route", RouteType.NEEDS_RETRIEVAL)
    if route == RouteType.DIRECT:
        return {
            "answer": "Hello! How can I assist you with SRKR Engineering College academic information?",
            "sources": [],
            "provider": "Direct",
        }

    draft = state.get("draft_answer", "No answer generated.")
    chunks = state.get("reranked_chunks", [])
    sources = list(dict.fromkeys(c.source for c in chunks if hasattr(c, "source")))

    return {
        "answer": draft,
        "sources": sources,
        "provider": state.get("provider", "Groq"),
    }
