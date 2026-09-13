"""agent/nodes/cache.py — in-memory cache check and cache write."""

from typing import Any, Dict
from agent.state import AgentState

# Simple in-memory LRU cache dictionary: {normalized_query: {"answer": ..., "sources": ..., "provider": ...}}
_QUERY_CACHE: Dict[str, Dict[str, Any]] = {}


def cache_node(state: AgentState) -> AgentState:
    query = state.get("query", "").strip().lower()
    if query in _QUERY_CACHE:
        return {
            "cache_hit": True,
            "cached_response": _QUERY_CACHE[query],
        }
    return {
        "cache_hit": False,
        "cached_response": None,
    }


def cache_write_node(state: AgentState) -> AgentState:
    query = state.get("query", "").strip().lower()
    if query and not state.get("cache_hit", False) and state.get("answer"):
        _QUERY_CACHE[query] = {
            "answer": state.get("answer", ""),
            "sources": state.get("sources", []),
            "provider": state.get("provider", "unknown"),
        }
    return {}
