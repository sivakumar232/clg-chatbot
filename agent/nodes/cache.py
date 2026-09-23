import threading
from collections import OrderedDict
from typing import Any, Dict
from langsmith import traceable
from agent.state import AgentState

# Thread-safe bounded LRU cache (max 500 entries)
_QUERY_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_CACHE_LOCK = threading.Lock()
_MAX_CACHE_SIZE = 500


@traceable(name="Cache Lookup Node", run_type="chain")
def cache_node(state: AgentState) -> AgentState:
    query = state.get("query", "").strip().lower()
    with _CACHE_LOCK:
        if query in _QUERY_CACHE:
            # Move to end to indicate recent use (LRU)
            _QUERY_CACHE.move_to_end(query)
            return {
                "cache_hit": True,
                "cached_response": _QUERY_CACHE[query],
            }
    return {
        "cache_hit": False,
        "cached_response": None,
    }


@traceable(name="Cache Write Node", run_type="chain")
def cache_write_node(state: AgentState) -> AgentState:
    query = state.get("query", "").strip().lower()
    if query and not state.get("cache_hit", False) and state.get("answer"):
        with _CACHE_LOCK:
            _QUERY_CACHE[query] = {
                "answer": state.get("answer", ""),
                "sources": state.get("sources", []),
                "provider": state.get("provider", "unknown"),
            }
            _QUERY_CACHE.move_to_end(query)
            if len(_QUERY_CACHE) > _MAX_CACHE_SIZE:
                _QUERY_CACHE.popitem(last=False)  # Evict oldest entry
    return {}

