"""agent/nodes/planner.py — rewrite, classify, and decompose in one unified pass."""

from agent.state import AgentState, RouteType, QueryType


def planner_node(state: AgentState) -> AgentState:
    query = state.get("query", "")
    # Default stub behavior: passes through query and marks as needs_retrieval
    return {
        "rewritten_query": query,
        "route": RouteType.NEEDS_RETRIEVAL,
        "query_type": QueryType.SINGLE,
        "intent": {},
        "sub_queries": [{"query": query, "metadata_filter": None}],
    }
