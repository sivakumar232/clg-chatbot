"""agent/nodes/reformulator.py — query reformulator for targeted retrieval."""

from agent.state import AgentState


def reformulator_node(state: AgentState) -> AgentState:
    retries = state.get("retrieval_retry_count", 0) + 1
    return {
        "retrieval_retry_count": retries,
        "reformulated_query": state.get("rewritten_query") or state.get("query", ""),
    }
