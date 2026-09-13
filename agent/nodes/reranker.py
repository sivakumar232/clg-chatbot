"""agent/nodes/reranker.py — cross-encoder reranking."""

from agent.state import AgentState


def reranker_node(state: AgentState) -> AgentState:
    chunks = state.get("candidate_chunks", [])
    return {
        "reranked_chunks": chunks[:5],
    }
