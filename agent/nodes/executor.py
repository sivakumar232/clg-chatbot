"""agent/nodes/executor.py — parallel executor with hybrid search + fusion."""

from agent.state import AgentState


def executor_node(state: AgentState) -> AgentState:
    # Stub: will run Qdrant + BM25 + RRF and merge with accumulated_chunks
    return {
        "candidate_chunks": [],
        "accumulated_chunks": state.get("accumulated_chunks", []),
    }
