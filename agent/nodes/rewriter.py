"""Stub — will be implemented in the next step."""
from agent.state import AgentState
def query_rewriter_node(state: AgentState) -> AgentState:
    return {"rewritten_query": state.get("query", "")}
