"""agent/nodes/guard.py — answer grounding & faithfulness guard."""

from agent.state import AgentState, GuardStatus


def guard_node(state: AgentState) -> AgentState:
    retry_count = state.get("guard_retry_count", 0)
    max_retries = state.get("max_guard_retries", 1)

    # Stub: default to grounded
    return {
        "guard_status": GuardStatus.GROUNDED,
        "guard_feedback": None,
    }
