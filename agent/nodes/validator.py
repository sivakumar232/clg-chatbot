"""agent/nodes/validator.py — evidence validation."""

from agent.state import AgentState, EvidenceStatus


def validator_node(state: AgentState) -> AgentState:
    chunks = state.get("reranked_chunks", [])
    retry_count = state.get("retrieval_retry_count", 0)
    max_retries = state.get("max_retrieval_retries", 1)

    # Basic heuristic check
    if len(chunks) > 0:
        return {"evidence_status": EvidenceStatus.SUFFICIENT, "degraded": False}
    
    if retry_count < max_retries:
        return {"evidence_status": EvidenceStatus.INSUFFICIENT_RETRY, "degraded": False}
    
    return {
        "evidence_status": EvidenceStatus.INSUFFICIENT_EXHAUSTED,
        "degraded": True,
        "degraded_reason": "No relevant documentation found in the knowledge base.",
    }
