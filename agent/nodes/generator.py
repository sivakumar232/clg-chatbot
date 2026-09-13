"""agent/nodes/generator.py — answer generation."""

from agent.state import AgentState


def generator_node(state: AgentState) -> AgentState:
    return {
        "draft_answer": "This is a draft generated answer based on retrieved evidence.",
        "provider": "Groq",
    }
