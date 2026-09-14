"""
agent/graph.py
──────────────
Compiles the LangGraph StateGraph matching the updated Agentic RAG architecture:

    START
      │
      ▼
    cache ─────────── [hit] ───────────────────────────► responder ──► cachewrite ──► END
      │                                                     ▲
    [miss]                                                  │
      ▼                                                     │
    planner ───────── [direct] ─────────────────────────────┤
      │                                                     │
    [needs retrieval]                                       │
      ▼                                                     │
    executor ◄──────────────────┐                           │
      │                         │                           │
      ▼                         │                           │
    reranker                    │                           │
      │                         │                           │
      ▼                         │                           │
    validator ── [retry left] ──┴──► reformulator           │
      │                                                     │
    [sufficient / exhausted]                                │
      ▼                                                     │
    generator ◄─────────────────┐                           │
      │                         │                           │
      ▼                         │                           │
    guard ────── [retry left] ──┴                           │
      │                                                     │
    [grounded / exhausted] ─────────────────────────────────┘
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langgraph.graph import StateGraph, END

from agent.state import AgentState, RouteType, EvidenceStatus, GuardStatus
from agent.nodes import (
    cache_node,
    cache_write_node,
    planner_node,
    executor_node,
    reranker_node,
    validator_node,
    reformulator_node,
    generator_node,
    guard_node,
    responder_node,
)


# ─────────────────────────────────────────────────────────────────────────────
# Routing / Conditional Edge Functions
# ─────────────────────────────────────────────────────────────────────────────

def route_after_cache(state: AgentState) -> str:
    """Routes based on cache lookup result."""
    if state.get("cache_hit", False):
        return "responder"
    return "planner"


def route_after_planner(state: AgentState) -> str:
    """Routes based on query intent classification."""
    route = state.get("route", RouteType.NEEDS_RETRIEVAL)
    if route == RouteType.DIRECT:
        return "responder"
    return "executor"


def route_after_validator(state: AgentState) -> str:
    """Routes based on evidence sufficiency."""
    status = state.get("evidence_status", EvidenceStatus.SUFFICIENT)
    if status == EvidenceStatus.INSUFFICIENT_RETRY:
        return "reformulator"
    # Both SUFFICIENT and INSUFFICIENT_EXHAUSTED proceed to generator
    return "generator"


def route_after_guard(state: AgentState) -> str:
    """Routes based on answer grounding / faithfulness."""
    status = state.get("guard_status", GuardStatus.GROUNDED)
    if status == GuardStatus.UNGROUNDED_RETRY:
        return "generator"
    # Both GROUNDED and UNGROUNDED_EXHAUSTED proceed to responder
    return "responder"


# ─────────────────────────────────────────────────────────────────────────────
# Graph Construction
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Constructs and compiles the full LangGraph state machine.
    """
    graph = StateGraph(AgentState)

    # ── Register all nodes ────────────────────────────────────────────────────
    graph.add_node("cache",        cache_node)
    graph.add_node("planner",      planner_node)
    graph.add_node("executor",     executor_node)
    graph.add_node("reranker",     reranker_node)
    graph.add_node("validator",    validator_node)
    graph.add_node("reformulator", reformulator_node)
    graph.add_node("generator",    generator_node)
    graph.add_node("guard",        guard_node)
    graph.add_node("responder",    responder_node)
    graph.add_node("cachewrite",   cache_write_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("cache")

    # ── Conditional edges from cache ──────────────────────────────────────────
    graph.add_conditional_edges(
        "cache",
        route_after_cache,
        {
            "responder": "responder",
            "planner":   "planner",
        },
    )

    # ── Conditional edges from planner ────────────────────────────────────────
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "responder": "responder",
            "executor":  "executor",
        },
    )

    # ── Retrieval loop ────────────────────────────────────────────────────────
    graph.add_edge("executor",  "reranker")
    graph.add_edge("reranker",  "validator")

    graph.add_conditional_edges(
        "validator",
        route_after_validator,
        {
            "generator":    "generator",
            "reformulator": "reformulator",
        },
    )

    # Retry path back to executor
    graph.add_edge("reformulator", "executor")

    # ── Generation & Guard loop ───────────────────────────────────────────────
    graph.add_edge("generator", "guard")

    graph.add_conditional_edges(
        "guard",
        route_after_guard,
        {
            "responder": "responder",
            "generator": "generator",
        },
    )

    # ── Termination path ──────────────────────────────────────────────────────
    graph.add_edge("responder",  "cachewrite")
    graph.add_edge("cachewrite", END)

    return graph


# ─────────────────────────────────────────────────────────────────────────────
# Compiled App
# ─────────────────────────────────────────────────────────────────────────────

_graph = build_graph()
app    = _graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Public Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def get_mermaid_diagram() -> str:
    """
    Returns Mermaid markdown string of the compiled LangGraph.
    """
    return app.get_graph().draw_mermaid()


def run_agent(
    query:        str,
    chat_history: list | None = None,
) -> AgentState:
    """
    Executes the agentic RAG graph with initial state.
    """
    initial_state: AgentState = {
        "query":                 query,
        "chat_history":          chat_history or [],
        "retrieval_retry_count": 0,
        "max_retrieval_retries": 2,
        "guard_retry_count":     0,
        "max_guard_retries":     1,
        "accumulated_chunks":    [],
        "degraded":              False,
    }
    return app.invoke(initial_state)


if __name__ == "__main__":
    if "--mermaid" in sys.argv or "-m" in sys.argv:
        print("\n" + "=" * 60)
        print("  LANGGRAPH MERMAID DIAGRAM")
        print("=" * 60 + "\n")
        diagram = get_mermaid_diagram()
        print(diagram)
        print("\n" + "=" * 60)
        print("  Tip: Copy the block above into https://mermaid.live")
        print("=" * 60 + "\n")
    else:
        sample_query = "What is the CSE syllabus for R23?"
        print(f"\nRunning test execution for query: '{sample_query}'")
        res = run_agent(sample_query)
        print("\nExecution Result:")
        print(f"  • Route:    {res.get('route')}")
        print(f"  • Answer:   {res.get('answer')}")
        print(f"  • Provider: {res.get('provider')}")
        print(f"  • Degraded: {res.get('degraded')}")
        print("\nTip: Run with `uv run python agent/graph.py --mermaid` to view the Mermaid chart.")
