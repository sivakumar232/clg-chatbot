"""
agent/nodes/input_rail.py
─────────────────────────
Dedicated Input Rail Node for Agentic RAG using NeMo Guardrails.

Responsibilities:
1. Inspects the raw user query before planning or retrieval begins.
2. Checks for prompt injections, jailbreak attempts, profanity, and disallowed intents.
3. If blocked:
   - Marks input_rail_status = "blocked"
   - Routes directly to responder with the safety refusal message.
   - Bypasses Planner, Retrieval, and Generation completely (0 token waste).
4. If allowed:
   - Marks input_rail_status = "allowed"
   - Proceeds seamlessly to the Planner node.
"""

import time
from typing import Any, Dict
from langsmith import traceable

from agent.state import AgentState, RouteType, QueryType


@traceable(name="Input Rail Node", run_type="chain")
def input_rail_node(state: AgentState) -> AgentState:
    """
    LangGraph Input Rail node:
    Reads:  query
    Writes: input_rail_status, (route, intent, answer, rewritten_query if blocked)
    """
    raw_query = state.get("query", "").strip()
    t0 = time.time()

    try:
        from guardrails import get_guardrails_service
        guard_service = get_guardrails_service()
        input_guard = guard_service.check_input(raw_query)
        elapsed = time.time() - t0

        if not input_guard.allowed:
            refusal = input_guard.refusal_message or (
                "I cannot fulfill this request as it violates academic safety and usage guidelines. "
                "Please ask a question related to college academics, regulations, faculty, or campus services."
            )
            print("=" * 60)
            print(f"  🛑 INPUT RAIL BLOCKED ({elapsed:.2f}s)")
            print("=" * 60)
            print(f"  • Query   : \"{raw_query}\"")
            print(f"  • Action  : {input_guard.action}")
            print(f"  • Reason  : {input_guard.reason}")
            print(f"  • Refusal : {refusal}")
            print("=" * 60 + "\n")

            return {
                "input_rail_status": "blocked",
                "route": RouteType.DIRECT,
                "query_type": QueryType.SINGLE,
                "rewritten_query": raw_query,
                "intent": {
                    "category": input_guard.action,
                    "refusal_message": refusal,
                    "reason": input_guard.reason,
                },
                "answer": refusal,
                "sub_queries": [],
            }

        print("=" * 60)
        print(f"  ✓ INPUT RAIL ALLOWED ({elapsed:.2f}s)")
        print("=" * 60)
        print(f"  • Query   : \"{raw_query}\"")
        print(f"  • Status  : Passed all safety and policy rails")
        print("=" * 60 + "\n")

        return {
            "input_rail_status": "allowed",
        }

    except Exception as exc:
        print(f"NeMo Guardrails input check warning (allowing query): {exc}")
        return {
            "input_rail_status": "allowed",
        }
