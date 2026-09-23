"""
agent/nodes/reformulator.py
───────────────────────────
Query Reformulator Node for Agentic RAG.

Responsibilities:
1. Analyzes the failure reason diagnosed by evidence_validator:
   - Missing sub-queries (Check 4)
   - Missing required entities (Check 5)
   - Low relevance score (Check 2)
2. Formulates a surgical, targeted expansion query using fast LLM or heuristic rules.
3. Increments retrieval_retry_count to guarantee bounded execution.
4. Emits reformulated_query for parallel_executor.
"""

import time
from typing import Any, Dict, List
from langsmith import traceable

from config import settings
from agent.state import AgentState
from agent.prompts import REFORMULATOR_SYSTEM_PROMPT



@traceable(name="Groq Reformulator Inference", run_type="llm")
def _call_groq_reformulator(original_query: str, failure_reason: str, missing_sq: List[str]) -> str | None:
    """Calls Groq to generate a targeted expansion query with full key-bucket failover."""
    keys = settings.GROQ_API_KEYS or ([settings.GROQ_API_KEY] if settings.GROQ_API_KEY else [])
    if not keys:
        return None

    prompt = (
        f"Original Query: \"{original_query}\"\n"
        f"Failure Reason: {failure_reason}\n"
    )
    if missing_sq:
        prompt += f"Missing Sub-Queries: {missing_sq}\n"
    prompt += "\nGenerate the revised search string:"

    from groq import Groq
    for key in keys:
        try:
            client = Groq(api_key=key)
            response = client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": REFORMULATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=60,
            )
            text = (response.choices[0].message.content or "").strip().replace('"', '')
            return text if text else None
        except Exception as e:
            print(f"Groq reformulator key failed: {e}")
            continue
    return None


@traceable(name="Query Reformulator Node", run_type="chain")
def reformulator_node(state: AgentState) -> AgentState:
    """
    LangGraph Reformulator node:
    Reads:  rewritten_query, query, degraded_reason, missing_sub_queries, intent, retrieval_retry_count
    Writes: reformulated_query, retrieval_retry_count
    """
    original_query = state.get("rewritten_query") or state.get("query", "")
    failure_reason = state.get("degraded_reason") or "Low relevance score"
    missing_sq = state.get("missing_sub_queries", [])
    current_retries = state.get("retrieval_retry_count", 0)
    new_retry_count = current_retries + 1

    t0 = time.time()
    reformulated = ""

    # Case 1: If specific sub-queries were missing, prioritize them directly
    if missing_sq:
        reformulated = " ".join(missing_sq)

    # Case 2: Use LLM reformulation if missing_sq is empty or to expand keywords
    if not reformulated:
        reformulated = _call_groq_reformulator(original_query, failure_reason, missing_sq) or ""

    # Case 3: Deterministic fallback if LLM returned empty
    if not reformulated:
        dept = state.get("intent", {}).get("department", "")
        reg = state.get("intent", {}).get("regulation", "")
        tokens = [original_query, dept, reg, "SRKR Engineering College"]
        reformulated = " ".join(t for t in tokens if t)

    elapsed = time.time() - t0

    print("=" * 60)
    print(f"  QUERY REFORMULATOR COMPLETE (Attempt {new_retry_count} in {elapsed:.2f}s)")
    print("=" * 60)
    print(f"  • Diagnosed Reason   : {failure_reason}")
    print(f"  • Original Query     : \"{original_query}\"")
    print(f"  • Reformulated Query : \"{reformulated}\"")
    print("=" * 60 + "\n")

    return {
        "reformulated_query":    reformulated,
        "retrieval_retry_count": new_retry_count,
    }
