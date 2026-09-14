"""
agent/nodes/planner.py
──────────────────────
Unified Query Planner node for Agentic RAG.

Performs 4 tasks in a single fast LLM call:
1. Coreference resolution (pronouns rewritten with chat history)
2. Routing classification (direct vs needs_retrieval)
3. Structural typing (single_query, sub_query, multi_hop_query)
4. Sub-query decomposition + payload filter generation
"""

import json
import time
from typing import Any, Dict, List, Optional
from langsmith import traceable

from config import settings
from agent.state import AgentState, RouteType, QueryType


PLANNER_SYSTEM_PROMPT = """You are the master Query Planner for the College AI Academic Assistant.
Analyze the user's latest query along with any chat history, and output a strictly valid JSON execution plan.

TASKS:
1. Coreference Resolution & Rewrite:
   - If the user uses pronouns or references previous context (e.g., "what about his cabin?", "give me its syllabus"), rewrite it into a self-contained question using the chat history.
   - If already self-contained, keep it unchanged.

2. Route Classification:
   - "direct":
     * Conversational greetings (e.g. "hi", "hello", "how are you"), compliments.
     * Queries totally out-of-scope of college academics (e.g., general world history, cooking, cricket, politics).
     * Privacy & Security Restrictions: Requests asking for personal phone numbers, mobile numbers, WhatsApp numbers, residential/home addresses, personal email IDs, salaries, or private personal data of faculty, staff, or students. Set intent category to "privacy_restriction".
   - "needs_retrieval": Legitimate queries seeking academic college information (syllabi, courses, departments, regulations like R19/R20/R23/R24, faculty designations, official department offices, administration, fees, exams, placements, admissions, campus facilities, clubs).

3. Query Type:
   - "single_query": Focused question on one entity or topic.
   - "sub_query": Query requiring multiple sub-searches (e.g. comparing R20 vs R23, or syllabus + lab curriculum).
   - "multi_hop_query": Aggregate/broad query across multiple departments/entities (e.g., "list all HODs", "all engineering branches").

4. Intent:
   - Extract the user's goal: category (e.g., "syllabus", "faculty", "placements", "admin", "exam", "general"), department (e.g. "CSE", "ECE", "AIDS", "MECH", "CIVIL", "IT", "CSBS", "EEE" if mentioned or inferred), regulation ("R20", "R23", "R24" if mentioned), is_aggregate (true/false), and specific entities (e.g. course codes like "CS3201", faculty names).

5. Sub-queries:
   - If route == "direct", sub_queries MUST be [].
   - If route == "needs_retrieval", produce 1 to 4 distinct, keyword-rich search queries optimized for hybrid search (dense + lexical). Avoid conversational filler words.
   - Optionally attach a metadata_filter dict (e.g. {"department": "CSE"} or {"regulation": "R23"}) only when explicitly confident; otherwise set to null.

Output MUST be a JSON object with this exact schema:
{
  "rewritten_query": "string",
  "route": "direct" | "needs_retrieval",
  "query_type": "single_query" | "sub_query" | "multi_hop_query",
  "intent": {
    "category": "string",
    "department": "string or null",
    "regulation": "string or null",
    "is_aggregate": boolean,
    "entities": ["string"]
  },
  "sub_queries": [
    {
      "query": "string",
      "metadata_filter": {"department": "CSE"} or null
    }
  ]
}
"""


def _format_history_context(chat_history: List[Dict[str, str]]) -> str:
    """
    Formats conversation history for the planner prompt.
    - Last 2 turns: kept verbatim (needed for coreference resolution).
    - Older turns: compressed into a compact summary to save tokens.
    """
    if not chat_history:
        return "None"

    recent = chat_history[-2:]
    older  = chat_history[:-2]

    parts: List[str] = []

    if older:
        # Extract topic keywords from older turns for a compact summary
        topics = []
        for msg in older:
            content = msg.get("content", "").strip()
            if content and msg.get("role") == "user":
                # Take first 60 chars of each older user message as topic hint
                topics.append(content[:60].rstrip() + ("..." if len(content) > 60 else ""))
        if topics:
            parts.append(f"[Earlier context — user asked about: {'; '.join(topics)}]")

    for msg in recent:
        role    = msg.get("role", "user").capitalize()
        content = msg.get("content", "").strip()
        parts.append(f"{role}: {content}")

    return "\n".join(parts)


@traceable(name="Groq Planner LLM Inference", run_type="llm")
def _call_groq_planner(prompt_text: str) -> Optional[Dict[str, Any]]:
    """Calls Groq with strict JSON output format and key failover across bucket."""
    keys = settings.GROQ_API_KEYS or ([settings.GROQ_API_KEY] if settings.GROQ_API_KEY else [])
    if not keys:
        return None

    from groq import Groq
    for key in keys:
        try:
            client = Groq(api_key=key)
            response = client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=512,
            )
            content = response.choices[0].message.content or ""
            parsed = json.loads(content)
            return parsed

        except Exception as e:
            print(f"Groq planner key failed: {e}")
            continue
    return None


@traceable(name="Gemini Planner Fallback Inference", run_type="llm")
def _call_gemini_planner(prompt_text: str) -> Optional[Dict[str, Any]]:
    """Fallback to Google Gemini for JSON planning with key failover across bucket."""
    keys = settings.GEMINI_API_KEYS or ([settings.GEMINI_API_KEY] if settings.GEMINI_API_KEY else [])
    if not keys:
        return None

    from google import genai
    for key in keys:
        try:
            client = genai.Client(api_key=key)
            combined_prompt = f"{PLANNER_SYSTEM_PROMPT}\n\nUser Input:\n{prompt_text}\n\nRespond with ONLY valid JSON."
            resp = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=combined_prompt,
            )
            raw_text = (resp.text or "").strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]

            parsed = json.loads(raw_text.strip())
            return parsed

        except Exception as e:
            print(f"Gemini planner key failed: {e}")
            continue
    return None


@traceable(name="Planner Node", run_type="chain")
def planner_node(state: AgentState) -> AgentState:
    """
    LangGraph Planner node:
    Reads:  query, chat_history
    Writes: rewritten_query, route, query_type, intent, sub_queries
    """
    raw_query = state.get("query", "").strip()
    chat_history = state.get("chat_history", [])

    history_str = _format_history_context(chat_history)
    prompt_payload = (
        f"Conversation History:\n{history_str}\n\n"
        f"Latest User Query: \"{raw_query}\"\n\n"
        f"Generate the JSON execution plan."
    )

    t0 = time.time()
    plan_dict = _call_groq_planner(prompt_payload)
    provider_used = "Groq"

    if plan_dict is None:
        plan_dict = _call_gemini_planner(prompt_payload)
        provider_used = "Gemini"

    elapsed = time.time() - t0

    # ── Fallback Resiliency ──────────────────────────────────────────────────
    if not plan_dict or not isinstance(plan_dict, dict):
        print(f"Planner failed to produce valid JSON for query: '{raw_query}'. Using safe fallback.")
        return {
            "rewritten_query": raw_query,
            "route": RouteType.NEEDS_RETRIEVAL,
            "query_type": QueryType.SINGLE,
            "intent": {"category": "general", "fallback": True},
            "sub_queries": [{"query": raw_query, "metadata_filter": None}],
        }

    # ── Parse and Validate Fields ────────────────────────────────────────────
    rewritten_query = plan_dict.get("rewritten_query") or raw_query
    
    # Route
    raw_route = str(plan_dict.get("route", "")).lower()
    if "direct" in raw_route:
        route = RouteType.DIRECT
    else:
        route = RouteType.NEEDS_RETRIEVAL

    # Query Type
    raw_type = str(plan_dict.get("query_type", "")).lower()
    if "multi_hop" in raw_type:
        query_type = QueryType.MULTI_HOP
    elif "sub_query" in raw_type:
        query_type = QueryType.SUB
    else:
        query_type = QueryType.SINGLE

    intent = plan_dict.get("intent") or {}
    
    # Sub-queries
    sub_queries = []
    if route == RouteType.NEEDS_RETRIEVAL:
        raw_subs = plan_dict.get("sub_queries") or []
        for item in raw_subs[:4]:  # Bounded to max 4 sub-queries
            if isinstance(item, dict) and item.get("query"):
                sub_queries.append({
                    "query": str(item["query"]).strip(),
                    "metadata_filter": item.get("metadata_filter"),
                })
        
        # Ensure at least 1 sub-query exists if needs_retrieval
        if not sub_queries:
            sub_queries.append({
                "query": rewritten_query,
                "metadata_filter": None,
            })

    print("=" * 60)
    print(f"  PLANNER COMPLETE ({provider_used} in {elapsed:.2f}s)")
    print("=" * 60)
    print(f"  • Raw Query       : \"{raw_query}\"")
    print(f"  • Rewritten Query : \"{rewritten_query}\"")
    print(f"  • Route Decision  : {route.value}")
    print(f"  • Query Type      : {query_type.value}")
    print(f"  • Intent Category : {intent.get('category')}")
    print(f"  • Sub-Queries ({len(sub_queries)}):")
    for idx, sq in enumerate(sub_queries, 1):
        flt_str = f" | Filter: {sq['metadata_filter']}" if sq.get("metadata_filter") else ""
        print(f"    [{idx}] \"{sq['query']}\"{flt_str}")
    print("=" * 60 + "\n")

    return {
        "rewritten_query": rewritten_query,
        "route": route,
        "query_type": query_type,
        "intent": intent,
        "sub_queries": sub_queries,
    }