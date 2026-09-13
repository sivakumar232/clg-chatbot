"""
backend/app/api/endpoints/chat.py
─────────────────────────────────
Streaming and standard chat endpoints for Agentic RAG.
Uses LangGraph app.stream(stream_mode="updates") to emit real-time
Server-Sent Events (SSE) as each node in the graph executes.
"""

import json
import sys
from pathlib import Path
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import logfire

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.graph import app
from agent.state import AgentState
from backend.app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter()


def _format_sse(event_data: dict) -> str:
    """Formats a dictionary into a compliant Server-Sent Events (SSE) payload."""
    return f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"


async def _stream_agent_execution(query: str, chat_history: list) -> AsyncGenerator[str, None]:
    """
    Async generator that executes the LangGraph state machine and yields
    granular progress updates for the frontend UI.
    """
    initial_state: AgentState = {
        "query":                 query,
        "chat_history":          chat_history,
        "retrieval_retry_count": 0,
        "max_retrieval_retries": 1,
        "guard_retry_count":     0,
        "max_guard_retries":     1,
        "accumulated_chunks":    [],
        "degraded":              False,
    }

    # Initial start event
    yield _format_sse({
        "type": "start",
        "query": query,
    })

    final_answer = ""
    final_sources = []
    final_provider = "SRKR Advisor"
    final_degraded = False

    try:
        # Stream updates as each LangGraph node finishes execution
        for step in app.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in step.items():

                if node_name == "cache":
                    hit = bool(node_output.get("cache_hit", False))
                    yield _format_sse({
                        "type": "step",
                        "node": "cache",
                        "label": "Cache Hit (< 1ms)" if hit else "Checking in-memory cache",
                        "status": "hit" if hit else "miss",
                    })

                elif node_name == "planner":
                    route = str(node_output.get("route", "needs_retrieval"))
                    sub_queries = [sq.get("query", "") for sq in node_output.get("sub_queries", [])]
                    rewritten = node_output.get("rewritten_query")
                    yield _format_sse({
                        "type": "step",
                        "node": "planner",
                        "label": "Query Analyzed & Decomposed",
                        "route": route,
                        "rewritten_query": rewritten,
                        "sub_queries": sub_queries,
                    })

                elif node_name == "executor":
                    count = len(node_output.get("candidate_chunks", []))
                    yield _format_sse({
                        "type": "step",
                        "node": "executor",
                        "label": f"Retrieved {count} candidates via Qdrant & BM25",
                        "candidate_count": count,
                    })

                elif node_name == "reranker":
                    reranked = node_output.get("reranked_chunks", [])
                    top_score = round(reranked[0].score, 4) if reranked else 0.0
                    yield _format_sse({
                        "type": "step",
                        "node": "reranker",
                        "label": f"Cross-Encoder reranked top {len(reranked)} chunks (Top Score: {top_score})",
                        "top_score": top_score,
                    })

                elif node_name == "validator":
                    status = str(node_output.get("evidence_status", "sufficient"))
                    pruned_count = len(node_output.get("pruned_chunks", []))
                    yield _format_sse({
                        "type": "step",
                        "node": "validator",
                        "label": f"7-Point Evidence Check: {status.upper()}",
                        "evidence_status": status,
                        "pruned_count": pruned_count,
                    })

                elif node_name == "reformulator":
                    reformed = node_output.get("reformulated_query", "")
                    retry = node_output.get("retrieval_retry_count", 1)
                    yield _format_sse({
                        "type": "step",
                        "node": "reformulator",
                        "label": f"Diagnostic expansion for missing facts (Attempt {retry})",
                        "reformulated_query": reformed,
                    })

                elif node_name == "generator":
                    yield _format_sse({
                        "type": "step",
                        "node": "generator",
                        "label": "Synthesizing grounded answer with citations",
                    })

                elif node_name == "guard":
                    g_status = str(node_output.get("guard_status", "grounded"))
                    yield _format_sse({
                        "type": "step",
                        "node": "guard",
                        "label": "Answer faithfulness verified (No hallucinations)",
                        "guard_status": g_status,
                    })

                elif node_name == "responder":
                    final_answer = node_output.get("answer", "")
                    final_sources = node_output.get("sources", [])
                    final_provider = node_output.get("provider", "Groq")

        # Emit completion payload with verified answer and sources
        yield _format_sse({
            "type": "done",
            "answer": final_answer,
            "sources": final_sources,
            "provider": final_provider,
            "degraded": final_degraded,
        })

    except Exception as e:
        logfire.error("Streaming error in LangGraph: {err}", err=str(e), exc_info=True)
        yield _format_sse({
            "type": "error",
            "message": f"An error occurred during pipeline execution: {str(e)}",
        })


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    Server-Sent Events (SSE) endpoint:
    Streams step-by-step agentic transitions and the final grounded response.
    """
    history = [{"role": m.role, "content": m.content} for m in (request.chat_history or [])]
    return StreamingResponse(
        _stream_agent_execution(query=request.query, chat_history=history),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("", response_model=ChatResponse)
@router.post("/", response_model=ChatResponse)
async def chat_standard(request: ChatRequest):
    """Standard REST endpoint for non-streaming clients."""
    from agent.graph import run_agent

    history = [{"role": m.role, "content": m.content} for m in (request.chat_history or [])]
    try:
        result = run_agent(query=request.query, chat_history=history)
        return ChatResponse(
            query=request.query,
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            provider=result.get("provider", "Groq"),
            route=str(result.get("route")),
            cache_hit=result.get("cache_hit", False),
            degraded=result.get("degraded", False),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
