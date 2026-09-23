"""
backend/app/api/endpoints/chat.py
─────────────────────────────────
Streaming and standard chat endpoints for Agentic RAG.
Uses LangGraph app.stream(stream_mode="updates") to emit real-time
Server-Sent Events (SSE) as each node in the graph executes.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
import threading
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

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


async def _stream_agent_execution(
    query: str,
    chat_history: list,
    raw_request: Request | None = None,
) -> AsyncGenerator[str, None]:
    """
    Async generator that executes the LangGraph state machine and yields
    granular progress updates for the frontend UI.
    Streams node events in real-time as they finish and immediately halts
    execution if client disconnects or hits Stop.
    """
    initial_state: AgentState = {
        "query":                 query,
        "chat_history":          chat_history,
        "retrieval_retry_count": 0,
        "max_retrieval_retries": 2,
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
    final_provider = "Campus Advisor"
    final_degraded = False
    final_clarification = None

    stop_event = threading.Event()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _worker():
        try:
            for step in app.stream(initial_state, stream_mode="updates"):
                if stop_event.is_set():
                    logger.info("LangGraph stream iteration stopped via cancellation token.")
                    break
                loop.call_soon_threadsafe(queue.put_nowait, ("step", step))
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
        except Exception as exc:
            logger.exception("Error in LangGraph worker thread: %s", exc)
            loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))

    worker_thread = threading.Thread(target=_worker, daemon=True)
    worker_thread.start()

    async def _disconnect_watcher():
        if raw_request is None:
            return
        while not stop_event.is_set():
            try:
                if await raw_request.is_disconnected():
                    logger.info("Client disconnect detected by watcher. Halting agent thread.")
                    stop_event.set()
                    break
            except Exception:
                pass
            await asyncio.sleep(0.2)

    watcher_task = asyncio.create_task(_disconnect_watcher())

    try:
        while True:
            if stop_event.is_set():
                break

            if raw_request is not None and await raw_request.is_disconnected():
                logger.info("Client disconnected. Halting agent execution.")
                stop_event.set()
                break

            try:
                item_type, data = await asyncio.wait_for(queue.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue

            if item_type == "done":
                break
            elif item_type == "error":
                raise data
            elif item_type == "step":
                step = data
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

                    elif node_name == "clarifier":
                        clar_payload = node_output.get("clarification")
                        yield _format_sse({
                            "type": "step",
                            "node": "clarifier",
                            "label": "Disambiguating query with quick-reply options",
                            "clarification": clar_payload,
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
                            "label": "NeMo Guardrails: Output verified & PII sanitized" if g_status == "grounded" else "NeMo Guardrails: Ungrounded claim flagged",
                            "guard_status": g_status,
                        })

                    elif node_name == "responder":
                        final_answer = node_output.get("answer", "")
                        final_sources = node_output.get("sources", [])
                        final_provider = node_output.get("provider", "Campus Advisor")
                        final_clarification = node_output.get("clarification")
                        final_degraded = node_output.get("degraded", False)

        # Emit completion payload with verified answer, sources, and clarification
        if not stop_event.is_set():
            yield _format_sse({
                "type": "done",
                "answer": final_answer,
                "sources": final_sources,
                "provider": final_provider,
                "clarification": final_clarification,
                "degraded": final_degraded,
            })

    except Exception as e:
        if not stop_event.is_set():
            logger.exception("Streaming error in LangGraph: %s", str(e))
            yield _format_sse({
                "type": "error",
                "message": f"An error occurred during pipeline execution: {str(e)}",
            })
    finally:
        stop_event.set()
        watcher_task.cancel()


@router.post("/stream")
async def chat_stream(request: ChatRequest, raw_request: Request):
    """
    Server-Sent Events (SSE) endpoint:
    Streams step-by-step agentic transitions and the final grounded response.
    Supports cancellation when client disconnects or aborts.
    """
    history = [{"role": m.role, "content": m.content} for m in (request.chat_history or [])]
    return StreamingResponse(
        _stream_agent_execution(query=request.query, chat_history=history, raw_request=raw_request),
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
            clarification=result.get("clarification"),
            cache_hit=result.get("cache_hit", False),
            degraded=result.get("degraded", False),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
