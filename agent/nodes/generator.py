"""
agent/nodes/generator.py
────────────────────────
Answer Generator Node for Agentic RAG.

Responsibilities:
1. Formats reranked/pruned chunks into numbered context blocks with metadata.
2. Dynamically adapts prompt based on evidence state:
   - Standard Grounded Mode: Strict attribution, quotes regulations & course codes.
   - Degraded Mode: Injects explicit disclaimer about incomplete documentation.
   - Contradiction Notes: Injects instructions to explain conflicting document claims.
   - Guard Revision: Incorporates feedback if rejected by answer_guard.
3. Generates response via Groq with automatic fallback to Google Gemini.
4. Emits draft_answer for the answer_guard node.
"""

import time
from typing import Any, Dict, List, Tuple
import logfire

from config import settings
from agent.state import AgentState
from app.models import RetrievedChunk


SYSTEM_PROMPT_BASE = """You are the official AI Academic Advisor for SRKR Engineering College (Autonomous), Bhimavaram.
Your responsibility is to provide accurate, structured, and strictly grounded answers to student and faculty questions.

CORE RULES:
1. Base your answer SOLELY on the provided Context Blocks. Do not extrapolate, assume, or hallucinate facts not present in the text.
2. Quote specific course codes (e.g. CS3201), credit hours (L-T-P-C), regulations (R20, R23, R24), and committee details whenever present.
3. Structure your response clearly using markdown headings, bold terms, and bullet points.
4. When stating a fact, add an in-text source marker matching the context block, e.g. [Source 1] or [Source 2].
"""


def _build_generator_prompt(
    query: str,
    chunks: List[RetrievedChunk],
    degraded: bool,
    degraded_reason: str | None,
    contradictions: List[str],
    guard_feedback: str | None,
) -> str:
    """Assembles the complete prompt with context blocks and dynamic constraints."""
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        page = f" (Page {chunk.metadata.get('page') + 1})" if isinstance(chunk.metadata.get("page"), int) else ""
        block = f"--- [Context Block {i} | Source: {chunk.source}{page}] ---\n{chunk.text.strip()}\n"
        context_blocks.append(block)

    joined_context = "\n".join(context_blocks) if context_blocks else "No relevant context found."

    instructions = []

    if degraded:
        instructions.append(
            f"CRITICAL (INCOMPLETE DOCUMENTATION NOTICE):\n"
            f"- Official college records are incomplete for this query (Reason: {degraded_reason or 'Partial documentation'}).\n"
            f"- You MUST begin or conclude your answer with an explicit notice:\n"
            f"  'Notice: Official college records are incomplete regarding this query. The following details are what could be verified from available documentation:'\n"
            f"- State ONLY what is verified in the context blocks. Do NOT invent missing details."
        )

    if contradictions:
        contra_str = "; ".join(contradictions)
        instructions.append(
            f"DOCUMENTATION CONFLICT DETECTED:\n"
            f"- {contra_str}\n"
            f"- Present BOTH conflicting values transparently to the student (e.g. 'One official record states X, while another states Y')."
        )

    if guard_feedback:
        instructions.append(
            f"GUARD REVISION FEEDBACK (Fix previous draft):\n"
            f"- The previous draft was flagged: {guard_feedback}\n"
            f"- Make sure your revised answer strictly complies and eliminates ungrounded claims."
        )

    special_instructions = "\n\n".join(instructions)
    special_section = f"\n\nSPECIAL GUIDELINES:\n{special_instructions}\n" if special_instructions else ""

    prompt = (
        f"Context Information:\n"
        f"====================\n"
        f"{joined_context}\n"
        f"====================\n"
        f"{special_section}\n"
        f"Student/User Query: \"{query}\"\n\n"
        f"Please provide your official grounded answer based strictly on the context blocks above."
    )
    return prompt


def _call_groq_generator(prompt: str) -> Tuple[str, str]:
    """Generates response via Groq."""
    from groq import Groq
    client = Groq(api_key=settings.GROQ_API_KEY)
    response = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_BASE},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=896,
    )
    answer = response.choices[0].message.content or ""
    return answer, f"Groq ({settings.GROQ_MODEL})"


def _call_gemini_generator(prompt: str) -> Tuple[str, str]:
    """Generates response via Google Gemini fallback."""
    from google import genai
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    combined = f"{SYSTEM_PROMPT_BASE}\n\n{prompt}"
    resp = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=combined,
    )
    answer = resp.text or ""
    return answer, f"Gemini ({settings.GEMINI_MODEL})"


def generator_node(state: AgentState) -> AgentState:
    """
    LangGraph Generator node:
    Reads:  reranked_chunks, pruned_chunks, rewritten_query, query, degraded, degraded_reason, contradictions, guard_feedback
    Writes: draft_answer, provider
    """
    # Prefer pruned_chunks from validator Check 3 if available, else reranked_chunks
    chunks = state.get("pruned_chunks") or state.get("reranked_chunks", [])
    query = state.get("rewritten_query") or state.get("query", "")
    degraded = state.get("degraded", False)
    degraded_reason = state.get("degraded_reason")
    contradictions = state.get("contradictions", [])
    guard_feedback = state.get("guard_feedback")

    prompt = _build_generator_prompt(
        query=query,
        chunks=chunks,
        degraded=degraded,
        degraded_reason=degraded_reason,
        contradictions=contradictions,
        guard_feedback=guard_feedback,
    )

    with logfire.span("Answer Generator Node", chunks_used=len(chunks), degraded=degraded) as span:
        t0 = time.time()
        answer = ""
        provider = "Unknown"

        # 1. Primary: Groq
        if settings.GROQ_API_KEY:
            try:
                answer, provider = _call_groq_generator(prompt)
            except Exception as e:
                logfire.warn(f"Groq generator failed: {e}", exc_info=True)
                print(f"  ⚠️ Groq generation failed: {e}. Switching to Gemini fallback...")

        # 2. Fallback: Gemini
        if not answer and settings.GEMINI_API_KEY:
            try:
                answer, provider = _call_gemini_generator(prompt)
            except Exception as e:
                logfire.error(f"Gemini generator fallback failed: {e}", exc_info=True)
                raise RuntimeError(f"All LLM generation providers failed: {e}")

        elapsed = time.time() - t0
        span.set_attribute("provider", provider)
        span.set_attribute("latency_seconds", round(elapsed, 3))
        span.set_attribute("answer_length", len(answer))

        print("=" * 60)
        print(f"  GENERATOR COMPLETE ({provider} in {elapsed:.2f}s)")
        print("=" * 60)
        print(f"  • Chunks Used     : {len(chunks)}")
        print(f"  • Degraded Mode   : {degraded}")
        print(f"  • Draft Answer Preview:")
        preview = answer.strip().replace("\n", " ")[:160]
        print(f"    \"{preview}...\"\n")
        print("=" * 60 + "\n")

        return {
            "draft_answer": answer,
            "provider":     provider,
        }
