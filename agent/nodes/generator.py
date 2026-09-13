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


SYSTEM_PROMPT_BASE = """You are the official AI Academic Advisor and Campus Assistant for the college.
Your role is to converse naturally, helpfully, and professionally with students and faculty, delivering accurate, well-structured academic information.

STYLE & PRESENTATION GUIDELINES:
1. Conversational & Professional Flow:
   - Speak naturally like an attentive, knowledgeable academic advisor.
   - Start immediately with a clear, direct answer to the user's question without robotic disclaimers or meta-talk (do NOT say "Notice: Official college records are incomplete..." or "Based on the available documentation...").
   - Write cleanly with natural transitions.

2. Clean Visual Structure & Markdown:
   - Organize related details into thematic sections with clean Markdown headings (e.g., `### Laboratory Infrastructure`, `### Computational Facilities`, `### Research & Innovation`).
   - Leave a blank line before and after each heading.
   - Format bullet lists cleanly using standard bullet markers (`- `) with a space after each dash. Ensure sub-items and lists have proper line breaks rather than being squashed together.
   - When presenting structured course data, subject codes, credits, or regulations, format them into clean, well-aligned Markdown TABLES.
   - Bold key names, lab titles, tools, and technical terms to make the response scannable and visually appealing.

3. Strict Factual Grounding & Clean Text:
   - Base all statements SOLELY on the provided Context Blocks. Never speculate beyond what is documented.
   - Do NOT insert distracting in-text tags like [Source 1] or [Source 2] in the body.
   - Do NOT append a manual "Sources:" URL list at the end of your response, as verified sources are automatically parsed and displayed by the interface.

4. Silent Self-Verification (before writing your response):
   - Mentally verify every course code, credit count, faculty name, and regulation number against the Context Blocks.
   - If a specific fact (e.g., a course code or credit) does NOT appear in any Context Block, do NOT include it.
   - Do NOT mention this verification step in your response — just produce clean, grounded output.
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
            f"INCOMPLETE DOCUMENTATION NOTICE:\n"
            f"- Information in records is partial (Reason: {degraded_reason or 'Partial documentation'}).\n"
            f"- State ONLY the facts explicitly verified in the context blocks. Do NOT invent missing details.\n"
            f"- Do NOT output disclaimers or phrases like 'Notice: Official college records are incomplete...'. Present verified facts directly and cleanly."
        )

    if contradictions:
        contra_str = "; ".join(contradictions)
        instructions.append(
            f"DOCUMENTATION CONFLICT DETECTED:\n"
            f"- {contra_str}\n"
            f"- Present conflicting records transparently (e.g., 'One official document notes X, while another lists Y')."
        )

    if guard_feedback:
        instructions.append(
            f"GUARD REVISION FEEDBACK (Fix previous draft):\n"
            f"- {guard_feedback}\n"
            f"- Correct any unverified claims or course codes strictly."
        )

    special_section = ""
    if instructions:
        special_section = "\nSPECIAL INSTRUCTIONS:\n" + "\n".join(f"- {inst}" for inst in instructions) + "\n"

    prompt = (
        f"CONTEXT BLOCKS:\n"
        f"====================\n"
        f"{joined_context}\n"
        f"====================\n"
        f"{special_section}\n"
        f"Student/User Question: \"{query}\"\n\n"
        f"Please provide a well-structured response using clean markdown headings (### ), bullet points, and tables where suitable. Jump straight into the verified information without meta-disclaimers or manual source URLs."
    )
    return prompt


def _call_groq_generator(prompt: str) -> Tuple[str, str]:
    """Generates response via Groq with automatic key failover across bucket."""
    from groq import Groq
    keys = settings.GROQ_API_KEYS or ([settings.GROQ_API_KEY] if settings.GROQ_API_KEY else [])
    last_err = None
    for key in keys:
        try:
            client = Groq(api_key=key)
            response = client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_BASE},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=2048,
            )
            answer = response.choices[0].message.content or ""
            return answer, f"Groq ({settings.GROQ_MODEL})"
        except Exception as e:
            last_err = e
            continue
    raise last_err or RuntimeError("No working Groq API keys available")


def _call_gemini_generator(prompt: str) -> Tuple[str, str]:
    """Generates response via Google Gemini fallback with automatic key failover across bucket."""
    from google import genai
    keys = settings.GEMINI_API_KEYS or ([settings.GEMINI_API_KEY] if settings.GEMINI_API_KEY else [])
    combined = f"{SYSTEM_PROMPT_BASE}\n\n{prompt}"
    last_err = None
    for key in keys:
        try:
            client = genai.Client(api_key=key)
            resp = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=combined,
            )
            answer = resp.text or ""
            return answer, f"Gemini ({settings.GEMINI_MODEL})"
        except Exception as e:
            last_err = e
            continue
    raise last_err or RuntimeError("No working Gemini API keys available")


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
                logfire.warn("Groq generator failed: {err}", err=str(e), exc_info=True)
                print(f"  ⚠️ Groq generation failed: {e}. Switching to Gemini fallback...")

        # 2. Fallback: Gemini
        if not answer and settings.GEMINI_API_KEY:
            try:
                answer, provider = _call_gemini_generator(prompt)
            except Exception as e:
                logfire.error("Gemini generator fallback failed: {err}", err=str(e), exc_info=True)
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
