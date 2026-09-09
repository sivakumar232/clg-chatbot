import os
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure project root is importable (for config and Ingestion)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from .retriever import RetrievedChunk


SYSTEM_PROMPT = """You are an official AI academic advisor for SRKR Engineering College (Autonomous), Bhimavaram.
Your responsibility is to provide accurate, helpful, and strictly grounded answers to student and faculty queries.

RULES:
1. Base your answer SOLELY on the provided context sections. Do not extrapolate, assume, or hallucinate facts not present in the text.
2. If the provided context does not contain enough information to answer the question, clearly state:
   "I could not find specific information about that in the official college documentation."
3. Quote specific course codes, credit hours, regulations (R20, R23, etc.), or committee details whenever relevant.
4. Structure your response clearly using bullet points and Markdown formatting.
5. List the sources you referenced at the end under a '### Sources' heading.
"""


import time

class LLMGenerator:
    """
    Handles prompt assembly and LLM response generation with automatic
    fallback from Groq to Google Gemini.
    """

    def __init__(self):
        self.groq_model = getattr(settings, "GROQ_MODEL", "qwen/qwen3.8-27b")
        self.gemini_model = getattr(settings, "GEMINI_MODEL", "gemini-flash-latest")

    def build_prompt(self, query: str, chunks: List[RetrievedChunk]) -> str:
        """
        Assembles context from retrieved chunks and constructs the prompt.
        """
        print("=" * 60)
        print("  STEP 4: Context & Prompt Assembly (After Re-Ranking)")
        print("=" * 60)

        context_blocks = []
        for i, chunk in enumerate(chunks, start=1):
            block = f"--- [Context Block {i} | Source: {chunk.source}] ---\n{chunk.text.strip()}\n"
            context_blocks.append(block)

        joined_context = "\n".join(context_blocks)
        total_chars = len(joined_context)

        print(f"  • Assembled {len(chunks)} reranked context blocks ({total_chars:,} characters).")
        print(f"  • Grounding instructions and system prompt applied.\n")

        user_content = (
            f"Context Information:\n"
            f"====================\n"
            f"{joined_context}\n"
            f"====================\n\n"
            f"Student Question: {query}\n\n"
            f"Please answer the question accurately based only on the above context."
        )
        return user_content

    def generate(self, query: str, chunks: List[RetrievedChunk]) -> Tuple[str, str]:
        """
        Generates response using Groq with fallback to Google Gemini.
        Returns: (answer_text, provider_used)
        """
        prompt = self.build_prompt(query, chunks)

        print("=" * 60)
        print("  STEP 5: Generation with Fallback (Groq ──► Gemini)")
        print("=" * 60)

        # 1. Attempt Groq
        if settings.GROQ_API_KEY:
            try:
                print(f"  • Attempting Primary: Groq ({self.groq_model})...")
                from groq import Groq

                groq_client = Groq(api_key=settings.GROQ_API_KEY)
                response = groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=1024,
                )
                answer = response.choices[0].message.content or ""
                print(f"  ✓ Answer generated successfully via Groq ({self.groq_model}).\n")
                return answer, f"Groq ({self.groq_model})"

            except Exception as e:
                print(f"  ⚠️ Groq generation failed: {e}")
                print(f"  • Switching to Fallback: Google Gemini ({self.gemini_model})...")

        # 2. Fallback to Gemini with Retry & Model Fallback
        if settings.GEMINI_API_KEY:
            from google import genai

            gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
            combined_contents = f"{SYSTEM_PROMPT}\n\n{prompt}"
            fallback_models = [self.gemini_model, "gemini-2.5-flash-lite", "gemini-2.5-flash"]

            for model_name in fallback_models:
                for attempt in range(3):
                    try:
                        resp = gemini_client.models.generate_content(
                            model=model_name,
                            contents=combined_contents,
                        )
                        answer = resp.text or ""
                        print(f"  ✓ Answer generated successfully via Gemini ({model_name}).\n")
                        return answer, f"Gemini ({model_name})"

                    except Exception as e:
                        wait = 2 ** (attempt + 1)
                        if "503" in str(e) or "UNAVAILABLE" in str(e) or "429" in str(e):
                            print(f"  ⚠️ Gemini ({model_name}) busy (attempt {attempt + 1}/3). Retrying in {wait}s...")
                            time.sleep(wait)
                        else:
                            print(f"  ⚠️ Gemini ({model_name}) failed: {e}")
                            break

            raise RuntimeError("Gemini fallback failed across all retry attempts and fallback models.")

        raise RuntimeError("No LLM API keys (GROQ_API_KEY or GEMINI_API_KEY) configured.")

