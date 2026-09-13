"""
agent/test_suite.py
───────────────────
End-to-End Test Suite for Agentic RAG.

Tests all major production paths:
1. Academic Syllabus Query (Full Pipeline)
2. Faculty / Administrative Query
3. Multi-Turn Pronoun Resolution
4. Out-of-Scope Polite Rejection (0 retrieval cost)
5. Conversational Greeting (0 retrieval cost)
6. Semantic/LRU Cache Hit (< 1ms delivery)
"""

import sys
from pathlib import Path
import time
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.graph import run_agent


def _print_test_header(test_num: int, title: str, description: str):
    print("\n" + "█" * 70)
    print(f"  TEST {test_num}: {title.upper()}")
    print(f"  Goal: {description}")
    print("█" * 70 + "\n")


def _print_test_summary(result: Dict[str, Any], elapsed: float):
    print("\n" + "─" * 70)
    print(f"  Execution Time   : {elapsed:.2f}s")
    print(f"  Route Taken      : {result.get('route')}")
    print(f"  Provider Used    : {result.get('provider')}")
    print(f"  Cache Hit        : {result.get('cache_hit', False)}")
    print(f"  Evidence Status  : {result.get('evidence_status', 'N/A')}")
    print(f"  Guard Status     : {result.get('guard_status', 'N/A')}")
    print(f"  Sources Count    : {len(result.get('sources', []))}")
    if result.get("sources"):
        print("  Referenced Sources:")
        for s in result.get("sources", []):
            print(f"    • {s}")
    print("─" * 70)
    print("\n[FINAL RESPONSE]:")
    print(result.get("answer", "").strip())
    print("─" * 70 + "\n")


def run_all_tests():
    # ── Test 1: Academic Syllabus Query ──────────────────────────────────────
    _print_test_header(
        1,
        "Academic Syllabus Query",
        "Tests Planner decomposition, Qdrant+BM25 retrieval, Reranker scoring, Validator checks, and Grounded Generator"
    )
    t0 = time.time()
    r1 = run_agent("What are the core subjects and course codes for CSE 1st Year under R23 regulation?")
    _print_test_summary(r1, time.time() - t0)

    # ── Test 2: In-Memory Cache Hit Verification ─────────────────────────────
    _print_test_header(
        2,
        "In-Memory Cache Hit",
        "Re-runs the exact same query from Test 1 to verify instant cache delivery (< 1ms, 0 LLM/vector calls)"
    )
    t0 = time.time()
    r2 = run_agent("What are the core subjects and course codes for CSE 1st Year under R23 regulation?")
    _print_test_summary(r2, time.time() - t0)

    # ── Test 3: Faculty / HOD Query ──────────────────────────────────────────
    _print_test_header(
        3,
        "Faculty / Department Query",
        "Tests retrieval for departmental faculty and HOD records"
    )
    t0 = time.time()
    r3 = run_agent("Who is the Head of Department (HOD) for Civil Engineering at SRKR?")
    _print_test_summary(r3, time.time() - t0)

    # ── Test 4: Multi-Turn Pronoun Resolution ────────────────────────────────
    _print_test_header(
        4,
        "Multi-Turn Pronoun Resolution",
        "Tests if the Planner correctly resolves 'its' and 'their' using conversation history"
    )
    history = [
        {"role": "user", "content": "Tell me about the Artificial Intelligence and Data Science (AIDS) branch at SRKR."},
        {"role": "assistant", "content": "The AIDS department offers B.Tech in Artificial Intelligence & Data Science under R20 and R23 regulations."}
    ]
    t0 = time.time()
    r4 = run_agent(
        query="What are the subjects taught in its 2nd year syllabus?",
        chat_history=history,
    )
    _print_test_summary(r4, time.time() - t0)

    # ── Test 5: Out-of-Scope Rejection ───────────────────────────────────────
    _print_test_header(
        5,
        "Out-of-Scope Query Rejection",
        "Tests if the Planner immediately detects out-of-scope topics and triggers polite rejection (0 vector calls)"
    )
    t0 = time.time()
    r5 = run_agent("Can you give me a recipe for making Hyderabadi chicken biryani?")
    _print_test_summary(r5, time.time() - t0)

    # ── Test 6: Conversational Greeting ──────────────────────────────────────
    _print_test_header(
        6,
        "Conversational Greeting",
        "Tests fast greeting short-circuit without retrieval"
    )
    t0 = time.time()
    r6 = run_agent("Hi, good afternoon! Can you tell me what you can do?")
    _print_test_summary(r6, time.time() - t0)


if __name__ == "__main__":
    run_all_tests()
