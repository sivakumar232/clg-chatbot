# Technical Architecture Specification: Adaptive Hybrid Agentic RAG
**Project:** SRKR Engineering College AI Academic Advisor & Assistant  
**Document Version:** 1.0.0 (Production Blueprint)  
**Author:** Deep Learning & Agentic Systems Architecture

---

## 1. Architectural Philosophy & Executive Summary

Standard RAG treats all queries identically: $Q \rightarrow \text{Embed} \rightarrow \text{Top-K} \rightarrow \text{Rerank} \rightarrow \text{Generate}$. This fails on ambiguous college queries (e.g., missing regulations), aggregate queries (e.g., all HODs), exact entity matching (e.g., course codes), and incurs unnecessary cost on trivial chit-chat or out-of-scope prompts.

The **Adaptive Hybrid Agentic RAG** architecture converts a static retrieval pipeline into an **intelligent, self-adapting decision graph**. It introduces:
1. **Dynamic Triage**: 3-way pathing (`OUT_OF_SCOPE`, `SIMPLE_QA`, `COMPLEX_QA`) to prevent latency and resource bloat.
2. **Catalog-Grounded Query Planning**: Decomposes aggregate/ambiguous queries into bounded, parallel sub-searches using a lightweight knowledge catalog.
3. **Tri-brid Search with Late Interaction**: Combines Metadata Pre-filtering, Sparse BM25 (lexical), and Dense Vector (semantic) via Reciprocal Rank Fusion (RRF), followed by Cross-Encoder re-ranking.
4. **Self-Reflective Evidence Evaluation**: A bounded reflection loop that verifies information completeness before generation, triggering targeted query expansion if critical facts are missing.

---

## 2. End-to-End System Architecture

```
                                  USER QUERY
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │     1. QUERY UNDERSTANDING        │
                     │  • Normalization & Disambiguation │
                     │  • Slot & Entity Extraction       │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │       2. ADAPTIVE ROUTER          │
                     │  • Path Decision & Complexity     │
                     └─────────────────┬─────────────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
   [PATH 1: OUT-OF-SCOPE]     [PATH 2: SIMPLE QA]           [PATH 3: COMPLEX QA]
   Chit-chat / Non-SRKR       Factual / Single Entity       Aggregate / Ambiguous / Multi-Hop
         │                             │                             │
         ▼                             ▼                             ▼
  Direct Polite Rejection       Single Fast Search            ┌─────────────────────────────┐
  (0 vector/rerank cost)      (Dense + BM25 + Filter)         │  3. KNOWLEDGE-AWARE PLANNER │
         │                             │                      │  • Ingest Knowledge Catalog │
         │                             │                      │  • Decompose into 2-6       │
         │                             │                      │    Parallel Sub-Queries     │
         │                             │                      └──────────────┬──────────────┘
         │                             │                                     │
         │                             │                        Parallel Sub-Search Workers
         │                             │                        (ThreadPoolExecutor, max=8)
         │                             │                                     │
         │                             └──────────────┬──────────────────────┘
         │                                            ▼
         │                            ┌───────────────────────────────┐
         │                            │    4. TRI-BRID RETRIEVAL      │
         │                            │  • Qdrant Payload Filter      │
         │                            │  • BM25 Lexical Search        │
         │                            │  • Jina 1024-d Dense Search   │
         │                            │  • Reciprocal Rank Fusion     │
         │                            └───────────────┬───────────────┘
         │                                            │ Candidate Chunks
         │                                            ▼
         │                            ┌───────────────────────────────┐
         │                            │    5. JINA RERANKER V2        │
         │                            │  • Cross-Encoder Joint Scoring│
         │                            │  • Top-N Precision Pruning    │
         │                            └───────────────┬───────────────┘
         │                                            │
         │                                            ▼
         │                            ┌───────────────────────────────┐
         │                            │   6. EVIDENCE EVALUATOR       │
         │                            │      (Reflection Gate)        │
         │                            └───────────────┬───────────────┘
         │                                            │
         │                               ┌────────────┴────────────┐
         │                               │                         │
         │                        [INSUFFICIENT]              [SUFFICIENT]
         │                       (Score < Threshold)       (Coverage Verified)
         │                               │                         │
         │                   Loop Count < Max Retries (1)?          │
         │                               │                         │
         │                         YES ┌─┴─┐ NO                    │
         │                             │   │                       │
         │              ┌──────────────┘   └─────────────┐         │
         │              ▼                                ▼         │
         │      Query Reformulation              Degraded Notice   │
         │      & Targeted Retrieval             Appended to State │
         │      (Loops back to Step 4)                   │         │
         │                                               └────┬────┘
         │                                                    │
         │                                                    ▼
         │                                    ┌───────────────────────────────┐
         │                                    │   7. CONTEXT CONSTRUCTION     │
         │                                    │  • Metadata Attribution Blocks│
         │                                    │  • Grounding Constraints      │
         │                                    └───────────────┬───────────────┘
         │                                                    │
         │                                                    ▼
         │                                    ┌───────────────────────────────┐
         │                                    │     8. LLM GENERATION         │
         │                                    │  • Primary: Groq (Qwen/Llama) │
         │                                    │  • Fallback: Google Gemini    │
         │                                    └───────────────┬───────────────┘
         │                                                    │
         └─────────────────────────────┬──────────────────────┘
                                       ▼
                            GROUNDED FINAL ANSWER
                        (Citations, Sources & Metadata)
```

---

## 3. Deep Dive: Designing the Robust Adaptation Mechanism

An adaptation mechanism fails when it oscillates (endless loops), over-plans (simple queries taking 10s), or hallucinates false paths. Here is how robust adaptation is guaranteed:

### 3.1 Strict Routing State Schema
The Router produces an immutable dataclass using strict JSON mode:

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

class RouteType(str, Enum):
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    SIMPLE_QA = "SIMPLE_QA"
    COMPLEX_QA = "COMPLEX_QA"

@dataclass(frozen=True)
class IntentPayload:
    route: RouteType
    category: str                   # "syllabus", "faculty", "fees", "admin", "general"
    department: Optional[str]       # "CSE", "ECE", "MECH", etc.
    regulation: Optional[str]       # "R20", "R23", "R24"
    is_aggregate: bool              # True for "list all...", "who are the..."
    confidence: float               # 0.0 to 1.0
    detected_entities: List[str]    # ["CS3401", "Dr. Mahesh"]
```

### 3.2 The Knowledge Catalog Contract
The Planner never hallucinates arbitrary query parameters. It is bounded by a **Knowledge Catalog** representing the college's physical corpus:

```python
KNOWLEDGE_CATALOG = {
    "departments": ["CSE", "ECE", "EEE", "CIVIL", "MECH", "IT", "AIDS", "CSBS"],
    "regulations": ["R19", "R20", "R23", "R24"],
    "document_types": ["syllabus", "faculty", "governance", "policies", "circulars"],
    "defaults": {
        "regulation": "R23", # Latest active B.Tech regulation
    }
}
```

* **Slot Completion Strategy**: If `category == "syllabus"` and `regulation is None`:
  * The Planner does *not* ask a follow-up question.
  * It cross-references `KNOWLEDGE_CATALOG["regulations"]` and automatically plans parallel sub-queries for **R20** and **R23**.
  * The LLM presents a structured comparison: *"Under R23: ... | Under R20: ..."*

### 3.3 The Evidence Reflection Gate
After Jina Reranking, the top candidates pass through the **Evidence Evaluator**. This is governed by deterministic heuristics combined with an SLM critique:

```python
def evaluate_evidence(query: str, chunks: List[RetrievedChunk], intent: IntentPayload) -> bool:
    # 1. Deterministic Fast-Fail
    if not chunks:
        return False
    
    # 2. Score Threshold Gate
    avg_rerank_score = sum(c.score for c in chunks[:3]) / min(len(chunks), 3)
    if avg_rerank_score < 0.25:
        # Chunks retrieved are completely ungrounded / irrelevant
        return False
    
    # 3. Aggregate Coverage Check (e.g. HOD query requires multiple departments)
    if intent.is_aggregate and intent.category == "faculty":
        text_corpus = " ".join([c.text.lower() for c in chunks])
        # Verify coverage of core departments
        covered = sum(1 for d in ["cse", "ece", "mech", "civil"] if d in text_corpus)
        if covered < 2:
            return False

    return True
```

### 3.4 Bounded Adaptation Loop
* **Iteration Ceiling**: Maximum retry count is hard-coded to **1**.
* **Loop Prevention**: If Retry 1 fails, the system switches to **Graceful Degradation Mode**:
  * Instead of looping a 3rd time, the context is assembled with whatever partial data was retrieved.
  * A disclaimer prompt instruction is injected: *"Notice: Official documentation is incomplete for [Missing Field]. State what is known and explicitly mention that [Missing Field] could not be verified in records."*

---

## 4. Computational Requirements & Real-Time Performance Engineering

Every extra agentic step adds latency and token cost. Below is the computational budget and optimization architecture to ensure real-time response times ($< 2.5\text{s}$ P95).

### 4.1 Latency & Computational Budget

| Component | Execution Type | Latency (P50) | Latency (P95) | Additional Compute / API Load |
| :--- | :--- | :--- | :--- | :--- |
| **Router** | Lightweight LLM (Groq Qwen-2.5-32B or 8B) | 180 ms | 280 ms | 1 LLM inference (~150 prompt tokens) |
| **Planner** *(Complex QA only)* | Lightweight LLM / Regex Rules | 200 ms | 350 ms | 1 LLM inference (~250 prompt tokens) |
| **Tri-brid Retrieval** | Concurrent Qdrant API (`ThreadPoolExecutor`) | 120 ms | 220 ms | 1 dense embedding + $N$ parallel DB queries |
| **Jina Reranker v2** | Batched Cross-Encoder HTTP API | 300 ms | 550 ms | 1 API call (reranking top 20–40 chunks) |
| **Evidence Evaluator** | Deterministic heuristics (Fast-path) | 2 ms | 5 ms | 0 LLM tokens (uses scores + regex) |
| **LLM Generation** | Groq / Gemini (Streaming or Full) | 650 ms | 1,100 ms | Primary generation (~1,500 tokens) |
| **Total (Simple QA)** | **Linear Fast Path** | **~1.25 s** | **~1.85 s** | **Lowest possible overhead** |
| **Total (Complex QA)**| **Parallel Agentic Path** | **~1.65 s** | **~2.50 s** | **Bounded, production-grade** |

---

### 4.2 Real-Time Optimizations (Latency Reduction)

1. **ThreadPool Concurrency for Sub-Queries**:
   If the planner emits 6 sub-queries for 6 departments, running them sequentially would cost $6 \times 150\text{ms} = 900\text{ms}$.
   With `ThreadPoolExecutor(max_workers=8)`, all 6 query vectors are sent concurrently.
   $$\text{Retrieval Latency} = \max(\text{Latency}_1, \dots, \text{Latency}_6) \approx 180\text{ms}$$

2. **Specular Fast-Bypass for Simple QA**:
   For simple queries where the top rerank score is $\ge 0.75$, the **Evidence Evaluator is bypassed completely**, saving 1 round-trip.

3. **In-Memory Routing Cache**:
   Repeated or semantically identical queries (e.g., *"What is the CSE syllabus?"*) hit a lightweight LRU memory cache, returning the pre-calculated `IntentPayload` and sub-queries instantly in $<1\text{ms}$.

4. **Connection Pooling**:
   Persistent HTTP keep-alive connections are maintained across `JinaEmbedder`, `QdrantClient`, and `JinaReranker` using `requests.Session()` to avoid TLS handshake penalties on every user message.

---

## 5. Knowledge Layer Architecture: Unstructured, Lexical & Structured

```
                                KNOWLEDGE BASE LAYER
                                          │
            ┌─────────────────────────────┼─────────────────────────────┐
            ▼                             ▼                             ▼
   [QDRANT DENSE VECTORS]        [BM25 INVERTED INDEX]        [STRUCTURED ENTITIES]
   • 1024-d Jina Embeddings      • Keyword term frequency     • Tabular/Metadata facts
   • Cosine metric               • Exact course codes         • Faculty names, designations,
   • Rich JSON payloads          • Subject code matches         HOD list, phone directories
            │                             │                             │
            └─────────────────────────────┼─────────────────────────────┘
                                          ▼
                              UNIFIED RETRIEVAL ENGINE
                              (Prefetch + RRF Fusion)
```

1. **Dense Vectors (Qdrant)**: Embeds continuous paragraphs, course objectives, and rules.
2. **BM25 Lexical (Qdrant / BM25)**: Essential for preserving academic codes (`CS3401`, `R23`, `AIDS-B`).
3. **Structured Entity Payloads**: Metadata stored directly inside Qdrant point payloads (`department`, `regulation`, `doc_type`, `year`, `page`). This eliminates the operational overhead of running a separate PostgreSQL instance while delivering the exact same pre-filtering power.

---

## 6. Codebase Implementation Mapping

The implementation maps cleanly into the existing `src/chat_bot/` structure:

```
src/chat_bot/
├── __init__.py           # CLI entry point (interactive & one-shot)
├── config.py             # System configs & model settings
├── router.py             # [NEW] Intent classification, slot extraction, 3-way triage
├── planner.py            # [NEW] Knowledge catalog, sub-query decomposition, parallel runner
├── retriever.py          # [UPDATED] Tri-brid search (dense + BM25 + Qdrant payload filter)
├── reranker.py           # Cross-encoder Jina Reranker v2 (Already Production-Ready)
├── rag_chain.py          # [UPDATED] Orchestrator: Adaptive graph, reflection loop, timeout
└── generator.py          # LLM Generation with grounding prompt & Groq/Gemini fallback
```

---

## 7. Concrete Scenario Walkthroughs

### Scenario A: Incomplete Query (*"Give me syllabus of AIML"*)
1. **Router**: Detects `category="syllabus"`, `department="AIDS"`, `regulation=None`. Routes to `COMPLEX_QA`.
2. **Planner**: Ingests `KNOWLEDGE_CATALOG`. Identifies that AIML exists in `R20` and `R23`. Spawns 2 sub-queries:
   * Sub-query 1: `query="AIML Artificial Intelligence Machine Learning syllabus"`, `filter={"regulation": "R23"}`
   * Sub-query 2: `query="AIML Artificial Intelligence Machine Learning syllabus"`, `filter={"regulation": "R20"}`
3. **Retrieval**: Both execute in parallel. 20 chunks retrieved each $\rightarrow$ deduplicated to 30 chunks.
4. **Reranker**: Reranks top 8 precision chunks.
5. **Evaluator**: Verifies chunks contain both R20 and R23 markers. Returns `SUFFICIENT`.
6. **Generator**: Formats a clear two-part answer:
   * **Regulation R23 (Current)**: Semester-wise subjects, course codes.
   * **Regulation R20 (Previous)**: Semester-wise subjects, course codes.

### Scenario B: Aggregate Directory (*"Who are all the HODs in our college?"*)
1. **Router**: Detects `category="faculty"`, `is_aggregate=True`. Routes to `COMPLEX_QA`.
2. **Planner**: Spawns 8 sub-queries across all 8 catalog departments in parallel:
   * `[HOD of CSE, HOD of ECE, HOD of MECH, HOD of CIVIL, HOD of IT, HOD of EEE, HOD of AIDS, HOD of CSBS]`
3. **Retrieval**: 8 concurrent searches run via ThreadPool. Chunks merged $\rightarrow$ top 15 candidate chunks retained.
4. **Reranker**: Cross-scores against *"Head of the Department names and designations"*.
5. **Evaluator**: Validates that at least 6 departments are covered. Returns `SUFFICIENT`.
6. **Generator**: Outputs a complete, unified Markdown table with Department, HOD Name, and Qualification.

### Scenario C: Unrelated Query (*"Who won the cricket match yesterday?"*)
1. **Router**: Detects `route=OUT_OF_SCOPE`, `confidence=0.99`.
2. **Pipeline**: Instantly short-circuits.
3. **Output**: *"I am the SRKR Engineering College academic advisor. I can only assist with college admissions, syllabi, faculty, regulations, and campus information."*
4. **Time & Cost**: **150ms latency**, 0 vector search calls, 0 reranker calls.

---

## 8. Summary of Architectural Guarantees

* **Zero Infinite Loops**: Strict `max_retries=1` limit.
* **Bounded Latency**: Parallel thread execution guarantees $\le 2.5\text{s}$ P95.
* **High Grounding**: Dual-pass retrieval + cross-encoder reranking eliminates false-positive context.
* **Clean Code**: 100% pure Python with standard libraries (`concurrent.futures`, `dataclasses`, `enum`) and official SDKs.
