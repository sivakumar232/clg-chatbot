# Adaptive Hybrid Agentic RAG — Implementation Plan
**Project:** SRKR Engineering College AI Academic Advisor
**Branch:** `feature/agenticrag`
**Current State:** Standard 2-pass RAG (embed → vector search → rerank → generate)
**Target State:** Adaptive Hybrid Agentic RAG with closed-loop reflection

---

## Current State of the Codebase

```
chat_bot/
├── config.py                          # API keys + model settings
├── pyproject.toml                     # Dependencies (qdrant, jina, groq, gemini, etc.)
├── Ingestion/
│   ├── pipeline.py                    # Main ingestion orchestrator
│   ├── processor.py                   # Text cleaning + basic category metadata (SPARSE)
│   ├── embedder.py                    # Jina v5 embedder (1024-d)
│   ├── chunking/
│   │   ├── chunker.py
│   │   ├── semantic_chunker.py        # Structure-aware semantic chunker (MiniLM-L6-v2)
│   │   └── strategies/
│   └── loaders/
│       ├── md_loader.py
│       └── pdf_loader.py
└── src/chat_bot/
    ├── __init__.py                    # CLI entry
    ├── rag_chain.py                   # 2-pass orchestrator (retrieve → rerank → generate)
    ├── retriever.py                   # Dense vector search ONLY (Qdrant cosine, top-K)
    ├── reranker.py                    # Jina Reranker v2 (cross-encoder) — production-ready
    └── generator.py                  # LLM generation (Groq primary, Gemini fallback)
```

### What the current system CANNOT do
- Detect that "ML syllabus" is ambiguous across R20/R23/R24
- Find ALL HODs across 8 departments in one shot (returns partial results silently)
- Differentiate between bus timetable, hostel rules, and exam fee questions
- Short-circuit irrelevant out-of-scope queries without burning tokens
- Know that "Petroleum Engineering" does not exist in SRKR before searching

---

## Target Architecture

```
                         User Query
                              │
                              ▼
              ┌───────────────────────────────┐
              │   1. FAST ADAPTIVE TRIAGE     │ ← Regex guard (<0.1ms)
              │      & INTENT EXTRACTION      │   + 1 Lightweight LLM call (~180ms)
              └───────────────┬───────────────┘
                              │ DynamicIntentPayload
                              ▼
              ┌───────────────────────────────┐
              │  2. KNOWLEDGE FACET REGISTRY  │ ← In-memory dict (<2ms, no LLM)
              │     Catalog Lookup            │   Built automatically during ingestion
              └───────────────┬───────────────┘
                              │ CatalogContext
               ┌──────────────┼──────────────────┐
               ▼              ▼                   ▼
       [OUT_OF_SCOPE]    [SIMPLE_QA]         [COMPLEX_QA]
       Polite Refusal    Single Search    Parallel Sub-Query Plan
         (<150ms)          (~1.2s)              │
                              │                  ▼
                              │    ┌─────────────────────────────┐
                              │    │  3. RETRIEVAL PLANNING      │
                              │    │     Sub-query decomposition │
                              │    │     ThreadPoolExecutor      │
                              │    └──────────────┬──────────────┘
                              └───────────────────┘
                                                  │
                              ┌───────────────────▼───────────────────┐
                              │       4. TRI-BRID RETRIEVAL           │
                              │  Qdrant Payload Filter (structured)   │
                              │  + BM25 Lexical (exact codes)         │
                              │  + Jina 1024-d Dense (semantic)       │
                              │  → RRF Fusion                         │
                              │  → Jina Reranker v2 (cross-encoder)   │
                              └───────────────────┬───────────────────┘
                                                  │ List[RetrievedChunk]
                              ┌───────────────────▼───────────────────┐
                              │   5. EVIDENCE EVALUATION GATE         │
                              │  Deterministic heuristics (<5ms)      │
                              │  Score threshold + Coverage check     │
                              │  Hard ceiling: max_retries=1          │
                              └───────────────────┬───────────────────┘
                                                  │
                              ┌───────────────────▼───────────────────┐
                              │   6. CONTEXT ASSEMBLY & GENERATION    │
                              │  Authority-tiered provenance headers  │
                              │  Domain-aware system prompt           │
                              │  Groq primary → Gemini fallback       │
                              └───────────────────────────────────────┘
```

---

## Phase 0: Schema & Metadata Upgrade (Ingestion Layer)

> **This must be done FIRST.** Every later component depends on rich, consistent metadata in Qdrant.

### Problem with current metadata

`processor.py` currently extracts only:
- `category`: regex match on URL path (`cse`, `ece`, etc.) or `"general"`
- `content_type`: `"markdown"` or `"pdf"`
- `source_url`: file path or scraped URL

This is far too sparse. The router and planner need: `domain`, `doc_type`, `authority_tier`, `regulation`, `department`, and open `attributes{}`.

---

### Two-Tier Polymorphic Schema

**Tier 1 — Universal Envelope (enforced on 100% of all documents):**

```python
{
  "domain":         str,  # "academics" | "transport" | "hostel" | "placements" | "governance" | "general"
  "doc_type":       str,  # "syllabus" | "circular" | "handbook" | "timetable" | "directory" | "policy"
  "authority_tier": int,  # 1=official regulation | 2=dept brochure | 3=notice/flyer
  "effective_date": str,  # "2024-06-01" or null
  "source_url":     str,  # file path or scraped URL
  "chunk_id":       str,  # UUID
  "ingested_at":    str,  # ISO timestamp
}
```

**Tier 2 — Dynamic Domain Attributes (open JSON blob, domain-specific):**

```python
# Academic documents:
"attributes": {
  "regulation":   "R23",
  "department":   "CSE",
  "semester":     "VI",
  "subject":      "Machine Learning",
  "subject_code": "CS3201",
  "year":         4,
}

# Transport documents:
"attributes": {
  "route_number": 14,
  "destination":  "Tanuku",
}

# Hostel documents:
"attributes": {
  "hostel_type": "boys",
  "block":       "A",
}
```

> **Why open `attributes: Dict`?** When you add bus timetables tomorrow, the ingestion pipeline just
> writes `{"route_number": 14}` into `attributes`. The router, planner, and retriever
> work on it generically. Zero code changes needed anywhere.

---

### `Ingestion/metadata_extractor.py` [NEW]

Runs during ingestion after text cleaning. Strategy: domain-detection first, schema-extraction second.

```
Document Text + Source URL
         │
         ▼
  Domain Classifier (fast keyword + path regex rules — no LLM)
         │
    ┌────┴─────────────────────────────────────┐
    ▼                                           ▼
 "academics"                     "transport" / "hostel" / "placements" / "governance"
    │
    ▼
 Regex Extraction (Regulation, Department, Subject Code)
    │
    ▼ (only if regex yields 0 academic attributes)
 LLM Extraction — called ONCE per DOCUMENT, NOT per chunk
 (constrained output: response_format={"type": "json_object"}, temp=0.0)
    │
    ▼
 Canonical Normalization (ML → Machine Learning, AI&DS → AIDS)
    │
    ▼
 Tier 1 + Tier 2 Metadata Dict
```

**Key implementation details:**

```python
# Regulation: compiled regex on filename + first 500 chars
REGULATION_RE = re.compile(r"\b(R19|R20|R23|R24|R25)\b")

# Department: canonical alias map
DEPT_ALIASES = {
    "cse": "CSE",  "computer science": "CSE", "computer science & engineering": "CSE",
    "ece": "ECE",  "electronics": "ECE", "electronics and communication": "ECE",
    "ai&ds": "AIDS", "aiml": "AIDS", "aids": "AIDS", "artificial intelligence": "AIDS",
    "it":   "IT",  "information technology": "IT",
    "eee":  "EEE", "electrical": "EEE",
    "civil": "CIVIL",
    "mech": "MECH", "mechanical": "MECH",
    "csbs": "CSBS",
}

# Subject code: e.g. CS3201, AIDS304
SUBJECT_CODE_RE = re.compile(r"\b[A-Z]{2,4}\d{3,4}\b")

# Domain keyword detection sets (checked on filename + first 200 chars)
TRANSPORT_KEYWORDS = {"bus", "route", "timing", "transport", "shuttle"}
HOSTEL_KEYWORDS    = {"hostel", "warden", "mess", "dormitory", "block"}
PLACEMENT_KEYWORDS = {"placement", "company", "package", "recruit", "lpa", "offer letter"}
```

**LLM fallback** (only when regex finds 0 academic attributes on an `academics` doc):

```python
EXTRACTOR_PROMPT = """
Extract metadata from this academic document chunk.
Return ONLY JSON with keys: regulation, department, subject, subject_code, semester, year.
Use null for any field not found.
Document text: {text_sample}
"""
```

---

### `Ingestion/facet_registry.py` [NEW]

Builds and saves the Knowledge Facet Registry at the end of each ingestion run.

```python
# Saved as: Data/processeddata/facet_registry.json
{
  "domains": ["academics", "transport", "hostel", "placements", "governance"],
  "academics": {
    "departments": ["CSE", "ECE", "EEE", "MECH", "CIVIL", "IT", "AIDS", "CSBS"],
    "regulations": ["R19", "R20", "R23", "R24"],
    "active_regulation": "R23",
    "subjects": {
      "machine learning": {
        "canonical": "Machine Learning",
        "aliases":   ["ML", "AIML", "AI&ML"],
        "departments": ["CSE", "AIDS"],
        "regulations": ["R20", "R23"],
        "subject_codes": {"R20": "CS4105", "R23": "CS3201"}
      },
      ...
    }
  },
  "transport": {
    "routes":       [1, 2, 4, 7, 12, 14, 18],
    "destinations": ["Tanuku", "Bhimavaram", "Eluru", "Nidadavole"]
  },
  "updated_at": "2026-09-10T12:00:00Z"
}
```

**How it is built**: After ingestion completes, `facet_registry.py` scans `chunks.jsonl`,
aggregates all `attributes` fields per domain, normalizes via `ALIAS_MAP`,
and writes `facet_registry.json`. Takes `<1s` even for 10,000 chunks.

**Qdrant Facet API** (live alternative for 100,000+ chunks):

```python
# Zero maintenance — live from Qdrant
result = client.facets(collection_name=col, key="attributes.regulation")
# Returns: [FacetValue(value="R20", count=342), FacetValue(value="R23", count=519), ...]
```

---

### `Ingestion/bm25.py` [NEW]

Builds the shared BM25 vocabulary and computes sparse vectors for each chunk.

```python
# Vocabulary built from ALL chunk texts during ingestion
# Saved as: Data/processeddata/bm25_vocab.json
# Format: {"machine": 0, "learning": 1, "cs3201": 2, ...} (token → integer index)

def build_vocabulary(all_texts: List[str]) -> Dict[str, int]:
    """Tokenizes all chunk texts and assigns integer IDs to unique tokens."""
    from collections import Counter
    token_counts = Counter()
    for text in all_texts:
        tokens = re.findall(r"\b[A-Za-z0-9]+\b", text.lower())
        token_counts.update(tokens)
    # Keep tokens appearing ≥2 times to reduce noise
    vocab = {tok: idx for idx, (tok, _) in enumerate(
        (t for t in token_counts.items() if t[1] >= 2)
    )}
    return vocab

def compute_sparse_vector(text: str, vocab: Dict[str, int]) -> SparseVector:
    """Converts text to a BM25-style sparse vector using term frequency."""
    from collections import Counter
    tokens = re.findall(r"\b[A-Za-z0-9]+\b", text.lower())
    tf = Counter(t for t in tokens if t in vocab)
    indices = [vocab[tok] for tok in tf]
    values  = [float(count) for count in tf.values()]
    return SparseVector(indices=indices, values=values)
```

---

### `Ingestion/pipeline.py` [MODIFY]

Changes:
1. After `process_documents()`, call `metadata_extractor.enrich_metadata(chunks)`.
2. Build `bm25_vocab` from all chunk texts before embedding.
3. Compute `SparseVector` for each chunk alongside dense embedding.
4. Update `PointStruct` to include both vectors and enriched payload.
5. After all upserts, call `facet_registry.build_and_save(chunks)`.

```python
# New PointStruct structure:
PointStruct(
    id=chunk_uuid,
    vectors={
        "text":        dense_vector_1024d,      # Jina v5 dense embedding
        "text-sparse": sparse_vector_bm25,      # BM25 sparse vector
    },
    payload={
        # Tier 1 (universal)
        "domain":         "academics",
        "doc_type":       "syllabus",
        "authority_tier": 1,
        "effective_date": "2023-07-01",
        "source_url":     "PDF: R23_CSE_Syllabus.pdf",
        "chunk_id":       str(chunk_uuid),
        "ingested_at":    "2026-09-10T12:00:00Z",
        # Content
        "text":           chunk_text,
        # Tier 2 (dynamic, domain-specific)
        "attributes": {
            "regulation":   "R23",
            "department":   "CSE",
            "subject":      "Machine Learning",
            "subject_code": "CS3201",
            "semester":     "VI",
        }
    }
)
```

**Qdrant collection schema change** (one-time re-index to `srkr_kb_v2`):

```python
client.recreate_collection(
    collection_name="srkr_kb_v2",
    vectors_config={
        "text": VectorParams(size=1024, distance=Distance.COSINE),
    },
    sparse_vectors_config={
        "text-sparse": SparseVectorParams(
            index=SparseIndexParams(on_disk=False)  # Keep in RAM for speed
        ),
    },
)
# Create payload indexes for fast filtering:
client.create_payload_index(col, "domain",           PayloadSchemaType.KEYWORD)
client.create_payload_index(col, "doc_type",         PayloadSchemaType.KEYWORD)
client.create_payload_index(col, "authority_tier",   PayloadSchemaType.INTEGER)
client.create_payload_index(col, "attributes.regulation",  PayloadSchemaType.KEYWORD)
client.create_payload_index(col, "attributes.department",  PayloadSchemaType.KEYWORD)
client.create_payload_index(col, "attributes.route_number",PayloadSchemaType.INTEGER)
```

### `Ingestion/processor.py` [MODIFY]

- **Remove** the regex `category` extraction block (lines 128–136). Replaced by `metadata_extractor`.
- Keep all text cleaning and URL extraction logic unchanged.

---

## Phase 1: Intent Triage & Query Understanding

### `src/chat_bot/router.py` [NEW]

Entry point for every user query. Replaces the direct `retriever.retrieve()` call.

#### 1.1 — Greeting/Chit-chat Regex Guard (< 0.1 ms)

```python
GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|bye|good morning|good evening)\s*[!.]?\s*$",
    re.IGNORECASE,
)
OUT_OF_SCOPE_RE = re.compile(
    r"\b(cricket|ipl|bollywood|netflix|stock market|bitcoin|weather)\b",
    re.IGNORECASE,
)
```

- Greeting matched → return `RouteType.DIRECT_ANSWER` instantly. **Zero LLM token cost.**
- Strong OOS keyword matched → return `RouteType.OUT_OF_SCOPE` instantly.

#### 1.2 — Single Lightweight LLM Extraction (~180ms)

```python
ROUTER_SYSTEM_PROMPT = """
You are an intent classifier for SRKR Engineering College chatbot.
Extract structured intent from the user query.
Return ONLY a valid JSON object with these exact keys — no other text:

{
  "route":              "OUT_OF_SCOPE" | "SIMPLE_QA" | "COMPLEX_QA",
  "domain":             "academics" | "transport" | "hostel" | "placements" | "governance" | "general",
  "intent_type":        "lookup" | "comparison" | "list_all" | "procedure" | "unknown",
  "filters":            { ...dict of constraints extracted from query... },
  "extracted_entities": [ ...list of named entities mentioned... ],
  "is_aggregate":       true | false,
  "confidence":         0.0 to 1.0
}

Route rules:
- OUT_OF_SCOPE: cricket, politics, movies, anything unrelated to SRKR college
- SIMPLE_QA: single specific fact — one dept, one subject, one person
- COMPLEX_QA: "all HODs", missing regulation/dept, "compare R20 vs R23", multi-hop

Filters examples:
- "ML syllabus R23 CSE" → {"subject": "Machine Learning", "regulation": "R23", "department": "CSE"}
- "bus route 14 timing" → {"route_number": 14}
- "all HODs"            → {} (is_aggregate=true handles dept expansion)
"""
```

Model: Groq `llama-3.1-8b-instant`, `temperature=0.0`, `response_format={"type": "json_object"}`.

#### 1.3 — DynamicIntentPayload (Open-Constraint Design)

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List
from enum import Enum

class RouteType(str, Enum):
    DIRECT_ANSWER = "DIRECT_ANSWER"  # Greetings → pre-written response
    SIMPLE_QA     = "SIMPLE_QA"      # Single fact → fast path
    COMPLEX_QA    = "COMPLEX_QA"     # Aggregate, ambiguous, multi-hop
    OUT_OF_SCOPE  = "OUT_OF_SCOPE"   # Non-SRKR content

@dataclass(frozen=True)
class DynamicIntentPayload:
    raw_query:          str
    route:              RouteType
    domain:             str              # "academics", "transport", "hostel", ...
    intent_type:        str              # "lookup", "comparison", "list_all", "procedure"
    filters:            Dict[str, Any]   # Open-ended: {"regulation":"R23"} or {"route_number":14}
    extracted_entities: List[str]        # ["Machine Learning", "R23"]
    is_aggregate:       bool
    confidence:         float
```

> **Why `filters: Dict[str, Any]` instead of hardcoded `department: str, regulation: str`?**
>
> When you add bus timetables next month, the router just emits `{"route_number": 14}` in `filters`.
> The planner maps it directly to Qdrant: `attributes.route_number == 14`.
> **Zero code changes needed in router, planner, or retriever.**

#### 1.4 — LRU Triage Cache

```python
from functools import lru_cache

@lru_cache(maxsize=512)
def _cached_triage(query_normalized: str) -> DynamicIntentPayload:
    ...
```

Two students asking "CSE HOD" 30 seconds apart → second response in `<1ms`.

---

## Phase 2: Knowledge Facet Registry (In-Memory Awareness Layer)

### `src/chat_bot/facet_registry_loader.py` [NEW]

Loads `facet_registry.json` into RAM on chatbot startup. All lookups are `O(1)` dict access.

#### 2.1 — Negative Entity Filtering

```python
def entity_exists(domain: str, entity_type: str, value: str) -> bool:
    """
    Returns True only if entity is present in the KB.
    Called BEFORE any Qdrant search is dispatched.
    """
    registry = FACET_REGISTRY.get(domain, {})
    known = registry.get(entity_type, [])
    normalized = canonicalize(value).upper()
    return normalized in [v.upper() for v in known] or normalized in ALIAS_MAP
```

**Behaviour on miss:**
```
entity_exists("academics", "departments", "Petroleum Engineering") → False
  → Immediate short-circuit. No Qdrant call. No reranker call. No LLM call.
  → Response: "SRKR Engineering College does not offer Petroleum Engineering.
               Available departments: CSE, ECE, EEE, MECH, CIVIL, IT, AIDS, CSBS."
```

#### 2.2 — Ambiguity Detection

```python
@dataclass
class AmbiguityResult:
    is_ambiguous:      bool
    ambiguity_type:    str           # "MULTI_REGULATION" | "MULTI_DEPARTMENT" | "FEE_TYPE" | "NONE"
    candidate_values:  List[str]     # ["R20", "R23", "R24"]
    default_value:     Optional[str] # "R23" (proactive default)

def detect_ambiguity(intent: DynamicIntentPayload) -> AmbiguityResult:
    if intent.domain == "academics":
        subject    = intent.filters.get("subject")
        regulation = intent.filters.get("regulation")
        if subject and not regulation:
            known_regs = (
                FACET_REGISTRY["academics"]["subjects"]
                .get(subject.lower(), {})
                .get("regulations", [])
            )
            if len(known_regs) > 1:
                return AmbiguityResult(
                    is_ambiguous=True,
                    ambiguity_type="MULTI_REGULATION",
                    candidate_values=known_regs,
                    default_value=FACET_REGISTRY["academics"]["active_regulation"],
                )
    return AmbiguityResult(is_ambiguous=False, ambiguity_type="NONE",
                           candidate_values=[], default_value=None)
```

#### 2.3 — Canonical Entity Resolution

```python
ALIAS_MAP = {
    # Subject aliases
    "ml":              "Machine Learning",
    "aiml":            "Machine Learning",
    "dbms":            "Database Management Systems",
    "os":              "Operating Systems",
    # Dept aliases
    "cse hod":         "Head of Department CSE",
    "aids":            "AIDS",
    "ai&ds":           "AIDS",
    # Transport aliases
    "bus 14":          "Route 14",
    "route14":         "Route 14",
    "tanuku bus":      "Route 14",
}

def canonicalize(entity: str) -> str:
    return ALIAS_MAP.get(entity.lower().strip(), entity)
```

---

## Phase 3: Retrieval Planning Agent

### `src/chat_bot/planner.py` [NEW]

Takes `DynamicIntentPayload` + `AmbiguityResult` → produces `List[SubQuery]`.

```python
@dataclass
class SubQuery:
    query_text:      str              # Text to embed + search
    payload_filter:  Dict[str, Any]   # Maps to Qdrant payload conditions
    top_k:           int              # Chunks to retrieve
    priority:        int              # 1=primary, 2=secondary
    label:           str              # e.g. "R23_CSE_ML" (for logging + context labeling)
```

#### 3.1 — Strategy Selection

```
SIMPLE_QA, No Ambiguity
    → 1 SubQuery with intent.filters mapped directly
    → top_k=20, priority=1

COMPLEX_QA, MULTI_REGULATION ambiguity (e.g. "ML syllabus" with no regulation)
    → 1 SubQuery per known regulation (parallel)
    → Primary (active R23): priority=1, top_k=15
    → Secondary (R20):      priority=2, top_k=15
    → Proactive: answer R23 first, show R20 as comparison

COMPLEX_QA, is_aggregate=True, domain="academics", category="faculty"
    → 1 SubQuery per department from FACET_REGISTRY
    → 8 sub-queries in parallel via ThreadPoolExecutor
    → top_k=5 each (1-2 chunks per dept is enough)

COMPLEX_QA, intent_type="comparison"
    → 1 SubQuery per entity being compared
    → Results kept labeled separately before generation

COMPLEX_QA, domain="transport"
    → 1 SubQuery with {"route_number": N} filter (or broad transport filter)
    → top_k=10
```

#### 3.2 — Generic Filter Construction

```python
from qdrant_client import models

TIER1_FIELDS = {"domain", "doc_type", "authority_tier"}

def build_qdrant_filter(sub_filters: Dict[str, Any]) -> Optional[models.Filter]:
    """
    Converts {"regulation": "R23", "department": "CSE"}
    into Qdrant Filter with payload conditions on "attributes.*".
    Works generically for any domain. No hardcoded academic fields.
    """
    conditions = []
    for key, value in sub_filters.items():
        field = key if key in TIER1_FIELDS else f"attributes.{key}"
        conditions.append(
            models.FieldCondition(key=field, match=models.MatchValue(value=value))
        )
    return models.Filter(must=conditions) if conditions else None
```

#### 3.3 — Proactive Defaulting Rule

When ambiguity type is `MULTI_REGULATION`:
1. **Do NOT ask the user.** Never block the pipeline with a clarification question.
2. Set primary sub-query to `active_regulation` (e.g., `R23`).
3. Set secondary sub-query to the previous regulation (e.g., `R20`).
4. After generation, the generator appends:
   > *"Showing R23 (Current Active Regulation). Historical R20 version retrieved below for comparison."*
5. Reserve "Ask User" **only** for truly irreversible ambiguities:
   - "What are the fees?" → Cannot proactively default (tuition ≠ hostel ≠ transport fee).

#### 3.4 — Retry Sub-query Builder

```python
def build_retry_subqueries(eval_result: EvalResult) -> List[SubQuery]:
    """
    Called when evidence evaluation fails.
    Builds a TARGETED sub-query only for the MISSING entities.
    Never re-runs the full original query set.
    """
    return [
        SubQuery(
            query_text=f"Head of Department HOD {dept}",
            payload_filter={"department": dept, "doc_type": "directory"},
            top_k=5, priority=1,
            label=f"RETRY_{dept}_HOD",
        )
        for dept in eval_result.missing_entities
    ]
```

---

## Phase 4: Tri-brid Retrieval Engine

### `src/chat_bot/retriever.py` [MODIFY — significant upgrade]

Current: dense vector search only.
Target: tri-brid (dense + BM25 sparse + payload filter) + RRF + concurrent multi-subquery.

#### 4.1 — Concurrent Multi-SubQuery Dispatch

```python
import concurrent.futures

def retrieve_all(sub_queries: List[SubQuery]) -> List[RetrievedChunk]:
    """
    Dispatches all sub-queries concurrently.
    Total retrieval latency = max(individual latencies), NOT sum.
    8 dept searches in parallel = ~200ms instead of 8 * 200ms = 1.6s.
    """
    all_chunks: List[RetrievedChunk] = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(len(sub_queries), 8)
    ) as pool:
        futures = {
            pool.submit(execute_tribrid_search, sq): sq
            for sq in sub_queries
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                all_chunks.extend(future.result())
            except Exception as e:
                sq = futures[future]
                print(f"  ⚠️  Sub-query '{sq.label}' failed: {e}")

    # Deduplicate by chunk_id, keep highest RRF score
    seen: Dict[str, RetrievedChunk] = {}
    for chunk in all_chunks:
        if chunk.chunk_id not in seen or chunk.score > seen[chunk.chunk_id].score:
            seen[chunk.chunk_id] = chunk

    return sorted(seen.values(), key=lambda c: c.score, reverse=True)
```

#### 4.2 — Per-SubQuery Tri-brid Search

```python
def execute_tribrid_search(sub_query: SubQuery) -> List[RetrievedChunk]:
    """Runs dense + sparse + payload filter for one SubQuery, fuses with RRF."""

    # 1. Embed query text (Jina v5, 1024-d)
    dense_vector  = embedder.embed_query(sub_query.query_text)

    # 2. Compute BM25 sparse vector (local, <1ms)
    sparse_vector = bm25.compute_sparse_vector(sub_query.query_text, BM25_VOCAB)

    # 3. Build Qdrant payload filter from sub_query.payload_filter
    qdrant_filter = build_qdrant_filter(sub_query.payload_filter)

    # 4. Dense search (cosine, 1024-d Jina vectors)
    dense_hits = client.query_points(
        collection_name=COLLECTION,
        query=dense_vector,
        using="text",
        query_filter=qdrant_filter,
        limit=sub_query.top_k,
        with_payload=True,
    ).points

    # 5. Sparse BM25 search (exact token matching)
    sparse_hits = client.query_points(
        collection_name=COLLECTION,
        query=SparseVector(indices=sparse_vector.indices, values=sparse_vector.values),
        using="text-sparse",
        query_filter=qdrant_filter,
        limit=sub_query.top_k,
        with_payload=True,
    ).points

    # 6. RRF Fusion → deduplicated, merged, sorted by RRF score
    return rrf_fuse(dense_hits, sparse_hits, k=60, label=sub_query.label)
```

#### 4.3 — Reciprocal Rank Fusion (RRF)

```python
def rrf_fuse(
    dense_hits: List,
    sparse_hits: List,
    k: int = 60,
    label: str = "",
) -> List[RetrievedChunk]:
    """
    RRF formula: score(d) = Σ_m  1 / (k + rank_m(d))

    k=60 is the standard RRF constant (from Cormack et al. 2009).
    Higher k → reduces impact of top ranks (more stable fusion).
    Lower k → amplifies top ranks (more aggressive).
    """
    scores:    Dict[str, float] = {}
    chunk_map: Dict[str, Any]   = {}

    for rank, point in enumerate(dense_hits, start=1):
        pid = str(point.id)
        scores[pid]    = scores.get(pid, 0.0) + 1.0 / (k + rank)
        chunk_map[pid] = point

    for rank, point in enumerate(sparse_hits, start=1):
        pid = str(point.id)
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
        chunk_map.setdefault(pid, point)

    sorted_ids = sorted(scores, key=lambda pid: scores[pid], reverse=True)

    return [
        point_to_retrieved_chunk(chunk_map[pid], rrf_score=scores[pid], label=label)
        for pid in sorted_ids
    ]
```

---

## Phase 5: Evidence Evaluation Gate

### `src/chat_bot/evidence_evaluator.py` [NEW]

Runs after reranking. **No LLM involved — pure deterministic Python.**

```python
@dataclass
class EvalResult:
    is_sufficient:    bool
    reason:           str             # "OK" | "NO_RESULTS" | "LOW_CONFIDENCE" | "MISSING_COVERAGE"
    missing_entities: List[str]       # e.g. ["MECH", "CIVIL"]
    degraded_notice:  Optional[str]   # Warning injected into the final answer context
```

#### Check 1: Empty Fast-Fail

```python
if not chunks:
    return EvalResult(is_sufficient=False, reason="NO_RESULTS",
                      missing_entities=[], degraded_notice=None)
```

#### Check 2: Relevance Score Threshold

```python
avg_score = sum(c.score for c in chunks[:3]) / min(len(chunks), 3)
if avg_score < 0.30:
    if retry_count < 1:
        return EvalResult(is_sufficient=False, reason="LOW_CONFIDENCE", ...)
    else:
        # After 1 retry, accept with notice
        return EvalResult(
            is_sufficient=True,
            degraded_notice="⚠️ Low confidence results. Verify with official SRKR sources.",
        )
```

#### Check 3: Aggregate Coverage Verification

```python
if intent.is_aggregate and intent.domain == "academics":
    required_depts = set(FACET_REGISTRY["academics"]["departments"])
    covered_depts = {
        c.metadata.get("attributes", {}).get("department")
        for c in chunks
        if c.metadata.get("attributes", {}).get("department")
    }
    missing = required_depts - covered_depts
    if len(missing) > 1 and retry_count == 0:
        return EvalResult(
            is_sufficient=False,
            reason="MISSING_COVERAGE",
            missing_entities=list(missing),
        )
    if missing:
        return EvalResult(
            is_sufficient=True,
            missing_entities=list(missing),
            degraded_notice=f"⚠️ Notice: HOD information for {', '.join(missing)} could not be verified in records.",
        )
```

#### Check 4: Authority Conflict Detection

```python
# Group chunks by entity value they describe
# If two chunks give different values for the same entity:
# Prefer authority_tier=1 (official regulation) over tier=2 (dept brochure)
# Inject a note if conflict exists but tier-1 wins
```

#### The Hard Retry Ceiling

```python
# In rag_chain.py — this logic NEVER retries more than once
MAX_RETRIES = 1

eval_result = evaluator.evaluate(intent, sub_queries, precision_chunks, retry_count=0)
if not eval_result.is_sufficient and MAX_RETRIES > 0:
    retry_sq      = planner.build_retry_subqueries(eval_result)
    retry_raw     = retriever.retrieve_all(retry_sq)
    retry_prec    = reranker.rerank(query, retry_raw, top_n=5)
    precision_chunks = merge_deduplicate(precision_chunks, retry_prec)
    eval_result   = evaluator.evaluate(intent, retry_sq, precision_chunks, retry_count=1)
    # retry_count=1 → evaluator will ALWAYS return is_sufficient=True (with notice if needed)
    # NO 3rd loop. EVER.
```

---

## Phase 6: Context Assembly & Generation Upgrade

### `src/chat_bot/generator.py` [MODIFY]

#### 6.1 — Authority-Tiered Context Block Headers

```python
TIER_LABELS = {1: "OFFICIAL REGULATION", 2: "DEPT SOURCE", 3: "NOTICE/FLYER"}

def build_context_blocks(chunks: List[RetrievedChunk]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        meta  = chunk.metadata
        attrs = meta.get("attributes", {})
        tier  = meta.get("authority_tier", 2)

        header = (
            f"[Block {i} | {TIER_LABELS.get(tier, 'SOURCE')} | "
            f"Score: {chunk.score:.3f} | Source: {chunk.source} | "
            f"Domain: {meta.get('domain', '?')}"
        )
        if attrs.get("regulation"):
            header += f" | Reg: {attrs['regulation']}"
        if attrs.get("department"):
            header += f" | Dept: {attrs['department']}"
        header += "]"

        blocks.append(f"{header}\n{chunk.text.strip()}")
    return "\n\n".join(blocks)
```

#### 6.2 — Domain-Aware System Prompts

```python
SYSTEM_PROMPTS = {
    "academics": """You are the official academic advisor for SRKR Engineering College (Autonomous), Bhimavaram.

RULES:
1. Answer ONLY from the provided context blocks. Never invent facts.
2. When multiple regulations are present, structure your answer by regulation (e.g., "Under R23: ... | Under R20: ...").
3. Always cite course codes (CS3201) and regulation numbers (R23) from the context.
4. If a regulation is missing from context, explicitly state it is not in records.
5. List sources under a ### Sources heading at the end.""",

    "transport": """You are the SRKR Engineering College transport information assistant.

RULES:
1. Answer bus schedule queries based ONLY on the provided timetable context.
2. Always include route number, departure times, and stop names.
3. If the stop or route is not in context, say so clearly.""",

    "hostel": """You are the SRKR hostel information assistant.

RULES:
1. Answer based ONLY on provided hostel handbook or circular context.
2. Distinguish between boys hostel and girls hostel rules where different.
3. If information is unavailable, direct the student to the Warden's office.""",

    "placements": """You are the SRKR placement cell information assistant.

RULES:
1. Answer based ONLY on the provided placement records context.
2. Always cite academic year, company name, and package figures from context.
3. Do not extrapolate or predict future placements.""",

    "general": """You are the SRKR Engineering College AI assistant.

RULES:
1. Answer ONLY from the provided context. If not in context, say so clearly.
2. Do not guess or extrapolate.""",
}
```

#### 6.3 — Degraded Notice Injection

```python
def generate(
    self,
    query: str,
    chunks: List[RetrievedChunk],
    intent: DynamicIntentPayload,
    degraded_notice: Optional[str] = None,
) -> Tuple[str, str]:
    system_prompt  = SYSTEM_PROMPTS.get(intent.domain, SYSTEM_PROMPTS["general"])
    context_blocks = build_context_blocks(chunks)

    # If evidence was incomplete, inject notice into context
    notice_block = f"\n\n[SYSTEM NOTICE]: {degraded_notice}" if degraded_notice else ""

    user_content = (
        f"Context Information:\n"
        f"{'='*50}\n"
        f"{context_blocks}{notice_block}\n"
        f"{'='*50}\n\n"
        f"Student/Faculty Question: {query}\n\n"
        "Answer based strictly on the context above."
    )
    # ... (Groq → Gemini fallback logic unchanged)
```

---

## Phase 7: Orchestrator Rebuild

### `src/chat_bot/rag_chain.py` [MODIFY — full rewrite of `run()`]

```python
class AgenticRAGPipeline:
    def __init__(self, collection_name: Optional[str] = None):
        self.router    = AdaptiveRouter()
        self.registry  = FacetRegistryLoader()
        self.planner   = RetrievalPlanner(self.registry)
        self.retriever = TribridRetriever(collection_name=collection_name)
        self.reranker  = JinaReranker()
        self.evaluator = EvidenceEvaluator(self.registry)
        self.generator = LLMGenerator()

    def run(self, query: str) -> Dict[str, Any]:
        t0 = time.time()
        print(f"\n{'█'*65}\n  AGENTIC RAG PIPELINE: \"{query}\"\n{'█'*65}")

        # ── Phase 1: Adaptive Triage ──────────────────────────────
        intent = self.router.triage(query)
        print(f"  [ROUTER] Route={intent.route} | Domain={intent.domain} | "
              f"Filters={intent.filters} | Aggregate={intent.is_aggregate}")

        if intent.route == RouteType.DIRECT_ANSWER:
            return self._direct_response(query)
        if intent.route == RouteType.OUT_OF_SCOPE:
            return self._oos_response()

        # ── Phase 2: Facet Registry Lookup ───────────────────────
        catalog_ctx = self.registry.validate_and_enrich(intent)
        print(f"  [REGISTRY] Negative={catalog_ctx.has_negative_entity} | "
              f"Ambiguity={catalog_ctx.ambiguity.ambiguity_type}")

        if catalog_ctx.has_negative_entity:
            return self._negative_entity_response(catalog_ctx)

        # ── Phase 3: Retrieval Planning ──────────────────────────
        sub_queries = self.planner.plan(intent, catalog_ctx)
        print(f"  [PLANNER] Generated {len(sub_queries)} sub-queries")

        # ── Phase 4: Concurrent Tri-brid Retrieval ───────────────
        candidate_chunks = self.retriever.retrieve_all(sub_queries)
        precision_chunks = self.reranker.rerank(query, candidate_chunks, top_n=8)
        print(f"  [RETRIEVAL] {len(candidate_chunks)} candidates → {len(precision_chunks)} precision chunks")

        # ── Phase 5: Evidence Gate (max 1 retry) ─────────────────
        eval_result = self.evaluator.evaluate(intent, sub_queries, precision_chunks, retry_count=0)
        if not eval_result.is_sufficient:
            print(f"  [EVIDENCE] INSUFFICIENT ({eval_result.reason}). Retry for: {eval_result.missing_entities}")
            retry_sq     = self.planner.build_retry_subqueries(eval_result)
            retry_raw    = self.retriever.retrieve_all(retry_sq)
            retry_prec   = self.reranker.rerank(query, retry_raw, top_n=5)
            precision_chunks = self._merge_deduplicate(precision_chunks, retry_prec)
            eval_result  = self.evaluator.evaluate(intent, retry_sq, precision_chunks, retry_count=1)
            print(f"  [EVIDENCE] Post-retry: Sufficient={eval_result.is_sufficient}")

        # ── Phase 6: Generate ─────────────────────────────────────
        answer, provider = self.generator.generate(
            query=query,
            chunks=precision_chunks,
            intent=intent,
            degraded_notice=eval_result.degraded_notice,
        )

        elapsed = time.time() - t0
        print(f"\n  ✓ Total pipeline time: {elapsed:.2f}s | Provider: {provider}")

        return {
            "query":           query,
            "answer":          answer,
            "chunks":          precision_chunks,
            "provider":        provider,
            "route":           intent.route,
            "domain":          intent.domain,
            "sub_queries":     len(sub_queries),
            "elapsed_seconds": elapsed,
        }

    def _direct_response(self, query: str) -> Dict[str, Any]:
        return {"query": query, "answer": "Hello! I'm the SRKR College AI advisor. How can I help you?",
                "route": RouteType.DIRECT_ANSWER, "elapsed_seconds": 0.001}

    def _oos_response(self) -> Dict[str, Any]:
        return {"query": "", "answer": (
            "I'm the SRKR Engineering College academic advisor. "
            "I can only assist with college academics, syllabus, faculty, regulations, "
            "transport, hostel, placements, and campus information."
        ), "route": RouteType.OUT_OF_SCOPE, "elapsed_seconds": 0.18}

    def _merge_deduplicate(
        self, primary: List[RetrievedChunk], secondary: List[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        seen = {c.chunk_id: c for c in primary}
        for c in secondary:
            if c.chunk_id not in seen:
                seen[c.chunk_id] = c
        return sorted(seen.values(), key=lambda c: c.score, reverse=True)

    def close(self):
        self.retriever.close()
        self.reranker.close()
```

---

## Phase 8: Qdrant Re-indexing (One-Time Migration)

**Why a new collection is needed:**
- Adding sparse vectors (`text-sparse` for BM25) requires schema-level change.
- Adding enriched payload fields (`domain`, `authority_tier`, `attributes.*`) requires re-ingesting with new metadata.

**Migration Steps:**

```
Step 1: Create new collection "srkr_kb_v2" with updated schema (dense + sparse + payload indexes)
Step 2: Run upgraded Ingestion/pipeline.py
        - metadata_extractor.enrich_metadata() on each document
        - bm25.compute_sparse_vector() on each chunk
        - PointStruct with both vector types + full payload
Step 3: After ingestion: facet_registry.build_and_save() → facet_registry.json
Step 4: config.py → QDRANT_COLLECTION_NAME = "srkr_kb_v2"
Step 5: Smoke test 6 test queries (see Testing section below)
Step 6: Delete old "collection2" after validation
```

---

## New File Map

```
chat_bot/
├── config.py                           [MODIFY] Add FACET_REGISTRY_PATH, BM25_VOCAB_PATH
│
├── Ingestion/
│   ├── pipeline.py                     [MODIFY] Enrich metadata, compute BM25 sparse, build registry
│   ├── processor.py                    [MODIFY] Remove regex category → handled by metadata_extractor
│   ├── metadata_extractor.py           [NEW]    Domain detection + Tier 1+2 extraction
│   ├── facet_registry.py               [NEW]    Build + save facet_registry.json post-ingestion
│   ├── bm25.py                         [NEW]    Vocab builder + sparse vector computation
│   └── embedder.py                     [NO CHANGE]
│
└── src/chat_bot/
    ├── __init__.py                     [MODIFY] Use AgenticRAGPipeline instead of RAGPipeline
    ├── rag_chain.py                    [MODIFY] Full rewrite of run() as adaptive decision graph
    ├── router.py                       [NEW]    Regex guard + LLM triage → DynamicIntentPayload
    ├── planner.py                      [NEW]    SubQuery builder, filter construction, proactive defaulting
    ├── retriever.py                    [MODIFY] Tri-brid search, RRF, concurrent retrieve_all()
    ├── reranker.py                     [NO CHANGE]
    ├── evidence_evaluator.py           [NEW]    Coverage + score heuristics, hard retry ceiling
    ├── facet_registry_loader.py        [NEW]    In-memory registry loader, entity lookup, ambiguity API
    └── generator.py                    [MODIFY] Domain-aware prompts, authority-tiered context blocks
```

---

## Dependencies to Add

```toml
# pyproject.toml additions:
"rank-bm25>=0.2.2"   # BM25 vocab + term frequency (sparse vector computation)
```

No Neo4j. No PostgreSQL. No Redis. **Qdrant alone** handles structured filtering + dense semantic + BM25 sparse.

---

## Performance Targets

| Query Path | Latency P50 | Latency P95 | API Calls |
| :--- | :--- | :--- | :--- |
| `DIRECT_ANSWER` (greeting) | <5 ms | <10 ms | 0 LLM, 0 Qdrant |
| `OUT_OF_SCOPE` (regex match) | <5 ms | <10 ms | 0 LLM, 0 Qdrant |
| `OUT_OF_SCOPE` (router call) | ~180 ms | ~280 ms | 1 LLM, 0 Qdrant |
| `SIMPLE_QA` (single fact) | ~1.1 s | ~1.7 s | 1 LLM (router) + 1 Qdrant + 1 reranker + 1 LLM |
| `COMPLEX_QA` (8 depts parallel) | ~1.5 s | ~2.3 s | 1 LLM (router) + 8 Qdrant (parallel) + 1 reranker + 1 LLM |
| `COMPLEX_QA` (with 1 retry) | ~2.0 s | ~3.0 s | 1 LLM + 8+N Qdrant + 2 reranker + 1 LLM |

---

## Implementation Order

- `[ ]` **Phase 0** — Schema & Ingestion Upgrade
  - `[ ]` Write `Ingestion/metadata_extractor.py`
  - `[ ]` Write `Ingestion/bm25.py`
  - `[ ]` Write `Ingestion/facet_registry.py`
  - `[ ]` Modify `Ingestion/processor.py` (remove regex category)
  - `[ ]` Modify `Ingestion/pipeline.py` (sparse vectors + enriched payload)
  - `[ ]` Modify `config.py` (add new path constants)
  - `[ ]` Run full ingestion → `srkr_kb_v2`

- `[ ]` **Phase 1** — Router
  - `[ ]` Write `src/chat_bot/router.py`

- `[ ]` **Phase 2** — Facet Registry Loader
  - `[ ]` Write `src/chat_bot/facet_registry_loader.py`

- `[ ]` **Phase 3** — Planner
  - `[ ]` Write `src/chat_bot/planner.py`

- `[ ]` **Phase 4** — Tri-brid Retriever
  - `[ ]` Modify `src/chat_bot/retriever.py`

- `[ ]` **Phase 5** — Evidence Evaluator
  - `[ ]` Write `src/chat_bot/evidence_evaluator.py`

- `[ ]` **Phase 6** — Generator Upgrade
  - `[ ]` Modify `src/chat_bot/generator.py`

- `[ ]` **Phase 7** — Orchestrator Rebuild
  - `[ ]` Rewrite `src/chat_bot/rag_chain.py`

- `[ ]` **Phase 8** — CLI Update & Integration
  - `[ ]` Modify `src/chat_bot/__init__.py`

- `[ ]` **Smoke Tests**
  - `[ ]` "What is the ML syllabus?" → detects R20/R23 ambiguity, answers proactively
  - `[ ]` "Who are all the HODs?" → 8-dept parallel search, coverage check
  - `[ ]` "Who won the cricket match?" → OUT_OF_SCOPE instant refusal
  - `[ ]` "Petroleum Engineering syllabus" → negative entity fast-fail, no Qdrant call
  - `[ ]` "Compare R20 and R23 ML syllabus" → comparison path, side-by-side output
  - `[ ]` "When does bus 14 reach Tanuku?" → transport domain, route filter

---

## Open Questions Before Execution

> [!IMPORTANT]
> **Q1: Qdrant Collection Migration** — Can we create `srkr_kb_v2` and re-index from scratch, or does `collection2` need to be kept live during migration? A fresh re-index is required to add sparse vectors + enriched metadata.

> [!IMPORTANT]
> **Q2: Current Raw Data Domains** — Does `Data/raw/` currently contain only academic PDFs/markdowns, or does it already include transport timetables, hostel handbooks, or placement records? This determines how many domains the `metadata_extractor` needs to handle at Phase 0.

> [!NOTE]
> **Q3: Router LLM Choice** — `llama-3.1-8b-instant` (~120ms, fastest) vs `qwen-2.5-32b-instruct` (~200ms, more accurate extraction)? Recommend starting with 8B and upgrading if accuracy issues arise.

> [!NOTE]
> **Q4: Qdrant Cloud Version** — Sparse vector support requires Qdrant ≥ 1.7. The `qdrant-client>=1.19.0` in `pyproject.toml` is compatible. Confirm the Qdrant Cloud cluster version is also ≥ 1.7 before running Phase 0 re-index.
