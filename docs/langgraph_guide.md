# LangGraph: Complete Guide — From Zero to Agentic RAG
**Project Context:** SRKR Engineering College AI Academic Advisor  
**Audience:** Beginner to Intermediate  
**Goal:** Understand LangGraph deeply, then apply it to build the Adaptive Hybrid Agentic RAG system

---

## Table of Contents

1. [Mental Model: Why LangGraph Exists](#1-mental-model-why-langgraph-exists)
2. [Core Concepts: The 3 Pillars](#2-core-concepts-the-3-pillars)
3. [Hands-On: Hello World in LangGraph](#3-hands-on-hello-world-in-langgraph)
4. [Conditional Edges: Decision Making](#4-conditional-edges-decision-making)
5. [Loops & Cycles: The Superpower](#5-loops--cycles-the-superpower)
6. [Reducers: Merging State Updates](#6-reducers-merging-state-updates)
7. [Persistence & Checkpointing (Memory)](#7-persistence--checkpointing-memory)
8. [Practical Patterns You Will Use Daily](#8-practical-patterns-you-will-use-daily)
9. [Building Our Agentic RAG Graph (Full Implementation)](#9-building-our-agentic-rag-graph-full-implementation)
10. [Debugging & Visualization](#10-debugging--visualization)
11. [LangGraph vs Raw LangChain vs From Scratch](#11-langgraph-vs-raw-langchain-vs-from-scratch)
12. [Common Mistakes & How to Avoid Them](#12-common-mistakes--how-to-avoid-them)

---

## 1. Mental Model: Why LangGraph Exists

### The Problem with Linear Pipelines

A standard LangChain chain runs like a **one-way conveyor belt**:

```
Input → Step A → Step B → Step C → Output
```

This is fine for simple Q&A, but breaks for real-world AI applications because:

1. **You cannot loop back** — If the retrieved documents are bad, you can't retry automatically.
2. **You cannot branch** — You can't say "if the query is simple, skip 3 steps; if complex, do extra planning."
3. **You cannot parallelize cleanly** — Running 6 sub-queries concurrently and merging results needs custom code.
4. **No shared memory** — Each step doesn't easily know what previous steps decided.

### The LangGraph Solution: Stateful Computation Graph

LangGraph models your application as a **Directed Graph** where:

- **Nodes** are workers (Python functions doing the actual work)
- **Edges** are the connections between workers
- **State** is the shared memory flowing through the entire graph

```
                    ┌─────────────────────────────────┐
                    │  GRAPH STATE (shared dictionary) │
                    │  { query, route, chunks, answer }│
                    └────────────────┬────────────────┘
                                     │ flows through every node
                    ┌────────────────▼────────────────┐
                    │          Node A (Router)         │
                    └────┬───────────────────┬─────────┘
                         │                   │
              (if simple)│                   │(if complex)
                    ┌────▼────┐         ┌────▼────┐
                    │ Node B  │         │ Node C  │
                    └────┬────┘         └────┬────┘
                         │                   │
                         └─────────┬─────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Node D (Generate Answer)   │
                    └─────────────────────────────┘
```

**The key insight:** LangGraph is just a sophisticated way to write `if/else` logic and `while` loops over Python functions, but with state management built-in.

---

## 2. Core Concepts: The 3 Pillars

### Pillar 1: State (TypedDict)

The **State** is a Python `TypedDict` — a typed dictionary shared across all nodes.
Every node reads from it and writes partial updates back to it.

```python
from typing import TypedDict, List, Annotated
import operator

class GraphState(TypedDict):
    # Each field is a piece of shared information
    query: str           # User's question (set once, read everywhere)
    route: str           # "SIMPLE_QA", "COMPLEX_QA", "OUT_OF_SCOPE"
    answer: str          # Final answer (set by generator node)
    retry_count: int     # How many times we retried retrieval
```

**Rules about State:**
- Nodes return a **partial dict** (only the keys they changed). LangGraph merges it automatically.
- If two nodes run in parallel and both update the same key, you need a **reducer** (explained in Section 6).

---

### Pillar 2: Nodes (Python Functions)

A node is any Python function that:
1. Takes the current **state** as input
2. Does some work (calls LLM, queries DB, filters data...)
3. Returns a **partial dict** with the fields it wants to update

```python
def my_node(state: GraphState) -> dict:
    # READ from state
    query = state["query"]
    
    # DO WORK
    result = do_something(query)
    
    # RETURN only the fields you are updating
    return {"answer": result}
    # LangGraph will merge {"answer": result} into the existing state
    # All other fields (query, route, etc.) remain unchanged
```

**Important:** Nodes do NOT return the full state — they return only what changed.

---

### Pillar 3: Edges (Connections)

There are 3 types of edges:

#### Type 1: Normal Edge (Always go from A to B)
```python
workflow.add_edge("node_a", "node_b")
# After node_a finishes, always go to node_b
```

#### Type 2: Conditional Edge (Branch based on state)
```python
def routing_function(state: GraphState) -> str:
    """Looks at state, returns the NAME of the next node."""
    if state["route"] == "OUT_OF_SCOPE":
        return "direct_reply"
    elif state["route"] == "COMPLEX_QA":
        return "plan_queries"
    else:
        return "retrieve"

workflow.add_conditional_edges(
    "router_node",          # from this node
    routing_function,       # call this function to decide
    {
        "direct_reply": "direct_reply",   # string -> actual node name mapping
        "plan_queries": "plan_queries",
        "retrieve": "retrieve",
    }
)
```

#### Type 3: Edge to END
```python
from langgraph.graph import END
workflow.add_edge("generator_node", END)  # This terminates the graph
```

---

## 3. Hands-On: Hello World in LangGraph

Let's build the simplest possible graph to understand how it works.

### Install LangGraph
```bash
uv add langgraph langchain-groq
```

### Simple 2-Node Graph

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# ── Step 1: Define State ──────────────────────────────────────────
class State(TypedDict):
    name: str
    greeting: str

# ── Step 2: Define Nodes ──────────────────────────────────────────
def greet_node(state: State) -> dict:
    """Reads 'name' from state, creates a greeting."""
    name = state["name"]
    return {"greeting": f"Hello, {name}!"}

def shout_node(state: State) -> dict:
    """Reads 'greeting', converts to uppercase."""
    greeting = state["greeting"]
    return {"greeting": greeting.upper()}

# ── Step 3: Build the Graph ───────────────────────────────────────
workflow = StateGraph(State)

# Add nodes
workflow.add_node("greet", greet_node)
workflow.add_node("shout", shout_node)

# Set entry point using START node (Modern standard)
workflow.add_edge(START, "greet")

# Connect nodes
workflow.add_edge("greet", "shout")
workflow.add_edge("shout", END)

# Compile the graph into an executable app
app = workflow.compile()

# ── Step 4: Run the Graph ─────────────────────────────────────────
result = app.invoke({"name": "Sivakumar"})
print(result)
# Output: {'name': 'Sivakumar', 'greeting': 'HELLO, SIVAKUMAR!'}
```

**What happened:**
1. `app.invoke({"name": "Sivakumar"})` → initializes State
2. `greet_node` runs → updates `greeting` to `"Hello, Sivakumar!"`
3. `shout_node` runs → updates `greeting` to `"HELLO, SIVAKUMAR!"`
4. Graph ends, returns final state

---

## 4. Conditional Edges: Decision Making

This is where LangGraph becomes powerful. Let's add an if/else branch:

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    query: str
    is_college_related: bool
    answer: str

# ── Nodes ──────────────────────────────────────────────────────────
def classify_node(state: State) -> dict:
    """Decides if query is college-related."""
    query = state["query"].lower()
    keywords = ["college", "syllabus", "hod", "faculty", "fees", "admission"]
    is_related = any(k in query for k in keywords)
    return {"is_college_related": is_related}

def college_answer_node(state: State) -> dict:
    """Handles college queries."""
    return {"answer": f"College info about: {state['query']}"}

def rejection_node(state: State) -> dict:
    """Rejects out-of-scope queries."""
    return {"answer": "I only answer SRKR College related questions."}

# ── Routing Function ───────────────────────────────────────────────
def route_after_classify(state: State) -> str:
    """This function's return value determines the next node."""
    if state["is_college_related"]:
        return "answer"    # goes to college_answer_node
    else:
        return "reject"    # goes to rejection_node

# ── Build Graph ────────────────────────────────────────────────────
workflow = StateGraph(State)
workflow.add_node("classify", classify_node)
workflow.add_node("answer", college_answer_node)
workflow.add_node("reject", rejection_node)

workflow.set_entry_point("classify")

workflow.add_conditional_edges(
    "classify",           # Source node
    route_after_classify, # Decision function
    {
        "answer": "answer",   # Return value -> next node mapping
        "reject": "reject",
    }
)

workflow.add_edge("answer", END)
workflow.add_edge("reject", END)

app = workflow.compile()

# Test 1: College related
print(app.invoke({"query": "What is the CSE syllabus?"})["answer"])
# → "College info about: What is the CSE syllabus?"

# Test 2: Out of scope
print(app.invoke({"query": "Who won the IPL match?"})["answer"])
# → "I only answer SRKR College related questions."
```

---

## 5. Loops & Cycles: The Superpower

The most unique feature of LangGraph is **cycles** — you can loop back to a previous node.

This is what enables **self-correction / reflection** in Agentic RAG.

```
retrieve → grade_docs → (bad docs?) → rewrite_query → retrieve → grade_docs → ...
                      → (good docs?) → generate_answer → END
```

### The Loop Pattern

```python
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

class State(TypedDict):
    query: str
    docs: List[str]
    retry_count: int
    answer: str

# ── Nodes ──────────────────────────────────────────────────────────
def retrieve_node(state: State) -> dict:
    query = state["query"]
    # Simulated retrieval
    if "R23" in query:
        docs = ["R23 syllabus document chunk 1", "R23 course details"]
    else:
        docs = []  # Bad retrieval — no docs found
    return {"docs": docs}

def grade_docs_node(state: State) -> dict:
    """Checks if retrieved docs are useful."""
    return {}  # no update needed — routing function reads state directly

def rewrite_query_node(state: State) -> dict:
    """Reformulates the query to be more specific."""
    old_query = state["query"]
    new_query = old_query + " R23 regulation SRKR"
    return {
        "query": new_query,
        "retry_count": state.get("retry_count", 0) + 1
    }

def generate_node(state: State) -> dict:
    docs = state["docs"]
    return {"answer": f"Based on {len(docs)} documents: [answer here]"}

# ── Routing Function with Loop Logic ──────────────────────────────
def decide_after_grade(state: State) -> str:
    """The key function — controls whether we loop or continue."""
    docs = state.get("docs", [])
    retry_count = state.get("retry_count", 0)
    
    if docs:
        return "generate"           # Good docs → proceed to generation
    elif retry_count < 1:
        return "rewrite"            # Bad docs, retries left → loop back
    else:
        return "generate"           # No more retries → generate with what we have

# ── Build Graph ────────────────────────────────────────────────────
workflow = StateGraph(State)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_docs_node)
workflow.add_node("rewrite", rewrite_query_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade")

# The conditional edge that creates the loop
workflow.add_conditional_edges(
    "grade",
    decide_after_grade,
    {
        "generate": "generate",
        "rewrite": "rewrite",       # This creates a cycle!
    }
)

# The loop: rewrite goes back to retrieve (NOT to START)
workflow.add_edge("rewrite", "retrieve")    # ← THIS IS THE LOOP
workflow.add_edge("generate", END)

app = workflow.compile()

# Execution trace for a query needing retry:
# → retrieve("syllabus")         → docs = []
# → grade()                      → checks: docs empty, retry_count=0
# → decide: "rewrite"            → loop back!
# → rewrite()                    → query = "syllabus R23 regulation SRKR", retry_count=1
# → retrieve("syllabus R23 ...") → docs = ["R23 chunk 1", ...]
# → grade()                      → checks: docs found
# → decide: "generate"           → exit loop
# → generate()                   → Final answer
# → END
```

---

## 6. Reducers: Merging State Updates

By default, when a node returns a value for a key, it **replaces** the previous value.

But sometimes you want to **accumulate** (append to a list), especially when multiple nodes or parallel branches update the same field.

```python
from typing import TypedDict, List, Annotated
import operator

class State(TypedDict):
    # Normal field — REPLACED on each update
    query: str
    
    # Reducer field — APPENDED on each update (operator.add on lists = concat)
    chunks: Annotated[List[str], operator.add]
    
    # Custom reducer — only keep the maximum score
    best_score: Annotated[float, lambda a, b: max(a, b)]
```

### When Do You Need Reducers?

**Scenario: Parallel Execution (Fan-Out & Fan-In)**

In our Agentic RAG, when searching multiple departments simultaneously, parallel branches run at the same time.

- **Fan-Out**: Add edges from one source node (or `START`) to multiple target nodes.
- **Fan-In**: Add edges from multiple target nodes to a single downstream join node.
- **Reducer**: Combine parallel outputs (e.g. `operator.add`) so outputs append to state instead of overwriting each other.

```python
from typing import TypedDict, List, Annotated
import operator
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    query: str
    # Using reducer — each branch's results get APPENDED, not replaced
    all_chunks: Annotated[List[str], operator.add]

def search_cse(state: State) -> dict:
    return {"all_chunks": ["CSE faculty chunk 1", "CSE HOD Dr. Kumar"]}

def search_ece(state: State) -> dict:
    return {"all_chunks": ["ECE faculty chunk 1", "ECE HOD Dr. Reddy"]}

def combine_results(state: State) -> dict:
    return {"summary": f"Retrieved {len(state['all_chunks'])} total chunks."}

workflow = StateGraph(State)
workflow.add_node("search_cse", search_cse)
workflow.add_node("search_ece", search_ece)
workflow.add_node("combine_results", combine_results)

# Fan-Out: START triggers both CSE and ECE in parallel
workflow.add_edge(START, "search_cse")
workflow.add_edge(START, "search_ece")

# Fan-In: Both parallel branches converge into combine_results
workflow.add_edge("search_cse", "combine_results")
workflow.add_edge("search_ece", "combine_results")
workflow.add_edge("combine_results", END)

# Result: all_chunks = ["CSE faculty chunk 1", "CSE HOD Dr. Kumar",
#                       "ECE faculty chunk 1", "ECE HOD Dr. Reddy"]
# Both results are preserved because of the Annotated[List, operator.add] reducer
```

---

## 7. Persistence & Checkpointing (Memory)

LangGraph can save the state at every step to a database, enabling:
- **Resume interrupted conversations**
- **Multi-turn chat memory** (remember what was said before)
- **Debugging** (replay a failed conversation step by step)

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

# ── Create a checkpointer ─────────────────────────────────────────
checkpointer = InMemorySaver()  # In-memory checkpointer for dev/tests.

# ── Compile WITH a checkpointer ──────────────────────────────────
app = workflow.compile(checkpointer=checkpointer)

# ── Run with a unique thread_id (one thread = one conversation) ───
config = {"configurable": {"thread_id": "user-123-session-1"}}

# First message
result1 = app.invoke({"query": "What is the CSE HOD name?"}, config=config)

# Second message (same thread_id = same conversation, state is remembered)
result2 = app.invoke({"query": "What is their qualification?"}, config=config)
# The graph can now reference context from the first message!
```

### For Production — Use SQLite Checkpointer

Install: `pip install langgraph-checkpoint-sqlite`

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Use context manager or connection string
with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    app = workflow.compile(checkpointer=checkpointer)
    result = app.invoke({"query": "What is the fee structure?"}, config=config)
```

---

## 8. Practical Patterns You Will Use Daily

### Pattern 1: Tool Calling Node

```python
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

def llm_node(state: State) -> dict:
    """Calls LLM and gets response."""
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    response = llm.invoke([HumanMessage(content=state["query"])])
    return {"answer": response.content}
```

### Pattern 2: Structured Output from LLM (Critical for Routing)

```python
from pydantic import BaseModel
from langchain_groq import ChatGroq
from typing import Literal

# Define structured output schema
class RouteDecision(BaseModel):
    route: Literal["OUT_OF_SCOPE", "SIMPLE_QA", "COMPLEX_QA"]
    confidence: float
    category: str

def router_node(state: State) -> dict:
    """LLM returns structured RouteDecision, not raw text."""
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    structured_llm = llm.with_structured_output(RouteDecision)
    
    prompt = f"""
    Classify this query for SRKR Engineering College chatbot.
    
    OUT_OF_SCOPE: Not about SRKR college (cricket, movies, general knowledge)
    SIMPLE_QA: Single, direct college question
    COMPLEX_QA: Aggregate, ambiguous, or multi-hop college query
    
    Query: {state['query']}
    """
    
    decision: RouteDecision = structured_llm.invoke(prompt)
    
    return {
        "route":      decision.route,
        "confidence": decision.confidence,
        "category":   decision.category,
    }
```

### Pattern 3: Streaming Responses

LangGraph supports multiple `stream_mode` options:
- `"updates"`: Returns only the dictionary updates made by each node.
- `"values"`: Returns the full graph state after each step.
- `"messages"`: Streams LLM tokens & message chunks in real time as they are generated.

```python
# Stream updates node-by-node
for chunk in app.stream({"query": "What is the fee structure?"}, stream_mode="updates"):
    for node_name, node_output in chunk.items():
        print(f"Node '{node_name}' returned: {node_output}")
```

### Pattern 4: Human-in-the-Loop (Interrupts)

Modern LangGraph uses `interrupt()` inside node functions to dynamically pause graph execution and prompt for human approval or input.

```python
from langgraph.types import interrupt

def human_review_node(state: State) -> dict:
    """Pauses execution and waits for human input before continuing."""
    # Interrupt sends payload to UI/User and halts execution until resumed
    user_response = interrupt({
        "question": "Please verify retrieved context",
        "chunks": state.get("precision_chunks", [])
    })
    
    return {"approved": user_response.get("approved", False)}

# Resume an interrupted graph execution:
# app.invoke(Command(resume={"approved": True}), config=config)
```

*(Note: Static `interrupt_before=["node_name"]` on `workflow.compile()` is still supported for simple static breakpoints).*

---

## 9. Building Our Agentic RAG Graph (Full Implementation)

### File Structure
```
app/
├── agents/
│   ├── __init__.py
│   ├── state.py          # GraphState definition
│   ├── graph.py          # Graph assembly & compilation
│   └── nodes/
│       ├── __init__.py
│       ├── router.py     # Intent classification
│       ├── planner.py    # Query decomposition
│       ├── retriever.py  # Hybrid retrieval node
│       ├── grader.py     # Evidence evaluation
│       ├── rewriter.py   # Query reformulation
│       └── generator.py  # Answer generation
```

---

### `app/agents/state.py`

```python
from typing import TypedDict, List, Annotated, Optional
import operator
from app.models import RetrievedChunk

class GraphState(TypedDict):
    # ── Input ────────────────────────────────────────────────────
    query: str                              # Original user question
    
    # ── Router Output ────────────────────────────────────────────
    route: str                              # "OUT_OF_SCOPE" | "SIMPLE_QA" | "COMPLEX_QA"
    category: str                           # "syllabus" | "faculty" | "fees" | "admin"
    department: Optional[str]              # "CSE", "ECE", None
    regulation: Optional[str]             # "R20", "R23", None
    is_aggregate: bool                     # True for "list all HODs"
    
    # ── Planner Output ───────────────────────────────────────────
    sub_queries: List[str]                 # Decomposed sub-searches
    
    # ── Retrieval Output ─────────────────────────────────────────
    # Reducer: parallel sub-query results are APPENDED, not replaced
    candidate_chunks: Annotated[List[RetrievedChunk], operator.add]
    precision_chunks: List[RetrievedChunk] # After cross-encoder reranking
    
    # ── Grader Output ────────────────────────────────────────────
    is_sufficient: bool                    # True = enough evidence
    
    # ── Loop Control ─────────────────────────────────────────────
    retry_count: int                       # Prevents infinite loops (max=1)
    
    # ── Final Output ─────────────────────────────────────────────
    answer: str
    sources: List[str]
    elapsed_seconds: float
```

---

### `app/agents/nodes/router.py`

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional
from langchain_groq import ChatGroq
from app.agents.state import GraphState
import logfire

KNOWLEDGE_CATALOG = {
    "departments": ["CSE", "ECE", "EEE", "CIVIL", "MECH", "IT", "AIDS", "CSBS"],
    "regulations":  ["R19", "R20", "R23", "R24"],
    "default_regulation": "R23",
}

class RouteDecision(BaseModel):
    route: Literal["OUT_OF_SCOPE", "SIMPLE_QA", "COMPLEX_QA"]
    category: str = Field(description="syllabus | faculty | fees | admin | general")
    department: Optional[str] = Field(default=None)
    regulation: Optional[str] = Field(default=None)
    is_aggregate: bool = Field(default=False)
    confidence: float = Field(ge=0.0, le=1.0)

ROUTER_PROMPT = """
You are a query classifier for SRKR Engineering College AI assistant.

ROUTING RULES:
- OUT_OF_SCOPE: Query is NOT about SRKR college
- SIMPLE_QA:    Single, factual college question with a clear direct answer
- COMPLEX_QA:   Aggregate queries ("list all HODs"), ambiguous (no regulation), multi-dept

CATALOG:
- Departments: CSE, ECE, EEE, CIVIL, MECH, IT, AIDS, CSBS
- Regulations: R19, R20, R23, R24 (Default: R23)

Query: {query}

Classify strictly. Return structured JSON.
"""

@logfire.instrument("Router Node")
def router_node(state: GraphState) -> dict:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    structured_llm = llm.with_structured_output(RouteDecision)
    
    decision: RouteDecision = structured_llm.invoke(
        ROUTER_PROMPT.format(query=state["query"])
    )
    
    logfire.info("Router decision", route=decision.route, confidence=decision.confidence)
    
    return {
        "route":        decision.route,
        "category":     decision.category,
        "department":   decision.department,
        "regulation":   decision.regulation,
        "is_aggregate": decision.is_aggregate,
    }
```

---

### `app/agents/nodes/planner.py`

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_groq import ChatGroq
from app.agents.state import GraphState
import logfire

KNOWLEDGE_CATALOG = {
    "departments": ["CSE", "ECE", "EEE", "CIVIL", "MECH", "IT", "AIDS", "CSBS"],
    "regulations":  ["R20", "R23"],
}

class QueryPlan(BaseModel):
    sub_queries: List[str] = Field(
        description="List of 2-8 specific sub-queries to run in parallel",
        min_length=1,
        max_length=8,
    )
    reasoning: str = Field(description="Why you decomposed the query this way")

PLANNER_PROMPT = """
You are a query planner for SRKR Engineering College vector database.

TASK: Decompose the complex query into 2-8 specific, targeted sub-queries for parallel search.

RULES:
1. If department is unknown but dept-specific, create one sub-query per relevant department.
2. If regulation is unknown but regulation-specific, create one sub-query per regulation.
3. For aggregate queries (e.g., "all HODs"), create one sub-query per department.
4. Each sub-query must be self-contained and specific.
5. Maximum 8 sub-queries total. Keep each under 15 words.

ORIGINAL QUERY: {query}
DETECTED CATEGORY: {category}
DETECTED DEPARTMENT: {department}
DETECTED REGULATION: {regulation}
IS AGGREGATE: {is_aggregate}

Available Departments: {departments}
Available Regulations: {regulations}
"""

@logfire.instrument("Planner Node")
def planner_node(state: GraphState) -> dict:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    structured_llm = llm.with_structured_output(QueryPlan)
    
    plan: QueryPlan = structured_llm.invoke(
        PLANNER_PROMPT.format(
            query=state["query"],
            category=state.get("category", "general"),
            department=state.get("department", "unknown"),
            regulation=state.get("regulation", "unknown"),
            is_aggregate=state.get("is_aggregate", False),
            departments=", ".join(KNOWLEDGE_CATALOG["departments"]),
            regulations=", ".join(KNOWLEDGE_CATALOG["regulations"]),
        )
    )
    
    logfire.info("Planner output", sub_queries=plan.sub_queries, count=len(plan.sub_queries))
    
    return {"sub_queries": plan.sub_queries}
```

---

### `app/agents/nodes/retriever.py`

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.agents.state import GraphState
from app.services.retrieval.retriever import HybridRetriever
from app.services.retrieval.reranker import JinaReranker
from app.models import RetrievedChunk
import logfire

# Create once at module level — NOT inside the function
retriever = HybridRetriever()
reranker  = JinaReranker()

@logfire.instrument("Retrieval Node")
def retrieve_and_rerank_node(state: GraphState) -> dict:
    """
    Executes hybrid retrieval across all sub-queries in parallel,
    merges results, and precision-reranks with Jina Cross-Encoder.
    """
    queries = state.get("sub_queries") or [state["query"]]
    original_query = state["query"]
    
    all_chunks: list[RetrievedChunk] = []
    
    # ── Parallel Retrieval via ThreadPool ─────────────────────────────
    with ThreadPoolExecutor(max_workers=min(len(queries), 8)) as executor:
        futures = {
            executor.submit(
                retriever.retrieve,
                query=q,
                top_k_dense=40,
                top_k_final=20,
            ): q
            for q in queries
        }
        for future in as_completed(futures):
            try:
                chunks = future.result()
                all_chunks.extend(chunks)
            except Exception as e:
                logfire.error("Retrieval sub-query failed", error=str(e))
    
    # ── Deduplicate by chunk ID ───────────────────────────────────────
    seen_ids = set()
    unique_chunks = []
    for chunk in all_chunks:
        if chunk.id not in seen_ids:
            seen_ids.add(chunk.id)
            unique_chunks.append(chunk)
    
    if not unique_chunks:
        return {"candidate_chunks": [], "precision_chunks": []}
    
    # ── Cross-Encoder Reranking ───────────────────────────────────────
    precision_chunks = reranker.rerank(
        query=original_query,
        chunks=unique_chunks,
        top_n=5,
    )
    
    logfire.info(
        "Retrieval complete",
        total_retrieved=len(all_chunks),
        unique=len(unique_chunks),
        precision=len(precision_chunks),
    )
    
    return {
        "candidate_chunks": unique_chunks,
        "precision_chunks": precision_chunks,
    }
```

---

### `app/agents/nodes/grader.py`

```python
from app.agents.state import GraphState
import logfire

@logfire.instrument("Grader Node")
def grade_evidence_node(state: GraphState) -> dict:
    """
    Deterministic evidence evaluator.
    Checks if retrieved chunks are sufficient to generate a grounded answer.
    """
    chunks = state.get("precision_chunks", [])
    is_aggregate = state.get("is_aggregate", False)
    category = state.get("category", "general")
    
    # ── Rule 1: No chunks at all ──────────────────────────────────────
    if not chunks:
        return {"is_sufficient": False}
    
    # ── Rule 2: Low confidence score ─────────────────────────────────
    avg_score = sum(c.score for c in chunks[:3]) / min(len(chunks), 3)
    if avg_score < 0.25:
        return {"is_sufficient": False}
    
    # ── Rule 3: Aggregate faculty coverage check ──────────────────────
    if is_aggregate and category == "faculty":
        text_corpus = " ".join(c.text.lower() for c in chunks)
        covered_depts = sum(
            1 for dept in ["cse", "ece", "mech", "civil", "it", "eee", "aids", "csbs"]
            if dept in text_corpus
        )
        if covered_depts < 3:
            return {"is_sufficient": False}
    
    logfire.info("Grader: Evidence is sufficient", avg_score=avg_score)
    return {"is_sufficient": True}
```

---

### `app/agents/nodes/rewriter.py`

```python
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from app.agents.state import GraphState
import logfire

REWRITE_PROMPT = """
The following search query failed to retrieve relevant documents from SRKR Engineering College database.

ORIGINAL QUERY: {query}
RETRY COUNT: {retry_count}

Rewrite the query to be more specific, including relevant academic keywords (regulation numbers, department codes, subject codes).

Return ONLY the rewritten query. No explanation.
"""

@logfire.instrument("Rewriter Node")
def rewrite_query_node(state: GraphState) -> dict:
    """Self-correction: Reformulates the query for better retrieval."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
    
    new_query = llm.invoke(
        [HumanMessage(content=REWRITE_PROMPT.format(
            query=state["query"],
            retry_count=state.get("retry_count", 0),
        ))]
    ).content.strip()
    
    logfire.info("Query rewritten", original=state["query"], rewritten=new_query)
    
    return {
        "query":            new_query,
        "retry_count":      state.get("retry_count", 0) + 1,
        "candidate_chunks": [],
        "precision_chunks": [],
        "sub_queries":      [],
    }
```

---

### `app/agents/nodes/generator.py`

```python
from app.agents.state import GraphState
from app.services.generation.generator import LLMGenerator
import logfire

# Create once at module level
generator = LLMGenerator()

@logfire.instrument("Generator Node")
def generate_answer_node(state: GraphState) -> dict:
    """Generates grounded answer from precision chunks."""
    chunks = state.get("precision_chunks", [])
    query = state["query"]
    
    degraded_notice = ""
    if not state.get("is_sufficient", True):
        degraded_notice = (
            "\n\n⚠️ Note: Official records may not contain complete information "
            "for all aspects of this query. The answer is based on available data."
        )
    
    answer, provider = generator.generate(query=query, chunks=chunks)
    sources = list(dict.fromkeys(c.source for c in chunks))
    
    logfire.info("Answer generated", provider=provider, source_count=len(sources))
    
    return {
        "answer":  answer + degraded_notice,
        "sources": sources,
    }

def direct_reply_node(state: GraphState) -> dict:
    """Instant rejection for out-of-scope queries."""
    logfire.info("Out-of-scope query rejected", query=state["query"])
    return {
        "answer": (
            "I am the SRKR Engineering College Academic Advisor. "
            "I can only assist with questions about SRKR college — admissions, "
            "syllabi, faculty, regulations, fees, and campus information."
        ),
        "sources":          [],
        "precision_chunks": [],
    }
```

---

### `app/agents/graph.py` — The Main Assembly

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.state import GraphState
from app.agents.nodes.router    import router_node
from app.agents.nodes.planner   import planner_node
from app.agents.nodes.retriever import retrieve_and_rerank_node
from app.agents.nodes.grader    import grade_evidence_node
from app.agents.nodes.rewriter  import rewrite_query_node
from app.agents.nodes.generator import generate_answer_node, direct_reply_node

# ── Routing Functions ─────────────────────────────────────────────────────────

def decide_after_routing(state: GraphState) -> str:
    route = state.get("route", "SIMPLE_QA")
    if route == "OUT_OF_SCOPE":
        return "direct_reply"
    elif route == "COMPLEX_QA":
        return "plan_queries"
    else:
        return "retrieve"

def decide_after_grading(state: GraphState) -> str:
    if state.get("is_sufficient", False):
        return "generate"
    elif state.get("retry_count", 0) < 1:
        return "rewrite"        # ← LOOP BACK
    else:
        return "generate"       # Graceful degradation after 1 retry

# ── Graph Assembly ────────────────────────────────────────────────────────────

def create_agentic_rag_graph(use_memory: bool = False):
    workflow = StateGraph(GraphState)
    
    # 1. Register Nodes
    workflow.add_node("route_query",     router_node)
    workflow.add_node("plan_queries",    planner_node)
    workflow.add_node("retrieve",        retrieve_and_rerank_node)
    workflow.add_node("grade_evidence",  grade_evidence_node)
    workflow.add_node("rewrite_query",   rewrite_query_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("direct_reply",    direct_reply_node)
    
    # 2. Entry Point using START node
    workflow.add_edge(START, "route_query")
    
    # 3. Conditional Edge: Routing → 3-way branch
    workflow.add_conditional_edges(
        "route_query",
        decide_after_routing,
        {
            "direct_reply": "direct_reply",
            "plan_queries": "plan_queries",
            "retrieve":     "retrieve",
        }
    )
    
    # 4. Planner → Retrieval (always)
    workflow.add_edge("plan_queries", "retrieve")
    
    # 5. Retrieval → Grader (always)
    workflow.add_edge("retrieve", "grade_evidence")
    
    # 6. Conditional Edge: Grader → Generate OR Rewrite (the loop decision)
    workflow.add_conditional_edges(
        "grade_evidence",
        decide_after_grading,
        {
            "generate": "generate_answer",
            "rewrite":  "rewrite_query",
        }
    )
    
    # 7. THE LOOP: Rewriter → back to Retrieval
    workflow.add_edge("rewrite_query", "retrieve")
    
    # 8. Terminal Edges
    workflow.add_edge("direct_reply",    END)
    workflow.add_edge("generate_answer", END)
    
    checkpointer = InMemorySaver() if use_memory else None
    return workflow.compile(checkpointer=checkpointer)


# Singleton — import and use this in your API
rag_graph = create_agentic_rag_graph(use_memory=True)
```

---

### Running the Graph

```python
from app.agents.graph import rag_graph

def run_query(query: str, session_id: str = "default") -> dict:
    config = {"configurable": {"thread_id": session_id}}
    initial_state = {
        "query":            query,
        "retry_count":      0,
        "candidate_chunks": [],
        "precision_chunks": [],
        "sub_queries":      [],
    }
    result = rag_graph.invoke(initial_state, config=config)
    return {
        "answer":  result["answer"],
        "sources": result.get("sources", []),
        "route":   result.get("route"),
    }

# Tests
print(run_query("Who are all the HODs in our college?"))   # Complex QA
print(run_query("Who won the IPL?"))                       # Out-of-scope
print(run_query("What is AIML syllabus?"))                 # Complex (ambiguous)
```

---

## 10. Debugging & Visualization

### Visualize the Graph (Mermaid Diagram)

```python
from app.agents.graph import rag_graph

# Print as Mermaid (paste at mermaid.live)
print(rag_graph.get_graph().draw_mermaid())

# Save as PNG (requires: pip install pygraphviz)
rag_graph.get_graph().draw_mermaid_png(output_file_path="docs/agent_graph.png")
```

### Stream Events (See Every Step Live)

```python
for event in rag_graph.stream(
    {"query": "What is the CSE R23 syllabus?", "retry_count": 0},
    stream_mode="updates"
):
    node_name = list(event.keys())[0]
    node_output = event[node_name]
    print(f"\n🔵 Node: {node_name}")
    for key, val in node_output.items():
        print(f"   {key}: {str(val)[:120]}")
```

### Sample Stream Output

```
🔵 Node: route_query
   route: SIMPLE_QA
   category: syllabus
   department: CSE
   regulation: R23

🔵 Node: retrieve
   candidate_chunks: [<20 chunks>]
   precision_chunks: [<5 top chunks>]

🔵 Node: grade_evidence
   is_sufficient: True

🔵 Node: generate_answer
   answer: The CSE R23 syllabus includes the following subjects...
   sources: ['syllabus/cse_r23_semester1.pdf', ...]
```

---

## 11. LangGraph vs Raw LangChain vs From Scratch

| Feature | Current Code (From Scratch) | LangChain Chains | LangGraph |
| :--- | :--- | :--- | :--- |
| **Loops/Retries** | Manual `while` + complex state | ❌ Not supported | ✅ Native cycles |
| **Branching** | Manual `if/else` | Routers (limited) | ✅ Conditional edges |
| **Parallel Execution** | Manual `ThreadPoolExecutor` | ❌ Sequential | ✅ Fan-out/Fan-in |
| **State Management** | Manual `dict` passing | Limited | ✅ TypedDict + Reducers |
| **Streaming** | Custom | Basic | ✅ Event streaming per node |
| **Memory/Persistence** | Custom | Custom | ✅ Built-in Checkpointing |
| **Debugging** | Print statements | LangSmith traces | ✅ LangSmith + Step streaming |
| **Maintainability** | Hard as complexity grows | Medium | ✅ Modular & visual |

---

## 12. Common Mistakes & How to Avoid Them

### ❌ Mistake 1: Returning full state from node

```python
# WRONG
def my_node(state: State) -> dict:
    return state  # Causes unpredictable merging issues

# CORRECT — return only what changed
def my_node(state: State) -> dict:
    return {"answer": "computed answer"}
```

### ❌ Mistake 2: Infinite loops (no retry counter)

```python
# WRONG — no exit condition
def decide(state):
    if not state["docs"]:
        return "retry"  # Can loop forever!

# CORRECT — always check retry_count
def decide(state):
    if state["docs"]:                       return "generate"
    elif state.get("retry_count", 0) < 1:  return "retry"
    else:                                   return "generate"  # Force exit
```

### ❌ Mistake 3: Not using reducers for parallel branches

```python
# WRONG — last branch to finish overwrites all others
class State(TypedDict):
    chunks: List[str]   # No reducer → DANGER in parallel

# CORRECT
class State(TypedDict):
    chunks: Annotated[List[str], operator.add]  # Appends from each branch
```

### ❌ Mistake 4: Creating resource instances inside node functions

```python
# WRONG — creates new connections on every call (slow + resource leak)
def retrieve_node(state):
    retriever = HybridRetriever()  # NEW instance every time!
    ...

# CORRECT — create once at module level
retriever = HybridRetriever()  # Created once on import

def retrieve_node(state):
    results = retriever.retrieve(state["query"])
    ...
```

### ❌ Mistake 5: Forgetting entry point or END edge

```python
# Both of these are REQUIRED or the graph will error at compile time:
workflow.add_edge(START, "first_node")   # Where does it start?
workflow.add_edge("last_node", END)      # Where does it end?
```

---

## Summary: LangGraph in One Picture

```
Your Agent = State Machine
                │
    ┌───────────┼───────────┐
    │           │           │
  State        Nodes      Edges
(TypedDict)  (Python     (add_edge /
  Shared      functions)  add_conditional_edges)
  memory
    │           │           │
    └───────────┼───────────┘
                │
         compile() → app
                │
    app.invoke()    → Full final state
    app.stream()    → Step-by-step events (great for UI)
```

### Build Order for This Project

1. ✅ Define `GraphState` in `state.py`
2. ✅ Write each node as an isolated, testable Python function in `nodes/`
3. ✅ Unit-test each node with mock state dicts (no LLM needed)
4. ✅ Assemble the graph in `graph.py`
5. ✅ Integration test with `app.stream()` to see every step
6. ✅ Wrap `rag_graph` in FastAPI endpoints with SSE streaming
7. ✅ Build the chat UI consuming the SSE stream

---
*Related docs: [`architecture_specification.md`](./architecture_specification.md) · [`implement.md`](./implement.md)*
