import argparse
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path


# =========================================================
# Make project root importable
# Allows:
# uv run Ingestion/pipeline.py
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# External imports
# =========================================================

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)


# reduce the chunks complete ingestions 
from config import settings

from app.ingestion.loaders.md_loader import load_markdown_files
from app.ingestion.loaders.pdf_loader import load_pdf_files

from app.ingestion.processor import process_documents

from app.ingestion.chunking.chunker import chunk_documents

from app.services.retrieval.embedding import (
    JinaEmbedder,
    EMBEDDING_DIM,
)


# =========================================================
# Paths
# =========================================================

RAW_DATA_DIR = Path("Data/raw")

DATA_DIR = Path("Data/processeddata")

CHUNKS_FILE = DATA_DIR / "chunks.jsonl"

EMBEDDINGS_FILE = DATA_DIR / "embeddings.jsonl"


# =========================================================
# Settings
# =========================================================

EMBED_BATCH_SIZE = 32

QDRANT_BATCH_SIZE = 100

TEST_DOCUMENTS_PER_TYPE = 3

TEST_CHUNKS = 10

TEST_TOP_K = 3


# =========================================================
# Unicode safety
# =========================================================

def safe_unicode(value):
    """
    Replace invalid Unicode surrogate characters.

    Some PDFs contain malformed/incomplete Unicode such as:
        \\ud83e

    This prevents UTF-8 / JSON / hashing errors.
    """

    if isinstance(value, str):
        return value.encode(
            "utf-8",
            errors="replace",
        ).decode("utf-8")

    if isinstance(value, dict):
        return {
            key: safe_unicode(val)
            for key, val in value.items()
        }

    if isinstance(value, list):
        return [
            safe_unicode(item)
            for item in value
        ]

    return value


# =========================================================
# JSONL helpers
# =========================================================

def save_jsonl(
    path: Path,
    rows: list[dict],
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:

            row = safe_unicode(row)

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def append_jsonl(
    path: Path,
    rows: list[dict],
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:

        for row in rows:

            row = safe_unicode(row)

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_jsonl(
    path: Path,
) -> list[dict]:

    if not path.exists():
        return []

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                rows.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Invalid JSON in "
                    f"{path} at line "
                    f"{line_number}: {e}"
                ) from e

    return rows


# =========================================================
# Chunk ID
# =========================================================

def make_chunk_id(
    text: str,
    metadata: dict,
) -> str:

    text = safe_unicode(text)
    metadata = safe_unicode(metadata)

    source = (
        metadata.get("source_url")
        or metadata.get("source")
        or "unknown"
    )

    source = safe_unicode(str(source))

    raw = (
        f"{source}|"
        f"{text}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


# =========================================================
# Build chunks
# =========================================================

def build_chunks():
    """
    Load → process → chunk → save chunks.

    If chunks.jsonl already exists, reuse it.
    """

    print("\n" + "─" * 50)
    print("  STAGE 1 — Document Chunking")
    print("─" * 50)

    if CHUNKS_FILE.exists():

        print(
            f"  [CACHE HIT] Loading from "
            f"{CHUNKS_FILE}"
        )

        chunks = load_jsonl(
            CHUNKS_FILE
        )

        print(
            f"  ✓ Loaded {len(chunks):,} cached chunks."
        )

        return chunks

    print("  Loading raw documents from disk...")

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    t0 = time.time()

    markdown_docs = load_markdown_files(
        str(RAW_DATA_DIR)
    )

    pdf_docs = load_pdf_files(
        str(RAW_DATA_DIR)
    )

    # Sort documents to ensure deterministic order across runs
    markdown_docs = sorted(
        markdown_docs,
        key=lambda d: str(d.metadata.get("source", "")),
    )

    pdf_docs = sorted(
        pdf_docs,
        key=lambda d: (
            str(d.metadata.get("source", "")),
            d.metadata.get("page", 0),
        ),
    )

    print(
        f"  ✓ Loaded {len(markdown_docs):,} markdown pages "
        f"and {len(pdf_docs):,} PDF pages "
        f"({time.time() - t0:.1f}s)"
    )

    # -----------------------------------------------------
    # Process
    # -----------------------------------------------------

    print("  Processing documents (cleaning / filtering)...")

    t1 = time.time()

    markdown_docs = process_documents(
        markdown_docs,
        "markdown",
    )

    pdf_docs = process_documents(
        pdf_docs,
        "pdf",
    )

    print(
        f"  ✓ Processed {len(markdown_docs) + len(pdf_docs):,} documents "
        f"({time.time() - t1:.1f}s)"
    )

    # -----------------------------------------------------
    # Chunk
    # -----------------------------------------------------

    print("  Chunking documents...")

    t2 = time.time()

    markdown_chunks = chunk_documents(
        markdown_docs,
        "markdown",
    )

    pdf_chunks = chunk_documents(
        pdf_docs,
        "pdf",
    )

    chunks = (
        markdown_chunks +
        pdf_chunks
    )

    print(
        f"  ✓ Chunked: {len(markdown_chunks):,} markdown "
        f"+ {len(pdf_chunks):,} PDF "
        f"= {len(chunks):,} total "
        f"({time.time() - t2:.1f}s)"
    )

    # -----------------------------------------------------
    # Prepare cache
    # -----------------------------------------------------

    rows = []

    for index, doc in enumerate(chunks):

        if not isinstance(
            doc.page_content,
            str,
        ):
            print(
                f"Skipping chunk {index}: "
                f"invalid page_content type"
            )
            continue

        text = safe_unicode(
            doc.page_content
        ).strip()

        if not text:
            continue

        metadata = safe_unicode(
            dict(doc.metadata)
        )

        rows.append(
            {
                "index": index,
                "chunk_id": make_chunk_id(
                    text,
                    metadata,
                ),
                "text": text,
                "metadata": metadata,
            }
        )

    t3 = time.time()

    save_jsonl(
        CHUNKS_FILE,
        rows,
    )

    print(
        f"  ✓ Saved {len(rows):,} chunks → "
        f"{CHUNKS_FILE} "
        f"({time.time() - t3:.1f}s)"
    )

    return rows


# =========================================================
# Build embeddings
# =========================================================

def build_embeddings(
    chunks: list[dict],
):
    """
    Embed chunks that don't already have embeddings.

    Every successful batch is immediately written to disk.
    Therefore the pipeline can safely resume.
    """

    print("\n" + "─" * 50)
    print("  STAGE 2 — Jina Embeddings")
    print("─" * 50)
    print("  Loading existing checkpoint embeddings...")

    existing_rows = load_jsonl(
        EMBEDDINGS_FILE
    )

    existing = {}

    for row in existing_rows:

        chunk_id = row.get("chunk_id")

        vector = row.get("vector")

        if not chunk_id or not vector:
            continue

        if len(vector) != EMBEDDING_DIM:
            print(
                f"Skipping cached embedding "
                f"with wrong dimension: "
                f"{len(vector)}"
            )
            continue

        existing[chunk_id] = row

    print(
        f"  ✓ Checkpoint: {len(existing):,} embeddings already done"
    )

    remaining = [
        chunk
        for chunk in chunks
        if chunk["chunk_id"] not in existing
    ]

    total_batches = (
        len(remaining) + EMBED_BATCH_SIZE - 1
    ) // EMBED_BATCH_SIZE

    print(
        f"  Remaining: {len(remaining):,} chunks "
        f"({total_batches} API batches of {EMBED_BATCH_SIZE})"
    )

    if not remaining:

        print("  ✓ All chunks already embedded — skipping Jina.")

        return existing

    embedder = JinaEmbedder()

    print(
        f"  API keys loaded: {len(embedder.api_keys)} key(s) "
        f"(key rotation enabled)"
    )

    embed_start = time.time()

    try:

        for start in range(
            0,
            len(remaining),
            EMBED_BATCH_SIZE,
        ):

            batch = remaining[
                start:
                start + EMBED_BATCH_SIZE
            ]

            texts = [
                safe_unicode(
                    item["text"]
                )
                for item in batch
            ]

            batch_num = start // EMBED_BATCH_SIZE + 1
            elapsed = time.time() - embed_start
            chunks_done = start + len(existing)
            pct = (start / len(remaining) * 100) if remaining else 100

            print(
                f"\n  [{batch_num}/{total_batches}] "
                f"Embedding chunks {start + 1}–{start + len(batch)} "
                f"/ {len(remaining):,} "
                f"({pct:.1f}%) | "
                f"elapsed {elapsed:.0f}s"
            )

            # -------------------------------------------------
            # Jina
            # -------------------------------------------------

            vectors = (
                embedder.embed_documents(
                    texts
                )
            )

            if len(vectors) != len(batch):

                raise RuntimeError(
                    "Jina returned "
                    f"{len(vectors)} vectors "
                    f"for {len(batch)} texts."
                )

            rows = []

            for chunk, vector in zip(
                batch,
                vectors,
            ):

                if len(vector) != EMBEDDING_DIM:

                    raise RuntimeError(
                        "Wrong embedding dimension: "
                        f"expected {EMBEDDING_DIM}, "
                        f"got {len(vector)}"
                    )

                row = {
                    "index": chunk["index"],
                    "chunk_id": chunk["chunk_id"],
                    "vector": vector,
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                }

                rows.append(row)

                existing[
                    chunk["chunk_id"]
                ] = row

            # -------------------------------------------------
            # Checkpoint
            # -------------------------------------------------

            append_jsonl(
                EMBEDDINGS_FILE,
                rows,
            )

            print(
                f"  ✓ Checkpoint saved. "
                f"Total embedded so far: {len(existing):,}"
            )

        print(
            f"\n  ✓ Embedding complete. "
            f"{len(existing):,} total embeddings "
            f"(took {time.time() - embed_start:.1f}s)"
        )

    finally:

        embedder.close()


    return existing


# =========================================================
# Qdrant client
# =========================================================

def get_qdrant_client():

    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
    )


# =========================================================
# Get collection name
# =========================================================

def get_production_collection_name():

    return settings.QDRANT_COLLECTION_NAME


def get_test_collection_name():

    return (
        f"{get_production_collection_name()}_test"
    )


# =========================================================
# Ensure Qdrant collection
# =========================================================

def ensure_collection(
    client: QdrantClient,
    collection_name: str,
):
    """
    Create collection if it doesn't exist.

    If it already exists, verify its dimension.
    """

    collections = client.get_collections()

    collection_names = {
        collection.name
        for collection in collections.collections
    }

    # -----------------------------------------------------
    # Create
    # -----------------------------------------------------

    if collection_name not in collection_names:

        print(
            f"Creating Qdrant collection: "
            f"{collection_name}"
        )

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )

        return

    # -----------------------------------------------------
    # Validate existing collection
    # -----------------------------------------------------

    print(
        f"Qdrant collection already exists: "
        f"{collection_name}"
    )

    info = client.get_collection(
        collection_name
    )

    vectors_config = (
        info.config.params.vectors
    )

    actual_dimension = getattr(
        vectors_config,
        "size",
        None,
    )

    if actual_dimension is None:

        raise RuntimeError(
            "Could not determine Qdrant "
            "vector dimension."
        )

    if actual_dimension != EMBEDDING_DIM:

        raise RuntimeError(
            f"Qdrant collection "
            f"'{collection_name}' uses "
            f"{actual_dimension} dimensions, "
            f"but current embedding model "
            f"uses {EMBEDDING_DIM}."
        )

    print(
        f"Qdrant dimension OK: "
        f"{actual_dimension}"
    )


# =========================================================
# Create Qdrant points
# =========================================================

def create_points(
    chunks: list[dict],
    embeddings: dict,
):
    points = []

    missing = 0

    for chunk in chunks:

        row = embeddings.get(
            chunk["chunk_id"]
        )

        if row is None:

            missing += 1
            continue

        vector = row["vector"]

        if len(vector) != EMBEDDING_DIM:

            raise RuntimeError(
                f"Invalid vector dimension "
                f"for chunk "
                f"{chunk['chunk_id']}"
            )

        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                chunk["chunk_id"],
            )
        )

        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "page_content": row["text"],
                    "text": row["text"],
                    "chunk_id": row["chunk_id"],
                    **row["metadata"],
                },
            )
        )

    if missing:

        raise RuntimeError(
            f"{missing} chunks have no "
            f"embedding. Refusing to upload "
            f"an incomplete dataset."
        )

    return points


# =========================================================
# Upload to Qdrant
# =========================================================

def upload_to_qdrant(
    chunks: list[dict],
    embeddings: dict,
    collection_name: str | None = None,
):
    """
    Upload all embeddings to Qdrant.
    """

    if collection_name is None:
        collection_name = (
            get_production_collection_name()
        )

    client = get_qdrant_client()

    try:

        ensure_collection(
            client,
            collection_name,
        )

        points = create_points(
            chunks,
            embeddings,
        )

        total_batches_q = (
            len(points) + QDRANT_BATCH_SIZE - 1
        ) // QDRANT_BATCH_SIZE

        print(
            f"  Uploading {len(points):,} vectors → "
            f"'{collection_name}' "
            f"({total_batches_q} upload batches of {QDRANT_BATCH_SIZE})"
        )

        upload_start = time.time()

        for start in range(
            0,
            len(points),
            QDRANT_BATCH_SIZE,
        ):

            batch = points[
                start:
                start + QDRANT_BATCH_SIZE
            ]

            client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=True,
            )

            uploaded = min(
                start + len(batch),
                len(points),
            )
            pct = uploaded / len(points) * 100
            batch_num_q = start // QDRANT_BATCH_SIZE + 1

            print(
                f"  [{batch_num_q}/{total_batches_q}] "
                f"Uploaded {uploaded:,}/{len(points):,} "
                f"({pct:.1f}%)"
            )

        print(
            f"  ✓ Qdrant upload complete. "
            f"{len(points):,} vectors "
            f"(took {time.time() - upload_start:.1f}s)"
        )

    finally:

        client.close()


# =========================================================
# Full pipeline
# =========================================================

def run_pipeline(
    collection_name: str | None = None,
):

    pipeline_start = time.time()

    print("\n" + "═" * 50)
    print("        FULL INGESTION PIPELINE")
    print("═" * 50)
    print(f"  Collection : {collection_name or settings.QDRANT_COLLECTION_NAME}")
    print(f"  Chunk cache: {CHUNKS_FILE}")
    print(f"  Embed cache: {EMBEDDINGS_FILE}")
    print("═" * 50)

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ─────────────────────────────────────────────────────
    # 1. Build/load chunks
    # ─────────────────────────────────────────────────────

    chunks = build_chunks()

    # ─────────────────────────────────────────────────────
    # 2. Build/resume embeddings
    # ─────────────────────────────────────────────────────

    embeddings = build_embeddings(
        chunks
    )

    # ─────────────────────────────────────────────────────
    # 3. Upload
    # ─────────────────────────────────────────────────────

    print("\n" + "─" * 50)
    print("  STAGE 3 — Qdrant Upload")
    print("─" * 50)

    upload_to_qdrant(
        chunks,
        embeddings,
        collection_name=collection_name,
    )

    elapsed = time.time() - pipeline_start
    mins, secs = divmod(int(elapsed), 60)

    print("\n" + "═" * 50)
    print("     ✅ INGESTION COMPLETE")
    print("═" * 50)
    print(f"  Total chunks  : {len(chunks):,}")
    print(f"  Total embedded: {len(embeddings):,}")
    print(f"  Collection    : {collection_name or settings.QDRANT_COLLECTION_NAME}")
    print(f"  Total time    : {mins}m {secs}s")
    print("═" * 50)


# =========================================================
# Test helpers
# =========================================================

def get_test_documents():

    print(
        "\nLoading a small test sample..."
    )

    from langchain_community.document_loaders import TextLoader, PyPDFLoader

    md_files = sorted(list(RAW_DATA_DIR.glob("*.md")))[:TEST_DOCUMENTS_PER_TYPE]
    pdf_dir = RAW_DATA_DIR / "pdfs"
    pdf_files = sorted(
        list(pdf_dir.glob("*.pdf")) or list(RAW_DATA_DIR.rglob("*.pdf"))
    )[:TEST_DOCUMENTS_PER_TYPE]

    markdown_docs = []
    for f in md_files:
        try:
            markdown_docs.extend(
                TextLoader(str(f), encoding="utf-8").load()
            )
        except Exception as e:
            print(f"Warning: Could not load test markdown {f}: {e}")

    pdf_docs = []
    for f in pdf_files:
        try:
            pdf_docs.extend(
                PyPDFLoader(str(f)).load()
            )
        except Exception as e:
            print(f"Warning: Could not load test PDF {f}: {e}")

    markdown_docs = markdown_docs[:TEST_DOCUMENTS_PER_TYPE]
    pdf_docs = pdf_docs[:TEST_DOCUMENTS_PER_TYPE]

    print(
        f"Test Markdown documents: "
        f"{len(markdown_docs)}"
    )

    print(
        f"Test PDF documents: "
        f"{len(pdf_docs)}"
    )

    return (
        markdown_docs,
        pdf_docs,
    )


# =========================================================
# Test pipeline
# =========================================================

def test_pipeline():

    print("\n================================")
    print("         PIPELINE TEST")
    print("================================\n")

    # =====================================================
    # 1. Check paths
    # =====================================================

    print("[1/7] Checking data directory...")

    if not RAW_DATA_DIR.exists():

        raise RuntimeError(
            f"Raw data directory does not exist: "
            f"{RAW_DATA_DIR}"
        )

    print(
        f"✓ Raw data directory: "
        f"{RAW_DATA_DIR}"
    )

    # =====================================================
    # 2. Load
    # =====================================================

    print("\n[2/7] Testing document loading...")

    markdown_docs, pdf_docs = (
        get_test_documents()
    )

    if not markdown_docs and not pdf_docs:

        raise RuntimeError(
            "No test documents were loaded."
        )

    print("✓ Document loading works")

    # =====================================================
    # 3. Process
    # =====================================================

    print("\n[3/7] Testing processing...")

    markdown_docs = process_documents(
        markdown_docs,
        "markdown",
    )

    pdf_docs = process_documents(
        pdf_docs,
        "pdf",
    )

    documents = (
        markdown_docs +
        pdf_docs
    )

    if not documents:

        raise RuntimeError(
            "Processing produced zero documents."
        )

    print(
        f"✓ Processing works: "
        f"{len(documents)} documents"
    )

    # =====================================================
    # 4. Chunk
    # =====================================================

    print("\n[4/7] Testing chunking...")

    markdown_chunks = chunk_documents(
        markdown_docs,
        "markdown",
    )

    pdf_chunks = chunk_documents(
        pdf_docs,
        "pdf",
    )

    chunks = (
        markdown_chunks +
        pdf_chunks
    )

    if not chunks:

        raise RuntimeError(
            "Chunking produced zero chunks."
        )

    chunks = chunks[:TEST_CHUNKS]

    test_rows: list[dict] = []

    for index, doc in enumerate(chunks):

        if not isinstance(
            doc.page_content,
            str,
        ):
            continue

        text = safe_unicode(
            doc.page_content
        ).strip()

        if not text:
            continue

        metadata = safe_unicode(
            dict(doc.metadata)
        )

        test_rows.append(
            {
                "index": index,
                "chunk_id": make_chunk_id(
                    text,
                    metadata,
                ),
                "text": text,
                "metadata": metadata,
            }
        )

    if not test_rows:

        raise RuntimeError(
            "No valid test chunks."
        )

    print(
        f"✓ Chunking works: "
        f"{len(test_rows)} test chunks"
    )

    # =====================================================
    # 5. Jina
    # =====================================================

    print(
        "\n[5/7] Testing Jina embeddings..."
    )

    embedder = JinaEmbedder()

    try:

        texts = [
            row["text"]
            for row in test_rows
        ]

        vectors = (
            embedder.embed_documents(
                texts
            )
        )

        # Also test query embedding.
        query_vector = (
            embedder.embed_query(
                "What information is available?"
            )
        )

    finally:

        embedder.close()

    # -----------------------------------------------------
    # Validate document vectors
    # -----------------------------------------------------

    if len(vectors) != len(texts):

        raise RuntimeError(
            f"Expected {len(texts)} vectors, "
            f"got {len(vectors)}"
        )

    for i, vector in enumerate(vectors):

        if len(vector) != EMBEDDING_DIM:

            raise RuntimeError(
                f"Document vector {i} has "
                f"{len(vector)} dimensions. "
                f"Expected {EMBEDDING_DIM}."
            )

    # -----------------------------------------------------
    # Validate query vector
    # -----------------------------------------------------

    if len(query_vector) != EMBEDDING_DIM:

        raise RuntimeError(
            f"Query vector has "
            f"{len(query_vector)} dimensions. "
            f"Expected {EMBEDDING_DIM}."
        )

    print(
        f"✓ Jina document embeddings work"
    )

    print(
        f"✓ Jina query embedding works"
    )

    print(
        f"✓ Vector dimension: "
        f"{EMBEDDING_DIM}"
    )

    # =====================================================
    # 6. Qdrant
    # =====================================================

    print(
        "\n[6/7] Testing Qdrant..."
    )

    test_collection = (
        get_test_collection_name()
    )

    client = get_qdrant_client()

    try:

        ensure_collection(
            client,
            test_collection,
        )

        points = []

        for row, vector in zip(
            test_rows,
            vectors,
        ):

            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    str(row["chunk_id"]),
                )
            )

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "page_content": row["text"],
                        "text": row["text"],
                        "chunk_id": row["chunk_id"],
                        **row["metadata"],
                    },
                )
            )

        client.upsert(
            collection_name=test_collection,
            points=points,
            wait=True,
        )

        print(
            f"✓ Uploaded {len(points)} "
            f"test vectors"
        )

    finally:

        client.close()

    # =====================================================
    # 7. Test retrieval
    # =====================================================

    print(
        "\n[7/7] Testing vector search..."
    )

    client = get_qdrant_client()

    try:

        results = client.query_points(
            collection_name=test_collection,
            query=query_vector,
            limit=TEST_TOP_K,
            with_payload=True,
        ).points

    finally:

        client.close()

    if not results:

        raise RuntimeError(
            "Qdrant search returned no results."
        )

    print(
        f"✓ Qdrant search returned "
        f"{len(results)} results"
    )

    print("\nTop test result:")

    top_result = results[0]

    print(
        f"Score: {top_result.score}"
    )

    if top_result.payload:

        print(
            f"Chunk ID: "
            f"{top_result.payload.get('chunk_id')}"
        )

        text = top_result.payload.get(
            "text",
            "",
        )

        preview = text[:300].replace(
            "\n",
            " ",
        )

        print(
            f"Text: {preview}..."
        )

    # =====================================================
    # Success
    # =====================================================

    print("\n================================")
    print("       ALL TESTS PASSED")
    print("================================")

    print(
        "\nPipeline components verified:"
    )

    print("  ✓ Data directory")
    print("  ✓ Markdown loader")
    print("  ✓ PDF loader")
    print("  ✓ Document processor")
    print("  ✓ Chunker")
    print("  ✓ Unicode safety")
    print("  ✓ Jina document embeddings")
    print("  ✓ Jina query embedding")
    print("  ✓ 1024-dimensional vectors")
    print("  ✓ Qdrant collection")
    print("  ✓ Qdrant upsert")
    print("  ✓ Qdrant similarity search")

    print(
        "\nTest collection:"
    )

    print(
        f"  {test_collection}"
    )

    print(
        "\nYour embedding pipeline is ready "
        "for the full ingestion."
    )


# =========================================================
# Delete cache
# =========================================================

def rebuild():

    print(
        "\nRebuilding local ingestion cache..."
    )

    if DATA_DIR.exists():
        for file_path in DATA_DIR.glob("*.jsonl"):
            try:
                file_path.unlink()
                print(
                    f"Deleted {file_path}"
                )
            except Exception as e:
                print(
                    f"Could not delete {file_path}: {e}"
                )

    print(
        "Local cache cleared."
    )


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description="SRKR RAG ingestion pipeline"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Run a small end-to-end test "
            "including Jina and Qdrant."
        ),
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Delete local chunk and "
            "embedding caches before ingestion."
        ),
    )

    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help=(
            "Name of the Qdrant collection to ingest into "
            "(defaults to settings.QDRANT_COLLECTION_NAME)."
        ),
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Test
    # -----------------------------------------------------

    if args.test:

        test_pipeline()
        return

    # -----------------------------------------------------
    # Rebuild
    # -----------------------------------------------------

    if args.rebuild:

        rebuild()

    # -----------------------------------------------------
    # Full pipeline
    # -----------------------------------------------------

    run_pipeline(
        collection_name=args.collection
    )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    main()