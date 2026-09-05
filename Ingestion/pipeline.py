import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import settings
from Ingestion.loaders import load_markdown_files, load_pdf_files
from Ingestion.processor import process_documents
from Ingestion.chunking.chunker import chunk_documents

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# File paths
PROCESSED_DIR = Path(PROJECT_ROOT) / "Data" / "processeddata"
CHUNKS_FILE = PROCESSED_DIR / "chunks_inspection.jsonl"
EMBEDDINGS_FILE = PROCESSED_DIR / "embeddings_cache.jsonl"

# Local embedding model — 768d, no API key, no rate limits
LOCAL_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM = 768
BATCH_SIZE = 64  # chunks per forward pass; lower if you run out of RAM


def get_chunks() -> list[dict]:
    """Return chunks from cache if available, otherwise parse raw data and save."""
    if CHUNKS_FILE.exists():
        print(f"Loading cached chunks from {CHUNKS_FILE}")
        chunks = []
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
        print(f"  -> {len(chunks)} chunks loaded.")
        return chunks

    print("No cache found. Parsing raw data (may take a while)...")
    DATA_DIR = Path(PROJECT_ROOT) / "Data" / "raw"
    pdf_dir = DATA_DIR / "pdfs"

    raw_md = load_markdown_files(str(DATA_DIR))
    raw_pdf = load_pdf_files(str(pdf_dir)) if pdf_dir.exists() else []

    md_chunks = chunk_documents(process_documents(raw_md, "markdown"), "markdown")
    pdf_chunks = chunk_documents(process_documents(raw_pdf, "pdf"), "pdf")
    all_chunks = md_chunks + pdf_chunks

    if not all_chunks:
        print("No chunks produced from raw data.")
        return []

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    chunks = []
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            record = {"page_content": chunk.page_content, "metadata": chunk.metadata}
            chunks.append(record)
            f.write(json.dumps(record) + "\n")

    print(f"  -> {len(chunks)} chunks saved to cache.")
    return chunks


def run_ingestion() -> None:
    """
    Full ingestion pipeline:
      1. Load / parse chunks
      2. Embed with a local model (resume-safe)
      3. Upload vectors to Qdrant
    """
    # --- 1. Load chunks ---
    chunks = get_chunks()
    if not chunks:
        print("Nothing to ingest. Exiting.")
        return

    total = len(chunks)

    # --- 2. Embed chunks ---
    # Count already-embedded chunks so we can resume after a crash
    already_done = 0
    if EMBEDDINGS_FILE.exists():
        with open(EMBEDDINGS_FILE, "r") as f:
            already_done = sum(1 for _ in f)

    if already_done >= total:
        print(f"All {total} chunks already embedded. Skipping to upload.")
    else:
        if already_done > 0:
            print(f"Resuming from chunk {already_done} ({total - already_done} remaining).")

        print(f"\n--- LOADING MODEL: {LOCAL_MODEL} ---")
        t0 = time.time()
        model = SentenceTransformer(LOCAL_MODEL)
        print(f"  Model ready in {time.time() - t0:.1f}s")

        remaining = chunks[already_done:]
        print(f"\n--- EMBEDDING {len(remaining)} CHUNKS (batch size {BATCH_SIZE}) ---")

        with open(EMBEDDINGS_FILE, "a", encoding="utf-8") as f:
            for i in range(0, len(remaining), BATCH_SIZE):
                batch = remaining[i : i + BATCH_SIZE]
                texts = [c["page_content"] for c in batch]
                start_idx = already_done + i

                print(
                    f"  [{start_idx}–{start_idx + len(batch)}/{total}]",
                    end=" ",
                    flush=True,
                )
                t0 = time.time()
                vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                print(f"✓ {time.time() - t0:.1f}s")

                for j, vec in enumerate(vectors):
                    f.write(json.dumps({"idx": start_idx + j, "vector": vec.tolist()}) + "\n")
                f.flush()

    # --- 3. Upload to Qdrant ---
    if not EMBEDDINGS_FILE.exists():
        print("Embeddings file not found. Cannot upload.")
        return

    print(f"\n--- UPLOADING TO QDRANT ({settings.QDRANT_URL}) ---")
    qdrant = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

    # Create collection if it doesn't exist
    existing = [c.name for c in qdrant.get_collections().collections]
    if settings.QDRANT_COLLECTION_NAME not in existing:
        print(f"Creating collection '{settings.QDRANT_COLLECTION_NAME}' ({EMBEDDING_DIM}d, cosine)...")
        qdrant.create_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

    # Build Qdrant PointStructs from saved embeddings
    embeddings: dict[int, list[float]] = {}
    with open(EMBEDDINGS_FILE, "r") as f:
        for line in f:
            rec = json.loads(line)
            embeddings[rec["idx"]] = rec["vector"]

    points = [
        PointStruct(
            id=idx + 1,  # Qdrant IDs start at 1
            vector=embeddings[idx],
            payload={
                "page_content": chunks[idx]["page_content"],
                "metadata": chunks[idx]["metadata"],
            },
        )
        for idx in range(total)
        if idx in embeddings
    ]

    print(f"Uploading {len(points)} points...")
    upload_batch = 200
    for i in range(0, len(points), upload_batch):
        batch = points[i : i + upload_batch]
        qdrant.upsert(collection_name=settings.QDRANT_COLLECTION_NAME, points=batch)
        print(f"  -> {i + len(batch)}/{len(points)}")

    print(f"\n✅ Done! {len(points)} vectors uploaded to '{settings.QDRANT_COLLECTION_NAME}'.")


if __name__ == "__main__":
    run_ingestion()
