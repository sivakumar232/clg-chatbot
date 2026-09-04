import os
import sys

# Add project root to path so we can import modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import settings
from Ingestion.loaders import load_markdown_files, load_pdf_files
from Ingestion.processor import process_documents
from Ingestion.chunking.chunker import chunk_documents
from Ingestion.embedder import get_embedding_model

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

def run_ingestion():
    DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "raw")
    
    # 1. LOAD RAW DATA
    print("\n--- 1. LOADING RAW DATA ---")
    raw_md_docs = load_markdown_files(DATA_DIR)
    
    pdf_dir = os.path.join(DATA_DIR, "pdfs")
    raw_pdf_docs = load_pdf_files(pdf_dir) if os.path.exists(pdf_dir) else []
    
    # 2. PROCESS & CLEAN
    print("\n--- 2. PROCESSING & CLEANING METADATA ---")
    processed_md = process_documents(raw_md_docs, "markdown")
    processed_pdf = process_documents(raw_pdf_docs, "pdf")
    
    # 3. CHUNK
    print("\n--- 3. CHUNKING TEXT ---")
    md_chunks = chunk_documents(processed_md, "markdown")
    pdf_chunks = chunk_documents(processed_pdf, "pdf")
    
    all_chunks = md_chunks + pdf_chunks
    
    if not all_chunks:
        print("No chunks were generated. Exiting.")
        return
        
    print(f"\nTotal ready chunks: {len(all_chunks)}")
    
    # 3.5 SAVE TO DISK FOR INSPECTION
    print("\n--- Saving processed data for inspection ---")
    processed_dir = os.path.join(PROJECT_ROOT, "Data", "processeddata")
    os.makedirs(processed_dir, exist_ok=True)
    inspection_file = os.path.join(processed_dir, "chunks_inspection.jsonl")
    
    import json
    with open(inspection_file, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            chunk_data = {
                "metadata": chunk.metadata,
                "page_content": chunk.page_content
            }
            f.write(json.dumps(chunk_data) + "\n")
    print(f"Saved {len(all_chunks)} chunks to {inspection_file} so you can review them!")
    
    # 4. EMBED AND UPLOAD
    print("\n--- 4. EMBEDDING & UPLOADING TO QDRANT ---")
    embeddings = get_embedding_model()
    
    print(f"Connecting to Qdrant at {settings.QDRANT_URL}...")
    client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY
    )
    
    # Initialize VectorStore
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        embedding=embeddings
    )
    
    print(f"Uploading {len(all_chunks)} vectors to Qdrant... (This might take a few minutes)")
    # Langchain Qdrant will automatically handle the embedding generation and batch uploading
    vector_store.add_documents(all_chunks)
    
    print("\n✅ INGESTION COMPLETE! Your Qdrant database is now populated and ready for RAG.")

if __name__ == "__main__":
    run_ingestion()
