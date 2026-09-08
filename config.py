import os 
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "collection2")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")
    GROQ_MODEL="llama-3.3-70b-versatile"
    GEMINI_EMB_MODEL="gemini-embedding-001"

    JINA_API_KEY = os.getenv("JINA_API_KEY") or os.getenv("JINA_API_KEYS")
    JINA_EMB_MODEL = "jina-embeddings-v5-text-small"
    EMBEDDING_DIM = 1024
    
settings = Settings()