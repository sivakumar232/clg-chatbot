import os 
from dotenv import load_dotenv
import logfire

load_dotenv()

class Settings:
    LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "collection2")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    GEMINI_EMB_MODEL = "gemini-embedding-001"

    JINA_API_KEY = os.getenv("JINA_API_KEY") or os.getenv("JINA_API_KEYS")
    JINA_EMB_MODEL = "jina-embeddings-v5-text-small"
    EMBEDDING_DIM = 1024
    
settings = Settings()

# Centralized Logfire configuration
logfire.configure(
    service_name="srkr_academic_advisor",
    send_to_logfire=True if settings.LOGFIRE_TOKEN else "if-token-present",
    token=settings.LOGFIRE_TOKEN or None,
    inspect_arguments=False,
)