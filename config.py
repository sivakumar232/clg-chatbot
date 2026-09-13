import os
from pathlib import Path
from dotenv import load_dotenv
import logfire

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

class Settings:
    LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")
    
    # Gemini keys bucket (comma-separated or single)
    _raw_gemini = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or ""
    GEMINI_API_KEYS: list[str] = [k.strip() for k in _raw_gemini.split(",") if k.strip()]
    GEMINI_API_KEY: str | None = GEMINI_API_KEYS[0] if GEMINI_API_KEYS else None

    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "collection2")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")

    # Groq keys bucket (comma-separated or single)
    _raw_groq = os.getenv("GROQ_API_KEYS") or os.getenv("GROQ_API_KEY") or ""
    GROQ_API_KEYS: list[str] = [k.strip() for k in _raw_groq.split(",") if k.strip()]
    GROQ_API_KEY: str | None = GROQ_API_KEYS[0] if GROQ_API_KEYS else None
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")

    GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    GEMINI_EMB_MODEL = "gemini-embedding-001"

    # Jina keys bucket (comma-separated or single)
    _raw_jina = os.getenv("JINA_API_KEYS") or os.getenv("JINA_API_KEY") or ""
    JINA_API_KEYS: list[str] = [k.strip() for k in _raw_jina.split(",") if k.strip()]
    JINA_API_KEY: str | None = JINA_API_KEYS[0] if JINA_API_KEYS else None
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