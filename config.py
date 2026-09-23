import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

class Settings:
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
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    GEMINI_EMB_MODEL = "gemini-embedding-001"

    # Jina keys bucket (comma-separated or single)
    _raw_jina = os.getenv("JINA_API_KEYS") or os.getenv("JINA_API_KEY") or ""
    JINA_API_KEYS: list[str] = [k.strip() for k in _raw_jina.split(",") if k.strip()]
    JINA_API_KEY: str | None = JINA_API_KEYS[0] if JINA_API_KEYS else None
    JINA_EMB_MODEL = "jina-embeddings-v5-text-small"
    EMBEDDING_DIM = 1024
    # LangSmith Observability
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT", "srkr-academic-advisor")
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2", "true" if (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")) else "false")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    # NeMo Guardrails
    GUARDRAILS_DIR: Path = ROOT_DIR / "guardrails"
    GAURDRAILS_DIR: Path = GUARDRAILS_DIR  # Backward-compatible alias
    
settings = Settings()

# Configure NeMo Guardrails / OpenAI compatible environment variables
if settings.GROQ_API_KEY and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = settings.GROQ_API_KEY
if settings.GROQ_API_KEY and not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY

# Configure LangChain / LangSmith environment variables for LangGraph tracing
if settings.LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if str(settings.LANGSMITH_TRACING).lower() in ("true", "1") else "false"
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
    os.environ["LANGSMITH_TRACING"] = str(settings.LANGSMITH_TRACING)
    os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT