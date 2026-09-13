"""
backend/app/core/config.py
──────────────────────────
Configuration settings for the FastAPI server.
"""

import os
from typing import List
from pydantic import BaseModel


class APISettings:
    PROJECT_NAME: str = "SRKR Agentic RAG API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    HOST: str = os.getenv("API_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("API_PORT", "8000"))
    
    # Allowed CORS origins for Next.js frontend (local dev & production)
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "*",  # Allow all for local dev
    ]


api_settings = APISettings()
