"""
backend/app/api/endpoints/health.py
───────────────────────────────────
Health check endpoint.
"""

from fastapi import APIRouter
from config import settings

router = APIRouter()


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "SRKR Agentic RAG Assistant",
        "collection": settings.QDRANT_COLLECTION_NAME,
        "groq_model": settings.GROQ_MODEL,
        "gemini_model": settings.GEMINI_MODEL,
    }
