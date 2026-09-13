"""
backend/app/api/router.py
─────────────────────────
Central API router combining health and chat endpoints.
"""

from fastapi import APIRouter
from backend.app.api.endpoints import chat, health

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(chat.router, prefix="/chat", tags=["Chat"])
