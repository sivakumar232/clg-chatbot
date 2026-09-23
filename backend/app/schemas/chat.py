"""
backend/app/schemas/chat.py
───────────────────────────
Pydantic schemas for chat requests and responses.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's query or question")
    chat_history: Optional[List[ChatMessage]] = Field(
        default_factory=list,
        description="Previous conversation turns for context and pronoun resolution"
    )


class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: List[str] = Field(default_factory=list)
    provider: str
    route: Optional[str] = None
    clarification: Optional[Dict[str, Any]] = None
    cache_hit: bool = False
    degraded: bool = False
