"""
backend/app/main.py
───────────────────
Main FastAPI application entrypoint.
"""

import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("uvicorn.info")

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import api_settings
from backend.app.api.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("SRKR Agentic RAG API starting up...")
    yield
    # Shutdown
    logger.info("SRKR Agentic RAG API shutting down...")


def create_application() -> FastAPI:
    app = FastAPI(
        title=api_settings.PROJECT_NAME,
        version=api_settings.VERSION,
        openapi_url=f"{api_settings.API_V1_STR}/openapi.json",
        docs_url=f"{api_settings.API_V1_STR}/docs",
        redoc_url=f"{api_settings.API_V1_STR}/redoc",
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routes
    app.include_router(api_router, prefix=api_settings.API_V1_STR)

    @app.get("/")
    async def root():
        return {
            "name": api_settings.PROJECT_NAME,
            "version": api_settings.VERSION,
            "docs": f"{api_settings.API_V1_STR}/docs",
            "health": f"{api_settings.API_V1_STR}/health",
        }

    return app


app = create_application()
