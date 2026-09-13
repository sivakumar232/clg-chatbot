"""
backend/run.py
──────────────
Convenience script to run the FastAPI development server.
Usage:
    uv run python backend/run.py
"""

import sys
from pathlib import Path
import uvicorn

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.core.config import api_settings

if __name__ == "__main__":
    uvicorn.run(
        "backend.app.main:app",
        host=api_settings.HOST,
        port=api_settings.PORT,
        reload=True,
        log_level="info",
    )
