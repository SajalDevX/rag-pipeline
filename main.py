"""
Main Entry Point

This is the main entry point for the RAG Pipeline API.

Run with:
    python main.py
    
Or with uvicorn directly:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import uvicorn

from src.api import app
from src.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.is_development,
        log_level=settings.log_level.lower()
    )