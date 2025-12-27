"""
API Routes Package

This package contains all API route modules:
- ingest: Document ingestion endpoints
- query: Query/search endpoints
- health: Health check endpoints
"""

from src.api.routes.health import router as health_router
from src.api.routes.ingest import router as ingest_router
from src.api.routes.query import router as query_router

__all__ = [
    "ingest_router",
    "query_router",
    "health_router",
]