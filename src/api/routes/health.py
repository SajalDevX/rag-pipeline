"""
Health Check API Routes

Endpoints for monitoring system health:
- GET /health - Check overall system health
- GET /health/detailed - Detailed component health
"""

from typing import Any

from fastapi import APIRouter

from src.config import get_logger, settings
from src.models import HealthResponse
from src.services import RAGService

logger = get_logger(__name__)

router = APIRouter(tags=["Health"])

# Shared service instance for health checks
_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    """Get or create RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Basic health check endpoint.
    
    Returns:
        HealthResponse: System health status
    
    Example:
        GET /health
        
        Response:
        {
            "status": "healthy",
            "version": "1.0.0",
            "components": {},
            "timestamp": "2024-01-15T10:30:00Z"
        }
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0"
    )


@router.get("/health/detailed", response_model=HealthResponse)
async def detailed_health_check() -> HealthResponse:
    """
    Detailed health check of all components.
    
    Checks:
    - Vector store connection
    - Embedding service
    - Reranker service
    - LLM service
    
    Returns:
        HealthResponse: Detailed health status
    
    Example:
        GET /health/detailed
        
        Response:
        {
            "status": "healthy",
            "version": "1.0.0",
            "components": {
                "vector_store": "healthy",
                "embedder": "healthy",
                "reranker": "healthy",
                "llm": "healthy"
            },
            "timestamp": "2024-01-15T10:30:00Z"
        }
    """
    logger.info("Running detailed health check")
    
    try:
        service = get_rag_service()
        health = await service.health_check()
        
        # Convert boolean to string status
        components = {
            k: "healthy" if v else "unhealthy"
            for k, v in health.get("components", {}).items()
        }
        
        overall_status = health.get("status", "unknown")
        
        return HealthResponse(
            status=overall_status,
            version="1.0.0",
            components=components
        )
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            components={"error": str(e)}
        )


@router.get("/info")
async def system_info() -> dict[str, Any]:
    """
    Get system information and configuration.
    
    Returns:
        dict: System configuration (non-sensitive)
    
    Example:
        GET /info
    """
    return {
        "app_name": settings.app_name,
        "environment": settings.app_env,
        "version": "1.0.0",
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        "llm_model": settings.llm_model,
        "reranker_model": settings.reranker_model,
        "reranker_enabled": settings.reranker_enabled,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "default_top_k": settings.default_top_k,
    }