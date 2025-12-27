"""
FastAPI Application

This module creates and configures the FastAPI application.
It includes:
- Route registration
- Middleware configuration
- Error handlers
- CORS settings
- Lifespan events
"""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import get_logger, settings, setup_logging
from src.models.errors import ErrorResponse, RAGException

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    
    Runs setup on startup and cleanup on shutdown.
    """
    # Startup
    setup_logging()
    logger.info(
        "Application starting",
        app_name=settings.app_name,
        environment=settings.app_env
    )
    
    yield
    
    # Shutdown
    logger.info("Application shutting down")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured application
    """
    app = FastAPI(
        title="RAG Pipeline API",
        description="""
## RAG (Retrieval-Augmented Generation) Pipeline API

This API provides endpoints for:
- **Document Ingestion**: Upload and process documents
- **Querying**: Ask questions and get AI-generated answers
- **Health Monitoring**: Check system status

### Key Features
- PDF, TXT, Markdown, HTML, and URL support
- Semantic search with vector embeddings
- Optional reranking for improved relevance
- Fast LLM inference with Groq

### Getting Started
1. Ingest a document: `POST /api/ingest`
2. Ask a question: `POST /api/query`
3. Check health: `GET /health`
        """,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify allowed origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register exception handlers
    register_exception_handlers(app)
    
    # Register routes
    register_routes(app)
    
    return app


def register_routes(app: FastAPI) -> None:
    """
    Register all API routes.
    
    Args:
        app: FastAPI application
    """
    from src.api.routes import health_router, ingest_router, query_router
    
    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(query_router)
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root() -> dict[str, str]:
        """Root endpoint with API information."""
        return {
            "message": "Welcome to the RAG Pipeline API",
            "docs": "/docs",
            "health": "/health"
        }


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register global exception handlers.
    
    Args:
        app: FastAPI application
    """
    
    @app.exception_handler(RAGException)
    async def rag_exception_handler(
        request: Request,
        exc: RAGException
    ) -> JSONResponse:
        """Handle RAG-specific exceptions."""
        logger.error(
            "RAG exception",
            error_type=exc.error_type,
            message=exc.message,
            path=request.url.path
        )
        
        error_response = exc.to_response()
        
        return JSONResponse(
            status_code=400,
            content=error_response.model_dump(mode="json")
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request,
        exc: Exception
    ) -> JSONResponse:
        """Handle unexpected exceptions."""
        logger.error(
            "Unexpected exception",
            error=str(exc),
            path=request.url.path
        )
        
        # Don't expose internal errors in production
        if settings.is_production:
            message = "An internal error occurred"
        else:
            message = str(exc)
        
        error_response = ErrorResponse(
            error="InternalError",
            message=message
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response.model_dump(mode="json")
        )


# Create the app instance
app = create_app()