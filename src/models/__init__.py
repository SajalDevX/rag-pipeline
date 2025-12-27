"""
Data Models Package

This package contains all Pydantic models used throughout the RAG pipeline:

Documents:
    - DocumentType: Enum of supported document types
    - DocumentMetadata: Metadata associated with a document
    - Document: A loaded document before chunking
    - Chunk: A piece of a document after chunking
    - ChunkedDocument: A document with its chunks

Queries:
    - QueryRequest: User query input
    - QueryResponse: System response with answer and sources
    - RetrievedChunk: A chunk retrieved from vector search
    - IngestRequest: Document ingestion input
    - IngestResponse: Ingestion result
    - HealthResponse: Health check response
    - DocumentListResponse: List of documents

Errors:
    - ErrorDetail: Detailed error information
    - ErrorResponse: Standard error response format
    - RAGException: Base exception class
    - DocumentLoadError, ChunkingError, EmbeddingError, etc.

Usage:
    from src.models import Document, Chunk, QueryRequest, QueryResponse
    from src.models import RAGException, DocumentLoadError
"""

# Document models
from src.models.documents import (
    Chunk,
    ChunkedDocument,
    Document,
    DocumentMetadata,
    DocumentType,
)

# Query/Response models
from src.models.queries import (
    DocumentListResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
)

# Error models
from src.models.errors import (
    ChunkingError,
    DocumentLoadError,
    EmbeddingError,
    ErrorDetail,
    ErrorResponse,
    LLMError,
    RAGException,
    RerankerError,
    RetrievalError,
    ValidationError,
    VectorStoreError,
)

__all__ = [
    # Documents
    "DocumentType",
    "DocumentMetadata",
    "Document",
    "Chunk",
    "ChunkedDocument",
    # Queries
    "QueryRequest",
    "QueryResponse",
    "RetrievedChunk",
    "IngestRequest",
    "IngestResponse",
    "HealthResponse",
    "DocumentListResponse",
    # Errors
    "ErrorDetail",
    "ErrorResponse",
    "RAGException",
    "DocumentLoadError",
    "ChunkingError",
    "EmbeddingError",
    "VectorStoreError",
    "LLMError",
    "RerankerError",
    "RetrievalError",
    "ValidationError",
]