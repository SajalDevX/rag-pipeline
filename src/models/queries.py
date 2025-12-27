"""
Query and Response Models

This module defines the data structures for querying the RAG system:
- QueryRequest: What the user sends to query the system
- QueryResponse: What the system returns to the user
- RetrievedChunk: A chunk retrieved from the vector database
- IngestRequest/Response: For document ingestion
- HealthResponse: For health checks

These models are used by:
1. FastAPI for request/response validation
2. Services for data passing
3. API documentation generation
"""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """
    A chunk retrieved from the vector database.
    
    This represents a search result, including the similarity score
    and optionally the rerank score.
    
    Attributes:
        chunk_id: Unique identifier of the chunk
        document_id: ID of the parent document
        content: Text content of the chunk
        score: Similarity score from vector search (0.0 to 1.0)
        chunk_index: Position in the original document
        source: Original document source
        document_type: Type of the source document
        title: Document title if available
        rerank_score: Score from reranker if reranking was used
    """
    
    chunk_id: str = Field(
        ...,
        description="Unique identifier of the chunk"
    )
    
    document_id: str = Field(
        ...,
        description="ID of the parent document"
    )
    
    content: str = Field(
        ...,
        description="Text content of the chunk"
    )
    
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity score from vector search"
    )
    
    chunk_index: int = Field(
        ...,
        ge=0,
        description="Position of chunk in original document"
    )
    
    source: str = Field(
        ...,
        description="Original document source (file path or URL)"
    )
    
    document_type: str = Field(
        ...,
        description="Type of the source document"
    )
    
    title: str | None = Field(
        default=None,
        description="Document title if available"
    )
    
    rerank_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Score from reranker (if reranking was used)"
    )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"RetrievedChunk(score={self.score:.3f}, content='{preview}')"


class QueryRequest(BaseModel):
    """
    Request model for querying the RAG system.
    
    This is what users send when they want to ask a question.
    
    Attributes:
        query: The user's question
        top_k: Number of chunks to retrieve
        rerank: Whether to use reranking
        filter_metadata: Optional metadata filters
        include_sources: Whether to include source chunks in response
    
    Example JSON:
        {
            "query": "What is machine learning?",
            "top_k": 5,
            "rerank": true,
            "include_sources": true
        }
    """
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The question to ask",
        examples=["What is machine learning?", "How does RAG work?"]
    )
    
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of chunks to retrieve and use for generation"
    )
    
    rerank: bool = Field(
        default=True,
        description="Whether to rerank results for better relevance"
    )
    
    filter_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata filters (e.g., {'document_type': 'pdf'})"
    )
    
    include_sources: bool = Field(
        default=True,
        description="Whether to include source chunks in the response"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "top_k": 5,
                "rerank": True,
                "filter_metadata": None,
                "include_sources": True
            }
        }


class QueryResponse(BaseModel):
    """
    Response model for RAG queries.
    
    This is what the system returns after processing a query.
    
    Attributes:
        query_id: Unique identifier for this query
        query: The original question
        answer: The generated answer
        sources: List of source chunks used (if include_sources was True)
        total_chunks_retrieved: How many chunks were retrieved
        processing_time_ms: Total processing time in milliseconds
        model_used: Which LLM model generated the answer
        timestamp: When the query was processed
    """
    
    query_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this query"
    )
    
    query: str = Field(
        ...,
        description="The original question"
    )
    
    answer: str = Field(
        ...,
        description="The generated answer"
    )
    
    sources: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Source chunks used to generate the answer"
    )
    
    total_chunks_retrieved: int = Field(
        ...,
        ge=0,
        description="Total number of chunks retrieved from vector database"
    )
    
    processing_time_ms: float = Field(
        ...,
        ge=0,
        description="Total processing time in milliseconds"
    )
    
    model_used: str = Field(
        ...,
        description="LLM model used for generation"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the query was processed"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query_id": "123e4567-e89b-12d3-a456-426614174000",
                "query": "What is machine learning?",
                "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed...",
                "sources": [
                    {
                        "chunk_id": "abc-123",
                        "document_id": "def-456",
                        "content": "Machine learning is...",
                        "score": 0.92,
                        "chunk_index": 0,
                        "source": "ml_guide.pdf",
                        "document_type": "pdf",
                        "title": "Machine Learning Guide"
                    }
                ],
                "total_chunks_retrieved": 5,
                "processing_time_ms": 1234.5,
                "model_used": "llama-3.1-8b-instant",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class IngestRequest(BaseModel):
    """
    Request model for document ingestion.
    
    This is what users send when they want to add a document.
    
    Attributes:
        source: File path or URL of the document
        document_type: Type of document (auto-detected if not provided)
        title: Document title (extracted if not provided)
        custom_metadata: Additional metadata for filtering
    
    Example JSON:
        {
            "source": "/path/to/document.pdf",
            "title": "My Document",
            "custom_metadata": {"department": "engineering"}
        }
    """
    
    source: str = Field(
        ...,
        min_length=1,
        description="File path or URL of the document",
        examples=["/documents/report.pdf", "https://example.com/article"]
    )
    
    document_type: str | None = Field(
        default=None,
        description="Document type (auto-detected from extension if not provided)"
    )
    
    title: str | None = Field(
        default=None,
        max_length=500,
        description="Document title (extracted from document if not provided)"
    )
    
    custom_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for filtering"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "source": "/documents/report.pdf",
                "title": "Q4 Financial Report",
                "custom_metadata": {
                    "department": "finance",
                    "year": 2024
                }
            }
        }


class IngestResponse(BaseModel):
    """
    Response model for document ingestion.
    
    This is what the system returns after ingesting a document.
    
    Attributes:
        document_id: Unique identifier of the ingested document
        source: Original source path/URL
        chunks_created: Number of chunks created
        processing_time_ms: Time taken to process
        status: Success or error status
        message: Human-readable status message
    """
    
    document_id: UUID = Field(
        ...,
        description="Unique identifier of the ingested document"
    )
    
    source: str = Field(
        ...,
        description="Original source path or URL"
    )
    
    chunks_created: int = Field(
        ...,
        ge=0,
        description="Number of chunks created from the document"
    )
    
    processing_time_ms: float = Field(
        ...,
        ge=0,
        description="Processing time in milliseconds"
    )
    
    status: str = Field(
        default="success",
        description="Status of the ingestion"
    )
    
    message: str = Field(
        default="Document ingested successfully",
        description="Human-readable status message"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "source": "/documents/report.pdf",
                "chunks_created": 15,
                "processing_time_ms": 2345.6,
                "status": "success",
                "message": "Document ingested successfully"
            }
        }


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.
    
    Used to verify the system is running and all components are healthy.
    
    Attributes:
        status: Overall system status
        version: API version
        components: Status of individual components
        timestamp: When the check was performed
    """
    
    status: str = Field(
        default="healthy",
        description="Overall system status"
    )
    
    version: str = Field(
        default="1.0.0",
        description="API version"
    )
    
    components: dict[str, str] = Field(
        default_factory=dict,
        description="Status of individual components"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the health check was performed"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "components": {
                    "vector_store": "healthy",
                    "embedding_service": "healthy",
                    "llm_service": "healthy",
                    "reranker_service": "healthy"
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class DocumentListResponse(BaseModel):
    """
    Response model for listing documents.
    
    Returns a list of ingested documents with their metadata.
    """
    
    total_documents: int = Field(
        ...,
        ge=0,
        description="Total number of documents"
    )
    
    documents: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of document summaries"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_documents": 3,
                "documents": [
                    {
                        "document_id": "abc-123",
                        "source": "report.pdf",
                        "title": "Annual Report",
                        "chunk_count": 15,
                        "created_at": "2024-01-15T10:30:00Z"
                    }
                ]
            }
        }