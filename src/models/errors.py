"""
Error Models

This module defines error structures for consistent error handling:
- ErrorDetail: Detailed information about a specific error
- ErrorResponse: Standard API error response format
- Custom exceptions for different error types

These are used for:
1. Consistent API error responses
2. Exception handling throughout the application
3. Clear error messages for debugging
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """
    Detailed information about a specific error.
    
    Used when an error relates to a specific field or has
    additional context that would be helpful for debugging.
    
    Attributes:
        field: The field that caused the error (if applicable)
        message: Human-readable error message
        code: Machine-readable error code (optional)
    """
    
    field: str | None = Field(
        default=None,
        description="Field that caused the error"
    )
    
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    
    code: str | None = Field(
        default=None,
        description="Machine-readable error code"
    )


class ErrorResponse(BaseModel):
    """
    Standard error response format.
    
    All API errors should return this structure for consistency.
    
    Attributes:
        error: Error type/category
        message: Human-readable error message
        details: List of detailed error information
        timestamp: When the error occurred
        request_id: Unique identifier for the request (for tracking)
    
    Example:
        {
            "error": "ValidationError",
            "message": "Invalid request parameters",
            "details": [
                {"field": "query", "message": "Query cannot be empty"}
            ],
            "timestamp": "2024-01-15T10:30:00Z"
        }
    """
    
    error: str = Field(
        ...,
        description="Error type or category"
    )
    
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    
    details: list[ErrorDetail] = Field(
        default_factory=list,
        description="Detailed error information"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the error occurred"
    )
    
    request_id: str | None = Field(
        default=None,
        description="Unique request identifier for tracking"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid request parameters",
                "details": [
                    {
                        "field": "query",
                        "message": "Query cannot be empty",
                        "code": "EMPTY_QUERY"
                    }
                ],
                "timestamp": "2024-01-15T10:30:00Z",
                "request_id": "req_abc123"
            }
        }


# =========================================================
# CUSTOM EXCEPTIONS
# =========================================================

class RAGException(Exception):
    """
    Base exception for all RAG pipeline errors.
    
    All custom exceptions inherit from this class.
    Provides a consistent way to convert exceptions to API responses.
    
    Usage:
        raise RAGException("Something went wrong")
        
        # Or with details
        raise RAGException(
            message="Failed to process document",
            error_type="ProcessingError",
            details=[ErrorDetail(field="file", message="File is corrupted")]
        )
    """
    
    def __init__(
        self,
        message: str,
        error_type: str = "RAGError",
        details: list[ErrorDetail] | None = None,
        request_id: str | None = None
    ):
        self.message = message
        self.error_type = error_type
        self.details = details or []
        self.request_id = request_id
        super().__init__(self.message)
    
    def to_response(self) -> ErrorResponse:
        """
        Convert exception to ErrorResponse.
        
        Returns:
            ErrorResponse suitable for API response
        """
        return ErrorResponse(
            error=self.error_type,
            message=self.message,
            details=self.details,
            request_id=self.request_id
        )
    
    def __str__(self) -> str:
        """String representation for logging."""
        return f"{self.error_type}: {self.message}"


class DocumentLoadError(RAGException):
    """
    Error loading a document.
    
    Raised when:
    - File doesn't exist
    - File format is unsupported
    - File is corrupted
    - URL is unreachable
    """
    
    def __init__(
        self,
        message: str,
        details: list[ErrorDetail] | None = None,
        request_id: str | None = None
    ):
        super().__init__(
            message=message,
            error_type="DocumentLoadError",
            details=details,
            request_id=request_id
        )


class ChunkingError(RAGException):
    """
    Error during document chunking.
    
    Raised when:
    - Document is empty
    - Chunking parameters are invalid
    - Text processing fails
    """
    
    def __init__(
        self,
        message: str,
        details: list[ErrorDetail] | None = None,
        request_id: str | None = None
    ):
        super().__init__(
            message=message,
            error_type="ChunkingError",
            details=details,
            request_id=request_id
        )


class EmbeddingError(RAGException):
    """
    Error generating embeddings.
    
    Raised when:
    - HuggingFace API is unavailable
    - Rate limit exceeded
    - Model not found
    - Invalid input
    """
    
    def __init__(
        self,
        message: str,
        details: list[ErrorDetail] | None = None,
        request_id: str | None = None
    ):
        super().__init__(
            message=message,
            error_type="EmbeddingError",
            details=details,
            request_id=request_id
        )


class VectorStoreError(RAGException):
    """
    Error with vector store operations.
    
    Raised when:
    - Zilliz connection fails
    - Insert/query operation fails
    - Collection doesn't exist
    """
    
    def __init__(
        self,
        message: str,
        details: list[ErrorDetail] | None = None,
        request_id: str | None = None
    ):
        super().__init__(
            message=message,
            error_type="VectorStoreError",
            details=details,
            request_id=request_id
        )


class LLMError(RAGException):
    """
    Error with LLM operations.
    
    Raised when:
    - Groq API is unavailable
    - Rate limit exceeded
    - Model not found
    - Generation fails
    """
    
    def __init__(
        self,
        message: str,
        details: list[ErrorDetail] | None = None,
        request_id: str | None = None
    ):
        super().__init__(
            message=message,
            error_type="LLMError",
            details=details,
            request_id=request_id
        )


class RerankerError(RAGException):
    """
    Error with reranking operations.
    
    Raised when:
    - Cohere API is unavailable
    - Rate limit exceeded
    - Reranking fails
    """
    
    def __init__(
        self,
        message: str,
        details: list[ErrorDetail] | None = None,
        request_id: str | None = None
    ):
        super().__init__(
            message=message,
            error_type="RerankerError",
            details=details,
            request_id=request_id
        )


class RetrievalError(RAGException):
    """
    Error during retrieval process.
    
    Raised when:
    - Search fails
    - No results found
    - Filtering fails
    """
    
    def __init__(
        self,
        message: str,
        details: list[ErrorDetail] | None = None,
        request_id: str | None = None
    ):
        super().__init__(
            message=message,
            error_type="RetrievalError",
            details=details,
            request_id=request_id
        )


class ValidationError(RAGException):
    """
    Error with input validation.
    
    Raised when:
    - Request parameters are invalid
    - File size exceeds limit
    - Required fields are missing
    """
    
    def __init__(
        self,
        message: str,
        details: list[ErrorDetail] | None = None,
        request_id: str | None = None
    ):
        super().__init__(
            message=message,
            error_type="ValidationError",
            details=details,
            request_id=request_id
        )