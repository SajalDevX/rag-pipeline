"""
Query API Routes

Endpoints for querying the RAG system:
- POST /query - Ask a question and get an answer
"""

from fastapi import APIRouter, HTTPException

from src.config import get_logger
from src.models import QueryRequest, QueryResponse
from src.models.errors import RAGException
from src.services import RAGService

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["Query"])

# Shared service instance
_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    """Get or create RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Query the RAG system with a question.
    
    This endpoint:
    1. Retrieves relevant chunks from the vector database
    2. Optionally reranks them for better relevance
    3. Generates an answer using the LLM
    
    Args:
        request: QueryRequest with question and options
    
    Returns:
        QueryResponse: Answer with sources and metadata
    
    Example:
        POST /api/query
        {
            "query": "What is machine learning?",
            "top_k": 5,
            "rerank": true,
            "include_sources": true
        }
    """
    logger.info(
        "Query request received",
        query_length=len(request.query),
        top_k=request.top_k,
        rerank=request.rerank
    )
    
    try:
        service = get_rag_service()
        
        response = await service.query(
            query=request.query,
            top_k=request.top_k,
            rerank=request.rerank,
            include_sources=request.include_sources,
            filter_metadata=request.filter_metadata
        )
        
        logger.info(
            "Query processed successfully",
            answer_length=len(response.answer),
            chunks_used=response.total_chunks_retrieved
        )
        
        return response
        
    except RAGException as e:
        logger.error("Query failed", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Unexpected error during query", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/query/simple")
async def simple_query(query: str, top_k: int = 5) -> QueryResponse:
    """
    Simple query endpoint with minimal parameters.
    
    Args:
        query: The question to ask
        top_k: Number of chunks to retrieve
    
    Returns:
        QueryResponse: Answer with sources
    
    Example:
        POST /api/query/simple?query=What is ML?&top_k=3
    """
    request = QueryRequest(query=query, top_k=top_k)
    return await query(request)