"""
RAG Service

This is the main service that combines retrieval and generation:
1. Retrieve relevant chunks
2. Generate answer using LLM

Usage:
    service = RAGService()
    response = await service.query("How does ML work?")
"""

import time
from typing import Any

from src.config import get_logger, settings
from src.core.llm import GroqLLM
from src.infrastructure.vector_store import ZillizStore
from src.models import QueryRequest, QueryResponse, RetrievedChunk
from src.models.errors import RAGException
from src.services.retrieval_service import RetrievalService

logger = get_logger(__name__)


class RAGService:
    """
    Main RAG service that combines retrieval and generation.
    
    Orchestrates the complete RAG pipeline:
    Query → Retrieve Chunks → Generate Answer
    
    Attributes:
        retrieval_service: Service for retrieving chunks
        llm: Language model for generation
    
    Usage:
        service = RAGService()
        
        # Simple query
        response = await service.query("What is machine learning?")
        print(response.answer)
        
        # With options
        response = await service.query(
            query="What is ML?",
            top_k=5,
            rerank=True,
            include_sources=True
        )
    """
    
    def __init__(
        self,
        vector_store: ZillizStore | None = None,
        enable_reranking: bool | None = None
    ):
        """
        Initialize the RAG service.
        
        Args:
            vector_store: Vector store client
            enable_reranking: Whether to enable reranking
        """
        # Shared vector store
        self.vector_store = vector_store or ZillizStore()
        
        # Initialize services
        self.retrieval_service = RetrievalService(
            vector_store=self.vector_store,
            enable_reranking=enable_reranking
        )
        self.llm = GroqLLM()
        
        self.logger = get_logger(self.__class__.__name__)
        
        self.logger.info("RAG service initialized")
    
    async def query(
        self,
        query: str | QueryRequest,
        top_k: int | None = None,
        rerank: bool | None = None,
        include_sources: bool = True,
        filter_metadata: dict[str, Any] | None = None
    ) -> QueryResponse:
        """
        Process a query and generate an answer.
        
        Args:
            query: User query (string or QueryRequest)
            top_k: Number of chunks to use
            rerank: Whether to rerank results
            include_sources: Whether to include sources in response
            filter_metadata: Metadata filters
        
        Returns:
            QueryResponse: Answer with sources and metadata
        
        Raises:
            RAGException: If query processing fails
        """
        start_time = time.time()
        
        # Handle QueryRequest object
        if isinstance(query, QueryRequest):
            query_text = query.query
            top_k = top_k or query.top_k
            rerank = rerank if rerank is not None else query.rerank
            include_sources = query.include_sources
            filter_metadata = filter_metadata or query.filter_metadata
        else:
            query_text = query
        
        self.logger.info(
            "Processing query",
            query_length=len(query_text),
            top_k=top_k,
            rerank=rerank
        )
        
        try:
            # Step 1: Retrieve relevant chunks
            chunks = await self.retrieval_service.retrieve(
                query=query_text,
                top_k=top_k,
                rerank=rerank,
                filter_metadata=filter_metadata
            )
            
            # Step 2: Generate answer
            if chunks:
                answer = await self.llm.generate(
                    query=query_text,
                    context_chunks=chunks
                )
            else:
                answer = "I couldn't find any relevant information to answer your question. Please try rephrasing or adding more documents to the knowledge base."
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Build response
            response = QueryResponse(
                query=query_text,
                answer=answer,
                sources=chunks if include_sources else [],
                total_chunks_retrieved=len(chunks),
                processing_time_ms=processing_time,
                model_used=self.llm.model
            )
            
            self.logger.info(
                "Query processed",
                answer_length=len(answer),
                chunks_used=len(chunks),
                processing_time_ms=processing_time
            )
            
            return response
            
        except RAGException:
            raise
        except Exception as e:
            self.logger.error("Query processing failed", error=str(e))
            raise RAGException(
                message=f"Failed to process query: {str(e)}"
            )
    
    async def health_check(self) -> dict[str, Any]:
        """
        Check health of all components.
        
        Returns:
            dict: Health status of each component
        """
        retrieval_health = await self.retrieval_service.health_check()
        llm_health = await self.llm.health_check()
        
        all_healthy = all(retrieval_health.values()) and llm_health
        
        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "components": {
                **retrieval_health,
                "llm": llm_health
            }
        }