"""
Cohere Reranker

This module provides reranking functionality using Cohere's API.
Reranking improves search results by re-scoring them based on
relevance to the query using a cross-encoder model.

Cohere Free Tier:
- 1,000 requests/month
- Rate limited

Usage:
    reranker = CohereReranker()
    reranked = await reranker.rerank(query, chunks, top_k=5)
"""

import asyncio
from typing import Any

import cohere

from src.config import get_logger, settings
from src.models import RetrievedChunk
from src.models.errors import ErrorDetail, RerankerError

logger = get_logger(__name__)


class CohereReranker:
    """
    Reranker using Cohere's rerank API.
    
    Takes search results and re-scores them based on relevance
    to the query, providing more accurate ranking than vector
    similarity alone.
    
    Attributes:
        api_key: Cohere API key
        model: Reranker model to use
        enabled: Whether reranking is enabled
    
    Usage:
        reranker = CohereReranker()
        
        # Rerank retrieved chunks
        reranked = await reranker.rerank(
            query="How does ML work?",
            chunks=retrieved_chunks,
            top_k=5
        )
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        enabled: bool | None = None
    ):
        """
        Initialize the Cohere reranker.
        
        Args:
            api_key: Cohere API key (default from settings)
            model: Reranker model (default from settings)
            enabled: Whether reranking is enabled (default from settings)
        """
        self.api_key = api_key or settings.cohere_api_key
        self.model = model or settings.reranker_model
        self.enabled = enabled if enabled is not None else settings.reranker_enabled
        
        self._client: cohere.Client | None = None
        
        self.logger = get_logger(self.__class__.__name__)
        
        # Validate API key if enabled
        if self.enabled and not self.api_key:
            raise RerankerError(
                message="Cohere API key not provided",
                details=[ErrorDetail(field="api_key", message="Set COHERE_API_KEY in .env")]
            )
        
        # Initialize client
        if self.enabled:
            self._client = cohere.Client(api_key=self.api_key)
        
        self.logger.info(
            "Cohere reranker initialized",
            model=self.model,
            enabled=self.enabled
        )
    
    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None
    ) -> list[RetrievedChunk]:
        """
        Rerank chunks based on relevance to query.
        
        Args:
            query: The search query
            chunks: List of chunks to rerank
            top_k: Number of top results to return (default: all)
        
        Returns:
            list[RetrievedChunk]: Reranked chunks with updated scores
        
        Raises:
            RerankerError: If reranking fails
        """
        # If disabled, return original chunks
        if not self.enabled:
            self.logger.debug("Reranking disabled, returning original order")
            return chunks[:top_k] if top_k else chunks
        
        # Handle empty input
        if not chunks:
            return []
        
        if not query or not query.strip():
            self.logger.warning("Empty query, returning original order")
            return chunks[:top_k] if top_k else chunks
        
        self.logger.info(
            "Reranking chunks",
            query_length=len(query),
            chunk_count=len(chunks),
            top_k=top_k
        )
        
        try:
            # Prepare documents for reranking
            documents = [chunk.content for chunk in chunks]
            
            # Call Cohere API in thread pool (sync client)
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.rerank(
                    query=query,
                    documents=documents,
                    model=self.model,
                    top_n=top_k or len(chunks)
                )
            )
            
            # Create reranked list
            reranked_chunks = []
            
            for item in result.results:
                # Get original chunk
                original_chunk = chunks[item.index]
                
                # Create new chunk with rerank score
                reranked_chunk = RetrievedChunk(
                    chunk_id=original_chunk.chunk_id,
                    document_id=original_chunk.document_id,
                    content=original_chunk.content,
                    score=original_chunk.score,  # Keep original vector score
                    chunk_index=original_chunk.chunk_index,
                    source=original_chunk.source,
                    document_type=original_chunk.document_type,
                    title=original_chunk.title,
                    rerank_score=item.relevance_score  # Add rerank score
                )
                reranked_chunks.append(reranked_chunk)
            
            self.logger.info(
                "Reranking complete",
                input_count=len(chunks),
                output_count=len(reranked_chunks)
            )
            
            return reranked_chunks
            
        except cohere.errors.TooManyRequestsError:
            self.logger.error("Cohere rate limit exceeded")
            raise RerankerError(
                message="Cohere API rate limit exceeded",
                details=[ErrorDetail(
                    field="rate_limit",
                    message="Free tier: 1,000 requests/month. Try again later."
                )]
            )
        
        except Exception as e:
            self.logger.error("Reranking failed", error=str(e))
            raise RerankerError(
                message=f"Reranking failed: {str(e)}"
            )
    
    async def health_check(self) -> bool:
        """
        Check if the reranker is healthy.
        
        Returns:
            bool: True if healthy
        """
        if not self.enabled:
            return True
        
        try:
            # Try a simple rerank call
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.rerank(
                    query="test",
                    documents=["test document"],
                    model=self.model,
                    top_n=1
                )
            )
            return len(result.results) > 0
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return False
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CohereReranker("
            f"model='{self.model}', "
            f"enabled={self.enabled})"
        )