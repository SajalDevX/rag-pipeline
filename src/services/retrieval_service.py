"""
Retrieval Service

This service handles searching for relevant chunks:
1. Embed the query
2. Search vector database
3. Optionally rerank results

Usage:
    service = RetrievalService()
    chunks = await service.retrieve("How does ML work?", top_k=5)
"""

import time
from typing import Any

from src.config import get_logger, settings
from src.core.embeddings import EmbeddingFactory
from src.core.rerankers import CohereReranker
from src.infrastructure.vector_store import ZillizStore
from src.models import RetrievedChunk
from src.models.errors import RetrievalError

logger = get_logger(__name__)


class RetrievalService:
    """
    Service for retrieving relevant chunks from the vector database.
    
    Orchestrates the retrieval pipeline:
    Query → Embed → Search → Rerank → Results
    
    Attributes:
        vector_store: Vector database client
        embedder: Embedding generator
        reranker: Result reranker (optional)
        default_top_k: Default number of results
    
    Usage:
        service = RetrievalService()
        
        # Simple retrieval
        chunks = await service.retrieve("What is ML?")
        
        # With options
        chunks = await service.retrieve(
            query="What is ML?",
            top_k=10,
            rerank=True,
            rerank_top_k=5
        )
    """
    
    def __init__(
        self,
        vector_store: ZillizStore | None = None,
        enable_reranking: bool | None = None
    ):
        """
        Initialize the retrieval service.
        
        Args:
            vector_store: Vector store client (creates new if None)
            enable_reranking: Whether to enable reranking (default from settings)
        """
        self.vector_store = vector_store or ZillizStore()
        self.embedder = EmbeddingFactory.get_embedder()
        
        # Initialize reranker if enabled
        self.enable_reranking = enable_reranking if enable_reranking is not None else settings.reranker_enabled
        self.reranker = CohereReranker() if self.enable_reranking else None
        
        self.default_top_k = settings.default_top_k
        self.rerank_top_k = settings.rerank_top_k
        
        self.logger = get_logger(self.__class__.__name__)
        
        self.logger.info(
            "Retrieval service initialized",
            default_top_k=self.default_top_k,
            reranking_enabled=self.enable_reranking
        )
    
    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        rerank: bool | None = None,
        rerank_top_k: int | None = None,
        filter_metadata: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Search query
            top_k: Number of chunks to retrieve from vector DB
            rerank: Whether to rerank results
            rerank_top_k: Number of chunks to keep after reranking
            filter_metadata: Metadata filters
        
        Returns:
            list[RetrievedChunk]: Retrieved chunks with scores
        
        Raises:
            RetrievalError: If retrieval fails
        """
        start_time = time.time()
        
        # Set defaults
        top_k = top_k or self.default_top_k
        rerank = rerank if rerank is not None else self.enable_reranking
        rerank_top_k = rerank_top_k or self.rerank_top_k
        
        # If reranking, retrieve more candidates
        search_top_k = top_k * 2 if rerank else top_k
        
        self.logger.info(
            "Starting retrieval",
            query_length=len(query),
            top_k=top_k,
            rerank=rerank
        )
        
        try:
            # Step 1: Embed the query
            self.logger.debug("Embedding query...")
            query_embedding = await self.embedder.embed_text(query)
            
            # Step 2: Search vector database
            self.logger.debug("Searching vector database...")
            
            # Build filter expression if metadata provided
            filter_expr = self._build_filter_expression(filter_metadata)
            
            chunks = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=search_top_k,
                filter_expr=filter_expr
            )
            
            # Step 3: Rerank if enabled
            if rerank and self.reranker and len(chunks) > 0:
                self.logger.debug("Reranking results...")
                chunks = await self.reranker.rerank(
                    query=query,
                    chunks=chunks,
                    top_k=rerank_top_k
                )
            else:
                # Just take top_k without reranking
                chunks = chunks[:top_k]
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            self.logger.info(
                "Retrieval complete",
                results_count=len(chunks),
                processing_time_ms=processing_time
            )
            
            return chunks
            
        except Exception as e:
            self.logger.error("Retrieval failed", error=str(e))
            raise RetrievalError(
                message=f"Failed to retrieve chunks: {str(e)}"
            )
    
    def _build_filter_expression(
        self,
        filter_metadata: dict[str, Any] | None
    ) -> str | None:
        """
        Build Milvus filter expression from metadata.
        
        Args:
            filter_metadata: Metadata to filter by
        
        Returns:
            str | None: Filter expression or None
        """
        if not filter_metadata:
            return None
        
        expressions = []
        
        for key, value in filter_metadata.items():
            if isinstance(value, str):
                expressions.append(f'{key} == "{value}"')
            elif isinstance(value, (int, float)):
                expressions.append(f'{key} == {value}')
            elif isinstance(value, list):
                # IN expression
                if all(isinstance(v, str) for v in value):
                    values = ", ".join(f'"{v}"' for v in value)
                    expressions.append(f'{key} in [{values}]')
                else:
                    values = ", ".join(str(v) for v in value)
                    expressions.append(f'{key} in [{values}]')
        
        if not expressions:
            return None
        
        return " && ".join(expressions)
    
    async def health_check(self) -> dict[str, bool]:
        """
        Check health of all components.
        
        Returns:
            dict: Health status of each component
        """
        health = {
            "vector_store": await self.vector_store.health_check(),
            "embedder": await self.embedder.health_check(),
        }
        
        if self.reranker:
            health["reranker"] = await self.reranker.health_check()
        
        return health