"""
Rerankers Package

This package provides reranking functionality to improve search results.

Usage:
    from src.core.rerankers import CohereReranker
    
    reranker = CohereReranker()
    
    # Rerank search results
    reranked = await reranker.rerank(
        query="How does machine learning work?",
        chunks=retrieved_chunks,
        top_k=5
    )
"""

from src.core.rerankers.cohere_reranker import CohereReranker

__all__ = [
    "CohereReranker",
]