"""
Cache Package

This package provides caching functionality:
- EmbeddingCache: File-based cache for embeddings

Usage:
    from src.infrastructure.cache import EmbeddingCache
    
    cache = EmbeddingCache()
    
    # Check cache
    embedding = await cache.get("Hello world")
    
    # Store in cache
    await cache.set("Hello world", embedding)
"""

from src.infrastructure.cache.embedding_cache import EmbeddingCache

__all__ = [
    "EmbeddingCache",
]