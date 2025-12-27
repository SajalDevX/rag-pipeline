"""
Vector Store Package

This package provides vector database functionality using Zilliz Cloud.

Usage:
    from src.infrastructure.vector_store import ZillizStore
    
    store = ZillizStore()
    await store.connect()
    
    # Insert chunks with embeddings
    await store.insert_chunks(chunks)
    
    # Search for similar chunks
    results = await store.search(query_embedding, top_k=5)
    
    # Cleanup
    await store.disconnect()
"""

from src.infrastructure.vector_store.zilliz_store import ZillizStore

__all__ = [
    "ZillizStore",
]