"""
Embeddings Package

This package provides text embedding functionality using cloud APIs.

Available Providers:
- HuggingFaceEmbedding: Free HuggingFace Inference API (recommended)

Usage:
    # Recommended: Use the factory
    from src.core.embeddings import EmbeddingFactory
    
    embedder = EmbeddingFactory.get_embedder()
    
    # Single text
    vector = await embedder.embed_text("Hello world")
    
    # Multiple texts
    vectors = await embedder.embed_texts(["Hello", "World"])
    
    # Embed chunks
    chunks_with_embeddings = await embedder.embed_chunks(chunks)
    
    # Or use specific provider directly
    from src.core.embeddings import HuggingFaceEmbedding
    
    embedder = HuggingFaceEmbedding()
    vector = await embedder.embed_text("Hello world")
"""

from src.core.embeddings.base import BaseEmbeddingManager
from src.core.embeddings.factory import EmbeddingFactory, EmbeddingProvider
from src.core.embeddings.huggingface_embedding import HuggingFaceEmbedding

__all__ = [
    "BaseEmbeddingManager",
    "HuggingFaceEmbedding",
    "EmbeddingFactory",
    "EmbeddingProvider",
]