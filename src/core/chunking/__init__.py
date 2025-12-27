"""
Chunking Package

This package provides text chunking functionality for the RAG pipeline.

Available Chunkers:
- RecursiveChunker: Splits on natural boundaries (recommended)
- SentenceChunker: Splits by sentences

Usage:
    # Recommended: Use the factory
    from src.core.chunking import ChunkingFactory
    
    chunked_doc = ChunkingFactory.chunk(document)
    chunked_doc = ChunkingFactory.chunk(document, strategy="sentence")
    
    # Or use specific chunkers directly
    from src.core.chunking import RecursiveChunker, SentenceChunker
    
    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
    chunked_doc = chunker.chunk(document)
"""

from src.core.chunking.base import BaseChunker
from src.core.chunking.factory import ChunkingFactory, ChunkingStrategy
from src.core.chunking.recursive_chunker import RecursiveChunker
from src.core.chunking.sentence_chunker import SentenceChunker

__all__ = [
    "BaseChunker",
    "RecursiveChunker",
    "SentenceChunker",
    "ChunkingFactory",
    "ChunkingStrategy",
]