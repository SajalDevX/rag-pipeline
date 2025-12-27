"""
Chunking Factory

This module provides a factory class for creating chunkers
and a convenience function for chunking documents.

Usage:
    from src.core.chunking import ChunkingFactory
    
    # Use default chunker (recursive)
    chunked_doc = ChunkingFactory.chunk(document)
    
    # Use specific chunker
    chunked_doc = ChunkingFactory.chunk(document, strategy="sentence")
    
    # Get chunker instance
    chunker = ChunkingFactory.get_chunker("recursive", chunk_size=1000)
"""

from enum import Enum
from typing import Literal

from src.config import get_logger
from src.models import ChunkedDocument, Document
from src.models.errors import ChunkingError

from .base import BaseChunker
from .recursive_chunker import RecursiveChunker
from .sentence_chunker import SentenceChunker

logger = get_logger(__name__)


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""
    RECURSIVE = "recursive"
    SENTENCE = "sentence"


class ChunkingFactory:
    """
    Factory class for creating and using chunkers.
    
    Provides:
    1. Easy creation of chunkers by strategy name
    2. Convenience method for one-step chunking
    3. Default chunker selection
    
    Usage:
        # Quick chunking with defaults
        chunked = ChunkingFactory.chunk(document)
        
        # With specific strategy
        chunked = ChunkingFactory.chunk(document, strategy="sentence")
        
        # With custom parameters
        chunked = ChunkingFactory.chunk(
            document,
            strategy="recursive",
            chunk_size=1000,
            chunk_overlap=100
        )
    """
    
    # Map strategy names to chunker classes
    CHUNKERS = {
        ChunkingStrategy.RECURSIVE: RecursiveChunker,
        ChunkingStrategy.SENTENCE: SentenceChunker,
    }
    
    # Default strategy
    DEFAULT_STRATEGY = ChunkingStrategy.RECURSIVE
    
    @classmethod
    def get_chunker(
        cls,
        strategy: str | ChunkingStrategy | None = None,
        **kwargs
    ) -> BaseChunker:
        """
        Get a chunker instance.
        
        Args:
            strategy: Chunking strategy name or enum
            **kwargs: Additional arguments for the chunker
        
        Returns:
            BaseChunker: Configured chunker instance
        
        Raises:
            ChunkingError: If strategy is unknown
        
        Example:
            chunker = ChunkingFactory.get_chunker("recursive", chunk_size=500)
        """
        # Use default if not specified
        if strategy is None:
            strategy = cls.DEFAULT_STRATEGY
        
        # Convert string to enum
        if isinstance(strategy, str):
            try:
                strategy = ChunkingStrategy(strategy.lower())
            except ValueError:
                raise ChunkingError(
                    message=f"Unknown chunking strategy: {strategy}",
                )
        
        # Get chunker class
        chunker_class = cls.CHUNKERS.get(strategy)
        
        if chunker_class is None:
            raise ChunkingError(
                message=f"No chunker for strategy: {strategy}",
            )
        
        logger.debug(
            "Creating chunker",
            strategy=strategy.value,
            chunker_class=chunker_class.__name__,
            kwargs=kwargs
        )
        
        return chunker_class(**kwargs)
    
    @classmethod
    def chunk(
        cls,
        document: Document,
        strategy: str | ChunkingStrategy | None = None,
        **kwargs
    ) -> ChunkedDocument:
        """
        Chunk a document using the specified strategy.
        
        This is a convenience method that creates a chunker
        and immediately chunks the document.
        
        Args:
            document: Document to chunk
            strategy: Chunking strategy (default: recursive)
            **kwargs: Additional arguments for the chunker
        
        Returns:
            ChunkedDocument: Document with its chunks
        
        Raises:
            ChunkingError: If chunking fails
        
        Example:
            chunked = ChunkingFactory.chunk(document)
            chunked = ChunkingFactory.chunk(document, strategy="sentence")
            chunked = ChunkingFactory.chunk(
                document,
                strategy="recursive",
                chunk_size=1000
            )
        """
        chunker = cls.get_chunker(strategy, **kwargs)
        return chunker.chunk(document)
    
    @classmethod
    def get_available_strategies(cls) -> list[str]:
        """
        Get list of available chunking strategies.
        
        Returns:
            list[str]: Strategy names
        """
        return [s.value for s in ChunkingStrategy]
    
    @classmethod
    def get_default_strategy(cls) -> str:
        """
        Get the default chunking strategy.
        
        Returns:
            str: Default strategy name
        """
        return cls.DEFAULT_STRATEGY.value