"""
Base Embedding Manager

This module defines the abstract base class for embedding providers.
All embedding implementations must inherit from this class.

The base class provides:
1. Common interface (embed methods)
2. Batch processing utilities
3. Caching support
"""

from abc import ABC, abstractmethod

from src.config import get_logger
from src.models import Chunk
from src.models.errors import EmbeddingError

logger = get_logger(__name__)


class BaseEmbeddingManager(ABC):
    """
    Abstract base class for embedding managers.
    
    All embedding providers must implement:
    - embed_text(): Embed a single text string
    - embed_texts(): Embed multiple texts in batch
    
    Attributes:
        model_name: Name of the embedding model
        dimension: Dimension of output vectors
    
    Usage:
        # Don't use BaseEmbeddingManager directly
        # Use specific implementations:
        embedder = HuggingFaceEmbedding()
        vector = await embedder.embed_text("Hello world")
    """
    
    def __init__(self, model_name: str, dimension: int):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Name/ID of the embedding model
            dimension: Dimension of output vectors
        """
        self.model_name = model_name
        self.dimension = dimension
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
        
        Returns:
            list[float]: Embedding vector
        
        Raises:
            EmbeddingError: If embedding fails
        """
        pass
    
    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            list[list[float]]: List of embedding vectors
        
        Raises:
            EmbeddingError: If embedding fails
        """
        pass
    
    async def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Generate embeddings for chunks and attach them.
        
        This method:
        1. Extracts text from chunks
        2. Generates embeddings in batch
        3. Attaches embeddings to chunk objects
        
        Args:
            chunks: List of Chunk objects
        
        Returns:
            list[Chunk]: Chunks with embeddings attached
        
        Raises:
            EmbeddingError: If embedding fails
        """
        if not chunks:
            return chunks
        
        self.logger.info(
            "Embedding chunks",
            chunk_count=len(chunks),
            model=self.model_name
        )
        
        # Extract texts
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        embeddings = await self.embed_texts(texts)
        
        # Attach embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        self.logger.info(
            "Chunks embedded successfully",
            chunk_count=len(chunks)
        )
        
        return chunks
    
    def _validate_text(self, text: str) -> str:
        """
        Validate and clean text before embedding.
        
        Args:
            text: Text to validate
        
        Returns:
            str: Cleaned text
        
        Raises:
            EmbeddingError: If text is invalid
        """
        if not text:
            raise EmbeddingError(
                message="Cannot embed empty text"
            )
        
        # Clean whitespace
        text = " ".join(text.split())
        
        if not text.strip():
            raise EmbeddingError(
                message="Cannot embed whitespace-only text"
            )
        
        return text
    
    def _validate_embedding(self, embedding: list[float]) -> None:
        """
        Validate embedding vector.
        
        Args:
            embedding: Vector to validate
        
        Raises:
            EmbeddingError: If embedding is invalid
        """
        if not embedding:
            raise EmbeddingError(
                message="Received empty embedding"
            )
        
        if len(embedding) != self.dimension:
            raise EmbeddingError(
                message=f"Embedding dimension mismatch. "
                       f"Expected {self.dimension}, got {len(embedding)}"
            )
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"model='{self.model_name}', "
            f"dimension={self.dimension})"
        )