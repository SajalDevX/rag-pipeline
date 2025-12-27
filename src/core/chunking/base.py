"""
Base Chunker

This module defines the abstract base class that all chunkers
must inherit from. It ensures a consistent interface.

The base class provides:
1. Common interface (chunk method)
2. Shared utility methods
3. Overlap handling
"""

from abc import ABC, abstractmethod

from src.config import get_logger, settings
from src.models import Chunk, ChunkedDocument, Document
from src.models.errors import ChunkingError

logger = get_logger(__name__)


class BaseChunker(ABC):
    """
    Abstract base class for all text chunkers.
    
    All chunkers must inherit from this class and implement
    the `_split_text` method.
    
    Attributes:
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of overlapping characters between chunks
    
    Usage:
        # You don't use BaseChunker directly
        # Instead, use specific chunkers:
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
        chunked_doc = chunker.chunk(document)
    """
    
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Maximum characters per chunk (default from settings)
            chunk_overlap: Characters to overlap (default from settings)
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        # Validate settings
        self._validate_settings()
        
        self.logger = get_logger(self.__class__.__name__)
    
    def _validate_settings(self) -> None:
        """
        Validate chunking settings.
        
        Raises:
            ChunkingError: If settings are invalid
        """
        if self.chunk_size < 50:
            raise ChunkingError(
                message=f"Chunk size too small: {self.chunk_size}. Minimum is 50."
            )
        
        if self.chunk_overlap < 0:
            raise ChunkingError(
                message=f"Chunk overlap cannot be negative: {self.chunk_overlap}"
            )
        
        if self.chunk_overlap >= self.chunk_size:
            raise ChunkingError(
                message=f"Chunk overlap ({self.chunk_overlap}) must be less than "
                       f"chunk size ({self.chunk_size})"
            )
    
    def chunk(self, document: Document) -> ChunkedDocument:
        """
        Split a document into chunks.
        
        This is the main public method. It:
        1. Validates the document
        2. Splits the text using the subclass implementation
        3. Creates Chunk objects with proper metadata
        4. Returns a ChunkedDocument
        
        Args:
            document: The document to chunk
        
        Returns:
            ChunkedDocument: Document with its chunks
        
        Raises:
            ChunkingError: If chunking fails
        """
        self.logger.info(
            "Starting chunking",
            document_id=str(document.id),
            content_length=document.content_length,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Validate document
        if not document.content or not document.content.strip():
            raise ChunkingError(
                message="Cannot chunk empty document",
            )
        
        try:
            # Split the text
            text_chunks = self._split_text(document.content)
            
            # Create Chunk objects
            chunks = self._create_chunks(document, text_chunks)
            
            # Create ChunkedDocument
            chunked_doc = ChunkedDocument(
                document=document,
                chunks=chunks
            )
            
            self.logger.info(
                "Chunking complete",
                document_id=str(document.id),
                chunk_count=chunked_doc.chunk_count,
                average_chunk_size=chunked_doc.average_chunk_size
            )
            
            return chunked_doc
            
        except ChunkingError:
            raise
        except Exception as e:
            raise ChunkingError(
                message=f"Failed to chunk document: {str(e)}"
            )
    
    @abstractmethod
    def _split_text(self, text: str) -> list[str]:
        """
        Split text into chunks.
        
        This method must be implemented by subclasses.
        
        Args:
            text: The text to split
        
        Returns:
            list[str]: List of text chunks
        """
        pass
    
    def _create_chunks(
        self,
        document: Document,
        text_chunks: list[str]
    ) -> list[Chunk]:
        """
        Create Chunk objects from text chunks.
        
        Calculates character positions and creates proper Chunk objects.
        
        Args:
            document: The source document
            text_chunks: List of text strings
        
        Returns:
            list[Chunk]: List of Chunk objects
        """
        chunks = []
        current_pos = 0
        
        for index, text in enumerate(text_chunks):
            # Find the actual position in the original document
            # This handles overlap correctly
            start_char = document.content.find(text, current_pos)
            
            if start_char == -1:
                # If exact match not found, use estimated position
                start_char = current_pos
            
            end_char = start_char + len(text)
            
            # Create chunk
            chunk = Chunk(
                document_id=document.id,
                content=text,
                chunk_index=index,
                start_char=start_char,
                end_char=end_char,
                metadata=document.metadata
            )
            
            chunks.append(chunk)
            
            # Update position for next chunk (account for overlap)
            current_pos = max(current_pos, end_char - self.chunk_overlap)
        
        return chunks
    
    def _merge_small_chunks(
        self,
        chunks: list[str],
        min_size: int = 50
    ) -> list[str]:
        """
        Merge chunks that are too small.
        
        Args:
            chunks: List of text chunks
            min_size: Minimum chunk size
        
        Returns:
            list[str]: Merged chunks
        """
        if not chunks:
            return chunks
        
        merged = []
        current = ""
        
        for chunk in chunks:
            if len(current) + len(chunk) <= self.chunk_size:
                current = current + " " + chunk if current else chunk
            else:
                if current:
                    merged.append(current.strip())
                current = chunk
        
        # Don't forget the last chunk
        if current:
            # If last chunk is too small, merge with previous
            if len(current) < min_size and merged:
                merged[-1] = merged[-1] + " " + current
            else:
                merged.append(current.strip())
        
        return merged
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap})"
        )