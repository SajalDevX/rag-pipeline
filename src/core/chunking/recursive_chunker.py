"""
Recursive Character Text Splitter

This is the most commonly used chunking strategy. It tries to split
on natural boundaries (paragraphs, sentences, words) before falling
back to character-level splitting.

This approach:
1. Preserves semantic meaning by respecting natural boundaries
2. Ensures chunks don't exceed the size limit
3. Handles overlap between chunks
"""

import re

from src.config import get_logger
from src.models.errors import ChunkingError

from .base import BaseChunker

logger = get_logger(__name__)


class RecursiveChunker(BaseChunker):
    """
    Recursive character text splitter.
    
    Tries to split on natural boundaries in this order:
    1. Double newlines (paragraphs)
    2. Single newlines
    3. Sentences (. ! ?)
    4. Commas and semicolons
    5. Spaces (words)
    6. Characters (last resort)
    
    Attributes:
        chunk_size: Maximum characters per chunk
        chunk_overlap: Characters to overlap between chunks
        separators: List of separators to try in order
    
    Usage:
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
        chunked_doc = chunker.chunk(document)
    """
    
    # Default separators in order of preference
    DEFAULT_SEPARATORS = [
        "\n\n",      # Paragraphs
        "\n",        # Lines
        ". ",        # Sentences
        "? ",        # Questions
        "! ",        # Exclamations
        "; ",        # Semicolons
        ", ",        # Commas
        " ",         # Words
        "",          # Characters (last resort)
    ]
    
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None
    ):
        """
        Initialize the recursive chunker.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap
            separators: Custom list of separators (optional)
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.separators = separators or self.DEFAULT_SEPARATORS
    
    def _split_text(self, text: str) -> list[str]:
        """
        Split text recursively using separators.
        
        Args:
            text: Text to split
        
        Returns:
            list[str]: List of text chunks
        """
        return self._recursive_split(text, self.separators)
    
    def _recursive_split(
        self,
        text: str,
        separators: list[str]
    ) -> list[str]:
        """
        Recursively split text using the given separators.
        
        Args:
            text: Text to split
            separators: Remaining separators to try
        
        Returns:
            list[str]: List of text chunks
        """
        # Base case: no more separators, split by characters
        if not separators:
            return self._split_by_characters(text)
        
        # Get current separator
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Split by current separator
        if separator == "":
            # Character-level split
            splits = list(text)
        else:
            # Use the separator
            splits = text.split(separator)
        
        # Process each split
        chunks = []
        current_chunk = ""
        
        for i, split in enumerate(splits):
            # Add separator back (except for last split and character split)
            piece = split
            if separator and i < len(splits) - 1:
                piece = split + separator
            
            # Check if adding this piece exceeds chunk size
            if len(current_chunk) + len(piece) <= self.chunk_size:
                current_chunk += piece
            else:
                # Current chunk is full
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Check if the piece itself is too large
                if len(piece) > self.chunk_size:
                    # Recursively split with remaining separators
                    sub_chunks = self._recursive_split(piece, remaining_separators)
                    
                    # Add all but last sub-chunk
                    chunks.extend(sub_chunks[:-1])
                    
                    # Start new current_chunk with last sub-chunk
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    current_chunk = piece
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Apply overlap
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)
        
        # Merge small chunks
        chunks = self._merge_small_chunks(chunks)
        
        return chunks
    
    def _split_by_characters(self, text: str) -> list[str]:
        """
        Split text by characters as last resort.
        
        Args:
            text: Text to split
        
        Returns:
            list[str]: Character-level chunks
        """
        chunks = []
        
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """
        Apply overlap between chunks.
        
        Takes the end of each chunk and prepends it to the next chunk.
        
        Args:
            chunks: List of chunks without overlap
        
        Returns:
            list[str]: Chunks with overlap applied
        """
        if len(chunks) <= 1:
            return chunks
        
        overlapped = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]
            
            # Get overlap from end of previous chunk
            overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
            
            # Try to find a good break point in the overlap
            overlap_text = self._find_overlap_boundary(overlap_text)
            
            # Prepend overlap to current chunk
            new_chunk = overlap_text + " " + curr_chunk
            
            # Trim if too long
            if len(new_chunk) > self.chunk_size:
                new_chunk = new_chunk[:self.chunk_size]
            
            overlapped.append(new_chunk.strip())
        
        return overlapped
    
    def _find_overlap_boundary(self, text: str) -> str:
        """
        Find a good boundary for overlap text.
        
        Tries to start the overlap at a word boundary.
        
        Args:
            text: Overlap text
        
        Returns:
            str: Adjusted overlap text
        """
        # Try to find a space near the start
        space_index = text.find(" ")
        
        if space_index != -1 and space_index < len(text) // 2:
            return text[space_index + 1:]
        
        return text