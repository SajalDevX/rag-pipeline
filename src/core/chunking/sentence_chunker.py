"""
Sentence-Based Chunker

This chunker splits text by sentences and groups them into chunks.
It ensures that sentences are never split in the middle.

Best for:
- Text where sentence boundaries are important
- Documents with clear sentence structure
- When you want complete thoughts in each chunk
"""

import nltk
from nltk.tokenize import sent_tokenize

from src.config import get_logger
from src.models.errors import ChunkingError

from .base import BaseChunker

logger = get_logger(__name__)


class SentenceChunker(BaseChunker):
    """
    Sentence-based text chunker.
    
    Splits text into sentences using NLTK, then groups sentences
    into chunks that don't exceed the size limit.
    
    Attributes:
        chunk_size: Maximum characters per chunk
        chunk_overlap: Number of sentences to overlap (not characters)
        min_sentences: Minimum sentences per chunk
    
    Usage:
        chunker = SentenceChunker(chunk_size=500, chunk_overlap=1)
        chunked_doc = chunker.chunk(document)
    """
    
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_sentences: int = 1
    ):
        """
        Initialize the sentence chunker.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Number of sentences to overlap between chunks
            min_sentences: Minimum sentences per chunk
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.min_sentences = min_sentences
        
        # Ensure NLTK data is available
        self._ensure_nltk_data()
    
    def _ensure_nltk_data(self) -> None:
        """
        Ensure required NLTK data is downloaded.
        """
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            self.logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            self.logger.info("Downloading NLTK punkt_tab...")
            nltk.download('punkt_tab', quiet=True)
    
    def _split_text(self, text: str) -> list[str]:
        """
        Split text into sentence-based chunks.
        
        Args:
            text: Text to split
        
        Returns:
            list[str]: List of text chunks
        """
        # Tokenize into sentences
        try:
            sentences = sent_tokenize(text)
        except Exception as e:
            self.logger.warning(
                "NLTK sentence tokenization failed, falling back to simple split",
                error=str(e)
            )
            sentences = self._simple_sentence_split(text)
        
        if not sentences:
            return [text] if text.strip() else []
        
        # Group sentences into chunks
        chunks = []
        current_chunk_sentences = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed limit
            # +1 for the space between sentences
            new_length = current_length + sentence_length + (1 if current_chunk_sentences else 0)
            
            if new_length <= self.chunk_size:
                # Add to current chunk
                current_chunk_sentences.append(sentence)
                current_length = new_length
            else:
                # Current chunk is full
                if current_chunk_sentences:
                    chunks.append(" ".join(current_chunk_sentences))
                
                # Handle sentence longer than chunk_size
                if sentence_length > self.chunk_size:
                    # Split long sentence
                    sub_chunks = self._split_long_sentence(sentence)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk_sentences = [sub_chunks[-1]] if sub_chunks else []
                    current_length = len(current_chunk_sentences[0]) if current_chunk_sentences else 0
                else:
                    # Start new chunk with this sentence
                    current_chunk_sentences = [sentence]
                    current_length = sentence_length
        
        # Don't forget the last chunk
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
        
        # Apply sentence overlap
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_sentence_overlap(chunks, sentences)
        
        return chunks
    
    def _simple_sentence_split(self, text: str) -> list[str]:
        """
        Simple sentence splitting fallback.
        
        Uses basic punctuation-based splitting.
        
        Args:
            text: Text to split
        
        Returns:
            list[str]: List of sentences
        """
        import re
        
        # Split on sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_long_sentence(self, sentence: str) -> list[str]:
        """
        Split a sentence that's longer than chunk_size.
        
        Tries to split on commas, semicolons, or spaces.
        
        Args:
            sentence: Long sentence to split
        
        Returns:
            list[str]: Sub-chunks
        """
        chunks = []
        
        # Try to split on punctuation first
        import re
        parts = re.split(r'([,;:])\s*', sentence)
        
        current = ""
        for i, part in enumerate(parts):
            if len(current) + len(part) <= self.chunk_size:
                current += part
            else:
                if current.strip():
                    chunks.append(current.strip())
                
                # If part is still too long, split by words
                if len(part) > self.chunk_size:
                    words = part.split()
                    current = ""
                    for word in words:
                        if len(current) + len(word) + 1 <= self.chunk_size:
                            current = current + " " + word if current else word
                        else:
                            if current.strip():
                                chunks.append(current.strip())
                            current = word
                else:
                    current = part
        
        if current.strip():
            chunks.append(current.strip())
        
        return chunks if chunks else [sentence[:self.chunk_size]]
    
    def _apply_sentence_overlap(
        self,
        chunks: list[str],
        original_sentences: list[str]
    ) -> list[str]:
        """
        Apply sentence-based overlap between chunks.
        
        Instead of character overlap, overlaps by number of sentences.
        
        Args:
            chunks: List of chunks
            original_sentences: Original sentences list
        
        Returns:
            list[str]: Chunks with sentence overlap
        """
        # For sentence overlap, we need to track which sentences are in each chunk
        # This is a simplified version that adds the last sentence(s) of each chunk
        # to the beginning of the next chunk
        
        overlapped = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]
            
            # Get last N sentences from previous chunk
            prev_sentences = sent_tokenize(prev_chunk)
            overlap_sentences = prev_sentences[-self.chunk_overlap:] if len(prev_sentences) >= self.chunk_overlap else prev_sentences
            
            # Prepend to current chunk
            overlap_text = " ".join(overlap_sentences)
            new_chunk = overlap_text + " " + curr_chunk
            
            # Trim if too long
            if len(new_chunk) > self.chunk_size:
                # Remove overlap sentences one by one until it fits
                while len(new_chunk) > self.chunk_size and overlap_sentences:
                    overlap_sentences = overlap_sentences[1:]
                    overlap_text = " ".join(overlap_sentences)
                    new_chunk = overlap_text + " " + curr_chunk if overlap_sentences else curr_chunk
            
            overlapped.append(new_chunk.strip())
        
        return overlapped