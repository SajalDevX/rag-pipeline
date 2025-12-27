"""
Base Document Loader

This module defines the abstract base class that all document loaders
must inherit from. It ensures consistent interface across all loaders.

The base class provides:
1. Common interface (load method)
2. Shared utility methods
3. Error handling patterns
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.config import get_logger
from src.models import Document, DocumentMetadata, DocumentType
from src.models.errors import DocumentLoadError

# Get logger for this module
logger = get_logger(__name__)


class BaseLoader(ABC):
    """
    Abstract base class for all document loaders.
    
    All document loaders must inherit from this class and implement
    the `load` method. This ensures a consistent interface throughout
    the application.
    
    Attributes:
        source: The source path or URL of the document
        document_type: The type of document being loaded
    
    Usage:
        # You don't use BaseLoader directly
        # Instead, use specific loaders:
        loader = PDFLoader("/path/to/file.pdf")
        document = loader.load()
    """
    
    def __init__(self, source: str, document_type: DocumentType):
        """
        Initialize the loader.
        
        Args:
            source: File path or URL to load
            document_type: Type of document
        """
        self.source = source
        self.document_type = document_type
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    def load(self) -> Document:
        """
        Load the document and return a Document object.
        
        This method must be implemented by all subclasses.
        
        Returns:
            Document: The loaded document with content and metadata
        
        Raises:
            DocumentLoadError: If loading fails
        """
        pass
    
    @abstractmethod
    def extract_metadata(self) -> dict[str, Any]:
        """
        Extract metadata from the document.
        
        This method must be implemented by all subclasses.
        
        Returns:
            dict: Extracted metadata
        """
        pass
    
    def _create_document(
        self,
        content: str,
        title: str | None = None,
        author: str | None = None,
        file_size: int | None = None,
        page_count: int | None = None,
        custom_metadata: dict[str, Any] | None = None
    ) -> Document:
        """
        Create a Document object with the given content and metadata.
        
        This is a helper method used by subclasses to create
        standardized Document objects.
        
        Args:
            content: Text content of the document
            title: Document title
            author: Document author
            file_size: File size in bytes
            page_count: Number of pages
            custom_metadata: Additional metadata
        
        Returns:
            Document: Complete document object
        
        Raises:
            DocumentLoadError: If content is empty
        """
        # Validate content
        if not content or not content.strip():
            raise DocumentLoadError(
                message=f"Document has no text content: {self.source}",
                details=[]
            )
        
        # Clean the content
        content = self._clean_text(content)
        
        # Create metadata
        metadata = DocumentMetadata(
            source=self.source,
            document_type=self.document_type,
            title=title,
            author=author,
            file_size=file_size,
            page_count=page_count,
            custom_metadata=custom_metadata or {}
        )
        
        # Create and return document
        document = Document(
            content=content,
            metadata=metadata
        )
        
        self.logger.info(
            "Document loaded successfully",
            source=self.source,
            content_length=document.content_length,
            word_count=document.word_count
        )
        
        return document
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        This method:
        1. Removes excessive whitespace
        2. Normalizes line endings
        3. Strips leading/trailing whitespace
        
        Args:
            text: Raw text content
        
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Replace multiple spaces with single space
        import re
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Strip overall leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _get_file_size(self, path: Path) -> int | None:
        """
        Get file size in bytes.
        
        Args:
            path: Path to the file
        
        Returns:
            int: File size in bytes, or None if cannot determine
        """
        try:
            return path.stat().st_size
        except Exception:
            return None
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"{self.__class__.__name__}(source='{self.source}')"