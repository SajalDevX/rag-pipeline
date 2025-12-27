"""
Document Loader Factory

This module provides a factory class that automatically selects
the appropriate loader based on the source (file path or URL).

Usage:
    from src.core.document_loaders import DocumentLoaderFactory
    
    # Load any file type - factory selects the right loader
    document = DocumentLoaderFactory.load("/path/to/file.pdf")
    document = DocumentLoaderFactory.load("/path/to/file.txt")
    document = DocumentLoaderFactory.load("https://example.com/article")
"""

from pathlib import Path
from urllib.parse import urlparse

from src.config import get_logger
from src.models import Document, DocumentType
from src.models.errors import DocumentLoadError, ErrorDetail

from .base import BaseLoader
from .pdf_loader import PDFLoader
from .text_loader import TextLoader
from .web_loader import WebLoader

logger = get_logger(__name__)


class DocumentLoaderFactory:
    """
    Factory class for creating document loaders.
    
    Automatically selects the appropriate loader based on:
    - File extension for local files
    - URL scheme for web pages
    
    Usage:
        # Load a PDF
        doc = DocumentLoaderFactory.load("/path/to/file.pdf")
        
        # Load a text file
        doc = DocumentLoaderFactory.load("/path/to/file.txt")
        
        # Load a web page
        doc = DocumentLoaderFactory.load("https://example.com/page")
        
        # Get loader instance without loading
        loader = DocumentLoaderFactory.get_loader("/path/to/file.pdf")
        doc = loader.load()
    """
    
    # Map file extensions to loader classes
    EXTENSION_LOADERS = {
        ".pdf": PDFLoader,
        ".txt": TextLoader,
        ".text": TextLoader,
        ".md": TextLoader,
        ".markdown": TextLoader,
        ".html": TextLoader,
        ".htm": TextLoader,
    }
    
    @classmethod
    def get_loader(
        cls,
        source: str,
        document_type: DocumentType | str | None = None,
        **kwargs
    ) -> BaseLoader:
        """
        Get the appropriate loader for a source.
        
        Args:
            source: File path or URL
            document_type: Optional explicit document type
            **kwargs: Additional arguments passed to the loader
        
        Returns:
            BaseLoader: Appropriate loader instance
        
        Raises:
            DocumentLoadError: If source type cannot be determined
                             or is not supported
        
        Example:
            loader = DocumentLoaderFactory.get_loader("/path/to/file.pdf")
            loader = DocumentLoaderFactory.get_loader(
                "https://example.com",
                timeout=60
            )
        """
        # Check if it's a URL
        if cls._is_url(source):
            logger.debug("Creating WebLoader", source=source)
            return WebLoader(source, **kwargs)
        
        # It's a file path
        path = Path(source)
        extension = path.suffix.lower()
        
        # If document type is explicitly provided, use it
        if document_type:
            if isinstance(document_type, str):
                try:
                    document_type = DocumentType(document_type)
                except ValueError:
                    raise DocumentLoadError(
                        message=f"Unknown document type: {document_type}",
                        details=[ErrorDetail(
                            field="document_type",
                            message=f"Valid types: {[t.value for t in DocumentType]}"
                        )]
                    )
            
            return cls._get_loader_for_type(source, document_type, **kwargs)
        
        # Auto-detect from extension
        if extension in cls.EXTENSION_LOADERS:
            loader_class = cls.EXTENSION_LOADERS[extension]
            logger.debug(
                "Creating loader from extension",
                source=source,
                extension=extension,
                loader=loader_class.__name__
            )
            return loader_class(source, **kwargs)
        
        # Unknown extension
        raise DocumentLoadError(
            message=f"Unsupported file type: {extension}",
            details=[ErrorDetail(
                field="source",
                message=f"Supported extensions: {list(cls.EXTENSION_LOADERS.keys())}"
            )]
        )
    
    @classmethod
    def load(
        cls,
        source: str,
        document_type: DocumentType | str | None = None,
        **kwargs
    ) -> Document:
        """
        Load a document from a source.
        
        This is a convenience method that creates the loader
        and immediately loads the document.
        
        Args:
            source: File path or URL
            document_type: Optional explicit document type
            **kwargs: Additional arguments passed to the loader
        
        Returns:
            Document: Loaded document
        
        Raises:
            DocumentLoadError: If loading fails
        
        Example:
            doc = DocumentLoaderFactory.load("/path/to/file.pdf")
            print(doc.content_length)
        """
        loader = cls.get_loader(source, document_type, **kwargs)
        return loader.load()
    
    @classmethod
    def _is_url(cls, source: str) -> bool:
        """
        Check if a source is a URL.
        
        Args:
            source: Source string to check
        
        Returns:
            bool: True if source is a URL
        """
        try:
            parsed = urlparse(source)
            return parsed.scheme in ("http", "https")
        except Exception:
            return False
    
    @classmethod
    def _get_loader_for_type(
        cls,
        source: str,
        document_type: DocumentType,
        **kwargs
    ) -> BaseLoader:
        """
        Get loader for explicit document type.
        
        Args:
            source: File path
            document_type: Document type
            **kwargs: Additional arguments
        
        Returns:
            BaseLoader: Appropriate loader
        
        Raises:
            DocumentLoadError: If type is not supported
        """
        type_to_loader = {
            DocumentType.PDF: PDFLoader,
            DocumentType.TEXT: TextLoader,
            DocumentType.MARKDOWN: TextLoader,
            DocumentType.HTML: TextLoader,
            DocumentType.URL: WebLoader,
        }
        
        if document_type not in type_to_loader:
            raise DocumentLoadError(
                message=f"No loader for document type: {document_type}",
                details=[ErrorDetail(
                    field="document_type",
                    message=f"Supported: {list(type_to_loader.keys())}"
                )]
            )
        
        loader_class = type_to_loader[document_type]
        return loader_class(source, **kwargs)
    
    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        """
        Get list of supported file extensions.
        
        Returns:
            list[str]: Supported extensions
        """
        return list(cls.EXTENSION_LOADERS.keys())
    
    @classmethod
    def is_supported(cls, source: str) -> bool:
        """
        Check if a source is supported.
        
        Args:
            source: File path or URL
        
        Returns:
            bool: True if source can be loaded
        """
        # URLs are always supported
        if cls._is_url(source):
            return True
        
        # Check file extension
        extension = Path(source).suffix.lower()
        return extension in cls.EXTENSION_LOADERS