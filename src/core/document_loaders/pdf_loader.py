"""
PDF Document Loader

This module provides functionality to load and extract text from PDF files.
It uses the pypdf library for PDF parsing.

Features:
- Extracts text from all pages
- Extracts metadata (title, author, page count)
- Handles encrypted PDFs (if not password protected)
- Handles malformed PDFs gracefully
"""

from pathlib import Path
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.config import get_logger
from src.models import Document, DocumentType
from src.models.errors import DocumentLoadError, ErrorDetail

from .base import BaseLoader

logger = get_logger(__name__)


class PDFLoader(BaseLoader):
    """
    Loader for PDF documents.
    
    Uses pypdf to extract text and metadata from PDF files.
    
    Attributes:
        source: Path to the PDF file
        password: Optional password for encrypted PDFs
    
    Usage:
        loader = PDFLoader("/path/to/document.pdf")
        document = loader.load()
        
        # With password
        loader = PDFLoader("/path/to/encrypted.pdf", password="secret")
        document = loader.load()
    """
    
    def __init__(self, source: str, password: str | None = None):
        """
        Initialize the PDF loader.
        
        Args:
            source: Path to the PDF file
            password: Optional password for encrypted PDFs
        """
        super().__init__(source=source, document_type=DocumentType.PDF)
        self.password = password
        self._reader: PdfReader | None = None
    
    def load(self) -> Document:
        """
        Load the PDF and extract text content.
        
        Returns:
            Document: Loaded document with content and metadata
        
        Raises:
            DocumentLoadError: If file doesn't exist, is not a PDF,
                             or cannot be read
        """
        path = Path(self.source)
        
        # Check if file exists
        if not path.exists():
            raise DocumentLoadError(
                message=f"PDF file not found: {self.source}",
                details=[ErrorDetail(field="source", message="File does not exist")]
            )
        
        # Check if it's a file (not directory)
        if not path.is_file():
            raise DocumentLoadError(
                message=f"Path is not a file: {self.source}",
                details=[ErrorDetail(field="source", message="Path is a directory, not a file")]
            )
        
        # Check file extension
        if path.suffix.lower() != '.pdf':
            raise DocumentLoadError(
                message=f"File is not a PDF: {self.source}",
                details=[ErrorDetail(field="source", message=f"Expected .pdf, got {path.suffix}")]
            )
        
        self.logger.info("Loading PDF file", source=self.source)
        
        try:
            # Open and read the PDF
            self._reader = PdfReader(path, password=self.password)
            
            # Extract text from all pages
            text_parts = []
            for page_num, page in enumerate(self._reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    self.logger.warning(
                        "Failed to extract text from page",
                        page=page_num,
                        error=str(e)
                    )
            
            # Combine all text
            content = "\n\n".join(text_parts)
            
            # Extract metadata
            metadata = self.extract_metadata()
            
            # Get file size
            file_size = self._get_file_size(path)
            
            # Create and return document
            return self._create_document(
                content=content,
                title=metadata.get("title"),
                author=metadata.get("author"),
                file_size=file_size,
                page_count=len(self._reader.pages),
                custom_metadata=metadata.get("custom", {})
            )
            
        except PdfReadError as e:
            raise DocumentLoadError(
                message=f"Failed to read PDF: {self.source}",
                details=[ErrorDetail(field="source", message=f"PDF read error: {str(e)}")]
            )
        except Exception as e:
            raise DocumentLoadError(
                message=f"Unexpected error loading PDF: {self.source}",
                details=[ErrorDetail(field="source", message=str(e))]
            )
    
    def extract_metadata(self) -> dict[str, Any]:
        """
        Extract metadata from the PDF.
        
        Returns:
            dict: Metadata including title, author, and other PDF properties
        """
        metadata = {
            "title": None,
            "author": None,
            "custom": {}
        }
        
        if self._reader is None:
            return metadata
        
        try:
            pdf_metadata = self._reader.metadata
            
            if pdf_metadata:
                # Extract standard fields
                metadata["title"] = pdf_metadata.get("/Title")
                metadata["author"] = pdf_metadata.get("/Author")
                
                # Extract additional metadata as custom
                custom = {}
                
                if pdf_metadata.get("/Subject"):
                    custom["subject"] = pdf_metadata.get("/Subject")
                
                if pdf_metadata.get("/Creator"):
                    custom["creator"] = pdf_metadata.get("/Creator")
                
                if pdf_metadata.get("/Producer"):
                    custom["producer"] = pdf_metadata.get("/Producer")
                
                if pdf_metadata.get("/CreationDate"):
                    custom["creation_date"] = str(pdf_metadata.get("/CreationDate"))
                
                if pdf_metadata.get("/ModDate"):
                    custom["modification_date"] = str(pdf_metadata.get("/ModDate"))
                
                metadata["custom"] = custom
                
        except Exception as e:
            self.logger.warning(
                "Failed to extract PDF metadata",
                source=self.source,
                error=str(e)
            )
        
        return metadata