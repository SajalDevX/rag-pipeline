"""
Text Document Loader

This module provides functionality to load plain text files including:
- .txt files
- .md (Markdown) files
- .html files (extracts text, strips tags)

Features:
- Auto-detects encoding
- Handles different line endings
- Strips HTML tags from .html files
"""

from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from src.config import get_logger
from src.models import Document, DocumentType
from src.models.errors import DocumentLoadError, ErrorDetail

from .base import BaseLoader

logger = get_logger(__name__)


class TextLoader(BaseLoader):
    """
    Loader for text-based documents.
    
    Supports:
    - Plain text (.txt)
    - Markdown (.md)
    - HTML (.html, .htm) - strips tags, extracts text
    
    Attributes:
        source: Path to the text file
        encoding: File encoding (default: utf-8)
    
    Usage:
        loader = TextLoader("/path/to/document.txt")
        document = loader.load()
        
        # With specific encoding
        loader = TextLoader("/path/to/document.txt", encoding="latin-1")
        document = loader.load()
    """
    
    # Supported extensions and their document types
    SUPPORTED_EXTENSIONS = {
        ".txt": DocumentType.TEXT,
        ".text": DocumentType.TEXT,
        ".md": DocumentType.MARKDOWN,
        ".markdown": DocumentType.MARKDOWN,
        ".html": DocumentType.HTML,
        ".htm": DocumentType.HTML,
    }
    
    def __init__(self, source: str, encoding: str = "utf-8"):
        """
        Initialize the text loader.
        
        Args:
            source: Path to the text file
            encoding: File encoding (default: utf-8)
        """
        # Determine document type from extension
        path = Path(source)
        extension = path.suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise DocumentLoadError(
                message=f"Unsupported text file extension: {extension}",
                details=[ErrorDetail(
                    field="source",
                    message=f"Supported: {list(self.SUPPORTED_EXTENSIONS.keys())}"
                )]
            )
        
        document_type = self.SUPPORTED_EXTENSIONS[extension]
        super().__init__(source=source, document_type=document_type)
        
        self.encoding = encoding
        self._content: str | None = None
    
    def load(self) -> Document:
        """
        Load the text file and extract content.
        
        Returns:
            Document: Loaded document with content and metadata
        
        Raises:
            DocumentLoadError: If file doesn't exist or cannot be read
        """
        path = Path(self.source)
        
        # Check if file exists
        if not path.exists():
            raise DocumentLoadError(
                message=f"File not found: {self.source}",
                details=[ErrorDetail(field="source", message="File does not exist")]
            )
        
        # Check if it's a file
        if not path.is_file():
            raise DocumentLoadError(
                message=f"Path is not a file: {self.source}",
                details=[ErrorDetail(field="source", message="Path is a directory")]
            )
        
        self.logger.info(
            "Loading text file",
            source=self.source,
            document_type=self.document_type
        )
        
        try:
            # Try to read with specified encoding
            content = self._read_file(path)
            
            # If HTML, extract text from tags
            if self.document_type == DocumentType.HTML:
                content = self._extract_html_text(content)
            
            self._content = content
            
            # Extract metadata
            metadata = self.extract_metadata()
            
            # Get file size
            file_size = self._get_file_size(path)
            
            # Create and return document
            return self._create_document(
                content=content,
                title=metadata.get("title") or path.stem,  # Use filename as fallback title
                file_size=file_size
            )
            
        except UnicodeDecodeError as e:
            raise DocumentLoadError(
                message=f"Failed to decode file with encoding '{self.encoding}': {self.source}",
                details=[ErrorDetail(field="encoding", message=str(e))]
            )
        except Exception as e:
            raise DocumentLoadError(
                message=f"Failed to load text file: {self.source}",
                details=[ErrorDetail(field="source", message=str(e))]
            )
    
    def _read_file(self, path: Path) -> str:
        """
        Read file content with encoding handling.
        
        Tries the specified encoding first, then falls back to
        common encodings if that fails.
        
        Args:
            path: Path to the file
        
        Returns:
            str: File content
        
        Raises:
            UnicodeDecodeError: If no encoding works
        """
        # List of encodings to try
        encodings = [self.encoding, "utf-8", "latin-1", "cp1252"]
        
        # Remove duplicates while preserving order
        seen = set()
        encodings = [e for e in encodings if not (e in seen or seen.add(e))]
        
        last_error = None
        
        for encoding in encodings:
            try:
                with open(path, "r", encoding=encoding) as f:
                    content = f.read()
                
                # If not the expected encoding, log a warning
                if encoding != self.encoding:
                    self.logger.warning(
                        "Used fallback encoding",
                        expected=self.encoding,
                        actual=encoding,
                        source=self.source
                    )
                
                return content
                
            except UnicodeDecodeError as e:
                last_error = e
                continue
        
        # If we get here, no encoding worked
        raise last_error or UnicodeDecodeError(
            "utf-8", b"", 0, 0, "Could not decode file"
        )
    
    def _extract_html_text(self, html_content: str) -> str:
        """
        Extract plain text from HTML content.
        
        Uses BeautifulSoup to parse HTML and extract text,
        removing all tags and scripts.
        
        Args:
            html_content: Raw HTML string
        
        Returns:
            str: Plain text extracted from HTML
        """
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Remove script and style elements
            for element in soup(["script", "style", "head", "meta", "link"]):
                element.decompose()
            
            # Get text
            text = soup.get_text(separator="\n")
            
            return text
            
        except Exception as e:
            self.logger.warning(
                "Failed to parse HTML, returning raw content",
                source=self.source,
                error=str(e)
            )
            return html_content
    
    def extract_metadata(self) -> dict[str, Any]:
        """
        Extract metadata from the text file.
        
        For HTML files, tries to extract title from <title> tag.
        For Markdown, tries to extract title from first # heading.
        
        Returns:
            dict: Extracted metadata
        """
        metadata = {"title": None}
        
        if not self._content:
            return metadata
        
        try:
            if self.document_type == DocumentType.HTML:
                # Try to extract title from HTML
                soup = BeautifulSoup(self._content, "html.parser")
                title_tag = soup.find("title")
                if title_tag:
                    metadata["title"] = title_tag.get_text().strip()
            
            elif self.document_type == DocumentType.MARKDOWN:
                # Try to extract title from first # heading
                lines = self._content.split("\n")
                for line in lines:
                    line = line.strip()
                    if line.startswith("# "):
                        metadata["title"] = line[2:].strip()
                        break
            
        except Exception as e:
            self.logger.warning(
                "Failed to extract metadata",
                source=self.source,
                error=str(e)
            )
        
        return metadata