"""
Web Document Loader

This module provides functionality to load content from web URLs.
It fetches the page and extracts text content from HTML.

Features:
- Fetches web pages via HTTP/HTTPS
- Extracts text from HTML
- Extracts metadata (title, description)
- Handles redirects
- Configurable timeout
"""

from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from src.config import get_logger
from src.models import Document, DocumentType
from src.models.errors import DocumentLoadError, ErrorDetail

from .base import BaseLoader

logger = get_logger(__name__)


class WebLoader(BaseLoader):
    """
    Loader for web pages.
    
    Fetches content from URLs and extracts text from HTML.
    
    Attributes:
        source: URL to fetch
        timeout: Request timeout in seconds
        headers: Custom HTTP headers
    
    Usage:
        loader = WebLoader("https://example.com/article")
        document = loader.load()
        
        # With custom timeout
        loader = WebLoader("https://example.com/article", timeout=30)
        document = loader.load()
    """
    
    # Default headers to mimic a browser
    DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    
    def __init__(
        self,
        source: str,
        timeout: int = 30,
        headers: dict[str, str] | None = None
    ):
        """
        Initialize the web loader.
        
        Args:
            source: URL to fetch
            timeout: Request timeout in seconds (default: 30)
            headers: Custom HTTP headers (merged with defaults)
        """
        super().__init__(source=source, document_type=DocumentType.URL)
        
        self.timeout = timeout
        self.headers = {**self.DEFAULT_HEADERS, **(headers or {})}
        
        self._html_content: str | None = None
        self._soup: BeautifulSoup | None = None
        
        # Validate URL
        self._validate_url()
    
    def _validate_url(self) -> None:
        """
        Validate that the source is a valid URL.
        
        Raises:
            DocumentLoadError: If URL is invalid
        """
        try:
            parsed = urlparse(self.source)
            
            if not parsed.scheme:
                raise DocumentLoadError(
                    message=f"Invalid URL (missing scheme): {self.source}",
                    details=[ErrorDetail(
                        field="source",
                        message="URL must start with http:// or https://"
                    )]
                )
            
            if parsed.scheme not in ("http", "https"):
                raise DocumentLoadError(
                    message=f"Unsupported URL scheme: {parsed.scheme}",
                    details=[ErrorDetail(
                        field="source",
                        message="Only http:// and https:// are supported"
                    )]
                )
            
            if not parsed.netloc:
                raise DocumentLoadError(
                    message=f"Invalid URL (missing domain): {self.source}",
                    details=[ErrorDetail(field="source", message="URL must have a domain")]
                )
                
        except DocumentLoadError:
            raise
        except Exception as e:
            raise DocumentLoadError(
                message=f"Invalid URL: {self.source}",
                details=[ErrorDetail(field="source", message=str(e))]
            )
    
    def load(self) -> Document:
        """
        Fetch the web page and extract text content.
        
        Returns:
            Document: Loaded document with content and metadata
        
        Raises:
            DocumentLoadError: If fetch fails or content cannot be extracted
        """
        self.logger.info("Fetching web page", url=self.source)
        
        try:
            # Fetch the page
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                response = client.get(self.source, headers=self.headers)
                response.raise_for_status()
            
            self._html_content = response.text
            
            # Check content type
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                self.logger.warning(
                    "Unexpected content type",
                    url=self.source,
                    content_type=content_type
                )
            
            # Parse HTML
            self._soup = BeautifulSoup(self._html_content, "html.parser")
            
            # Extract text content
            content = self._extract_text()
            
            # Extract metadata
            metadata = self.extract_metadata()
            
            # Create and return document
            return self._create_document(
                content=content,
                title=metadata.get("title"),
                author=metadata.get("author"),
                custom_metadata={
                    "url": self.source,
                    "description": metadata.get("description"),
                    "final_url": str(response.url),  # After redirects
                }
            )
            
        except httpx.TimeoutException:
            raise DocumentLoadError(
                message=f"Request timed out after {self.timeout}s: {self.source}",
                details=[ErrorDetail(field="source", message="Connection timed out")]
            )
        except httpx.HTTPStatusError as e:
            raise DocumentLoadError(
                message=f"HTTP error {e.response.status_code}: {self.source}",
                details=[ErrorDetail(
                    field="source",
                    message=f"Server returned status {e.response.status_code}"
                )]
            )
        except httpx.RequestError as e:
            raise DocumentLoadError(
                message=f"Failed to fetch URL: {self.source}",
                details=[ErrorDetail(field="source", message=str(e))]
            )
        except Exception as e:
            raise DocumentLoadError(
                message=f"Unexpected error fetching URL: {self.source}",
                details=[ErrorDetail(field="source", message=str(e))]
            )
    
    def _extract_text(self) -> str:
        """
        Extract text content from HTML.
        
        Removes scripts, styles, and other non-content elements.
        
        Returns:
            str: Extracted text content
        """
        if not self._soup:
            return ""
        
        # Make a copy to avoid modifying the original
        soup = BeautifulSoup(str(self._soup), "html.parser")
        
        # Remove non-content elements
        for element in soup([
            "script", "style", "head", "meta", "link",
            "nav", "footer", "header", "aside",
            "noscript", "iframe", "svg"
        ]):
            element.decompose()
        
        # Try to find main content area
        main_content = (
            soup.find("main") or
            soup.find("article") or
            soup.find(id="content") or
            soup.find(class_="content") or
            soup.find("body") or
            soup
        )
        
        # Get text with proper spacing
        text = main_content.get_text(separator="\n")
        
        return text
    
    def extract_metadata(self) -> dict[str, Any]:
        """
        Extract metadata from HTML.
        
        Extracts:
        - Title from <title> tag
        - Description from meta tags
        - Author from meta tags
        
        Returns:
            dict: Extracted metadata
        """
        metadata = {
            "title": None,
            "description": None,
            "author": None
        }
        
        if not self._soup:
            return metadata
        
        try:
            # Extract title
            title_tag = self._soup.find("title")
            if title_tag:
                metadata["title"] = title_tag.get_text().strip()
            
            # Also try og:title
            og_title = self._soup.find("meta", property="og:title")
            if og_title and og_title.get("content"):
                metadata["title"] = metadata["title"] or og_title["content"]
            
            # Extract description
            desc_tag = self._soup.find("meta", attrs={"name": "description"})
            if desc_tag and desc_tag.get("content"):
                metadata["description"] = desc_tag["content"]
            
            # Also try og:description
            og_desc = self._soup.find("meta", property="og:description")
            if og_desc and og_desc.get("content"):
                metadata["description"] = metadata["description"] or og_desc["content"]
            
            # Extract author
            author_tag = self._soup.find("meta", attrs={"name": "author"})
            if author_tag and author_tag.get("content"):
                metadata["author"] = author_tag["content"]
            
        except Exception as e:
            self.logger.warning(
                "Failed to extract metadata from HTML",
                url=self.source,
                error=str(e)
            )
        
        return metadata