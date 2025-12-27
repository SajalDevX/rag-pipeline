"""
Document Loaders Package

This package provides document loaders for various file types:
- PDFLoader: Load PDF files
- TextLoader: Load text, markdown, and HTML files
- WebLoader: Load web pages from URLs
- DocumentLoaderFactory: Automatically select the right loader

Usage:
    # Recommended: Use the factory for automatic loader selection
    from src.core.document_loaders import DocumentLoaderFactory
    
    document = DocumentLoaderFactory.load("/path/to/file.pdf")
    document = DocumentLoaderFactory.load("https://example.com/article")
    
    # Or use specific loaders directly
    from src.core.document_loaders import PDFLoader, TextLoader, WebLoader
    
    loader = PDFLoader("/path/to/file.pdf")
    document = loader.load()
"""

from src.core.document_loaders.base import BaseLoader
from src.core.document_loaders.factory import DocumentLoaderFactory
from src.core.document_loaders.pdf_loader import PDFLoader
from src.core.document_loaders.text_loader import TextLoader
from src.core.document_loaders.web_loader import WebLoader

__all__ = [
    "BaseLoader",
    "PDFLoader",
    "TextLoader",
    "WebLoader",
    "DocumentLoaderFactory",
]