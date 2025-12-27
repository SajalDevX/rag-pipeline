"""
Services Package

This package provides the service layer that orchestrates core components:
- IngestionService: Load, chunk, embed, and store documents
- RetrievalService: Search and rerank chunks
- RAGService: Complete RAG pipeline (retrieve + generate)

Usage:
    from src.services import IngestionService, RAGService
    
    # Ingest a document
    ingestion = IngestionService()
    result = await ingestion.ingest("/path/to/document.pdf")
    
    # Query the system
    rag = RAGService()
    response = await rag.query("What is machine learning?")
    print(response.answer)
"""

from src.services.ingestion_service import IngestionService
from src.services.rag_service import RAGService
from src.services.retrieval_service import RetrievalService

__all__ = [
    "IngestionService",
    "RetrievalService",
    "RAGService",
]