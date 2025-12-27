"""
Ingestion Service

This service orchestrates the document ingestion pipeline:
1. Load document from file/URL
2. Split into chunks
3. Generate embeddings
4. Store in vector database

Usage:
    service = IngestionService()
    result = await service.ingest("/path/to/document.pdf")
"""

import time
from pathlib import Path
from typing import Any

from src.config import get_logger, settings
from src.core.chunking import ChunkingFactory
from src.core.document_loaders import DocumentLoaderFactory
from src.core.embeddings import EmbeddingFactory
from src.infrastructure.vector_store import ZillizStore
from src.models import (
    ChunkedDocument,
    Document,
    DocumentType,
    IngestResponse,
)
from src.models.errors import DocumentLoadError, RAGException

logger = get_logger(__name__)


class IngestionService:
    """
    Service for ingesting documents into the RAG system.
    
    Orchestrates the complete ingestion pipeline:
    Document → Load → Chunk → Embed → Store
    
    Attributes:
        vector_store: Vector database client
        embedder: Embedding generator
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    
    Usage:
        service = IngestionService()
        
        # Ingest a document
        result = await service.ingest("/path/to/doc.pdf")
        print(f"Created {result.chunks_created} chunks")
        
        # Ingest with custom metadata
        result = await service.ingest(
            source="/path/to/doc.pdf",
            title="My Document",
            custom_metadata={"department": "engineering"}
        )
    """
    
    def __init__(
        self,
        vector_store: ZillizStore | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None
    ):
        """
        Initialize the ingestion service.
        
        Args:
            vector_store: Vector store client (creates new if None)
            chunk_size: Chunk size (default from settings)
            chunk_overlap: Chunk overlap (default from settings)
        """
        self.vector_store = vector_store or ZillizStore()
        self.embedder = EmbeddingFactory.get_embedder()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        self.logger = get_logger(self.__class__.__name__)
        
        self.logger.info(
            "Ingestion service initialized",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    async def ingest(
        self,
        source: str,
        document_type: DocumentType | str | None = None,
        title: str | None = None,
        custom_metadata: dict[str, Any] | None = None,
        chunking_strategy: str = "recursive"
    ) -> IngestResponse:
        """
        Ingest a document into the RAG system.
        
        Args:
            source: File path or URL
            document_type: Type of document (auto-detected if None)
            title: Document title (extracted if None)
            custom_metadata: Additional metadata
            chunking_strategy: Chunking strategy to use
        
        Returns:
            IngestResponse: Result of ingestion
        
        Raises:
            RAGException: If ingestion fails
        """
        start_time = time.time()
        
        self.logger.info(
            "Starting document ingestion",
            source=source,
            document_type=document_type
        )
        
        try:
            # Step 1: Load document
            self.logger.debug("Loading document...")
            document = await self._load_document(
                source=source,
                document_type=document_type,
                title=title,
                custom_metadata=custom_metadata
            )
            
            # Step 2: Chunk document
            self.logger.debug("Chunking document...")
            chunked_doc = await self._chunk_document(
                document=document,
                strategy=chunking_strategy
            )
            
            # Step 3: Generate embeddings
            self.logger.debug("Generating embeddings...")
            chunks_with_embeddings = await self.embedder.embed_chunks(
                chunked_doc.chunks
            )
            
            # Step 4: Store in vector database
            self.logger.debug("Storing in vector database...")
            await self.vector_store.insert_chunks(chunks_with_embeddings)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            self.logger.info(
                "Document ingestion complete",
                document_id=str(document.id),
                chunks_created=len(chunks_with_embeddings),
                processing_time_ms=processing_time
            )
            
            return IngestResponse(
                document_id=document.id,
                source=source,
                chunks_created=len(chunks_with_embeddings),
                processing_time_ms=processing_time,
                status="success",
                message=f"Document ingested successfully. Created {len(chunks_with_embeddings)} chunks."
            )
            
        except RAGException:
            raise
        except Exception as e:
            self.logger.error("Ingestion failed", error=str(e), source=source)
            raise DocumentLoadError(
                message=f"Failed to ingest document: {str(e)}"
            )
    
    async def _load_document(
        self,
        source: str,
        document_type: DocumentType | str | None = None,
        title: str | None = None,
        custom_metadata: dict[str, Any] | None = None
    ) -> Document:
        """
        Load a document from source.
        
        Args:
            source: File path or URL
            document_type: Type of document
            title: Override title
            custom_metadata: Additional metadata
        
        Returns:
            Document: Loaded document
        """
        # Load the document
        document = DocumentLoaderFactory.load(
            source=source,
            document_type=document_type
        )
        
        # Override title if provided
        if title:
            document.metadata.title = title
        
        # Add custom metadata
        if custom_metadata:
            document.metadata.custom_metadata.update(custom_metadata)
        
        return document
    
    async def _chunk_document(
        self,
        document: Document,
        strategy: str = "recursive"
    ) -> ChunkedDocument:
        """
        Split document into chunks.
        
        Args:
            document: Document to chunk
            strategy: Chunking strategy
        
        Returns:
            ChunkedDocument: Document with chunks
        """
        return ChunkingFactory.chunk(
            document=document,
            strategy=strategy,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    async def delete_document(self, document_id: str) -> int:
        """
        Delete a document and its chunks.
        
        Args:
            document_id: ID of document to delete
        
        Returns:
            int: Number of chunks deleted
        """
        self.logger.info("Deleting document", document_id=document_id)
        
        deleted = await self.vector_store.delete_by_document_id(document_id)
        
        self.logger.info(
            "Document deleted",
            document_id=document_id,
            chunks_deleted=deleted
        )
        
        return deleted
    
    async def health_check(self) -> dict[str, bool]:
        """
        Check health of all components.
        
        Returns:
            dict: Health status of each component
        """
        return {
            "vector_store": await self.vector_store.health_check(),
            "embedder": await self.embedder.health_check(),
        }