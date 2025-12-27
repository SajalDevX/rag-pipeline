"""
Zilliz Cloud Vector Store

This module provides a client for Zilliz Cloud (managed Milvus)
for storing and searching vector embeddings.

Features:
- Automatic collection creation
- Batch insert operations
- Similarity search with filtering
- Metadata storage and retrieval

Zilliz Cloud Free Tier:
- 1 million vectors
- 2 collections
- Serverless deployment
"""

import asyncio
from typing import Any
from uuid import uuid4

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
)

from src.config import get_logger, settings
from src.models import Chunk, RetrievedChunk
from src.models.errors import ErrorDetail, VectorStoreError

logger = get_logger(__name__)


class ZillizStore:
    """
    Vector store client for Zilliz Cloud.
    
    Handles all vector database operations:
    - Collection management (create, delete)
    - Vector insertion
    - Similarity search
    - Metadata filtering
    
    Attributes:
        uri: Zilliz Cloud endpoint
        token: API token for authentication
        collection_name: Name of the collection to use
        dimension: Vector dimension (must match embedding model)
    
    Usage:
        store = ZillizStore()
        await store.connect()
        
        # Insert chunks with embeddings
        await store.insert_chunks(chunks)
        
        # Search for similar chunks
        results = await store.search(query_embedding, top_k=5)
        
        # Cleanup
        await store.disconnect()
    """
    
    # Field names in the collection
    FIELD_ID = "id"
    FIELD_VECTOR = "vector"
    FIELD_CONTENT = "content"
    FIELD_CHUNK_ID = "chunk_id"
    FIELD_DOCUMENT_ID = "document_id"
    FIELD_CHUNK_INDEX = "chunk_index"
    FIELD_SOURCE = "source"
    FIELD_DOCUMENT_TYPE = "document_type"
    FIELD_TITLE = "title"
    
    def __init__(
        self,
        uri: str | None = None,
        token: str | None = None,
        collection_name: str | None = None,
        dimension: int | None = None
    ):
        """
        Initialize the Zilliz store.
        
        Args:
            uri: Zilliz Cloud endpoint (default from settings)
            token: API token (default from settings)
            collection_name: Collection name (default from settings)
            dimension: Vector dimension (default from settings)
        """
        self.uri = uri or settings.zilliz_uri
        self.token = token or settings.zilliz_token
        self.collection_name = collection_name or settings.zilliz_collection_name
        self.dimension = dimension or settings.embedding_dimension
        
        self._client: MilvusClient | None = None
        self._connected = False
        
        self.logger = get_logger(self.__class__.__name__)
        
        # Validate configuration
        if not self.uri:
            raise VectorStoreError(
                message="Zilliz URI not provided",
                details=[ErrorDetail(field="uri", message="Set ZILLIZ_URI in .env")]
            )
        
        if not self.token:
            raise VectorStoreError(
                message="Zilliz token not provided",
                details=[ErrorDetail(field="token", message="Set ZILLIZ_TOKEN in .env")]
            )
    
    async def connect(self) -> None:
        """
        Connect to Zilliz Cloud.
        
        Creates the collection if it doesn't exist.
        
        Raises:
            VectorStoreError: If connection fails
        """
        if self._connected:
            return
        
        self.logger.info(
            "Connecting to Zilliz Cloud",
            uri=self.uri[:50] + "..." if len(self.uri) > 50 else self.uri,
            collection=self.collection_name
        )
        
        try:
            # Run connection in thread pool (pymilvus is sync)
            await asyncio.get_event_loop().run_in_executor(
                None, self._connect_sync
            )
            
            self._connected = True
            
            self.logger.info(
                "Connected to Zilliz Cloud",
                collection=self.collection_name
            )
            
        except Exception as e:
            self.logger.error("Failed to connect to Zilliz", error=str(e))
            raise VectorStoreError(
                message=f"Failed to connect to Zilliz Cloud: {str(e)}",
                details=[ErrorDetail(field="connection", message=str(e))]
            )
    
    def _connect_sync(self) -> None:
        """
        Synchronous connection logic.
        
        Called from connect() in a thread pool.
        """
        # Create client
        self._client = MilvusClient(
            uri=self.uri,
            token=self.token
        )
        
        # Check if collection exists
        collections = self._client.list_collections()
        
        if self.collection_name not in collections:
            self.logger.info(
                "Creating collection",
                collection=self.collection_name,
                dimension=self.dimension
            )
            self._create_collection_sync()
        else:
            self.logger.info(
                "Collection already exists",
                collection=self.collection_name
            )
    
    def _create_collection_sync(self) -> None:
        """
        Create the collection with schema.
        
        Schema:
        - id: Primary key (auto-generated)
        - vector: The embedding vector
        - content: Text content of the chunk
        - chunk_id: UUID of the chunk
        - document_id: UUID of the parent document
        - chunk_index: Position in document
        - source: Original file/URL
        - document_type: Type of document
        - title: Document title
        """
        # Create collection with auto-generated ID
        self._client.create_collection(
            collection_name=self.collection_name,
            dimension=self.dimension,
            metric_type="COSINE",  # Cosine similarity
            auto_id=True,
            # Additional fields beyond vector
            # Note: MilvusClient simplified API handles basic schema
        )
        
        self.logger.info(
            "Collection created",
            collection=self.collection_name,
            dimension=self.dimension
        )
    
    async def disconnect(self) -> None:
        """
        Disconnect from Zilliz Cloud.
        """
        if not self._connected:
            return
        
        try:
            if self._client:
                await asyncio.get_event_loop().run_in_executor(
                    None, self._client.close
                )
            self._connected = False
            self.logger.info("Disconnected from Zilliz Cloud")
        except Exception as e:
            self.logger.warning("Error disconnecting", error=str(e))
    
    async def insert_chunks(self, chunks: list[Chunk]) -> list[str]:
        """
        Insert chunks with embeddings into the vector store.
        
        Args:
            chunks: List of chunks with embeddings attached
        
        Returns:
            list[str]: List of inserted IDs
        
        Raises:
            VectorStoreError: If insertion fails
        """
        if not chunks:
            return []
        
        # Ensure connected
        await self.connect()
        
        # Validate chunks have embeddings
        for chunk in chunks:
            if chunk.embedding is None:
                raise VectorStoreError(
                    message=f"Chunk {chunk.id} has no embedding"
                )
        
        self.logger.info(
            "Inserting chunks",
            count=len(chunks),
            collection=self.collection_name
        )
        
        try:
            # Prepare data for insertion
            data = []
            for chunk in chunks:
                record = {
                    "vector": chunk.embedding,
                    "chunk_id": str(chunk.id),
                    "document_id": str(chunk.document_id),
                    "content": chunk.content[:65535],  # Max length for VARCHAR
                    "chunk_index": chunk.chunk_index,
                    "source": chunk.metadata.source[:512],
                    "document_type": chunk.metadata.document_type.value if hasattr(chunk.metadata.document_type, 'value') else str(chunk.metadata.document_type),
                    "title": (chunk.metadata.title or "")[:512],
                }
                data.append(record)
            
            # Insert in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.insert(
                    collection_name=self.collection_name,
                    data=data
                )
            )
            
            # Extract IDs from result
            inserted_ids = [str(chunk.id) for chunk in chunks]
            
            self.logger.info(
                "Chunks inserted successfully",
                count=len(inserted_ids)
            )
            
            return inserted_ids
            
        except Exception as e:
            self.logger.error("Failed to insert chunks", error=str(e))
            raise VectorStoreError(
                message=f"Failed to insert chunks: {str(e)}"
            )
    
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_expr: str | None = None,
        **kwargs
    ) -> list[RetrievedChunk]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_expr: Optional filter expression (e.g., "source == 'doc.pdf'")
            **kwargs: Additional search parameters
        
        Returns:
            list[RetrievedChunk]: List of matching chunks with scores
        
        Raises:
            VectorStoreError: If search fails
        """
        # Ensure connected
        await self.connect()
        
        self.logger.debug(
            "Searching vector store",
            top_k=top_k,
            filter=filter_expr,
            collection=self.collection_name
        )
        
        try:
            # Build search parameters
            search_params = {
                "collection_name": self.collection_name,
                "data": [query_embedding],
                "limit": top_k,
                "output_fields": [
                    "chunk_id", "document_id", "content",
                    "chunk_index", "source", "document_type", "title"
                ],
            }
            
            # Add filter if provided
            if filter_expr:
                search_params["filter"] = filter_expr
            
            # Execute search in thread pool
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.search(**search_params)
            )
            
            # Parse results
            retrieved_chunks = []
            
            if results and len(results) > 0:
                for hit in results[0]:  # First query results
                    # Extract entity data
                    entity = hit.get("entity", {})
                    
                    chunk = RetrievedChunk(
                        chunk_id=entity.get("chunk_id", ""),
                        document_id=entity.get("document_id", ""),
                        content=entity.get("content", ""),
                        score=float(hit.get("distance", 0)),  # Cosine similarity
                        chunk_index=entity.get("chunk_index", 0),
                        source=entity.get("source", ""),
                        document_type=entity.get("document_type", ""),
                        title=entity.get("title"),
                    )
                    retrieved_chunks.append(chunk)
            
            self.logger.debug(
                "Search complete",
                results_count=len(retrieved_chunks)
            )
            
            return retrieved_chunks
            
        except Exception as e:
            self.logger.error("Search failed", error=str(e))
            raise VectorStoreError(
                message=f"Vector search failed: {str(e)}"
            )
    
    async def delete_by_document_id(self, document_id: str) -> int:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: Document ID to delete
        
        Returns:
            int: Number of chunks deleted
        
        Raises:
            VectorStoreError: If deletion fails
        """
        await self.connect()
        
        self.logger.info(
            "Deleting chunks by document ID",
            document_id=document_id
        )
        
        try:
            # Delete by filter
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.delete(
                    collection_name=self.collection_name,
                    filter=f'document_id == "{document_id}"'
                )
            )
            
            # Result contains deleted count
            deleted_count = result.get("delete_count", 0) if isinstance(result, dict) else 0
            
            self.logger.info(
                "Chunks deleted",
                document_id=document_id,
                count=deleted_count
            )
            
            return deleted_count
            
        except Exception as e:
            self.logger.error("Delete failed", error=str(e))
            raise VectorStoreError(
                message=f"Failed to delete chunks: {str(e)}"
            )
    
    async def get_collection_stats(self) -> dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            dict: Collection statistics
        """
        await self.connect()
        
        try:
            stats = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.get_collection_stats(self.collection_name)
            )
            
            return {
                "collection_name": self.collection_name,
                "dimension": self.dimension,
                "row_count": stats.get("row_count", 0),
            }
            
        except Exception as e:
            self.logger.warning("Failed to get stats", error=str(e))
            return {
                "collection_name": self.collection_name,
                "dimension": self.dimension,
                "row_count": -1,
                "error": str(e)
            }
    
    async def drop_collection(self) -> bool:
        """
        Drop (delete) the entire collection.
        
        WARNING: This deletes all data!
        
        Returns:
            bool: True if dropped successfully
        """
        await self.connect()
        
        self.logger.warning(
            "Dropping collection",
            collection=self.collection_name
        )
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.drop_collection(self.collection_name)
            )
            
            self.logger.info("Collection dropped", collection=self.collection_name)
            return True
            
        except Exception as e:
            self.logger.error("Failed to drop collection", error=str(e))
            return False
    
    async def health_check(self) -> bool:
        """
        Check if the vector store is healthy.
        
        Returns:
            bool: True if healthy
        """
        try:
            await self.connect()
            stats = await self.get_collection_stats()
            return "error" not in stats
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return False
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ZillizStore("
            f"collection='{self.collection_name}', "
            f"dimension={self.dimension}, "
            f"connected={self._connected})"
        )