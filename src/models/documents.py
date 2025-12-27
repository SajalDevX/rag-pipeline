"""
Document Models

This module defines the data structures for documents throughout the pipeline:
- DocumentType: Enum of supported document types
- DocumentMetadata: Metadata associated with a document
- Document: A loaded document before chunking
- Chunk: A piece of a document after chunking
- ChunkedDocument: A document with its chunks

These models are used for:
1. Loading documents from files/URLs
2. Splitting documents into chunks
3. Storing chunks in the vector database
4. Retrieving chunks during queries
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field


class DocumentType(str, Enum):
    """
    Supported document types.
    
    Using str, Enum allows the value to be serialized as a string
    while still providing enum benefits (autocomplete, validation).
    
    Usage:
        doc_type = DocumentType.PDF
        print(doc_type)        # "pdf"
        print(doc_type.value)  # "pdf"
    """
    PDF = "pdf"
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    DOCX = "docx"
    URL = "url"
    
    @classmethod
    def from_extension(cls, extension: str) -> "DocumentType":
        """
        Get DocumentType from file extension.
        
        Args:
            extension: File extension (with or without dot)
        
        Returns:
            Corresponding DocumentType
        
        Raises:
            ValueError: If extension is not supported
        
        Example:
            doc_type = DocumentType.from_extension(".pdf")  # DocumentType.PDF
            doc_type = DocumentType.from_extension("txt")   # DocumentType.TEXT
        """
        # Remove leading dot if present
        ext = extension.lower().lstrip(".")
        
        mapping = {
            "pdf": cls.PDF,
            "txt": cls.TEXT,
            "text": cls.TEXT,
            "md": cls.MARKDOWN,
            "markdown": cls.MARKDOWN,
            "html": cls.HTML,
            "htm": cls.HTML,
            "docx": cls.DOCX,
            "doc": cls.DOCX,
        }
        
        if ext not in mapping:
            raise ValueError(
                f"Unsupported file extension: {extension}. "
                f"Supported: {list(mapping.keys())}"
            )
        
        return mapping[ext]


class DocumentMetadata(BaseModel):
    """
    Metadata associated with a document.
    
    This metadata is:
    1. Extracted during document loading
    2. Inherited by all chunks of the document
    3. Stored in the vector database for filtering
    
    Attributes:
        source: Original file path or URL
        document_type: Type of document (PDF, TEXT, etc.)
        title: Document title (extracted or provided)
        author: Document author if available
        created_at: When the document was ingested
        file_size: Size in bytes (for files)
        page_count: Number of pages (for PDFs)
        language: Document language code
        custom_metadata: User-provided additional metadata
    """
    
    source: str = Field(
        ...,
        description="Original source path or URL",
        examples=["/documents/report.pdf", "https://example.com/article"]
    )
    
    document_type: DocumentType = Field(
        ...,
        description="Type of the document"
    )
    
    title: str | None = Field(
        default=None,
        description="Document title",
        examples=["Annual Report 2024", "Machine Learning Guide"]
    )
    
    author: str | None = Field(
        default=None,
        description="Document author"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when document was ingested"
    )
    
    file_size: int | None = Field(
        default=None,
        ge=0,
        description="File size in bytes"
    )
    
    page_count: int | None = Field(
        default=None,
        ge=0,
        description="Number of pages (for PDFs)"
    )
    
    language: str = Field(
        default="en",
        min_length=2,
        max_length=10,
        description="Language code (ISO 639-1)",
        examples=["en", "es", "fr", "de"]
    )
    
    custom_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional user-defined metadata for filtering"
    )
    
    class Config:
        # Use enum values (strings) instead of enum objects in serialization
        use_enum_values = True


class Document(BaseModel):
    """
    Represents a document after loading, before chunking.
    
    This is the intermediate representation after a file/URL
    has been loaded but before it's split into chunks.
    
    Attributes:
        id: Unique identifier for the document
        content: Full text content of the document
        metadata: Associated metadata
    
    Example:
        doc = Document(
            content="This is the full document text...",
            metadata=DocumentMetadata(
                source="/path/to/file.pdf",
                document_type=DocumentType.PDF,
                title="My Document"
            )
        )
    """
    
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique document identifier"
    )
    
    content: str = Field(
        ...,
        min_length=1,
        description="Full text content of the document"
    )
    
    metadata: DocumentMetadata = Field(
        ...,
        description="Document metadata"
    )
    
    @computed_field
    @property
    def content_length(self) -> int:
        """
        Length of content in characters.
        
        This is a computed field - calculated on access, not stored.
        """
        return len(self.content)
    
    @computed_field
    @property
    def word_count(self) -> int:
        """
        Approximate word count.
        
        Uses simple whitespace splitting for estimation.
        """
        return len(self.content.split())
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Document(id={self.id}, "
            f"source='{self.metadata.source}', "
            f"length={self.content_length} chars)"
        )


class Chunk(BaseModel):
    """
    Represents a chunk of a document.
    
    This is what gets embedded and stored in the vector database.
    Each chunk maintains a reference to its parent document.
    
    Attributes:
        id: Unique chunk identifier
        document_id: ID of the parent document
        content: Text content of the chunk
        chunk_index: Position in the document (0-based)
        start_char: Starting character position in original document
        end_char: Ending character position in original document
        metadata: Inherited from parent document
        embedding: Vector embedding (set after embedding generation)
    
    Example:
        chunk = Chunk(
            document_id=doc.id,
            content="This is chunk content...",
            chunk_index=0,
            start_char=0,
            end_char=500,
            metadata=doc.metadata
        )
    """
    
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique chunk identifier"
    )
    
    document_id: UUID = Field(
        ...,
        description="ID of the parent document"
    )
    
    content: str = Field(
        ...,
        min_length=1,
        description="Text content of the chunk"
    )
    
    chunk_index: int = Field(
        ...,
        ge=0,
        description="Position of chunk in document (0-based)"
    )
    
    start_char: int = Field(
        ...,
        ge=0,
        description="Starting character position in original document"
    )
    
    end_char: int = Field(
        ...,
        ge=0,
        description="Ending character position in original document"
    )
    
    metadata: DocumentMetadata = Field(
        ...,
        description="Metadata inherited from parent document"
    )
    
    # Embedding is optional - set after calling embedding service
    # exclude=True means it won't be included in JSON serialization by default
    embedding: list[float] | None = Field(
        default=None,
        exclude=True,
        description="Vector embedding (set after embedding generation)"
    )
    
    @computed_field
    @property
    def content_length(self) -> int:
        """Length of chunk content in characters."""
        return len(self.content)
    
    def to_vector_payload(self) -> dict[str, Any]:
        """
        Convert chunk to payload format for vector database.
        
        This creates a flat dictionary suitable for storing
        in Zilliz/Milvus alongside the vector.
        
        Returns:
            Dictionary with all fields needed for storage and retrieval
        
        Example:
            payload = chunk.to_vector_payload()
            # {
            #     "chunk_id": "abc-123",
            #     "document_id": "def-456",
            #     "content": "...",
            #     "source": "/path/to/file.pdf",
            #     ...
            # }
        """
        return {
            # IDs as strings for vector DB compatibility
            "chunk_id": str(self.id),
            "document_id": str(self.document_id),
            
            # Content
            "content": self.content,
            
            # Position info
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            
            # Flattened metadata for filtering
            "source": self.metadata.source,
            "document_type": self.metadata.document_type,
            "title": self.metadata.title or "",
            "author": self.metadata.author or "",
            "created_at": self.metadata.created_at.isoformat(),
            "language": self.metadata.language,
            
            # Custom metadata as JSON string for flexibility
            # (Zilliz supports JSON fields)
            "custom_metadata": self.metadata.custom_metadata,
        }
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return (
            f"Chunk(id={self.id}, "
            f"index={self.chunk_index}, "
            f"content='{preview}')"
        )


class ChunkedDocument(BaseModel):
    """
    A document that has been split into chunks.
    
    This is the result of the chunking process, containing
    both the original document and all its chunks.
    
    Attributes:
        document: The original document
        chunks: List of chunks created from the document
    
    Example:
        chunked = ChunkedDocument(
            document=doc,
            chunks=[chunk1, chunk2, chunk3]
        )
        print(f"Document split into {chunked.chunk_count} chunks")
    """
    
    document: Document = Field(
        ...,
        description="The original document"
    )
    
    chunks: list[Chunk] = Field(
        default_factory=list,
        description="List of chunks from the document"
    )
    
    @computed_field
    @property
    def chunk_count(self) -> int:
        """Number of chunks created from the document."""
        return len(self.chunks)
    
    @computed_field
    @property
    def total_chunk_chars(self) -> int:
        """Total characters across all chunks."""
        return sum(chunk.content_length for chunk in self.chunks)
    
    @computed_field
    @property
    def average_chunk_size(self) -> float:
        """Average chunk size in characters."""
        if not self.chunks:
            return 0.0
        return self.total_chunk_chars / len(self.chunks)
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"ChunkedDocument("
            f"source='{self.document.metadata.source}', "
            f"chunks={self.chunk_count}, "
            f"avg_size={self.average_chunk_size:.0f} chars)"
        )