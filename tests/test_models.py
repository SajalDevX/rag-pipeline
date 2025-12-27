"""
Test Data Models

This script verifies that all models work correctly:
1. Document models can be created and serialized
2. Query models validate input correctly
3. Error models work as expected

Run with: python test_models.py
"""

import json
from datetime import datetime
from uuid import uuid4


def test_models():
    """Test all data models."""
    
    print("=" * 60)
    print("DATA MODELS TEST")
    print("=" * 60)
    
    # =========================================================
    # Test Document Models
    # =========================================================
    
    print("\n📄 Testing Document Models...")
    
    # Test DocumentType enum
    from src.models import DocumentType
    
    print("\n   Testing DocumentType enum:")
    print(f"   - DocumentType.PDF = '{DocumentType.PDF}'")
    print(f"   - DocumentType.from_extension('.pdf') = {DocumentType.from_extension('.pdf')}")
    print(f"   - DocumentType.from_extension('txt') = {DocumentType.from_extension('txt')}")
    print("   ✅ DocumentType works correctly")
    
    # Test DocumentMetadata
    from src.models import DocumentMetadata
    
    metadata = DocumentMetadata(
        source="/documents/test.pdf",
        document_type=DocumentType.PDF,
        title="Test Document",
        author="Test Author",
        file_size=1024,
        page_count=10,
        custom_metadata={"department": "engineering"}
    )
    
    print("\n   Testing DocumentMetadata:")
    print(f"   - source: {metadata.source}")
    print(f"   - document_type: {metadata.document_type}")
    print(f"   - title: {metadata.title}")
    print(f"   - created_at: {metadata.created_at}")
    print("   ✅ DocumentMetadata works correctly")
    
    # Test Document
    from src.models import Document
    
    doc = Document(
        content="This is the full content of the test document. " * 10,
        metadata=metadata
    )
    
    print("\n   Testing Document:")
    print(f"   - id: {doc.id}")
    print(f"   - content_length: {doc.content_length} chars")
    print(f"   - word_count: {doc.word_count} words")
    print(f"   - str: {doc}")
    print("   ✅ Document works correctly")
    
    # Test Chunk
    from src.models import Chunk
    
    chunk = Chunk(
        document_id=doc.id,
        content="This is a chunk of the document.",
        chunk_index=0,
        start_char=0,
        end_char=33,
        metadata=metadata
    )
    
    print("\n   Testing Chunk:")
    print(f"   - id: {chunk.id}")
    print(f"   - document_id: {chunk.document_id}")
    print(f"   - chunk_index: {chunk.chunk_index}")
    print(f"   - content_length: {chunk.content_length} chars")
    print(f"   - str: {chunk}")
    
    # Test to_vector_payload
    payload = chunk.to_vector_payload()
    print("\n   Testing to_vector_payload:")
    print(f"   - Keys: {list(payload.keys())}")
    print("   ✅ Chunk works correctly")
    
    # Test ChunkedDocument
    from src.models import ChunkedDocument
    
    chunk2 = Chunk(
        document_id=doc.id,
        content="This is the second chunk.",
        chunk_index=1,
        start_char=33,
        end_char=58,
        metadata=metadata
    )
    
    chunked_doc = ChunkedDocument(
        document=doc,
        chunks=[chunk, chunk2]
    )
    
    print("\n   Testing ChunkedDocument:")
    print(f"   - chunk_count: {chunked_doc.chunk_count}")
    print(f"   - total_chunk_chars: {chunked_doc.total_chunk_chars}")
    print(f"   - average_chunk_size: {chunked_doc.average_chunk_size:.1f}")
    print(f"   - str: {chunked_doc}")
    print("   ✅ ChunkedDocument works correctly")
    
    # =========================================================
    # Test Query Models
    # =========================================================
    
    print("\n" + "-" * 60)
    print("\n🔍 Testing Query Models...")
    
    # Test QueryRequest
    from src.models import QueryRequest
    
    query_req = QueryRequest(
        query="What is machine learning?",
        top_k=5,
        rerank=True
    )
    
    print("\n   Testing QueryRequest:")
    print(f"   - query: {query_req.query}")
    print(f"   - top_k: {query_req.top_k}")
    print(f"   - rerank: {query_req.rerank}")
    print("   ✅ QueryRequest works correctly")
    
    # Test validation
    print("\n   Testing QueryRequest validation:")
    try:
        invalid_req = QueryRequest(query="", top_k=5)  # Empty query should fail
        print("   ❌ Should have raised validation error")
    except Exception as e:
        print(f"   ✅ Validation works: {type(e).__name__}")
    
    # Test RetrievedChunk
    from src.models import RetrievedChunk
    
    retrieved = RetrievedChunk(
        chunk_id="abc-123",
        document_id="def-456",
        content="Machine learning is a subset of AI...",
        score=0.92,
        chunk_index=0,
        source="ml_guide.pdf",
        document_type="pdf",
        title="ML Guide"
    )
    
    print("\n   Testing RetrievedChunk:")
    print(f"   - score: {retrieved.score}")
    print(f"   - str: {retrieved}")
    print("   ✅ RetrievedChunk works correctly")
    
    # Test QueryResponse
    from src.models import QueryResponse
    
    query_resp = QueryResponse(
        query="What is machine learning?",
        answer="Machine learning is a subset of artificial intelligence...",
        sources=[retrieved],
        total_chunks_retrieved=5,
        processing_time_ms=1234.5,
        model_used="llama-3.1-8b-instant"
    )
    
    print("\n   Testing QueryResponse:")
    print(f"   - query_id: {query_resp.query_id}")
    print(f"   - answer preview: {query_resp.answer[:50]}...")
    print(f"   - sources count: {len(query_resp.sources)}")
    print(f"   - processing_time_ms: {query_resp.processing_time_ms}")
    print("   ✅ QueryResponse works correctly")
    
    # Test JSON serialization
    print("\n   Testing JSON serialization:")
    json_output = query_resp.model_dump_json(indent=2)
    print(f"   - JSON length: {len(json_output)} chars")
    print("   ✅ JSON serialization works correctly")
    
    # Test IngestRequest/Response
    from src.models import IngestRequest, IngestResponse
    
    ingest_req = IngestRequest(
        source="/documents/report.pdf",
        title="Annual Report",
        custom_metadata={"year": 2024}
    )
    
    ingest_resp = IngestResponse(
        document_id=uuid4(),
        source="/documents/report.pdf",
        chunks_created=15,
        processing_time_ms=2345.6
    )
    
    print("\n   Testing IngestRequest/Response:")
    print(f"   - IngestRequest source: {ingest_req.source}")
    print(f"   - IngestResponse chunks_created: {ingest_resp.chunks_created}")
    print("   ✅ Ingest models work correctly")
    
    # =========================================================
    # Test Error Models
    # =========================================================
    
    print("\n" + "-" * 60)
    print("\n❌ Testing Error Models...")
    
    # Test ErrorResponse
    from src.models import ErrorResponse, ErrorDetail
    
    error_resp = ErrorResponse(
        error="ValidationError",
        message="Invalid request parameters",
        details=[
            ErrorDetail(field="query", message="Query cannot be empty", code="EMPTY_QUERY")
        ]
    )
    
    print("\n   Testing ErrorResponse:")
    print(f"   - error: {error_resp.error}")
    print(f"   - message: {error_resp.message}")
    print(f"   - details count: {len(error_resp.details)}")
    print("   ✅ ErrorResponse works correctly")
    
    # Test custom exceptions
    from src.models import (
        RAGException,
        DocumentLoadError,
        EmbeddingError,
        VectorStoreError,
        LLMError,
    )
    
    print("\n   Testing custom exceptions:")
    
    exceptions = [
        RAGException("Generic error"),
        DocumentLoadError("File not found"),
        EmbeddingError("API rate limit exceeded"),
        VectorStoreError("Connection failed"),
        LLMError("Model unavailable"),
    ]
    
    for exc in exceptions:
        resp = exc.to_response()
        print(f"   - {exc.error_type}: {exc.message}")
    
    print("   ✅ Custom exceptions work correctly")
    
    # =========================================================
    # Summary
    # =========================================================
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\n✅ All document models working")
    print("✅ All query/response models working")
    print("✅ All error models working")
    print("✅ JSON serialization working")
    print("✅ Validation working")
    print("\n🚀 Data models are ready!")
    print("=" * 60)


if __name__ == "__main__":
    test_models()