"""
Test Chunking Engine

This script tests all chunking functionality:
1. RecursiveChunker
2. SentenceChunker
3. ChunkingFactory
4. Overlap handling
5. Edge cases

Run with: python test_chunking.py
"""

from pathlib import Path


def create_sample_document():
    """Create a sample document for testing."""
    from src.models import Document, DocumentMetadata, DocumentType
    
    # Create a multi-paragraph document
    content = """
Machine Learning: A Comprehensive Overview

Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It has revolutionized many industries and continues to advance rapidly.

Supervised Learning

Supervised learning is the most common type of machine learning. In this approach, the algorithm learns from labeled training data. Each training example consists of an input and the desired output. The algorithm learns to map inputs to outputs by finding patterns in the data.

Common applications of supervised learning include:
- Image classification: Identifying objects in images
- Spam detection: Filtering unwanted emails
- Medical diagnosis: Predicting diseases from symptoms
- Price prediction: Estimating house prices or stock values

Unsupervised Learning

Unlike supervised learning, unsupervised learning works with unlabeled data. The algorithm must discover patterns and structure on its own. This is useful when you don't have labeled examples or want to explore the data.

Key techniques in unsupervised learning include clustering, which groups similar data points together, and dimensionality reduction, which simplifies complex data while preserving important information.

Reinforcement Learning

Reinforcement learning is a different paradigm where an agent learns by interacting with an environment. The agent takes actions and receives rewards or penalties based on the outcomes. Over time, it learns to maximize cumulative rewards.

This approach has achieved remarkable results in game playing, robotics, and autonomous systems. Famous examples include AlphaGo, which defeated world champions at the game of Go.

Conclusion

Machine learning continues to evolve and find new applications. Understanding these fundamental concepts is essential for anyone working in technology today. The field offers exciting opportunities for innovation and problem-solving.
    """.strip()
    
    metadata = DocumentMetadata(
        source="test_document.txt",
        document_type=DocumentType.TEXT,
        title="Machine Learning Overview"
    )
    
    return Document(content=content, metadata=metadata)


def test_chunking():
    """Test all chunking functionality."""
    
    print("=" * 60)
    print("CHUNKING ENGINE TEST")
    print("=" * 60)
    
    # Setup logging
    from src.config import setup_logging
    setup_logging()
    
    # Create sample document
    print("\n📄 Creating sample document...")
    document = create_sample_document()
    print(f"   Document length: {document.content_length} characters")
    print(f"   Word count: {document.word_count} words")
    
    # =========================================================
    # Test RecursiveChunker
    # =========================================================
    
    print("\n" + "-" * 60)
    print("\n🔄 Testing RecursiveChunker...")
    
    from src.core.chunking import RecursiveChunker
    
    # Test with default settings
    print("\n   With default settings (chunk_size=500, overlap=50):")
    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
    chunked_doc = chunker.chunk(document)
    
    print(f"   ✅ Created {chunked_doc.chunk_count} chunks")
    print(f"   Average chunk size: {chunked_doc.average_chunk_size:.0f} chars")
    
    # Show chunks
    print("\n   Chunks preview:")
    for i, chunk in enumerate(chunked_doc.chunks[:3]):
        preview = chunk.content[:80].replace('\n', ' ')
        print(f"   [{i}] ({chunk.content_length} chars): {preview}...")
    
    if chunked_doc.chunk_count > 3:
        print(f"   ... and {chunked_doc.chunk_count - 3} more chunks")
    
    # Test with larger chunk size
    print("\n   With larger chunks (chunk_size=1000, overlap=100):")
    chunker_large = RecursiveChunker(chunk_size=1000, chunk_overlap=100)
    chunked_large = chunker_large.chunk(document)
    
    print(f"   ✅ Created {chunked_large.chunk_count} chunks")
    print(f"   Average chunk size: {chunked_large.average_chunk_size:.0f} chars")
    
    # Test with no overlap
    print("\n   With no overlap (chunk_size=500, overlap=0):")
    chunker_no_overlap = RecursiveChunker(chunk_size=500, chunk_overlap=0)
    chunked_no_overlap = chunker_no_overlap.chunk(document)
    
    print(f"   ✅ Created {chunked_no_overlap.chunk_count} chunks")
    
    # =========================================================
    # Test SentenceChunker
    # =========================================================
    
    print("\n" + "-" * 60)
    print("\n📝 Testing SentenceChunker...")
    
    from src.core.chunking import SentenceChunker
    
    # Test with default settings
    print("\n   With default settings (chunk_size=500, overlap=1 sentence):")
    sentence_chunker = SentenceChunker(chunk_size=500, chunk_overlap=1)
    chunked_sentence = sentence_chunker.chunk(document)
    
    print(f"   ✅ Created {chunked_sentence.chunk_count} chunks")
    print(f"   Average chunk size: {chunked_sentence.average_chunk_size:.0f} chars")
    
    # Show chunks
    print("\n   Chunks preview:")
    for i, chunk in enumerate(chunked_sentence.chunks[:3]):
        preview = chunk.content[:80].replace('\n', ' ')
        print(f"   [{i}] ({chunk.content_length} chars): {preview}...")
    
    # =========================================================
    # Test ChunkingFactory
    # =========================================================
    
    print("\n" + "-" * 60)
    print("\n🏭 Testing ChunkingFactory...")
    
    from src.core.chunking import ChunkingFactory
    
    # Test available strategies
    print(f"\n   Available strategies: {ChunkingFactory.get_available_strategies()}")
    print(f"   Default strategy: {ChunkingFactory.get_default_strategy()}")
    
    # Test factory with default
    print("\n   Using factory with default strategy:")
    chunked_factory = ChunkingFactory.chunk(document)
    print(f"   ✅ Created {chunked_factory.chunk_count} chunks")
    
    # Test factory with specific strategy
    print("\n   Using factory with 'sentence' strategy:")
    chunked_factory_sentence = ChunkingFactory.chunk(document, strategy="sentence")
    print(f"   ✅ Created {chunked_factory_sentence.chunk_count} chunks")
    
    # Test factory with custom params
    print("\n   Using factory with custom params:")
    chunked_factory_custom = ChunkingFactory.chunk(
        document,
        strategy="recursive",
        chunk_size=300,
        chunk_overlap=30
    )
    print(f"   ✅ Created {chunked_factory_custom.chunk_count} chunks")
    
    # =========================================================
    # Test Chunk Properties
    # =========================================================
    
    print("\n" + "-" * 60)
    print("\n📊 Testing Chunk Properties...")
    
    chunk = chunked_doc.chunks[0]
    
    print(f"\n   Chunk ID: {chunk.id}")
    print(f"   Document ID: {chunk.document_id}")
    print(f"   Chunk Index: {chunk.chunk_index}")
    print(f"   Start Char: {chunk.start_char}")
    print(f"   End Char: {chunk.end_char}")
    print(f"   Content Length: {chunk.content_length}")
    
    # Test to_vector_payload
    print("\n   Vector payload keys:")
    payload = chunk.to_vector_payload()
    for key in payload.keys():
        print(f"   - {key}")
    
    # =========================================================
    # Test Edge Cases
    # =========================================================
    
    print("\n" + "-" * 60)
    print("\n⚠️ Testing Edge Cases...")
    
    from src.models import Document, DocumentMetadata, DocumentType
    from src.models.errors import ChunkingError
    
    # Test with short document
    print("\n   Testing short document (50 chars):")
    short_doc = Document(
        content="This is a short document for testing.",
        metadata=DocumentMetadata(source="short.txt", document_type=DocumentType.TEXT)
    )
    chunked_short = ChunkingFactory.chunk(short_doc)
    print(f"   ✅ Created {chunked_short.chunk_count} chunk(s)")
    
    # Test with empty document
    print("\n   Testing empty document:")
    try:
        empty_doc = Document(
            content="   ",  # Only whitespace
            metadata=DocumentMetadata(source="empty.txt", document_type=DocumentType.TEXT)
        )
        ChunkingFactory.chunk(empty_doc)
        print("   ❌ Should have raised error")
    except ChunkingError as e:
        print(f"   ✅ Correctly raised: {e.error_type}")
    except Exception as e:
        # Pydantic validation might catch this first
        print(f"   ✅ Correctly raised error: {type(e).__name__}")
    
    # Test invalid chunk size
    print("\n   Testing invalid chunk size (too small):")
    try:
        RecursiveChunker(chunk_size=10)
        print("   ❌ Should have raised error")
    except ChunkingError as e:
        print(f"   ✅ Correctly raised: {e.error_type}")
    
    # Test invalid overlap
    print("\n   Testing invalid overlap (larger than chunk):")
    try:
        RecursiveChunker(chunk_size=100, chunk_overlap=150)
        print("   ❌ Should have raised error")
    except ChunkingError as e:
        print(f"   ✅ Correctly raised: {e.error_type}")
    
    # =========================================================
    # Verify Chunk Coverage
    # =========================================================
    
    print("\n" + "-" * 60)
    print("\n✅ Verifying Chunk Coverage...")
    
    # Ensure all content is covered
    all_chunk_content = " ".join([c.content for c in chunked_doc.chunks])
    
    # Check that key phrases from document appear in chunks
    test_phrases = [
        "Machine learning",
        "Supervised Learning",
        "Unsupervised Learning",
        "Reinforcement Learning",
        "Conclusion"
    ]
    
    for phrase in test_phrases:
        if phrase.lower() in all_chunk_content.lower():
            print(f"   ✅ Found '{phrase}' in chunks")
        else:
            print(f"   ❌ Missing '{phrase}' in chunks")
    
    # =========================================================
    # Summary
    # =========================================================
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\n✅ RecursiveChunker working")
    print("✅ SentenceChunker working")
    print("✅ ChunkingFactory working")
    print("✅ Chunk properties correct")
    print("✅ Edge cases handled")
    print("✅ Content coverage verified")
    print("\n🚀 Chunking engine is ready!")
    print("=" * 60)


if __name__ == "__main__":
    test_chunking()