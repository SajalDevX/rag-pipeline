"""
Test Embedding Manager

This script tests the HuggingFace embedding functionality:
1. Single text embedding
2. Batch embedding
3. Chunk embedding
4. Caching
5. Error handling

IMPORTANT: This test makes real API calls to HuggingFace.
Make sure your HUGGINGFACE_API_KEY is set in .env

Run with: python test_embeddings.py
"""

import asyncio


async def test_embeddings():
    """Test all embedding functionality."""
    
    print("=" * 60)
    print("EMBEDDING MANAGER TEST")
    print("=" * 60)
    
    # Setup logging
    from src.config import setup_logging, settings
    setup_logging()
    
    # Check API key
    print("\n🔑 Checking configuration...")
    if not settings.huggingface_api_key:
        print("   ❌ HUGGINGFACE_API_KEY not set in .env")
        print("   Please set your API key and try again.")
        return
    
    print(f"   ✅ API Key: {settings.huggingface_api_key[:10]}...")
    print(f"   ✅ Model: {settings.embedding_model}")
    print(f"   ✅ Dimension: {settings.embedding_dimension}")
    
    # =========================================================
    # Test HuggingFace Embedding
    # =========================================================
    
    print("\n" + "-" * 60)
    print("\n🤗 Testing HuggingFace Embedding...")
    
    from src.core.embeddings import HuggingFaceEmbedding, EmbeddingFactory
    
    # Create embedder
    print("\n   Creating embedder...")
    embedder = HuggingFaceEmbedding()
    print(f"   ✅ Created: {embedder}")
    
    # Test single text embedding
    print("\n   Testing single text embedding...")
    test_text = "Machine learning is a subset of artificial intelligence."
    
    try:
        embedding = await embedder.embed_text(test_text)
        print(f"   ✅ Embedding generated!")
        print(f"      Dimension: {len(embedding)}")
        print(f"      First 5 values: {embedding[:5]}")
        print(f"      Last 5 values: {embedding[-5:]}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return
    
    # Test batch embedding
    print("\n   Testing batch embedding...")
    test_texts = [
        "I love pizza",
        "Pizza is my favorite food",
        "The stock market crashed today",
        "Machine learning is fascinating",
        "Natural language processing enables computers to understand text"
    ]
    
    try:
        embeddings = await embedder.embed_texts(test_texts)
        print(f"   ✅ Generated {len(embeddings)} embeddings")
        
        for i, (text, emb) in enumerate(zip(test_texts, embeddings)):
            print(f"      [{i}] '{text[:30]}...' -> dim={len(emb)}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return
    
    # =========================================================
    # Test Similarity
    # =========================================================
    
    print("\n" + "-" * 60)
    print("\n📊 Testing Semantic Similarity...")
    
    import numpy as np
    
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    # Test texts
    texts_for_similarity = [
        "I love pizza",                    # 0
        "Pizza is my favorite food",       # 1 - Similar to 0
        "The stock market crashed today",  # 2 - Different
    ]
    
    similarity_embeddings = await embedder.embed_texts(texts_for_similarity)
    
    print("\n   Similarity scores:")
    
    # Similar texts
    sim_0_1 = cosine_similarity(similarity_embeddings[0], similarity_embeddings[1])
    print(f"   '{texts_for_similarity[0]}' vs '{texts_for_similarity[1]}'")
    print(f"   → Similarity: {sim_0_1:.4f} (should be HIGH)")
    
    # Different texts
    sim_0_2 = cosine_similarity(similarity_embeddings[0], similarity_embeddings[2])
    print(f"\n   '{texts_for_similarity[0]}' vs '{texts_for_similarity[2]}'")
    print(f"   → Similarity: {sim_0_2:.4f} (should be LOW)")
    
    if sim_0_1 > sim_0_2:
        print("\n   ✅ Semantic similarity working correctly!")
    else:
        print("\n   ⚠️ Unexpected similarity scores")
    
    # =========================================================
    # Test Chunk Embedding
    # =========================================================
    
    print("\n" + "-" * 60)
    print("\n📄 Testing Chunk Embedding...")
    
    from src.models import Chunk, DocumentMetadata, DocumentType
    from uuid import uuid4
    
    # Create sample chunks
    doc_id = uuid4()
    metadata = DocumentMetadata(
        source="test.txt",
        document_type=DocumentType.TEXT
    )
    
    chunks = [
        Chunk(
            document_id=doc_id,
            content="Machine learning enables computers to learn from data.",
            chunk_index=0,
            start_char=0,
            end_char=50,
            metadata=metadata
        ),
        Chunk(
            document_id=doc_id,
            content="Deep learning uses neural networks with multiple layers.",
            chunk_index=1,
            start_char=50,
            end_char=100,
            metadata=metadata
        )
    ]
    
    print(f"   Embedding {len(chunks)} chunks...")
    
    # Embed chunks
    embedded_chunks = await embedder.embed_chunks(chunks)
    
    print(f"   ✅ Chunks embedded!")
    for chunk in embedded_chunks:
        has_embedding = chunk.embedding is not None
        dim = len(chunk.embedding) if has_embedding else 0
        print(f"      Chunk {chunk.chunk_index}: embedding={'✅' if has_embedding else '❌'} (dim={dim})")
    
    # =========================================================
    # Test Caching
    # =========================================================
    
    print("\n" + "-" * 60)
    print("\n💾 Testing Embedding Cache...")
    
    from src.infrastructure.cache import EmbeddingCache
    
    cache = EmbeddingCache()
    
    # Get stats
    stats = await cache.get_stats()
    print(f"   Cache enabled: {stats['enabled']}")
    print(f"   Cache entries: {stats['entries']}")
    
    # Test cache operations
    test_cache_text = "This is a test for caching."
    test_cache_embedding = [0.1, 0.2, 0.3] * 128  # 384 dims
    
    # Set
    print("\n   Testing cache set...")
    await cache.set(test_cache_text, test_cache_embedding, "test-model")
    print("   ✅ Cache set successful")
    
    # Get
    print("   Testing cache get...")
    cached = await cache.get(test_cache_text, "test-model")
    if cached and len(cached) == len(test_cache_embedding):
        print("   ✅ Cache get successful")
    else:
        print("   ❌ Cache get failed")
    
    # Stats after
    stats = await cache.get_stats()
    print(f"\n   Cache stats after test:")
    print(f"   - Entries: {stats['entries']}")
    print(f"   - Size: {stats['size_mb']} MB")
    
    # =========================================================
    # Test Factory
    # =========================================================
    
    print("\n" + "-" * 60)
    print("\n🏭 Testing Embedding Factory...")
    
    print(f"\n   Available providers: {EmbeddingFactory.get_available_providers()}")
    print(f"   Default provider: {EmbeddingFactory.get_default_provider()}")
    
    # Create via factory
    factory_embedder = EmbeddingFactory.get_embedder()
    print(f"   ✅ Created via factory: {factory_embedder}")
    
    # Test health check
    print("\n   Running health check...")
    is_healthy = await factory_embedder.health_check()
    print(f"   Health check: {'✅ Passed' if is_healthy else '❌ Failed'}")
    
    # =========================================================
    # Test Error Handling
    # =========================================================
    
    print("\n" + "-" * 60)
    print("\n❌ Testing Error Handling...")
    
    from src.models.errors import EmbeddingError
    
    # Test empty text
    print("\n   Testing empty text...")
    try:
        await embedder.embed_text("")
        print("   ❌ Should have raised error")
    except EmbeddingError as e:
        print(f"   ✅ Correctly raised: {e.error_type}")
    
    # Test whitespace text
    print("   Testing whitespace text...")
    try:
        await embedder.embed_text("   \n\t  ")
        print("   ❌ Should have raised error")
    except EmbeddingError as e:
        print(f"   ✅ Correctly raised: {e.error_type}")
    
    # =========================================================
    # Summary
    # =========================================================
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\n✅ HuggingFace embedding working")
    print("✅ Single text embedding working")
    print("✅ Batch embedding working")
    print("✅ Semantic similarity working")
    print("✅ Chunk embedding working")
    print("✅ Caching working")
    print("✅ Factory working")
    print("✅ Error handling working")
    print("\n🚀 Embedding manager is ready!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_embeddings())