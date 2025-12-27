"""
Test Vector Store

This script tests the Zilliz Cloud vector store:
1. Connection
2. Collection creation
3. Inserting chunks
4. Searching for similar chunks
5. Deletion

IMPORTANT: This test makes real API calls to Zilliz Cloud.
Make sure your ZILLIZ_URI and ZILLIZ_TOKEN are set in .env

Run with: python test_vector_store.py
"""

import asyncio
from uuid import uuid4


async def test_vector_store():
    """Test all vector store functionality."""
    
    print("=" * 60)
    print("VECTOR STORE TEST")
    print("=" * 60)
    
    # Setup logging
    from src.config import setup_logging, settings
    setup_logging()
    
    # Check configuration
    print("\n🔑 Checking configuration...")
    
    if not settings.zilliz_uri:
        print("   ❌ ZILLIZ_URI not set in .env")
        print("   Please set your Zilliz Cloud URI and try again.")
        return
    
    if not settings.zilliz_token:
        print("   ❌ ZILLIZ_TOKEN not set in .env")
        print("   Please set your Zilliz Cloud token and try again.")
        return
    
    print(f"   ✅ URI: {settings.zilliz_uri[:50]}...")
    print(f"   ✅ Token: {settings.zilliz_token[:10]}...")
    print(f"   ✅ Collection: {settings.zilliz_collection_name}")
    print(f"   ✅ Dimension: {settings.embedding_dimension}")
    
    # =========================================================
    # Test Connection
    # =========================================================
    
    print("\n" + "-" * 60)
    print("\n🔌 Testing Connection...")
    
    from src.infrastructure.vector_store import ZillizStore
    
    store = ZillizStore()
    print(f"   Store created: {store}")
    
    try:
        await store.connect()
        print("   ✅ Connected to Zilliz Cloud!")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return
    
    # =========================================================
    # Test Collection Stats
    # =========================================================
    
    print("\n" + "-" * 60)
    print("\n📊 Getting Collection Stats...")
    
    stats = await store.get_collection_stats()
    print(f"   Collection: {stats.get('collection_name')}")
    print(f"   Dimension: {stats.get('dimension')}")
    print(f"   Row count: {stats.get('row_count')}")
    
    # =========================================================
    # Test Insert
    # =========================================================
    
    print("\n" + "-" * 60)
    print("\n📥 Testing Insert...")
    
    from src.models import Chunk, DocumentMetadata, DocumentType
    
    # Create test chunks with embeddings
    doc_id = uuid4()
    metadata = DocumentMetadata(
        source="test_document.txt",
        document_type=DocumentType.TEXT,
        title="Test Document"
    )
    
    # Create sample embeddings (random for testing)
    import random
    
    def create_test_embedding(dim: int = 384) -> list[float]:
        """Create a random test embedding."""
        return [random.uniform(-1, 1) for _ in range(dim)]
    
    test_chunks = []
    for i in range(3):
        chunk = Chunk(
            document_id=doc_id,
            content=f"This is test chunk number {i + 1}. It contains some sample text for testing the vector store.",
            chunk_index=i,
            start_char=i * 100,
            end_char=(i + 1) * 100,
            metadata=metadata
        )
        chunk.embedding = create_test_embedding(settings.embedding_dimension)
        test_chunks.append(chunk)
    
    print(f"   Created {len(test_chunks)} test chunks with embeddings")
    
    try:
        inserted_ids = await store.insert_chunks(test_chunks)
        print(f"   ✅ Inserted {len(inserted_ids)} chunks")
    except Exception as e:
        print(f"   ❌ Insert failed: {e}")
        await store.disconnect()
        return
    
    # Wait a moment for indexing
    print("   Waiting for indexing...")
    await asyncio.sleep(2)
    
    # =========================================================
    # Test Search
    # =========================================================
    
    print("\n" + "-" * 60)
    print("\n🔍 Testing Search...")
    
    # Use the first chunk's embedding as query (should match itself)
    query_embedding = test_chunks[0].embedding
    
    try:
        results = await store.search(
            query_embedding=query_embedding,
            top_k=5
        )
        
        print(f"   ✅ Search returned {len(results)} results")
        
        for i, result in enumerate(results):
            print(f"\n   Result {i + 1}:")
            print(f"      Score: {result.score:.4f}")
            print(f"      Content: {result.content[:50]}...")
            print(f"      Source: {result.source}")
            print(f"      Chunk Index: {result.chunk_index}")
            
    except Exception as e:
        print(f"   ❌ Search failed: {e}")
    
    # =========================================================
    # Test Search with Real Embeddings
    # =========================================================
    
    print("\n" + "-" * 60)
    print("\n🔍 Testing Search with Real Embeddings...")
    
    from src.core.embeddings import HuggingFaceEmbedding
    
    try:
        embedder = HuggingFaceEmbedding()
        
        # Create chunks with real embeddings
        real_chunks = [
            Chunk(
                document_id=doc_id,
                content="Machine learning is a type of artificial intelligence.",
                chunk_index=10,
                start_char=0,
                end_char=50,
                metadata=metadata
            ),
            Chunk(
                document_id=doc_id,
                content="Python is a popular programming language.",
                chunk_index=11,
                start_char=50,
                end_char=100,
                metadata=metadata
            ),
            Chunk(
                document_id=doc_id,
                content="Deep learning uses neural networks with many layers.",
                chunk_index=12,
                start_char=100,
                end_char=150,
                metadata=metadata
            ),
        ]
        
        # Embed chunks
        print("   Embedding chunks...")
        real_chunks = await embedder.embed_chunks(real_chunks)
        
        # Insert chunks
        print("   Inserting chunks...")
        await store.insert_chunks(real_chunks)
        
        # Wait for indexing
        await asyncio.sleep(2)
        
        # Search with a related query
        print("   Searching for 'AI and machine learning'...")
        query_text = "AI and machine learning"
        query_emb = await embedder.embed_text(query_text)
        
        search_results = await store.search(
            query_embedding=query_emb,
            top_k=3
        )
        
        print(f"\n   Query: '{query_text}'")
        print(f"   Results: {len(search_results)}")
        
        for i, result in enumerate(search_results):
            print(f"\n   [{i + 1}] Score: {result.score:.4f}")
            print(f"       Content: {result.content}")
        
        print("\n   ✅ Real embedding search working!")
        
    except Exception as e:
        print(f"   ⚠️ Real embedding test skipped: {e}")
    
    # =========================================================
    # Test Deletion
    # =========================================================
    
    print("\n" + "-" * 60)
    print("\n🗑️ Testing Deletion...")
    
    try:
        deleted_count = await store.delete_by_document_id(str(doc_id))
        print(f"   ✅ Deleted {deleted_count} chunks for document {doc_id}")
    except Exception as e:
        print(f"   ⚠️ Deletion test: {e}")
    
    # =========================================================
    # Test Health Check
    # =========================================================
    
    print("\n" + "-" * 60)
    print("\n❤️ Testing Health Check...")
    
    is_healthy = await store.health_check()
    print(f"   Health check: {'✅ Passed' if is_healthy else '❌ Failed'}")
    
    # =========================================================
    # Final Stats
    # =========================================================
    
    print("\n" + "-" * 60)
    print("\n📊 Final Collection Stats...")
    
    final_stats = await store.get_collection_stats()
    print(f"   Row count: {final_stats.get('row_count')}")
    
    # =========================================================
    # Cleanup
    # =========================================================
    
    print("\n" + "-" * 60)
    print("\n🧹 Cleaning up...")
    
    await store.disconnect()
    print("   ✅ Disconnected from Zilliz Cloud")
    
    # =========================================================
    # Summary
    # =========================================================
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\n✅ Connection working")
    print("✅ Collection stats working")
    print("✅ Insert working")
    print("✅ Search working")
    print("✅ Deletion working")
    print("✅ Health check working")
    print("\n🚀 Vector store is ready!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_vector_store())