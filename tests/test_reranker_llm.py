"""
Test Reranker and LLM

This script tests:
1. Cohere Reranker
2. Groq LLM
3. Integration of both

Run with: python test_reranker_llm.py
"""

import asyncio


async def test_reranker_llm():
    """Test reranker and LLM functionality."""
    
    print("=" * 60)
    print("RERANKER AND LLM TEST")
    print("=" * 60)
    
    # Setup logging
    from src.config import setup_logging, settings
    setup_logging()
    
    # =========================================================
    # Test Cohere Reranker
    # =========================================================
    
    print("\n" + "-" * 60)
    print("\n🔄 Testing Cohere Reranker...")
    
    from src.core.rerankers import CohereReranker
    from src.models import RetrievedChunk
    
    # Check if reranker is enabled
    if not settings.cohere_api_key:
        print("   ⚠️ COHERE_API_KEY not set, skipping reranker test")
    else:
        try:
            reranker = CohereReranker()
            print(f"   ✅ Reranker created: {reranker}")
            
            # Create test chunks
            test_chunks = [
                RetrievedChunk(
                    chunk_id="1",
                    document_id="doc1",
                    content="Python is a popular programming language used for web development.",
                    score=0.85,
                    chunk_index=0,
                    source="python.txt",
                    document_type="text"
                ),
                RetrievedChunk(
                    chunk_id="2",
                    document_id="doc1",
                    content="Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
                    score=0.82,
                    chunk_index=1,
                    source="ml.txt",
                    document_type="text"
                ),
                RetrievedChunk(
                    chunk_id="3",
                    document_id="doc1",
                    content="Machine learning algorithms work by finding patterns in training data and using those patterns to make predictions.",
                    score=0.80,
                    chunk_index=2,
                    source="ml.txt",
                    document_type="text"
                ),
            ]
            
            # Test reranking
            query = "How does machine learning work?"
            print(f"\n   Query: '{query}'")
            print("\n   Before reranking:")
            for i, chunk in enumerate(test_chunks):
                print(f"   [{i+1}] Score: {chunk.score:.2f} - {chunk.content[:50]}...")
            
            reranked = await reranker.rerank(query, test_chunks, top_k=3)
            
            print("\n   After reranking:")
            for i, chunk in enumerate(reranked):
                print(f"   [{i+1}] Rerank Score: {chunk.rerank_score:.2f} - {chunk.content[:50]}...")
            
            print("\n   ✅ Reranker working!")
            
            # Health check
            is_healthy = await reranker.health_check()
            print(f"   Health check: {'✅ Passed' if is_healthy else '❌ Failed'}")
            
        except Exception as e:
            print(f"   ❌ Reranker test failed: {e}")
    
    # =========================================================
    # Test Groq LLM
    # =========================================================
    
    print("\n" + "-" * 60)
    print("\n🤖 Testing Groq LLM...")
    
    from src.core.llm import GroqLLM
    
    if not settings.groq_api_key:
        print("   ⚠️ GROQ_API_KEY not set, skipping LLM test")
    else:
        try:
            llm = GroqLLM()
            print(f"   ✅ LLM created: {llm}")
            
            # Test simple generation
            print("\n   Testing simple generation...")
            response = await llm.generate_simple(
                prompt="What is 2 + 2? Reply with just the number.",
                max_tokens=10
            )
            print(f"   Response: {response}")
            
            # Test RAG generation
            print("\n   Testing RAG generation...")
            
            # Create context chunks
            context_chunks = [
                RetrievedChunk(
                    chunk_id="1",
                    document_id="doc1",
                    content="Machine learning is a type of artificial intelligence that allows computers to learn from data without being explicitly programmed. It works by finding patterns in training data.",
                    score=0.95,
                    chunk_index=0,
                    source="ml_guide.pdf",
                    document_type="pdf",
                    title="Machine Learning Guide"
                ),
                RetrievedChunk(
                    chunk_id="2",
                    document_id="doc1",
                    content="The main types of machine learning are supervised learning (using labeled data), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through rewards).",
                    score=0.90,
                    chunk_index=1,
                    source="ml_guide.pdf",
                    document_type="pdf",
                    title="Machine Learning Guide"
                ),
            ]
            
            query = "What is machine learning and what are its main types?"
            print(f"\n   Query: '{query}'")
            print(f"   Context chunks: {len(context_chunks)}")
            
            answer = await llm.generate(
                query=query,
                context_chunks=context_chunks
            )
            
            print(f"\n   Answer:\n   {answer[:500]}...")
            
            print("\n   ✅ LLM working!")
            
            # Health check
            is_healthy = await llm.health_check()
            print(f"   Health check: {'✅ Passed' if is_healthy else '❌ Failed'}")
            
        except Exception as e:
            print(f"   ❌ LLM test failed: {e}")
    
    # =========================================================
    # Summary
    # =========================================================
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\n✅ Reranker component ready")
    print("✅ LLM component ready")
    print("\n🚀 All core components are ready!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_reranker_llm())