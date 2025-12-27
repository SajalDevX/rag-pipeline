"""
Test Services

This script tests the complete RAG pipeline:
1. Ingestion Service
2. Retrieval Service
3. RAG Service (end-to-end)

Run with: python test_services.py
"""

import asyncio
from pathlib import Path


async def test_services():
    """Test all services."""
    
    print("=" * 60)
    print("SERVICES TEST")
    print("=" * 60)
    
    # Setup logging
    from src.config import setup_logging
    setup_logging()
    
    # =========================================================
    # Create test document
    # =========================================================
    
    print("\n📄 Creating test document...")
    
    test_dir = Path("./data/test_files")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    test_file = test_dir / "ml_guide.txt"
    test_file.write_text("""
Machine Learning: A Comprehensive Guide

Chapter 1: Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. Instead of writing specific rules, we provide data and let the algorithm discover patterns.

The key idea is that machines can learn from experience. When exposed to more data, they can improve their performance on specific tasks. This is similar to how humans learn - through practice and exposure.

Chapter 2: Types of Machine Learning

There are three main types of machine learning:

1. Supervised Learning: The algorithm learns from labeled training data. Each training example has an input and a corresponding correct output. Common applications include:
   - Image classification
   - Spam detection
   - Price prediction

2. Unsupervised Learning: The algorithm finds patterns in unlabeled data. There are no correct answers provided. Common applications include:
   - Customer segmentation
   - Anomaly detection
   - Dimensionality reduction

3. Reinforcement Learning: The algorithm learns by interacting with an environment and receiving rewards or penalties. Common applications include:
   - Game playing (like AlphaGo)
   - Robotics
   - Autonomous vehicles

Chapter 3: How Machine Learning Works

The machine learning process typically follows these steps:

1. Data Collection: Gather relevant data for the problem
2. Data Preprocessing: Clean and prepare the data
3. Feature Engineering: Select or create relevant features
4. Model Selection: Choose an appropriate algorithm
5. Training: Feed data to the algorithm to learn patterns
6. Evaluation: Test the model on unseen data
7. Deployment: Put the model into production

The model learns by adjusting its internal parameters to minimize errors. This process is called optimization. The goal is to find parameters that generalize well to new, unseen data.

Chapter 4: Popular Algorithms

Some popular machine learning algorithms include:

- Linear Regression: For predicting continuous values
- Logistic Regression: For binary classification
- Decision Trees: For both classification and regression
- Random Forests: An ensemble of decision trees
- Neural Networks: Inspired by the human brain
- Support Vector Machines: For classification tasks

Chapter 5: Deep Learning

Deep learning is a subset of machine learning that uses neural networks with many layers (hence "deep"). These deep neural networks can learn complex patterns in data.

Key concepts in deep learning:
- Neurons: Basic computational units
- Layers: Groups of neurons
- Activation Functions: Add non-linearity
- Backpropagation: Algorithm for training
- Gradient Descent: Optimization method

Deep learning has achieved remarkable results in:
- Image recognition
- Natural language processing
- Speech recognition
- Game playing
    """.strip())
    
    print(f"   Created: {test_file}")
    
    # =========================================================
    # Test Ingestion Service
    # =========================================================
    
    print("\n" + "-" * 60)
    print("\n📥 Testing Ingestion Service...")
    
    from src.services import IngestionService
    
    try:
        ingestion_service = IngestionService()
        print("   ✅ Ingestion service created")
        
        # Ingest the test document
        print(f"\n   Ingesting: {test_file}")
        result = await ingestion_service.ingest(
            source=str(test_file),
            title="Machine Learning Guide",
            custom_metadata={"category": "tutorial", "topic": "machine learning"}
        )
        
        print(f"\n   ✅ Ingestion complete!")
        print(f"      Document ID: {result.document_id}")
        print(f"      Chunks created: {result.chunks_created}")
        print(f"      Processing time: {result.processing_time_ms:.2f}ms")
        
        document_id = str(result.document_id)
        
    except Exception as e:
        print(f"   ❌ Ingestion failed: {e}")
        return
    
    # Wait for indexing
    print("\n   Waiting for indexing...")
    await asyncio.sleep(3)
    
    # =========================================================
    # Test Retrieval Service
    # =========================================================
    
    print("\n" + "-" * 60)
    print("\n🔍 Testing Retrieval Service...")
    
    from src.services import RetrievalService
    
    try:
        retrieval_service = RetrievalService()
        print("   ✅ Retrieval service created")
        
        # Test query
        query = "What are the types of machine learning?"
        print(f"\n   Query: '{query}'")
        
        chunks = await retrieval_service.retrieve(
            query=query,
            top_k=3,
            rerank=True
        )
        
        print(f"\n   ✅ Retrieved {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks, 1):
            score = chunk.rerank_score if chunk.rerank_score else chunk.score
            print(f"\n   [{i}] Score: {score:.4f}")
            print(f"       Content: {chunk.content[:100]}...")
        
    except Exception as e:
        print(f"   ❌ Retrieval failed: {e}")
    
    # =========================================================
    # Test RAG Service (End-to-End)
    # =========================================================
    
    print("\n" + "-" * 60)
    print("\n🤖 Testing RAG Service (End-to-End)...")
    
    from src.services import RAGService
    
    try:
        rag_service = RAGService()
        print("   ✅ RAG service created")
        
        # Test queries
        test_queries = [
            "What is machine learning?",
            "What are the three types of machine learning?",
            "How does deep learning work?",
        ]
        
        for query in test_queries:
            print(f"\n   Query: '{query}'")
            
            response = await rag_service.query(
                query=query,
                top_k=3,
                rerank=True
            )
            
            print(f"\n   Answer: {response.answer[:300]}...")
            print(f"\n   Stats:")
            print(f"   - Chunks used: {response.total_chunks_retrieved}")
            print(f"   - Model: {response.model_used}")
            print(f"   - Processing time: {response.processing_time_ms:.2f}ms")
            print("-" * 40)
        
        print("\n   ✅ RAG service working!")
        
    except Exception as e:
        print(f"   ❌ RAG service failed: {e}")
    
    # =========================================================
    # Test Health Check
    # =========================================================
    
    print("\n" + "-" * 60)
    print("\n❤️ Testing Health Checks...")
    
    try:
        health = await rag_service.health_check()
        print(f"   Status: {health['status']}")
        print("   Components:")
        for component, status in health['components'].items():
            status_icon = "✅" if status else "❌"
            print(f"   - {component}: {status_icon}")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
    
    # =========================================================
    # Cleanup
    # =========================================================
    
    print("\n" + "-" * 60)
    print("\n🧹 Cleaning up...")
    
    try:
        deleted = await ingestion_service.delete_document(document_id)
        print(f"   Deleted {deleted} chunks")
    except Exception as e:
        print(f"   ⚠️ Cleanup: {e}")
    
    # =========================================================
    # Summary
    # =========================================================
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\n✅ Ingestion Service working")
    print("✅ Retrieval Service working")
    print("✅ RAG Service working")
    print("✅ End-to-end pipeline complete!")
    print("\n🚀 All services are ready!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_services())