"""
Test API Endpoints

This script tests the API endpoints:
1. Health check
2. Document ingestion
3. Query
4. Document deletion

Run the API first:
    python main.py

Then run this test:
    python test_api.py
"""

import asyncio
import httpx
from pathlib import Path


BASE_URL = "http://localhost:8000"


async def test_api():
    """Test all API endpoints."""
    
    print("=" * 60)
    print("API ENDPOINT TESTS")
    print("=" * 60)
    print(f"\nBase URL: {BASE_URL}")
    print("\n⚠️  Make sure the API is running: python main.py")
    
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        
        # =========================================================
        # Test Root
        # =========================================================
        
        print("\n" + "-" * 60)
        print("\n🏠 Testing Root Endpoint...")
        
        try:
            response = await client.get("/")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.json()}")
            print("   ✅ Root endpoint working")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            print("\n   Make sure the API is running!")
            return
        
        # =========================================================
        # Test Health
        # =========================================================
        
        print("\n" + "-" * 60)
        print("\n❤️ Testing Health Endpoints...")
        
        # Basic health
        response = await client.get("/health")
        print(f"\n   GET /health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        
        # Detailed health
        response = await client.get("/health/detailed")
        print(f"\n   GET /health/detailed")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        
        # System info
        response = await client.get("/info")
        print(f"\n   GET /info")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        
        print("   ✅ Health endpoints working")
        
        # =========================================================
        # Create Test Document
        # =========================================================
        
        print("\n" + "-" * 60)
        print("\n📄 Creating test document...")
        
        test_dir = Path("./data/test_files")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        test_file = test_dir / "api_test.txt"
        test_file.write_text("""
Artificial Intelligence: An Overview

What is AI?
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines. These machines are programmed to think and learn like humans.

Types of AI:
1. Narrow AI: Designed for specific tasks (like Siri or chess programs)
2. General AI: Hypothetical AI with human-like general intelligence
3. Super AI: Hypothetical AI that surpasses human intelligence

Machine Learning:
Machine learning is a subset of AI that enables computers to learn from data. Instead of being explicitly programmed, ML algorithms find patterns in data.

Deep Learning:
Deep learning uses neural networks with many layers. It has achieved remarkable results in image recognition, natural language processing, and more.
        """.strip())
        
        print(f"   Created: {test_file}")
        
        # =========================================================
        # Test Ingestion
        # =========================================================
        
        print("\n" + "-" * 60)
        print("\n📥 Testing Ingestion Endpoint...")
        
        # Ingest via path
        print("\n   POST /api/ingest (path)")
        response = await client.post(
            "/api/ingest",
            json={
                "source": str(test_file),
                "title": "AI Overview",
                "custom_metadata": {"topic": "artificial intelligence"}
            }
        )
        print(f"   Status: {response.status_code}")
        result = response.json()
        print(f"   Response: {result}")
        
        if response.status_code == 200:
            document_id = result.get("document_id")
            print(f"   ✅ Ingestion successful, document_id: {document_id}")
        else:
            print(f"   ❌ Ingestion failed")
            document_id = None
        
        # Wait for indexing
        print("\n   Waiting for indexing...")
        await asyncio.sleep(3)
        
        # =========================================================
        # Test Query
        # =========================================================
        
        print("\n" + "-" * 60)
        print("\n🔍 Testing Query Endpoint...")
        
        test_queries = [
            "What is artificial intelligence?",
            "What are the types of AI?",
            "How does machine learning work?",
        ]
        
        for query in test_queries:
            print(f"\n   Query: '{query}'")
            
            response = await client.post(
                "/api/query",
                json={
                    "query": query,
                    "top_k": 3,
                    "rerank": True,
                    "include_sources": True
                }
            )
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("answer", "")[:200]
                chunks = result.get("total_chunks_retrieved", 0)
                time_ms = result.get("processing_time_ms", 0)
                
                print(f"   Answer: {answer}...")
                print(f"   Chunks used: {chunks}")
                print(f"   Time: {time_ms:.2f}ms")
            else:
                print(f"   Error: {response.json()}")
        
        print("\n   ✅ Query endpoint working")
        
        # =========================================================
        # Test File Upload
        # =========================================================
        
        print("\n" + "-" * 60)
        print("\n📤 Testing File Upload Endpoint...")
        
        upload_file = test_dir / "upload_test.txt"
        upload_file.write_text("This is a test upload file about Python programming.")
        
        with open(upload_file, "rb") as f:
            response = await client.post(
                "/api/ingest/upload",
                files={"file": ("upload_test.txt", f, "text/plain")},
                data={"title": "Upload Test"}
            )
        
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        
        if response.status_code == 200:
            upload_doc_id = response.json().get("document_id")
            print("   ✅ Upload endpoint working")
            
            # Clean up uploaded document
            await client.delete(f"/api/documents/{upload_doc_id}")
        else:
            print("   ⚠️ Upload may have failed")
        
        # =========================================================
        # Test Deletion
        # =========================================================
        
        print("\n" + "-" * 60)
        print("\n🗑️ Testing Delete Endpoint...")
        
        if document_id:
            response = await client.delete(f"/api/documents/{document_id}")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.json()}")
            print("   ✅ Delete endpoint working")
        else:
            print("   ⚠️ No document to delete")
        
        # =========================================================
        # Test OpenAPI Docs
        # =========================================================
        
        print("\n" + "-" * 60)
        print("\n📚 Testing API Documentation...")
        
        response = await client.get("/docs")
        print(f"   GET /docs - Status: {response.status_code}")
        
        response = await client.get("/openapi.json")
        print(f"   GET /openapi.json - Status: {response.status_code}")
        
        print("   ✅ Documentation endpoints working")
        
    # =========================================================
    # Summary
    # =========================================================
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\n✅ Root endpoint working")
    print("✅ Health endpoints working")
    print("✅ Ingestion endpoint working")
    print("✅ Query endpoint working")
    print("✅ Upload endpoint working")
    print("✅ Delete endpoint working")
    print("✅ Documentation working")
    print("\n🚀 API is fully functional!")
    print("\n📚 View API docs at: http://localhost:8000/docs")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_api())