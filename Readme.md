# 🚀 RAG Pipeline

A production-ready Retrieval-Augmented Generation (RAG) system built with Python. Ingest documents, store them as vector embeddings, and query them using natural language to get AI-generated answers grounded in your data.

**Built with 100% Free Cloud Services** - No infrastructure costs for development.

---

## 📑 Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Tech Stack](#tech-stack)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Quick Start](#quick-start)
8. [API Reference](#api-reference)
9. [Pipeline Workflow](#pipeline-workflow)
10. [Project Structure](#project-structure)
11. [Testing](#testing)
12. [Troubleshooting](#troubleshooting)

---

## Overview

### What is RAG?

**Retrieval-Augmented Generation (RAG)** enhances Large Language Models by grounding responses in your actual documents. Instead of relying solely on training data, RAG retrieves relevant information and uses it to generate accurate, contextual answers.

### Key Benefits

| Problem | RAG Solution |
|---------|--------------|
| LLMs hallucinate facts | Answers grounded in your documents |
| Training data is outdated | Use your latest documents |
| Generic responses | Domain-specific answers |
| Privacy concerns | Your data stays in your control |

### Use Cases

- 📚 Documentation Q&A
- 🏢 Enterprise Knowledge Base
- 📖 Research Assistant
- 💬 Customer Support Bot
- 📝 Legal/Compliance Search

---

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| Multi-format Support | PDF, TXT, Markdown, HTML, DOCX, URLs |
| Smart Chunking | Recursive and sentence-based strategies |
| Semantic Search | Vector similarity search |
| Reranking | Cross-encoder reranking for better relevance |
| Contextual Answers | LLM answers grounded in documents |
| Source Attribution | Every answer includes references |

### Technical Features

| Feature | Description |
|---------|-------------|
| Async Architecture | High performance with async/await |
| RESTful API | FastAPI with OpenAPI docs |
| Structured Logging | JSON logging for production |
| Type Safety | Full type hints with Pydantic |
| Modular Design | Easy to extend components |

---

## Architecture

### High-Level Overview
```
+------------------------------------------------------------------+
|                        RAG PIPELINE                               |
+------------------------------------------------------------------+
|                                                                   |
|    +-------------+      +-------------+      +-------------+      |
|    |   FastAPI   | ---> |  Services   | ---> |    Core     |      |
|    |   (API)     |      |   Layer     |      | Components  |      |
|    +-------------+      +-------------+      +-------------+      |
|                                                     |             |
|                                                     v             |
|                                            +-------------+        |
|                                            |   Zilliz    |        |
|                                            |   Cloud     |        |
|                                            | (Vector DB) |        |
|                                            +-------------+        |
|                                                                   |
+------------------------------------------------------------------+
```

### Service Layer Components
```
+------------------------------------------------------------------+
|                       SERVICES LAYER                              |
+------------------------------------------------------------------+
|                                                                   |
|  +--------------------+  +--------------------+                   |
|  |  INGESTION SERVICE |  |  RETRIEVAL SERVICE |                   |
|  +--------------------+  +--------------------+                   |
|  | - Load documents   |  | - Embed queries    |                   |
|  | - Chunk text       |  | - Search vectors   |                   |
|  | - Generate embeds  |  | - Rerank results   |                   |
|  | - Store vectors    |  +--------------------+                   |
|  +--------------------+            |                              |
|           |                        v                              |
|           |              +--------------------+                   |
|           +------------> |    RAG SERVICE     |                   |
|                          +--------------------+                   |
|                          | - Orchestrates all |                   |
|                          | - Generates answer |                   |
|                          +--------------------+                   |
|                                                                   |
+------------------------------------------------------------------+
```

### Core Components
```
+------------------------------------------------------------------+
|                      CORE COMPONENTS                              |
+------------------------------------------------------------------+
|                                                                   |
|  +-----------+  +-----------+  +-----------+  +-----------+      |
|  |  LOADERS  |  |  CHUNKING |  | EMBEDDINGS|  |    LLM    |      |
|  +-----------+  +-----------+  +-----------+  +-----------+      |
|  | - PDF     |  | - Recurs. |  | - HuggingF|  | - Groq    |      |
|  | - TXT     |  | - Sentence|  | - 384 dim |  | - Llama   |      |
|  | - HTML    |  |           |  |           |  |           |      |
|  | - URL     |  |           |  |           |  |           |      |
|  +-----------+  +-----------+  +-----------+  +-----------+      |
|                                                                   |
|  +-----------+  +-----------+                                    |
|  |  RERANKER |  |  VECTOR   |                                    |
|  +-----------+  |   STORE   |                                    |
|  | - Cohere  |  +-----------+                                    |
|  |           |  | - Zilliz  |                                    |
|  +-----------+  +-----------+                                    |
|                                                                   |
+------------------------------------------------------------------+
```

---

## Tech Stack

### Cloud Services (All Free Tiers)

| Service | Purpose | Free Tier |
|---------|---------|-----------|
| Zilliz Cloud | Vector database | 1M vectors |
| HuggingFace | Embeddings | 1K req/day |
| Cohere | Reranking | 1K req/month |
| Groq | LLM inference | 14K req/day |

### Python Libraries

| Category | Libraries |
|----------|-----------|
| API | FastAPI, Uvicorn, Pydantic |
| Vector DB | PyMilvus |
| HTTP | httpx, aiofiles |
| LLM | groq, cohere |
| Documents | pypdf, beautifulsoup4 |
| NLP | nltk |
| Logging | structlog |

---

## Installation

### Prerequisites

- Python 3.11+
- pip package manager

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/rag-pipeline.git
cd rag-pipeline
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
.\venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -e ".[dev]"
```

### Step 4: Download NLTK Data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Step 5: Create Directories
```bash
mkdir -p data/uploads data/cache data/test_files
```

---

## Configuration

### Get API Keys (All Free)

1. **Zilliz Cloud**: https://cloud.zilliz.com
   - Create free serverless cluster
   - Copy URI and Token

2. **HuggingFace**: https://huggingface.co/settings/tokens
   - Create account, generate token

3. **Groq**: https://console.groq.com
   - Create account, generate API key

4. **Cohere**: https://dashboard.cohere.com
   - Create account, generate API key

### Create .env File
```bash
cp .env.example .env
```

Edit `.env`:
```env
# Application
APP_NAME=rag-pipeline
APP_ENV=development
DEBUG=true
LOG_LEVEL=INFO

# API
API_HOST=0.0.0.0
API_PORT=8000

# Zilliz Cloud
ZILLIZ_URI=https://your-cluster.zillizcloud.com
ZILLIZ_TOKEN=your-token
ZILLIZ_COLLECTION_NAME=documents

# HuggingFace
HUGGINGFACE_API_KEY=hf_your_key

# Groq
GROQ_API_KEY=gsk_your_key

# Cohere
COHERE_API_KEY=your_key

# Models
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DIMENSION=384
LLM_MODEL=llama-3.1-8b-instant
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1024
RERANKER_MODEL=rerank-english-v3.0
RERANKER_ENABLED=true

# Processing
CHUNK_SIZE=500
CHUNK_OVERLAP=50
DEFAULT_TOP_K=10
RERANK_TOP_K=5

# Storage
CACHE_DIR=./data/cache
CACHE_ENABLED=true
UPLOAD_DIR=./data/uploads
MAX_FILE_SIZE_MB=10
```

---

## Quick Start

### 1. Test Configuration
```bash
python test_config.py
```

### 2. Run Full Pipeline Test
```bash
python test_full_pipeline.py
```

This creates sample documents, ingests them, and tests queries.

### 3. Start the API
```bash
python main.py
```

### 4. Open API Documentation

Visit: http://localhost:8000/docs

### 5. Try a Query
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Python?", "top_k": 3}'
```

### 6. Try Web Demo (Optional)
```bash
python web_demo.py
# Open http://localhost:8001
```

---

## API Reference

### Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | API info |
| GET | /health | Health check |
| GET | /health/detailed | Component health |
| GET | /info | System config |
| POST | /api/ingest | Ingest from path/URL |
| POST | /api/ingest/upload | Upload file |
| POST | /api/query | Query RAG system |
| DELETE | /api/documents/{id} | Delete document |

---

### POST /api/ingest

Ingest a document from file path or URL.

**Request:**
```json
{
  "source": "/path/to/document.pdf",
  "title": "My Document",
  "custom_metadata": {
    "category": "technical"
  }
}
```

**Response:**
```json
{
  "document_id": "uuid-here",
  "source": "/path/to/document.pdf",
  "chunks_created": 15,
  "processing_time_ms": 2340.5,
  "status": "success",
  "message": "Document ingested successfully."
}
```

---

### POST /api/ingest/upload

Upload and ingest a file.

**Request:** multipart/form-data
```bash
curl -X POST http://localhost:8000/api/ingest/upload \
  -F "file=@document.pdf" \
  -F "title=My Document"
```

---

### POST /api/query

Query the RAG system.

**Request:**
```json
{
  "query": "What is machine learning?",
  "top_k": 5,
  "rerank": true,
  "include_sources": true
}
```

**Response:**
```json
{
  "query_id": "uuid-here",
  "query": "What is machine learning?",
  "answer": "Machine learning is a subset of AI...",
  "sources": [
    {
      "chunk_id": "...",
      "content": "...",
      "score": 0.89,
      "source": "ml_guide.pdf"
    }
  ],
  "total_chunks_retrieved": 5,
  "processing_time_ms": 1523.4,
  "model_used": "llama-3.1-8b-instant"
}
```

---

### DELETE /api/documents/{id}

Delete a document and its chunks.

**Response:**
```json
{
  "status": "success",
  "document_id": "uuid-here",
  "chunks_deleted": 15
}
```

---

### GET /health

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

---

### GET /health/detailed

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "vector_store": "healthy",
    "embedder": "healthy",
    "reranker": "healthy",
    "llm": "healthy"
  }
}
```

---

## Pipeline Workflow

### Ingestion Pipeline

When you ingest a document, here's what happens:
```
STEP 1: LOAD DOCUMENT
+------------------------------------------+
| Input: File path or URL                  |
| Action: Detect type, extract text        |
| Output: Document object with metadata    |
+------------------------------------------+
                    |
                    v
STEP 2: CHUNK DOCUMENT
+------------------------------------------+
| Input: Full document text                |
| Action: Split into 500-char chunks       |
| Action: 50-char overlap for context      |
| Output: List of Chunk objects            |
+------------------------------------------+
                    |
                    v
STEP 3: GENERATE EMBEDDINGS
+------------------------------------------+
| Input: List of text chunks               |
| Action: Call HuggingFace API             |
| Action: Convert to 384-dim vectors       |
| Output: Chunks with embeddings           |
+------------------------------------------+
                    |
                    v
STEP 4: STORE IN VECTOR DATABASE
+------------------------------------------+
| Input: Chunks with embeddings            |
| Action: Store in Zilliz Cloud            |
| Action: Index for similarity search      |
| Output: Document ID                      |
+------------------------------------------+
```

### Query Pipeline

When you ask a question, here's what happens:
```
STEP 1: EMBED QUERY
+------------------------------------------+
| Input: User's question                   |
| Action: Convert to 384-dim vector        |
| Output: Query embedding                  |
+------------------------------------------+
                    |
                    v
STEP 2: VECTOR SEARCH
+------------------------------------------+
| Input: Query embedding                   |
| Action: Find similar vectors in Zilliz  |
| Action: Return top 10 candidates         |
| Output: Candidate chunks with scores     |
+------------------------------------------+
                    |
                    v
STEP 3: RERANK (Optional)
+------------------------------------------+
| Input: Query + candidate chunks          |
| Action: Cohere cross-encoder scoring     |
| Action: Keep top 5 most relevant         |
| Output: Reranked chunks                  |
+------------------------------------------+
                    |
                    v
STEP 4: GENERATE ANSWER
+------------------------------------------+
| Input: Query + relevant chunks           |
| Action: Build prompt with context        |
| Action: Call Groq LLM                    |
| Output: Answer with sources              |
+------------------------------------------+
```

### Visual Flow
```
INGESTION:
Document --> [Loader] --> [Chunker] --> [Embedder] --> [Zilliz]

QUERY:
Question --> [Embedder] --> [Zilliz Search] --> [Reranker] --> [LLM] --> Answer
```

---

## Project Structure
```
rag-pipeline/
|
|-- main.py                 # Entry point
|-- pyproject.toml          # Dependencies
|-- .env                    # Configuration
|-- README.md               # Documentation
|
|-- data/
|   |-- uploads/            # Uploaded files
|   |-- cache/              # Embedding cache
|   |-- test_files/         # Test documents
|
|-- src/
|   |-- __init__.py
|   |
|   |-- config/
|   |   |-- __init__.py
|   |   |-- settings.py     # Settings management
|   |   |-- logging_config.py
|   |
|   |-- models/
|   |   |-- __init__.py
|   |   |-- documents.py    # Document, Chunk models
|   |   |-- queries.py      # Request/Response models
|   |   |-- errors.py       # Custom exceptions
|   |
|   |-- core/
|   |   |-- __init__.py
|   |   |
|   |   |-- document_loaders/
|   |   |   |-- __init__.py
|   |   |   |-- base.py
|   |   |   |-- pdf_loader.py
|   |   |   |-- text_loader.py
|   |   |   |-- web_loader.py
|   |   |   |-- factory.py
|   |   |
|   |   |-- chunking/
|   |   |   |-- __init__.py
|   |   |   |-- base.py
|   |   |   |-- recursive_chunker.py
|   |   |   |-- sentence_chunker.py
|   |   |   |-- factory.py
|   |   |
|   |   |-- embeddings/
|   |   |   |-- __init__.py
|   |   |   |-- base.py
|   |   |   |-- huggingface_embedding.py
|   |   |   |-- factory.py
|   |   |
|   |   |-- rerankers/
|   |   |   |-- __init__.py
|   |   |   |-- cohere_reranker.py
|   |   |
|   |   |-- llm/
|   |       |-- __init__.py
|   |       |-- groq_llm.py
|   |
|   |-- infrastructure/
|   |   |-- __init__.py
|   |   |
|   |   |-- vector_store/
|   |   |   |-- __init__.py
|   |   |   |-- zilliz_store.py
|   |   |
|   |   |-- cache/
|   |       |-- __init__.py
|   |       |-- embedding_cache.py
|   |
|   |-- services/
|   |   |-- __init__.py
|   |   |-- ingestion_service.py
|   |   |-- retrieval_service.py
|   |   |-- rag_service.py
|   |
|   |-- api/
|       |-- __init__.py
|       |-- app.py
|       |-- routes/
|           |-- __init__.py
|           |-- ingest.py
|           |-- query.py
|           |-- health.py
|
|-- test_config.py
|-- test_models.py
|-- test_loaders.py
|-- test_chunking.py
|-- test_embeddings.py
|-- test_vector_store.py
|-- test_reranker_llm.py
|-- test_services.py
|-- test_api.py
|-- test_full_pipeline.py
|-- inspect_data.py
|-- web_demo.py
```

---

## Testing

### Test Scripts

| Script | Tests |
|--------|-------|
| test_config.py | Configuration loading |
| test_models.py | Data models |
| test_loaders.py | Document loaders |
| test_chunking.py | Chunking engine |
| test_embeddings.py | Embedding generation |
| test_vector_store.py | Zilliz connection |
| test_reranker_llm.py | Reranker and LLM |
| test_services.py | Service layer |
| test_api.py | API endpoints |
| test_full_pipeline.py | End-to-end test |

### Running Tests
```bash
# Test individual components
python test_config.py
python test_embeddings.py
python test_vector_store.py

# Full pipeline test (recommended)
python test_full_pipeline.py

# API tests (start server first)
python main.py  # Terminal 1
python test_api.py  # Terminal 2
```

### Inspect Data
```bash
# View stored documents
python inspect_data.py

# Interactive search
python inspect_data.py search

# Delete all data
python inspect_data.py delete
```

---

## Troubleshooting

### Connection Errors

**Problem:** Cannot connect to Zilliz Cloud

**Solution:**
1. Check ZILLIZ_URI format: `https://xxx.api.region.zillizcloud.com`
2. Verify ZILLIZ_TOKEN is correct
3. Ensure cluster is running in Zilliz console

---

**Problem:** HuggingFace API error 401

**Solution:**
1. Check HUGGINGFACE_API_KEY starts with `hf_`
2. Verify token has read access
3. Check token is not expired

---

**Problem:** Rate limit exceeded

**Solution:**
1. HuggingFace: 1,000 requests/day - wait until reset
2. Cohere: 1,000 requests/month - upgrade or wait
3. Groq: 14,400 requests/day - wait until reset

---

### No Results Found

**Problem:** Query returns no chunks

**Solution:**
1. Run `python test_full_pipeline.py` to ingest sample docs
2. Check `python inspect_data.py` for stored data
3. Verify documents were chunked and embedded

---

### Import Errors

**Problem:** ModuleNotFoundError

**Solution:**
1. Activate virtual environment: `source venv/bin/activate`
2. Install package: `pip install -e ".[dev]"`
3. Check you're in project root directory

---

### Slow Performance

**Problem:** Queries are slow

**Solution:**
1. First query loads models - subsequent queries faster
2. Enable caching: `CACHE_ENABLED=true`
3. Reduce `top_k` parameter
4. Disable reranking for faster (less accurate) results

---

## Common Commands
```bash
# Start API server
python main.py

# Run full test
python test_full_pipeline.py

# View stored data
python inspect_data.py

# Interactive search
python inspect_data.py search

# Web demo
python web_demo.py

# Check health
curl http://localhost:8000/health
```

---

## License

MIT License - see LICENSE file for details.

---

## Support

- Open an issue for bugs
- Submit PRs for improvements
- Star the repo if helpful!

---

Built with ❤️ using Python, FastAPI, and free cloud services.