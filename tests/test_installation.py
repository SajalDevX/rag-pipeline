"""Test that all dependencies are installed correctly."""

def test_imports():
    print("Testing imports...")
    
    # API Framework
    import fastapi
    print(f"✅ FastAPI {fastapi.__version__}")
    
    import uvicorn
    print(f"✅ Uvicorn installed")
    
    # Vector Store
    from pymilvus import MilvusClient
    print(f"✅ PyMilvus installed")
    
    # Cloud APIs
    import httpx
    print(f"✅ HTTPX {httpx.__version__}")
    
    import groq
    print(f"✅ Groq installed")
    
    import cohere
    print(f"✅ Cohere installed")
    
    # Document Processing
    import pypdf
    print(f"✅ PyPDF {pypdf.__version__}")
    
    import docx
    print(f"✅ python-docx installed")
    
    from bs4 import BeautifulSoup
    print(f"✅ BeautifulSoup installed")
    
    # Text Processing
    import nltk
    print(f"✅ NLTK {nltk.__version__}")
    
    # Configuration
    import pydantic
    print(f"✅ Pydantic {pydantic.__version__}")
    
    from pydantic_settings import BaseSettings
    print(f"✅ Pydantic Settings installed")
    
    # Logging
    import structlog
    print(f"✅ Structlog installed")
    
    # Utilities
    import tenacity
    print(f"✅ Tenacity installed")
    
    import numpy
    print(f"✅ NumPy {numpy.__version__}")
    
    print("\n" + "="*50)
    print("🎉 All dependencies installed successfully!")
    print("="*50)


if __name__ == "__main__":
    test_imports()