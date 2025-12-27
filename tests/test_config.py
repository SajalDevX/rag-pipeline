"""
Test Configuration Loading

This script verifies that:
1. All settings load correctly from .env
2. Logging is working
3. All required API keys are present

Run with: python test_config.py
"""

import sys


def test_configuration():
    """Test that all configuration loads correctly."""
    
    print("=" * 60)
    print("CONFIGURATION TEST")
    print("=" * 60)
    
    # Step 1: Test settings import
    print("\n📋 Step 1: Loading settings...")
    try:
        from src.config import settings
        print("   ✅ Settings loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load settings: {e}")
        sys.exit(1)
    
    # Step 2: Test logging setup
    print("\n📋 Step 2: Setting up logging...")
    try:
        from src.config import setup_logging, get_logger
        setup_logging()
        logger = get_logger(__name__)
        print("   ✅ Logging configured successfully")
    except Exception as e:
        print(f"   ❌ Failed to setup logging: {e}")
        sys.exit(1)
    
    # Step 3: Display all settings
    print("\n" + "=" * 60)
    print("CURRENT CONFIGURATION")
    print("=" * 60)
    
    # Application Settings
    print("\n🔧 Application Settings:")
    print(f"   App Name: {settings.app_name}")
    print(f"   Environment: {settings.app_env}")
    print(f"   Debug Mode: {settings.debug}")
    print(f"   Log Level: {settings.log_level}")
    
    # API Settings
    print("\n🌐 API Settings:")
    print(f"   Host: {settings.api_host}")
    print(f"   Port: {settings.api_port}")
    
    # Zilliz Settings
    print("\n🗄️ Zilliz Cloud Settings:")
    print(f"   URI: {settings.zilliz_uri[:50]}..." if len(settings.zilliz_uri) > 50 else f"   URI: {settings.zilliz_uri}")
    print(f"   Token: {'*' * 10}...{settings.zilliz_token[-4:]}" if len(settings.zilliz_token) > 4 else "   Token: [SET]")
    print(f"   Collection: {settings.zilliz_collection_name}")
    
    # HuggingFace Settings
    print("\n🤗 HuggingFace Settings:")
    print(f"   API Key: {'*' * 10}...{settings.huggingface_api_key[-4:]}" if len(settings.huggingface_api_key) > 4 else "   API Key: [SET]")
    print(f"   Model: {settings.embedding_model}")
    print(f"   Dimension: {settings.embedding_dimension}")
    print(f"   API URL: {settings.huggingface_api_url}")
    
    # Groq Settings
    print("\n⚡ Groq Settings:")
    print(f"   API Key: {'*' * 10}...{settings.groq_api_key[-4:]}" if len(settings.groq_api_key) > 4 else "   API Key: [SET]")
    print(f"   Model: {settings.llm_model}")
    print(f"   Temperature: {settings.llm_temperature}")
    print(f"   Max Tokens: {settings.llm_max_tokens}")
    
    # Cohere Settings
    print("\n🔄 Cohere Settings:")
    print(f"   API Key: {'*' * 10}...{settings.cohere_api_key[-4:]}" if len(settings.cohere_api_key) > 4 else "   API Key: [SET]")
    print(f"   Model: {settings.reranker_model}")
    print(f"   Enabled: {settings.reranker_enabled}")
    
    # Chunking Settings
    print("\n✂️ Chunking Settings:")
    print(f"   Chunk Size: {settings.chunk_size} characters")
    print(f"   Chunk Overlap: {settings.chunk_overlap} characters")
    
    # Retrieval Settings
    print("\n🔍 Retrieval Settings:")
    print(f"   Default Top K: {settings.default_top_k}")
    print(f"   Rerank Top K: {settings.rerank_top_k}")
    print(f"   Similarity Threshold: {settings.similarity_threshold}")
    
    # Cache Settings
    print("\n💾 Cache Settings:")
    print(f"   Cache Directory: {settings.cache_dir}")
    print(f"   Cache Enabled: {settings.cache_enabled}")
    
    # File Settings
    print("\n📁 File Settings:")
    print(f"   Upload Directory: {settings.upload_dir}")
    print(f"   Max File Size: {settings.max_file_size_mb} MB ({settings.max_file_size_bytes} bytes)")
    
    # Computed Properties
    print("\n📊 Computed Properties:")
    print(f"   Is Development: {settings.is_development}")
    print(f"   Is Production: {settings.is_production}")
    
    # Step 4: Test logging
    print("\n" + "=" * 60)
    print("LOGGING TEST")
    print("=" * 60)
    print("\nSending test log messages:\n")
    
    logger.debug("This is a DEBUG message", test=True)
    logger.info("This is an INFO message", user="test_user", action="config_test")
    logger.warning("This is a WARNING message", warning_code=123)
    
    # Step 5: Validate settings
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    
    errors = []
    
    # Check Zilliz URI format
    if not settings.zilliz_uri.startswith("https://"):
        errors.append("Zilliz URI should start with https://")
    
    # Check HuggingFace key format
    if not settings.huggingface_api_key.startswith("hf_"):
        errors.append("HuggingFace API key should start with 'hf_'")
    
    # Check Groq key format
    if not settings.groq_api_key.startswith("gsk_"):
        errors.append("Groq API key should start with 'gsk_'")
    
    # Check chunk overlap is less than chunk size
    if settings.chunk_overlap >= settings.chunk_size:
        errors.append(f"Chunk overlap ({settings.chunk_overlap}) must be less than chunk size ({settings.chunk_size})")
    
    # Check rerank_top_k is less than or equal to default_top_k
    if settings.rerank_top_k > settings.default_top_k:
        errors.append(f"Rerank top_k ({settings.rerank_top_k}) should not exceed default top_k ({settings.default_top_k})")
    
    if errors:
        print("\n⚠️ Warnings:")
        for error in errors:
            print(f"   - {error}")
    else:
        print("\n✅ All validations passed!")
    
    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\n✅ Configuration loaded successfully!")
    print("✅ Logging is working!")
    print("\n🚀 You're ready to proceed to the next step!")
    print("=" * 60)


if __name__ == "__main__":
    test_configuration()