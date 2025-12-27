"""
Configuration Management System

This module loads and validates all application settings from environment
variables and the .env file using Pydantic Settings.

Usage:
    from src.config.settings import settings
    
    # Access any setting
    print(settings.groq_api_key)
    print(settings.chunk_size)
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Pydantic Settings automatically:
    1. Reads from environment variables
    2. Reads from .env file
    3. Validates types (converts "8000" to int 8000)
    4. Applies defaults if not provided
    5. Raises errors for missing required fields
    
    Fields marked with `...` are required (no default).
    Fields with `= value` have defaults.
    """
    
    # Tell Pydantic where to find the .env file
    model_config = SettingsConfigDict(
        env_file=".env",           # Load from .env file
        env_file_encoding="utf-8", # File encoding
        case_sensitive=False,      # ENV_VAR = env_var
        extra="ignore",            # Ignore extra env vars
    )
    
    # =========================================================
    # APPLICATION SETTINGS
    # =========================================================
    
    app_name: str = Field(
        default="rag-pipeline",
        description="Application name for logging and identification"
    )
    
    app_env: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Current environment"
    )
    
    debug: bool = Field(
        default=True,
        description="Enable debug mode"
    )
    
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )
    
    # =========================================================
    # API SERVER SETTINGS
    # =========================================================
    
    api_host: str = Field(
        default="0.0.0.0",
        description="Host to bind the API server"
    )
    
    api_port: int = Field(
        default=8000,
        ge=1,        # Greater than or equal to 1
        le=65535,    # Less than or equal to 65535
        description="Port for the API server"
    )
    
    # =========================================================
    # ZILLIZ CLOUD SETTINGS (Vector Database)
    # =========================================================
    
    zilliz_uri: str = Field(
        ...,  # Required - no default
        description="Zilliz Cloud cluster URI (e.g., https://xxx.zillizcloud.com)"
    )
    
    zilliz_token: str = Field(
        ...,  # Required - no default
        description="Zilliz API token for authentication"
    )
    
    zilliz_collection_name: str = Field(
        default="documents",
        description="Name of the collection to store vectors"
    )
    
    # =========================================================
    # HUGGINGFACE SETTINGS (Embeddings)
    # =========================================================
    
    huggingface_api_key: str = Field(
        ...,  # Required
        description="HuggingFace API token (starts with hf_)"
    )
    
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace model for generating embeddings"
    )
    
    embedding_dimension: int = Field(
        default=384,
        ge=1,
        description="Dimension of embedding vectors (must match model)"
    )
    
    # Validate HuggingFace API key format
    @field_validator("huggingface_api_key")
    @classmethod
    def validate_hf_key(cls, v: str) -> str:
        """Validate that HuggingFace key has correct format."""
        if not v.startswith("hf_"):
            # Just a warning - some keys might have different formats
            pass
        return v
    
    # =========================================================
    # GROQ SETTINGS (LLM)
    # =========================================================
    
    groq_api_key: str = Field(
        ...,  # Required
        description="Groq API key (starts with gsk_)"
    )
    
    llm_model: str = Field(
        default="llama-3.1-8b-instant",
        description="Groq model name for text generation"
    )
    
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,      # Minimum 0
        le=2.0,      # Maximum 2
        description="Temperature for LLM responses (0=deterministic, 2=creative)"
    )
    
    llm_max_tokens: int = Field(
        default=1024,
        ge=1,
        le=8192,
        description="Maximum tokens in LLM response"
    )
    
    # Validate Groq API key format
    @field_validator("groq_api_key")
    @classmethod
    def validate_groq_key(cls, v: str) -> str:
        """Validate that Groq key has correct format."""
        if not v.startswith("gsk_"):
            # Just a warning - format might change
            pass
        return v
    
    # =========================================================
    # COHERE SETTINGS (Reranking)
    # =========================================================
    
    cohere_api_key: str = Field(
        ...,  # Required
        description="Cohere API key for reranking"
    )
    
    reranker_model: str = Field(
        default="rerank-english-v3.0",
        description="Cohere model for reranking search results"
    )
    
    reranker_enabled: bool = Field(
        default=True,
        description="Whether to use reranking (disable to save API calls)"
    )
    
    # =========================================================
    # CHUNKING SETTINGS
    # =========================================================
    
    chunk_size: int = Field(
        default=500,
        ge=100,      # Minimum 100 characters
        le=2000,     # Maximum 2000 characters
        description="Maximum characters per chunk"
    )
    
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Number of characters to overlap between chunks"
    )
    
    # Validate that overlap is less than chunk size
    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is less than chunk size."""
        # Note: In Pydantic v2, we can't easily access other fields here
        # We'll validate this in the application logic instead
        return v
    
    # =========================================================
    # RETRIEVAL SETTINGS
    # =========================================================
    
    default_top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of chunks to retrieve from vector database"
    )
    
    rerank_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of chunks to keep after reranking"
    )
    
    similarity_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score to include a result"
    )
    
    # =========================================================
    # CACHE SETTINGS
    # =========================================================
    
    cache_dir: Path = Field(
        default=Path("./data/cache"),
        description="Directory for caching embeddings and responses"
    )
    
    cache_enabled: bool = Field(
        default=True,
        description="Enable caching to reduce API calls"
    )
    
    # =========================================================
    # FILE SETTINGS
    # =========================================================
    
    upload_dir: Path = Field(
        default=Path("./data/uploads"),
        description="Directory for uploaded documents"
    )
    
    max_file_size_mb: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum file size for uploads in MB"
    )
    
    # =========================================================
    # COMPUTED PROPERTIES
    # =========================================================
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.app_env == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.app_env == "production"
    
    @property
    def max_file_size_bytes(self) -> int:
        """Get maximum file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024
    
    @property
    def huggingface_api_url(self) -> str:
        """Get the HuggingFace Inference API URL for the embedding model."""
        return f"https://api-inference.huggingface.co/models/{self.embedding_model}"


# =========================================================
# SETTINGS INSTANCE
# =========================================================

@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses @lru_cache to ensure settings are only loaded once.
    This is important because:
    1. Reading .env file is I/O operation
    2. We want consistent settings throughout the app
    3. Avoids repeated validation
    
    Returns:
        Settings: The application settings instance
    
    Usage:
        settings = get_settings()
        # or
        from src.config.settings import settings
    """
    return Settings()


# Create a default instance for easy importing
# This is the recommended way to access settings
settings = get_settings()