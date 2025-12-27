"""
Embedding Factory

This module provides a factory class for creating embedding managers.
Currently supports HuggingFace Inference API (free).

Usage:
    from src.core.embeddings import EmbeddingFactory
    
    # Get default embedder (HuggingFace)
    embedder = EmbeddingFactory.get_embedder()
    
    # Embed text
    vector = await embedder.embed_text("Hello world")
"""

from enum import Enum

from src.config import get_logger
from src.models.errors import EmbeddingError

from .base import BaseEmbeddingManager
from .huggingface_embedding import HuggingFaceEmbedding

logger = get_logger(__name__)


class EmbeddingProvider(str, Enum):
    """Available embedding providers."""
    HUGGINGFACE = "huggingface"


class EmbeddingFactory:
    """
    Factory class for creating embedding managers.
    
    Provides a unified interface for creating embedders
    regardless of the underlying provider.
    
    Usage:
        # Get default embedder
        embedder = EmbeddingFactory.get_embedder()
        
        # Get specific provider
        embedder = EmbeddingFactory.get_embedder(provider="huggingface")
        
        # With custom settings
        embedder = EmbeddingFactory.get_embedder(
            provider="huggingface",
            model_name="BAAI/bge-base-en-v1.5"
        )
    """
    
    # Map providers to classes
    PROVIDERS = {
        EmbeddingProvider.HUGGINGFACE: HuggingFaceEmbedding,
    }
    
    # Default provider
    DEFAULT_PROVIDER = EmbeddingProvider.HUGGINGFACE
    
    @classmethod
    def get_embedder(
        cls,
        provider: str | EmbeddingProvider | None = None,
        **kwargs
    ) -> BaseEmbeddingManager:
        """
        Get an embedding manager instance.
        
        Args:
            provider: Embedding provider name
            **kwargs: Additional arguments for the embedder
        
        Returns:
            BaseEmbeddingManager: Configured embedder instance
        
        Raises:
            EmbeddingError: If provider is unknown
        
        Example:
            embedder = EmbeddingFactory.get_embedder()
            embedder = EmbeddingFactory.get_embedder(
                provider="huggingface",
                model_name="BAAI/bge-base-en-v1.5"
            )
        """
        # Use default if not specified
        if provider is None:
            provider = cls.DEFAULT_PROVIDER
        
        # Convert string to enum
        if isinstance(provider, str):
            try:
                provider = EmbeddingProvider(provider.lower())
            except ValueError:
                raise EmbeddingError(
                    message=f"Unknown embedding provider: {provider}. "
                           f"Available: {cls.get_available_providers()}"
                )
        
        # Get embedder class
        embedder_class = cls.PROVIDERS.get(provider)
        
        if embedder_class is None:
            raise EmbeddingError(
                message=f"No embedder for provider: {provider}"
            )
        
        logger.info(
            "Creating embedder",
            provider=provider.value,
            embedder_class=embedder_class.__name__
        )
        
        return embedder_class(**kwargs)
    
    @classmethod
    def get_available_providers(cls) -> list[str]:
        """
        Get list of available embedding providers.
        
        Returns:
            list[str]: Provider names
        """
        return [p.value for p in EmbeddingProvider]
    
    @classmethod
    def get_default_provider(cls) -> str:
        """
        Get the default embedding provider.
        
        Returns:
            str: Default provider name
        """
        return cls.DEFAULT_PROVIDER.value