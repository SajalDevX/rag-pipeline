"""
HuggingFace Inference API Embedding Manager

This module provides embedding generation using HuggingFace's free
Inference API. No local GPU required - all processing happens in the cloud.

Features:
- Free tier: 1,000 requests/day
- Fast cloud-based inference
- Multiple model options
- Automatic retry on failures
- Batch processing support

Models available:
- BAAI/bge-small-en-v1.5 (384 dims) - Recommended, good balance
- BAAI/bge-base-en-v1.5 (768 dims) - Better quality, slower
- sentence-transformers/all-MiniLM-L6-v2 (384 dims) - Fast, good quality
"""

import asyncio
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import get_logger, settings
from src.models.errors import EmbeddingError, ErrorDetail

from .base import BaseEmbeddingManager

logger = get_logger(__name__)


class HuggingFaceEmbedding(BaseEmbeddingManager):
    """
    Embedding manager using HuggingFace Inference API.
    
    Uses the free HuggingFace Inference API to generate embeddings.
    No local GPU or model download required.
    
    Attributes:
        api_key: HuggingFace API token
        model_name: Model to use for embeddings
        dimension: Output vector dimension
        api_url: HuggingFace API endpoint
    
    Usage:
        embedder = HuggingFaceEmbedding()
        
        # Single text
        vector = await embedder.embed_text("Hello world")
        
        # Multiple texts
        vectors = await embedder.embed_texts(["Hello", "World"])
        
        # Embed chunks
        chunks_with_embeddings = await embedder.embed_chunks(chunks)
    """
    
    # Model configurations
    MODEL_DIMENSIONS = {
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
    }
    
    # API configuration
    BASE_URL = "https://router.huggingface.co/hf-inference/models"
    TIMEOUT = 60  # seconds
    MAX_BATCH_SIZE = 100  # HuggingFace limit
    
    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        dimension: int | None = None
    ):
        """
        Initialize the HuggingFace embedding manager.
        
        Args:
            api_key: HuggingFace API token (default from settings)
            model_name: Model to use (default from settings)
            dimension: Vector dimension (auto-detected from model)
        """
        # Get values from settings if not provided
        self.api_key = api_key or settings.huggingface_api_key
        model_name = model_name or settings.embedding_model
        
        # Auto-detect dimension from model
        if dimension is None:
            dimension = self.MODEL_DIMENSIONS.get(model_name)
            if dimension is None:
                # Use setting if model not in our list
                dimension = settings.embedding_dimension
        
        super().__init__(model_name=model_name, dimension=dimension)
        
        # Build API URL
        self.api_url = f"{self.BASE_URL}/{self.model_name}"
        
        # Validate API key
        if not self.api_key:
            raise EmbeddingError(
                message="HuggingFace API key not provided",
                details=[ErrorDetail(
                    field="api_key",
                    message="Set HUGGINGFACE_API_KEY in .env file"
                )]
            )
        
        self.logger.info(
            "HuggingFace embedding manager initialized",
            model=self.model_name,
            dimension=self.dimension,
            api_url=self.api_url
        )
    
    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
        
        Returns:
            list[float]: Embedding vector of size self.dimension
        
        Raises:
            EmbeddingError: If embedding fails
        """
        # Validate text
        text = self._validate_text(text)
        
        # Embed
        embeddings = await self._call_api([text])
        
        if not embeddings or len(embeddings) == 0:
            raise EmbeddingError(
                message="No embedding returned from API"
            )
        
        embedding = embeddings[0]
        
        # Validate result
        self._validate_embedding(embedding)
        
        return embedding
    
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        Handles batching automatically if texts exceed API limits.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            list[list[float]]: List of embedding vectors
        
        Raises:
            EmbeddingError: If embedding fails
        """
        if not texts:
            return []
        
        # Validate all texts
        validated_texts = [self._validate_text(t) for t in texts]
        
        # Process in batches if needed
        if len(validated_texts) <= self.MAX_BATCH_SIZE:
            embeddings = await self._call_api(validated_texts)
        else:
            # Split into batches
            embeddings = []
            for i in range(0, len(validated_texts), self.MAX_BATCH_SIZE):
                batch = validated_texts[i:i + self.MAX_BATCH_SIZE]
                batch_embeddings = await self._call_api(batch)
                embeddings.extend(batch_embeddings)
                
                # Small delay between batches to avoid rate limits
                if i + self.MAX_BATCH_SIZE < len(validated_texts):
                    await asyncio.sleep(0.1)
        
        # Validate all embeddings
        for emb in embeddings:
            self._validate_embedding(emb)
        
        return embeddings
    
    async def _call_api(self, texts: list[str]) -> list[list[float]]:
        """
        Make API call to HuggingFace Inference API.
        
        Args:
            texts: Texts to embed
        
        Returns:
            list[list[float]]: Embedding vectors
        
        Raises:
            EmbeddingError: If API call fails after retries
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Payload format for HuggingFace Inference API
        payload = {
            "inputs": texts,
            "options": {
                "wait_for_model": True  # Wait if model is loading
            }
        }
        
        self.logger.debug(
            "Calling HuggingFace API",
            text_count=len(texts),
            model=self.model_name
        )
        
        # Retry logic
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
                    response = await client.post(
                        self.api_url,
                        headers=headers,
                        json=payload
                    )
                    
                    # Check for errors
                    if response.status_code == 401:
                        raise EmbeddingError(
                            message="Invalid HuggingFace API key",
                            details=[ErrorDetail(
                                field="api_key",
                                message="Check your HUGGINGFACE_API_KEY"
                            )]
                        )
                    
                    if response.status_code == 429:
                        raise EmbeddingError(
                            message="HuggingFace API rate limit exceeded",
                            details=[ErrorDetail(
                                field="rate_limit",
                                message="Free tier: 1,000 requests/day. Try again later."
                            )]
                        )
                    
                    if response.status_code == 503:
                        # Model is loading, retry
                        self.logger.warning(
                            "Model is loading, retrying...",
                            model=self.model_name,
                            attempt=attempt + 1
                        )
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    
                    response.raise_for_status()
                    
                    # Parse response
                    result = response.json()
                    
                    # Handle different response formats
                    embeddings = self._parse_response(result)
                    
                    self.logger.debug(
                        "API call successful",
                        embedding_count=len(embeddings)
                    )
                    
                    return embeddings
                    
            except httpx.TimeoutException:
                self.logger.warning(
                    "HuggingFace API timeout, retrying...",
                    attempt=attempt + 1
                )
                last_error = EmbeddingError(
                    message=f"HuggingFace API timeout after {self.TIMEOUT}s"
                )
                await asyncio.sleep(2 ** attempt)
                continue
            
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 503:
                    # Model loading, retry
                    await asyncio.sleep(2 ** attempt)
                    continue
                
                self.logger.error(
                    "HuggingFace API error",
                    status_code=e.response.status_code,
                    response=e.response.text[:200]
                )
                raise EmbeddingError(
                    message=f"HuggingFace API error: {e.response.status_code}",
                    details=[ErrorDetail(
                        field="api",
                        message=e.response.text[:200]
                    )]
                )
            
            except EmbeddingError:
                raise
            
            except Exception as e:
                self.logger.error(
                    "Unexpected error calling HuggingFace API",
                    error=str(e)
                )
                raise EmbeddingError(
                    message=f"Failed to call HuggingFace API: {str(e)}"
                )
        
        # If we get here, all retries failed
        if last_error:
            raise last_error
        raise EmbeddingError(message="Failed to call HuggingFace API after retries")
    
    def _parse_response(self, result: Any) -> list[list[float]]:
        """
        Parse API response into embedding vectors.
        
        HuggingFace API can return different formats depending on
        the model and input.
        
        Args:
            result: API response JSON
        
        Returns:
            list[list[float]]: Parsed embeddings
        
        Raises:
            EmbeddingError: If response format is unexpected
        """
        # Check for error response
        if isinstance(result, dict) and "error" in result:
            raise EmbeddingError(
                message=f"HuggingFace API error: {result['error']}"
            )
        
        # Format 1: List of embeddings directly
        # [[0.1, 0.2, ...], [0.3, 0.4, ...]]
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list) and len(result[0]) > 0 and isinstance(result[0][0], (int, float)):
                return result
        
        # Format 2: Single embedding (for single input)
        # [0.1, 0.2, ...]
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], (int, float)):
                return [result]
        
        # Format 3: Nested in 'embeddings' key
        # {"embeddings": [[0.1, 0.2, ...], ...]}
        if isinstance(result, dict) and "embeddings" in result:
            return result["embeddings"]
        
        # Unknown format
        self.logger.error(
            "Unexpected API response format",
            result_type=type(result).__name__,
            result_preview=str(result)[:200]
        )
        raise EmbeddingError(
            message=f"Unexpected API response format: {type(result).__name__}"
        )
    
    async def health_check(self) -> bool:
        """
        Check if the HuggingFace API is accessible.
        
        Returns:
            bool: True if API is healthy
        """
        try:
            # Try to embed a simple text
            embedding = await self.embed_text("test")
            return len(embedding) == self.dimension
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return False