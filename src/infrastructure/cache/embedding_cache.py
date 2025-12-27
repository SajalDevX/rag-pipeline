"""
Embedding Cache

This module provides caching for embeddings to:
1. Reduce API calls (save money and rate limits)
2. Speed up repeated queries
3. Enable offline operation for cached content

Uses file-based caching (no Redis required).
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import aiofiles

from src.config import get_logger, settings

logger = get_logger(__name__)


class EmbeddingCache:
    """
    File-based cache for embeddings.
    
    Stores embeddings as JSON files, keyed by a hash of the text.
    This avoids re-computing embeddings for the same text.
    
    Attributes:
        cache_dir: Directory to store cache files
        enabled: Whether caching is enabled
    
    Usage:
        cache = EmbeddingCache()
        
        # Check cache
        embedding = await cache.get("Hello world")
        
        if embedding is None:
            # Compute embedding
            embedding = await embedder.embed_text("Hello world")
            # Store in cache
            await cache.set("Hello world", embedding)
    """
    
    def __init__(
        self,
        cache_dir: Path | str | None = None,
        enabled: bool | None = None
    ):
        """
        Initialize the embedding cache.
        
        Args:
            cache_dir: Directory for cache files (default from settings)
            enabled: Whether caching is enabled (default from settings)
        """
        self.cache_dir = Path(cache_dir or settings.cache_dir) / "embeddings"
        self.enabled = enabled if enabled is not None else settings.cache_enabled
        
        # Create cache directory
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger(self.__class__.__name__)
        
        self.logger.info(
            "Embedding cache initialized",
            cache_dir=str(self.cache_dir),
            enabled=self.enabled
        )
    
    def _get_cache_key(self, text: str, model: str = "") -> str:
        """
        Generate a cache key for text.
        
        Uses SHA256 hash of text + model name.
        
        Args:
            text: Text to hash
            model: Model name (different models = different embeddings)
        
        Returns:
            str: Cache key (hash)
        """
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """
        Get the file path for a cache key.
        
        Uses first 2 chars as subdirectory for better file distribution.
        
        Args:
            cache_key: Cache key (hash)
        
        Returns:
            Path: File path for cache entry
        """
        # Use first 2 chars as subdirectory
        subdir = cache_key[:2]
        return self.cache_dir / subdir / f"{cache_key}.json"
    
    async def get(
        self,
        text: str,
        model: str = ""
    ) -> list[float] | None:
        """
        Get embedding from cache.
        
        Args:
            text: Text to look up
            model: Model name
        
        Returns:
            list[float] | None: Cached embedding or None if not found
        """
        if not self.enabled:
            return None
        
        cache_key = self._get_cache_key(text, model)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            async with aiofiles.open(cache_path, "r") as f:
                content = await f.read()
                data = json.loads(content)
                
                self.logger.debug("Cache hit", cache_key=cache_key[:8])
                return data.get("embedding")
                
        except Exception as e:
            self.logger.warning(
                "Failed to read from cache",
                cache_key=cache_key[:8],
                error=str(e)
            )
            return None
    
    async def set(
        self,
        text: str,
        embedding: list[float],
        model: str = ""
    ) -> bool:
        """
        Store embedding in cache.
        
        Args:
            text: Original text
            embedding: Embedding vector
            model: Model name
        
        Returns:
            bool: True if stored successfully
        """
        if not self.enabled:
            return False
        
        cache_key = self._get_cache_key(text, model)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            # Create subdirectory
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Store data
            data = {
                "text_preview": text[:100],  # For debugging
                "model": model,
                "dimension": len(embedding),
                "embedding": embedding
            }
            
            async with aiofiles.open(cache_path, "w") as f:
                await f.write(json.dumps(data))
            
            self.logger.debug("Cache set", cache_key=cache_key[:8])
            return True
            
        except Exception as e:
            self.logger.warning(
                "Failed to write to cache",
                cache_key=cache_key[:8],
                error=str(e)
            )
            return False
    
    async def get_many(
        self,
        texts: list[str],
        model: str = ""
    ) -> dict[int, list[float]]:
        """
        Get multiple embeddings from cache.
        
        Args:
            texts: List of texts
            model: Model name
        
        Returns:
            dict[int, list[float]]: Map of index -> embedding for cache hits
        """
        if not self.enabled:
            return {}
        
        results = {}
        
        for i, text in enumerate(texts):
            embedding = await self.get(text, model)
            if embedding is not None:
                results[i] = embedding
        
        return results
    
    async def set_many(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        model: str = ""
    ) -> int:
        """
        Store multiple embeddings in cache.
        
        Args:
            texts: List of texts
            embeddings: Corresponding embeddings
            model: Model name
        
        Returns:
            int: Number of embeddings stored
        """
        if not self.enabled:
            return 0
        
        stored = 0
        
        for text, embedding in zip(texts, embeddings):
            if await self.set(text, embedding, model):
                stored += 1
        
        return stored
    
    async def clear(self) -> int:
        """
        Clear all cached embeddings.
        
        Returns:
            int: Number of entries cleared
        """
        if not self.cache_dir.exists():
            return 0
        
        count = 0
        
        for cache_file in self.cache_dir.rglob("*.json"):
            try:
                cache_file.unlink()
                count += 1
            except Exception:
                pass
        
        self.logger.info("Cache cleared", entries_removed=count)
        return count
    
    async def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            dict: Cache statistics
        """
        if not self.cache_dir.exists():
            return {"enabled": self.enabled, "entries": 0, "size_bytes": 0}
        
        entries = list(self.cache_dir.rglob("*.json"))
        total_size = sum(f.stat().st_size for f in entries)
        
        return {
            "enabled": self.enabled,
            "entries": len(entries),
            "size_bytes": total_size,
            "size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir)
        }
    
    def get_stats_sync(self) -> dict[str, Any]:
        """
        Get cache statistics (synchronous version).
        
        Returns:
            dict: Cache statistics
        """
        if not self.cache_dir.exists():
            return {"enabled": self.enabled, "entries": 0, "size_bytes": 0}
        
        entries = list(self.cache_dir.rglob("*.json"))
        total_size = sum(f.stat().st_size for f in entries)
        
        return {
            "enabled": self.enabled,
            "entries": len(entries),
            "size_bytes": total_size,
            "size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir)
        }