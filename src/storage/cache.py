"""
Caching Layer

Implements caching for frequently accessed data including question sets,
model configurations, and benchmark results with TTL and invalidation.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
import logging
from functools import wraps

from core.config import get_config
from core.exceptions import CacheError

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    data: Any
    created_at: datetime
    ttl: int  # Time to live in seconds
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() - self.created_at > timedelta(seconds=self.ttl)

    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return (datetime.now() - self.created_at).total_seconds()

    def access(self) -> None:
        """Mark entry as accessed."""
        self.access_count += 1
        self.last_accessed = datetime.now()


class CacheManager:
    """In-memory cache manager with persistence."""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.config = get_config()
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load persisted cache on startup
        self._load_persisted_cache()

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if key in self.cache:
            entry = self.cache[key]
            if entry.is_expired:
                self.delete(key)
                return None

            entry.access()
            return entry.data

        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        if ttl is None:
            ttl = self.default_ttl

        # Remove expired entries if cache is full
        if len(self.cache) >= self.max_size:
            self._cleanup_expired()

        if len(self.cache) >= self.max_size:
            self._evict_lru()

        entry = CacheEntry(
            key=key,
            data=value,
            created_at=datetime.now(),
            ttl=ttl
        )

        self.cache[key] = entry

        # Persist important cache entries
        if self._should_persist(key):
            self._persist_entry(entry)

    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.

        Args:
            key: Cache key

        Returns:
            True if entry was deleted
        """
        if key in self.cache:
            del self.cache[key]
            self._delete_persisted(key)
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self._clear_persisted()

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate cache entries matching pattern.

        Args:
            pattern: Pattern to match keys

        Returns:
            Number of entries invalidated
        """
        keys_to_delete = [key for key in self.cache.keys() if pattern in key]
        for key in keys_to_delete:
            self.delete(key)
        return len(keys_to_delete)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self.cache)
        expired_entries = sum(1 for entry in self.cache.values() if entry.is_expired)
        total_accesses = sum(entry.access_count for entry in self.cache.values())

        return {
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "active_entries": total_entries - expired_entries,
            "total_accesses": total_accesses,
            "max_size": self.max_size,
            "hit_rate": self._calculate_hit_rate()
        }

    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        expired_keys = [key for key, entry in self.cache.items() if entry.is_expired]
        for key in expired_keys:
            self.delete(key)

    def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        if not self.cache:
            return

        # Find entry with oldest last_accessed (or created_at if never accessed)
        lru_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].last_accessed or self.cache[k].created_at
        )
        self.delete(lru_key)

    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate (simplified)."""
        # This is a simplified calculation - in practice you'd track hits/misses
        return 0.0  # Placeholder

    def _should_persist(self, key: str) -> bool:
        """Determine if cache entry should be persisted."""
        # Persist question sets and model configurations
        return key.startswith(("questions:", "model_config:", "benchmark_results:"))

    def _persist_entry(self, entry: CacheEntry) -> None:
        """Persist cache entry to disk."""
        try:
            cache_file = self.cache_dir / f"{hashlib.md5(entry.key.encode()).hexdigest()}.json"

            data = {
                "key": entry.key,
                "data": entry.data,
                "created_at": entry.created_at.isoformat(),
                "ttl": entry.ttl,
                "access_count": entry.access_count,
                "last_accessed": entry.last_accessed.isoformat() if entry.last_accessed else None
            }

            with open(cache_file, 'w') as f:
                json.dump(data, f, default=str)

        except Exception as e:
            logger.warning(f"Failed to persist cache entry {entry.key}: {e}")

    def _load_persisted_cache(self) -> None:
        """Load persisted cache entries."""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)

                    # Check if entry is still valid
                    created_at = datetime.fromisoformat(data["created_at"])
                    ttl = data["ttl"]
                    if datetime.now() - created_at > timedelta(seconds=ttl):
                        cache_file.unlink()  # Remove expired file
                        continue

                    entry = CacheEntry(
                        key=data["key"],
                        data=data["data"],
                        created_at=created_at,
                        ttl=ttl,
                        access_count=data.get("access_count", 0),
                        last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None
                    )

                    self.cache[entry.key] = entry

                except Exception as e:
                    logger.warning(f"Failed to load cache file {cache_file}: {e}")
                    cache_file.unlink()  # Remove corrupted file

        except Exception as e:
            logger.error(f"Failed to load persisted cache: {e}")

    def _delete_persisted(self, key: str) -> None:
        """Delete persisted cache entry."""
        try:
            cache_file = self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.json"
            if cache_file.exists():
                cache_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete persisted cache entry {key}: {e}")

    def _clear_persisted(self) -> None:
        """Clear all persisted cache entries."""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to clear persisted cache: {e}")


# Global cache instance
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        config = get_config()
        _cache_manager = CacheManager(
            max_size=config.cache.max_size,
            default_ttl=config.cache.ttl
        )
    return _cache_manager


# Cache decorators and utilities
def cached(ttl: Optional[int] = None, key_prefix: str = ""):
    """
    Decorator to cache function results.

    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache keys
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix or func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}:{v}" for k, v in kwargs.items())
            cache_key = ":".join(key_parts)

            cache = get_cache_manager()

            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result

        return wrapper
    return decorator


def cache_questions(questions: List[Dict[str, Any]], key: str) -> None:
    """
    Cache question set.

    Args:
        questions: List of question dictionaries
        key: Cache key
    """
    cache = get_cache_manager()
    cache_key = f"questions:{key}"
    cache.set(cache_key, questions, ttl=3600)  # 1 hour TTL


def get_cached_questions(key: str) -> Optional[List[Dict[str, Any]]]:
    """
    Get cached question set.

    Args:
        key: Cache key

    Returns:
        Cached questions or None
    """
    cache = get_cache_manager()
    cache_key = f"questions:{key}"
    return cache.get(cache_key)


def cache_model_config(model_name: str, config: Dict[str, Any]) -> None:
    """
    Cache model configuration.

    Args:
        model_name: Model name
        config: Model configuration
    """
    cache = get_cache_manager()
    cache_key = f"model_config:{model_name}"
    cache.set(cache_key, config, ttl=3600)  # 1 hour TTL


def get_cached_model_config(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get cached model configuration.

    Args:
        model_name: Model name

    Returns:
        Cached configuration or None
    """
    cache = get_cache_manager()
    cache_key = f"model_config:{model_name}"
    return cache.get(cache_key)


def cache_benchmark_results(benchmark_id: int, results: List[Dict[str, Any]]) -> None:
    """
    Cache benchmark results.

    Args:
        benchmark_id: Benchmark ID
        results: Benchmark results
    """
    cache = get_cache_manager()
    cache_key = f"benchmark_results:{benchmark_id}"
    cache.set(cache_key, results, ttl=1800)  # 30 minutes TTL


def get_cached_benchmark_results(benchmark_id: int) -> Optional[List[Dict[str, Any]]]:
    """
    Get cached benchmark results.

    Args:
        benchmark_id: Benchmark ID

    Returns:
        Cached results or None
    """
    cache = get_cache_manager()
    cache_key = f"benchmark_results:{benchmark_id}"
    return cache.get(cache_key)


def invalidate_question_cache(pattern: str = "") -> int:
    """
    Invalidate question cache entries.

    Args:
        pattern: Pattern to match in cache keys

    Returns:
        Number of entries invalidated
    """
    cache = get_cache_manager()
    return cache.invalidate_pattern(f"questions:{pattern}")


def invalidate_model_cache(model_name: Optional[str] = None) -> int:
    """
    Invalidate model configuration cache.

    Args:
        model_name: Specific model name or None for all

    Returns:
        Number of entries invalidated
    """
    cache = get_cache_manager()
    if model_name:
        return cache.invalidate_pattern(f"model_config:{model_name}")
    else:
        return cache.invalidate_pattern("model_config:")


def invalidate_benchmark_cache(benchmark_id: Optional[int] = None) -> int:
    """
    Invalidate benchmark results cache.

    Args:
        benchmark_id: Specific benchmark ID or None for all

    Returns:
        Number of entries invalidated
    """
    cache = get_cache_manager()
    if benchmark_id:
        return cache.invalidate_pattern(f"benchmark_results:{benchmark_id}")
    else:
        return cache.invalidate_pattern("benchmark_results:")


# CLI Helper Functions
def cli_clear_cache() -> None:
    """CLI command to clear all cache."""
    cache = get_cache_manager()
    cache.clear()
    print("Cache cleared")


def cli_cache_stats() -> None:
    """CLI command to show cache statistics."""
    cache = get_cache_manager()
    stats = cache.get_stats()
    print("Cache Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def cli_invalidate_pattern(pattern: str) -> None:
    """CLI command to invalidate cache entries by pattern."""
    cache = get_cache_manager()
    count = cache.invalidate_pattern(pattern)
    print(f"Invalidated {count} cache entries matching '{pattern}'")