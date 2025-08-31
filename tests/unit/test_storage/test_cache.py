"""
Unit tests for CacheManager.
"""

import pytest
import json
import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from storage.cache import CacheManager, get_cache_manager, cached
from core.exceptions import CacheError


class TestCacheManager:
    """Test cases for CacheManager."""

    def setup_method(self):
        """Setup test fixtures."""
        self.cache_dir = Path("test_cache")
        self.cache = CacheManager(max_size=10, default_ttl=60)
        self.cache.cache_dir = self.cache_dir

    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up cache directory
        if self.cache_dir.exists():
            for file in self.cache_dir.glob("*"):
                file.unlink()
            self.cache_dir.rmdir()

        # Clear cache
        self.cache.clear()

    def test_get_cache_manager_singleton(self):
        """Test that get_cache_manager returns singleton."""
        cache1 = get_cache_manager()
        cache2 = get_cache_manager()
        assert cache1 is cache2

    def test_cache_set_and_get(self):
        """Test basic cache set and get operations."""
        # Set a value
        self.cache.set("test_key", "test_value", ttl=300)

        # Get the value
        value = self.cache.get("test_key")
        assert value == "test_value"

    def test_cache_get_nonexistent_key(self):
        """Test getting a non-existent key."""
        value = self.cache.get("nonexistent_key")
        assert value is None

    def test_cache_get_expired_key(self):
        """Test getting an expired key."""
        # Set a value with very short TTL
        self.cache.set("expired_key", "expired_value", ttl=0.001)

        # Wait for expiration
        import time
        time.sleep(0.01)

        # Try to get expired value
        value = self.cache.get("expired_key")
        assert value is None

    def test_cache_delete(self):
        """Test cache deletion."""
        # Set a value
        self.cache.set("delete_key", "delete_value")

        # Verify it exists
        assert self.cache.get("delete_key") == "delete_value"

        # Delete it
        result = self.cache.delete("delete_key")
        assert result is True

        # Verify it's gone
        assert self.cache.get("delete_key") is None

    def test_cache_delete_nonexistent_key(self):
        """Test deleting a non-existent key."""
        result = self.cache.delete("nonexistent_key")
        assert result is False

    def test_cache_clear(self):
        """Test cache clearing."""
        # Set multiple values
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        self.cache.set("key3", "value3")

        # Clear cache
        self.cache.clear()

        # Verify all are gone
        assert self.cache.get("key1") is None
        assert self.cache.get("key2") is None
        assert self.cache.get("key3") is None

        # Verify cache is empty
        assert len(self.cache.cache) == 0

    def test_cache_max_size_eviction(self):
        """Test cache size limit and LRU eviction."""
        # Set cache max size to 3
        small_cache = CacheManager(max_size=3, default_ttl=300)

        try:
            # Fill cache to max
            small_cache.set("key1", "value1")
            small_cache.set("key2", "value2")
            small_cache.set("key3", "value3")

            assert len(small_cache.cache) == 3

            # Add one more (should trigger eviction)
            small_cache.set("key4", "value4")

            # Cache should still have max 3 items
            assert len(small_cache.cache) <= 3

        finally:
            small_cache.clear()

    def test_cache_invalidate_pattern(self):
        """Test pattern-based cache invalidation."""
        # Set multiple keys with similar patterns
        self.cache.set("questions:category1", "data1")
        self.cache.set("questions:category2", "data2")
        self.cache.set("model_config:gpt4", "config1")
        self.cache.set("other:key", "other_data")

        # Invalidate questions pattern
        invalidated = self.cache.invalidate_pattern("questions:")
        assert invalidated == 2

        # Verify questions are gone but others remain
        assert self.cache.get("questions:category1") is None
        assert self.cache.get("questions:category2") is None
        assert self.cache.get("model_config:gpt4") == "config1"
        assert self.cache.get("other:key") == "other_data"

    def test_cache_stats(self):
        """Test cache statistics."""
        # Set some values
        self.cache.set("active_key", "active_value", ttl=300)
        self.cache.set("expired_key", "expired_value", ttl=0.001)

        # Wait for expiration
        import time
        time.sleep(0.01)

        # Get stats
        stats = self.cache.get_stats()

        assert "total_entries" in stats
        assert "expired_entries" in stats
        assert "active_entries" in stats
        assert stats["total_entries"] >= 1  # At least the active entry
        assert stats["expired_entries"] >= 1  # At least the expired entry

    def test_cache_persistence(self):
        """Test cache persistence to disk."""
        # Set a value that should be persisted
        self.cache.set("questions:test", "test_data", ttl=300)

        # Force cleanup to trigger persistence
        self.cache._cleanup_expired()

        # Verify file was created
        expected_filename = hashlib.md5("questions:test".encode()).hexdigest() + ".json"
        cache_file = self.cache_dir / expected_filename
        assert cache_file.exists()

        # Verify file contents
        with open(cache_file, 'r') as f:
            data = json.load(f)

        assert data["key"] == "questions:test"
        assert data["data"] == "test_data"

    def test_cache_load_persisted(self):
        """Test loading persisted cache entries."""
        # Create a persisted cache file manually
        cache_key = "persisted:key"
        cache_data = {
            "key": cache_key,
            "data": "persisted_value",
            "created_at": datetime.now().isoformat(),
            "ttl": 300,
            "access_count": 5,
            "last_accessed": datetime.now().isoformat()
        }

        filename = hashlib.md5(cache_key.encode()).hexdigest() + ".json"
        cache_file = self.cache_dir / filename

        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)

        # Create new cache manager (should load persisted data)
        new_cache = CacheManager(max_size=10, default_ttl=60)
        new_cache.cache_dir = self.cache_dir

        # Manually trigger load
        new_cache._load_persisted_cache()

        # Verify data was loaded
        loaded_value = new_cache.get(cache_key)
        assert loaded_value == "persisted_value"

        new_cache.clear()

    def test_cache_access_tracking(self):
        """Test cache access count tracking."""
        # Set a value
        self.cache.set("access_test", "access_value")

        # Get the entry
        entry = self.cache.cache["access_test"]

        # Initial access count should be 0
        assert entry.access_count == 0

        # Access the value
        self.cache.get("access_test")

        # Access count should be 1
        assert entry.access_count == 1
        assert entry.last_accessed is not None

    def test_cache_ttl_default(self):
        """Test default TTL usage."""
        # Set without TTL (should use default)
        self.cache.set("default_ttl_key", "default_value")

        entry = self.cache.cache["default_ttl_key"]
        assert entry.ttl == self.cache.default_ttl

    def test_cache_key_with_special_characters(self):
        """Test cache keys with special characters."""
        special_key = "key:with:colons/and/slashes"
        self.cache.set(special_key, "special_value")

        value = self.cache.get(special_key)
        assert value == "special_value"


class TestCacheDecorators:
    """Test cases for cache decorators."""

    def setup_method(self):
        """Setup test fixtures."""
        self.cache = CacheManager(max_size=10, default_ttl=60)
        # Clear any existing cache
        self.cache.clear()

    def teardown_method(self):
        """Clean up test fixtures."""
        self.cache.clear()

    def test_cached_decorator(self):
        """Test the cached decorator."""
        call_count = 0

        @cached(ttl=300, key_prefix="test_func")
        def test_function(x, y=10):
            nonlocal call_count
            call_count += 1
            return x + y

        # First call should execute function
        result1 = test_function(5, y=15)
        assert result1 == 20
        assert call_count == 1

        # Second call with same args should use cache
        result2 = test_function(5, y=15)
        assert result2 == 20
        assert call_count == 1  # Should not have increased

        # Call with different args should execute again
        result3 = test_function(10, y=15)
        assert result3 == 25
        assert call_count == 2

    def test_cached_decorator_with_different_prefixes(self):
        """Test cached decorator with different key prefixes."""
        call_count = 0

        @cached(ttl=300, key_prefix="func1")
        def func1(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        @cached(ttl=300, key_prefix="func2")
        def func2(x):
            nonlocal call_count
            call_count += 1
            return x * 3

        # Call both functions with same arg
        result1 = func1(5)
        result2 = func2(5)

        assert result1 == 10
        assert result2 == 15
        assert call_count == 2  # Both should have executed


class TestCacheUtilityFunctions:
    """Test cases for cache utility functions."""

    def setup_method(self):
        """Setup test fixtures."""
        self.cache = CacheManager(max_size=10, default_ttl=60)
        self.cache.clear()

    def teardown_method(self):
        """Clean up test fixtures."""
        self.cache.clear()

    @patch('storage.cache.get_cache_manager')
    def test_cache_questions(self, mock_get_cache):
        """Test cache_questions utility function."""
        mock_get_cache.return_value = self.cache

        from storage.cache import cache_questions

        questions = [{"id": "q1", "text": "Question?"}]
        cache_questions(questions, "test_key")

        # Verify data was cached
        cached = self.cache.get("questions:test_key")
        assert cached == questions

    @patch('storage.cache.get_cache_manager')
    def test_get_cached_questions(self, mock_get_cache):
        """Test get_cached_questions utility function."""
        mock_get_cache.return_value = self.cache

        from storage.cache import get_cached_questions

        # Set up cached data
        questions = [{"id": "q1", "text": "Question?"}]
        self.cache.set("questions:test_key", questions)

        # Retrieve cached data
        cached = get_cached_questions("test_key")
        assert cached == questions

    @patch('storage.cache.get_cache_manager')
    def test_invalidate_question_cache(self, mock_get_cache):
        """Test invalidate_question_cache utility function."""
        mock_get_cache.return_value = self.cache

        from storage.cache import invalidate_question_cache

        # Set up some cached questions
        self.cache.set("questions:key1", "data1")
        self.cache.set("questions:key2", "data2")
        self.cache.set("other:key", "other_data")

        # Invalidate questions
        invalidated = invalidate_question_cache()
        assert invalidated == 2

        # Verify questions are gone
        assert self.cache.get("questions:key1") is None
        assert self.cache.get("questions:key2") is None
        assert self.cache.get("other:key") == "other_data"