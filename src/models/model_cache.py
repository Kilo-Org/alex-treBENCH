"""
Model Cache System

Simple file-based caching system for OpenRouter model data with TTL support.
"""

import json
import os
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


class ModelCache:
    """
    File-based cache for storing OpenRouter model data.
    
    Provides simple caching with TTL (Time-To-Live) support to avoid
    frequent API calls to OpenRouter for model information.
    """
    
    def __init__(self, cache_path: str = "data/cache/models.json", ttl_seconds: int = 3600):
        """
        Initialize model cache.
        
        Args:
            cache_path: Path to the cache file
            ttl_seconds: Time-to-live in seconds (default: 1 hour)
        """
        self.cache_path = Path(cache_path)
        self.ttl_seconds = ttl_seconds
        
        # Ensure cache directory exists
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"ModelCache initialized with path: {self.cache_path}, TTL: {ttl_seconds}s")
    
    def save_cache(self, models: List[Dict[str, Any]]) -> bool:
        """
        Save model data to cache file with timestamp.
        
        Args:
            models: List of model dictionaries to cache
            
        Returns:
            True if cache was saved successfully, False otherwise
        """
        try:
            cache_data = {
                'timestamp': time.time(),
                'cached_at': datetime.now().isoformat(),
                'ttl_seconds': self.ttl_seconds,
                'model_count': len(models),
                'models': models
            }
            
            # Write to temporary file first, then rename (atomic operation)
            temp_path = self.cache_path.with_suffix('.tmp')
            
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            temp_path.replace(self.cache_path)
            
            logger.info(f"Successfully cached {len(models)} models to {self.cache_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model cache: {e}")
            # Clean up temp file if it exists
            temp_path = self.cache_path.with_suffix('.tmp')
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            return False
    
    def load_cache(self) -> Optional[List[Dict[str, Any]]]:
        """
        Load model data from cache file.
        
        Returns:
            List of cached model dictionaries, or None if cache is invalid/missing
        """
        try:
            if not self.cache_path.exists():
                logger.debug("Cache file does not exist")
                return None
            
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Validate cache structure
            required_keys = ['timestamp', 'models']
            if not all(key in cache_data for key in required_keys):
                logger.warning("Cache file has invalid structure, ignoring")
                return None
            
            # Check if cache is still valid
            if not self._is_cache_data_valid(cache_data):
                logger.info("Cache has expired, will fetch fresh data")
                return None
            
            models = cache_data['models']
            model_count = len(models)
            cached_at = cache_data.get('cached_at', 'unknown')
            
            logger.info(f"Loaded {model_count} models from cache (cached at: {cached_at})")
            return models
            
        except json.JSONDecodeError as e:
            logger.error(f"Cache file is corrupted (invalid JSON): {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load model cache: {e}")
            return None
    
    def is_cache_valid(self) -> bool:
        """
        Check if cache exists and is still valid (within TTL).
        
        Returns:
            True if cache is valid, False otherwise
        """
        try:
            if not self.cache_path.exists():
                return False
            
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            return self._is_cache_data_valid(cache_data)
            
        except Exception as e:
            logger.debug(f"Cache validation failed: {e}")
            return False
    
    def _is_cache_data_valid(self, cache_data: Dict[str, Any]) -> bool:
        """
        Check if cache data is within TTL.
        
        Args:
            cache_data: Loaded cache data dictionary
            
        Returns:
            True if cache is within TTL, False otherwise
        """
        try:
            cache_timestamp = cache_data.get('timestamp')
            if cache_timestamp is None:
                return False
            
            current_time = time.time()
            age_seconds = current_time - cache_timestamp
            
            # Use TTL from cache data if available, otherwise use instance TTL
            ttl = cache_data.get('ttl_seconds', self.ttl_seconds)
            
            is_valid = age_seconds < ttl
            
            if not is_valid:
                logger.debug(f"Cache expired: age={age_seconds:.1f}s, ttl={ttl}s")
            else:
                logger.debug(f"Cache valid: age={age_seconds:.1f}s, ttl={ttl}s")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Error validating cache data: {e}")
            return False
    
    def clear_cache(self) -> bool:
        """
        Remove the cache file.
        
        Returns:
            True if cache was cleared successfully, False otherwise
        """
        try:
            if self.cache_path.exists():
                self.cache_path.unlink()
                logger.info(f"Cache cleared: {self.cache_path}")
                return True
            else:
                logger.debug("No cache file to clear")
                return True
                
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the current cache.
        
        Returns:
            Dictionary with cache information
        """
        info = {
            'cache_path': str(self.cache_path),
            'ttl_seconds': self.ttl_seconds,
            'exists': self.cache_path.exists(),
            'valid': False,
            'size_bytes': 0,
            'model_count': 0,
            'cached_at': None,
            'age_seconds': None
        }
        
        try:
            if self.cache_path.exists():
                info['size_bytes'] = self.cache_path.stat().st_size
                
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                info['valid'] = self._is_cache_data_valid(cache_data)
                info['model_count'] = cache_data.get('model_count', len(cache_data.get('models', [])))
                info['cached_at'] = cache_data.get('cached_at')
                
                cache_timestamp = cache_data.get('timestamp')
                if cache_timestamp:
                    info['age_seconds'] = time.time() - cache_timestamp
                    
        except Exception as e:
            logger.debug(f"Error getting cache info: {e}")
        
        return info


# Global cache instance with default settings
default_model_cache = ModelCache()


def get_model_cache(cache_path: Optional[str] = None, ttl_seconds: Optional[int] = None) -> ModelCache:
    """
    Get a model cache instance.
    
    Args:
        cache_path: Optional custom cache path
        ttl_seconds: Optional custom TTL in seconds
        
    Returns:
        ModelCache instance
    """
    if cache_path is None and ttl_seconds is None:
        return default_model_cache
    
    return ModelCache(
        cache_path=cache_path or "data/cache/models.json",
        ttl_seconds=ttl_seconds or 3600
    )