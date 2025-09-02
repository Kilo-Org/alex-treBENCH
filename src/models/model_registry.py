"""
Model Registry

Registry of available models with their configurations, pricing, and metadata.
Enhanced with dynamic fetching from OpenRouter API and caching support.
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .model_cache import get_model_cache
from .openrouter import OpenRouterClient
from src.utils.logging import get_logger
from src.core.config import get_config

logger = get_logger(__name__)


class ModelProvider(str, Enum):
    """Model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    META = "meta"
    MISTRAL = "mistral"
    GOOGLE = "google"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    model_id: str
    display_name: str
    provider: ModelProvider
    context_window: int
    input_cost_per_1m_tokens: float
    output_cost_per_1m_tokens: float
    supports_streaming: bool = True
    max_tokens_default: int = 150
    temperature_default: float = 0.1
    capabilities: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []


# Default model constant - Claude 3.5 Sonnet
DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"


class ModelRegistry:
    """Registry of available models with their configurations."""
    
    # Static model configurations (fallback when API is unavailable)
    MODELS = {
        # OpenAI Models
        "openai/gpt-3.5-turbo": ModelConfig(
            model_id="openai/gpt-3.5-turbo",
            display_name="GPT-3.5 Turbo",
            provider=ModelProvider.OPENAI,
            context_window=4096,
            input_cost_per_1m_tokens=0.5,
            output_cost_per_1m_tokens=1.5,
            supports_streaming=True,
            capabilities=["chat", "reasoning"]
        ),
        "openai/gpt-4": ModelConfig(
            model_id="openai/gpt-4",
            display_name="GPT-4",
            provider=ModelProvider.OPENAI,
            context_window=8192,
            input_cost_per_1m_tokens=30.0,
            output_cost_per_1m_tokens=60.0,
            supports_streaming=True,
            capabilities=["chat", "reasoning", "analysis"]
        ),
        "openai/gpt-4-turbo": ModelConfig(
            model_id="openai/gpt-4-turbo",
            display_name="GPT-4 Turbo",
            provider=ModelProvider.OPENAI,
            context_window=128000,
            input_cost_per_1m_tokens=10.0,
            output_cost_per_1m_tokens=30.0,
            supports_streaming=True,
            capabilities=["chat", "reasoning", "analysis", "long-context"]
        ),
        
        # Anthropic Models
        "anthropic/claude-3-haiku": ModelConfig(
            model_id="anthropic/claude-3-haiku",
            display_name="Claude 3 Haiku",
            provider=ModelProvider.ANTHROPIC,
            context_window=200000,
            input_cost_per_1m_tokens=0.25,
            output_cost_per_1m_tokens=1.25,
            supports_streaming=True,
            capabilities=["chat", "reasoning", "analysis", "long-context"]
        ),
        "anthropic/claude-3-sonnet": ModelConfig(
            model_id="anthropic/claude-3-sonnet",
            display_name="Claude 3 Sonnet",
            provider=ModelProvider.ANTHROPIC,
            context_window=200000,
            input_cost_per_1m_tokens=3.0,
            output_cost_per_1m_tokens=15.0,
            supports_streaming=True,
            capabilities=["chat", "reasoning", "analysis", "long-context", "creative"]
        ),
        "anthropic/claude-3.5-sonnet": ModelConfig(
            model_id="anthropic/claude-3.5-sonnet",
            display_name="Claude 3.5 Sonnet",
            provider=ModelProvider.ANTHROPIC,
            context_window=200000,
            input_cost_per_1m_tokens=3.0,
            output_cost_per_1m_tokens=15.0,
            supports_streaming=True,
            capabilities=["chat", "reasoning", "analysis", "long-context", "creative"]
        ),
        "anthropic/claude-3-opus": ModelConfig(
            model_id="anthropic/claude-3-opus",
            display_name="Claude 3 Opus",
            provider=ModelProvider.ANTHROPIC,
            context_window=200000,
            input_cost_per_1m_tokens=15.0,
            output_cost_per_1m_tokens=75.0,
            supports_streaming=True,
            capabilities=["chat", "reasoning", "analysis", "long-context", "creative", "complex-reasoning"]
        ),
        
        # Meta Models
        "meta-llama/llama-2-70b-chat": ModelConfig(
            model_id="meta-llama/llama-2-70b-chat",
            display_name="Llama 2 70B Chat",
            provider=ModelProvider.META,
            context_window=4096,
            input_cost_per_1m_tokens=0.7,
            output_cost_per_1m_tokens=0.8,
            supports_streaming=True,
            capabilities=["chat", "reasoning"]
        ),
        
        # Mistral Models
        "mistralai/mixtral-8x7b-instruct": ModelConfig(
            model_id="mistralai/mixtral-8x7b-instruct",
            display_name="Mixtral 8x7B Instruct",
            provider=ModelProvider.MISTRAL,
            context_window=32768,
            input_cost_per_1m_tokens=0.24,
            output_cost_per_1m_tokens=0.24,
            supports_streaming=True,
            capabilities=["chat", "reasoning", "multilingual"]
        ),
        
        # Google Models
        "google/palm-2-chat": ModelConfig(
            model_id="google/palm-2-chat",
            display_name="PaLM 2 Chat",
            provider=ModelProvider.GOOGLE,
            context_window=8192,
            input_cost_per_1m_tokens=0.5,
            output_cost_per_1m_tokens=0.5,
            supports_streaming=False,
            capabilities=["chat", "reasoning"]
        ),
    }
    
    def __init__(self):
        """Initialize model registry with caching support."""
        self._config = get_config()
        self._cache = None
        self._cached_models = None
        
    def _get_cache(self):
        """Get or create model cache instance."""
        if self._cache is None:
            cache_config = getattr(self._config, 'model_cache', None)
            if cache_config is not None:
                cache_path = cache_config.path
                cache_ttl = cache_config.ttl_seconds
            else:
                cache_path = 'data/cache/models.json'
                cache_ttl = 3600
            self._cache = get_model_cache(cache_path, cache_ttl)
        return self._cache
    
    async def fetch_models(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch models directly from OpenRouter API.
        
        Returns:
            List of model dictionaries from API, or None if fetch fails
        """
        try:
            logger.info("Fetching models from OpenRouter API")
            
            # Create OpenRouter client
            client = OpenRouterClient()
            
            # Fetch models from API
            models = await client.list_available_models()
            
            if models:
                logger.info(f"Successfully fetched {len(models)} models from OpenRouter")
                
                # Cache the results
                cache = self._get_cache()
                cache.save_cache(models)
                
                return models
            else:
                logger.warning("No models returned from OpenRouter API")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch models from OpenRouter: {e}")
            return None
        finally:
            # Clean up client
            try:
                await client.close()
            except:
                pass
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get available models trying API first, then cache, then fallback.
        
        This method implements the priority order:
        1. Fresh data from OpenRouter API
        2. Valid cached data
        3. Static fallback models
        
        Returns:
            List of model dictionaries
        """
        try:
            # Try to fetch fresh data from API
            api_models = await self.fetch_models()
            if api_models:
                logger.debug("Using fresh models from API")
                return api_models
            
        except Exception as e:
            logger.warning(f"API fetch failed: {e}")
        
        # Try to load from cache
        try:
            cache = self._get_cache()
            cached_models = cache.load_cache()
            if cached_models:
                logger.info("Using cached models")
                return cached_models
                
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
        
        # Fall back to static models
        logger.info("Using static fallback models")
        return self._convert_static_models_to_api_format()
    
    def _convert_static_models_to_api_format(self) -> List[Dict[str, Any]]:
        """Convert static ModelConfig objects to API-compatible format."""
        api_models = []
        
        for model_id, config in self.MODELS.items():
            api_model = {
                'id': config.model_id,
                'name': config.display_name,
                'description': f"{config.display_name} model from {config.provider.value}",
                'provider': config.provider.value,
                'context_length': config.context_window,
                'pricing': {
                    'input_cost_per_1m_tokens': config.input_cost_per_1m_tokens,
                    'output_cost_per_1m_tokens': config.output_cost_per_1m_tokens
                },
                'capabilities': config.capabilities,
                'available': True,
                'per_request_limits': {},
                'top_provider': {},
                'architecture': {},
                'modality': 'text',
                'updated_at': None
            }
            api_models.append(api_model)
        
        return api_models
    
    def search_models(self, query: str, models: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Search through model names and IDs.
        
        Args:
            query: Search query string
            models: Optional list of models to search through (if None, uses cached models)
            
        Returns:
            List of matching model dictionaries
        """
        if models is None:
            # Try to use cached models synchronously
            try:
                cache = self._get_cache()
                models = cache.load_cache()
                if not models:
                    # Fall back to static models
                    models = self._convert_static_models_to_api_format()
            except Exception:
                models = self._convert_static_models_to_api_format()
        
        if not query or not models:
            return models or []
        
        query_lower = query.lower()
        matching_models = []
        
        for model in models:
            # Search in model ID
            if query_lower in model.get('id', '').lower():
                matching_models.append(model)
                continue
                
            # Search in display name
            if query_lower in model.get('name', '').lower():
                matching_models.append(model)
                continue
                
            # Search in provider
            if query_lower in model.get('provider', '').lower():
                matching_models.append(model)
                continue
                
            # Search in capabilities
            capabilities = model.get('capabilities', [])
            if any(query_lower in cap.lower() for cap in capabilities):
                matching_models.append(model)
                continue
        
        logger.debug(f"Found {len(matching_models)} models matching query: '{query}'")
        return matching_models
    
    @classmethod
    def list_available_models(cls) -> List[str]:
        """List all available static model IDs (for backward compatibility)."""
        return list(cls.MODELS.keys())
    
    @classmethod
    def get_model_config(cls, model_id: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model (static only)."""
        return cls.MODELS.get(model_id)
    
    @classmethod
    def validate_model_availability(cls, model_id: str) -> bool:
        """Check if a model is available in the static registry."""
        return model_id in cls.MODELS
    
    @classmethod
    def get_models_by_provider(cls, provider: ModelProvider) -> List[ModelConfig]:
        """Get all static models from a specific provider."""
        return [config for config in cls.MODELS.values() if config.provider == provider]
    
    @classmethod
    def get_models_by_capability(cls, capability: str) -> List[ModelConfig]:
        """Get static models that have a specific capability."""
        return [
            config for config in cls.MODELS.values()
            if config.capabilities is not None and capability in config.capabilities
        ]
    
    @classmethod
    def get_cheapest_models(cls, limit: int = 5) -> List[ModelConfig]:
        """Get the cheapest static models by combined input/output cost."""
        models = list(cls.MODELS.values())
        models.sort(key=lambda x: (x.input_cost_per_1m_tokens + x.output_cost_per_1m_tokens))
        return models[:limit]
    
    @classmethod
    def get_models_with_long_context(cls, min_context: int = 32000) -> List[ModelConfig]:
        """Get static models with context window larger than specified size."""
        return [
            config for config in cls.MODELS.values()
            if config.context_window >= min_context
        ]
    
    @classmethod
    def estimate_cost(cls, model_id: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a model given token counts."""
        # First try to get pricing from cached dynamic models
        pricing_info = cls._get_model_pricing(model_id)
        if pricing_info:
            # OpenRouter API pricing is per million tokens (despite field names)
            input_cost_per_1m_tokens = pricing_info.get('input_cost_per_1m_tokens', 0.0)
            output_cost_per_1m_tokens = pricing_info.get('output_cost_per_1m_tokens', 0.0)
            
            input_cost = (input_tokens / 1_000_000) * input_cost_per_1m_tokens
            output_cost = (output_tokens / 1_000_000) * output_cost_per_1m_tokens
            
            return input_cost + output_cost
        
        # Fall back to static models
        config = cls.get_model_config(model_id)
        if not config:
            return 0.0
        
        input_cost = (input_tokens / 1_000_000) * config.input_cost_per_1m_tokens
        output_cost = (output_tokens / 1_000_000) * config.output_cost_per_1m_tokens
        
        return input_cost + output_cost
    
    @classmethod
    def _get_model_pricing(cls, model_id: str) -> Optional[Dict[str, float]]:
        """
        Get pricing information for a model from cached dynamic models.
        
        Args:
            model_id: The model ID to get pricing for
            
        Returns:
            Dictionary with pricing info, or None if not found
        """
        try:
            # Create a temporary registry instance to access cache
            registry = cls()
            cache = registry._get_cache()
            
            # Try to load cached models
            cached_models = cache.load_cache()
            if not cached_models:
                return None
            
            # Find the specific model by ID
            for model in cached_models:
                if model.get('id') == model_id:
                    pricing = model.get('pricing', {})
                    if pricing and isinstance(pricing, dict):
                        return pricing
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to get pricing for model {model_id}: {e}")
            return None

    @classmethod
    def get_model_summary(cls) -> Dict[str, Any]:
        """Get summary statistics about available static models."""
        providers = set(config.provider for config in cls.MODELS.values())
        total_models = len(cls.MODELS)
        
        cost_ranges = {
            'min_input_cost': min(config.input_cost_per_1m_tokens for config in cls.MODELS.values()),
            'max_input_cost': max(config.input_cost_per_1m_tokens for config in cls.MODELS.values()),
            'min_output_cost': min(config.output_cost_per_1m_tokens for config in cls.MODELS.values()),
            'max_output_cost': max(config.output_cost_per_1m_tokens for config in cls.MODELS.values()),
        }
        
        context_ranges = {
            'min_context': min(config.context_window for config in cls.MODELS.values()),
            'max_context': max(config.context_window for config in cls.MODELS.values()),
        }
        
        return {
            'total_models': total_models,
            'providers': list(providers),
            'cost_ranges': cost_ranges,
            'context_ranges': context_ranges,
            'streaming_support': len([c for c in cls.MODELS.values() if c.supports_streaming])
        }


# Create a global registry instance
model_registry = ModelRegistry()


# Convenience functions for async operations
async def fetch_available_models() -> List[Dict[str, Any]]:
    """Convenience function to fetch available models."""
    return await model_registry.get_available_models()


async def search_available_models(query: str) -> List[Dict[str, Any]]:
    """Convenience function to search available models."""
    models = await model_registry.get_available_models()
    return model_registry.search_models(query, models)


def get_default_model() -> str:
    """Get the default model ID."""
    return DEFAULT_MODEL