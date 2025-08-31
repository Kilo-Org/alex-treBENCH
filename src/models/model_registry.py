"""
Model Registry

Registry of available models with their configurations, pricing, and metadata.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


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
    capabilities: List[str] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []


class ModelRegistry:
    """Registry of available models with their configurations."""
    
    # Model configurations
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
    
    @classmethod
    def list_available_models(cls) -> List[str]:
        """List all available model IDs."""
        return list(cls.MODELS.keys())
    
    @classmethod
    def get_model_config(cls, model_id: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        return cls.MODELS.get(model_id)
    
    @classmethod
    def validate_model_availability(cls, model_id: str) -> bool:
        """Check if a model is available in the registry."""
        return model_id in cls.MODELS
    
    @classmethod
    def get_models_by_provider(cls, provider: ModelProvider) -> List[ModelConfig]:
        """Get all models from a specific provider."""
        return [config for config in cls.MODELS.values() if config.provider == provider]
    
    @classmethod
    def get_models_by_capability(cls, capability: str) -> List[ModelConfig]:
        """Get models that have a specific capability."""
        return [
            config for config in cls.MODELS.values()
            if capability in config.capabilities
        ]
    
    @classmethod
    def get_cheapest_models(cls, limit: int = 5) -> List[ModelConfig]:
        """Get the cheapest models by combined input/output cost."""
        models = list(cls.MODELS.values())
        models.sort(key=lambda x: (x.input_cost_per_1m_tokens + x.output_cost_per_1m_tokens))
        return models[:limit]
    
    @classmethod
    def get_models_with_long_context(cls, min_context: int = 32000) -> List[ModelConfig]:
        """Get models with context window larger than specified size."""
        return [
            config for config in cls.MODELS.values()
            if config.context_window >= min_context
        ]
    
    @classmethod
    def estimate_cost(cls, model_id: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a model given token counts."""
        config = cls.get_model_config(model_id)
        if not config:
            return 0.0
        
        input_cost = (input_tokens / 1_000_000) * config.input_cost_per_1m_tokens
        output_cost = (output_tokens / 1_000_000) * config.output_cost_per_1m_tokens
        
        return input_cost + output_cost
    
    @classmethod
    def get_model_summary(cls) -> Dict[str, Any]:
        """Get summary statistics about available models."""
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