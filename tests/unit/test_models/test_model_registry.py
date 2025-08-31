"""
Tests for Model Registry
"""

import pytest
from src.models.model_registry import ModelRegistry, ModelConfig, ModelProvider


class TestModelRegistry:
    """Test cases for ModelRegistry class."""
    
    def test_list_available_models(self):
        """Test listing available models."""
        models = ModelRegistry.list_available_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert "openai/gpt-3.5-turbo" in models
        assert "anthropic/claude-3-haiku" in models
    
    def test_get_model_config(self):
        """Test getting model configuration."""
        # Test existing model
        config = ModelRegistry.get_model_config("openai/gpt-3.5-turbo")
        
        assert config is not None
        assert isinstance(config, ModelConfig)
        assert config.model_id == "openai/gpt-3.5-turbo"
        assert config.display_name == "GPT-3.5 Turbo"
        assert config.provider == ModelProvider.OPENAI
        assert config.context_window == 4096
        assert config.input_cost_per_1m_tokens == 0.5
        assert config.output_cost_per_1m_tokens == 1.5
        
        # Test non-existing model
        config = ModelRegistry.get_model_config("non-existing-model")
        assert config is None
    
    def test_validate_model_availability(self):
        """Test model availability validation."""
        # Test existing model
        assert ModelRegistry.validate_model_availability("openai/gpt-3.5-turbo") == True
        
        # Test non-existing model
        assert ModelRegistry.validate_model_availability("non-existing-model") == False
    
    def test_get_models_by_provider(self):
        """Test getting models by provider."""
        openai_models = ModelRegistry.get_models_by_provider(ModelProvider.OPENAI)
        
        assert isinstance(openai_models, list)
        assert len(openai_models) > 0
        
        # All models should be from OpenAI
        for model in openai_models:
            assert model.provider == ModelProvider.OPENAI
        
        # Test with Anthropic
        anthropic_models = ModelRegistry.get_models_by_provider(ModelProvider.ANTHROPIC)
        assert len(anthropic_models) > 0
        
        for model in anthropic_models:
            assert model.provider == ModelProvider.ANTHROPIC
    
    def test_get_models_by_capability(self):
        """Test getting models by capability."""
        chat_models = ModelRegistry.get_models_by_capability("chat")
        
        assert isinstance(chat_models, list)
        assert len(chat_models) > 0
        
        # All models should have chat capability
        for model in chat_models:
            assert "chat" in model.capabilities
        
        # Test with non-existing capability
        fake_models = ModelRegistry.get_models_by_capability("fake-capability")
        assert len(fake_models) == 0
    
    def test_get_cheapest_models(self):
        """Test getting cheapest models."""
        cheapest = ModelRegistry.get_cheapest_models(limit=3)
        
        assert isinstance(cheapest, list)
        assert len(cheapest) <= 3
        
        # Should be sorted by combined cost (ascending)
        if len(cheapest) > 1:
            for i in range(len(cheapest) - 1):
                cost1 = cheapest[i].input_cost_per_1m_tokens + cheapest[i].output_cost_per_1m_tokens
                cost2 = cheapest[i + 1].input_cost_per_1m_tokens + cheapest[i + 1].output_cost_per_1m_tokens
                assert cost1 <= cost2
    
    def test_get_models_with_long_context(self):
        """Test getting models with long context windows."""
        long_context = ModelRegistry.get_models_with_long_context(min_context=100000)
        
        assert isinstance(long_context, list)
        
        # All models should have context window >= 100000
        for model in long_context:
            assert model.context_window >= 100000
    
    def test_estimate_cost(self):
        """Test cost estimation."""
        # Test with existing model
        cost = ModelRegistry.estimate_cost("openai/gpt-3.5-turbo", 1000, 500)
        
        assert isinstance(cost, float)
        assert cost > 0
        
        # Manual calculation check
        expected_cost = (1000 / 1_000_000) * 0.5 + (500 / 1_000_000) * 1.5
        assert abs(cost - expected_cost) < 0.0001
        
        # Test with non-existing model
        cost = ModelRegistry.estimate_cost("non-existing-model", 1000, 500)
        assert cost == 0.0
    
    def test_get_model_summary(self):
        """Test getting model summary statistics."""
        summary = ModelRegistry.get_model_summary()
        
        assert isinstance(summary, dict)
        assert "total_models" in summary
        assert "providers" in summary
        assert "cost_ranges" in summary
        assert "context_ranges" in summary
        assert "streaming_support" in summary
        
        assert summary["total_models"] > 0
        assert isinstance(summary["providers"], list)
        assert len(summary["providers"]) > 0
        
        # Check cost ranges
        cost_ranges = summary["cost_ranges"]
        assert "min_input_cost" in cost_ranges
        assert "max_input_cost" in cost_ranges
        assert cost_ranges["min_input_cost"] <= cost_ranges["max_input_cost"]
        
        # Check context ranges
        context_ranges = summary["context_ranges"]
        assert "min_context" in context_ranges
        assert "max_context" in context_ranges
        assert context_ranges["min_context"] <= context_ranges["max_context"]


class TestModelConfig:
    """Test cases for ModelConfig class."""
    
    def test_model_config_creation(self):
        """Test creating a ModelConfig."""
        config = ModelConfig(
            model_id="test/model",
            display_name="Test Model",
            provider=ModelProvider.OPENAI,
            context_window=4096,
            input_cost_per_1m_tokens=1.0,
            output_cost_per_1m_tokens=2.0
        )
        
        assert config.model_id == "test/model"
        assert config.display_name == "Test Model"
        assert config.provider == ModelProvider.OPENAI
        assert config.context_window == 4096
        assert config.input_cost_per_1m_tokens == 1.0
        assert config.output_cost_per_1m_tokens == 2.0
        assert config.supports_streaming == True  # Default
        assert config.capabilities == []  # Default empty list after __post_init__
    
    def test_model_config_with_capabilities(self):
        """Test creating a ModelConfig with capabilities."""
        capabilities = ["chat", "reasoning", "analysis"]
        
        config = ModelConfig(
            model_id="test/model",
            display_name="Test Model",
            provider=ModelProvider.ANTHROPIC,
            context_window=8192,
            input_cost_per_1m_tokens=3.0,
            output_cost_per_1m_tokens=6.0,
            capabilities=capabilities
        )
        
        assert config.capabilities == capabilities
        assert "chat" in config.capabilities
        assert "reasoning" in config.capabilities