"""
Unit tests for model registry functionality.
"""

from unittest.mock import AsyncMock, Mock, patch, MagicMock

from src.models.model_registry import (
    ModelRegistry, 
    ModelConfig, 
    ModelProvider,
    get_default_model,
    fetch_available_models,
    search_available_models
)


class TestModelConfig:
    """Test cases for ModelConfig dataclass."""

    def test_model_config_creation(self):
        """Test basic ModelConfig creation."""
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
        assert config.supports_streaming is True
        assert config.max_tokens_default == 150
        assert config.temperature_default == 0.1
        assert config.capabilities == []

    def test_model_config_with_capabilities(self):
        """Test ModelConfig with custom capabilities."""
        config = ModelConfig(
            model_id="test/model",
            display_name="Test Model",
            provider=ModelProvider.ANTHROPIC,
            context_window=8192,
            input_cost_per_1m_tokens=1.0,
            output_cost_per_1m_tokens=2.0,
            capabilities=["chat", "reasoning"]
        )
        
        assert config.capabilities == ["chat", "reasoning"]

    def test_model_config_post_init(self):
        """Test ModelConfig post_init behavior."""
        # Test with None capabilities
        config = ModelConfig(
            model_id="test/model",
            display_name="Test Model",
            provider=ModelProvider.GOOGLE,
            context_window=4096,
            input_cost_per_1m_tokens=1.0,
            output_cost_per_1m_tokens=2.0,
            capabilities=None
        )
        
        assert config.capabilities == []


class TestModelRegistry:
    """Test cases for ModelRegistry class."""

    def get_registry(self):
        """Helper method to get a ModelRegistry instance."""
        with patch('src.models.model_registry.get_config') as mock_config:
            mock_config.return_value = Mock()
            return ModelRegistry()

    def get_sample_api_models(self):
        """Helper method providing sample API model data."""
        return [
            {
                'id': 'openai/gpt-4',
                'name': 'GPT-4',
                'description': 'OpenAI GPT-4 model',
                'provider': 'openai',
                'context_length': 8192,
                'pricing': {
                    'input_cost_per_1m_tokens': 30.0,
                    'output_cost_per_1m_tokens': 60.0
                },
                'capabilities': ['chat', 'reasoning'],
                'available': True
            },
            {
                'id': 'anthropic/claude-3-sonnet',
                'name': 'Claude 3 Sonnet',
                'description': 'Anthropic Claude 3 Sonnet',
                'provider': 'anthropic',
                'context_length': 200000,
                'pricing': {
                    'input_cost_per_1m_tokens': 3.0,
                    'output_cost_per_1m_tokens': 15.0
                },
                'capabilities': ['chat', 'reasoning', 'creative'],
                'available': True
            }
        ]

    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = self.get_registry()
        
        assert registry._config is not None
        assert registry._cache is None
        assert registry._cached_models is None

    def test_get_cache_default(self):
        """Test _get_cache method with default config."""
        registry = self.get_registry()
        cache = registry._get_cache()
        assert cache is not None

    def test_get_cache_with_config(self):
        """Test _get_cache method with custom config."""
        registry = self.get_registry()
        
        mock_cache_config = Mock()
        mock_cache_config.path = 'custom/path.json'
        mock_cache_config.ttl_seconds = 7200
        
        registry._config.model_cache = mock_cache_config
        
        with patch('src.models.model_registry.get_model_cache') as mock_get_cache:
            mock_get_cache.return_value = Mock()
            cache = registry._get_cache()
            
            mock_get_cache.assert_called_once_with('custom/path.json', 7200)

    def test_fetch_models_success(self):
        """Test successful model fetching from API."""
        registry = self.get_registry()
        sample_api_models = self.get_sample_api_models()
        
        async def run_test():
            with patch('src.models.model_registry.OpenRouterClient') as mock_client_class:
                mock_client = AsyncMock()
                mock_client.list_available_models.return_value = sample_api_models
                mock_client_class.return_value = mock_client
                
                with patch.object(registry, '_get_cache') as mock_get_cache:
                    mock_cache = Mock()
                    mock_cache.save_cache.return_value = True
                    mock_get_cache.return_value = mock_cache
                    
                    result = await registry.fetch_models()
                    
                    assert result == sample_api_models
                    mock_cache.save_cache.assert_called_once_with(sample_api_models)
                    mock_client.close.assert_called_once()
        
        # Run the test
        import asyncio
        asyncio.run(run_test())

    def test_fetch_models_api_failure(self):
        """Test model fetching when API fails."""
        registry = self.get_registry()
        
        async def run_test():
            with patch('src.models.model_registry.OpenRouterClient') as mock_client_class:
                mock_client = AsyncMock()
                mock_client.list_available_models.side_effect = Exception("API Error")
                mock_client_class.return_value = mock_client
                
                result = await registry.fetch_models()
                
                assert result is None
                mock_client.close.assert_called_once()
        
        import asyncio
        asyncio.run(run_test())

    def test_fetch_models_no_results(self):
        """Test model fetching when API returns no models."""
        registry = self.get_registry()
        
        async def run_test():
            with patch('src.models.model_registry.OpenRouterClient') as mock_client_class:
                mock_client = AsyncMock()
                mock_client.list_available_models.return_value = None
                mock_client_class.return_value = mock_client
                
                result = await registry.fetch_models()
                
                assert result is None
        
        import asyncio
        asyncio.run(run_test())

    def test_get_available_models_api_success(self):
        """Test get_available_models with successful API fetch."""
        registry = self.get_registry()
        sample_api_models = self.get_sample_api_models()
        
        async def run_test():
            with patch.object(registry, 'fetch_models') as mock_fetch:
                mock_fetch.return_value = sample_api_models
                
                result = await registry.get_available_models()
                
                assert result == sample_api_models
        import asyncio
        asyncio.run(run_test())

    def test_get_available_models_cache_fallback(self):
        """Test get_available_models falling back to cache."""
        registry = self.get_registry()
        sample_api_models = self.get_sample_api_models()
        
        async def run_test():
            with patch.object(registry, 'fetch_models') as mock_fetch:
                mock_fetch.return_value = None
                
                with patch.object(registry, '_get_cache') as mock_get_cache:
                    mock_cache = Mock()
                    mock_cache.load_cache.return_value = sample_api_models
                    mock_get_cache.return_value = mock_cache
                    
                    result = await registry.get_available_models()
                    
                    assert result == sample_api_models
        
        import asyncio
        asyncio.run(run_test())

    def test_get_available_models_static_fallback(self):
        """Test get_available_models falling back to static models."""
        registry = self.get_registry()
        
        async def run_test():
            with patch.object(registry, 'fetch_models') as mock_fetch:
                mock_fetch.return_value = None
                
                with patch.object(registry, '_get_cache') as mock_get_cache:
                    mock_cache = Mock()
                    mock_cache.load_cache.return_value = None
                    mock_get_cache.return_value = mock_cache
                    
                    result = await registry.get_available_models()
                    
                    # Should return static models converted to API format
                    assert result is not None
                    assert len(result) > 0
                    assert all(isinstance(model, dict) for model in result)
                    assert all('id' in model for model in result)
        
        import asyncio
        asyncio.run(run_test())

    def test_convert_static_models_to_api_format(self):
        """Test conversion of static models to API format."""
        registry = self.get_registry()
        result = registry._convert_static_models_to_api_format()
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Check first model has expected structure
        model = result[0]
        required_fields = ['id', 'name', 'description', 'provider', 'context_length', 
                          'pricing', 'capabilities', 'available']
        assert all(field in model for field in required_fields)
        
        # Check pricing structure
        assert 'input_cost_per_1m_tokens' in model['pricing']
        assert 'output_cost_per_1m_tokens' in model['pricing']

    def test_search_models_with_models(self):
        """Test search_models with provided models."""
        registry = self.get_registry()
        sample_api_models = self.get_sample_api_models()
        
        # Test searching by provider
        results = registry.search_models("openai", sample_api_models)
        assert len(results) == 1
        assert results[0]['id'] == 'openai/gpt-4'
        
        # Test searching by name
        results = registry.search_models("claude", sample_api_models)
        assert len(results) == 1
        assert results[0]['id'] == 'anthropic/claude-3-sonnet'
        
        # Test case insensitive search
        results = registry.search_models("CLAUDE", sample_api_models)
        assert len(results) == 1
        
        # Test searching by capability
        results = registry.search_models("creative", sample_api_models)
        assert len(results) == 1
        assert results[0]['id'] == 'anthropic/claude-3-sonnet'

    def test_search_models_no_query(self):
        """Test search_models with empty query returns all models."""
        registry = self.get_registry()
        sample_api_models = self.get_sample_api_models()
        
        results = registry.search_models("", sample_api_models)
        assert len(results) == len(sample_api_models)

    def test_search_models_no_matches(self):
        """Test search_models with no matches."""
        registry = self.get_registry()
        sample_api_models = self.get_sample_api_models()
        
        results = registry.search_models("nonexistent-model", sample_api_models)
        assert len(results) == 0

    def test_list_available_models_static(self):
        """Test list_available_models class method."""
        models = ModelRegistry.list_available_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert 'openai/gpt-3.5-turbo' in models
        assert 'anthropic/claude-3.5-sonnet' in models

    def test_get_model_config_existing(self):
        """Test get_model_config for existing model."""
        config = ModelRegistry.get_model_config('openai/gpt-4')
        assert config is not None
        assert config.model_id == 'openai/gpt-4'
        assert config.display_name == 'GPT-4'
        assert config.provider == ModelProvider.OPENAI

    def test_get_model_config_nonexistent(self):
        """Test get_model_config for nonexistent model."""
        config = ModelRegistry.get_model_config('nonexistent/model')
        assert config is None

    def test_validate_model_availability_existing(self):
        """Test validate_model_availability for existing model."""
        result = ModelRegistry.validate_model_availability('openai/gpt-4')
        assert result is True

    def test_validate_model_availability_nonexistent(self):
        """Test validate_model_availability for nonexistent model."""
        result = ModelRegistry.validate_model_availability('nonexistent/model')
        assert result is False

    def test_get_models_by_provider(self):
        """Test get_models_by_provider."""
        openai_models = ModelRegistry.get_models_by_provider(ModelProvider.OPENAI)
        assert len(openai_models) > 0
        assert all(model.provider == ModelProvider.OPENAI for model in openai_models)
        
        anthropic_models = ModelRegistry.get_models_by_provider(ModelProvider.ANTHROPIC)
        assert len(anthropic_models) > 0
        assert all(model.provider == ModelProvider.ANTHROPIC for model in anthropic_models)

    def test_get_models_by_capability(self):
        """Test get_models_by_capability."""
        chat_models = ModelRegistry.get_models_by_capability("chat")
        assert len(chat_models) > 0
        assert all(model.capabilities is not None and "chat" in model.capabilities for model in chat_models)
        
        reasoning_models = ModelRegistry.get_models_by_capability("reasoning")
        assert len(reasoning_models) > 0
        assert all(model.capabilities is not None and "reasoning" in model.capabilities for model in reasoning_models)

    def test_get_cheapest_models(self):
        """Test get_cheapest_models."""
        cheapest = ModelRegistry.get_cheapest_models(3)
        assert len(cheapest) <= 3
        
        # Should be sorted by combined cost
        if len(cheapest) > 1:
            for i in range(len(cheapest) - 1):
                cost1 = cheapest[i].input_cost_per_1m_tokens + cheapest[i].output_cost_per_1m_tokens
                cost2 = cheapest[i+1].input_cost_per_1m_tokens + cheapest[i+1].output_cost_per_1m_tokens
                assert cost1 <= cost2

    def test_get_models_with_long_context(self):
        """Test get_models_with_long_context."""
        long_context_models = ModelRegistry.get_models_with_long_context(100000)
        assert len(long_context_models) > 0
        assert all(model.context_window >= 100000 for model in long_context_models)

    def test_estimate_cost(self):
        """Test estimate_cost method."""
        # Test with existing model
        cost = ModelRegistry.estimate_cost('openai/gpt-4', 1000000, 500000)
        assert cost > 0
        
        # Test with nonexistent model
        cost = ModelRegistry.estimate_cost('nonexistent/model', 1000000, 500000)
        assert cost == 0.0

    def test_get_model_summary(self):
        """Test get_model_summary method."""
        summary = ModelRegistry.get_model_summary()
        
        assert 'total_models' in summary
        assert 'providers' in summary
        assert 'cost_ranges' in summary
        assert 'context_ranges' in summary
        assert 'streaming_support' in summary
        
        assert summary['total_models'] > 0
        assert len(summary['providers']) > 0
        assert summary['cost_ranges']['min_input_cost'] >= 0
        assert summary['cost_ranges']['max_input_cost'] >= summary['cost_ranges']['min_input_cost']
        assert summary['streaming_support'] >= 0


class TestModuleFunctions:
    """Test module-level functions."""

    def test_get_default_model(self):
        """Test get_default_model function."""
        default = get_default_model()
        assert default == "anthropic/claude-3.5-sonnet"

    def test_fetch_available_models(self):
        """Test fetch_available_models convenience function."""
        async def run_test():
            with patch('src.models.model_registry.model_registry') as mock_registry:
                mock_registry.get_available_models = AsyncMock()
                mock_registry.get_available_models.return_value = [{'id': 'test/model'}]
                
                result = await fetch_available_models()
                assert result == [{'id': 'test/model'}]
                mock_registry.get_available_models.assert_called_once()
        
        import asyncio
        asyncio.run(run_test())

    def test_search_available_models(self):
        """Test search_available_models convenience function."""
        async def run_test():
            with patch('src.models.model_registry.model_registry') as mock_registry:
                mock_registry.get_available_models = AsyncMock()
                mock_registry.search_models = Mock()
                
                mock_registry.get_available_models.return_value = [{'id': 'test/model'}]
                mock_registry.search_models.return_value = [{'id': 'test/model'}]
                
                result = await search_available_models("test")
                
                assert result == [{'id': 'test/model'}]
                mock_registry.get_available_models.assert_called_once()
                mock_registry.search_models.assert_called_once_with("test", [{'id': 'test/model'}])
        
        import asyncio
        asyncio.run(run_test())


class TestModelProvider:
    """Test ModelProvider enum."""

    def test_model_provider_values(self):
        """Test ModelProvider enum values."""
        assert ModelProvider.OPENAI == "openai"
        assert ModelProvider.ANTHROPIC == "anthropic"
        assert ModelProvider.META == "meta"
        assert ModelProvider.MISTRAL == "mistral"
        assert ModelProvider.GOOGLE == "google"

    def test_model_provider_iteration(self):
        """Test iterating over ModelProvider values."""
        providers = list(ModelProvider)
        assert len(providers) == 5
        assert ModelProvider.OPENAI in providers
        assert ModelProvider.ANTHROPIC in providers


# Integration test scenarios
class TestRegistryIntegration:
    """Integration tests for model registry functionality."""

    def test_registry_with_real_static_models(self):
        """Test registry operations with actual static model data."""
        registry = ModelRegistry()
        
        # Test static model access
        static_models = registry._convert_static_models_to_api_format()
        assert len(static_models) > 10  # Should have several static models
        
        # Test searching static models
        claude_models = registry.search_models("claude", static_models)
        assert len(claude_models) > 0
        
        gpt_models = registry.search_models("gpt", static_models)
        assert len(gpt_models) > 0

    def test_default_model_exists_in_static(self):
        """Test that default model exists in static model registry."""
        default_model = get_default_model()
        registry = ModelRegistry()
        static_models = registry._convert_static_models_to_api_format()
        
        default_found = any(model['id'] == default_model for model in static_models)
        assert default_found, f"Default model {default_model} not found in static models"