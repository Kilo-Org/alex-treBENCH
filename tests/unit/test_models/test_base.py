"""
Tests for Base Model Classes
"""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from src.models.base import ModelResponse, ModelConfig, ModelAdapter


class TestModelResponse:
    """Test cases for ModelResponse class."""
    
    def test_model_response_creation(self):
        """Test creating a ModelResponse."""
        timestamp = datetime.now()
        metadata = {"tokens_input": 100, "tokens_output": 50}
        
        response = ModelResponse(
            model_id="openai/gpt-3.5-turbo",
            prompt="Test question",
            response="Test answer",
            latency_ms=250.5,
            tokens_used=150,
            cost=0.001,
            timestamp=timestamp,
            metadata=metadata
        )
        
        assert response.model_id == "openai/gpt-3.5-turbo"
        assert response.prompt == "Test question"
        assert response.response == "Test answer"
        assert response.latency_ms == 250.5
        assert response.tokens_used == 150
        assert response.cost == 0.001
        assert response.timestamp == timestamp
        assert response.metadata == metadata
    
    def test_model_response_post_init_timestamp(self):
        """Test that __post_init__ sets timestamp if None."""
        response = ModelResponse(
            model_id="test/model",
            prompt="test",
            response="test",
            latency_ms=100.0,
            tokens_used=50,
            cost=0.001,
            timestamp=None,  # Should be set by __post_init__
            metadata={}
        )
        
        assert response.timestamp is not None
        assert isinstance(response.timestamp, datetime)
    
    def test_model_response_post_init_metadata(self):
        """Test that __post_init__ sets metadata if None."""
        response = ModelResponse(
            model_id="test/model",
            prompt="test",
            response="test",
            latency_ms=100.0,
            tokens_used=50,
            cost=0.001,
            timestamp=datetime.now(),
            metadata=None  # Should be set by __post_init__
        )
        
        assert response.metadata == {}


class TestModelConfig:
    """Test cases for ModelConfig class."""
    
    def test_model_config_creation(self):
        """Test creating a ModelConfig."""
        config = ModelConfig(
            model_name="openai/gpt-3.5-turbo",
            max_tokens=200,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            stop_sequences=["<|endoftext|>"],
            timeout_seconds=45
        )
        
        assert config.model_name == "openai/gpt-3.5-turbo"
        assert config.max_tokens == 200
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.frequency_penalty == 0.1
        assert config.presence_penalty == 0.2
        assert config.stop_sequences == ["<|endoftext|>"]
        assert config.timeout_seconds == 45
    
    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig(model_name="test/model")
        
        assert config.model_name == "test/model"
        assert config.max_tokens == 150
        assert config.temperature == 0.1
        assert config.top_p is None
        assert config.frequency_penalty is None
        assert config.presence_penalty is None
        assert config.stop_sequences is None
        assert config.timeout_seconds == 30


class MockModelAdapter(ModelAdapter):
    """Mock implementation of ModelAdapter for testing."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.available = True
        self.pricing = {"input": 1.0, "output": 2.0}
    
    async def query(self, prompt: str, **kwargs) -> ModelResponse:
        """Mock query implementation."""
        return ModelResponse(
            model_id=self.config.model_name,
            prompt=prompt,
            response="Mock response",
            latency_ms=100.0,
            tokens_used=50,
            cost=0.001,
            timestamp=datetime.now(),
            metadata={"mock": True}
        )
    
    async def batch_query(self, prompts, **kwargs):
        """Mock batch query implementation."""
        return [await self.query(prompt, **kwargs) for prompt in prompts]
    
    def is_available(self) -> bool:
        """Mock availability check."""
        return self.available
    
    def get_pricing_info(self):
        """Mock pricing info."""
        return self.pricing


class TestModelAdapter:
    """Test cases for ModelAdapter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ModelConfig(
            model_name="test/model",
            max_tokens=100,
            temperature=0.2
        )
        self.adapter = MockModelAdapter(self.config)
    
    def test_adapter_initialization(self):
        """Test adapter initialization."""
        assert self.adapter.config == self.config
        assert self.adapter._total_requests == 0
        assert self.adapter._total_tokens == 0
        assert self.adapter._total_cost == 0.0
    
    def test_format_jeopardy_prompt(self):
        """Test Jeopardy prompt formatting."""
        question = "This element has the chemical symbol 'Au'"
        formatted = self.adapter.format_jeopardy_prompt(question)
        
        assert isinstance(formatted, str)
        assert "Jeopardy!" in formatted
        assert "form of a question" in formatted
        assert question in formatted
        assert "Answer:" in formatted
    
    def test_extract_answer_jeopardy_format(self):
        """Test extracting Jeopardy-format answers."""
        test_cases = [
            ("What is gold?", "What is gold?"),
            ("Who is Shakespeare?", "Who is Shakespeare?"),
            ("What is the capital of France? It's a beautiful city.", "What is the capital of France?"),
            ("Where was the battle fought?\nIt was a historic event.", "Where was the battle fought?"),
        ]
        
        for response_text, expected in test_cases:
            extracted = self.adapter.extract_answer(response_text)
            assert expected.lower() in extracted.lower()
    
    def test_extract_answer_non_jeopardy_format(self):
        """Test extracting non-Jeopardy format answers."""
        test_cases = [
            ("Gold", "Gold"),
            ("The answer is Paris.", "The answer is Paris"),
            ("Shakespeare wrote many plays.\nHe was a great playwright.", "Shakespeare wrote many plays"),
        ]
        
        for response_text, expected_substring in test_cases:
            extracted = self.adapter.extract_answer(response_text)
            assert expected_substring.lower() in extracted.lower()
    
    def test_update_usage_stats(self):
        """Test updating usage statistics."""
        response = ModelResponse(
            model_id="test/model",
            prompt="test",
            response="test",
            latency_ms=100.0,
            tokens_used=50,
            cost=0.001,
            timestamp=datetime.now(),
            metadata={}
        )
        
        # Initial stats
        assert self.adapter._total_requests == 0
        assert self.adapter._total_tokens == 0
        assert self.adapter._total_cost == 0.0
        
        # Update stats
        self.adapter.update_usage_stats(response)
        
        assert self.adapter._total_requests == 1
        assert self.adapter._total_tokens == 50
        assert self.adapter._total_cost == 0.001
        
        # Update again
        self.adapter.update_usage_stats(response)
        
        assert self.adapter._total_requests == 2
        assert self.adapter._total_tokens == 100
        assert self.adapter._total_cost == 0.002
    
    def test_get_usage_stats(self):
        """Test getting usage statistics."""
        # Add some usage
        response = ModelResponse(
            model_id="test/model",
            prompt="test",
            response="test",
            latency_ms=100.0,
            tokens_used=50,
            cost=0.001,
            timestamp=datetime.now(),
            metadata={}
        )
        
        self.adapter.update_usage_stats(response)
        self.adapter.update_usage_stats(response)  # Add twice
        
        stats = self.adapter.get_usage_stats()
        
        assert isinstance(stats, dict)
        assert stats['total_requests'] == 2
        assert stats['total_tokens'] == 100
        assert stats['total_cost_usd'] == 0.002
        assert stats['average_cost_per_request'] == 0.001
        assert stats['average_tokens_per_request'] == 50.0
    
    def test_reset_usage_stats(self):
        """Test resetting usage statistics."""
        # Add some usage first
        response = ModelResponse(
            model_id="test/model",
            prompt="test",
            response="test",
            latency_ms=100.0,
            tokens_used=50,
            cost=0.001,
            timestamp=datetime.now(),
            metadata={}
        )
        self.adapter.update_usage_stats(response)
        
        assert self.adapter._total_requests == 1
        
        # Reset
        self.adapter.reset_usage_stats()
        
        assert self.adapter._total_requests == 0
        assert self.adapter._total_tokens == 0
        assert self.adapter._total_cost == 0.0
    
    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        health = await self.adapter.health_check()
        
        assert isinstance(health, dict)
        assert health['status'] == 'healthy'
        assert 'response_time_ms' in health
        assert health['model_name'] == 'test/model'
        assert 'timestamp' in health
        assert health['response_time_ms'] > 0
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health check when query fails."""
        # Make the adapter unavailable to trigger failure
        self.adapter.available = False
        
        # Mock the query to raise an exception
        original_query = self.adapter.query
        async def failing_query(*args, **kwargs):
            raise Exception("Connection failed")
        
        self.adapter.query = failing_query
        
        health = await self.adapter.health_check()
        
        assert health['status'] == 'unhealthy'
        assert 'error' in health
        assert health['model_name'] == 'test/model'
        assert 'timestamp' in health
        
        # Restore original query
        self.adapter.query = original_query
    
    def test_repr(self):
        """Test string representation of adapter."""
        repr_str = repr(self.adapter)
        
        assert "MockModelAdapter" in repr_str
        assert "test/model" in repr_str
    
    @pytest.mark.asyncio
    async def test_query_abstract_method(self):
        """Test that query is properly implemented."""
        result = await self.adapter.query("Test prompt")
        
        assert isinstance(result, ModelResponse)
        assert result.prompt == "Test prompt"
        assert result.response == "Mock response"
    
    @pytest.mark.asyncio
    async def test_batch_query_abstract_method(self):
        """Test that batch_query is properly implemented."""
        prompts = ["Prompt 1", "Prompt 2"]
        results = await self.adapter.batch_query(prompts)
        
        assert len(results) == 2
        assert all(isinstance(r, ModelResponse) for r in results)
        assert results[0].prompt == "Prompt 1"
        assert results[1].prompt == "Prompt 2"
    
    def test_is_available_abstract_method(self):
        """Test that is_available is properly implemented."""
        assert self.adapter.is_available() == True
        
        self.adapter.available = False
        assert self.adapter.is_available() == False
    
    def test_get_pricing_info_abstract_method(self):
        """Test that get_pricing_info is properly implemented."""
        pricing = self.adapter.get_pricing_info()
        
        assert isinstance(pricing, dict)
        assert pricing == {"input": 1.0, "output": 2.0}


class TestModelRegistry:
    """Test cases for ModelRegistry class from base.py."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from src.models.base import ModelRegistry
        self.registry = ModelRegistry()
    
    def test_registry_initialization(self):
        """Test registry initialization."""
        assert hasattr(self.registry, '_adapters')
        assert isinstance(self.registry._adapters, dict)
        assert len(self.registry._adapters) == 0
    
    def test_register_adapter(self):
        """Test registering an adapter."""
        config = ModelConfig(model_name="test/model")
        adapter = MockModelAdapter(config)
        
        self.registry.register("test-adapter", adapter)
        
        assert "test-adapter" in self.registry._adapters
        assert self.registry._adapters["test-adapter"] is adapter
    
    def test_get_adapter(self):
        """Test getting a registered adapter."""
        config = ModelConfig(model_name="test/model")
        adapter = MockModelAdapter(config)
        
        self.registry.register("test-adapter", adapter)
        
        retrieved = self.registry.get("test-adapter")
        assert retrieved is adapter
        
        # Test non-existing adapter
        non_existing = self.registry.get("non-existing")
        assert non_existing is None
    
    def test_list_models(self):
        """Test listing all registered models."""
        config1 = ModelConfig(model_name="model1")
        config2 = ModelConfig(model_name="model2")
        adapter1 = MockModelAdapter(config1)
        adapter2 = MockModelAdapter(config2)
        
        self.registry.register("adapter1", adapter1)
        self.registry.register("adapter2", adapter2)
        
        models = self.registry.list_models()
        
        assert isinstance(models, list)
        assert len(models) == 2
        assert "adapter1" in models
        assert "adapter2" in models
    
    def test_get_available_models(self):
        """Test getting available models."""
        config1 = ModelConfig(model_name="model1")
        config2 = ModelConfig(model_name="model2")
        adapter1 = MockModelAdapter(config1)
        adapter2 = MockModelAdapter(config2)
        
        # Make one adapter unavailable
        adapter2.available = False
        
        self.registry.register("adapter1", adapter1)
        self.registry.register("adapter2", adapter2)
        
        available = self.registry.get_available_models()
        
        assert isinstance(available, list)
        assert len(available) == 1
        assert "adapter1" in available
        assert "adapter2" not in available
    
    @pytest.mark.asyncio
    async def test_health_check_all(self):
        """Test health checking all registered models."""
        config1 = ModelConfig(model_name="model1")
        config2 = ModelConfig(model_name="model2")
        adapter1 = MockModelAdapter(config1)
        adapter2 = MockModelAdapter(config2)
        
        self.registry.register("adapter1", adapter1)
        self.registry.register("adapter2", adapter2)
        
        health_results = await self.registry.health_check_all()
        
        assert isinstance(health_results, dict)
        assert len(health_results) == 2
        assert "adapter1" in health_results
        assert "adapter2" in health_results
        
        assert health_results["adapter1"]["status"] == "healthy"
        assert health_results["adapter2"]["status"] == "healthy"


def test_global_model_registry():
    """Test that global model registry is available."""
    from src.models.base import model_registry
    
    # Should be a ModelRegistry instance
    assert hasattr(model_registry, 'register')
    assert hasattr(model_registry, 'get')
    assert hasattr(model_registry, 'list_models')
    assert callable(model_registry.register)
    assert callable(model_registry.get)
    assert callable(model_registry.list_models)


class TestAbstractMethods:
    """Test that abstract methods raise NotImplementedError when not overridden."""
    
    def test_model_adapter_abstract_methods(self):
        """Test that ModelAdapter abstract methods raise NotImplementedError."""
        config = ModelConfig(model_name="test/model")
        
        # Can't instantiate abstract class directly
        with pytest.raises(TypeError):
            ModelAdapter(config)


@pytest.mark.asyncio
async def test_integration_adapter_workflow():
    """Integration test for complete adapter workflow."""
    config = ModelConfig(
        model_name="test/model",
        max_tokens=150,
        temperature=0.1
    )
    
    adapter = MockModelAdapter(config)
    
    # Test single query
    response = await adapter.query("Test Jeopardy question")
    
    assert response.model_id == "test/model"
    assert response.prompt == "Test Jeopardy question"
    assert response.tokens_used > 0
    assert response.cost > 0
    
    # Test batch query
    prompts = ["Question 1", "Question 2", "Question 3"]
    batch_responses = await adapter.batch_query(prompts)
    
    assert len(batch_responses) == 3
    assert all(isinstance(r, ModelResponse) for r in batch_responses)
    
    # Test usage stats
    stats = adapter.get_usage_stats()
    assert stats['total_requests'] == 4  # 1 single + 3 batch
    assert stats['total_tokens'] > 0
    assert stats['total_cost_usd'] > 0
    
    # Test health check
    health = await adapter.health_check()
    assert health['status'] == 'healthy'
    
    # Test Jeopardy-specific features
    jeopardy_prompt = adapter.format_jeopardy_prompt("Test clue")
    assert "Jeopardy!" in jeopardy_prompt
    
    extracted = adapter.extract_answer("What is gold?")
    assert "gold" in extracted.lower()
    
    # Test availability and pricing
    assert adapter.is_available() == True
    pricing = adapter.get_pricing_info()
    assert "input" in pricing
    assert "output" in pricing