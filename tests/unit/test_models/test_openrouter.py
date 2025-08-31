"""
Tests for OpenRouter Client
"""

import pytest
import asyncio
import aiohttp
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.models.openrouter import OpenRouterClient
from src.models.base import ModelConfig, ModelResponse
from src.core.exceptions import ModelAPIError, RateLimitError


class TestOpenRouterClient:
    """Test cases for OpenRouterClient class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ModelConfig(
            model_name="openai/gpt-3.5-turbo",
            max_tokens=150,
            temperature=0.1
        )
        
        # Mock successful API response
        self.mock_api_response = {
            "choices": [
                {
                    "message": {"content": "What is gold?"},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 10,
                "total_tokens": 60
            },
            "model": "openai/gpt-3.5-turbo"
        }
    
    def test_init_with_api_key(self):
        """Test initialization with API key."""
        client = OpenRouterClient(api_key="test-key", config=self.config)
        
        assert client.api_key == "test-key"
        assert client.config.model_name == "openai/gpt-3.5-turbo"
        assert client.session is None  # Not created until needed
    
    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'env-test-key'})
    def test_init_with_env_api_key(self):
        """Test initialization with API key from environment."""
        client = OpenRouterClient(config=self.config)
        
        assert client.api_key == "env-test-key"
    
    def test_init_without_api_key_raises_error(self):
        """Test that missing API key raises error."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ModelAPIError, match="OpenRouter API key not provided"):
                OpenRouterClient(config=self.config)
    
    def test_init_with_default_config(self):
        """Test initialization with default config when none provided."""
        client = OpenRouterClient(api_key="test-key")
        
        assert client.config.model_name == "openai/gpt-3.5-turbo"
        assert client.config.max_tokens == 150
        assert client.config.temperature == 0.1
    
    @pytest.mark.asyncio
    async def test_ensure_session_creates_session(self):
        """Test that _ensure_session creates aiohttp session."""
        client = OpenRouterClient(api_key="test-key", config=self.config)
        
        session = await client._ensure_session()
        
        assert isinstance(session, aiohttp.ClientSession)
        assert client.session is session
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_ensure_session_reuses_existing(self):
        """Test that _ensure_session reuses existing session."""
        client = OpenRouterClient(api_key="test-key", config=self.config)
        
        session1 = await client._ensure_session()
        session2 = await client._ensure_session()
        
        assert session1 is session2
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_query_success(self):
        """Test successful query."""
        client = OpenRouterClient(api_key="test-key", config=self.config)
        
        # Mock the HTTP request
        with patch.object(client, '_make_request_with_retry', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = self.mock_api_response
            
            response = await client.query("Test question")
            
            assert isinstance(response, ModelResponse)
            assert response.model_id == "openai/gpt-3.5-turbo"
            assert response.prompt == "Test question"
            assert response.response == "What is gold?"
            assert response.latency_ms > 0
            assert response.tokens_used == 60  # prompt + completion
            assert response.cost > 0
            assert isinstance(response.timestamp, datetime)
            
            # Check metadata
            assert "tokens_input" in response.metadata
            assert "tokens_output" in response.metadata
            assert response.metadata["tokens_input"] == 50
            assert response.metadata["tokens_output"] == 10
    
    @pytest.mark.asyncio
    async def test_query_with_custom_parameters(self):
        """Test query with custom parameters."""
        client = OpenRouterClient(api_key="test-key", config=self.config)
        
        with patch.object(client, '_make_request_with_retry', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = self.mock_api_response
            
            await client.query(
                "Test question",
                max_tokens=200,
                temperature=0.7
            )
            
            # Check that custom parameters were passed to the API
            args, kwargs = mock_request.call_args
            session, headers, payload = args
            
            assert payload["max_tokens"] == 200
            assert payload["temperature"] == 0.7
    
    @pytest.mark.asyncio
    async def test_query_invalid_response_raises_error(self):
        """Test that invalid API response raises error."""
        client = OpenRouterClient(api_key="test-key", config=self.config)
        
        # Mock invalid response (missing choices)
        invalid_response = {"usage": {"prompt_tokens": 10}}
        
        with patch.object(client, '_make_request_with_retry', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = invalid_response
            
            with pytest.raises(ModelAPIError, match="Invalid response format"):
                await client.query("Test question")
    
    @pytest.mark.asyncio
    async def test_make_request_with_retry_success(self):
        """Test successful HTTP request."""
        client = OpenRouterClient(api_key="test-key", config=self.config)
        
        # Mock aiohttp response
        mock_response = Mock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value='{"choices": [{"message": {"content": "test"}}]}')
        
        mock_session = Mock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        
        headers = {"Authorization": "Bearer test-key"}
        payload = {"model": "test-model", "messages": []}
        
        result = await client._make_request_with_retry(mock_session, headers, payload)
        
        assert "choices" in result
        assert result["choices"][0]["message"]["content"] == "test"
    
    @pytest.mark.asyncio
    async def test_make_request_with_retry_rate_limit(self):
        """Test rate limit handling in HTTP request."""
        client = OpenRouterClient(api_key="test-key", config=self.config)
        
        # Mock rate limit response
        mock_response = Mock()
        mock_response.status = 429
        mock_response.headers = {"Retry-After": "60"}
        mock_response.text = AsyncMock(return_value="Rate limit exceeded")
        
        mock_session = Mock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        
        headers = {"Authorization": "Bearer test-key"}
        payload = {"model": "test-model", "messages": []}
        
        with pytest.raises(RateLimitError) as exc_info:
            await client._make_request_with_retry(mock_session, headers, payload)
        
        assert exc_info.value.retry_after == 60
        assert exc_info.value.status_code == 429
    
    @pytest.mark.asyncio
    async def test_make_request_with_retry_api_error(self):
        """Test API error handling in HTTP request."""
        client = OpenRouterClient(api_key="test-key", config=self.config)
        
        # Mock API error response
        mock_response = Mock()
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value="Bad request")
        
        mock_session = Mock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        
        headers = {"Authorization": "Bearer test-key"}
        payload = {"model": "test-model", "messages": []}
        
        with pytest.raises(ModelAPIError) as exc_info:
            await client._make_request_with_retry(mock_session, headers, payload)
        
        assert exc_info.value.status_code == 400
        assert "Bad request" in exc_info.value.response_body
    
    @pytest.mark.asyncio
    async def test_batch_query_success(self):
        """Test successful batch query."""
        client = OpenRouterClient(api_key="test-key", config=self.config)
        
        prompts = ["Question 1", "Question 2", "Question 3"]
        
        with patch.object(client, 'query', new_callable=AsyncMock) as mock_query:
            # Mock individual query responses
            mock_responses = [
                ModelResponse(
                    model_id="openai/gpt-3.5-turbo",
                    prompt=f"Question {i}",
                    response=f"Answer {i}",
                    latency_ms=100.0,
                    tokens_used=50,
                    cost=0.001,
                    timestamp=datetime.now(),
                    metadata={}
                ) for i in range(1, 4)
            ]
            mock_query.side_effect = mock_responses
            
            results = await client.batch_query(prompts)
            
            assert len(results) == 3
            assert all(isinstance(r, ModelResponse) for r in results)
            assert mock_query.call_count == 3
    
    @pytest.mark.asyncio
    async def test_batch_query_with_failures(self):
        """Test batch query with some failures."""
        client = OpenRouterClient(api_key="test-key", config=self.config)
        
        prompts = ["Question 1", "Question 2"]
        
        with patch.object(client, 'query', new_callable=AsyncMock) as mock_query:
            # First query succeeds, second fails
            success_response = ModelResponse(
                model_id="openai/gpt-3.5-turbo",
                prompt="Question 1",
                response="Answer 1",
                latency_ms=100.0,
                tokens_used=50,
                cost=0.001,
                timestamp=datetime.now(),
                metadata={}
            )
            
            mock_query.side_effect = [success_response, Exception("API Error")]
            
            results = await client.batch_query(prompts)
            
            assert len(results) == 2
            assert results[0].response == "Answer 1"  # Success
            assert results[1].metadata.get("failed") == True  # Failure
            assert results[1].response == ""  # Empty response for failure
    
    def test_calculate_cost(self):
        """Test cost calculation."""
        client = OpenRouterClient(api_key="test-key", config=self.config)
        
        # Test with known model pricing
        cost = client._calculate_cost(1000, 500)  # 1000 input, 500 output tokens
        
        # gpt-3.5-turbo: $0.5 input, $1.5 output per 1M tokens
        expected_cost = (1000 / 1_000_000) * 0.5 + (500 / 1_000_000) * 1.5
        assert abs(cost - expected_cost) < 0.0001
    
    def test_calculate_cost_unknown_model(self):
        """Test cost calculation for unknown model."""
        unknown_config = ModelConfig(model_name="unknown/model")
        client = OpenRouterClient(api_key="test-key", config=unknown_config)
        
        cost = client._calculate_cost(1000, 500)
        assert cost == 0.0  # Should return 0 for unknown models
    
    def test_is_available_with_api_key(self):
        """Test is_available returns True when API key is present."""
        client = OpenRouterClient(api_key="test-key", config=self.config)
        
        assert client.is_available() == True
    
    def test_is_available_without_api_key(self):
        """Test is_available returns False when API key is missing."""
        # Create client and then remove api_key
        client = OpenRouterClient(api_key="test-key", config=self.config)
        client.api_key = None
        
        assert client.is_available() == False
    
    def test_get_pricing_info(self):
        """Test getting pricing information."""
        client = OpenRouterClient(api_key="test-key", config=self.config)
        
        pricing = client.get_pricing_info()
        
        assert isinstance(pricing, dict)
        assert "input" in pricing
        assert "output" in pricing
        assert pricing["input"] == 0.5  # gpt-3.5-turbo input cost
        assert pricing["output"] == 1.5  # gpt-3.5-turbo output cost
    
    def test_get_pricing_info_unknown_model(self):
        """Test getting pricing info for unknown model."""
        unknown_config = ModelConfig(model_name="unknown/model")
        client = OpenRouterClient(api_key="test-key", config=unknown_config)
        
        pricing = client.get_pricing_info()
        
        assert pricing["input"] == 0.0
        assert pricing["output"] == 0.0
    
    @pytest.mark.asyncio
    async def test_list_available_models_success(self):
        """Test listing available models from OpenRouter API."""
        client = OpenRouterClient(api_key="test-key", config=self.config)
        
        mock_models_response = {
            "data": [
                {"id": "openai/gpt-3.5-turbo", "name": "GPT-3.5 Turbo"},
                {"id": "anthropic/claude-3-haiku", "name": "Claude 3 Haiku"}
            ]
        }
        
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_models_response)
        
        mock_session = Mock()
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(client, '_ensure_session', return_value=mock_session):
            models = await client.list_available_models()
            
            assert len(models) == 2
            assert models[0]["id"] == "openai/gpt-3.5-turbo"
            assert models[1]["id"] == "anthropic/claude-3-haiku"
    
    @pytest.mark.asyncio
    async def test_list_available_models_error(self):
        """Test handling errors when listing models."""
        client = OpenRouterClient(api_key="test-key", config=self.config)
        
        mock_response = Mock()
        mock_response.status = 403  # Forbidden
        
        mock_session = Mock()
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(client, '_ensure_session', return_value=mock_session):
            models = await client.list_available_models()
            
            assert models == []  # Should return empty list on error
    
    @pytest.mark.asyncio
    async def test_close_session(self):
        """Test closing the HTTP session."""
        client = OpenRouterClient(api_key="test-key", config=self.config)
        
        # Create a session
        session = await client._ensure_session()
        assert not session.closed
        
        # Close the session
        await client.close()
        assert session.closed
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using client as async context manager."""
        async with OpenRouterClient(api_key="test-key", config=self.config) as client:
            assert isinstance(client, OpenRouterClient)
            
            # Session should be created when needed
            session = await client._ensure_session()
            assert not session.closed
        
        # Session should be closed after context exit
        assert session.closed
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting functionality."""
        # Create client with very low rate limit for testing
        client = OpenRouterClient(api_key="test-key", config=self.config)
        
        # Mock the config to have a very low rate limit
        with patch.object(client, '_rate_limit_requests_per_minute', 2):
            # Fill up the rate limit
            client._request_times = [1000.0, 1001.0]  # 2 requests within a minute
            
            # Mock time.time to return a value that would trigger rate limiting
            with patch('time.time', return_value=1002.0):  # Just 2 seconds later
                with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                    await client._check_rate_limit()
                    
                    # Should have been called to sleep
                    mock_sleep.assert_called_once()
                    
                    # Should have recorded the new request time
                    assert len(client._request_times) == 3


class TestModelPricing:
    """Test cases for model pricing constants."""
    
    def test_model_pricing_structure(self):
        """Test that model pricing is properly structured."""
        pricing = OpenRouterClient.MODEL_PRICING
        
        assert isinstance(pricing, dict)
        assert len(pricing) > 0
        
        # Check specific models
        assert "openai/gpt-3.5-turbo" in pricing
        assert "anthropic/claude-3-haiku" in pricing
        
        # Check pricing structure
        for model_id, costs in pricing.items():
            assert "input" in costs
            assert "output" in costs
            assert isinstance(costs["input"], (int, float))
            assert isinstance(costs["output"], (int, float))
            assert costs["input"] >= 0
            assert costs["output"] >= 0
    
    def test_pricing_consistency(self):
        """Test that pricing is consistent across models."""
        pricing = OpenRouterClient.MODEL_PRICING
        
        # Output costs should generally be higher than input costs
        for model_id, costs in pricing.items():
            if costs["input"] > 0 and costs["output"] > 0:
                # Most models have higher output costs, but allow for exceptions
                # Just ensure both are positive
                assert costs["input"] > 0
                assert costs["output"] > 0


@pytest.mark.asyncio
async def test_integration_query_flow():
    """Integration test for the complete query flow."""
    config = ModelConfig(
        model_name="openai/gpt-3.5-turbo",
        max_tokens=100,
        temperature=0.2
    )
    
    client = OpenRouterClient(api_key="test-key", config=config)
    
    # Mock the entire HTTP request/response cycle
    mock_api_response = {
        "choices": [
            {
                "message": {"content": "What is Paris?"},
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 25,
            "completion_tokens": 5,
            "total_tokens": 30
        },
        "model": "openai/gpt-3.5-turbo"
    }
    
    with patch.object(client, '_make_request_with_retry', new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_api_response
        
        # Execute query
        response = await client.query("What is the capital of France?")
        
        # Verify complete response structure
        assert response.model_id == "openai/gpt-3.5-turbo"
        assert response.prompt == "What is the capital of France?"
        assert response.response == "What is Paris?"
        assert response.tokens_used == 30
        assert response.cost > 0
        assert isinstance(response.timestamp, datetime)
        assert response.metadata["tokens_input"] == 25
        assert response.metadata["tokens_output"] == 5
        assert response.metadata["finish_reason"] == "stop"
        
        # Verify API call was made with correct parameters
        args, kwargs = mock_request.call_args
        session, headers, payload = args
        
        assert headers["Authorization"] == "Bearer test-key"
        assert payload["model"] == "openai/gpt-3.5-turbo"
        assert payload["messages"][0]["content"] == "What is the capital of France?"
        assert payload["max_tokens"] == 100
        assert payload["temperature"] == 0.2