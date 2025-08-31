"""
OpenRouter API Client

OpenRouter API client implementation with async HTTP requests,
rate limiting, retry logic, and cost tracking for language model benchmarking.
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional
import aiohttp
import time
from datetime import datetime, timedelta

from .base import ModelAdapter, ModelResponse, ModelConfig
from src.core.config import get_config
from src.core.exceptions import ModelAPIError, RateLimitError
from src.utils.logging import get_logger
from src.utils.async_helpers import retry_with_backoff, throttle_requests

logger = get_logger(__name__)


class OpenRouterClient(ModelAdapter):
    """OpenRouter API client for accessing multiple language models."""
    
    # Model pricing information (per 1M tokens)
    MODEL_PRICING = {
        'anthropic/claude-3-haiku': {'input': 0.25, 'output': 1.25},
        'anthropic/claude-3-sonnet': {'input': 3.0, 'output': 15.0},
        'anthropic/claude-3-opus': {'input': 15.0, 'output': 75.0},
        'openai/gpt-3.5-turbo': {'input': 0.5, 'output': 1.5},
        'openai/gpt-4': {'input': 30.0, 'output': 60.0},
        'openai/gpt-4-turbo': {'input': 10.0, 'output': 30.0},
        'meta-llama/llama-2-70b-chat': {'input': 0.7, 'output': 0.8},
        'mistralai/mixtral-8x7b-instruct': {'input': 0.24, 'output': 0.24},
    }
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[ModelConfig] = None):
        """
        Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key (from env if not provided)
            config: Model configuration
        """
        if config is None:
            config = ModelConfig(model_name="openai/gpt-3.5-turbo")
            
        super().__init__(config)
        
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ModelAPIError(
                "OpenRouter API key not provided. Set OPENROUTER_API_KEY environment variable."
            )
        
        self.app_config = get_config()
        self.base_url = self.app_config.openrouter.base_url
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting
        self._request_times: List[float] = []
        self._rate_limit_requests_per_minute = self.app_config.openrouter.rate_limit['requests_per_minute']
        
    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session exists."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting."""
        current_time = time.time()
        
        # Remove requests older than 1 minute
        cutoff_time = current_time - 60
        self._request_times = [t for t in self._request_times if t > cutoff_time]
        
        # Check if we're at the rate limit
        if len(self._request_times) >= self._rate_limit_requests_per_minute:
            sleep_time = 60 - (current_time - self._request_times[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
        
        # Record this request
        self._request_times.append(current_time)
    
    async def query(self, prompt: str, **kwargs) -> ModelResponse:
        """
        Query the OpenRouter API with a single prompt.
        
        Args:
            prompt: Input prompt for the model
            **kwargs: Additional parameters
            
        Returns:
            ModelResponse with generated text and metadata
            
        Raises:
            ModelAPIError: If API request fails
            RateLimitError: If rate limits are exceeded
        """
        try:
            await self._check_rate_limit()
            
            session = await self._ensure_session()
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://alex-trebench.local",
                "X-Title": "alex-treBENCH Benchmarking System"
            }
            
            # Merge config with any override parameters
            payload = {
                "model": self.config.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
                "temperature": kwargs.get('temperature', self.config.temperature),
            }
            
            # Add optional parameters if configured
            if self.config.top_p is not None:
                payload["top_p"] = self.config.top_p
            if self.config.frequency_penalty is not None:
                payload["frequency_penalty"] = self.config.frequency_penalty
            if self.config.presence_penalty is not None:
                payload["presence_penalty"] = self.config.presence_penalty
            if self.config.stop_sequences:
                payload["stop"] = self.config.stop_sequences
            
            start_time = time.time()
            
            # Make the API request with retry logic
            response_data = await self._make_request_with_retry(session, headers, payload)
            
            end_time = time.time()
            response_time_ms = int((end_time - start_time) * 1000)
            
            # Extract response text
            if 'choices' not in response_data or not response_data['choices']:
                raise ModelAPIError(
                    "Invalid response format: no choices in response",
                    model_name=self.config.model_name,
                    response_body=str(response_data)
                )
            
            response_text = response_data['choices'][0]['message']['content']
            
            # Calculate usage and cost
            usage = response_data.get('usage', {})
            tokens_generated = usage.get('completion_tokens', 0)
            tokens_input = usage.get('prompt_tokens', 0)
            
            cost_usd = self._calculate_cost(tokens_input, tokens_generated)
            
            # Create response object
            model_response = ModelResponse(
                model_id=self.config.model_name,
                prompt=prompt,
                response=response_text,
                latency_ms=float(response_time_ms),
                tokens_used=tokens_input + tokens_generated,
                cost=cost_usd,
                timestamp=datetime.now(),
                metadata={
                    'tokens_input': tokens_input,
                    'tokens_output': tokens_generated,
                    'usage': usage,
                    'model': response_data.get('model'),
                    'finish_reason': response_data['choices'][0].get('finish_reason')
                }
            )
            
            # Update usage stats
            self.update_usage_stats(model_response)
            
            logger.debug(f"OpenRouter query completed in {response_time_ms}ms, "
                        f"tokens: {tokens_input + tokens_generated}, cost: ${cost_usd:.6f}")
            
            return model_response
            
        except Exception as e:
            if isinstance(e, (ModelAPIError, RateLimitError)):
                raise
            raise ModelAPIError(
                f"OpenRouter query failed: {str(e)}",
                model_name=self.config.model_name
            ) from e
    
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def _make_request_with_retry(self, session: aiohttp.ClientSession, 
                                     headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request with retry logic."""
        try:
            async with session.post(f"{self.base_url}/chat/completions", 
                                  headers=headers, json=payload) as response:
                response_text = await response.text()
                
                if response.status == 429:
                    # Rate limit error
                    retry_after = response.headers.get('Retry-After')
                    retry_after_seconds = int(retry_after) if retry_after else 60
                    
                    raise RateLimitError(
                        f"Rate limit exceeded, retry after {retry_after_seconds} seconds",
                        retry_after=retry_after_seconds,
                        model_name=self.config.model_name,
                        status_code=response.status
                    )
                
                elif response.status != 200:
                    raise ModelAPIError(
                        f"API request failed with status {response.status}",
                        model_name=self.config.model_name,
                        status_code=response.status,
                        response_body=response_text
                    )
                
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError as e:
                    raise ModelAPIError(
                        f"Invalid JSON response: {str(e)}",
                        model_name=self.config.model_name,
                        response_body=response_text
                    ) from e
                    
        except aiohttp.ClientError as e:
            raise ModelAPIError(
                f"HTTP client error: {str(e)}",
                model_name=self.config.model_name
            ) from e
    
    async def batch_query(self, prompts: List[str], **kwargs) -> List[ModelResponse]:
        """
        Query multiple prompts in batch with rate limiting.
        
        Args:
            prompts: List of prompts to query
            **kwargs: Additional parameters
            
        Returns:
            List of ModelResponse objects
        """
        logger.info(f"Starting batch query with {len(prompts)} prompts")
        
        responses = []
        semaphore = asyncio.Semaphore(self.app_config.benchmark.max_concurrent_requests)
        
        async def query_single(prompt: str) -> ModelResponse:
            async with semaphore:
                return await self.query(prompt, **kwargs)
        
        # Execute all queries concurrently with rate limiting
        tasks = [query_single(prompt) for prompt in prompts]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        successful_responses = []
        failed_count = 0
        
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Query {i} failed: {str(response)}")
                failed_count += 1
                # Create a failed response
                successful_responses.append(ModelResponse(
                    model_id=self.config.model_name,
                    prompt=prompts[i] if i < len(prompts) else "",
                    response="",
                    latency_ms=0.0,
                    tokens_used=0,
                    cost=0.0,
                    timestamp=datetime.now(),
                    metadata={'error': str(response), 'failed': True}
                ))
            else:
                successful_responses.append(response)
        
        logger.info(f"Batch query completed: {len(successful_responses) - failed_count} successful, "
                   f"{failed_count} failed")
        
        return successful_responses
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage."""
        pricing = self.MODEL_PRICING.get(self.config.model_name)
        if not pricing:
            logger.warning(f"No pricing info for model {self.config.model_name}")
            return 0.0
        
        # Convert from per-1M-token pricing
        input_cost = (input_tokens / 1_000_000) * pricing['input']
        output_cost = (output_tokens / 1_000_000) * pricing['output']
        
        return input_cost + output_cost
    
    def is_available(self) -> bool:
        """Check if OpenRouter API is available."""
        return bool(self.api_key)
    
    def get_pricing_info(self) -> Dict[str, float]:
        """Get pricing information for the current model."""
        return self.MODEL_PRICING.get(self.config.model_name, {'input': 0.0, 'output': 0.0})
    
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models on OpenRouter."""
        try:
            session = await self._ensure_session()
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            async with session.get(f"{self.base_url}/models", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', [])
                else:
                    logger.error(f"Failed to list models: HTTP {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []
    
    async def close(self) -> None:
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
