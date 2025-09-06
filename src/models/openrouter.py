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
        'anthropic/claude-3.5-sonnet': {'input': 3.0, 'output': 15.0},
        'anthropic/claude-sonnet-4': {'input': 3.0, 'output': 15.0},  # Added missing model
        'openai/gpt-3.5-turbo': {'input': 0.5, 'output': 1.5},
        'openai/gpt-4': {'input': 30.0, 'output': 60.0},
        'openai/gpt-4-turbo': {'input': 10.0, 'output': 30.0},
        'openai/gpt-4o': {'input': 5.0, 'output': 15.0},
        'openai/gpt-4o-mini': {'input': 0.15, 'output': 0.6},
        'openai/gpt-5': {'input': 1.0, 'output': 10.0},
        'meta-llama/llama-2-70b-chat': {'input': 0.7, 'output': 0.8},
        'meta-llama/llama-3.1-405b-instruct': {'input': 3.0, 'output': 3.0},
        'mistralai/mixtral-8x7b-instruct': {'input': 0.24, 'output': 0.24},
        'google/gemini-pro': {'input': 0.5, 'output': 1.5},
        'google/gemini-pro-1.5': {'input': 3.5, 'output': 10.5},
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
            logger.debug(f"🔍 DEBUG: Starting OpenRouter query for model {self.config.model_name}")
            await self._check_rate_limit()
            logger.debug("🔍 DEBUG: Rate limit check passed")
            
            session = await self._ensure_session()
            logger.debug("🔍 DEBUG: HTTP session ensured")
            
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
            logger.info(f"🔍 DEBUG: Making OpenRouter API request to model {self.config.model_name}...")
            response_data = await self._make_request_with_retry(session, headers, payload)
            logger.info(f"🔍 DEBUG: OpenRouter API request completed successfully")
            
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
        logger.info(f"TRACE: OpenRouterClient._calculate_cost called for '{self.config.model_name}' with input_tokens={input_tokens}, output_tokens={output_tokens}")
        
        pricing = self.MODEL_PRICING.get(self.config.model_name)
        if not pricing:
            logger.warning(f"TRACE: No static MODEL_PRICING entry for '{self.config.model_name}'. Falling back to registry estimate.")
            # Fallback to registry for dynamic pricing
            from .model_registry import ModelRegistry
            return ModelRegistry.estimate_cost(self.config.model_name, input_tokens, output_tokens)
        
        logger.info(f"TRACE: Using static pricing for '{self.config.model_name}': input={pricing['input']}, output={pricing['output']}")
        # Convert from per-1M-token pricing
        input_cost = (input_tokens / 1_000_000) * pricing['input']
        output_cost = (output_tokens / 1_000_000) * pricing['output']
        total_cost = input_cost + output_cost
        logger.info(f"TRACE: Static cost calculated: total=${total_cost:.6f}")
        
        return total_cost
    
    def is_available(self) -> bool:
        """Check if OpenRouter API is available."""
        return bool(self.api_key)
    
    def get_pricing_info(self) -> Dict[str, float]:
        """Get pricing information for the current model."""
        return self.MODEL_PRICING.get(self.config.model_name, {'input': 0.0, 'output': 0.0})
    
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available models on OpenRouter with detailed information.
        
        Returns:
            List of model dictionaries with keys:
            - id: Model ID (e.g., "anthropic/claude-3.5-sonnet")
            - name: Display name
            - pricing: Dict with input_cost and output_cost per 1M tokens
            - context_length: Maximum context window
            - capabilities: List of model capabilities
            - provider: Model provider
            - description: Model description
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
            
            logger.info("Fetching available models from OpenRouter API")
            
            async with session.get(f"{self.base_url}/models", headers=headers) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch models: HTTP {response.status}")
                    response_text = await response.text()
                    raise ModelAPIError(
                        f"Failed to fetch models from OpenRouter API: HTTP {response.status}",
                        status_code=response.status,
                        response_body=response_text
                    )
                
                data = await response.json()
                raw_models = data.get('data', [])
                
                if not raw_models:
                    logger.warning("No models returned from OpenRouter API")
                    return []
                
                # Parse and extract model information
                parsed_models = []
                for model_data in raw_models:
                    try:
                        parsed_model = self._parse_model_data(model_data)
                        if parsed_model:
                            parsed_models.append(parsed_model)
                    except Exception as e:
                        logger.warning(f"Failed to parse model data for {model_data.get('id', 'unknown')}: {e}")
                        continue
                
                logger.info(f"Successfully fetched and parsed {len(parsed_models)} models from OpenRouter")
                return parsed_models
                
        except ModelAPIError:
            # Re-raise API errors
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching models from OpenRouter: {str(e)}")
            raise ModelAPIError(
                f"Failed to fetch models from OpenRouter: {str(e)}"
            ) from e
    
    def _parse_model_data(self, model_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse raw model data from OpenRouter API into standardized format.
        
        Args:
            model_data: Raw model data from API
            
        Returns:
            Parsed model dictionary or None if parsing fails
        """
        try:
            model_id = model_data.get('id')
            if not model_id:
                logger.warning("Model data missing ID, skipping")
                return None
            
            # Extract basic model info
            name = model_data.get('name', model_id)
            description = model_data.get('description', '')
            context_length = model_data.get('context_length', 0)
            
            # Extract provider from model ID (e.g., "openai/gpt-4" -> "openai")
            provider = model_id.split('/')[0] if '/' in model_id else 'unknown'
            
            # Extract pricing information
            pricing_info = model_data.get('pricing', {})
            input_cost = 0.0
            output_cost = 0.0
            
            if pricing_info:
                # OpenRouter pricing can come in different formats - handle both
                # Format 1: Direct per-1M-token pricing
                input_cost = float(pricing_info.get('prompt', 0))
                output_cost = float(pricing_info.get('completion', 0))
                
                # Format 2: Already converted per-1M-token pricing
                if input_cost == 0 and output_cost == 0:
                    input_cost = float(pricing_info.get('input_cost_per_1m_tokens', 0))
                    output_cost = float(pricing_info.get('output_cost_per_1m_tokens', 0))
                
                # Convert from per-token to per-1M-tokens if values are very small (likely per-token)
                if input_cost > 0 and input_cost < 0.01:
                    input_cost = input_cost * 1_000_000
                if output_cost > 0 and output_cost < 0.01:
                    output_cost = output_cost * 1_000_000
            
            # Extract capabilities and features
            capabilities = []
            
            # Check for streaming support
            if model_data.get('supports_streaming', True):
                capabilities.append('streaming')
            
            # Check for function calling
            if model_data.get('supports_function_calling', False):
                capabilities.append('function_calling')
            
            # Check for vision capabilities
            if model_data.get('supports_vision', False):
                capabilities.append('vision')
            
            # Add context-based capabilities
            if context_length >= 100000:
                capabilities.append('long_context')
            
            # Add reasoning capability for most models
            capabilities.append('chat')
            capabilities.append('reasoning')
            
            # Check if model is currently available
            is_available = not model_data.get('disabled', False)
            
            parsed_model = {
                'id': model_id,
                'name': name,
                'description': description,
                'provider': provider,
                'context_length': context_length,
                'pricing': {
                    'input_cost_per_1m_tokens': input_cost,
                    'output_cost_per_1m_tokens': output_cost
                },
                'capabilities': capabilities,
                'available': is_available,
                'per_request_limits': model_data.get('per_request_limits', {}),
                'top_provider': model_data.get('top_provider', {}),
                'architecture': model_data.get('architecture', {}),
                'modality': model_data.get('modality', 'text'),
                'updated_at': model_data.get('updated', None)
            }
            
            return parsed_model
            
        except Exception as e:
            logger.error(f"Error parsing model data: {e}")
            return None
    
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
