"""
Base Model Interface

Abstract base class for model interfaces providing a common API
for different language model providers with standardized responses.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import time


@dataclass
class ModelResponse:
    """Standardized response from a language model."""
    model_id: str
    prompt: str
    response: str
    latency_ms: float
    tokens_used: int
    cost: float
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass 
class ModelConfig:
    """Configuration for a language model."""
    model_name: str
    max_tokens: int = 150
    temperature: float = 0.1
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    timeout_seconds: int = 30


class ModelAdapter(ABC):
    """Abstract base class for language model adapters."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the model adapter.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self._total_requests = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        
    @abstractmethod
    async def query(self, prompt: str, **kwargs) -> ModelResponse:
        """
        Query the language model with a prompt.
        
        Args:
            prompt: Input prompt for the model
            **kwargs: Additional model-specific parameters
            
        Returns:
            ModelResponse with generated text and metadata
            
        Raises:
            ModelAPIError: If the API request fails
        """
        pass
    
    @abstractmethod
    async def batch_query(self, prompts: List[str], **kwargs) -> List[ModelResponse]:
        """
        Query the model with multiple prompts in batch.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional model-specific parameters
            
        Returns:
            List of ModelResponse objects
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the model is available.
        
        Returns:
            True if model is available, False otherwise
        """
        pass
    
    @abstractmethod
    def get_pricing_info(self) -> Dict[str, float]:
        """
        Get pricing information for the model.
        
        Returns:
            Dictionary with pricing per token/request
        """
        pass
    
    def format_jeopardy_prompt(self, question: str) -> str:
        """
        Format a Jeopardy question as a prompt for the model.
        
        Args:
            question: The Jeopardy question
            
        Returns:
            Formatted prompt string
        """
        # Standard Jeopardy prompt format
        prompt = f"""You are playing Jeopardy! Please provide the answer to this question in the form of a question (starting with "What is", "Who is", "Where is", etc.).

Question: {question}

Answer:"""
        
        return prompt
    
    def extract_answer(self, response_text: str) -> str:
        """
        Extract the answer from model response text.
        
        Args:
            response_text: Raw response from the model
            
        Returns:
            Cleaned answer text
        """
        # Remove common prefixes and clean up response
        text = response_text.strip()
        
        # Common Jeopardy answer prefixes to preserve
        jeopardy_prefixes = [
            "What is", "Who is", "Where is", "When is", "Why is", "How is",
            "What are", "Who are", "Where are", "When are", "Why are", "How are",
            "What was", "Who was", "Where was", "When was", "Why was", "How was",
            "What were", "Who were", "Where were", "When were", "Why were", "How were"
        ]
        
        # Look for Jeopardy format answers
        for prefix in jeopardy_prefixes:
            if text.lower().startswith(prefix.lower()):
                # Find the end of the answer (period, question mark, or newline)
                for end_char in ['.', '?', '\n']:
                    if end_char in text:
                        text = text.split(end_char)[0] + ('?' if not text.endswith('?') else '')
                        break
                break
        else:
            # If no Jeopardy format found, take first sentence
            if '.' in text:
                text = text.split('.')[0]
            elif '\n' in text:
                text = text.split('\n')[0]
        
        return text.strip()
    
    def update_usage_stats(self, response: ModelResponse) -> None:
        """
        Update usage statistics.
        
        Args:
            response: Model response to record stats for
        """
        self._total_requests += 1
        if response.tokens_used:
            self._total_tokens += response.tokens_used
        if response.cost:
            self._total_cost += response.cost
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics for this adapter.
        
        Returns:
            Dictionary with usage statistics
        """
        return {
            'total_requests': self._total_requests,
            'total_tokens': self._total_tokens,
            'total_cost_usd': self._total_cost,
            'average_cost_per_request': self._total_cost / max(1, self._total_requests),
            'average_tokens_per_request': self._total_tokens / max(1, self._total_requests)
        }
    
    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self._total_requests = 0
        self._total_tokens = 0
        self._total_cost = 0.0
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the model.
        
        Returns:
            Dictionary with health check results
        """
        start_time = time.time()
        
        try:
            # Simple test query
            test_response = await self.query("Test query for health check")
            response_time = (time.time() - start_time) * 1000
            
            return {
                'status': 'healthy',
                'response_time_ms': response_time,
                'model_name': self.config.model_name,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'model_name': self.config.model_name,
                'timestamp': datetime.now().isoformat()
            }
    
    def __repr__(self) -> str:
        """String representation of the adapter."""
        return f"{self.__class__.__name__}(model={self.config.model_name})"


class ModelRegistry:
    """Registry for managing multiple model adapters."""
    
    def __init__(self):
        """Initialize the model registry."""
        self._adapters: Dict[str, ModelAdapter] = {}
    
    def register(self, name: str, adapter: ModelAdapter) -> None:
        """Register a model adapter."""
        self._adapters[name] = adapter
    
    def get(self, name: str) -> Optional[ModelAdapter]:
        """Get a registered model adapter."""
        return self._adapters.get(name)
    
    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._adapters.keys())
    
    def get_available_models(self) -> List[str]:
        """Get list of currently available models."""
        available = []
        for name, adapter in self._adapters.items():
            if adapter.is_available():
                available.append(name)
        return available
    
    async def health_check_all(self) -> Dict[str, Any]:
        """Perform health checks on all registered models."""
        results = {}
        for name, adapter in self._adapters.items():
            results[name] = await adapter.health_check()
        return results


# Global model registry instance
model_registry = ModelRegistry()