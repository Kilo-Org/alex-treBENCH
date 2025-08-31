"""
Models Module

Language model abstraction and API clients for benchmarking various models
through OpenRouter and other providers.
"""

from .base import ModelAdapter, ModelResponse
from .openrouter import OpenRouterClient

__all__ = [
    "ModelAdapter",
    "ModelResponse", 
    "OpenRouterClient",
]