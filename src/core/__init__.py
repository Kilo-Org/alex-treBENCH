"""
Core Module

Foundational components used across the application including configuration
management, database connections, and custom exceptions.
"""

from .config import get_config, AppConfig
from .exceptions import (
    AlexTreBenchException,
    ConfigurationError,
    DatabaseError,
    ModelAPIError,
    EvaluationError,
)

__all__ = [
    "get_config",
    "AppConfig",
    "AlexTreBenchException",
    "ConfigurationError", 
    "DatabaseError",
    "ModelAPIError",
    "EvaluationError",
]