"""
Jeopardy Benchmarking System

A comprehensive system for benchmarking language models using Jeopardy questions
from Kaggle datasets with statistical sampling and fuzzy answer matching.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core.config import get_config
from .core.exceptions import JeopardyBenchException

__all__ = [
    "get_config",
    "JeopardyBenchException",
]