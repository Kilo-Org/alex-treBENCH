"""
Evaluation Module

Answer evaluation and performance measurement with fuzzy string matching
for answer validation and confidence scoring.
"""

from .matcher import FuzzyMatcher
from .grader import AnswerGrader
from .metrics import MetricsCalculator

__all__ = [
    "FuzzyMatcher",
    "AnswerGrader",
    "MetricsCalculator",
]