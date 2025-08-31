"""
Evaluation Module

Answer evaluation and performance measurement with fuzzy string matching
for answer validation and confidence scoring.
"""

from .matcher import FuzzyAnswerMatcher
from .grader import AnswerGrader
from .metrics import MetricsCalculator

__all__ = [
    "FuzzyAnswerMatcher",
    "AnswerGrader",
    "MetricsCalculator",
]