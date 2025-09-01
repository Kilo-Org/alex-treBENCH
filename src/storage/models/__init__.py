"""
Storage Models

SQLAlchemy ORM models for the Jeopardy benchmarking system.
"""

from .question import Question
from .benchmark_run import BenchmarkRun
from .benchmark_result import BenchmarkResult
from .model_performance import ModelPerformance

# Backward compatibility aliases
Benchmark = BenchmarkRun
BenchmarkQuestion = Question
ModelResponse = BenchmarkResult
ModelPerformanceSummary = ModelPerformance

__all__ = [
    # New names
    "Question",
    "BenchmarkRun",
    "BenchmarkResult", 
    "ModelPerformance",
    # Backward compatibility
    "Benchmark",
    "BenchmarkQuestion",
    "ModelResponse",
    "ModelPerformanceSummary",
]