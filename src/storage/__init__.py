"""
Storage Module

Data persistence and access layer with SQLAlchemy ORM model definitions,
repository pattern for data access, and database migration management.
"""

# Import from new modular structure
from .models import (
    Question,
    BenchmarkRun,
    BenchmarkResult,
    ModelPerformance,
    # Backward compatibility aliases
    Benchmark,
    BenchmarkQuestion,
    ModelResponse,
    ModelPerformanceSummary,
)

from .repositories import (
    QuestionRepository,
    BenchmarkRepository,
    ResponseRepository,
    PerformanceRepository,
)

__all__ = [
    # Models
    "Question",
    "BenchmarkRun", 
    "BenchmarkResult",
    "ModelPerformance",
    # Backward compatibility
    "Benchmark",
    "BenchmarkQuestion",
    "ModelResponse", 
    "ModelPerformanceSummary",
    # Repositories
    "BenchmarkRepository",
    "QuestionRepository",
    "ResponseRepository",
    "PerformanceRepository",
]