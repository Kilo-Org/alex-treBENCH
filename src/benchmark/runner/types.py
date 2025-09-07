"""
Benchmark Types

Type definitions for benchmark execution results and progress tracking.
"""

from typing import List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from src.evaluation.metrics import ComprehensiveMetrics
from .config import BenchmarkConfig


@dataclass
class BenchmarkProgress:
    """Progress tracking for benchmark execution."""
    total_questions: int
    completed_questions: int
    successful_responses: int
    failed_responses: int
    current_phase: str
    start_time: datetime
    unanswered_responses: int = 0  # New field for rate-limited questions
    estimated_completion: Optional[datetime] = None
    
    @property
    def completion_percentage(self) -> float:
        if self.total_questions == 0:
            return 0.0
        return (self.completed_questions / self.total_questions) * 100.0
    
    @property
    def success_rate(self) -> float:
        """Success rate based on answered questions only."""
        answered_questions = self.successful_responses + self.failed_responses
        if answered_questions == 0:
            return 0.0
        return (self.successful_responses / answered_questions) * 100.0
    
    @property
    def answer_rate(self) -> float:
        """Percentage of questions that were answered (not rate limited)."""
        if self.completed_questions == 0:
            return 0.0
        answered_questions = self.successful_responses + self.failed_responses
        return (answered_questions / self.completed_questions) * 100.0


@dataclass
class BenchmarkRunResult:
    """Complete result of a benchmark run."""
    benchmark_id: int
    model_name: str
    config: BenchmarkConfig
    progress: BenchmarkProgress
    metrics: Optional[ComprehensiveMetrics]
    questions: List[Any]  # Question objects
    responses: List[Any]  # Response objects
    errors: List[str]
    total_cost: float
    execution_time_seconds: float
    
    @property
    def is_successful(self) -> bool:
        """Check if benchmark completed successfully."""
        return self.progress.completion_percentage >= 100.0
    
    @property
    def accuracy_rate(self) -> float:
        """Get overall accuracy rate."""
        if self.metrics:
            # Return basic accuracy calculation
            return self.progress.success_rate / 100.0
        return 0.0