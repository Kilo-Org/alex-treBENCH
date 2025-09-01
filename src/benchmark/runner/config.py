"""
Benchmark Configuration

Configuration classes and enums for benchmark execution.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from src.evaluation.grader import GradingMode
from src.models.prompt_formatter import PromptTemplate


class RunMode(str, Enum):
    """Benchmark run modes."""
    QUICK = "quick"
    STANDARD = "standard" 
    COMPREHENSIVE = "comprehensive"
    CUSTOM = "custom"


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    mode: RunMode
    sample_size: int
    categories: Optional[List[str]] = None
    timeout_seconds: int = 60
    max_concurrent_requests: int = 5
    confidence_level: float = 0.95
    margin_of_error: float = 0.05
    grading_mode: GradingMode = GradingMode.LENIENT
    prompt_template: PromptTemplate = PromptTemplate.JEOPARDY_STYLE
    enable_partial_credit: bool = True
    save_results: bool = True
    resume_from_checkpoint: bool = True
    
    # Sampling configuration
    sampling_method: str = "stratified"  # Options: random, stratified, balanced, temporal
    sampling_seed: Optional[int] = 42  # Fixed seed for reproducibility (None for random)
    stratify_columns: Optional[List[str]] = None
    difficulty_distribution: Optional[Dict[str, float]] = None
    enable_temporal_stratification: bool = False