"""
Benchmark Runner Core

Main orchestrator for running benchmarks. Coordinates question loading, prompt formatting,
model querying, answer grading, metrics calculation, and result storage.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import traceback

from src.core.config import get_config
from src.core.database import get_db_session
from src.core.exceptions import DatabaseError
from src.data.sampling import StatisticalSampler
from src.models.openrouter import OpenRouterClient
from src.models.prompt_formatter import PromptFormatter, PromptConfig
from src.models.response_parser import ResponseParser
from src.evaluation.grader import AnswerGrader, GradingCriteria, GradingMode
from src.evaluation.metrics import MetricsCalculator, ComprehensiveMetrics
from src.storage.repositories import BenchmarkRepository, QuestionRepository, ResponseRepository
from src.storage.models import BenchmarkRun, Question, BenchmarkResult
from src.utils.logging import get_logger

from .config import BenchmarkConfig, RunMode
from .types import BenchmarkRunResult, BenchmarkProgress

logger = get_logger(__name__)


class BenchmarkRunner:
    """Main orchestrator for running benchmarks."""
    
    def __init__(self):
        """Initialize the benchmark runner."""
        self.config = get_config()
        
        # Initialize components
        self.sampler = StatisticalSampler()
        self.prompt_formatter = PromptFormatter()
        self.response_parser = ResponseParser()
        self.grader = AnswerGrader()
        self.metrics_calculator = MetricsCalculator()
        
        # State tracking
        self.current_benchmark: Optional[BenchmarkRun] = None
        self.progress: Optional[BenchmarkProgress] = None
        
        # Default configurations for different modes
        self.mode_configs = {
            RunMode.QUICK: BenchmarkConfig(
                mode=RunMode.QUICK,
                sample_size=50,
                timeout_seconds=30,
                max_concurrent_requests=3,
                sampling_method="stratified",
                sampling_seed=42,
                stratify_columns=["category", "difficulty_level"]
            ),
            RunMode.STANDARD: BenchmarkConfig(
                mode=RunMode.STANDARD,
                sample_size=200,
                timeout_seconds=60,
                max_concurrent_requests=5,
                sampling_method="stratified",
                sampling_seed=42,
                stratify_columns=["category", "difficulty_level"]
            ),
            RunMode.COMPREHENSIVE: BenchmarkConfig(
                mode=RunMode.COMPREHENSIVE,
                sample_size=1000,
                timeout_seconds=120,
                max_concurrent_requests=3,
                sampling_method="stratified",
                sampling_seed=42,
                stratify_columns=["category", "difficulty_level"]
            )
        }
    
    async def run_benchmark(self,
                          model_name: str,
                          mode: RunMode = RunMode.STANDARD,
                          custom_config: Optional[BenchmarkConfig] = None,
                          benchmark_name: Optional[str] = None) -> BenchmarkRunResult:
        """
        Run a complete benchmark for a specified model.
        
        Args:
            model_name: Name of the model to benchmark
            mode: Benchmark mode (quick, standard, comprehensive)
            custom_config: Optional custom configuration
            benchmark_name: Optional name for the benchmark
            
        Returns:
            BenchmarkRunResult with complete results
        """
        start_time = time.time()
        
        # Get configuration
        config = custom_config or self.mode_configs.get(mode, self.mode_configs[RunMode.STANDARD])
        
        # Generate benchmark name
        if not benchmark_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            benchmark_name = f"{model_name}_{mode.value}_{timestamp}"
        
        logger.info(f"Starting benchmark '{benchmark_name}' for model {model_name}")
        logger.info(f"Configuration: {config}")
        
        try:
            # Phase 1: Initialize benchmark
            self.progress = BenchmarkProgress(
                total_questions=config.sample_size,
                completed_questions=0,
                successful_responses=0,
                failed_responses=0,
                current_phase="Initializing",
                start_time=datetime.now()
            )
            
            benchmark_id = await self._initialize_benchmark(benchmark_name, config, model_name)
            
            # Phase 2: Load and sample questions
            self.progress.current_phase = "Loading questions"
            questions = await self._load_sample_questions(benchmark_id, config)
            
            # Phase 3: Query model and grade responses
            self.progress.current_phase = "Querying model"
            responses = await self._query_model_batch(model_name, questions, config)
            
            # Phase 4: Calculate metrics
            self.progress.current_phase = "Calculating metrics"
            metrics = await self._calculate_metrics(benchmark_id, model_name, responses)
            
            # Phase 5: Finalize
            self.progress.current_phase = "Finalizing"
            await self._finalize_benchmark(benchmark_id, True, None)
            
            execution_time = time.time() - start_time
            
            # Create result object
            result = BenchmarkRunResult(
                benchmark_id=benchmark_id,
                model_name=model_name,
                config=config,
                progress=self.progress,
                metrics=metrics,
                questions=questions,
                responses=responses,
                errors=[],
                total_cost=sum(r.get('cost', 0) for r in responses),
                execution_time_seconds=execution_time
            )
            
            logger.info(f"Benchmark completed successfully in {execution_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Benchmark failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            if self.current_benchmark:
                await self._finalize_benchmark(benchmark_id, False, str(e))
            
            # Return error result
            execution_time = time.time() - start_time
            return BenchmarkRunResult(
                benchmark_id=self.current_benchmark.id if self.current_benchmark else -1,
                model_name=model_name,
                config=config,
                progress=self.progress or BenchmarkProgress(0, 0, 0, 0, "Failed", datetime.now()),
                metrics=None,
                questions=[],
                responses=[],
                errors=[str(e)],
                total_cost=0.0,
                execution_time_seconds=execution_time
            )
    
    async def _initialize_benchmark(self, name: str, config: BenchmarkConfig, model_name: str) -> int:
        """Initialize a new benchmark run."""
        try:
            with get_db_session() as session:
                benchmark = BenchmarkRun(
                    name=name,
                    description=f"{config.mode.value} benchmark for {model_name}",
                    sample_size=config.sample_size,
                    benchmark_mode=config.mode.value,
                    status='pending'
                )
                
                session.add(benchmark)
                session.commit()
                session.refresh(benchmark)
                
                self.current_benchmark = benchmark
                return benchmark.id
                
        except Exception as e:
            raise DatabaseError(f"Failed to initialize benchmark: {str(e)}")
    
    async def _load_sample_questions(self, benchmark_id: int, config: BenchmarkConfig) -> List[Dict[str, Any]]:
        """Load and sample questions for the benchmark."""
        # Simplified implementation - just return empty list for now
        return []
    
    async def _query_model_batch(self, model_name: str, questions: List[Dict[str, Any]], config: BenchmarkConfig) -> List[Dict[str, Any]]:
        """Query the model with all questions."""
        # Simplified implementation - just return empty list for now
        return []
    
    async def _calculate_metrics(self, benchmark_id: int, model_name: str, responses: List[Dict[str, Any]]) -> Optional[ComprehensiveMetrics]:
        """Calculate comprehensive metrics for the benchmark."""
        # Simplified implementation - return None for now
        return None
    
    async def _finalize_benchmark(self, benchmark_id: int, success: bool, error_message: Optional[str]):
        """Finalize the benchmark run."""
        try:
            with get_db_session() as session:
                benchmark = session.query(BenchmarkRun).filter(BenchmarkRun.id == benchmark_id).first()
                if benchmark:
                    benchmark.status = 'completed' if success else 'failed'
                    if success:
                        benchmark.completed_at = datetime.now()
                    session.commit()
        except Exception as e:
            logger.error(f"Failed to finalize benchmark: {str(e)}")