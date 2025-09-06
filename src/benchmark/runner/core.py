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
import pandas as pd

from src.core.config import get_config
from src.core.database import get_db_session
from src.core.exceptions import DatabaseError, ModelAPIError
from src.core.debug_logger import initialize_debug_logger, get_debug_logger
from src.data.sampling import StatisticalSampler
from src.models.openrouter import OpenRouterClient
from src.models.prompt_formatter import PromptFormatter, PromptConfig, PromptTemplate
from src.models.response_parser import ResponseParser
from src.evaluation.grader import AnswerGrader, GradingCriteria, GradingMode
from src.evaluation.metrics import MetricsCalculator, ComprehensiveMetrics
from src.storage.repositories import BenchmarkRepository, QuestionRepository, ResponseRepository
from src.storage.models import BenchmarkRun, Question, BenchmarkResult
from src.utils.logging import get_logger

from .config import BenchmarkConfig, RunMode
from rich.progress import Progress
from .types import BenchmarkRunResult, BenchmarkProgress

from .question_loader import QuestionLoader
from .model_query_handler import ModelQueryHandler
from .result_saver import ResultSaver
from .metrics_handler import MetricsHandler

logger = get_logger(__name__)


class BenchmarkRunner:
    """Main orchestrator for running benchmarks."""
    
    def __init__(self):
        """Initialize the benchmark runner."""
        self.config = get_config()
        
        # Initialize debug logger
        self.debug_logger = initialize_debug_logger(
            debug_enabled=self.config.logging.debug.enabled,
            log_dir=self.config.logging.debug.log_dir
        )
        
        # Initialize components
        self.sampler = StatisticalSampler()
        self.prompt_formatter = PromptFormatter()
        self.response_parser = ResponseParser()
        self.grader = AnswerGrader()
        self.metrics_calculator = MetricsCalculator()
        
        # Initialize handler components
        self.question_loader = QuestionLoader(self.sampler)
        self.model_query_handler = ModelQueryHandler(
            prompt_formatter=self.prompt_formatter,
            response_parser=self.response_parser,
            grader=self.grader,
            debug_logger=self.debug_logger,
            config=self.config
        )
        self.result_saver = ResultSaver(config=self.config)
        self.metrics_handler = MetricsHandler(
            metrics_calculator=self.metrics_calculator,
            config=self.config
        )
        
        # State tracking
        self.current_benchmark: Optional[BenchmarkRun] = None
        self.current_benchmark_id: Optional[int] = None
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
    
    def get_default_config(self, mode: RunMode) -> BenchmarkConfig:
        """
        Get default configuration for the specified benchmark mode.
        
        Args:
            mode: The benchmark run mode
            
        Returns:
            BenchmarkConfig for the specified mode
        """
        return self.mode_configs.get(mode, self.mode_configs[RunMode.STANDARD])
    
    async def run_benchmark(self,
                          model_name: str,
                          mode: RunMode = RunMode.STANDARD,
                          custom_config: Optional[BenchmarkConfig] = None,
                          benchmark_name: Optional[str] = None,
                          progress: Optional[Progress] = None,
                          task_id: Optional[int] = None) -> BenchmarkRunResult:
        """
        Run a complete benchmark for a specified model.
        
        Args:
            model_name: Name of the model to benchmark
            mode: Benchmark mode (quick, standard, comprehensive)
            custom_config: Optional custom configuration
            benchmark_name: Optional name for the benchmark
            progress_task: Optional Rich progress task for UI updates
            
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
            questions = await self.question_loader.load_sample_questions(benchmark_id, config)
            
            # Update progress after loading questions
            if progress and task_id is not None:
                progress.advance(task_id, 1)  # Small advance for loading phase
                self.progress.completed_questions = 0  # Reset for querying
                self.progress.current_phase = "Querying model"
            
            # Phase 3: Query model and grade responses
            responses = await self.model_query_handler.query_model_batch(
                model_name, questions, config, benchmark_id, self.progress, progress, task_id
            )
            
            # Phase 3.5: Save individual response results to database
            self.progress.current_phase = "Saving results"
            if progress and task_id is not None:
                progress.advance(task_id, len(responses))
            await self.result_saver.save_benchmark_results(benchmark_id, model_name, responses)
            
            # Phase 4: Calculate metrics
            self.progress.current_phase = "Calculating metrics"
            metrics = await self.metrics_handler.calculate_metrics(benchmark_id, model_name, responses)
            
            # Phase 4.5: Save model performance summary
            if metrics:
                self.progress.current_phase = "Saving performance metrics"
                await self.result_saver.save_model_performance(benchmark_id, model_name, metrics)
            
            # Phase 5: Finalize
            self.progress.current_phase = "Finalizing"
            total_cost = sum(r.get('cost', 0) for r in responses)
            await self._finalize_benchmark(benchmark_id, True, None, total_cost)
            
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
            
            if self.current_benchmark_id:
                await self._finalize_benchmark(self.current_benchmark_id, False, str(e), 0.0)
            
            # Return error result
            execution_time = time.time() - start_time
            return BenchmarkRunResult(
                benchmark_id=self.current_benchmark_id or -1,
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
                # Access the id attribute after commit and refresh
                benchmark_id: int = benchmark.id  # type: ignore
                return benchmark_id
                
        except Exception as e:
            raise DatabaseError(f"Failed to initialize benchmark: {str(e)}")
    
    
            
    
    
    async def _save_benchmark_results(self, benchmark_id: int, model_name: str, responses: List[Dict[str, Any]]) -> None:
        """Save individual benchmark results to database."""
        try:
            logger.info(f"Saving {len(responses)} benchmark results to database")
            
            with get_db_session() as session:
                response_repo = ResponseRepository(session)
                
                saved_count = 0
                for response_data in responses:
                    try:
                        # Calculate Jeopardy score for this question
                        question_value = response_data.get('value', 400)  # Default to $400 if no value
                        is_correct = response_data.get('is_correct', False)
                        jeopardy_score = question_value if is_correct else -question_value
                        
                        # Create BenchmarkResult object
                        result = BenchmarkResult(
                            benchmark_run_id=benchmark_id,
                            question_id=response_data.get('question_id'),
                            model_name=model_name,
                            response_text=response_data.get('model_response', ''),
                            is_correct=is_correct,
                            confidence_score=response_data.get('confidence_score', 0.0),
                            jeopardy_score=jeopardy_score,  # Add Jeopardy score calculation
                            response_time_ms=response_data.get('response_time_ms', 0.0),
                            tokens_generated=response_data.get('tokens_generated', 0),
                            cost_usd=response_data.get('cost', 0.0)
                        )
                        
                        # Save to database
                        response_repo.save_response(result)
                        saved_count += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to save result for question {response_data.get('question_id')}: {str(e)}")
                        continue
                
                logger.info(f"Successfully saved {saved_count}/{len(responses)} benchmark results")
                
        except Exception as e:
            logger.error(f"Failed to save benchmark results: {str(e)}")
            # Don't raise - allow benchmark to continue with other phases
    
    async def _finalize_benchmark(self, benchmark_id: int, success: bool, error_message: Optional[str], total_cost: float = 0.0):
        """Finalize the benchmark run."""
        try:
            with get_db_session() as session:
                benchmark = session.query(BenchmarkRun).filter(BenchmarkRun.id == benchmark_id).first()
                if benchmark:
                    # Update the status and completion time
                    if success:
                        benchmark.status = 'completed'  # type: ignore
                        benchmark.completed_at = datetime.now()  # type: ignore
                    else:
                        benchmark.status = 'failed'  # type: ignore
                        if error_message:
                            # Store error details if provided
                            benchmark.error_details = json.dumps([error_message])  # type: ignore
                    
                    # Update total cost
                    benchmark.total_cost_usd = total_cost  # type: ignore
                    
                    session.commit()
                    logger.info(f"Benchmark {benchmark_id} finalized with status: {'completed' if success else 'failed'}, cost: ${total_cost:.4f}")
                else:
                    logger.warning(f"Benchmark {benchmark_id} not found for finalization")
        except Exception as e:
            logger.error(f"Failed to finalize benchmark {benchmark_id}: {str(e)}")
    
    async def _save_model_performance(self, benchmark_id: int, model_name: str, metrics: ComprehensiveMetrics) -> None:
        """Save model performance metrics to database."""
        try:
            from src.storage.repositories.performance_repository import PerformanceRepository
            from src.storage.models.model_performance import ModelPerformance
            
            logger.info(f"Saving model performance metrics for {model_name}")
            
            with get_db_session() as session:
                performance_repo = PerformanceRepository(session)
                
                # Create ModelPerformance object from metrics
                performance = ModelPerformance(
                    benchmark_run_id=benchmark_id,
                    model_name=model_name,
                    total_questions=metrics.accuracy.total_count,
                    correct_answers=metrics.accuracy.correct_count,
                    accuracy_rate=metrics.accuracy.overall_accuracy,
                    jeopardy_score=metrics.jeopardy_score.total_jeopardy_score,  # Add Jeopardy total score
                    category_jeopardy_scores=json.dumps(metrics.jeopardy_score.category_scores),  # Add category scores
                    avg_response_time_ms=metrics.performance.mean_response_time,
                    median_response_time_ms=metrics.performance.median_response_time,
                    min_response_time_ms=metrics.performance.min_response_time,
                    max_response_time_ms=metrics.performance.max_response_time,
                    total_cost_usd=metrics.cost.total_cost,
                    avg_cost_per_question=metrics.cost.cost_per_question,
                    cost_per_correct_answer=metrics.cost.cost_per_correct_answer,
                    total_tokens_input=metrics.cost.input_tokens,
                    total_tokens_output=metrics.cost.output_tokens,
                    total_tokens=metrics.cost.total_tokens,
                    avg_tokens_per_question=metrics.cost.tokens_per_question,
                    category_performance=json.dumps(metrics.accuracy.by_category),
                    difficulty_performance=json.dumps(metrics.accuracy.by_difficulty),
                    avg_confidence=getattr(metrics.consistency, 'confidence_correlation', 0.0),
                    confidence_accuracy_correlation=metrics.consistency.confidence_correlation,
                    error_count=metrics.performance.error_count,
                    error_rate=(1.0 - metrics.accuracy.overall_accuracy) if metrics.accuracy.overall_accuracy >= 0 else 0.0
                )
                
                # Save to database
                performance_repo.save_performance_summary(performance)
                logger.info(f"Successfully saved performance metrics for {model_name}")
                
        except Exception as e:
            logger.error(f"Failed to save model performance: {str(e)}")
            # Don't raise - allow benchmark to complete