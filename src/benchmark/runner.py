"""
Benchmark Runner

Main orchestrator for running benchmarks. Coordinates question loading, prompt formatting,
model querying, answer grading, metrics calculation, and result storage.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import traceback

from src.core.config import get_config
from src.core.database import get_db_session
from src.core.exceptions import DatabaseError
from src.data.sampling import StatisticalSampler
from src.models.openrouter import OpenRouterClient
from src.models.prompt_formatter import PromptFormatter, PromptConfig, PromptTemplate
from src.models.response_parser import ResponseParser
from src.evaluation.grader import AnswerGrader, GradingCriteria, GradingMode
from src.evaluation.metrics import MetricsCalculator, ComprehensiveMetrics
from src.storage.repositories import BenchmarkRepository, QuestionRepository, ResponseRepository
from src.storage.models import Benchmark, BenchmarkQuestion, ModelResponse, create_benchmark_run, create_question, create_benchmark_result
from src.utils.logging import get_logger

logger = get_logger(__name__)


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


@dataclass 
class BenchmarkProgress:
    """Progress tracking for benchmark execution."""
    total_questions: int
    completed_questions: int
    successful_responses: int
    failed_responses: int
    current_phase: str
    start_time: datetime
    estimated_completion: Optional[datetime] = None
    
    @property
    def completion_percentage(self) -> float:
        if self.total_questions == 0:
            return 0.0
        return (self.completed_questions / self.total_questions) * 100.0
    
    @property
    def success_rate(self) -> float:
        if self.completed_questions == 0:
            return 0.0
        return (self.successful_responses / self.completed_questions) * 100.0


@dataclass
class BenchmarkResult:
    """Complete result of a benchmark run."""
    benchmark_id: int
    model_name: str
    config: BenchmarkConfig
    progress: BenchmarkProgress
    metrics: Optional[ComprehensiveMetrics]
    questions: List[BenchmarkQuestion]
    responses: List[ModelResponse]
    graded_responses: List[Any]  # GradedResponse objects
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


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
        self.current_benchmark: Optional[Benchmark] = None
        self.progress: Optional[BenchmarkProgress] = None
        
        # Default configurations for different modes
        self.mode_configs = {
            RunMode.QUICK: BenchmarkConfig(
                mode=RunMode.QUICK,
                sample_size=50,
                timeout_seconds=30,
                max_concurrent_requests=3
            ),
            RunMode.STANDARD: BenchmarkConfig(
                mode=RunMode.STANDARD,
                sample_size=200,
                timeout_seconds=60,
                max_concurrent_requests=5
            ),
            RunMode.COMPREHENSIVE: BenchmarkConfig(
                mode=RunMode.COMPREHENSIVE,
                sample_size=1000,
                timeout_seconds=120,
                max_concurrent_requests=3
            )
        }
    
    async def run_benchmark(self,
                          model_name: str,
                          mode: RunMode = RunMode.STANDARD,
                          custom_config: Optional[BenchmarkConfig] = None,
                          benchmark_name: Optional[str] = None) -> BenchmarkResult:
        """
        Run a complete benchmark for a specified model.
        
        Args:
            model_name: Name of the model to benchmark
            mode: Benchmark mode (quick, standard, comprehensive)
            custom_config: Optional custom configuration
            benchmark_name: Optional name for the benchmark
            
        Returns:
            BenchmarkResult with complete results
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
            questions = await self._load_questions(benchmark_id, config)
            
            # Phase 3: Format prompts
            self.progress.current_phase = "Formatting prompts"
            prompts, question_contexts = await self._format_prompts(questions, config)
            
            # Phase 4: Query model
            self.progress.current_phase = "Querying model"
            model_responses = await self._query_model(model_name, prompts, config)
            
            # Phase 5: Grade responses
            self.progress.current_phase = "Grading responses"
            graded_responses = await self._grade_responses(
                model_responses, questions, question_contexts, config
            )
            
            # Phase 6: Calculate metrics
            self.progress.current_phase = "Calculating metrics"
            metrics = await self._calculate_metrics(
                graded_responses, model_responses, model_name, benchmark_id, question_contexts
            )
            
            # Phase 7: Save results
            if config.save_results:
                self.progress.current_phase = "Saving results"
                await self._save_results(benchmark_id, model_responses, graded_responses, metrics)
            
            # Phase 8: Finalize
            self.progress.current_phase = "Complete"
            await self._finalize_benchmark(benchmark_id, True)
            
            execution_time = time.time() - start_time
            logger.info(f"Benchmark completed successfully in {execution_time:.2f} seconds")
            
            return BenchmarkResult(
                benchmark_id=benchmark_id,
                model_name=model_name,
                config=config,
                progress=self.progress,
                metrics=metrics,
                questions=questions,
                responses=model_responses,
                graded_responses=graded_responses,
                execution_time=execution_time,
                success=True,
                metadata={
                    'benchmark_name': benchmark_name,
                    'completion_time': datetime.now().isoformat(),
                    'total_cost': sum(r.cost or 0 for r in model_responses),
                    'avg_response_time': sum(r.latency_ms or 0 for r in model_responses) / len(model_responses) if model_responses else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Benchmark failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Clean up on failure
            if self.current_benchmark:
                await self._finalize_benchmark(self.current_benchmark.id, False, str(e))
            
            execution_time = time.time() - start_time
            
            return BenchmarkResult(
                benchmark_id=self.current_benchmark.id if self.current_benchmark else -1,
                model_name=model_name,
                config=config,
                progress=self.progress or BenchmarkProgress(0, 0, 0, 0, "Failed", datetime.now()),
                metrics=None,
                questions=[],
                responses=[],
                graded_responses=[],
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    async def _initialize_benchmark(self, name: str, config: BenchmarkConfig, model_name: str) -> int:
        """Initialize a new benchmark in the database."""
        with get_db_session() as session:
            benchmark_repo = BenchmarkRepository(session)
            
            # Create benchmark record
            benchmark = create_benchmark_run(
                name=name,
                description=f"{config.mode.value} benchmark for {model_name}",
                sample_size=config.sample_size
            )
            benchmark.models_tested_list = [model_name]
            
            benchmark = benchmark_repo.create_benchmark(benchmark)
            self.current_benchmark = benchmark
            
            logger.info(f"Created benchmark {benchmark.id}: {name}")
            return benchmark.id
    
    async def _load_questions(self, benchmark_id: int, config: BenchmarkConfig) -> List[Dict[str, Any]]:
        """Load and sample questions for the benchmark."""
        with get_db_session() as session:
            question_repo = QuestionRepository(session)
            
            logger.info(f"Loading {config.sample_size} questions for benchmark {benchmark_id}")
            
            # First check if we have real Jeopardy questions in the database
            existing_questions = question_repo.get_questions(limit=1)
            
            if not existing_questions:
                # No questions in database - recommend data initialization
                logger.error("No questions found in database. Please initialize the dataset first.")
                raise DatabaseError(
                    "No questions available in database. Run 'python -m src.scripts.init_data' to initialize the Jeopardy dataset first.",
                    operation="load_questions",
                    table="questions"
                )
            
            # Load questions from database using statistical sampling
            total_questions = len(question_repo.get_questions())
            logger.info(f"Found {total_questions} questions in database")
            
            if total_questions < config.sample_size:
                logger.warning(f"Requested sample size ({config.sample_size}) exceeds available questions ({total_questions}). Using all available questions.")
                sample_size = total_questions
            else:
                sample_size = config.sample_size
            
            # Use statistical sampling to get representative sample
            sampled_questions = question_repo.get_random_questions(sample_size)
            
            # Convert SQLAlchemy objects to dictionaries to prevent DetachedInstanceError
            question_dicts = []
            for question in sampled_questions:
                question_dict = {
                    'id': question.id,
                    'question_text': question.question_text,
                    'correct_answer': question.correct_answer,
                    'category': question.category,
                    'value': question.value,
                    'difficulty_level': question.difficulty_level
                }
                question_dicts.append(question_dict)
            
            logger.info(f"Loaded {len(question_dicts)} questions for benchmark")
            return question_dicts
    
    def _create_sample_questions(self, benchmark_id: int, config: BenchmarkConfig) -> List[BenchmarkQuestion]:
        """Create sample questions for testing (placeholder for real dataset loading)."""
        sample_questions = [
            {
                'question_id': f'q_{uuid.uuid4().hex[:8]}_{i}',  # Use unique UUID-based IDs
                'question_text': f'This is sample question {i}',
                'correct_answer': f'What is answer {i}?',
                'category': f'CATEGORY_{i % 5 + 1}',
                'value': (i % 5 + 1) * 200,
                'difficulty_level': ['Easy', 'Medium', 'Hard'][i % 3]
            }
            for i in range(config.sample_size)
        ]
        
        questions = []
        for q_data in sample_questions:
            question = create_question(
                question_id=q_data['question_id'],
                question_text=q_data['question_text'],
                correct_answer=q_data['correct_answer'],
                category=q_data['category'],
                value=q_data['value']
            )
            questions.append(question)
        
        return questions
    
    async def _format_prompts(self, questions: List[Dict[str, Any]], 
                            config: BenchmarkConfig) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Format questions into prompts."""
        logger.info(f"Formatting {len(questions)} questions into prompts")
        
        prompt_config = PromptConfig(
            template=config.prompt_template,
            include_category=True,
            include_value=True
        )
        
        prompts = []
        question_contexts = []
        
        for question in questions:
            # Format the prompt
            prompt = self.prompt_formatter.format_prompt(
                question=question['question_text'],
                category=question['category'],
                value=f"${question['value']}" if question['value'] else None,
                difficulty=question['difficulty_level'],
                config=prompt_config
            )
            
            # Create context for grading
            context = {
                'question_id': question['id'],
                'category': question['category'],
                'value': question['value'],
                'difficulty_level': question['difficulty_level'],
                'correct_answer': question['correct_answer']
            }
            
            prompts.append(prompt)
            question_contexts.append(context)
        
        logger.info(f"Successfully formatted {len(prompts)} prompts")
        return prompts, question_contexts
    
    async def _query_model(self, model_name: str, prompts: List[str], 
                          config: BenchmarkConfig) -> List[ModelResponse]:
        """Query the model with all prompts."""
        logger.info(f"Querying model {model_name} with {len(prompts)} prompts")
        
        # Initialize OpenRouter client
        from src.models.base import ModelConfig
        
        model_config = ModelConfig(
            model_name=model_name,
            timeout_seconds=config.timeout_seconds,
            max_tokens=150  # Reasonable for Jeopardy answers
        )
        
        async with OpenRouterClient(config=model_config) as client:
            # Use batch query with concurrency control
            responses = await client.batch_query(prompts)
            
            # Update progress
            if self.progress:
                self.progress.completed_questions = len(responses)
                self.progress.successful_responses = sum(
                    1 for r in responses if not r.metadata.get('failed', False)
                )
                self.progress.failed_responses = len(responses) - self.progress.successful_responses
                
                logger.info(f"Model querying complete: {self.progress.successful_responses} successful, "
                           f"{self.progress.failed_responses} failed")
            
            return responses
    
    async def _grade_responses(self, 
                             model_responses: List[ModelResponse],
                             questions: List[Dict[str, Any]],
                             question_contexts: List[Dict[str, Any]],
                             config: BenchmarkConfig) -> List[Any]:
        """Grade model responses."""
        logger.info(f"Grading {len(model_responses)} responses")
        
        # Create grading criteria
        grading_criteria = self.grader.default_criteria
        grading_criteria.mode = config.grading_mode
        grading_criteria.partial_credit_enabled = config.enable_partial_credit
        
        graded_responses = []
        
        for i, (response, question, context) in enumerate(zip(model_responses, questions, question_contexts)):
            # Grade the response
            graded = self.grader.grade_response(
                response_text=response.response,
                correct_answer=question['correct_answer'],
                question_context=context,
                criteria=grading_criteria
            )
            
            graded_responses.append(graded)
            
            # Log progress every 50 responses
            if (i + 1) % 50 == 0:
                logger.info(f"Graded {i + 1}/{len(model_responses)} responses")
        
        correct_count = sum(1 for r in graded_responses if r.is_correct)
        logger.info(f"Grading complete: {correct_count}/{len(graded_responses)} correct "
                   f"({correct_count/len(graded_responses)*100:.1f}%)")
        
        return graded_responses
    
    async def _calculate_metrics(self,
                               graded_responses: List[Any],
                               model_responses: List[ModelResponse],
                               model_name: str,
                               benchmark_id: int,
                               question_contexts: List[Dict[str, Any]]) -> ComprehensiveMetrics:
        """Calculate comprehensive metrics."""
        logger.info("Calculating comprehensive metrics")
        
        metrics = self.metrics_calculator.calculate_metrics(
            graded_responses=graded_responses,
            model_responses=model_responses,
            model_name=model_name,
            benchmark_id=benchmark_id,
            question_contexts=question_contexts
        )
        
        logger.info(f"Metrics calculated - Overall Score: {metrics.overall_score:.3f}, "
                   f"Accuracy: {metrics.accuracy.overall_accuracy:.3f}")
        
        return metrics
    
    async def _save_results(self,
                          benchmark_id: int,
                          model_responses: List[ModelResponse],
                          graded_responses: List[Any],
                          metrics: ComprehensiveMetrics):
        """Save benchmark results to database."""
        logger.info("Saving benchmark results")
        
        with get_db_session() as session:
            response_repo = ResponseRepository(session)
            
            # Convert model responses to database format
            db_responses = []
            for i, (model_resp, graded_resp) in enumerate(zip(model_responses, graded_responses)):
                db_response = create_benchmark_result(
                    benchmark_run_id=benchmark_id,
                    question_id=str(i + 1),  # Assuming sequential IDs as string
                    model_name=model_resp.model_id,
                    response_text=model_resp.response,
                    is_correct=graded_resp.is_correct,
                    confidence_score=graded_resp.confidence,
                    response_time_ms=int(model_resp.latency_ms or 0),
                    tokens_generated=model_resp.tokens_used,
                    cost_usd=model_resp.cost,
                    metadata={
                        'match_type': graded_resp.match_result.match_type.value,
                        'partial_credit': graded_resp.partial_credit,
                        'response_type': graded_resp.parsed_response.response_type.value
                    }
                )
                db_responses.append(db_response)
            
            # Save responses
            response_repo.save_responses(db_responses)
            
            logger.info(f"Saved {len(db_responses)} responses to database")
    
    async def _finalize_benchmark(self, benchmark_id: int, success: bool, error_message: Optional[str] = None):
        """Finalize the benchmark run."""
        with get_db_session() as session:
            benchmark_repo = BenchmarkRepository(session)
            
            status = 'completed' if success else 'failed'
            benchmark_repo.update_benchmark_status(benchmark_id, status)
            
            if error_message:
                logger.error(f"Benchmark {benchmark_id} failed: {error_message}")
            else:
                logger.info(f"Benchmark {benchmark_id} finalized with status: {status}")
    
    def get_progress(self) -> Optional[BenchmarkProgress]:
        """Get current benchmark progress."""
        return self.progress
    
    def cancel_benchmark(self):
        """Cancel the current benchmark run."""
        # Implementation would set a cancellation flag
        # and gracefully stop the benchmark
        logger.warning("Benchmark cancellation requested")
        if self.current_benchmark:
            asyncio.create_task(self._finalize_benchmark(self.current_benchmark.id, False, "Cancelled by user"))
    
    async def resume_benchmark(self, benchmark_id: int) -> Optional[BenchmarkResult]:
        """Resume a previously interrupted benchmark."""
        # Implementation would check the database for incomplete benchmarks
        # and resume from the last checkpoint
        logger.info(f"Resume functionality not yet implemented for benchmark {benchmark_id}")
        return None
    
    def get_default_config(self, mode: RunMode) -> BenchmarkConfig:
        """Get default configuration for a benchmark mode."""
        return self.mode_configs.get(mode, self.mode_configs[RunMode.STANDARD])