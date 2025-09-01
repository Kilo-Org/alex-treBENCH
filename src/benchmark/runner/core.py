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
            
            if self.current_benchmark_id:
                await self._finalize_benchmark(self.current_benchmark_id, False, str(e))
            
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
    
    async def _load_sample_questions(self, benchmark_id: int, config: BenchmarkConfig) -> List[Dict[str, Any]]:
        """Load and sample questions for the benchmark."""
        try:
            logger.info(f"Loading sample questions: {config.sample_size} questions using {config.sampling_method} sampling")
            
            # Get questions from database using repository
            with get_db_session() as session:
                question_repo = QuestionRepository(session)
                
                # Get all questions as dataframe for sampling
                all_questions = question_repo.get_all_questions()
                if not all_questions:
                    raise DatabaseError("No questions found in database. Please run data initialization first.")
                
                logger.info(f"Found {len(all_questions)} total questions in database")
                
                # Convert to DataFrame for sampling
                questions_data = []
                for q in all_questions:
                    questions_data.append({
                        'id': q.id,
                        'question_text': q.question_text,
                        'correct_answer': q.correct_answer,
                        'category': q.category,
                        'value': q.value or 400,  # Default value if None
                        'difficulty_level': q.difficulty_level or 'Medium',  # Default if None
                        'air_date': q.air_date,
                        'show_number': q.show_number,
                        'round': q.round
                    })
                
                df = pd.DataFrame(questions_data)
                
                # Use statistical sampler to get representative sample
                if config.sampling_method == "stratified":
                    sampled_df = self.sampler.stratified_sample(
                        df=df,
                        sample_size=config.sample_size,
                        stratify_columns=config.stratify_columns,
                        seed=config.sampling_seed
                    )
                else:
                    # Fall back to simple random sampling
                    sampled_df = df.sample(n=min(config.sample_size, len(df)), 
                                         random_state=config.sampling_seed)
                
                # Convert back to list of dicts with proper type handling
                sample_records = sampled_df.to_dict('records')
                sample_questions: List[Dict[str, Any]] = []
                for record in sample_records:
                    # Convert keys to strings and add to list
                    str_record: Dict[str, Any] = {str(k): v for k, v in record.items()}
                    sample_questions.append(str_record)
                
                logger.info(f"Sampled {len(sample_questions)} questions for benchmark")
                
                # Log sampling distribution
                if 'category' in sampled_df.columns:
                    category_counts = sampled_df['category'].value_counts()
                    logger.debug(f"Category distribution: {category_counts.head(10).to_dict()}")
                
                return sample_questions
                
        except Exception as e:
            logger.error(f"Failed to load sample questions: {str(e)}")
            raise DatabaseError(f"Failed to load sample questions: {str(e)}")
    
    async def _query_model_batch(self, model_name: str, questions: List[Dict[str, Any]], config: BenchmarkConfig) -> List[Dict[str, Any]]:
        """Query the model with all questions."""
        try:
            logger.info(f"Querying model {model_name} with {len(questions)} questions")
            
            # Initialize OpenRouter client
            openrouter_client = OpenRouterClient()
            openrouter_client.config.model_name = model_name
            openrouter_client.config.timeout_seconds = config.timeout_seconds
            
            # Configure prompt formatter  
            prompt_config = PromptConfig(
                template=PromptTemplate.JEOPARDY_STYLE,
                include_category=True,
                include_value=True,
                include_difficulty=False
            )
            
            responses = []
            semaphore = asyncio.Semaphore(config.max_concurrent_requests)
            
            async def query_single_question(question_data: Dict[str, Any]) -> Dict[str, Any]:
                async with semaphore:
                    try:
                        # Format the prompt
                        prompt = self.prompt_formatter.format_prompt(
                            question=question_data['question_text'],
                            category=question_data.get('category'),
                            value=f"${question_data.get('value', 400)}",
                            difficulty=question_data.get('difficulty_level'),
                            config=prompt_config
                        )
                        
                        # Query the model
                        model_response = await openrouter_client.query(prompt)
                        
                        # Parse the response
                        parsed_response = self.response_parser.parse_response(model_response.response)
                        
                        # Grade the response
                        graded_response = self.grader.grade_response(
                            response_text=model_response.response,
                            correct_answer=question_data['correct_answer'],
                            question_context={
                                'category': question_data.get('category'),
                                'value': question_data.get('value', 400),
                                'question_text': question_data['question_text']
                            },
                            multiple_acceptable=None,
                            criteria=GradingCriteria(
                                mode=GradingMode.JEOPARDY,
                                fuzzy_threshold=0.80,
                                semantic_threshold=0.70
                            )
                        )
                        
                        # Update progress
                        if self.progress:
                            self.progress.completed_questions += 1
                            if graded_response.is_correct:
                                self.progress.successful_responses += 1
                            else:
                                self.progress.failed_responses += 1
                        
                        # Return comprehensive response data
                        return {
                            'question_id': question_data['id'],
                            'question_text': question_data['question_text'],
                            'correct_answer': question_data['correct_answer'],
                            'category': question_data.get('category'),
                            'value': question_data.get('value', 400),
                            'model_response': model_response.response,
                            'parsed_answer': parsed_response.extracted_answer,
                            'is_correct': graded_response.is_correct,
                            'match_score': graded_response.match_result.confidence,
                            'match_type': graded_response.match_result.match_type.value,
                            'confidence_score': graded_response.confidence or 0.0,
                            'response_time_ms': model_response.latency_ms,
                            'tokens_generated': model_response.tokens_used,
                            'cost': model_response.cost,
                            'prompt': prompt,
                            'graded_response': graded_response,
                            'model_response_obj': model_response
                        }
                        
                    except Exception as e:
                        logger.error(f"Failed to query question {question_data.get('id', 'unknown')}: {str(e)}")
                        
                        # Update progress for failed response
                        if self.progress:
                            self.progress.completed_questions += 1
                            self.progress.failed_responses += 1
                        
                        # Return error response
                        return {
                            'question_id': question_data['id'],
                            'question_text': question_data['question_text'],
                            'correct_answer': question_data['correct_answer'],
                            'category': question_data.get('category'),
                            'value': question_data.get('value', 400),
                            'model_response': f"ERROR: {str(e)}",
                            'parsed_answer': "",
                            'is_correct': False,
                            'match_score': 0.0,
                            'match_type': 'error',
                            'confidence_score': 0.0,
                            'response_time_ms': 0.0,
                            'tokens_generated': 0,
                            'cost': 0.0,
                            'prompt': "",
                            'error': str(e)
                        }
            
            # Execute all queries concurrently
            query_tasks = [query_single_question(q) for q in questions]
            responses = await asyncio.gather(*query_tasks, return_exceptions=False)
            
            logger.info(f"Completed {len(responses)} model queries")
            successful_responses = sum(1 for r in responses if r.get('is_correct', False))
            logger.info(f"Successful responses: {successful_responses}/{len(responses)} ({successful_responses/len(responses)*100:.1f}%)")
            
            return responses
            
        except Exception as e:
            logger.error(f"Failed to query model batch: {str(e)}")
            raise ModelAPIError(f"Failed to query model batch: {str(e)}")
    
    async def _calculate_metrics(self, benchmark_id: int, model_name: str, responses: List[Dict[str, Any]]) -> Optional[ComprehensiveMetrics]:
        """Calculate comprehensive metrics for the benchmark."""
        try:
            if not responses:
                logger.warning("No responses provided for metrics calculation")
                return None
                
            logger.info(f"Calculating metrics for {len(responses)} responses")
            
            # Extract graded responses and model responses
            graded_responses = []
            model_responses = []
            question_contexts = []
            
            for response_data in responses:
                if 'graded_response' in response_data and 'model_response_obj' in response_data:
                    graded_responses.append(response_data['graded_response'])
                    model_responses.append(response_data['model_response_obj'])
                    
                    # Create question context for category/difficulty analysis
                    question_contexts.append({
                        'category': response_data.get('category'),
                        'value': response_data.get('value', 400),
                        'difficulty_level': response_data.get('difficulty_level', 'Medium'),
                        'question_id': response_data.get('question_id')
                    })
                else:
                    # Handle error responses - create minimal graded response
                    from src.evaluation.grader import GradedResponse
                    from src.evaluation.matcher import MatchResult, MatchType
                    from src.models.response_parser import ParsedResponse, ResponseType
                    from src.models.base import ModelResponse
                    from datetime import datetime
                    
                    # Create a minimal parsed response for errors
                    parsed_response = ParsedResponse(
                        original_text=response_data.get('model_response', ''),
                        extracted_answer=response_data.get('parsed_answer', ''),
                        response_type=ResponseType.ERROR,
                        confidence_indicators=[]
                    )
                    
                    # Create a minimal match result for errors
                    match_result = MatchResult(
                        is_match=False,
                        confidence=0.0,
                        match_type=MatchType.EXACT,
                        details={'error': response_data.get('error', 'Query failed')},
                        normalized_answer=response_data.get('parsed_answer', ''),
                        normalized_expected=response_data.get('correct_answer', '')
                    )
                    
                    graded_responses.append(GradedResponse(
                        is_correct=False,
                        score=0.0,
                        confidence=0.0,
                        partial_credit=0.0,
                        match_result=match_result,
                        parsed_response=parsed_response,
                        grading_metadata={'error': response_data.get('error', 'Query failed')},
                        grade_explanation=f"Query failed: {response_data.get('error', 'Unknown error')}",
                        timestamp=datetime.now()
                    ))
                    
                    model_responses.append(ModelResponse(
                        model_id=model_name,
                        prompt=response_data.get('prompt', ''),
                        response=response_data.get('model_response', ''),
                        latency_ms=response_data.get('response_time_ms', 0.0),
                        tokens_used=response_data.get('tokens_generated', 0),
                        cost=response_data.get('cost', 0.0),
                        timestamp=datetime.now(),
                        metadata={'error': response_data.get('error', 'Query failed')}
                    ))
                    
                    question_contexts.append({
                        'category': response_data.get('category'),
                        'value': response_data.get('value', 400),
                        'difficulty_level': response_data.get('difficulty_level', 'Medium'),
                        'question_id': response_data.get('question_id')
                    })
            
            # Calculate comprehensive metrics
            metrics = self.metrics_calculator.calculate_metrics(
                graded_responses=graded_responses,
                model_responses=model_responses,
                model_name=model_name,
                benchmark_id=benchmark_id,
                question_contexts=question_contexts
            )
            
            logger.info(f"Metrics calculation complete - Overall accuracy: {metrics.accuracy.overall_accuracy:.3f}")
            logger.info(f"Total cost: ${metrics.cost.total_cost:.4f}, Average response time: {metrics.performance.mean_response_time:.0f}ms")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate metrics: {str(e)}")
            # Don't raise exception, return None to allow benchmark to complete
            return None
    
    async def _finalize_benchmark(self, benchmark_id: int, success: bool, error_message: Optional[str]):
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
                    
                    session.commit()
                    logger.info(f"Benchmark {benchmark_id} finalized with status: {'completed' if success else 'failed'}")
                else:
                    logger.warning(f"Benchmark {benchmark_id} not found for finalization")
        except Exception as e:
            logger.error(f"Failed to finalize benchmark {benchmark_id}: {str(e)}")