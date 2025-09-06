"""
Model Query Handler Module

Handles querying models with questions, grading responses, and comprehensive debug logging.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.core.config import get_config
from src.core.exceptions import ModelAPIError
from src.models.openrouter import OpenRouterClient
from src.models.prompt_formatter import PromptFormatter, PromptConfig, PromptTemplate
from src.models.response_parser import ResponseParser
from src.evaluation.grader import AnswerGrader, GradingCriteria, GradingMode
from src.core.debug_logger import DebugLogger

logger = logging.getLogger(__name__)


class ModelQueryHandler:
    """Handles model querying, response parsing, grading, and debug logging for benchmarks."""
    
    def __init__(self, 
                 prompt_formatter: PromptFormatter,
                 response_parser: ResponseParser,
                 grader: AnswerGrader,
                 debug_logger: DebugLogger,
                 config=None):
        """
        Initialize the model query handler.
        
        Args:
            prompt_formatter: PromptFormatter instance
            response_parser: ResponseParser instance
            grader: AnswerGrader instance
            debug_logger: DebugLogger instance for logging interactions
            config: Configuration object (uses get_config() if None)
        """
        self.prompt_formatter = prompt_formatter
        self.response_parser = response_parser
        self.grader = grader
        self.debug_logger = debug_logger
        self.config = config or get_config()
    
    async def query_model_batch(self, 
                               model_name: str, 
                               questions: List[Dict[str, Any]], 
                               config: 'BenchmarkConfig',
                               benchmark_id: int,
                               progress: 'BenchmarkProgress') -> List[Dict[str, Any]]:
        """
        Query the model with all questions in a batch.
        
        Args:
            model_name: Name of the model to query
            questions: List of question dictionaries
            config: Benchmark configuration
            benchmark_id: Current benchmark ID
            progress: Progress tracker to update
            
        Returns:
            List of response dictionaries with grading and metrics
        """
        openrouter_client = None
        try:
            logger.info(f"🔍 DEBUG: Starting query_model_batch - model={model_name}, questions={len(questions)}")
            
            # Initialize OpenRouter client
            logger.info("🔍 DEBUG: Initializing OpenRouterClient...")
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
            
            semaphore = asyncio.Semaphore(config.max_concurrent_requests)
            
            # Execute all queries concurrently
            logger.info(f"🔍 DEBUG: Creating {len(questions)} concurrent query tasks...")
            query_tasks = [self._query_single_question(q, model_name, config, benchmark_id, prompt_config, semaphore, progress) for q in questions]
            logger.info("🔍 DEBUG: Starting asyncio.gather() for concurrent queries...")
            responses = await asyncio.gather(*query_tasks, return_exceptions=False)
            logger.info(f"🔍 DEBUG: asyncio.gather() completed with {len(responses)} responses")
            
            logger.info(f"Completed {len(responses)} model queries")
            successful_responses = sum(1 for r in responses if r.get('is_correct', False))
            logger.info(f"Successful responses: {successful_responses}/{len(responses)} ({successful_responses/len(responses)*100:.1f}%)")
            
            return responses
            
        except Exception as e:
            logger.error(f"Failed to query model batch: {str(e)}")
            raise ModelAPIError(f"Failed to query model batch: {str(e)}")
        finally:
            # Ensure proper session cleanup
            if openrouter_client:
                try:
                    await openrouter_client.close()
                    logger.debug("OpenRouter client session closed successfully")
                except Exception as e:
                    logger.warning(f"Error closing OpenRouter client session: {str(e)}")
    
    async def _query_single_question(self, 
                                   question_data: Dict[str, Any], 
                                   model_name: str,
                                   config: 'BenchmarkConfig',
                                   benchmark_id: int,
                                   prompt_config: PromptConfig,
                                   semaphore: asyncio.Semaphore,
                                   progress: 'BenchmarkProgress') -> Dict[str, Any]:
        """
        Query a single question and process the response.
        
        Args:
            question_data: Dictionary containing question information
            model_name: Name of the model
            config: Benchmark configuration
            benchmark_id: Current benchmark ID
            prompt_config: Prompt configuration
            semaphore: Async semaphore for concurrency control
            progress: Progress tracker
            
        Returns:
            Dictionary with comprehensive response data
        """
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
                
                # Debug log prompt if enabled
                if self.config.logging.debug.enabled and self.config.logging.debug.log_prompts:
                    self.debug_logger.log_prompt_only(
                        question_id=question_data['id'],
                        model_name=model_name,
                        question_text=question_data['question_text'],
                        formatted_prompt=prompt
                    )
                
                # Initialize OpenRouter client for this query (could be optimized to reuse)
                openrouter_client = OpenRouterClient()
                openrouter_client.config.model_name = model_name
                openrouter_client.config.timeout_seconds = config.timeout_seconds
                
                try:
                    # Query the model
                    model_response = await openrouter_client.query(prompt)
                finally:
                    await openrouter_client.close()
                
                # Parse the response
                parsed_response = self.response_parser.parse_response(model_response.response)
                
                # Debug log response if enabled
                if self.config.logging.debug.enabled and self.config.logging.debug.log_responses:
                    self.debug_logger.log_response_only(
                        question_id=question_data['id'],
                        model_name=model_name,
                        raw_response=model_response.response,
                        parsed_answer=parsed_response.extracted_answer
                    )
                
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
                
                # Debug log grading details if enabled
                if self.config.logging.debug.enabled and self.config.logging.debug.log_grading:
                    grading_details = {
                        'fuzzy_threshold': 0.80,
                        'semantic_threshold': 0.70,
                        'mode': 'JEOPARDY',
                        'match_details': graded_response.match_result.details if graded_response.match_result else {}
                    }
                    self.debug_logger.log_grading_details(
                        question_id=question_data['id'],
                        model_name=model_name,
                        parsed_answer=parsed_response.extracted_answer,
                        correct_answer=question_data['correct_answer'],
                        is_correct=graded_response.is_correct,
                        match_score=graded_response.match_result.confidence if graded_response.match_result else 0.0,
                        match_type=graded_response.match_result.match_type.value if graded_response.match_result else 'unknown',
                        grading_details=grading_details
                    )
                
                # Update progress
                progress.completed_questions += 1
                if graded_response.is_correct:
                    progress.successful_responses += 1
                else:
                    progress.failed_responses += 1
                
                # Comprehensive debug logging
                if self.config.logging.debug.enabled:
                    # Only log if not in errors_only mode, or if there's an error/incorrect answer
                    should_log = not self.config.logging.debug.log_errors_only or not graded_response.is_correct
                    
                    if should_log:
                        # Extract token information
                        tokens_input = model_response.metadata.get('tokens_input', 0) if model_response.metadata else 0
                        tokens_output = model_response.metadata.get('tokens_output', 0) if model_response.metadata else 0
                        
                        grading_details = {
                            'fuzzy_threshold': 0.80,
                            'semantic_threshold': 0.70,
                            'mode': 'JEOPARDY',
                            'match_details': graded_response.match_result.details if graded_response.match_result else {},
                            'confidence': graded_response.confidence,
                            'score': graded_response.score,
                            'partial_credit': graded_response.partial_credit
                        }
                        
                        self.debug_logger.log_model_interaction(
                            benchmark_id=benchmark_id,
                            question_id=question_data['id'],
                            model_name=model_name,
                            category=question_data.get('category'),
                            value=question_data.get('value', 400),
                            question_text=question_data['question_text'],
                            correct_answer=question_data['correct_answer'],
                            formatted_prompt=prompt,
                            raw_response=model_response.response,
                            parsed_answer=parsed_response.extracted_answer,
                            is_correct=graded_response.is_correct,
                            match_score=graded_response.match_result.confidence if graded_response.match_result else 0.0,
                            match_type=graded_response.match_result.match_type.value if graded_response.match_result else 'unknown',
                            confidence_score=graded_response.confidence or 0.0,
                            response_time_ms=model_response.latency_ms,
                            cost_usd=model_response.cost,
                            tokens_input=tokens_input,
                            tokens_output=tokens_output,
                            grading_details=grading_details
                        )
                
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
                progress.completed_questions += 1
                progress.failed_responses += 1
                
                # Debug log error if enabled
                if self.config.logging.debug.enabled:
                    try:
                        # Try to get the prompt that was being formatted when error occurred
                        error_prompt = ""
                        try:
                            error_prompt = self.prompt_formatter.format_prompt(
                                question=question_data['question_text'],
                                category=question_data.get('category'),
                                value=f"${question_data.get('value', 400)}",
                                difficulty=question_data.get('difficulty_level'),
                                config=prompt_config
                            )
                        except:
                            error_prompt = f"Failed to format prompt: {question_data['question_text']}"
                        
                        self.debug_logger.log_model_interaction(
                            benchmark_id=benchmark_id,
                            question_id=question_data['id'],
                            model_name=model_name,
                            category=question_data.get('category'),
                            value=question_data.get('value', 400),
                            question_text=question_data['question_text'],
                            correct_answer=question_data['correct_answer'],
                            formatted_prompt=error_prompt,
                            raw_response=f"ERROR: {str(e)}",
                            parsed_answer="",
                            is_correct=False,
                            match_score=0.0,
                            match_type='error',
                            confidence_score=0.0,
                            response_time_ms=0.0,
                            cost_usd=0.0,
                            tokens_input=0,
                            tokens_output=0,
                            error=str(e)
                        )
                    except Exception as debug_error:
                        logger.warning(f"Failed to log debug info for error: {debug_error}")
                
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