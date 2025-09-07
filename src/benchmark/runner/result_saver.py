"""
Result Saver Module

Handles saving benchmark results and performance metrics to the database.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from src.core.config import get_config
from src.core.database import get_db_session
from src.evaluation.metrics import ComprehensiveMetrics
from src.storage.repositories import ResponseRepository
from src.storage.repositories.performance_repository import PerformanceRepository
from src.storage.models import BenchmarkResult, ModelPerformance

logger = logging.getLogger(__name__)


def _convert_numpy_types(value: Any) -> Any:
    """
    Convert NumPy types to native Python types for database compatibility.
    
    Args:
        value: Value that might be a NumPy type
        
    Returns:
        Native Python type equivalent
    """
    if isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, (np.ndarray, list)):
        return [_convert_numpy_types(item) for item in value]
    elif isinstance(value, dict):
        return {key: _convert_numpy_types(val) for key, val in value.items()}
    else:
        return value


class ResultSaver:
    """Handles saving benchmark results and performance metrics to the database."""
    
    def __init__(self, config=None):
        """
        Initialize the result saver.
        
        Args:
            config: Configuration object (uses get_config() if None)
        """
        self.config = config or get_config()
    
    async def save_benchmark_results(self, benchmark_id: int, model_name: str, responses: List[Dict[str, Any]]) -> None:
        """
        Save individual benchmark results to database.
        
        Args:
            benchmark_id: ID of the benchmark run
            model_name: Name of the model
            responses: List of response dictionaries
        """
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
    
    async def save_model_performance(self, benchmark_id: int, model_name: str, metrics: ComprehensiveMetrics) -> None:
        """
        Save model performance metrics to database.
        
        Args:
            benchmark_id: ID of the benchmark run
            model_name: Name of the model
            metrics: ComprehensiveMetrics object with performance data
        """
        try:
            logger.info(f"Saving model performance metrics for {model_name}")
            
            with get_db_session() as session:
                performance_repo = PerformanceRepository(session)
                
                # Create ModelPerformance object from metrics
                # Convert all values to native Python types to avoid PostgreSQL schema errors
                performance = ModelPerformance(
                    benchmark_run_id=benchmark_id,
                    model_name=model_name,
                    total_questions=_convert_numpy_types(metrics.accuracy.total_count),
                    correct_answers=_convert_numpy_types(metrics.accuracy.correct_count),
                    accuracy_rate=_convert_numpy_types(metrics.accuracy.overall_accuracy),
                    jeopardy_score=_convert_numpy_types(metrics.jeopardy_score.total_jeopardy_score),
                    category_jeopardy_scores=json.dumps(_convert_numpy_types(metrics.jeopardy_score.category_scores)),
                    avg_response_time_ms=_convert_numpy_types(metrics.performance.mean_response_time),
                    median_response_time_ms=_convert_numpy_types(metrics.performance.median_response_time),
                    min_response_time_ms=_convert_numpy_types(metrics.performance.min_response_time),
                    max_response_time_ms=_convert_numpy_types(metrics.performance.max_response_time),
                    total_cost_usd=_convert_numpy_types(metrics.cost.total_cost),
                    avg_cost_per_question=_convert_numpy_types(metrics.cost.cost_per_question),
                    cost_per_correct_answer=_convert_numpy_types(metrics.cost.cost_per_correct_answer),
                    total_tokens_input=_convert_numpy_types(metrics.cost.input_tokens),
                    total_tokens_output=_convert_numpy_types(metrics.cost.output_tokens),
                    total_tokens=_convert_numpy_types(metrics.cost.total_tokens),
                    avg_tokens_per_question=_convert_numpy_types(metrics.cost.tokens_per_question),
                    category_performance=json.dumps(_convert_numpy_types(metrics.accuracy.by_category)),
                    difficulty_performance=json.dumps(_convert_numpy_types(metrics.accuracy.by_difficulty)),
                    avg_confidence=_convert_numpy_types(getattr(metrics.consistency, 'confidence_correlation', 0.0)),
                    confidence_accuracy_correlation=_convert_numpy_types(metrics.consistency.confidence_correlation),
                    error_count=_convert_numpy_types(metrics.performance.error_count),
                    error_rate=_convert_numpy_types((1.0 - metrics.accuracy.overall_accuracy) if metrics.accuracy.overall_accuracy >= 0 else 0.0)
                )
                
                # Save to database
                performance_repo.save_performance_summary(performance)
                logger.info(f"Successfully saved performance metrics for {model_name}")
                
        except Exception as e:
            logger.error(f"Failed to save model performance: {str(e)}")
            # Don't raise - allow benchmark to complete