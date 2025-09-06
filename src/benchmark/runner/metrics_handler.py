"""
Metrics Handler Module

Handles calculation of comprehensive metrics from benchmark responses.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.core.config import get_config
from src.evaluation.grader import GradedResponse
from src.evaluation.matcher import MatchResult, MatchType
from src.evaluation.metrics import MetricsCalculator, ComprehensiveMetrics
from src.models.response_parser import ParsedResponse, ResponseType
from src.models.base import ModelResponse

logger = logging.getLogger(__name__)


class MetricsHandler:
    """Handles calculation of comprehensive metrics from benchmark responses."""
    
    def __init__(self, metrics_calculator: MetricsCalculator, config=None):
        """
        Initialize the metrics handler.
        
        Args:
            metrics_calculator: MetricsCalculator instance
            config: Configuration object (uses get_config() if None)
        """
        self.metrics_calculator = metrics_calculator
        self.config = config or get_config()
    
    async def calculate_metrics(self, 
                               benchmark_id: int, 
                               model_name: str, 
                               responses: List[Dict[str, Any]]) -> Optional[ComprehensiveMetrics]:
        """
        Calculate comprehensive metrics for the benchmark.
        
        Args:
            benchmark_id: ID of the benchmark run
            model_name: Name of the model
            responses: List of response dictionaries
            
        Returns:
            ComprehensiveMetrics object or None if calculation fails
        """
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