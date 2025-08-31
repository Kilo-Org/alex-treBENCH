"""
Metrics Calculator

Calculates comprehensive benchmark metrics including accuracy, performance,
cost effectiveness, and consistency scores with support for aggregation and filtering.
"""

import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np
from datetime import datetime, timedelta

from .grader import GradedResponse
from ..models.base import ModelResponse
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AccuracyMetrics:
    """Accuracy-related metrics."""
    overall_accuracy: float
    correct_count: int
    total_count: int
    by_category: Dict[str, float] = field(default_factory=dict)
    by_difficulty: Dict[str, float] = field(default_factory=dict)
    by_value: Dict[str, float] = field(default_factory=dict)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance-related metrics."""
    mean_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    min_response_time: float
    max_response_time: float
    response_time_std: float
    timeout_count: int
    error_count: int


@dataclass
class CostMetrics:
    """Cost and efficiency metrics."""
    total_cost: float
    cost_per_question: float
    cost_per_correct_answer: float
    total_tokens: int
    tokens_per_question: float
    tokens_per_correct_answer: float
    input_tokens: int
    output_tokens: int
    cost_efficiency_score: float  # Cost per correct answer relative to baseline


@dataclass
class ConsistencyMetrics:
    """Consistency and reliability metrics."""
    performance_variance: float
    category_consistency_score: float
    difficulty_consistency_score: float
    confidence_correlation: float  # Correlation between confidence and correctness
    response_type_distribution: Dict[str, float]
    match_type_distribution: Dict[str, float]


@dataclass
class ComprehensiveMetrics:
    """Complete set of benchmark metrics."""
    model_name: str
    benchmark_id: Optional[int]
    timestamp: datetime
    
    accuracy: AccuracyMetrics
    performance: PerformanceMetrics
    cost: CostMetrics
    consistency: ConsistencyMetrics
    
    # Summary scores
    overall_score: float  # Composite score
    quality_score: float  # Based on accuracy and consistency
    efficiency_score: float  # Based on cost and performance
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCalculator:
    """Calculates comprehensive benchmark metrics."""
    
    def __init__(self, baseline_costs: Optional[Dict[str, float]] = None):
        """
        Initialize metrics calculator.
        
        Args:
            baseline_costs: Baseline costs for efficiency comparison
        """
        self.baseline_costs = baseline_costs or {
            'cost_per_correct_answer': 0.01,  # $0.01 baseline
            'tokens_per_correct_answer': 150   # 150 tokens baseline
        }
        
    def calculate_metrics(self, 
                         graded_responses: List[GradedResponse],
                         model_responses: List[ModelResponse],
                         model_name: str,
                         benchmark_id: Optional[int] = None,
                         question_contexts: Optional[List[Dict[str, Any]]] = None) -> ComprehensiveMetrics:
        """
        Calculate comprehensive metrics for a benchmark run.
        
        Args:
            graded_responses: List of graded responses
            model_responses: List of model responses with performance data
            model_name: Name of the model being evaluated
            benchmark_id: ID of the benchmark
            question_contexts: Optional question context information
            
        Returns:
            ComprehensiveMetrics object with all calculated metrics
        """
        if len(graded_responses) != len(model_responses):
            raise ValueError("Graded responses and model responses must have same length")
        
        logger.info(f"Calculating metrics for {len(graded_responses)} responses from {model_name}")
        
        # Calculate individual metric categories
        accuracy = self._calculate_accuracy_metrics(graded_responses, question_contexts)
        performance = self._calculate_performance_metrics(model_responses)
        cost = self._calculate_cost_metrics(model_responses, graded_responses)
        consistency = self._calculate_consistency_metrics(graded_responses, question_contexts)
        
        # Calculate composite scores
        overall_score = self._calculate_overall_score(accuracy, performance, cost, consistency)
        quality_score = self._calculate_quality_score(accuracy, consistency)
        efficiency_score = self._calculate_efficiency_score(cost, performance)
        
        # Create metadata
        metadata = self._create_metadata(graded_responses, model_responses, question_contexts)
        
        return ComprehensiveMetrics(
            model_name=model_name,
            benchmark_id=benchmark_id,
            timestamp=datetime.now(),
            accuracy=accuracy,
            performance=performance,
            cost=cost,
            consistency=consistency,
            overall_score=overall_score,
            quality_score=quality_score,
            efficiency_score=efficiency_score,
            metadata=metadata
        )
    
    def _calculate_accuracy_metrics(self, 
                                  graded_responses: List[GradedResponse],
                                  question_contexts: Optional[List[Dict[str, Any]]]) -> AccuracyMetrics:
        """Calculate accuracy-related metrics."""
        if not graded_responses:
            return AccuracyMetrics(overall_accuracy=0.0, correct_count=0, total_count=0)
        
        # Overall accuracy
        correct_count = sum(1 for r in graded_responses if r.is_correct)
        total_count = len(graded_responses)
        overall_accuracy = correct_count / total_count
        
        # By category accuracy
        by_category = defaultdict(list)
        by_difficulty = defaultdict(list)
        by_value = defaultdict(list)
        
        if question_contexts:
            for i, (response, context) in enumerate(zip(graded_responses, question_contexts)):
                category = context.get('category', 'Unknown')
                difficulty = context.get('difficulty_level', 'Unknown')
                value = context.get('value', 0)
                
                by_category[category].append(response.is_correct)
                by_difficulty[difficulty].append(response.is_correct)
                
                # Group values into ranges
                if value < 400:
                    value_range = 'Low ($1-399)'
                elif value < 800:
                    value_range = 'Medium ($400-799)'
                else:
                    value_range = 'High ($800+)'
                by_value[value_range].append(response.is_correct)
        
        # Calculate category accuracies
        category_accuracies = {
            cat: sum(results) / len(results) 
            for cat, results in by_category.items() if results
        }
        
        difficulty_accuracies = {
            diff: sum(results) / len(results)
            for diff, results in by_difficulty.items() if results
        }
        
        value_accuracies = {
            val_range: sum(results) / len(results)
            for val_range, results in by_value.items() if results
        }
        
        # Calculate confidence intervals (95%)
        confidence_intervals = {}
        if total_count > 30:  # Only calculate for sufficient sample size
            z_score = 1.96  # 95% confidence
            margin_of_error = z_score * np.sqrt((overall_accuracy * (1 - overall_accuracy)) / total_count)
            confidence_intervals['overall'] = (
                max(0.0, overall_accuracy - margin_of_error),
                min(1.0, overall_accuracy + margin_of_error)
            )
        
        return AccuracyMetrics(
            overall_accuracy=overall_accuracy,
            correct_count=correct_count,
            total_count=total_count,
            by_category=category_accuracies,
            by_difficulty=difficulty_accuracies,
            by_value=value_accuracies,
            confidence_intervals=confidence_intervals
        )
    
    def _calculate_performance_metrics(self, model_responses: List[ModelResponse]) -> PerformanceMetrics:
        """Calculate performance-related metrics."""
        if not model_responses:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Extract response times (convert to seconds)
        response_times = []
        timeout_count = 0
        error_count = 0
        
        for response in model_responses:
            if hasattr(response, 'latency_ms') and response.latency_ms is not None:
                response_times.append(response.latency_ms / 1000.0)
            elif hasattr(response, 'response_time_ms') and response.response_time_ms is not None:
                response_times.append(response.response_time_ms / 1000.0)
            
            # Count timeouts and errors
            if hasattr(response, 'metadata') and response.metadata:
                metadata = response.metadata if isinstance(response.metadata, dict) else {}
                if metadata.get('timeout', False):
                    timeout_count += 1
                if metadata.get('error', False) or metadata.get('failed', False):
                    error_count += 1
        
        if not response_times:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, timeout_count, error_count)
        
        # Calculate statistics
        mean_time = statistics.mean(response_times)
        median_time = statistics.median(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        std_time = statistics.stdev(response_times) if len(response_times) > 1 else 0
        
        # Calculate percentiles
        sorted_times = sorted(response_times)
        p95_idx = int(0.95 * len(sorted_times))
        p99_idx = int(0.99 * len(sorted_times))
        p95_time = sorted_times[min(p95_idx, len(sorted_times) - 1)]
        p99_time = sorted_times[min(p99_idx, len(sorted_times) - 1)]
        
        return PerformanceMetrics(
            mean_response_time=mean_time,
            median_response_time=median_time,
            p95_response_time=p95_time,
            p99_response_time=p99_time,
            min_response_time=min_time,
            max_response_time=max_time,
            response_time_std=std_time,
            timeout_count=timeout_count,
            error_count=error_count
        )
    
    def _calculate_cost_metrics(self, 
                               model_responses: List[ModelResponse],
                               graded_responses: List[GradedResponse]) -> CostMetrics:
        """Calculate cost and efficiency metrics."""
        if not model_responses:
            return CostMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Extract cost and token data
        total_cost = 0.0
        total_tokens = 0
        input_tokens = 0
        output_tokens = 0
        
        for response in model_responses:
            if hasattr(response, 'cost') and response.cost:
                total_cost += float(response.cost)
            
            if hasattr(response, 'tokens_used') and response.tokens_used:
                total_tokens += response.tokens_used
            
            # Try to get input/output token breakdown
            if hasattr(response, 'metadata') and response.metadata:
                metadata = response.metadata if isinstance(response.metadata, dict) else {}
                input_tokens += metadata.get('tokens_input', 0)
                output_tokens += metadata.get('tokens_output', 0)
        
        # Calculate per-question metrics
        total_questions = len(model_responses)
        cost_per_question = total_cost / total_questions if total_questions > 0 else 0
        tokens_per_question = total_tokens / total_questions if total_questions > 0 else 0
        
        # Calculate per-correct-answer metrics
        correct_count = sum(1 for r in graded_responses if r.is_correct)
        cost_per_correct = total_cost / correct_count if correct_count > 0 else float('inf')
        tokens_per_correct = total_tokens / correct_count if correct_count > 0 else float('inf')
        
        # Calculate efficiency score relative to baseline
        baseline_cost = self.baseline_costs['cost_per_correct_answer']
        cost_efficiency = baseline_cost / cost_per_correct if cost_per_correct > 0 else 0
        
        return CostMetrics(
            total_cost=total_cost,
            cost_per_question=cost_per_question,
            cost_per_correct_answer=cost_per_correct,
            total_tokens=total_tokens,
            tokens_per_question=tokens_per_question,
            tokens_per_correct_answer=tokens_per_correct,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_efficiency_score=cost_efficiency
        )
    
    def _calculate_consistency_metrics(self,
                                     graded_responses: List[GradedResponse],
                                     question_contexts: Optional[List[Dict[str, Any]]]) -> ConsistencyMetrics:
        """Calculate consistency and reliability metrics."""
        if not graded_responses:
            return ConsistencyMetrics(0, 0, 0, 0, {}, {})
        
        # Performance variance (standard deviation of confidence scores)
        confidences = [r.confidence for r in graded_responses]
        performance_variance = statistics.stdev(confidences) if len(confidences) > 1 else 0
        
        # Response type distribution
        response_types = [r.parsed_response.response_type.value for r in graded_responses]
        response_type_counts = Counter(response_types)
        total_responses = len(graded_responses)
        response_type_dist = {
            rt: count / total_responses 
            for rt, count in response_type_counts.items()
        }
        
        # Match type distribution  
        match_types = [r.match_result.match_type.value for r in graded_responses]
        match_type_counts = Counter(match_types)
        match_type_dist = {
            mt: count / total_responses
            for mt, count in match_type_counts.items()
        }
        
        # Category consistency (how consistent performance is across categories)
        category_consistency = 0.0
        difficulty_consistency = 0.0
        
        if question_contexts:
            category_scores = defaultdict(list)
            difficulty_scores = defaultdict(list)
            
            for response, context in zip(graded_responses, question_contexts):
                category = context.get('category', 'Unknown')
                difficulty = context.get('difficulty_level', 'Unknown')
                
                score = response.confidence if response.is_correct else 0.0
                category_scores[category].append(score)
                difficulty_scores[difficulty].append(score)
            
            # Calculate consistency as inverse of coefficient of variation
            if len(category_scores) > 1:
                category_means = [statistics.mean(scores) for scores in category_scores.values() if scores]
                if category_means:
                    cat_mean = statistics.mean(category_means)
                    cat_std = statistics.stdev(category_means) if len(category_means) > 1 else 0
                    category_consistency = 1.0 - (cat_std / cat_mean) if cat_mean > 0 else 0
            
            if len(difficulty_scores) > 1:
                diff_means = [statistics.mean(scores) for scores in difficulty_scores.values() if scores]
                if diff_means:
                    diff_mean = statistics.mean(diff_means)
                    diff_std = statistics.stdev(diff_means) if len(diff_means) > 1 else 0
                    difficulty_consistency = 1.0 - (diff_std / diff_mean) if diff_mean > 0 else 0
        
        # Confidence-correctness correlation
        confidence_correlation = 0.0
        if len(graded_responses) > 1:
            correct_scores = [1.0 if r.is_correct else 0.0 for r in graded_responses]
            try:
                confidence_correlation = np.corrcoef(confidences, correct_scores)[0, 1]
                if np.isnan(confidence_correlation):
                    confidence_correlation = 0.0
            except (ValueError, IndexError):
                confidence_correlation = 0.0
        
        return ConsistencyMetrics(
            performance_variance=performance_variance,
            category_consistency_score=category_consistency,
            difficulty_consistency_score=difficulty_consistency,
            confidence_correlation=confidence_correlation,
            response_type_distribution=response_type_dist,
            match_type_distribution=match_type_dist
        )
    
    def _calculate_overall_score(self, 
                               accuracy: AccuracyMetrics,
                               performance: PerformanceMetrics,
                               cost: CostMetrics,
                               consistency: ConsistencyMetrics) -> float:
        """Calculate composite overall score."""
        # Weighted combination of different aspects
        accuracy_weight = 0.4
        consistency_weight = 0.25
        efficiency_weight = 0.2
        performance_weight = 0.15
        
        accuracy_score = accuracy.overall_accuracy
        consistency_score = max(0.0, min(1.0, consistency.confidence_correlation + 0.5))
        
        # Normalize efficiency (inverse relationship with cost)
        efficiency_score = min(1.0, cost.cost_efficiency_score)
        
        # Normalize performance (inverse relationship with response time)
        max_acceptable_time = 30.0  # 30 seconds
        performance_score = max(0.0, 1.0 - (performance.mean_response_time / max_acceptable_time))
        
        overall = (accuracy_weight * accuracy_score +
                  consistency_weight * consistency_score +
                  efficiency_weight * efficiency_score +
                  performance_weight * performance_score)
        
        return min(1.0, max(0.0, overall))
    
    def _calculate_quality_score(self, accuracy: AccuracyMetrics, consistency: ConsistencyMetrics) -> float:
        """Calculate quality score based on accuracy and consistency."""
        accuracy_component = accuracy.overall_accuracy * 0.7
        consistency_component = max(0.0, min(1.0, consistency.confidence_correlation + 0.5)) * 0.3
        return accuracy_component + consistency_component
    
    def _calculate_efficiency_score(self, cost: CostMetrics, performance: PerformanceMetrics) -> float:
        """Calculate efficiency score based on cost and performance."""
        cost_component = min(1.0, cost.cost_efficiency_score) * 0.6
        performance_component = max(0.0, 1.0 - (performance.mean_response_time / 30.0)) * 0.4
        return cost_component + performance_component
    
    def _create_metadata(self,
                        graded_responses: List[GradedResponse],
                        model_responses: List[ModelResponse], 
                        question_contexts: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Create metadata for the metrics."""
        return {
            'total_responses': len(graded_responses),
            'grading_modes_used': list(set(r.grading_metadata.get('grading_mode', 'unknown') 
                                         for r in graded_responses)),
            'question_categories': list(set(ctx.get('category', 'Unknown') 
                                          for ctx in (question_contexts or []))),
            'calculation_timestamp': datetime.now().isoformat(),
            'has_partial_credit': any(r.partial_credit > 0 for r in graded_responses),
            'avg_grading_time': statistics.mean(
                r.grading_metadata.get('grading_time_ms', 0) 
                for r in graded_responses
            ) if graded_responses else 0
        }
    
    def compare_metrics(self, metrics1: ComprehensiveMetrics, 
                       metrics2: ComprehensiveMetrics) -> Dict[str, Any]:
        """Compare two sets of metrics."""
        comparison = {
            'model_comparison': f"{metrics1.model_name} vs {metrics2.model_name}",
            'accuracy_diff': metrics1.accuracy.overall_accuracy - metrics2.accuracy.overall_accuracy,
            'cost_diff': metrics1.cost.cost_per_correct_answer - metrics2.cost.cost_per_correct_answer,
            'performance_diff': metrics1.performance.mean_response_time - metrics2.performance.mean_response_time,
            'overall_score_diff': metrics1.overall_score - metrics2.overall_score,
            'winner': {
                'accuracy': metrics1.model_name if metrics1.accuracy.overall_accuracy > metrics2.accuracy.overall_accuracy else metrics2.model_name,
                'cost_efficiency': metrics1.model_name if metrics1.cost.cost_per_correct_answer < metrics2.cost.cost_per_correct_answer else metrics2.model_name,
                'performance': metrics1.model_name if metrics1.performance.mean_response_time < metrics2.performance.mean_response_time else metrics2.model_name,
                'overall': metrics1.model_name if metrics1.overall_score > metrics2.overall_score else metrics2.model_name
            },
            'detailed_comparison': {
                'accuracy': {
                    metrics1.model_name: metrics1.accuracy.overall_accuracy,
                    metrics2.model_name: metrics2.accuracy.overall_accuracy
                },
                'cost_per_correct': {
                    metrics1.model_name: metrics1.cost.cost_per_correct_answer,
                    metrics2.model_name: metrics2.cost.cost_per_correct_answer
                },
                'response_time': {
                    metrics1.model_name: metrics1.performance.mean_response_time,
                    metrics2.model_name: metrics2.performance.mean_response_time
                }
            }
        }
        
        return comparison
    
    def aggregate_metrics(self, metrics_list: List[ComprehensiveMetrics]) -> Dict[str, Any]:
        """Aggregate metrics across multiple benchmark runs."""
        if not metrics_list:
            return {}
        
        aggregated = {
            'models_included': [m.model_name for m in metrics_list],
            'run_count': len(metrics_list),
            'best_overall': max(metrics_list, key=lambda m: m.overall_score).model_name,
            'best_accuracy': max(metrics_list, key=lambda m: m.accuracy.overall_accuracy).model_name,
            'most_cost_effective': min(metrics_list, key=lambda m: m.cost.cost_per_correct_answer).model_name,
            'fastest': min(metrics_list, key=lambda m: m.performance.mean_response_time).model_name,
            
            'summary_stats': {
                'accuracy_range': (
                    min(m.accuracy.overall_accuracy for m in metrics_list),
                    max(m.accuracy.overall_accuracy for m in metrics_list)
                ),
                'cost_range': (
                    min(m.cost.cost_per_correct_answer for m in metrics_list),
                    max(m.cost.cost_per_correct_answer for m in metrics_list)
                ),
                'performance_range': (
                    min(m.performance.mean_response_time for m in metrics_list),
                    max(m.performance.mean_response_time for m in metrics_list)
                )
            }
        }
        
        return aggregated