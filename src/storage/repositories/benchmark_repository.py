"""
Benchmark Repository

Repository class for managing benchmark run data access.
"""

import json
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.core.exceptions import DatabaseError
from src.storage.models import BenchmarkRun, BenchmarkResult, Question


class BenchmarkRepository:
    """Repository for managing benchmark run data access."""
    
    def __init__(self, session: Optional[Session] = None):
        """Initialize repository with optional session."""
        self.session = session
    
    def get_benchmark_by_id(self, benchmark_id: int) -> Optional[BenchmarkRun]:
        """Get benchmark by ID."""
        try:
            return self.session.query(BenchmarkRun).filter(BenchmarkRun.id == benchmark_id).first()
        except Exception as e:
            raise DatabaseError(f"Failed to get benchmark {benchmark_id}: {str(e)}")
    
    def get_benchmark_summary_stats(self, run_id: int) -> Optional[Dict[str, Any]]:
        """Get comprehensive summary statistics for a benchmark run."""
        try:
            # Get the benchmark run
            benchmark_run = self.get_benchmark_by_id(run_id)
            if not benchmark_run:
                return None
            
            # Get all results for this benchmark run
            results = self.session.query(BenchmarkResult).filter(
                BenchmarkResult.benchmark_run_id == run_id
            ).all()
            
            # Get question IDs from results to fetch question metadata
            question_ids = [result.question_id for result in results]
            questions = []
            if question_ids:
                questions = self.session.query(Question).filter(
                    Question.id.in_(question_ids)
                ).all()
            
            # Create questions lookup dict
            questions_dict = {q.id: q for q in questions}
            
            # Calculate metrics
            total_responses = len(results)
            correct_responses = len([r for r in results if r.is_correct])
            overall_accuracy = correct_responses / total_responses if total_responses > 0 else 0
            
            # Parse models_tested if it's a JSON string
            models_tested = []
            if benchmark_run.models_tested:
                try:
                    if isinstance(benchmark_run.models_tested, str):
                        models_tested = json.loads(benchmark_run.models_tested)
                    elif isinstance(benchmark_run.models_tested, list):
                        models_tested = benchmark_run.models_tested
                    else:
                        models_tested = [str(benchmark_run.models_tested)]
                except (json.JSONDecodeError, TypeError):
                    models_tested = [str(benchmark_run.models_tested)]
            
            # Get unique categories and difficulty levels from questions
            categories = list(set([questions_dict[qid].category for qid in question_ids
                                 if qid in questions_dict and questions_dict[qid].category]))
            
            difficulty_levels = list(set([questions_dict[qid].difficulty_level for qid in question_ids
                                        if qid in questions_dict and questions_dict[qid].difficulty_level]))
            
            # Calculate value range
            question_values = [questions_dict[qid].value for qid in question_ids
                             if qid in questions_dict and questions_dict[qid].value is not None]
            value_range = {}
            if question_values:
                value_range = {
                    'min': min(question_values),
                    'max': max(question_values)
                }
            
            # Build the stats dictionary
            stats = {
                'name': benchmark_run.name,
                'status': benchmark_run.status,
                'created_at': benchmark_run.created_at,
                'completed_at': getattr(benchmark_run, 'completed_at', None),
                'models_tested': models_tested,
                'total_questions': len(question_ids),
                'total_responses': total_responses,
                'overall_accuracy': overall_accuracy,
                'categories': categories,
                'difficulty_levels': difficulty_levels,
                'value_range': value_range,
                'total_cost_usd': float(benchmark_run.total_cost_usd) if benchmark_run.total_cost_usd else 0.0,
                'total_tokens': benchmark_run.total_tokens,
                'avg_response_time_ms': float(benchmark_run.avg_response_time_ms) if benchmark_run.avg_response_time_ms else None,
                'error_count': benchmark_run.error_count
            }
            
            return stats
            
        except Exception as e:
            raise DatabaseError(f"Failed to get benchmark summary stats for run {run_id}: {str(e)}")
    
    def save_benchmark_run(self, benchmark: BenchmarkRun) -> BenchmarkRun:
        """Save a benchmark run."""
        try:
            self.session.add(benchmark)
            self.session.commit()
            self.session.refresh(benchmark)
            return benchmark
        except Exception as e:
            self.session.rollback()
            raise DatabaseError(f"Failed to save benchmark: {str(e)}")
    
    def list_benchmarks(self, limit: Optional[int] = None) -> List[BenchmarkRun]:
        """List benchmark runs ordered by most recent first."""
        try:
            query = self.session.query(BenchmarkRun).order_by(BenchmarkRun.created_at.desc())
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
        except Exception as e:
            raise DatabaseError(f"Failed to list benchmarks: {str(e)}")