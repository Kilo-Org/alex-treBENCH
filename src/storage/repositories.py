"""
Repository Pattern Implementation

Data access layer using repository pattern for clean separation
between business logic and database operations.
"""

import uuid
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc
import pandas as pd
from datetime import datetime, timedelta
import json

from .models import (
    Benchmark, BenchmarkQuestion, ModelResponse, ModelPerformanceSummary,
    Question, BenchmarkResult, BenchmarkRun, ModelPerformance, create_benchmark_run
)
from src.core.exceptions import DatabaseError, ValidationError
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BenchmarkRepository:
    """Repository for benchmark operations."""
    
    def __init__(self, session: Session):
        """Initialize repository with database session."""
        self.session = session
    
    def create_benchmark(self, benchmark: Benchmark) -> Benchmark:
        """Create a new benchmark."""
        try:
            self.session.add(benchmark)
            self.session.commit()
            self.session.refresh(benchmark)
            return benchmark
        except Exception as e:
            self.session.rollback()
            raise DatabaseError(
                f"Failed to create benchmark: {str(e)}",
                operation="create",
                table="benchmarks"
            ) from e
    
    def get_benchmark(self, benchmark_id: int) -> Optional[Benchmark]:
        """Get benchmark by ID."""
        try:
            return self.session.query(Benchmark).filter(
                Benchmark.id == benchmark_id
            ).first()
        except Exception as e:
            raise DatabaseError(
                f"Failed to get benchmark: {str(e)}",
                operation="get",
                table="benchmarks"
            ) from e
    
    def get_benchmark_by_name(self, name: str) -> Optional[Benchmark]:
        """Get benchmark by name."""
        try:
            return self.session.query(Benchmark).filter(
                Benchmark.name == name
            ).first()
        except Exception as e:
            raise DatabaseError(
                f"Failed to get benchmark by name: {str(e)}",
                operation="get",
                table="benchmarks"
            ) from e
    
    def list_benchmarks(self, limit: int = 100, offset: int = 0) -> List[Benchmark]:
        """List benchmarks with pagination."""
        try:
            return self.session.query(Benchmark).order_by(
                desc(Benchmark.created_at)
            ).limit(limit).offset(offset).all()
        except Exception as e:
            raise DatabaseError(
                f"Failed to list benchmarks: {str(e)}",
                operation="list",
                table="benchmarks"
            ) from e
    
    def update_benchmark_status(self, benchmark_id: int, status: str) -> None:
        """Update benchmark status."""
        try:
            benchmark = self.get_benchmark(benchmark_id)
            if benchmark:
                benchmark.status = status
                if status == 'completed':
                    benchmark.completed_at = func.current_timestamp()
                self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise DatabaseError(
                f"Failed to update benchmark status: {str(e)}",
                operation="update",
                table="benchmarks"
            ) from e
    
    def delete_benchmark(self, benchmark_id: int) -> bool:
        """Delete benchmark and all related data."""
        try:
            benchmark = self.get_benchmark(benchmark_id)
            if benchmark:
                self.session.delete(benchmark)
                self.session.commit()
                return True
            return False
        except Exception as e:
            self.session.rollback()
            raise DatabaseError(
                f"Failed to delete benchmark: {str(e)}",
                operation="delete",
                table="benchmarks"
            ) from e
    
    def save_benchmark_run(self, run_data: Dict[str, Any]) -> Benchmark:
        """Save a benchmark run with comprehensive metadata."""
        try:
            benchmark = create_benchmark_run(
                name=run_data.get('name', 'Unnamed Benchmark'),
                description=run_data.get('description', ''),
                sample_size=run_data.get('sample_size', 1000)
            )
            
            # Set additional fields
            if run_data.get('models_tested'):
                benchmark.models_tested = run_data['models_tested']
            
            if run_data.get('status'):
                benchmark.status = run_data['status']
            
            if run_data.get('completed_at'):
                benchmark.completed_at = run_data['completed_at']
            
            self.session.add(benchmark)
            self.session.commit()
            self.session.refresh(benchmark)
            
            logger.info(f"Saved benchmark run: {benchmark.id}")
            return benchmark
            
        except Exception as e:
            self.session.rollback()
            raise DatabaseError(
                f"Failed to save benchmark run: {str(e)}",
                operation="create",
                table="benchmarks"
            ) from e
    
    def save_benchmark_results(self, results: Dict[str, Any]) -> bool:
        """Save comprehensive benchmark results including metrics and summaries."""
        try:
            benchmark_id = results.get('benchmark_id')
            if not benchmark_id:
                raise ValueError("benchmark_id is required in results data")
            
            # Update benchmark status
            benchmark = self.get_benchmark(benchmark_id)
            if not benchmark:
                raise ValueError(f"Benchmark {benchmark_id} not found")
            
            benchmark.status = results.get('status', 'completed')
            benchmark.completed_at = results.get('completed_at', func.current_timestamp())
            
            # Save performance summaries if provided
            if results.get('performance_summaries'):
                perf_repo = PerformanceRepository(self.session)
                for summary_data in results['performance_summaries']:
                    perf_repo.save_performance_summary(summary_data)
            
            self.session.commit()
            logger.info(f"Saved benchmark results for benchmark {benchmark_id}")
            return True
            
        except Exception as e:
            self.session.rollback()
            raise DatabaseError(
                f"Failed to save benchmark results: {str(e)}",
                operation="update",
                table="benchmarks"
            ) from e
    
    def get_benchmark_history(self, model_id: str, limit: int = 50) -> List[Benchmark]:
        """Get benchmark history for a specific model."""
        try:
            # Search for benchmarks where the model was tested
            benchmarks = self.session.query(Benchmark).filter(
                Benchmark.models_tested.like(f'%{model_id}%')
            ).order_by(desc(Benchmark.created_at)).limit(limit).all()
            
            # Filter more precisely (JSON contains check would be better with proper JSON column)
            filtered_benchmarks = []
            for benchmark in benchmarks:
                if model_id in benchmark.models_tested_list:
                    filtered_benchmarks.append(benchmark)
            
            logger.info(f"Retrieved {len(filtered_benchmarks)} benchmark history entries for {model_id}")
            return filtered_benchmarks
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to get benchmark history for {model_id}: {str(e)}",
                operation="get",
                table="benchmarks"
            ) from e
    
    def get_comparative_results(self, model_ids: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Get comparative results across multiple models."""
        try:
            # Find benchmarks that include all specified models
            results = []
            
            for model_id in model_ids:
                benchmarks = self.session.query(Benchmark).filter(
                    Benchmark.models_tested.like(f'%{model_id}%')
                ).order_by(desc(Benchmark.created_at)).limit(limit).all()
                
                for benchmark in benchmarks:
                    if model_id in benchmark.models_tested_list:
                        # Get performance data for this model in this benchmark
                        perf_repo = PerformanceRepository(self.session)
                        performance = perf_repo.get_model_performance(benchmark.id, model_id)
                        
                        if performance:
                            results.append({
                                'benchmark_id': benchmark.id,
                                'benchmark_name': benchmark.name,
                                'model_name': model_id,
                                'created_at': benchmark.created_at,
                                'accuracy': float(performance.accuracy_rate) if performance.accuracy_rate else 0.0,
                                'avg_response_time': float(performance.avg_response_time_ms) if performance.avg_response_time_ms else 0.0,
                                'total_cost': float(performance.total_cost_usd) if performance.total_cost_usd else 0.0,
                                'total_questions': performance.total_questions,
                                'correct_answers': performance.correct_answers
                            })
            
            # Sort by benchmark creation date (most recent first)
            results.sort(key=lambda x: x['created_at'], reverse=True)
            
            logger.info(f"Retrieved comparative results for {len(model_ids)} models: {len(results)} entries")
            return results
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to get comparative results: {str(e)}",
                operation="get",
                table="benchmarks"
            ) from e
    
    def get_benchmark_by_id(self, benchmark_id: int) -> Optional[Benchmark]:
        """Get benchmark by ID (alias for existing get_benchmark method)."""
        return self.get_benchmark(benchmark_id)
    
    def get_benchmark_summary_stats(self, benchmark_id: int) -> Dict[str, Any]:
        """Get summary statistics for a benchmark."""
        try:
            benchmark = self.get_benchmark(benchmark_id)
            if not benchmark:
                return {}
            
            # Get related data
            question_repo = QuestionRepository(self.session)
            response_repo = ResponseRepository(self.session)
            perf_repo = PerformanceRepository(self.session)
            
            questions = question_repo.get_benchmark_questions(benchmark_id)
            responses = response_repo.get_benchmark_responses(benchmark_id)
            performance_summaries = perf_repo.get_benchmark_performance(benchmark_id)
            
            # Calculate summary stats
            total_responses = len(responses)
            successful_responses = len([r for r in responses if r.is_correct])
            
            stats = {
                'benchmark_id': benchmark_id,
                'name': benchmark.name,
                'status': benchmark.status,
                'created_at': benchmark.created_at,
                'completed_at': benchmark.completed_at,
                'models_tested': benchmark.models_tested_list,
                'total_questions': len(questions),
                'total_responses': total_responses,
                'successful_responses': successful_responses,
                'overall_accuracy': successful_responses / total_responses if total_responses > 0 else 0.0,
                'models_performance_count': len(performance_summaries),
                'categories': list(set(q.category for q in questions if q.category)),
                'difficulty_levels': list(set(q.difficulty_level for q in questions if q.difficulty_level)),
                'value_range': {
                    'min': min((q.value for q in questions if q.value), default=0),
                    'max': max((q.value for q in questions if q.value), default=0)
                }
            }
            
            return stats
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to get benchmark summary stats: {str(e)}",
                operation="get",
                table="benchmarks"
            ) from e


class QuestionRepository:
    """Repository for benchmark question operations."""
    
    def __init__(self, session: Session):
        """Initialize repository with database session."""
        self.session = session
    
    def add_questions(self, questions: List[BenchmarkQuestion]) -> List[BenchmarkQuestion]:
        """Add multiple questions to a benchmark."""
        try:
            self.session.add_all(questions)
            self.session.commit()
            return questions
        except Exception as e:
            self.session.rollback()
            raise DatabaseError(
                f"Failed to add questions: {str(e)}",
                operation="create",
                table="benchmark_questions"
            ) from e
    
    def get_benchmark_questions(self, benchmark_id: int) -> List[BenchmarkQuestion]:
        """Get all questions for a benchmark."""
        try:
            # Join through BenchmarkResult since Question is a global cache
            # and doesn't have benchmark_id directly
            from .models import BenchmarkResult
            
            results = self.session.query(BenchmarkQuestion).join(
                BenchmarkResult, BenchmarkResult.question_id == BenchmarkQuestion.id
            ).filter(
                BenchmarkResult.benchmark_run_id == benchmark_id
            ).distinct().all()
            
            return results
        except Exception as e:
            raise DatabaseError(
                f"Failed to get benchmark questions: {str(e)}",
                operation="get",
                table="benchmark_questions"
            ) from e
    
    def get_questions_by_category(self, benchmark_id: int, category: str) -> List[BenchmarkQuestion]:
        """Get questions by category."""
        try:
            from .models import BenchmarkResult
            
            return self.session.query(BenchmarkQuestion).join(
                BenchmarkResult, BenchmarkResult.question_id == BenchmarkQuestion.id
            ).filter(
                and_(
                    BenchmarkResult.benchmark_run_id == benchmark_id,
                    BenchmarkQuestion.category == category
                )
            ).distinct().all()
        except Exception as e:
            raise DatabaseError(
                f"Failed to get questions by category: {str(e)}",
                operation="get",
                table="benchmark_questions"
            ) from e
    
    def get_questions_by_difficulty(self, benchmark_id: int, difficulty: str) -> List[BenchmarkQuestion]:
        """Get questions by difficulty level."""
        try:
            from .models import BenchmarkResult
            
            return self.session.query(BenchmarkQuestion).join(
                BenchmarkResult, BenchmarkResult.question_id == BenchmarkQuestion.id
            ).filter(
                and_(
                    BenchmarkResult.benchmark_run_id == benchmark_id,
                    BenchmarkQuestion.difficulty_level == difficulty
                )
            ).distinct().all()
        except Exception as e:
            raise DatabaseError(
                f"Failed to get questions by difficulty: {str(e)}",
                operation="get",
                table="benchmark_questions"
            ) from e
    
    def save_questions(self, questions_df: pd.DataFrame, benchmark_id: int) -> List[BenchmarkQuestion]:
        """
        Bulk save questions from DataFrame to database.
        
        Args:
            questions_df: DataFrame with question data
            benchmark_id: ID of the benchmark these questions belong to (not used for Question model)
            
        Returns:
            List of saved BenchmarkQuestion objects
            
        Raises:
            DatabaseError: If save operation fails
        """
        try:
            logger.info(f"Saving {len(questions_df)} questions to database")
            
            questions = []
            for _, row in questions_df.iterrows():
                # Generate unique ID if not provided
                question_id = str(row.get('id', row.get('question_id', f"q_{uuid.uuid4().hex[:8]}")))
                
                question = BenchmarkQuestion(
                    id=question_id,
                    question_text=str(row.get('question', row.get('clue', ''))),
                    correct_answer=str(row.get('answer', row.get('response', ''))),
                    category=str(row.get('category', '')) if pd.notna(row.get('category')) else None,
                    value=int(row.get('value', 0)) if pd.notna(row.get('value')) else None,
                    difficulty_level=str(row.get('difficulty_level', '')) if pd.notna(row.get('difficulty_level')) else None,
                    air_date=pd.to_datetime(row.get('air_date')) if pd.notna(row.get('air_date')) else None,
                    show_number=int(row.get('show_number', 0)) if pd.notna(row.get('show_number')) else None,
                    round=str(row.get('round', '')) if pd.notna(row.get('round')) else None
                )
                questions.append(question)
            
            # Bulk insert with conflict handling
            for question in questions:
                existing = self.session.query(BenchmarkQuestion).filter(
                    BenchmarkQuestion.id == question.id
                ).first()
                
                if not existing:
                    self.session.add(question)
            
            self.session.commit()
            
            logger.info(f"Successfully saved {len(questions)} questions")
            return questions
            
        except Exception as e:
            self.session.rollback()
            raise DatabaseError(
                f"Failed to save questions: {str(e)}",
                operation="bulk_create",
                table="benchmark_questions"
            ) from e
    
    def get_questions(self, filters: Optional[Dict[str, Any]] = None,
                     limit: Optional[int] = None, offset: int = 0) -> List[BenchmarkQuestion]:
        """
        Retrieve questions with optional filtering.
        
        Args:
            filters: Dictionary of filter criteria
                - benchmark_id: int
                - categories: List[str]
                - difficulty_levels: List[str]
                - min_value: int
                - max_value: int
            limit: Maximum number of questions to return
            offset: Number of questions to skip
            
        Returns:
            List of BenchmarkQuestion objects
            
        Raises:
            DatabaseError: If query fails
        """
        try:
            query = self.session.query(BenchmarkQuestion)
            
            if filters:
                if filters.get('benchmark_id'):
                    # Join through BenchmarkResult since Question doesn't have benchmark_id directly
                    from .models import BenchmarkResult
                    query = query.join(BenchmarkResult, BenchmarkResult.question_id == BenchmarkQuestion.id).filter(
                        BenchmarkResult.benchmark_run_id == filters['benchmark_id']
                    ).distinct()
                
                if filters.get('categories'):
                    query = query.filter(BenchmarkQuestion.category.in_(filters['categories']))
                
                if filters.get('difficulty_levels'):
                    query = query.filter(BenchmarkQuestion.difficulty_level.in_(filters['difficulty_levels']))
                
                if filters.get('min_value') is not None:
                    query = query.filter(BenchmarkQuestion.value >= filters['min_value'])
                
                if filters.get('max_value') is not None:
                    query = query.filter(BenchmarkQuestion.value <= filters['max_value'])
            
            # Apply pagination
            query = query.offset(offset)
            if limit:
                query = query.limit(limit)
            
            return query.all()
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to get questions with filters: {str(e)}",
                operation="get",
                table="benchmark_questions"
            ) from e
    
    def get_random_questions(self, n: int, benchmark_id: Optional[int] = None,
                           category: Optional[str] = None,
                           difficulty: Optional[str] = None) -> List[BenchmarkQuestion]:
        """
        Get random questions with optional filtering.
        
        Args:
            n: Number of questions to return
            benchmark_id: Optional benchmark ID filter
            category: Optional category filter
            difficulty: Optional difficulty level filter
            
        Returns:
            List of randomly selected BenchmarkQuestion objects
            
        Raises:
            DatabaseError: If query fails
        """
        try:
            query = self.session.query(BenchmarkQuestion).order_by(func.random())
            
            # Apply filters
            if benchmark_id:
                from .models import BenchmarkResult
                query = query.join(BenchmarkResult, BenchmarkResult.question_id == BenchmarkQuestion.id).filter(
                    BenchmarkResult.benchmark_run_id == benchmark_id
                ).distinct()
            if category:
                query = query.filter(BenchmarkQuestion.category == category)
            if difficulty:
                query = query.filter(BenchmarkQuestion.difficulty_level == difficulty)
            
            return query.limit(n).all()
        except Exception as e:
            raise DatabaseError(
                f"Failed to get random questions: {str(e)}",
                operation="get",
                table="benchmark_questions"
            ) from e
    
    def get_question_statistics(self, benchmark_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get comprehensive statistics about questions in the database.
        
        Args:
            benchmark_id: Optional benchmark ID to filter by
            
        Returns:
            Dictionary with question statistics
            
        Raises:
            DatabaseError: If query fails
        """
        try:
            # Use Question model directly (BenchmarkQuestion is just an alias)
            base_query = self.session.query(Question)
            
            # If benchmark_id is provided, we need to join with BenchmarkResult 
            # to filter questions that were part of that benchmark
            if benchmark_id:
                base_query = base_query.join(BenchmarkResult).filter(
                    BenchmarkResult.benchmark_run_id == benchmark_id
                ).distinct()
            
            stats = {
                'total_questions': base_query.count(),
                'category_distribution': {},
                'difficulty_distribution': {},
                'value_distribution': {},
                'unique_categories': 0,
                'value_range': {}
            }
            
            # Category distribution
            category_counts = base_query.filter(Question.category.isnot(None))\
                                      .with_entities(Question.category, func.count())\
                                      .group_by(Question.category)\
                                      .all()
            
            stats['category_distribution'] = {cat: count for cat, count in category_counts}
            stats['unique_categories'] = len(category_counts)
            
            # Difficulty distribution
            difficulty_counts = base_query.filter(Question.difficulty_level.isnot(None))\
                                         .with_entities(Question.difficulty_level, func.count())\
                                         .group_by(Question.difficulty_level)\
                                         .all()
            
            stats['difficulty_distribution'] = {diff: count for diff, count in difficulty_counts}
            
            # Value statistics
            value_stats = base_query.filter(Question.value.isnot(None))\
                                   .with_entities(
                                       func.min(Question.value),
                                       func.max(Question.value),
                                       func.avg(Question.value),
                                       func.count(Question.value)
                                   ).first()
            
            if value_stats and value_stats[3] > 0:  # If we have values
                stats['value_range'] = {
                    'min': value_stats[0],
                    'max': value_stats[1],
                    'average': round(float(value_stats[2]), 2),
                    'count': value_stats[3]
                }
                
                # Value distribution by ranges
                value_ranges = [
                    ('Low ($1-600)', 1, 600),
                    ('Medium ($601-1200)', 601, 1200),
                    ('High ($1201+)', 1201, 999999)
                ]
                
                for range_name, min_val, max_val in value_ranges:
                    count = base_query.filter(
                        and_(
                            Question.value >= min_val,
                            Question.value <= max_val
                        )
                    ).count()
                    stats['value_distribution'][range_name] = count
            
            logger.info(f"Generated statistics for {stats['total_questions']} questions")
            return stats
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to get question statistics: {str(e)}",
                operation="get",
                table="questions"
            ) from e


class ResponseRepository:
    """Repository for model response operations."""
    
    def __init__(self, session: Session):
        """Initialize repository with database session."""
        self.session = session
    
    def save_response(self, response: ModelResponse) -> ModelResponse:
        """Save a model response."""
        try:
            self.session.add(response)
            self.session.commit()
            self.session.refresh(response)
            return response
        except Exception as e:
            self.session.rollback()
            raise DatabaseError(
                f"Failed to save response: {str(e)}",
                operation="create",
                table="model_responses"
            ) from e
    
    def save_responses(self, responses: List[ModelResponse]) -> List[ModelResponse]:
        """Save multiple model responses."""
        try:
            self.session.add_all(responses)
            self.session.commit()
            return responses
        except Exception as e:
            self.session.rollback()
            raise DatabaseError(
                f"Failed to save responses: {str(e)}",
                operation="create",
                table="model_responses"
            ) from e
    
    def get_benchmark_responses(self, benchmark_id: int, model_name: str = None) -> List[ModelResponse]:
        """Get all responses for a benchmark, optionally filtered by model."""
        try:
            query = self.session.query(ModelResponse).filter(
                ModelResponse.benchmark_run_id == benchmark_id
            )
            
            if model_name:
                query = query.filter(ModelResponse.model_name == model_name)
            
            return query.all()
        except Exception as e:
            raise DatabaseError(
                f"Failed to get benchmark responses: {str(e)}",
                operation="get",
                table="model_responses"
            ) from e
    
    def get_model_accuracy(self, benchmark_id: int, model_name: str) -> float:
        """Get accuracy rate for a specific model in a benchmark."""
        try:
            # Get total responses
            total_responses = self.session.query(ModelResponse).filter(
                and_(
                    ModelResponse.benchmark_run_id == benchmark_id,
                    ModelResponse.model_name == model_name
                )
            ).count()
            
            if total_responses == 0:
                return 0.0
            
            # Get correct responses
            correct_responses = self.session.query(ModelResponse).filter(
                and_(
                    ModelResponse.benchmark_run_id == benchmark_id,
                    ModelResponse.model_name == model_name,
                    ModelResponse.is_correct == True
                )
            ).count()
            
            return correct_responses / total_responses
        except Exception as e:
            raise DatabaseError(
                f"Failed to get model accuracy: {str(e)}",
                operation="get",
                table="model_responses"
            ) from e
    
    def get_response_statistics(self, benchmark_id: int, model_name: str) -> Dict[str, Any]:
        """Get comprehensive response statistics for a model."""
        try:
            responses = self.get_benchmark_responses(benchmark_id, model_name)
            
            if not responses:
                return {}
            
            total_responses = len(responses)
            correct_responses = sum(1 for r in responses if r.is_correct)
            total_cost = sum(float(r.cost_usd or 0) for r in responses)
            total_time = sum(r.response_time_ms or 0 for r in responses)
            total_tokens = sum(r.tokens_generated or 0 for r in responses)
            
            return {
                'total_responses': total_responses,
                'correct_responses': correct_responses,
                'accuracy_rate': correct_responses / total_responses if total_responses > 0 else 0,
                'total_cost_usd': total_cost,
                'avg_response_time_ms': total_time / total_responses if total_responses > 0 else 0,
                'total_tokens': total_tokens,
                'avg_tokens_per_response': total_tokens / total_responses if total_responses > 0 else 0
            }
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to get response statistics: {str(e)}",
                operation="get",
                table="model_responses"
            ) from e


class PerformanceRepository:
    """Repository for performance summary operations."""
    
    def __init__(self, session: Session):
        """Initialize repository with database session."""
        self.session = session
    
    def save_performance_summary(self, summary: ModelPerformanceSummary) -> ModelPerformanceSummary:
        """Save a performance summary."""
        try:
            self.session.add(summary)
            self.session.commit()
            self.session.refresh(summary)
            return summary
        except Exception as e:
            self.session.rollback()
            raise DatabaseError(
                f"Failed to save performance summary: {str(e)}",
                operation="create",
                table="model_performance_summary"
            ) from e
    
    def save_performance(self, performance: ModelPerformance) -> ModelPerformance:
        """Save a performance record with Jeopardy scores."""
        try:
            self.session.add(performance)
            self.session.commit()
            self.session.refresh(performance)
            return performance
        except Exception as e:
            self.session.rollback()
            raise DatabaseError(
                f"Failed to save performance record: {str(e)}",
                operation="create",
                table="model_performance"
            ) from e
    
    def get_performances_by_benchmark(self, benchmark_id: int) -> List[ModelPerformance]:
        """Get all performance records for a benchmark."""
        try:
            return self.session.query(ModelPerformance).filter(
                ModelPerformance.benchmark_run_id == benchmark_id
            ).all()
        except Exception as e:
            raise DatabaseError(
                f"Failed to get performances by benchmark: {str(e)}",
                operation="get",
                table="model_performance"
            ) from e
    
    def get_jeopardy_leaderboard(self, limit: int = 20) -> List[ModelPerformance]:
        """Get Jeopardy leaderboard sorted by total score."""
        try:
            return self.session.query(ModelPerformance).filter(
                ModelPerformance.jeopardy_score.isnot(None)
            ).order_by(desc(ModelPerformance.jeopardy_score)).limit(limit).all()
        except Exception as e:
            raise DatabaseError(
                f"Failed to get Jeopardy leaderboard: {str(e)}",
                operation="get",
                table="model_performance"
            ) from e
    
    def get_performance_by_model_and_benchmark(self, model_name: str, benchmark_id: int) -> Optional[ModelPerformance]:
        """Get performance record for a specific model and benchmark."""
        try:
            return self.session.query(ModelPerformance).filter(
                and_(
                    ModelPerformance.model_name == model_name,
                    ModelPerformance.benchmark_run_id == benchmark_id
                )
            ).first()
        except Exception as e:
            raise DatabaseError(
                f"Failed to get performance record: {str(e)}",
                operation="get",
                table="model_performance"
            ) from e

    def get_benchmark_performance(self, benchmark_id: int) -> List[ModelPerformanceSummary]:
        """Get performance summaries for all models in a benchmark."""
        try:
            return self.session.query(ModelPerformanceSummary).filter(
                ModelPerformanceSummary.benchmark_run_id == benchmark_id
            ).all()
        except Exception as e:
            raise DatabaseError(
                f"Failed to get benchmark performance: {str(e)}",
                operation="get",
                table="model_performance_summary"
            ) from e
    
    def get_model_performance(self, benchmark_id: int, model_name: str) -> Optional[ModelPerformanceSummary]:
        """Get performance summary for a specific model."""
        try:
            return self.session.query(ModelPerformanceSummary).filter(
                and_(
                    ModelPerformanceSummary.benchmark_run_id == benchmark_id,
                    ModelPerformanceSummary.model_name == model_name
                )
            ).first()
        except Exception as e:
            raise DatabaseError(
                f"Failed to get model performance: {str(e)}",
                operation="get",
                table="model_performance_summary"
            ) from e
    
    def get_top_performing_models(self, metric: str = 'accuracy', limit: int = 10) -> List[Dict[str, Any]]:
        """Get top performing models based on specified metric."""
        try:
            if metric == 'accuracy':
                order_column = desc(ModelPerformanceSummary.accuracy_rate)
            elif metric == 'cost_efficiency':
                order_column = asc(ModelPerformanceSummary.cost_per_correct_answer)
            elif metric == 'jeopardy_score':
                order_column = desc(ModelPerformance.jeopardy_score)
                # Use ModelPerformance instead of ModelPerformanceSummary for Jeopardy scores
                results = self.session.query(ModelPerformance).filter(
                    ModelPerformance.jeopardy_score.isnot(None)
                ).order_by(order_column).limit(limit).all()
                
                return [
                    {
                        'model_name': perf.model_name,
                        'benchmark_run_id': perf.benchmark_run_id,
                        'jeopardy_score': perf.jeopardy_score,
                        'accuracy_rate': float(perf.accuracy_rate) if perf.accuracy_rate else 0.0,
                        'total_questions': perf.total_questions,
                        'correct_answers': perf.correct_answers
                    }
                    for perf in results
                ]
            else:
                order_column = desc(ModelPerformanceSummary.accuracy_rate)  # Default fallback
            
            results = self.session.query(ModelPerformanceSummary).order_by(order_column).limit(limit).all()
            
            return [
                {
                    'model_name': perf.model_name,
                    'benchmark_run_id': perf.benchmark_run_id,
                    'accuracy_rate': float(perf.accuracy_rate) if perf.accuracy_rate else 0.0,
                    'total_cost': float(perf.total_cost_usd) if perf.total_cost_usd else 0.0,
                    'avg_response_time': float(perf.avg_response_time_ms) if perf.avg_response_time_ms else 0.0,
                    'total_questions': perf.total_questions,
                    'correct_answers': perf.correct_answers
                }
                for perf in results
            ]
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to get top performing models: {str(e)}",
                operation="get",
                table="model_performance_summary"
            ) from e
