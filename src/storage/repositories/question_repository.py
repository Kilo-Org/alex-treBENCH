"""
Question Repository

Repository class for managing Jeopardy question data access.
"""

import pandas as pd
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_

from src.core.database import get_db_session
from src.core.exceptions import DatabaseError
from src.storage.models import Question


class QuestionRepository:
    """Repository for managing question data access."""
    
    def __init__(self, session: Optional[Session] = None):
        """Initialize repository with optional session."""
        self.session = session
    
    def get_all_questions(self, limit: Optional[int] = None, offset: int = 0) -> List[Question]:
        """Get all questions with optional pagination."""
        try:
            query = self.session.query(Question).order_by(Question.id)
            
            if offset:
                query = query.offset(offset)
            if limit:
                query = query.limit(limit)
                
            return query.all()
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to get questions: {str(e)}",
                operation="query",
                table="questions"
            ) from e
    
    def get_questions_by_category(self, category: str, limit: Optional[int] = None) -> List[Question]:
        """Get questions filtered by category."""
        try:
            query = self.session.query(Question).filter(Question.category == category)
            
            if limit:
                query = query.limit(limit)
                
            return query.all()
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to get questions by category {category}: {str(e)}",
                operation="query",
                table="questions"
            ) from e
    
    def get_question_by_id(self, question_id: str) -> Optional[Question]:
        """Get question by ID."""
        try:
            return self.session.query(Question).filter(Question.id == question_id).first()
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to get question {question_id}: {str(e)}",
                operation="get",
                table="questions"
            ) from e
    
    def get_questions_by_difficulty(self, difficulty: str, limit: Optional[int] = None) -> List[Question]:
        """Get questions filtered by difficulty level."""
        try:
            query = self.session.query(Question).filter(Question.difficulty_level == difficulty)
            
            if limit:
                query = query.limit(limit)
                
            return query.all()
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to get questions by difficulty {difficulty}: {str(e)}",
                operation="query",
                table="questions"
            ) from e
    
    def get_questions_by_value_range(self, min_value: int, max_value: int, 
                                   limit: Optional[int] = None) -> List[Question]:
        """Get questions within a value range."""
        try:
            query = self.session.query(Question).filter(
                and_(Question.value >= min_value, Question.value <= max_value)
            )
            
            if limit:
                query = query.limit(limit)
                
            return query.all()
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to get questions by value range {min_value}-{max_value}: {str(e)}",
                operation="query",
                table="questions"
            ) from e
    
    def save_question(self, question: Question) -> Question:
        """Save a single question."""
        try:
            self.session.add(question)
            self.session.commit()
            self.session.refresh(question)
            return question
            
        except Exception as e:
            self.session.rollback()
            raise DatabaseError(
                f"Failed to save question: {str(e)}",
                operation="insert",
                table="questions"
            ) from e
    
    def save_questions_batch(self, questions: List[Question]) -> List[Question]:
        """Save multiple questions efficiently."""
        try:
            self.session.add_all(questions)
            self.session.commit()
            
            # Refresh all objects to get their IDs
            for question in questions:
                self.session.refresh(question)
                
            return questions
            
        except Exception as e:
            self.session.rollback()
            raise DatabaseError(
                f"Failed to save questions batch: {str(e)}",
                operation="bulk_insert",
                table="questions"
            ) from e
    
    def count_questions(self, category: Optional[str] = None, 
                       difficulty: Optional[str] = None) -> int:
        """Count questions with optional filters."""
        try:
            query = self.session.query(func.count(Question.id))
            
            if category:
                query = query.filter(Question.category == category)
            if difficulty:
                query = query.filter(Question.difficulty_level == difficulty)
                
            return query.scalar()
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to count questions: {str(e)}",
                operation="count",
                table="questions"
            ) from e
    
    def get_categories(self) -> List[str]:
        """Get all unique categories."""
        try:
            return [row[0] for row in self.session.query(Question.category.distinct()).all() if row[0]]
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to get categories: {str(e)}",
                operation="query",
                table="questions"
            ) from e
    
    def get_difficulty_levels(self) -> List[str]:
        """Get all unique difficulty levels."""
        try:
            return [row[0] for row in self.session.query(Question.difficulty_level.distinct()).all() if row[0]]
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to get difficulty levels: {str(e)}",
                operation="query",
                table="questions"
            ) from e
    
    def delete_question(self, question_id: str) -> bool:
        """Delete a question by ID."""
        try:
            question = self.get_question_by_id(question_id)
            if question:
                self.session.delete(question)
                self.session.commit()
                return True
            return False
            
        except Exception as e:
            self.session.rollback()
            raise DatabaseError(
                f"Failed to delete question {question_id}: {str(e)}",
                operation="delete",
                table="questions"
            ) from e
    
    def get_questions(self, filters: Optional[Dict[str, Any]] = None,
                     limit: Optional[int] = None, offset: int = 0) -> List[Question]:
        """Get questions with flexible filtering support.
        
        Args:
            filters: Dictionary of field filters. Supported keys:
                - category: Filter by category
                - difficulty_level: Filter by difficulty
                - value: Filter by exact value
                - min_value: Filter by minimum value
                - max_value: Filter by maximum value
                - round: Filter by round (Jeopardy, Double Jeopardy, Final Jeopardy)
                - benchmark_id: Filter by questions used in a specific benchmark (joins with benchmark_results)
            limit: Maximum number of questions to return
            offset: Number of questions to skip
            
        Returns:
            List of Question objects matching the filters
        """
        try:
            query = self.session.query(Question)
            
            if filters:
                # Handle benchmark_id filter by joining with benchmark_results
                if 'benchmark_id' in filters:
                    # Import here to avoid circular imports
                    from src.storage.models.benchmark_result import BenchmarkResult
                    from src.storage.models.benchmark_run import BenchmarkRun
                    
                    query = query.join(BenchmarkResult, Question.id == BenchmarkResult.question_id)\
                                .join(BenchmarkRun, BenchmarkResult.benchmark_run_id == BenchmarkRun.id)\
                                .filter(BenchmarkRun.id == filters['benchmark_id'])\
                                .distinct()
                
                # Handle standard question field filters
                if 'category' in filters:
                    query = query.filter(Question.category == filters['category'])
                    
                if 'difficulty_level' in filters:
                    query = query.filter(Question.difficulty_level == filters['difficulty_level'])
                    
                if 'value' in filters:
                    query = query.filter(Question.value == filters['value'])
                    
                if 'min_value' in filters:
                    query = query.filter(Question.value >= filters['min_value'])
                    
                if 'max_value' in filters:
                    query = query.filter(Question.value <= filters['max_value'])
                    
                if 'round' in filters:
                    query = query.filter(Question.round == filters['round'])
            
            # Apply ordering
            query = query.order_by(Question.id)
            
            # Apply pagination
            if offset:
                query = query.offset(offset)
            if limit:
                query = query.limit(limit)
                
            return query.all()
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to get questions with filters {filters}: {str(e)}",
                operation="query",
                table="questions"
            ) from e

    def clear_all_questions(self) -> int:
        """Delete all questions from the database.
        
        Returns:
            Number of questions deleted
        """
        try:
            # Get count before deletion
            count = self.session.query(func.count(Question.id)).scalar()
            
            if count > 0:
                # Delete all questions
                self.session.query(Question).delete()
                self.session.commit()
                
            return count
            
        except Exception as e:
            self.session.rollback()
            raise DatabaseError(
                f"Failed to clear all questions: {str(e)}",
                operation="delete_all",
                table="questions"
            ) from e
    
    def get_question_statistics(self, benchmark_id: Optional[int] = None) -> Dict[str, Any]:
        """Get comprehensive statistics about questions.
        
        Args:
            benchmark_id: Optional benchmark ID to filter statistics for specific benchmark
            
        Returns:
            Dictionary containing various statistics about the questions
        """
        try:
            # Start with base query
            if benchmark_id:
                # Import here to avoid circular imports
                from src.storage.models.benchmark_result import BenchmarkResult
                from src.storage.models.benchmark_run import BenchmarkRun
                
                base_query = self.session.query(Question)\
                    .join(BenchmarkResult, Question.id == BenchmarkResult.question_id)\
                    .join(BenchmarkRun, BenchmarkResult.benchmark_run_id == BenchmarkRun.id)\
                    .filter(BenchmarkRun.id == benchmark_id)\
                    .distinct()
            else:
                base_query = self.session.query(Question)
            
            # Total questions
            total_questions = base_query.count()
            
            if total_questions == 0:
                return {
                    'total_questions': 0,
                    'unique_categories': 0,
                    'value_range': None,
                    'category_distribution': {},
                    'difficulty_distribution': {}
                }
            
            # Get all questions for analysis
            questions = base_query.all()
            
            # Unique categories
            categories = set(q.category for q in questions if q.category)
            unique_categories = len(categories)
            
            # Value range statistics
            values = [q.value for q in questions if q.value is not None and q.value > 0]
            value_range = None
            if values:
                value_range = {
                    'min': min(values),
                    'max': max(values),
                    'average': sum(values) / len(values),
                    'count': len(values)
                }
            
            # Category distribution
            category_counts = {}
            for question in questions:
                if question.category:
                    category_counts[question.category] = category_counts.get(question.category, 0) + 1
            
            # Difficulty distribution
            difficulty_counts = {}
            for question in questions:
                if question.difficulty_level:
                    difficulty_counts[question.difficulty_level] = difficulty_counts.get(question.difficulty_level, 0) + 1
            
            return {
                'total_questions': total_questions,
                'unique_categories': unique_categories,
                'value_range': value_range,
                'category_distribution': category_counts,
                'difficulty_distribution': difficulty_counts
            }
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to get question statistics: {str(e)}",
                operation="statistics",
                table="questions"
            ) from e