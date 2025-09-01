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