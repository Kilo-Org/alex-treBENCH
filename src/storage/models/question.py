"""
Question Model

SQLAlchemy ORM model for Jeopardy questions.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Index, CheckConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from src.core.database import Base


class Question(Base):
    """Model for cached questions from the Jeopardy dataset."""

    __tablename__ = "questions"
    __table_args__ = (
        Index('idx_question_category', 'category'),
        Index('idx_question_value', 'value'),
        Index('idx_question_difficulty', 'difficulty_level'),
        Index('idx_question_air_date', 'air_date'),
        Index('idx_question_round', 'round'),
        CheckConstraint('value > 0', name='check_value_positive'),
        {'extend_existing': True}
    )

    id = Column(String(255), primary_key=True)  # From original dataset
    question_text = Column(Text, nullable=False)
    correct_answer = Column(Text, nullable=False)
    category = Column(String(255))
    value = Column(Integer)  # Dollar value
    air_date = Column(DateTime)
    show_number = Column(Integer)
    round = Column(String(50))  # Jeopardy, Double Jeopardy, Final Jeopardy
    difficulty_level = Column(String(50))  # Easy/Medium/Hard based on value
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())

    # Relationships
    benchmark_results = relationship("BenchmarkResult", back_populates="question", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Question(id={self.id}, category='{self.category}', value={self.value})>"