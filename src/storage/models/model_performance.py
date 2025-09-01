"""
Model Performance Model

SQLAlchemy ORM model for aggregated performance metrics.
"""

import json
from typing import Dict, Any
from sqlalchemy import Column, Integer, String, Text, DECIMAL, ForeignKey, Index, CheckConstraint
from sqlalchemy.orm import relationship

from src.core.database import Base


class ModelPerformance(Base):
    """Model for aggregated performance metrics."""

    __tablename__ = "model_performance"
    __table_args__ = (
        Index('idx_performance_benchmark_run_id', 'benchmark_run_id'),
        Index('idx_performance_model_name', 'model_name'),
        Index('idx_performance_accuracy', 'accuracy_rate'),
        Index('idx_performance_response_time', 'avg_response_time_ms'),
        Index('idx_performance_cost', 'total_cost_usd'),
        Index('idx_performance_jeopardy_score', 'jeopardy_score'),
        CheckConstraint('total_questions > 0', name='check_total_questions_positive'),
        CheckConstraint('correct_answers >= 0', name='check_correct_answers_non_negative'),
        CheckConstraint('accuracy_rate >= 0 AND accuracy_rate <= 1', name='check_accuracy_range'),
        {'extend_existing': True}
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    benchmark_run_id = Column(Integer, ForeignKey('benchmark_runs.id'), nullable=False)
    model_name = Column(String(255), nullable=False)

    # Basic metrics
    total_questions = Column(Integer, nullable=False)
    correct_answers = Column(Integer, nullable=False)
    accuracy_rate = Column(DECIMAL(5, 4))
    
    # Jeopardy scoring
    jeopardy_score = Column(Integer, default=0)  # Total Jeopardy score (can be negative)
    category_jeopardy_scores = Column(Text)  # JSON object with per-category Jeopardy scores

    # Performance metrics
    avg_response_time_ms = Column(DECIMAL(10, 2))
    median_response_time_ms = Column(DECIMAL(10, 2))
    min_response_time_ms = Column(DECIMAL(10, 2))
    max_response_time_ms = Column(DECIMAL(10, 2))
    response_time_std = Column(DECIMAL(10, 2))

    # Cost metrics
    total_cost_usd = Column(DECIMAL(10, 6))
    avg_cost_per_question = Column(DECIMAL(10, 6))
    cost_per_correct_answer = Column(DECIMAL(10, 6))

    # Token metrics
    total_tokens_input = Column(Integer)
    total_tokens_output = Column(Integer)
    total_tokens = Column(Integer)
    avg_tokens_per_question = Column(DECIMAL(10, 2))
    tokens_per_second = Column(DECIMAL(10, 2))

    # Category performance
    category_performance = Column(Text)  # JSON object with per-category stats

    # Difficulty performance
    difficulty_performance = Column(Text)  # JSON object with per-difficulty stats

    # Confidence analysis
    avg_confidence = Column(DECIMAL(3, 2))
    confidence_accuracy_correlation = Column(DECIMAL(5, 4))

    # Error analysis
    error_count = Column(Integer, default=0)
    error_rate = Column(DECIMAL(5, 4))

    # Relationships
    benchmark_run = relationship("BenchmarkRun", back_populates="performances")

    def __repr__(self) -> str:
        return f"<ModelPerformance(id={self.id}, model='{self.model_name}', accuracy={self.accuracy_rate}, jeopardy_score={self.jeopardy_score})>"

    @property
    def category_performance_dict(self) -> Dict[str, Any]:
        """Get category performance as a Python dictionary."""
        if self.category_performance:
            try:
                return json.loads(self.category_performance)
            except json.JSONDecodeError:
                return {}
        return {}

    @category_performance_dict.setter
    def category_performance_dict(self, value: Dict[str, Any]) -> None:
        """Set category performance from a Python dictionary."""
        self.category_performance = json.dumps(value) if value else None

    @property
    def difficulty_performance_dict(self) -> Dict[str, Any]:
        """Get difficulty performance as a Python dictionary."""
        if self.difficulty_performance:
            try:
                return json.loads(self.difficulty_performance)
            except json.JSONDecodeError:
                return {}
        return {}

    @difficulty_performance_dict.setter
    def difficulty_performance_dict(self, value: Dict[str, Any]) -> None:
        """Set difficulty performance from a Python dictionary."""
        self.difficulty_performance = json.dumps(value) if value else None

    @property
    def category_jeopardy_scores_dict(self) -> Dict[str, int]:
        """Get category Jeopardy scores as a Python dictionary."""
        if self.category_jeopardy_scores:
            try:
                return json.loads(self.category_jeopardy_scores)
            except json.JSONDecodeError:
                return {}
        return {}

    @category_jeopardy_scores_dict.setter
    def category_jeopardy_scores_dict(self, value: Dict[str, int]) -> None:
        """Set category Jeopardy scores from a Python dictionary."""
        self.category_jeopardy_scores = json.dumps(value) if value else None

    def calculate_efficiency_score(self) -> float:
        """Calculate efficiency score (accuracy / cost)."""
        if self.total_cost_usd and self.total_cost_usd > 0:
            return float(self.accuracy_rate) / float(self.total_cost_usd)
        return 0.0

    def calculate_speed_score(self) -> float:
        """Calculate speed score (1 / avg_response_time)."""
        if self.avg_response_time_ms and self.avg_response_time_ms > 0:
            return 1000 / float(self.avg_response_time_ms)  # responses per second
        return 0.0