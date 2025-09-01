"""
Benchmark Result Model

SQLAlchemy ORM model for individual question results from benchmark runs.
"""

import json
from typing import Dict, Any
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, DECIMAL, ForeignKey, Index, CheckConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from src.core.database import Base


class BenchmarkResult(Base):
    """Model for individual question results from benchmark runs."""

    __tablename__ = "benchmark_results"
    __table_args__ = (
        Index('idx_result_benchmark_run_id', 'benchmark_run_id'),
        Index('idx_result_question_id', 'question_id'),
        Index('idx_result_model_name', 'model_name'),
        Index('idx_result_is_correct', 'is_correct'),
        Index('idx_result_created_at', 'created_at'),
        Index('idx_result_model_benchmark', 'model_name', 'benchmark_run_id'),
        Index('idx_result_jeopardy_score', 'jeopardy_score'),
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 1', name='check_confidence_range'),
        CheckConstraint('response_time_ms >= 0', name='check_response_time_positive'),
        {'extend_existing': True}
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    benchmark_run_id = Column(Integer, ForeignKey('benchmark_runs.id'), nullable=False)
    question_id = Column(String(255), ForeignKey('questions.id'), nullable=False)
    model_name = Column(String(255), nullable=False)

    # Response data
    response_text = Column(Text)
    is_correct = Column(Boolean)
    confidence_score = Column(DECIMAL(3, 2))  # 0.00 to 1.00
    
    # Jeopardy scoring
    jeopardy_score = Column(Integer, default=0)  # Score for this question (+/- question value)

    # Performance metrics
    response_time_ms = Column(Integer)
    tokens_generated = Column(Integer)
    tokens_input = Column(Integer)
    tokens_output = Column(Integer)
    cost_usd = Column(DECIMAL(10, 6))

    # Metadata
    created_at = Column(DateTime, default=func.current_timestamp())
    retry_count = Column(Integer, default=0)
    error_message = Column(Text)
    result_metadata = Column(Text)  # JSON string for additional data

    # Relationships
    benchmark_run = relationship("BenchmarkRun", back_populates="results")
    question = relationship("Question", back_populates="benchmark_results")

    def __repr__(self) -> str:
        return f"<BenchmarkResult(id={self.id}, model='{self.model_name}', correct={self.is_correct}, jeopardy_score={self.jeopardy_score})>"

    @property
    def metadata_dict(self) -> Dict[str, Any]:
        """Get metadata as a Python dictionary."""
        if self.result_metadata:
            try:
                return json.loads(self.result_metadata)
            except json.JSONDecodeError:
                return {}
        return {}

    @metadata_dict.setter
    def metadata_dict(self, value: Dict[str, Any]) -> None:
        """Set metadata from a Python dictionary."""
        self.result_metadata = json.dumps(value) if value else None