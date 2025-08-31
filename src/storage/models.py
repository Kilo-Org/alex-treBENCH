"""
SQLAlchemy ORM Models

Database models for the Jeopardy benchmarking system based on the
technical specification schema with relationships and constraints.
"""

from datetime import datetime
from typing import Optional, Dict, Any
import json
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean,
    DECIMAL, ForeignKey, JSON, Index, CheckConstraint
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func

from core.database import Base


class Question(Base):
    """Model for cached questions from the Jeopardy dataset."""

    __tablename__ = "questions"

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

    # Indexes
    __table_args__ = (
        Index('idx_question_category', 'category'),
        Index('idx_question_value', 'value'),
        Index('idx_question_difficulty', 'difficulty_level'),
        Index('idx_question_air_date', 'air_date'),
        Index('idx_question_round', 'round'),
        CheckConstraint('value > 0', name='check_value_positive'),
    )

    def __repr__(self) -> str:
        return f"<Question(id={self.id}, category='{self.category}', value={self.value})>"


class BenchmarkRun(Base):
    """Model for benchmark runs with comprehensive metadata."""

    __tablename__ = "benchmark_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    benchmark_mode = Column(String(50), default='standard')  # quick, standard, comprehensive
    sample_size = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=func.current_timestamp())
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    status = Column(String(50), default='pending')  # pending, running, completed, failed
    models_tested = Column(Text)  # JSON array of model names
    total_questions = Column(Integer, default=0)
    completed_questions = Column(Integer, default=0)

    # Configuration metadata
    config_snapshot = Column(Text)  # JSON snapshot of configuration used
    environment = Column(String(50), default='production')

    # Performance metadata
    total_cost_usd = Column(DECIMAL(10, 6), default=0)
    total_tokens = Column(Integer, default=0)
    avg_response_time_ms = Column(DECIMAL(10, 2))

    # Error tracking
    error_count = Column(Integer, default=0)
    error_details = Column(Text)  # JSON array of errors

    # Relationships
    results = relationship("BenchmarkResult", back_populates="benchmark_run", cascade="all, delete-orphan")
    performances = relationship("ModelPerformance", back_populates="benchmark_run", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index('idx_benchmark_run_status', 'status'),
        Index('idx_benchmark_run_created_at', 'created_at'),
        Index('idx_benchmark_run_mode', 'benchmark_mode'),
        Index('idx_benchmark_run_environment', 'environment'),
        CheckConstraint("status IN ('pending', 'running', 'completed', 'failed')", name='check_status_valid'),
        CheckConstraint('sample_size > 0', name='check_sample_size_positive'),
    )

    def __repr__(self) -> str:
        return f"<BenchmarkRun(id={self.id}, name='{self.name}', status='{self.status}', mode='{self.benchmark_mode}')>"

    @property
    def models_tested_list(self) -> list:
        """Get models_tested as a Python list."""
        if self.models_tested:
            try:
                return json.loads(self.models_tested)
            except json.JSONDecodeError:
                return []
        return []

    @models_tested_list.setter
    def models_tested_list(self, value: list) -> None:
        """Set models_tested from a Python list."""
        self.models_tested = json.dumps(value) if value else None

    @property
    def config_snapshot_dict(self) -> Dict[str, Any]:
        """Get config snapshot as a Python dictionary."""
        if self.config_snapshot:
            try:
                return json.loads(self.config_snapshot)
            except json.JSONDecodeError:
                return {}
        return {}

    @config_snapshot_dict.setter
    def config_snapshot_dict(self, value: Dict[str, Any]) -> None:
        """Set config snapshot from a Python dictionary."""
        self.config_snapshot = json.dumps(value) if value else None

    @property
    def error_details_list(self) -> list:
        """Get error details as a Python list."""
        if self.error_details:
            try:
                return json.loads(self.error_details)
            except json.JSONDecodeError:
                return []
        return []

    @error_details_list.setter
    def error_details_list(self, value: list) -> None:
        """Set error details from a Python list."""
        self.error_details = json.dumps(value) if value else None

    def is_completed(self) -> bool:
        """Check if benchmark run is completed."""
        return self.status == 'completed' and self.completed_at is not None

    def progress_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_questions and self.total_questions > 0:
            return (self.completed_questions / self.total_questions) * 100
        return 0.0


class BenchmarkResult(Base):
    """Model for individual question results from benchmark runs."""

    __tablename__ = "benchmark_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    benchmark_run_id = Column(Integer, ForeignKey('benchmark_runs.id'), nullable=False)
    question_id = Column(String(255), ForeignKey('questions.id'), nullable=False)
    model_name = Column(String(255), nullable=False)

    # Response data
    response_text = Column(Text)
    is_correct = Column(Boolean)
    confidence_score = Column(DECIMAL(3, 2))  # 0.00 to 1.00

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

    # Indexes
    __table_args__ = (
        Index('idx_result_benchmark_run_id', 'benchmark_run_id'),
        Index('idx_result_question_id', 'question_id'),
        Index('idx_result_model_name', 'model_name'),
        Index('idx_result_is_correct', 'is_correct'),
        Index('idx_result_created_at', 'created_at'),
        Index('idx_result_model_benchmark', 'model_name', 'benchmark_run_id'),
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 1', name='check_confidence_range'),
        CheckConstraint('response_time_ms >= 0', name='check_response_time_positive'),
    )

    def __repr__(self) -> str:
        return f"<BenchmarkResult(id={self.id}, model='{self.model_name}', correct={self.is_correct})>"

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


class ModelPerformance(Base):
    """Model for aggregated performance metrics."""

    __tablename__ = "model_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    benchmark_run_id = Column(Integer, ForeignKey('benchmark_runs.id'), nullable=False)
    model_name = Column(String(255), nullable=False)

    # Basic metrics
    total_questions = Column(Integer, nullable=False)
    correct_answers = Column(Integer, nullable=False)
    accuracy_rate = Column(DECIMAL(5, 4))

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

    # Indexes
    __table_args__ = (
        Index('idx_performance_benchmark_run_id', 'benchmark_run_id'),
        Index('idx_performance_model_name', 'model_name'),
        Index('idx_performance_accuracy', 'accuracy_rate'),
        Index('idx_performance_response_time', 'avg_response_time_ms'),
        Index('idx_performance_cost', 'total_cost_usd'),
        CheckConstraint('total_questions > 0', name='check_total_questions_positive'),
        CheckConstraint('correct_answers >= 0', name='check_correct_answers_non_negative'),
        CheckConstraint('accuracy_rate >= 0 AND accuracy_rate <= 1', name='check_accuracy_range'),
    )

    def __repr__(self) -> str:
        return f"<ModelPerformance(id={self.id}, model='{self.model_name}', accuracy={self.accuracy_rate})>"

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


# Backward compatibility aliases
Benchmark = BenchmarkRun
BenchmarkQuestion = Question
ModelResponse = BenchmarkResult
ModelPerformanceSummary = ModelPerformance


# Utility functions for working with models

def create_question(question_id: str, question_text: str, correct_answer: str,
                   category: str = None, value: int = None, air_date: datetime = None,
                   show_number: int = None, round_name: str = None) -> Question:
    """Create a new question instance."""
    return Question(
        id=question_id,
        question_text=question_text,
        correct_answer=correct_answer,
        category=category,
        value=value,
        air_date=air_date,
        show_number=show_number,
        round=round_name
    )


def create_benchmark_run(name: str, description: str = None,
                        benchmark_mode: str = 'standard', sample_size: int = 1000,
                        models_tested: list = None) -> BenchmarkRun:
    """Create a new benchmark run instance."""
    run = BenchmarkRun(
        name=name,
        description=description,
        benchmark_mode=benchmark_mode,
        sample_size=sample_size,
        status='pending'
    )

    if models_tested:
        run.models_tested_list = models_tested

    return run


def create_benchmark_result(benchmark_run_id: int, question_id: str, model_name: str,
                           response_text: str, is_correct: bool, confidence_score: float,
                           response_time_ms: int, tokens_generated: int = None,
                           cost_usd: float = None, metadata: Dict[str, Any] = None) -> BenchmarkResult:
    """Create a new benchmark result instance."""
    result = BenchmarkResult(
        benchmark_run_id=benchmark_run_id,
        question_id=question_id,
        model_name=model_name,
        response_text=response_text,
        is_correct=is_correct,
        confidence_score=confidence_score,
        response_time_ms=response_time_ms,
        tokens_generated=tokens_generated,
        cost_usd=cost_usd
    )

    if metadata:
        result.metadata_dict = metadata

    return result


def create_model_performance(benchmark_run_id: int, model_name: str,
                           total_questions: int, correct_answers: int) -> ModelPerformance:
    """Create a new model performance instance."""
    accuracy_rate = correct_answers / total_questions if total_questions > 0 else 0

    return ModelPerformance(
        benchmark_run_id=benchmark_run_id,
        model_name=model_name,
        total_questions=total_questions,
        correct_answers=correct_answers,
        accuracy_rate=accuracy_rate
    )