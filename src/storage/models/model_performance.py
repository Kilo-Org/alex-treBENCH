"""
Model Performance Model - Clean Implementation

SQLAlchemy ORM model for aggregated performance metrics.
"""

from sqlalchemy import String, Integer, Text, DECIMAL, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.core.database import Base
from .mixins import StringTimestampMixin


class ModelPerformance(Base, StringTimestampMixin):
    """Model for aggregated performance metrics."""

    __tablename__ = "model_performance"
    __table_args__ = {'extend_existing': True}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    benchmark_run_id: Mapped[int] = mapped_column(Integer, ForeignKey('benchmark_runs.id'), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Basic metrics
    total_questions: Mapped[int] = mapped_column(Integer, nullable=False)
    correct_answers: Mapped[int] = mapped_column(Integer, nullable=False)
    accuracy_rate: Mapped[float] = mapped_column(DECIMAL(5, 4), nullable=True)
    
    # Jeopardy scoring
    jeopardy_score: Mapped[int] = mapped_column(Integer, default=0)  # Total Jeopardy score
    category_jeopardy_scores: Mapped[str] = mapped_column(Text, nullable=True)  # JSON

    # Performance metrics
    avg_response_time_ms: Mapped[float] = mapped_column(DECIMAL(10, 2), nullable=True)
    median_response_time_ms: Mapped[float] = mapped_column(DECIMAL(10, 2), nullable=True)
    min_response_time_ms: Mapped[float] = mapped_column(DECIMAL(10, 2), nullable=True)
    max_response_time_ms: Mapped[float] = mapped_column(DECIMAL(10, 2), nullable=True)

    # Cost metrics
    total_cost_usd: Mapped[float] = mapped_column(DECIMAL(10, 6), nullable=True)
    avg_cost_per_question: Mapped[float] = mapped_column(DECIMAL(10, 6), nullable=True)
    cost_per_correct_answer: Mapped[float] = mapped_column(DECIMAL(10, 6), nullable=True)

    # Token metrics
    total_tokens_input: Mapped[int] = mapped_column(Integer, nullable=True)
    total_tokens_output: Mapped[int] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=True)
    avg_tokens_per_question: Mapped[float] = mapped_column(DECIMAL(10, 2), nullable=True)

    # Category and difficulty performance (JSON strings)
    category_performance: Mapped[str] = mapped_column(Text, nullable=True)
    difficulty_performance: Mapped[str] = mapped_column(Text, nullable=True)

    # Confidence analysis
    avg_confidence: Mapped[float] = mapped_column(DECIMAL(3, 2), nullable=True)
    confidence_accuracy_correlation: Mapped[float] = mapped_column(DECIMAL(5, 4), nullable=True)

    # Error analysis
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    error_rate: Mapped[float] = mapped_column(DECIMAL(5, 4), nullable=True)

    # Relationships - temporarily disabled due to configuration issues
    # benchmark_run: Mapped["BenchmarkRun"] = relationship(
    #     "src.storage.models.benchmark_run.BenchmarkRun",
    #     back_populates="performances"
    # )

    def __repr__(self) -> str:
        return f"<ModelPerformance(id={self.id}, model='{self.model_name}', accuracy={self.accuracy_rate})>"