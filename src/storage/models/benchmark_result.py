"""
Benchmark Result Model - Clean Implementation

SQLAlchemy ORM model for individual question results from benchmark runs.
"""

from sqlalchemy import String, Integer, Text, Boolean, DECIMAL, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.core.database import Base


class TimestampMixin:
    """Mixin for models that need created/updated timestamps."""
    
    created_at: Mapped[str] = mapped_column(String, nullable=False, default="CURRENT_TIMESTAMP")
    updated_at: Mapped[str] = mapped_column(String, nullable=False, default="CURRENT_TIMESTAMP")


class BenchmarkResult(Base, TimestampMixin):
    """Model for individual question results from benchmark runs."""

    __tablename__ = "benchmark_results"
    __table_args__ = {'extend_existing': True}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    benchmark_run_id: Mapped[int] = mapped_column(Integer, ForeignKey('benchmark_runs.id'), nullable=False)
    question_id: Mapped[str] = mapped_column(String(255), ForeignKey('questions.id'), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Response data
    response_text: Mapped[str] = mapped_column(Text, nullable=True)
    is_correct: Mapped[bool] = mapped_column(Boolean, nullable=True)
    confidence_score: Mapped[float] = mapped_column(DECIMAL(3, 2), nullable=True)  # 0.00 to 1.00
    
    # Jeopardy scoring
    jeopardy_score: Mapped[int] = mapped_column(Integer, default=0)  # Score for this question (+/- question value)

    # Performance metrics
    response_time_ms: Mapped[int] = mapped_column(Integer, nullable=True)
    tokens_generated: Mapped[int] = mapped_column(Integer, nullable=True)
    tokens_input: Mapped[int] = mapped_column(Integer, nullable=True)
    tokens_output: Mapped[int] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[float] = mapped_column(DECIMAL(10, 6), nullable=True)

    # Metadata
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str] = mapped_column(Text, nullable=True)
    result_metadata: Mapped[str] = mapped_column(Text, nullable=True)  # JSON string

    # Relationships - simple and clean
    benchmark_run: Mapped["BenchmarkRun"] = relationship(
        "src.storage.models.benchmark_run.BenchmarkRun",
        back_populates="results"
    )
    
    question: Mapped["Question"] = relationship(
        "src.storage.models.question.Question",
        back_populates="benchmark_results"
    )

    def __repr__(self) -> str:
        return f"<BenchmarkResult(id={self.id}, model='{self.model_name}', correct={self.is_correct})>"