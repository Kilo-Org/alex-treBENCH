"""
Benchmark Run Model - Clean Implementation

SQLAlchemy ORM model for benchmark runs.
"""

from typing import List
from datetime import datetime
from sqlalchemy import String, Integer, Text, DECIMAL, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.core.database import Base
from .mixins import TimestampMixin


class BenchmarkRun(Base, TimestampMixin):
    """Model for benchmark runs with essential metadata."""

    __tablename__ = "benchmark_runs"
    __table_args__ = {'extend_existing': True}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    benchmark_mode: Mapped[str] = mapped_column(String(50), default='standard')
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default='pending')
    models_tested: Mapped[str] = mapped_column(Text, nullable=True)  # JSON string
    total_questions: Mapped[int] = mapped_column(Integer, default=0)
    completed_questions: Mapped[int] = mapped_column(Integer, default=0)
    
    # Performance metadata
    total_cost_usd: Mapped[float] = mapped_column(DECIMAL(10, 6), default=0.0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    avg_response_time_ms: Mapped[float] = mapped_column(DECIMAL(10, 2), nullable=True)
    
    # Error tracking
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    error_details: Mapped[str] = mapped_column(Text, nullable=True)  # JSON string

    # Relationships - temporarily disabled due to configuration issues
    # results: Mapped[List["BenchmarkResult"]] = relationship(
    #     "src.storage.models.benchmark_result.BenchmarkResult",
    #     back_populates="benchmark_run",
    #     cascade="all, delete-orphan"
    # )
    
    # performances: Mapped[List["ModelPerformance"]] = relationship(
    #     "src.storage.models.model_performance.ModelPerformance",
    #     back_populates="benchmark_run",
    #     cascade="all, delete-orphan"
    # )

    def __repr__(self) -> str:
        return f"<BenchmarkRun(id={self.id}, name='{self.name}', status='{self.status}')>"
