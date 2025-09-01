"""
Benchmark Run Model

SQLAlchemy ORM model for benchmark runs with comprehensive metadata.
"""

import json
from typing import Dict, Any
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, DECIMAL, Index, CheckConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from src.core.database import Base


class BenchmarkRun(Base):
    """Model for benchmark runs with comprehensive metadata."""

    __tablename__ = "benchmark_runs"
    __table_args__ = (
        Index('idx_benchmark_run_status', 'status'),
        Index('idx_benchmark_run_created_at', 'created_at'),
        Index('idx_benchmark_run_mode', 'benchmark_mode'),
        Index('idx_benchmark_run_environment', 'environment'),
        CheckConstraint("status IN ('pending', 'running', 'completed', 'failed')", name='check_status_valid'),
        CheckConstraint('sample_size > 0', name='check_sample_size_positive'),
        {'extend_existing': True}
    )

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
