"""
Clean Database System - Completely Isolated

Fresh SQLAlchemy setup with its own Base class and registry to avoid conflicts.
"""

from typing import Optional, Generator, List
from contextlib import contextmanager
from sqlalchemy import create_engine, Engine, String, Integer, Text, Boolean, DECIMAL, ForeignKey
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.pool import StaticPool

from src.core.config import get_config
from src.core.exceptions import DatabaseError


# Fresh Base class with no registry pollution
class CleanBase(DeclarativeBase):
    """Clean base class for all ORM models - isolated registry."""
    pass


class TimestampMixin:
    """Mixin for models that need created/updated timestamps."""
    
    created_at: Mapped[str] = mapped_column(String, nullable=False, default="CURRENT_TIMESTAMP")
    updated_at: Mapped[str] = mapped_column(String, nullable=False, default="CURRENT_TIMESTAMP")


# Define all models with the clean Base
class Question(CleanBase, TimestampMixin):
    """Model for cached questions from the Jeopardy dataset."""

    __tablename__ = "questions"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    correct_answer: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(255), nullable=True)
    value: Mapped[int] = mapped_column(Integer, nullable=True)
    air_date: Mapped[str] = mapped_column(String, nullable=True)
    show_number: Mapped[int] = mapped_column(Integer, nullable=True)
    round: Mapped[str] = mapped_column(String(50), nullable=True)
    difficulty_level: Mapped[str] = mapped_column(String(50), nullable=True)

    # Relationships - clean and isolated
    benchmark_results: Mapped[List["BenchmarkResult"]] = relationship(
        "BenchmarkResult", 
        back_populates="question", 
        cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Question(id={self.id}, category='{self.category}', value={self.value})>"


class BenchmarkRun(CleanBase, TimestampMixin):
    """Model for benchmark runs with essential metadata."""

    __tablename__ = "benchmark_runs"

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

    # Relationships - clean and isolated
    results: Mapped[List["BenchmarkResult"]] = relationship(
        "BenchmarkResult", 
        back_populates="benchmark_run", 
        cascade="all, delete-orphan"
    )
    
    performances: Mapped[List["ModelPerformance"]] = relationship(
        "ModelPerformance", 
        back_populates="benchmark_run", 
        cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<BenchmarkRun(id={self.id}, name='{self.name}', status='{self.status}')>"


class BenchmarkResult(CleanBase, TimestampMixin):
    """Model for individual question results from benchmark runs."""

    __tablename__ = "benchmark_results"

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

    # Relationships - clean and isolated
    benchmark_run: Mapped["BenchmarkRun"] = relationship(
        "BenchmarkRun", 
        back_populates="results"
    )
    
    question: Mapped["Question"] = relationship(
        "Question", 
        back_populates="benchmark_results"
    )

    def __repr__(self) -> str:
        return f"<BenchmarkResult(id={self.id}, model='{self.model_name}', correct={self.is_correct})>"


class ModelPerformance(CleanBase, TimestampMixin):
    """Model for aggregated performance metrics."""

    __tablename__ = "model_performance"

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

    # Relationships - clean and isolated
    benchmark_run: Mapped["BenchmarkRun"] = relationship(
        "BenchmarkRun", 
        back_populates="performances"
    )

    def __repr__(self) -> str:
        return f"<ModelPerformance(id={self.id}, model='{self.model_name}', accuracy={self.accuracy_rate})>"


# Clean database management
_clean_engine: Optional[Engine] = None
CleanSessionFactory: Optional[sessionmaker] = None


def get_clean_engine() -> Engine:
    """Get or create the clean SQLAlchemy engine."""
    global _clean_engine
    
    if _clean_engine is None:
        config = get_config()
        
        try:
            # Configure engine based on database URL
            if config.database.url.startswith('sqlite:'):
                # SQLite-specific configuration
                _clean_engine = create_engine(
                    config.database.url,
                    echo=config.database.echo,
                    poolclass=StaticPool,
                    connect_args={
                        'check_same_thread': False,
                        'timeout': 20,
                    }
                )
            else:
                # Generic database configuration
                _clean_engine = create_engine(
                    config.database.url,
                    echo=config.database.echo,
                    pool_size=config.database.pool_size,
                )
        except Exception as e:
            raise DatabaseError(
                f"Failed to create clean database engine: {str(e)}",
                operation="create_engine",
                url=config.database.url
            ) from e
    
    return _clean_engine


def get_clean_session_factory() -> sessionmaker:
    """Get or create the clean SQLAlchemy session factory."""
    global CleanSessionFactory
    
    if CleanSessionFactory is None:
        engine = get_clean_engine()
        CleanSessionFactory = sessionmaker(bind=engine)
    
    return CleanSessionFactory


def create_clean_tables() -> None:
    """Create all clean database tables."""
    try:
        engine = get_clean_engine()
        CleanBase.metadata.create_all(engine)
    except Exception as e:
        raise DatabaseError(
            f"Failed to create clean database tables: {str(e)}",
            operation="create_tables"
        ) from e


@contextmanager
def get_clean_db_session() -> Generator[Session, None, None]:
    """Context manager for clean database sessions with automatic cleanup."""
    SessionFactory = get_clean_session_factory()
    session = SessionFactory()
    
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise DatabaseError(
            f"Clean database session error: {str(e)}",
            operation="session_transaction"
        ) from e
    finally:
        session.close()


def init_clean_database() -> None:
    """Initialize the clean database with tables."""
    try:
        create_clean_tables()
    except Exception as e:
        raise DatabaseError(
            f"Failed to initialize clean database: {str(e)}",
            operation="init_database"
        ) from e


# Expose the clean models
__all__ = [
    'Question', 'BenchmarkRun', 'BenchmarkResult', 'ModelPerformance',
    'get_clean_db_session', 'init_clean_database'
]