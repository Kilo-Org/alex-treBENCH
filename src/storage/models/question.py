"""
Question Model - Clean Implementation

SQLAlchemy ORM model for Jeopardy questions.
"""

from typing import List
from sqlalchemy import String, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.core.database import Base
from .mixins import StringTimestampMixin


class Question(Base, StringTimestampMixin):
    """Model for cached questions from the Jeopardy dataset."""

    __tablename__ = "questions"
    __table_args__ = {'extend_existing': True}

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    correct_answer: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(255), nullable=True)
    value: Mapped[int] = mapped_column(Integer, nullable=True)
    air_date: Mapped[str] = mapped_column(String, nullable=True)
    show_number: Mapped[int] = mapped_column(Integer, nullable=True)
    round: Mapped[str] = mapped_column(String(50), nullable=True)
    difficulty_level: Mapped[str] = mapped_column(String(50), nullable=True)

    # Relationships - temporarily disabled due to configuration issues
    # benchmark_results: Mapped[List["BenchmarkResult"]] = relationship(
    #     "src.storage.models.benchmark_result.BenchmarkResult",
    #     back_populates="question",
    #     cascade="all, delete-orphan"
    # )

    def __repr__(self) -> str:
        return f"<Question(id={self.id}, category='{self.category}', value={self.value})>"