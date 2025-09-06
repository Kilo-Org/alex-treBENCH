"""
Performance Repository

Repository class for managing model performance data access.
"""

from typing import List, Optional
from sqlalchemy.orm import Session

from src.core.exceptions import DatabaseError
from src.storage.models import ModelPerformance


class PerformanceRepository:
    """Repository for managing performance data access."""
    
    def __init__(self, session: Optional[Session] = None):
        """Initialize repository with optional session."""
        self.session = session
    
    def save_performance_summary(self, summary: ModelPerformance) -> ModelPerformance:
        """Save a performance summary."""
        try:
            self.session.add(summary)
            self.session.commit()
            self.session.refresh(summary)
            return summary
        except Exception as e:
            self.session.rollback()
            raise DatabaseError(f"Failed to save performance summary: {str(e)}")

    def get_performances_by_benchmark(self, benchmark_id: int) -> List[ModelPerformance]:
        """Get all performance summaries for a specific benchmark."""
        try:
            performances = (
                self.session.query(ModelPerformance)
                .filter(ModelPerformance.benchmark_run_id == benchmark_id)
                .all()
            )
            return performances
        except Exception as e:
            raise DatabaseError(f"Failed to get performances for benchmark {benchmark_id}: {str(e)}")