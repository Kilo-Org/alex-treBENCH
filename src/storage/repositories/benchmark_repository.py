"""
Benchmark Repository

Repository class for managing benchmark run data access.
"""

from typing import List, Optional
from sqlalchemy.orm import Session

from src.core.exceptions import DatabaseError
from src.storage.models import BenchmarkRun


class BenchmarkRepository:
    """Repository for managing benchmark run data access."""
    
    def __init__(self, session: Optional[Session] = None):
        """Initialize repository with optional session."""
        self.session = session
    
    def get_benchmark_by_id(self, benchmark_id: int) -> Optional[BenchmarkRun]:
        """Get benchmark by ID."""
        try:
            return self.session.query(BenchmarkRun).filter(BenchmarkRun.id == benchmark_id).first()
        except Exception as e:
            raise DatabaseError(f"Failed to get benchmark {benchmark_id}: {str(e)}")
    
    def save_benchmark_run(self, benchmark: BenchmarkRun) -> BenchmarkRun:
        """Save a benchmark run."""
        try:
            self.session.add(benchmark)
            self.session.commit()
            self.session.refresh(benchmark)
            return benchmark
        except Exception as e:
            self.session.rollback()
            raise DatabaseError(f"Failed to save benchmark: {str(e)}")
    
    def list_benchmarks(self, limit: Optional[int] = None) -> List[BenchmarkRun]:
        """List benchmark runs ordered by most recent first."""
        try:
            query = self.session.query(BenchmarkRun).order_by(BenchmarkRun.created_at.desc())
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
        except Exception as e:
            raise DatabaseError(f"Failed to list benchmarks: {str(e)}")