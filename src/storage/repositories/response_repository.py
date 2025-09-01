"""
Response Repository

Repository class for managing benchmark response data access.
"""

from typing import List, Optional
from sqlalchemy.orm import Session

from src.core.exceptions import DatabaseError
from src.storage.models import BenchmarkResult


class ResponseRepository:
    """Repository for managing response data access."""
    
    def __init__(self, session: Optional[Session] = None):
        """Initialize repository with optional session."""
        self.session = session
    
    def save_response(self, response: BenchmarkResult) -> BenchmarkResult:
        """Save a response."""
        try:
            self.session.add(response)
            self.session.commit() 
            self.session.refresh(response)
            return response
        except Exception as e:
            self.session.rollback()
            raise DatabaseError(f"Failed to save response: {str(e)}")