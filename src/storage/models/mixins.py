"""
Database Model Mixins

Shared mixin classes for SQLAlchemy models.
"""

from datetime import datetime
from sqlalchemy import String, DateTime
from sqlalchemy.orm import Mapped, mapped_column


class TimestampMixin:
    """Mixin for models that need created/updated timestamps."""
    
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class StringTimestampMixin:
    """Mixin for models that use string-based timestamps (for backward compatibility)."""
    
    created_at: Mapped[str] = mapped_column(String, nullable=False, default="CURRENT_TIMESTAMP")
    updated_at: Mapped[str] = mapped_column(String, nullable=False, default="CURRENT_TIMESTAMP")