"""
Database Connection and Session Management

SQLAlchemy database connection, session management, and schema creation
for the benchmarking system with SQLite support.
"""

from typing import Optional, Generator
from contextlib import contextmanager
from sqlalchemy import create_engine, Engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .config import get_config
from .exceptions import DatabaseError


# SQLAlchemy declarative base for ORM models
Base = declarative_base()

# Global engine and session factory
_engine: Optional[Engine] = None
SessionFactory: Optional[sessionmaker] = None


def get_engine() -> Engine:
    """Get or create the SQLAlchemy engine."""
    global _engine
    
    if _engine is None:
        config = get_config()
        
        try:
            # Configure engine based on database URL
            if config.database.url.startswith('sqlite:'):
                # SQLite-specific configuration
                _engine = create_engine(
                    config.database.url,
                    echo=config.database.echo,
                    poolclass=StaticPool,
                    connect_args={
                        'check_same_thread': False,  # Allow SQLite to be used across threads
                        'timeout': 20,
                    }
                )
            else:
                # Generic database configuration
                _engine = create_engine(
                    config.database.url,
                    echo=config.database.echo,
                    pool_size=config.database.pool_size,
                )
        except Exception as e:
            raise DatabaseError(
                f"Failed to create database engine: {str(e)}",
                operation="create_engine",
                url=config.database.url
            ) from e
    
    return _engine


def get_session_factory() -> sessionmaker:
    """Get or create the SQLAlchemy session factory."""
    global SessionFactory
    
    if SessionFactory is None:
        engine = get_engine()
        SessionFactory = sessionmaker(bind=engine)
    
    return SessionFactory


def create_tables() -> None:
    """Create all database tables."""
    try:
        engine = get_engine()
        
        # Models are automatically registered with Base when imported elsewhere
        # No need to import them here as it causes duplicate table definitions
        Base.metadata.create_all(engine)
    except Exception as e:
        raise DatabaseError(
            f"Failed to create database tables: {str(e)}",
            operation="create_tables"
        ) from e


def drop_tables() -> None:
    """Drop all database tables (useful for testing)."""
    try:
        engine = get_engine()
        Base.metadata.drop_all(engine)
    except Exception as e:
        raise DatabaseError(
            f"Failed to drop database tables: {str(e)}",
            operation="drop_tables"
        ) from e


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for database sessions with automatic cleanup."""
    SessionFactory = get_session_factory()
    session = SessionFactory()
    
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise DatabaseError(
            f"Database session error: {str(e)}",
            operation="session_transaction"
        ) from e
    finally:
        session.close()


def init_database() -> None:
    """Initialize the database with tables and basic setup."""
    try:
        # Models are already imported elsewhere and registered with Base
        # No need to import them here as it causes duplicate table definitions
        
        # Create tables
        create_tables()
        
        # TODO: Add any initial data setup if needed
        # Example: default categories, sample data, etc.
        
    except Exception as e:
        raise DatabaseError(
            f"Failed to initialize database: {str(e)}",
            operation="init_database"
        ) from e


def reset_database() -> None:
    """Reset the database by dropping and recreating all tables."""
    try:
        drop_tables()
        create_tables()
    except Exception as e:
        raise DatabaseError(
            f"Failed to reset database: {str(e)}",
            operation="reset_database"
        ) from e


def check_database_connection() -> bool:
    """Check if database connection is working."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            # Simple query to test connection
            result = conn.execute(text("SELECT 1"))
            row = result.fetchone()
            return row is not None and row[0] == 1
    except Exception as e:
        raise DatabaseError(
            f"Database connection check failed: {str(e)}",
            operation="connection_check"
        ) from e


def close_connections() -> None:
    """Close all database connections (useful for testing and cleanup)."""
    global _engine, SessionFactory
    
    if _engine:
        _engine.dispose()
        _engine = None
    
    SessionFactory = None