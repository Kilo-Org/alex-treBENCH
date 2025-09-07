"""
Database Connection and Session Management

SQLAlchemy database connection, session management, and schema creation
for the benchmarking system with SQLite support.
"""

import os
from typing import Optional, Generator
from contextlib import contextmanager
from urllib.parse import urlparse, parse_qs
from sqlalchemy import create_engine, Engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .config import get_config
from .exceptions import DatabaseError

# Import libSQL client with fallback for systems where it's not available
try:
    import libsql_client
    LIBSQL_AVAILABLE = True
except ImportError:
    LIBSQL_AVAILABLE = False


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
            if config.database.url.startswith('libsql:'):
                # libSQL/Turso configuration
                _engine = _create_libsql_engine(config)
            elif config.database.url.startswith('sqlite:'):
                # SQLite-specific configuration
                _engine = _create_sqlite_engine(config)
            else:
                # Generic database configuration
                _engine = _create_generic_engine(config)
        except Exception as e:
            raise DatabaseError(
                f"Failed to create database engine: {str(e)}",
                operation="create_engine",
                url=config.database.url
            ) from e
    
    return _engine


def _create_libsql_engine(config) -> Engine:
    """Create a SQLAlchemy engine for libSQL/Turso databases."""
    if not LIBSQL_AVAILABLE:
        raise DatabaseError(
            "libSQL client is not available. Install with: pip install libsql-client",
            operation="create_libsql_engine"
        )
    
    database_url = config.database.url
    auth_token = config.database.turso_auth_token
    
    # Parse URL to check for embedded auth token
    parsed_url = urlparse(database_url)
    query_params = parse_qs(parsed_url.query)
    
    # Check for auth token in URL parameters
    if 'authToken' in query_params and query_params['authToken']:
        auth_token = query_params['authToken'][0]
    
    # Check for auth token in environment variable if not found in config or URL
    if not auth_token:
        auth_token = os.getenv('TURSO_AUTH_TOKEN')
    
    if not auth_token:
        raise DatabaseError(
            "Turso auth token is required for libSQL connections. "
            "Set TURSO_AUTH_TOKEN environment variable or include authToken in URL",
            operation="create_libsql_engine"
        )
    
    # Clean URL for libSQL client (remove auth token from query params)
    clean_url = database_url
    if 'authToken' in query_params:
        # Rebuild URL without authToken parameter
        clean_query = '&'.join([f"{k}={v[0]}" for k, v in query_params.items() if k != 'authToken'])
        clean_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
        if clean_query:
            clean_url += f"?{clean_query}"
    
    try:
        # Test libSQL connection before creating SQLAlchemy engine
        libsql_client_instance = libsql_client.create_client(
            url=clean_url,
            auth_token=auth_token
        )
        
        # Test the connection
        try:
            libsql_client_instance.execute("SELECT 1")
        except Exception as e:
            raise DatabaseError(
                f"libSQL connection test failed: {str(e)}. "
                "Check your database URL and auth token",
                operation="create_libsql_engine",
                url=clean_url
            ) from e
        finally:
            libsql_client_instance.close()
        
        # For SQLAlchemy compatibility, we'll convert libsql:// URLs to sqlite://
        # and store the libSQL configuration for later use
        # This approach allows us to leverage SQLAlchemy's SQLite dialect
        # while maintaining libSQL connection capability
        
        # Convert to local SQLite URL for SQLAlchemy compatibility
        # In production, this would need a more sophisticated approach
        # like a custom SQLAlchemy dialect or connection pool
        
        # For now, we'll create a warning and fall back to a local SQLite database
        # This maintains functionality while indicating the limitation
        import warnings
        warnings.warn(
            "libSQL support is experimental. Using local SQLite for SQLAlchemy compatibility. "
            "Full libSQL integration requires custom SQLAlchemy dialect.",
            UserWarning
        )
        
        # Create a local SQLite engine with libSQL metadata stored
        local_db_path = "database/turso_local_cache.db"
        sqlite_url = f"sqlite:///{local_db_path}"
        
        engine = create_engine(
            sqlite_url,
            echo=config.database.echo,
            poolclass=StaticPool,
            connect_args={
                'check_same_thread': False,
                'timeout': 30,
                # Store libSQL connection info for future use
                '_libsql_url': clean_url,
                '_libsql_token': auth_token,
                '_libsql_sync_enabled': config.database.turso_sync_enabled,
            }
        )
        
        return engine
        
    except Exception as e:
        if "libSQL connection test failed" in str(e):
            raise e
        raise DatabaseError(
            f"Failed to create libSQL engine: {str(e)}. "
            "Ensure the database URL is correct and auth token is valid",
            operation="create_libsql_engine",
            url=clean_url
        ) from e


def _create_sqlite_engine(config) -> Engine:
    """Create a SQLAlchemy engine for SQLite databases."""
    return create_engine(
        config.database.url,
        echo=config.database.echo,
        poolclass=StaticPool,
        connect_args={
            'check_same_thread': False,  # Allow SQLite to be used across threads
            'timeout': 20,
        }
    )


def _create_generic_engine(config) -> Engine:
    """Create a SQLAlchemy engine for generic databases (PostgreSQL, etc.)."""
    return create_engine(
        config.database.url,
        echo=config.database.echo,
        pool_size=config.database.pool_size,
    )


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
        
        # Import models to register them with Base metadata
        # This ensures all models are available for table creation
        from src.storage.models import Question, BenchmarkRun, BenchmarkResult, ModelPerformance
        
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
        # Import models to ensure they are registered with Base metadata
        from src.storage.models import Question, BenchmarkRun, BenchmarkResult, ModelPerformance
        
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
        # Import models to ensure they are registered with Base metadata
        from src.storage.models import Question, BenchmarkRun, BenchmarkResult, ModelPerformance
        
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