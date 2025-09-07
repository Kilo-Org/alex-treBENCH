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

import sqlalchemy_libsql

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
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Creating database engine with URL: {config.database.url}")
        
        try:
            # Configure engine based on database URL
            if config.database.url.startswith('libsql:'):
                # libSQL/Turso configuration
                logger.info("Using libSQL/Turso engine configuration")
                _engine = _create_libsql_engine(config)
            elif config.database.url.startswith('sqlite:'):
                # SQLite-specific configuration
                logger.info("Using SQLite engine configuration")
                _engine = _create_sqlite_engine(config)
            else:
                # Generic database configuration
                logger.info("Using generic engine configuration")
                _engine = _create_generic_engine(config)
        except Exception as e:
            raise DatabaseError(
                f"Failed to create database engine: {str(e)}",
                operation="create_engine",
                url=config.database.url
            ) from e
    
    return _engine


def _create_libsql_engine(config) -> Engine:
    """Create a SQLAlchemy engine for libSQL/Turso databases using sqlalchemy-libsql dialect."""
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
    
    # Clean URL and convert to sqlite+libsql:// dialect
    # Remove authToken from query params for security
    clean_query_params = {k: v for k, v in query_params.items() if k != 'authToken'}
    clean_query = '&'.join(f"{k}={v[0]}" for k, v in clean_query_params.items())
    
    # Build libSQL URL with dialect
    # The sqlite+libsql dialect handles the protocol internally
    libsql_url = f"sqlite+libsql://{parsed_url.netloc}{parsed_url.path}?secure=true"
    if clean_query:
        libsql_url += f"&{clean_query}"
    
    # Debug logging
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Creating libSQL engine with URL: {libsql_url}")
    logger.info(f"Using auth_token: {'***' if auth_token else 'None'}")
    
    # Create engine with libSQL dialect and auth_token
    engine = create_engine(
        libsql_url,
        echo=config.database.echo,
        connect_args={
            'auth_token': auth_token,
        }
    )
    
    return engine


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
        
        import logging
        logger = logging.getLogger(__name__)
        
        # Debug: Check what tables are registered
        table_names = list(Base.metadata.tables.keys())
        logger.info(f"Registered table names in metadata: {table_names}")
        
        # Create tables (PostgreSQL autocommits DDL)
        Base.metadata.create_all(engine)
        logger.info("create_all() completed")
        
        # Verify tables were created using database-agnostic approach
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        logger.info(f"Tables found in database: {tables}")
        
        # Only check for tables that are actually registered
        required_tables = [name for name in table_names if name in ['questions', 'benchmark_runs', 'benchmark_results', 'model_performance']]
        missing_tables = [t for t in required_tables if t not in tables]
        
        if missing_tables:
            logger.error(f"Missing tables: {missing_tables}")
            logger.error(f"Available tables: {tables}")
            logger.error(f"Expected tables: {required_tables}")
            raise DatabaseError(f"Failed to create required tables: {missing_tables}")
        else:
            logger.info(f"All required tables created successfully: {required_tables}")
            
        # Additional verification: try to query each table with a new session
        # This ensures tables are visible to new connections
        try:
            from sqlalchemy.orm import sessionmaker
            SessionLocal = sessionmaker(bind=engine)
            with SessionLocal() as test_session:
                from src.storage.models import Question
                test_session.query(Question).count()
                logger.info("Table visibility verified with new session")
        except Exception as e:
            logger.error(f"Table visibility check failed: {e}")
            raise DatabaseError(f"Tables created but not visible to new sessions: {e}")
                
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