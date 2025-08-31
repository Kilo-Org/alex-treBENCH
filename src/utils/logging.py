"""
Logging Configuration

Centralized logging setup with configurable levels, file rotation,
and structured logging for the benchmarking system.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

from core.config import get_config


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'benchmark_id'):
            log_entry['benchmark_id'] = record.benchmark_id
        if hasattr(record, 'model_name'):
            log_entry['model_name'] = record.model_name
        if hasattr(record, 'question_id'):
            log_entry['question_id'] = record.question_id
            
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


class BenchmarkLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds benchmark context to log records."""
    
    def __init__(self, logger: logging.Logger, extra: dict):
        """Initialize adapter with context."""
        super().__init__(logger, extra)
    
    def process(self, msg: str, kwargs: dict) -> tuple:
        """Add extra context to log record."""
        # Add the extra context to the log record
        extra = kwargs.get('extra', {})
        extra.update(self.extra)
        kwargs['extra'] = extra
        return msg, kwargs


def setup_logging(config=None, enable_json: bool = False) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        config: Optional configuration object (uses default if None)
        enable_json: Enable JSON formatted logging
    """
    if config is None:
        config = get_config()
    
    # Ensure logs directory exists
    log_file = Path(config.logging.file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.logging.level.upper()))
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler - use console_level for reduced terminal output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.logging.console_level.upper()))
    
    # File handler with rotation - use main level for comprehensive file logging
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=_parse_size(config.logging.max_size),
        backupCount=config.logging.backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, config.logging.level.upper()))
    
    # Set formatters
    if enable_json:
        json_formatter = JSONFormatter()
        console_handler.setFormatter(json_formatter)
        file_handler.setFormatter(json_formatter)
    else:
        # Standard formatters
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        file_handler.setFormatter(file_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    _configure_third_party_loggers()
    
    # Log startup message - this will only show in file if console_level is WARNING+
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Console: {config.logging.console_level}, File: {config.logging.level}, Path: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def get_benchmark_logger(benchmark_id: int, model_name: str = None) -> BenchmarkLoggerAdapter:
    """
    Get a logger adapter with benchmark context.
    
    Args:
        benchmark_id: Benchmark ID for context
        model_name: Optional model name for context
        
    Returns:
        Logger adapter with benchmark context
    """
    logger = get_logger('benchmark')
    extra = {'benchmark_id': benchmark_id}
    
    if model_name:
        extra['model_name'] = model_name
    
    return BenchmarkLoggerAdapter(logger, extra)


def _parse_size(size_str: str) -> int:
    """
    Parse size string (e.g., '10MB') to bytes.
    
    Args:
        size_str: Size string like '10MB', '1GB', etc.
        
    Returns:
        Size in bytes
    """
    size_str = size_str.upper().strip()
    
    # Size multipliers
    multipliers = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 ** 2,
        'GB': 1024 ** 3
    }
    
    # Extract number and unit
    for unit, multiplier in multipliers.items():
        if size_str.endswith(unit):
            number_str = size_str[:-len(unit)].strip()
            try:
                number = float(number_str)
                return int(number * multiplier)
            except ValueError:
                break
    
    # Default to 10MB if parsing fails
    return 10 * 1024 * 1024


def _configure_third_party_loggers() -> None:
    """Configure third-party library loggers."""
    # Reduce noise from third-party libraries
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)
    
    # Kaggle hub can be noisy
    logging.getLogger('kagglehub').setLevel(logging.WARNING)


# Performance logging helpers

class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, operation: str, logger: logging.Logger = None):
        """
        Initialize performance timer.
        
        Args:
            operation: Description of the operation being timed
            logger: Logger instance to use
        """
        self.operation = operation
        self.logger = logger or get_logger(__name__)
        self.start_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = datetime.now()
        self.logger.debug(f"Started {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log duration."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(f"Completed {self.operation} in {duration:.3f}s")
        else:
            self.logger.error(f"Failed {self.operation} after {duration:.3f}s: {exc_val}")


def log_function_call(func):
    """Decorator to log function calls with timing."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        func_name = f"{func.__module__}.{func.__name__}"
        
        with PerformanceTimer(f"function call: {func_name}", logger):
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Function {func_name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"Function {func_name} raised {type(e).__name__}: {str(e)}")
                raise
    
    return wrapper


# Async logging helpers

async def log_async_function_call(func):
    """Decorator to log async function calls with timing."""
    async def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        func_name = f"{func.__module__}.{func.__name__}"
        
        start_time = datetime.now()
        logger.debug(f"Started async function: {func_name}")
        
        try:
            result = await func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.debug(f"Async function {func_name} completed in {duration:.3f}s")
            return result
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.error(f"Async function {func_name} failed after {duration:.3f}s: {str(e)}")
            raise
    
    return wrapper