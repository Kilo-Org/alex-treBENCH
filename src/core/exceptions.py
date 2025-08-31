"""
Custom Exception Classes

Application-specific exception classes for better error handling
and debugging throughout the benchmarking system.
"""

from typing import Optional, Any, Dict


class JeopardyBenchException(Exception):
    """Base exception class for all Jeopardy Bench errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class ConfigurationError(JeopardyBenchException):
    """Raised when there's an issue with configuration setup or validation."""
    pass


class DatabaseError(JeopardyBenchException):
    """Raised when database operations fail."""
    
    def __init__(self, message: str, operation: Optional[str] = None, 
                 table: Optional[str] = None, **kwargs):
        super().__init__(message, kwargs)
        self.operation = operation
        self.table = table


class ModelAPIError(JeopardyBenchException):
    """Raised when API calls to language model providers fail."""
    
    def __init__(self, message: str, model_name: Optional[str] = None,
                 status_code: Optional[int] = None, 
                 response_body: Optional[str] = None, **kwargs):
        super().__init__(message, kwargs)
        self.model_name = model_name
        self.status_code = status_code
        self.response_body = response_body


class RateLimitError(ModelAPIError):
    """Raised when API rate limits are exceeded."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class EvaluationError(JeopardyBenchException):
    """Raised when answer evaluation fails."""
    
    def __init__(self, message: str, question_id: Optional[str] = None,
                 model_response: Optional[str] = None, 
                 expected_answer: Optional[str] = None, **kwargs):
        super().__init__(message, kwargs)
        self.question_id = question_id
        self.model_response = model_response
        self.expected_answer = expected_answer


class DataIngestionError(JeopardyBenchException):
    """Raised when data loading or preprocessing fails."""
    
    def __init__(self, message: str, dataset_name: Optional[str] = None,
                 file_path: Optional[str] = None, **kwargs):
        super().__init__(message, kwargs)
        self.dataset_name = dataset_name
        self.file_path = file_path


class SamplingError(JeopardyBenchException):
    """Raised when statistical sampling fails."""
    
    def __init__(self, message: str, sample_size: Optional[int] = None,
                 population_size: Optional[int] = None, **kwargs):
        super().__init__(message, kwargs)
        self.sample_size = sample_size
        self.population_size = population_size


class ValidationError(JeopardyBenchException):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, field_name: Optional[str] = None,
                 invalid_value: Optional[Any] = None, **kwargs):
        super().__init__(message, kwargs)
        self.field_name = field_name
        self.invalid_value = invalid_value


class BenchmarkExecutionError(JeopardyBenchException):
    """Raised when benchmark execution encounters critical errors."""
    
    def __init__(self, message: str, benchmark_id: Optional[str] = None,
                 current_step: Optional[str] = None, **kwargs):
        super().__init__(message, kwargs)
        self.benchmark_id = benchmark_id
        self.current_step = current_step