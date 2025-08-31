"""
Input Validation Utilities

Input validation utilities for configuration, question data,
and other system inputs with comprehensive error handling.
"""

import re
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import pandas as pd
from pydantic import BaseModel, validator, ValidationError as PydanticValidationError

from core.exceptions import ValidationError
from utils.logging import get_logger

logger = get_logger(__name__)


class QuestionDataValidator(BaseModel):
    """Pydantic model for validating question data."""
    
    question: str
    answer: str
    category: Optional[str] = None
    value: Optional[int] = None
    difficulty_level: Optional[str] = None
    
    @validator('question')
    def validate_question(cls, v):
        """Validate question text."""
        if not v or len(v.strip()) < 10:
            raise ValueError("Question must be at least 10 characters long")
        return v.strip()
    
    @validator('answer')
    def validate_answer(cls, v):
        """Validate answer text."""
        if not v or len(v.strip()) < 1:
            raise ValueError("Answer cannot be empty")
        return v.strip()
    
    @validator('value')
    def validate_value(cls, v):
        """Validate monetary value."""
        if v is not None and v < 0:
            raise ValueError("Value must be non-negative")
        return v
    
    @validator('difficulty_level')
    def validate_difficulty(cls, v):
        """Validate difficulty level."""
        if v is not None:
            valid_levels = ['Easy', 'Medium', 'Hard', 'Unknown']
            if v not in valid_levels:
                raise ValueError(f"Difficulty must be one of: {valid_levels}")
        return v


class ConfigValidator:
    """Configuration validation utilities."""
    
    @staticmethod
    def validate_database_url(url: str) -> bool:
        """
        Validate database URL format.
        
        Args:
            url: Database URL to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If URL is invalid
        """
        if not url:
            raise ValidationError("Database URL cannot be empty", field_name="database_url")
        
        # Basic URL pattern check
        url_pattern = r'^(sqlite:///|postgresql://|mysql://)'
        if not re.match(url_pattern, url, re.IGNORECASE):
            raise ValidationError(
                "Database URL must start with sqlite:///, postgresql://, or mysql://",
                field_name="database_url",
                invalid_value=url
            )
        
        return True
    
    @staticmethod
    def validate_api_key(api_key: str, key_name: str = "API key") -> bool:
        """
        Validate API key format.
        
        Args:
            api_key: API key to validate
            key_name: Name of the key for error messages
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If key is invalid
        """
        if not api_key:
            raise ValidationError(f"{key_name} cannot be empty", field_name="api_key")
        
        if len(api_key) < 10:
            raise ValidationError(
                f"{key_name} appears too short (minimum 10 characters)",
                field_name="api_key",
                invalid_value="<redacted>"
            )
        
        return True
    
    @staticmethod
    def validate_log_level(level: str) -> bool:
        """
        Validate logging level.
        
        Args:
            level: Log level to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If level is invalid
        """
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if level.upper() not in valid_levels:
            raise ValidationError(
                f"Log level must be one of: {valid_levels}",
                field_name="log_level",
                invalid_value=level
            )
        
        return True
    
    @staticmethod
    def validate_file_path(file_path: Union[str, Path], must_exist: bool = False) -> bool:
        """
        Validate file path.
        
        Args:
            file_path: File path to validate
            must_exist: Whether file must already exist
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If path is invalid
        """
        if not file_path:
            raise ValidationError("File path cannot be empty", field_name="file_path")
        
        path = Path(file_path)
        
        if must_exist and not path.exists():
            raise ValidationError(
                f"File does not exist: {file_path}",
                field_name="file_path",
                invalid_value=str(file_path)
            )
        
        # Check if parent directory is writable (for new files)
        if not must_exist:
            parent_dir = path.parent
            if parent_dir.exists() and not parent_dir.is_dir():
                raise ValidationError(
                    f"Parent path is not a directory: {parent_dir}",
                    field_name="file_path",
                    invalid_value=str(file_path)
                )
        
        return True


def validate_question_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate individual question data.
    
    Args:
        data: Question data dictionary
        
    Returns:
        Validated and cleaned data
        
    Raises:
        ValidationError: If data is invalid
    """
    try:
        validator = QuestionDataValidator(**data)
        return validator.dict()
    except PydanticValidationError as e:
        errors = []
        for error in e.errors():
            field = '.'.join(str(loc) for loc in error['loc'])
            message = error['msg']
            errors.append(f"{field}: {message}")
        
        raise ValidationError(
            f"Question data validation failed: {'; '.join(errors)}",
            invalid_value=data
        ) from e


def validate_question_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate entire question dataset.
    
    Args:
        df: DataFrame with question data
        
    Returns:
        Validated DataFrame
        
    Raises:
        ValidationError: If dataset is invalid
    """
    if df.empty:
        raise ValidationError("Question dataset cannot be empty")
    
    # Check required columns
    required_columns = ['question', 'answer']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValidationError(
            f"Missing required columns: {missing_columns}",
            field_name="dataset_columns"
        )
    
    # Validate each row
    validation_errors = []
    valid_rows = []
    
    for idx, row in df.iterrows():
        try:
            validated_data = validate_question_data(row.to_dict())
            valid_rows.append(validated_data)
        except ValidationError as e:
            validation_errors.append(f"Row {idx}: {e.message}")
            logger.warning(f"Invalid question data at row {idx}: {e.message}")
    
    if not valid_rows:
        raise ValidationError(
            f"No valid questions found. Validation errors: {'; '.join(validation_errors[:5])}"
        )
    
    # Log validation summary
    total_rows = len(df)
    valid_count = len(valid_rows)
    invalid_count = len(validation_errors)
    
    logger.info(f"Dataset validation: {valid_count}/{total_rows} valid questions, "
               f"{invalid_count} invalid questions discarded")
    
    if invalid_count > 0:
        logger.warning(f"Validation errors summary: {'; '.join(validation_errors[:10])}")
    
    # Return DataFrame with valid data only
    return pd.DataFrame(valid_rows)


def validate_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate application configuration.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Validated configuration
        
    Raises:
        ValidationError: If configuration is invalid
    """
    try:
        # Validate database configuration
        if 'database' in config_dict:
            db_config = config_dict['database']
            if 'url' in db_config:
                ConfigValidator.validate_database_url(db_config['url'])
        
        # Validate logging configuration
        if 'logging' in config_dict:
            log_config = config_dict['logging']
            if 'level' in log_config:
                ConfigValidator.validate_log_level(log_config['level'])
            if 'file' in log_config:
                ConfigValidator.validate_file_path(log_config['file'])
        
        # Validate numeric ranges
        if 'benchmarks' in config_dict:
            bench_config = config_dict['benchmarks']
            
            if 'default_sample_size' in bench_config:
                sample_size = bench_config['default_sample_size']
                if not isinstance(sample_size, int) or sample_size <= 0:
                    raise ValidationError(
                        "Sample size must be a positive integer",
                        field_name="default_sample_size",
                        invalid_value=sample_size
                    )
            
            if 'confidence_level' in bench_config:
                confidence = bench_config['confidence_level']
                if not isinstance(confidence, (int, float)) or not (0 < confidence < 1):
                    raise ValidationError(
                        "Confidence level must be between 0 and 1",
                        field_name="confidence_level",
                        invalid_value=confidence
                    )
            
            if 'max_concurrent_requests' in bench_config:
                concurrent = bench_config['max_concurrent_requests']
                if not isinstance(concurrent, int) or concurrent <= 0:
                    raise ValidationError(
                        "Max concurrent requests must be a positive integer",
                        field_name="max_concurrent_requests",
                        invalid_value=concurrent
                    )
        
        return config_dict
        
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(f"Configuration validation failed: {str(e)}") from e


def validate_model_name(model_name: str) -> bool:
    """
    Validate model name format.
    
    Args:
        model_name: Model name to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If model name is invalid
    """
    if not model_name:
        raise ValidationError("Model name cannot be empty", field_name="model_name")
    
    # Basic format validation (provider/model-name)
    if '/' not in model_name:
        raise ValidationError(
            "Model name should be in format 'provider/model-name'",
            field_name="model_name",
            invalid_value=model_name
        )
    
    parts = model_name.split('/')
    if len(parts) != 2 or not all(parts):
        raise ValidationError(
            "Model name should have exactly one '/' separator",
            field_name="model_name",
            invalid_value=model_name
        )
    
    return True


def validate_benchmark_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate benchmark execution parameters.
    
    Args:
        params: Benchmark parameters
        
    Returns:
        Validated parameters
        
    Raises:
        ValidationError: If parameters are invalid
    """
    validated = params.copy()
    
    # Validate sample size
    if 'sample_size' in params:
        sample_size = params['sample_size']
        if not isinstance(sample_size, int) or sample_size <= 0:
            raise ValidationError(
                "Sample size must be a positive integer",
                field_name="sample_size",
                invalid_value=sample_size
            )
    
    # Validate model names
    if 'models' in params:
        models = params['models']
        if not isinstance(models, list) or not models:
            raise ValidationError(
                "Models must be a non-empty list",
                field_name="models",
                invalid_value=models
            )
        
        for model in models:
            validate_model_name(model)
    
    # Validate timeout
    if 'timeout' in params:
        timeout = params['timeout']
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValidationError(
                "Timeout must be a positive number",
                field_name="timeout",
                invalid_value=timeout
            )
    
    return validated


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file system usage.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and periods
    sanitized = sanitized.strip(' .')
    
    # Limit length
    if len(sanitized) > 200:
        sanitized = sanitized[:200]
    
    # Ensure non-empty
    if not sanitized:
        sanitized = "unnamed"
    
    return sanitized


def validate_json_structure(data: Dict[str, Any], required_fields: List[str],
                          optional_fields: List[str] = None) -> bool:
    """
    Validate JSON structure has required fields.
    
    Args:
        data: JSON data to validate
        required_fields: List of required field names
        optional_fields: List of optional field names
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If structure is invalid
    """
    if not isinstance(data, dict):
        raise ValidationError("Data must be a dictionary", invalid_value=type(data))
    
    # Check required fields
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValidationError(
            f"Missing required fields: {missing_fields}",
            field_name="json_structure"
        )
    
    # Check for unexpected fields if optional_fields is provided
    if optional_fields is not None:
        allowed_fields = set(required_fields + optional_fields)
        unexpected_fields = [field for field in data.keys() if field not in allowed_fields]
        if unexpected_fields:
            logger.warning(f"Unexpected fields in data: {unexpected_fields}")
    
    return True