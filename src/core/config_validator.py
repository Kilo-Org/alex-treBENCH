"""
Configuration Validator

Validates configuration files against schema requirements,
provides helpful error messages, and supports configuration merging.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import asdict
import yaml
from copy import deepcopy

from .config import AppConfig, get_config
from src.core.exceptions import ConfigurationError


class ConfigValidator:
    """Configuration validator with schema validation and merging support."""

    def __init__(self):
        self.required_fields = {
            'app': ['name', 'version'],
            'database': ['url'],
            'openrouter': ['base_url', 'default_model'],
            'benchmark': ['default_sample_size', 'confidence_level'],
            'logging': ['level', 'file'],
            'kaggle': ['dataset', 'cache_dir']
        }

        self.field_validators = {
            'app.version': self._validate_version,
            'database.pool_size': self._validate_positive_int,
            'openrouter.timeout': self._validate_positive_int,
            'openrouter.max_retries': self._validate_non_negative_int,
            'benchmark.default_sample_size': self._validate_positive_int,
            'benchmark.confidence_level': self._validate_confidence_level,
            'benchmark.margin_of_error': self._validate_margin_of_error,
            'benchmark.answer_similarity_threshold': self._validate_similarity_threshold,
            'benchmark.max_concurrent_requests': self._validate_positive_int,
            'cache.ttl': self._validate_positive_int,
            'cache.max_size': self._validate_positive_int,
            'logging.backup_count': self._validate_non_negative_int
        }

        self.range_validators = {
            'benchmark.confidence_level': (0.0, 1.0),
            'benchmark.margin_of_error': (0.0, 1.0),
            'benchmark.answer_similarity_threshold': (0.0, 1.0),
            'models.defaults.temperature': (0.0, 2.0),
            'models.defaults.top_p': (0.0, 1.0)
        }

    def validate_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration data against schema.

        Args:
            config_data: Configuration dictionary to validate

        Returns:
            Validated and merged configuration data

        Raises:
            ConfigurationError: If validation fails
        """
        errors = []

        # Check required fields
        errors.extend(self._check_required_fields(config_data))

        # Validate field types and ranges
        errors.extend(self._validate_field_types(config_data))

        # Validate specific field values
        errors.extend(self._validate_field_values(config_data))

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ConfigurationError(error_msg)

        return config_data

    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge override configuration with base configuration.

        Args:
            base_config: Base configuration
            override_config: Override configuration

        Returns:
            Merged configuration
        """
        merged = deepcopy(base_config)

        def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
            return base

        return deep_merge(merged, override_config)

    def validate_and_load(self, config_path: Path) -> AppConfig:
        """
        Load and validate configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            Validated AppConfig instance

        Raises:
            ConfigurationError: If validation fails
        """
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in configuration file: {e}")

        # Validate the loaded configuration
        validated_data = self.validate_config(config_data)

        # Try to create AppConfig instance to catch any remaining issues
        try:
            config = AppConfig(**validated_data)
            return config
        except Exception as e:
            raise ConfigurationError(f"Configuration instantiation failed: {e}")

    def validate_environment_config(self, env_prefix: str = "JB_") -> Dict[str, Any]:
        """
        Validate environment variable configurations.

        Args:
            env_prefix: Prefix for environment variables

        Returns:
            Dictionary of validated environment configurations
        """
        env_config = {}
        errors = []

        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):].lower().replace('_', '.')
                try:
                    # Try to convert value to appropriate type
                    if value.lower() in ('true', 'false'):
                        env_config[config_key] = value.lower() == 'true'
                    elif value.isdigit():
                        env_config[config_key] = int(value)
                    elif value.replace('.', '').isdigit():
                        env_config[config_key] = float(value)
                    else:
                        env_config[config_key] = value
                except ValueError:
                    errors.append(f"Invalid value for {key}: {value}")

        if errors:
            raise ConfigurationError("Environment configuration errors:\n" + "\n".join(errors))

        return env_config

    def _check_required_fields(self, config_data: Dict[str, Any]) -> List[str]:
        """Check for required configuration fields."""
        errors = []

        for section, fields in self.required_fields.items():
            if section not in config_data:
                errors.append(f"Missing required section: {section}")
                continue

            section_data = config_data[section]
            if not isinstance(section_data, dict):
                errors.append(f"Section '{section}' must be a dictionary")
                continue

            for field in fields:
                if field not in section_data:
                    errors.append(f"Missing required field: {section}.{field}")

        return errors

    def _validate_field_types(self, config_data: Dict[str, Any]) -> List[str]:
        """Validate field types."""
        errors = []

        def validate_section(section_data: Dict[str, Any], path: str = ""):
            for key, value in section_data.items():
                current_path = f"{path}.{key}" if path else key

                if isinstance(value, dict):
                    validate_section(value, current_path)
                elif key in self.field_validators:
                    try:
                        self.field_validators[key](value)
                    except ValueError as e:
                        errors.append(f"Invalid value for {current_path}: {e}")

        validate_section(config_data)
        return errors

    def _validate_field_values(self, config_data: Dict[str, Any]) -> List[str]:
        """Validate specific field value ranges."""
        errors = []

        def validate_ranges(section_data: Dict[str, Any], path: str = ""):
            for key, value in section_data.items():
                current_path = f"{path}.{key}" if path else key

                if current_path in self.range_validators:
                    min_val, max_val = self.range_validators[current_path]
                    if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                        errors.append(f"Value for {current_path} must be between {min_val} and {max_val}")

                if isinstance(value, dict):
                    validate_ranges(value, current_path)

        validate_ranges(config_data)
        return errors

    def _validate_version(self, value: str) -> None:
        """Validate version string format."""
        if not isinstance(value, str):
            raise ValueError("Version must be a string")
        parts = value.split('.')
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            raise ValueError("Version must be in format x.y.z")

    def _validate_positive_int(self, value: Any) -> None:
        """Validate positive integer."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Must be a positive integer")

    def _validate_non_negative_int(self, value: Any) -> None:
        """Validate non-negative integer."""
        if not isinstance(value, int) or value < 0:
            raise ValueError("Must be a non-negative integer")

    def _validate_confidence_level(self, value: Any) -> None:
        """Validate confidence level."""
        if not isinstance(value, (int, float)) or not (0 < value < 1):
            raise ValueError("Confidence level must be between 0 and 1")

    def _validate_margin_of_error(self, value: Any) -> None:
        """Validate margin of error."""
        if not isinstance(value, (int, float)) or not (0 < value <= 0.5):
            raise ValueError("Margin of error must be between 0 and 0.5")

    def _validate_similarity_threshold(self, value: Any) -> None:
        """Validate similarity threshold."""
        if not isinstance(value, (int, float)) or not (0 <= value <= 1):
            raise ValueError("Similarity threshold must be between 0 and 1")


def validate_configuration(config_path: Optional[Path] = None) -> AppConfig:
    """
    Convenience function to validate and load configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Validated AppConfig instance
    """
    validator = ConfigValidator()
    if config_path is None:
        config_path = Path("config/default.yaml")

    return validator.validate_and_load(config_path)