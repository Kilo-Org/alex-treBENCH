"""
Configuration Manager

Singleton configuration manager with hot-reloading, environment-specific configs,
and CLI support for configuration operations.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import asdict
import yaml
from copy import deepcopy

from .config import AppConfig, get_config, set_config, reload_config
from .config_validator import ConfigValidator
from src.core.exceptions import ConfigurationError


class ConfigManager:
    """Singleton configuration manager with advanced features."""

    _instance: Optional['ConfigManager'] = None
    _config: Optional[AppConfig] = None
    _config_path: Optional[Path] = None
    _validator: ConfigValidator = None

    def __new__(cls) -> 'ConfigManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._validator = ConfigValidator()
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._environment_configs: Dict[str, Path] = {}
            self._model_configs: Dict[str, Path] = {}
            self._scan_config_files()

    def _scan_config_files(self):
        """Scan for available configuration files."""
        config_dir = Path("config")

        if config_dir.exists():
            # Scan environment configs
            for env_file in config_dir.glob("*.yaml"):
                if env_file.name != "default.yaml":
                    env_name = env_file.stem
                    self._environment_configs[env_name] = env_file

            # Scan model configs
            models_dir = config_dir / "models"
            if models_dir.exists():
                for model_file in models_dir.glob("*.yaml"):
                    model_name = model_file.stem
                    self._model_configs[model_name] = model_file

    def load_config(self, config_path: Optional[Path] = None, environment: Optional[str] = None) -> AppConfig:
        """
        Load configuration with optional environment overrides.

        Args:
            config_path: Path to base configuration file
            environment: Environment name for overrides

        Returns:
            Loaded and validated AppConfig
        """
        if config_path is None:
            config_path = Path("config/default.yaml")

        # Load base configuration
        base_config = self._validator.validate_and_load(config_path)

        # Apply environment overrides if specified
        if environment:
            env_config_path = self._environment_configs.get(environment)
            if env_config_path and env_config_path.exists():
                env_config_data = self._load_yaml_file(env_config_path)
                merged_data = self._validator.merge_configs(asdict(base_config), env_config_data)
                base_config = AppConfig(**merged_data)

        # Apply model-specific overrides
        model_overrides = self._load_model_overrides()
        if model_overrides:
            config_dict = asdict(base_config)
            merged_data = self._validator.merge_configs(config_dict, model_overrides)
            base_config = AppConfig(**merged_data)

        self._config = base_config
        self._config_path = config_path
        set_config(base_config)

        return base_config

    def reload_config(self) -> AppConfig:
        """Reload configuration from current path."""
        if self._config_path is None:
            raise ConfigurationError("No configuration path set")

        return self.load_config(self._config_path)

    def get_config(self) -> AppConfig:
        """Get current configuration instance."""
        if self._config is None:
            self.load_config()
        return self._config

    def set_environment(self, environment: str) -> AppConfig:
        """
        Switch to a different environment configuration.

        Args:
            environment: Environment name

        Returns:
            Updated configuration
        """
        if environment not in self._environment_configs:
            available = list(self._environment_configs.keys())
            raise ConfigurationError(f"Environment '{environment}' not found. Available: {available}")

        return self.load_config(self._config_path, environment)

    def export_config(self, output_path: Path, format: str = "yaml") -> None:
        """
        Export current configuration to file.

        Args:
            output_path: Path to export configuration
            format: Export format ('yaml' or 'json')
        """
        config_dict = asdict(self.get_config())

        if format.lower() == "yaml":
            with open(output_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        elif format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ConfigurationError(f"Unsupported export format: {format}")

    def import_config(self, input_path: Path) -> AppConfig:
        """
        Import configuration from file.

        Args:
            input_path: Path to configuration file

        Returns:
            Imported and validated configuration
        """
        if not input_path.exists():
            raise ConfigurationError(f"Configuration file not found: {input_path}")

        return self.load_config(input_path)

    def update_config_value(self, key_path: str, value: Any) -> AppConfig:
        """
        Update a specific configuration value.

        Args:
            key_path: Dot-separated path to configuration key (e.g., 'database.url')
            value: New value

        Returns:
            Updated configuration
        """
        config_dict = asdict(self.get_config())
        keys = key_path.split('.')

        # Navigate to the target location
        current = config_dict
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Update the value
        current[keys[-1]] = value

        # Validate and reload
        validated_data = self._validator.validate_config(config_dict)
        self._config = AppConfig(**validated_data)
        set_config(self._config)

        return self._config

    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """
        Get a specific configuration value.

        Args:
            key_path: Dot-separated path to configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        config_dict = asdict(self.get_config())
        keys = key_path.split('.')

        current = config_dict
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    def list_environments(self) -> List[str]:
        """List available environment configurations."""
        return list(self._environment_configs.keys())

    def list_models(self) -> List[str]:
        """List available model configurations."""
        return list(self._model_configs.keys())

    def validate_current_config(self) -> bool:
        """Validate current configuration."""
        try:
            config_dict = asdict(self.get_config())
            self._validator.validate_config(config_dict)
            return True
        except ConfigurationError:
            return False

    def get_config_info(self) -> Dict[str, Any]:
        """Get information about current configuration."""
        return {
            'config_path': str(self._config_path) if self._config_path else None,
            'environment': self.get_config().environment,
            'available_environments': self.list_environments(),
            'available_models': self.list_models(),
            'is_valid': self.validate_current_config()
        }

    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML file safely."""
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {file_path}: {e}")

    def _load_model_overrides(self) -> Dict[str, Any]:
        """Load and merge model-specific overrides."""
        model_overrides = {}

        for model_name, config_path in self._model_configs.items():
            if config_path.exists():
                model_config = self._load_yaml_file(config_path)
                model_overrides = self._validator.merge_configs(model_overrides, model_config)

        return model_overrides


# Global instance
config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    return config_manager


# CLI Helper Functions
def cli_show_config() -> None:
    """CLI command to display current configuration."""
    config = config_manager.get_config()
    config_dict = asdict(config)
    print(yaml.dump(config_dict, default_flow_style=False, sort_keys=False))


def cli_validate_config() -> None:
    """CLI command to validate current configuration."""
    if config_manager.validate_current_config():
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration has errors")
        raise SystemExit(1)


def cli_set_config_value(key: str, value: str) -> None:
    """CLI command to set a configuration value."""
    # Try to parse value as appropriate type
    if value.lower() in ('true', 'false'):
        parsed_value = value.lower() == 'true'
    elif value.isdigit():
        parsed_value = int(value)
    elif value.replace('.', '').isdigit():
        parsed_value = float(value)
    else:
        parsed_value = value

    config_manager.update_config_value(key, parsed_value)
    print(f"Updated {key} = {parsed_value}")


def cli_export_config(output_path: str, format: str = "yaml") -> None:
    """CLI command to export configuration."""
    config_manager.export_config(Path(output_path), format)
    print(f"Configuration exported to {output_path}")


def cli_list_environments() -> None:
    """CLI command to list available environments."""
    envs = config_manager.list_environments()
    if envs:
        print("Available environments:")
        for env in envs:
            print(f"  - {env}")
    else:
        print("No environment configurations found")


def cli_switch_environment(environment: str) -> None:
    """CLI command to switch environment."""
    config_manager.set_environment(environment)
    print(f"Switched to environment: {environment}")