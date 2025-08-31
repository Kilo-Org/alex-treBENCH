"""
Configuration Management

Centralized configuration management with YAML file support,
environment variable overrides, and Pydantic validation.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import yaml


@dataclass
class DatabaseBackupConfig:
    """Database backup configuration."""
    enabled: bool = True
    rotation: int = 5
    path: str = "./backups"


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    url: str = "sqlite:///database/benchmarks.db"
    echo: bool = False
    pool_size: int = 5
    backup: DatabaseBackupConfig = field(default_factory=DatabaseBackupConfig)


@dataclass
class OpenRouterConfig:
    """OpenRouter API configuration settings."""
    base_url: str = "https://openrouter.ai/api/v1"
    timeout: int = 30
    max_retries: int = 3
    rate_limit: Dict[str, Any] = field(default_factory=lambda: {
        "requests_per_minute": 60,
        "backoff_factor": 2.0,
        "max_backoff": 60.0
    })
    default_model: str = "openai/gpt-3.5-turbo"
    streaming: bool = False
    headers: Dict[str, str] = field(default_factory=lambda: {
        "referer": "https://jeopardy-bench.local",
        "title": "Jeopardy Benchmarking System"
    })


@dataclass
class CacheConfig:
    """Cache configuration."""
    enabled: bool = True
    ttl: int = 3600
    max_size: int = 1000


@dataclass
class BenchmarkModeConfig:
    """Configuration for a specific benchmark mode."""
    sample_size: int = 1000
    categories: Any = "all"  # Can be int or "all"
    timeout: int = 60
    max_retries: int = 3


@dataclass
class BenchmarkGradingConfig:
    """Grading configuration for benchmark evaluation."""
    default_mode: str = "lenient"
    confidence_threshold: float = 0.7
    partial_credit: bool = True
    jeopardy_format_required: bool = False


@dataclass
class BenchmarkParallelConfig:
    """Parallel processing configuration."""
    max_concurrent_models: int = 3
    max_concurrent_requests: int = 10
    rate_limit_per_minute: int = 60


@dataclass
class BenchmarkConfig:
    """Benchmark execution configuration."""
    modes: Dict[str, BenchmarkModeConfig] = field(default_factory=lambda: {
        "quick": BenchmarkModeConfig(sample_size=50, categories=5, timeout=30, max_retries=2),
        "standard": BenchmarkModeConfig(sample_size=200, categories=10, timeout=60, max_retries=3),
        "comprehensive": BenchmarkModeConfig(sample_size=1000, categories="all", timeout=120, max_retries=3)
    })
    grading: BenchmarkGradingConfig = field(default_factory=BenchmarkGradingConfig)
    parallel: BenchmarkParallelConfig = field(default_factory=BenchmarkParallelConfig)
    default_sample_size: int = 1000
    confidence_level: float = 0.95
    margin_of_error: float = 0.05
    answer_similarity_threshold: float = 0.7
    max_concurrent_requests: int = 5


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/benchmark.log"
    max_size: str = "10MB"
    backup_count: int = 5


@dataclass
class KaggleConfig:
    """Kaggle dataset configuration."""
    dataset: str = "aravindram11/jeopardy-dataset-updated"
    cache_dir: str = "data/raw"


@dataclass
class ModelDefaultsConfig:
    """Default model parameters."""
    temperature: float = 0.7
    max_tokens: int = 150
    top_p: float = 0.9


@dataclass
class ModelsConfig:
    """Model configuration with defaults and overrides."""
    defaults: ModelDefaultsConfig = field(default_factory=ModelDefaultsConfig)
    overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class ReportingConfig:
    """Reporting configuration."""
    default_format: str = "terminal"
    include_metadata: bool = True
    export_path: str = "./reports"
    visualization: bool = True


@dataclass
class AppConfig:
    """Main application configuration."""
    name: str = "Jeopardy Benchmark"
    version: str = "1.0.0"
    environment: str = "production"
    debug: bool = False

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    kaggle: KaggleConfig = field(default_factory=KaggleConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)

    @classmethod
    def from_yaml(cls, config_path: Path) -> "AppConfig":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Apply environment variable overrides
        config_data = cls._apply_env_overrides(config_data)

        return cls(**config_data)

    @staticmethod
    def _apply_env_overrides(config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        env_mappings = {
            'DATABASE_URL': ['database', 'url'],
            'LOG_LEVEL': ['logging', 'level'],
            'DEBUG': ['debug'],
            'ENVIRONMENT': ['environment'],
        }

        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                current = config_data
                for key in config_path[:-1]:
                    current = current.setdefault(key, {})
                current[config_path[-1]] = env_value

        return config_data


# Global configuration instance
_config: Optional[AppConfig] = None


def get_config(config_path: Optional[Path] = None) -> AppConfig:
    """Get the application configuration instance."""
    global _config

    if _config is None:
        if config_path is None:
            # Default configuration path
            config_path = Path("config/default.yaml")

        if config_path.exists():
            _config = AppConfig.from_yaml(config_path)
        else:
            # Use default configuration if file doesn't exist
            _config = AppConfig()

    return _config


def set_config(config: AppConfig) -> None:
    """Set the global configuration instance (useful for testing)."""
    global _config
    _config = config


def reload_config(config_path: Optional[Path] = None) -> AppConfig:
    """Reload configuration from file."""
    global _config
    _config = None
    return get_config(config_path)