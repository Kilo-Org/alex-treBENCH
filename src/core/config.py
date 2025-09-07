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

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass


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
    turso_auth_token: Optional[str] = None
    turso_sync_enabled: bool = False


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
        "referer": "https://alex-trebench.local",
        "title": "alex-treBENCH Benchmarking System"
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
    """Benchmark parallel execution configuration."""
    max_concurrent_models: int = 3
    max_concurrent_requests: int = 10
    rate_limit_per_minute: int = 60


@dataclass
class BenchmarkSamplingConfig:
    """Benchmark sampling configuration."""
    method: str = "stratified"  # Options: random, stratified, balanced, temporal
    seed: Optional[int] = 42  # Fixed seed for reproducibility (None for random)
    stratify_columns: List[str] = field(default_factory=lambda: ["category", "difficulty_level"])
    difficulty_distribution: Optional[Dict[str, float]] = field(default_factory=lambda: {
        "Easy": 0.4,
        "Medium": 0.4,
        "Hard": 0.2
    })
    enable_temporal_stratification: bool = False
    confidence_level: float = 0.95
    margin_of_error: float = 0.05


@dataclass
class BenchmarkConfig:
    """Benchmark execution configuration."""
    modes: Dict[str, BenchmarkModeConfig] = field(default_factory=lambda: {
        "quick": BenchmarkModeConfig(sample_size=50, categories=5, timeout=30, max_retries=2),
        "standard": BenchmarkModeConfig(sample_size=200, categories=10, timeout=60, max_retries=3),
        "comprehensive": BenchmarkModeConfig(sample_size=1000, categories="all", timeout=120, max_retries=3)
    })
    sampling: BenchmarkSamplingConfig = field(default_factory=BenchmarkSamplingConfig)
    grading: BenchmarkGradingConfig = field(default_factory=BenchmarkGradingConfig)
    parallel: BenchmarkParallelConfig = field(default_factory=BenchmarkParallelConfig)
    default_sample_size: int = 1000
    confidence_level: float = 0.95
    margin_of_error: float = 0.05
    answer_similarity_threshold: float = 0.7
    max_concurrent_requests: int = 5


@dataclass
class DebugModeConfig:
    """Debug mode configuration for detailed logging."""
    enabled: bool = False
    log_dir: str = "logs/debug"
    log_prompts: bool = True
    log_responses: bool = True
    log_grading: bool = True
    log_errors_only: bool = False  # If True, only log incorrect answers and errors
    include_tokens: bool = True
    include_costs: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    console_level: str = "WARNING"  # Separate level for console output
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/benchmark.log"
    max_size: str = "10MB"
    backup_count: int = 5
    debug: DebugModeConfig = field(default_factory=DebugModeConfig)


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
class ModelCacheConfig:
    """Model cache configuration."""
    enabled: bool = True
    path: str = "data/cache/models.json"
    ttl_seconds: int = 3600  # 1 hour
    auto_refresh: bool = True
    fallback_to_static: bool = True


@dataclass
class ModelsConfig:
    """Model configuration with defaults, overrides, and dynamic settings."""
    default: str = "anthropic/claude-3.5-sonnet"
    defaults: ModelDefaultsConfig = field(default_factory=ModelDefaultsConfig)
    overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    preferred: Optional[Dict[str, List[str]]] = None


@dataclass
class PromptTemplateConfig:
    """Configuration for a specific prompt template."""
    system_prompt: str


@dataclass
class PromptsConfig:
    """Prompt configuration."""
    default_template: str = "jeopardy_style"
    include_category: bool = True
    include_value: bool = True
    include_difficulty: bool = False
    max_length: int = 1000
    templates: Dict[str, PromptTemplateConfig] = field(default_factory=lambda: {
        "basic_qa": PromptTemplateConfig(
            system_prompt="You are an expert quiz contestant. Answer the question accurately and concisely."
        ),
        "jeopardy_style": PromptTemplateConfig(
            system_prompt="You are a Jeopardy! contestant. Respond to each clue in the form of a question."
        ),
        "chain_of_thought": PromptTemplateConfig(
            system_prompt="You are a Jeopardy! contestant. Think through the clue step by step, then provide your final answer in the form of a question."
        )
    })


@dataclass
class CostEstimationConfig:
    """Cost estimation configuration."""
    default_input_tokens_per_question: int = 100
    default_output_tokens_per_question: int = 50


@dataclass
class CostsConfig:
    """Cost tracking configuration."""
    billing_tier: str = "basic"
    track_usage: bool = True
    usage_file: str = "data/usage/model_usage.json"
    estimation: CostEstimationConfig = field(default_factory=CostEstimationConfig)


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
    model_cache: ModelCacheConfig = field(default_factory=ModelCacheConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    costs: CostsConfig = field(default_factory=CostsConfig)
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

        # Handle nested app configuration structure
        if 'app' in config_data:
            app_config = config_data.pop('app')
            # Merge app-level config into the main config
            config_data.update(app_config)

        # Convert nested dictionaries to their corresponding dataclass objects
        if 'database' in config_data and isinstance(config_data['database'], dict):
            db_data = config_data['database']
            if 'backup' in db_data and isinstance(db_data['backup'], dict):
                db_data['backup'] = DatabaseBackupConfig(**db_data['backup'])
            config_data['database'] = DatabaseConfig(**db_data)

        if 'openrouter' in config_data and isinstance(config_data['openrouter'], dict):
            config_data['openrouter'] = OpenRouterConfig(**config_data['openrouter'])

        if 'cache' in config_data and isinstance(config_data['cache'], dict):
            config_data['cache'] = CacheConfig(**config_data['cache'])

        if 'model_cache' in config_data and isinstance(config_data['model_cache'], dict):
            config_data['model_cache'] = ModelCacheConfig(**config_data['model_cache'])

        if 'prompts' in config_data and isinstance(config_data['prompts'], dict):
            prompts_data = config_data['prompts']
            if 'templates' in prompts_data and isinstance(prompts_data['templates'], dict):
                templates = {}
                for name, template in prompts_data['templates'].items():
                    templates[name] = PromptTemplateConfig(**template)
                prompts_data['templates'] = templates
            config_data['prompts'] = PromptsConfig(**prompts_data)

        if 'costs' in config_data and isinstance(config_data['costs'], dict):
            costs_data = config_data['costs']
            if 'estimation' in costs_data and isinstance(costs_data['estimation'], dict):
                costs_data['estimation'] = CostEstimationConfig(**costs_data['estimation'])
            config_data['costs'] = CostsConfig(**costs_data)

        if 'benchmark' in config_data and isinstance(config_data['benchmark'], dict):
            benchmark_data = config_data['benchmark']
            
            # Handle modes
            if 'modes' in benchmark_data and isinstance(benchmark_data['modes'], dict):
                modes = {}
                for name, mode in benchmark_data['modes'].items():
                    modes[name] = BenchmarkModeConfig(**mode)
                benchmark_data['modes'] = modes
            
            # Handle sampling
            if 'sampling' in benchmark_data and isinstance(benchmark_data['sampling'], dict):
                benchmark_data['sampling'] = BenchmarkSamplingConfig(**benchmark_data['sampling'])
            
            # Handle grading
            if 'grading' in benchmark_data and isinstance(benchmark_data['grading'], dict):
                benchmark_data['grading'] = BenchmarkGradingConfig(**benchmark_data['grading'])
            
            # Handle parallel
            if 'parallel' in benchmark_data and isinstance(benchmark_data['parallel'], dict):
                benchmark_data['parallel'] = BenchmarkParallelConfig(**benchmark_data['parallel'])
            
            config_data['benchmark'] = BenchmarkConfig(**benchmark_data)

        if 'logging' in config_data and isinstance(config_data['logging'], dict):
            logging_data = config_data['logging']
            if 'debug' in logging_data and isinstance(logging_data['debug'], dict):
                logging_data['debug'] = DebugModeConfig(**logging_data['debug'])
            config_data['logging'] = LoggingConfig(**logging_data)

        if 'kaggle' in config_data and isinstance(config_data['kaggle'], dict):
            config_data['kaggle'] = KaggleConfig(**config_data['kaggle'])

        if 'models' in config_data and isinstance(config_data['models'], dict):
            models_data = config_data['models']
            if 'defaults' in models_data and isinstance(models_data['defaults'], dict):
                models_data['defaults'] = ModelDefaultsConfig(**models_data['defaults'])
            # Keep preferred as-is (it's already a dict of lists)
            config_data['models'] = ModelsConfig(**models_data)

        if 'reporting' in config_data and isinstance(config_data['reporting'], dict):
            config_data['reporting'] = ReportingConfig(**config_data['reporting'])

        return cls(**config_data)

    @staticmethod
    def _apply_env_overrides(config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        env_mappings = {
            'DATABASE_URL': ['database', 'url'],
            'TURSO_AUTH_TOKEN': ['database', 'turso_auth_token'],
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
