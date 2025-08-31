# Jeopardy Benchmarking System - API Reference

## Overview

This document provides comprehensive API reference for the Jeopardy Benchmarking System, including all public classes, methods, configuration options, and available models.

## Table of Contents

- [Core Classes](#core-classes)
- [Model Interface](#model-interface)
- [Configuration](#configuration)
- [Metrics and Evaluation](#metrics-and-evaluation)
- [Benchmark Runner](#benchmark-runner)
- [Reporting](#reporting)
- [Available Models](#available-models)
- [Error Handling](#error-handling)
- [Examples](#examples)

## Core Classes

### ModelResponse

Standardized response from a language model.

```python
@dataclass
class ModelResponse:
    model_id: str
    prompt: str
    response: str
    latency_ms: float
    tokens_used: int
    cost: float
    timestamp: datetime
    metadata: Dict[str, Any]
```

**Attributes:**
- `model_id` (str): Identifier of the model that generated the response
- `prompt` (str): The input prompt sent to the model
- `response` (str): The generated response text
- `latency_ms` (float): Response time in milliseconds
- `tokens_used` (int): Number of tokens consumed
- `cost` (float): Cost of the request in USD
- `timestamp` (datetime): When the response was generated
- `metadata` (Dict[str, Any]): Additional metadata

### ModelConfig

Configuration for a language model instance.

```python
@dataclass
class ModelConfig:
    model_name: str
    max_tokens: int = 150
    temperature: float = 0.1
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    timeout_seconds: int = 30
```

**Attributes:**
- `model_name` (str): Name/identifier of the model
- `max_tokens` (int): Maximum tokens to generate
- `temperature` (float): Sampling temperature (0.0 to 1.0)
- `top_p` (Optional[float]): Nucleus sampling parameter
- `frequency_penalty` (Optional[float]): Frequency penalty
- `presence_penalty` (Optional[float]): Presence penalty
- `stop_sequences` (Optional[List[str]]): Stop generation sequences
- `timeout_seconds` (int): Request timeout in seconds

## Model Interface

### ModelAdapter (Abstract Base Class)

Abstract base class for language model adapters.

```python
class ModelAdapter(ABC):
    def __init__(self, config: ModelConfig)

    @abstractmethod
    async def query(self, prompt: str, **kwargs) -> ModelResponse:
        """Query the model with a single prompt."""

    @abstractmethod
    async def batch_query(self, prompts: List[str], **kwargs) -> List[ModelResponse]:
        """Query the model with multiple prompts."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available."""

    @abstractmethod
    def get_pricing_info(self) -> Dict[str, float]:
        """Get pricing information."""

    def format_jeopardy_prompt(self, question: str) -> str:
        """Format a Jeopardy question as a prompt."""

    def extract_answer(self, response_text: str) -> str:
        """Extract the answer from model response."""

    def update_usage_stats(self, response: ModelResponse) -> None:
        """Update usage statistics."""

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""

    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check."""
```

### OpenRouterClient

Concrete implementation for OpenRouter API.

```python
class OpenRouterClient(ModelAdapter):
    def __init__(self, config: ModelConfig)

    async def query(self, prompt: str, **kwargs) -> ModelResponse:
        """Query OpenRouter API."""

    async def batch_query(self, prompts: List[str], **kwargs) -> List[ModelResponse]:
        """Batch query OpenRouter API."""

    def is_available(self) -> bool:
        """Check OpenRouter availability."""

    def get_pricing_info(self) -> Dict[str, float]:
        """Get model pricing from OpenRouter."""
```

**Usage Example:**
```python
from src.models.openrouter import OpenRouterClient
from src.models.base import ModelConfig

config = ModelConfig(
    model_name="openai/gpt-3.5-turbo",
    max_tokens=150,
    temperature=0.1
)

async with OpenRouterClient(config) as client:
    response = await client.query("What is the capital of France?")
    print(f"Answer: {response.response}")
    print(f"Cost: ${response.cost}")
```

## Configuration

### AppConfig

Main application configuration class.

```python
@dataclass
class AppConfig:
    name: str = "Jeopardy Benchmark"
    version: str = "1.0.0"
    environment: str = "production"
    debug: bool = False

    database: DatabaseConfig
    openrouter: OpenRouterConfig
    cache: CacheConfig
    benchmark: BenchmarkConfig
    logging: LoggingConfig
    kaggle: KaggleConfig
    models: ModelsConfig
    reporting: ReportingConfig
```

### DatabaseConfig

Database configuration options.

```python
@dataclass
class DatabaseConfig:
    url: str = "sqlite:///database/benchmarks.db"
    echo: bool = False
    pool_size: int = 5
    backup: DatabaseBackupConfig
```

**Configuration Options:**
- `url`: Database connection URL
- `echo`: Enable SQL query logging
- `pool_size`: Connection pool size
- `backup.enabled`: Enable automatic backups
- `backup.rotation`: Number of backup files to keep
- `backup.path`: Backup directory path

### OpenRouterConfig

OpenRouter API configuration.

```python
@dataclass
class OpenRouterConfig:
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
```

### BenchmarkConfig

Benchmark execution configuration.

```python
@dataclass
class BenchmarkConfig:
    modes: Dict[str, BenchmarkModeConfig]
    grading: BenchmarkGradingConfig
    parallel: BenchmarkParallelConfig
    default_sample_size: int = 1000
    confidence_level: float = 0.95
    margin_of_error: float = 0.05
    answer_similarity_threshold: float = 0.7
    max_concurrent_requests: int = 5
```

**Benchmark Modes:**
- `quick`: 50 questions, 30s timeout
- `standard`: 200 questions, 60s timeout
- `comprehensive`: 1000 questions, 120s timeout

### Configuration Loading

```python
from src.core.config import get_config, reload_config

# Load default configuration
config = get_config()

# Load from specific file
config = reload_config(Path("path/to/config.yaml"))

# Access configuration values
db_url = config.database.url
max_concurrent = config.benchmark.max_concurrent_requests
```

## Metrics and Evaluation

### ComprehensiveMetrics

Complete set of benchmark metrics.

```python
@dataclass
class ComprehensiveMetrics:
    model_name: str
    benchmark_id: Optional[int]
    timestamp: datetime

    accuracy: AccuracyMetrics
    performance: PerformanceMetrics
    cost: CostMetrics
    consistency: ConsistencyMetrics

    overall_score: float
    quality_score: float
    efficiency_score: float

    metadata: Dict[str, Any]
```

### AccuracyMetrics

Accuracy-related metrics.

```python
@dataclass
class AccuracyMetrics:
    overall_accuracy: float
    correct_count: int
    total_count: int
    by_category: Dict[str, float]
    by_difficulty: Dict[str, float]
    by_value: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
```

### PerformanceMetrics

Performance-related metrics.

```python
@dataclass
class PerformanceMetrics:
    mean_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    min_response_time: float
    max_response_time: float
    response_time_std: float
    timeout_count: int
    error_count: int
```

### CostMetrics

Cost and efficiency metrics.

```python
@dataclass
class CostMetrics:
    total_cost: float
    cost_per_question: float
    cost_per_correct_answer: float
    total_tokens: int
    tokens_per_question: float
    tokens_per_correct_answer: float
    input_tokens: int
    output_tokens: int
    cost_efficiency_score: float
```

### ConsistencyMetrics

Consistency and reliability metrics.

```python
@dataclass
class ConsistencyMetrics:
    performance_variance: float
    category_consistency_score: float
    difficulty_consistency_score: float
    confidence_correlation: float
    response_type_distribution: Dict[str, float]
    match_type_distribution: Dict[str, float]
```

### MetricsCalculator

Calculates comprehensive benchmark metrics.

```python
class MetricsCalculator:
    def __init__(self, baseline_costs: Optional[Dict[str, float]] = None)

    def calculate_metrics(self,
                         graded_responses: List[GradedResponse],
                         model_responses: List[ModelResponse],
                         model_name: str,
                         benchmark_id: Optional[int] = None,
                         question_contexts: Optional[List[Dict[str, Any]]] = None) -> ComprehensiveMetrics:
        """Calculate comprehensive metrics."""

    def compare_metrics(self, metrics1: ComprehensiveMetrics,
                       metrics2: ComprehensiveMetrics) -> Dict[str, Any]:
        """Compare two sets of metrics."""

    def aggregate_metrics(self, metrics_list: List[ComprehensiveMetrics]) -> Dict[str, Any]:
        """Aggregate multiple metric sets."""
```

**Usage Example:**
```python
from src.evaluation.metrics import MetricsCalculator

calculator = MetricsCalculator()
metrics = calculator.calculate_metrics(
    graded_responses=graded_list,
    model_responses=model_responses,
    model_name="openai/gpt-4",
    benchmark_id=123
)

print(f"Overall Accuracy: {metrics.accuracy.overall_accuracy:.1%}")
print(f"Total Cost: ${metrics.cost.total_cost:.4f}")
print(f"Mean Response Time: {metrics.performance.mean_response_time:.2f}s")
```

## Benchmark Runner

### BenchmarkRunner

Main orchestrator for running benchmarks.

```python
class BenchmarkRunner:
    def __init__(self)

    async def run_benchmark(self,
                           model_name: str,
                           mode: RunMode = RunMode.STANDARD,
                           custom_config: Optional[BenchmarkConfig] = None,
                           benchmark_name: Optional[str] = None) -> BenchmarkResult:
        """Run a complete benchmark."""

    def get_default_config(self, mode: RunMode) -> BenchmarkConfig:
        """Get default configuration for a mode."""

    def cancel_benchmark(self):
        """Cancel current benchmark."""

    async def resume_benchmark(self, benchmark_id: int) -> Optional[BenchmarkResult]:
        """Resume a paused benchmark."""
```

### RunMode

Benchmark run modes enumeration.

```python
class RunMode(str, Enum):
    QUICK = "quick"           # 50 questions, fast
    STANDARD = "standard"     # 200 questions, balanced
    COMPREHENSIVE = "comprehensive"  # 1000 questions, thorough
    CUSTOM = "custom"         # Custom configuration
```

### BenchmarkResult

Complete result of a benchmark run.

```python
@dataclass
class BenchmarkResult:
    benchmark_id: int
    model_name: str
    config: BenchmarkConfig
    progress: BenchmarkProgress
    metrics: Optional[ComprehensiveMetrics]
    questions: List[BenchmarkQuestion]
    responses: List[ModelResponse]
    graded_responses: List[GradedResponse]
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Usage Example:**
```python
from src.benchmark.runner import BenchmarkRunner, RunMode

runner = BenchmarkRunner()

result = await runner.run_benchmark(
    model_name="openai/gpt-4",
    mode=RunMode.STANDARD,
    benchmark_name="My Benchmark"
)

if result.success:
    print(f"Benchmark completed in {result.execution_time:.2f}s")
    print(f"Accuracy: {result.metrics.accuracy.overall_accuracy:.1%}")
else:
    print(f"Benchmark failed: {result.error_message}")
```

## Reporting

### ReportGenerator

Generates reports in various formats.

```python
class ReportGenerator:
    def __init__(self)

    def generate_report(self, result: BenchmarkResult, format: ReportFormat) -> str:
        """Generate a report in specified format."""

    def generate_comparison_report(self,
                                 results: List[BenchmarkResult],
                                 format: ReportFormat,
                                 output_path: Optional[Path] = None) -> str:
        """Generate comparison report for multiple results."""

    def display_terminal_report(self, result: BenchmarkResult) -> None:
        """Display report in terminal."""

    def display_comparison_report(self, results: List[BenchmarkResult]) -> None:
        """Display comparison in terminal."""
```

### ReportFormat

Supported report formats.

```python
class ReportFormat(str, Enum):
    TERMINAL = "terminal"
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"
```

**Usage Example:**
```python
from src.benchmark.reporting import ReportGenerator, ReportFormat

generator = ReportGenerator()

# Generate markdown report
markdown_report = generator.generate_report(result, ReportFormat.MARKDOWN)

# Save to file
with open("benchmark_report.md", "w") as f:
    f.write(markdown_report)

# Display in terminal
generator.display_terminal_report(result)
```

## Available Models

### OpenAI Models

| Model | Context Length | Pricing (per 1K tokens) |
|-------|----------------|------------------------|
| gpt-3.5-turbo | 4K | $0.0015 / $0.002 |
| gpt-3.5-turbo-16k | 16K | $0.003 / $0.004 |
| gpt-4 | 8K | $0.03 / $0.06 |
| gpt-4-32k | 32K | $0.06 / $0.12 |
| gpt-4-turbo | 128K | $0.01 / $0.03 |

### Anthropic Models

| Model | Context Length | Pricing (per 1K tokens) |
|-------|----------------|------------------------|
| claude-3-haiku | 200K | $0.25 / $1.25 |
| claude-3-sonnet | 200K | $3.00 / $15.00 |
| claude-3-opus | 200K | $15.00 / $75.00 |
| claude-2 | 100K | $8.00 / $24.00 |

### Google Models

| Model | Context Length | Pricing (per 1K tokens) |
|-------|----------------|------------------------|
| gemini-pro | 32K | $0.50 / $1.50 |
| gemini-pro-vision | 16K | $0.50 / $1.50 |
| gemini-ultra | 32K | $10.00 / $30.00 |

### Meta Models

| Model | Context Length | Pricing (per 1K tokens) |
|-------|----------------|------------------------|
| llama-2-7b-chat | 4K | $0.001 / $0.001 |
| llama-2-13b-chat | 4K | $0.001 / $0.001 |
| llama-2-70b-chat | 4K | $0.001 / $0.001 |

### Model Selection Guide

**For Speed:**
- OpenAI: gpt-3.5-turbo
- Anthropic: claude-3-haiku

**For Quality:**
- OpenAI: gpt-4-turbo
- Anthropic: claude-3-opus

**For Cost Efficiency:**
- OpenAI: gpt-3.5-turbo
- Meta: llama-2-13b-chat

## Error Handling

### Exception Classes

```python
class JeopardyBenchException(Exception):
    """Base exception for the benchmarking system."""
    pass

class ModelAPIError(JeopardyBenchException):
    """Exception raised when model API calls fail."""
    pass

class ConfigurationError(JeopardyBenchException):
    """Exception raised for configuration errors."""
    pass

class DatabaseError(JeopardyBenchException):
    """Exception raised for database operations."""
    pass

class ValidationError(JeopardyBenchException):
    """Exception raised for data validation errors."""
    pass
```

### Error Handling Patterns

```python
from src.core.exceptions import JeopardyBenchException, ModelAPIError

try:
    result = await runner.run_benchmark("openai/gpt-4", RunMode.STANDARD)
except ModelAPIError as e:
    print(f"Model API error: {e}")
    # Handle API-specific errors
except JeopardyBenchException as e:
    print(f"Benchmark error: {e}")
    # Handle general benchmarking errors
except Exception as e:
    print(f"Unexpected error: {e}")
    # Handle unexpected errors
```

## Examples

### Basic Benchmark

```python
import asyncio
from src.benchmark.runner import BenchmarkRunner, RunMode

async def basic_benchmark():
    runner = BenchmarkRunner()

    result = await runner.run_benchmark(
        model_name="openai/gpt-3.5-turbo",
        mode=RunMode.QUICK
    )

    print(f"Accuracy: {result.metrics.accuracy.overall_accuracy:.1%}")
    print(f"Cost: ${result.metrics.cost.total_cost:.4f}")

asyncio.run(basic_benchmark())
```

### Model Comparison

```python
import asyncio
from src.benchmark.runner import BenchmarkRunner, RunMode
from src.benchmark.reporting import ReportGenerator, ReportFormat

async def compare_models():
    runner = BenchmarkRunner()
    generator = ReportGenerator()

    models = ["openai/gpt-3.5-turbo", "openai/gpt-4", "anthropic/claude-3-haiku"]
    results = []

    for model in models:
        result = await runner.run_benchmark(model, RunMode.STANDARD)
        results.append(result)

    # Generate comparison report
    comparison = generator.generate_comparison_report(
        results, ReportFormat.MARKDOWN
    )

    with open("model_comparison.md", "w") as f:
        f.write(comparison)

asyncio.run(compare_models())
```

### Custom Configuration

```python
from src.benchmark.runner import BenchmarkRunner, BenchmarkConfig
from src.models.base import ModelConfig

# Custom model configuration
model_config = ModelConfig(
    model_name="openai/gpt-4",
    temperature=0.3,
    max_tokens=200
)

# Custom benchmark configuration
benchmark_config = BenchmarkConfig(
    sample_size=100,
    timeout_seconds=45,
    grading_mode=GradingMode.STRICT
)

runner = BenchmarkRunner()
result = await runner.run_benchmark(
    model_name="openai/gpt-4",
    custom_config=benchmark_config
)
```

### Metrics Analysis

```python
from src.evaluation.metrics import MetricsCalculator

calculator = MetricsCalculator()

# Calculate metrics
metrics = calculator.calculate_metrics(
    graded_responses=graded_list,
    model_responses=model_responses,
    model_name="openai/gpt-4"
)

# Access different metric categories
print(f"Overall Score: {metrics.overall_score:.3f}")
print(f"Accuracy by Category: {metrics.accuracy.by_category}")
print(f"Cost Efficiency: {metrics.cost.cost_efficiency_score:.2f}")
print(f"Response Time (P95): {metrics.performance.p95_response_time:.2f}s")
```

### Configuration Management

```python
from src.core.config import get_config, AppConfig
from pathlib import Path

# Load configuration
config = get_config()

# Modify configuration
config.benchmark.max_concurrent_requests = 10
config.database.echo = True

# Save configuration
config_path = Path("custom_config.yaml")
with open(config_path, 'w') as f:
    # Use YAML library to save
    pass
```

---

For more examples and advanced usage, see the [User Guide](USER_GUIDE.md) and [Technical Documentation](../TECHNICAL_SPEC.md).