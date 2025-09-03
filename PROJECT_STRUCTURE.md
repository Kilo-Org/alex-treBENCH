# Project Structure: Language Model Benchmarking System

## 1. Directory Organization

```
alex-trebench/
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── setup.py
├── INSTALLATION.md
├── PROJECT_STRUCTURE.md
├── TECHNICAL_SPEC.md
├── config/
│   ├── default.yaml                # Main configuration
│   └── models/                     # Model-specific configurations
│       ├── anthropic.yaml
│       ├── openai.yaml
│       └── opensource.yaml
├── src/
│   ├── __init__.py
│   ├── main.py                     # CLI entry point (alex command)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py               # Configuration management
│   │   ├── config_manager.py       # Enhanced config management
│   │   ├── database.py             # Database connection and schema
│   │   ├── exceptions.py           # Custom exceptions
│   │   └── session.py              # Session management
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestion.py            # Kaggle dataset loading
│   │   ├── preprocessing.py        # Data cleaning and normalization
│   │   ├── sampling.py             # Statistical sampling algorithms
│   │   └── validation.py           # Data validation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract model interface
│   │   ├── openrouter.py           # OpenRouter API client
│   │   ├── model_registry.py       # Dynamic model registry
│   │   ├── model_cache.py          # Model metadata caching
│   │   ├── cost_calculator.py      # Cost tracking
│   │   ├── prompt_formatter.py     # Prompt templates
│   │   ├── response_parser.py      # Response parsing
│   │   └── adapters/               # Future model adapters
│   │       └── __init__.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── matcher.py              # Fuzzy answer matching
│   │   ├── grader.py               # Answer grading logic
│   │   └── metrics.py              # Performance metrics calculation
│   ├── benchmark/
│   │   ├── __init__.py
│   │   ├── reporting.py            # Results analysis and reporting
│   │   ├── scheduler.py            # Task scheduling and queue management
│   │   └── runner/                 # Benchmark execution engine
│   │       ├── __init__.py
│   │       ├── core.py
│   │       ├── config.py
│   │       └── types.py
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── backup.py               # Database backup utilities
│   │   ├── cache.py                # Response caching
│   │   ├── state_manager.py        # State management for pause/resume
│   │   ├── migrations.py           # Migration management
│   │   ├── models/                 # SQLAlchemy ORM models
│   │   │   ├── __init__.py
│   │   │   ├── benchmark_result.py
│   │   │   ├── benchmark_run.py
│   │   │   ├── mixins.py
│   │   │   ├── model_performance.py
│   │   │   └── question.py
│   │   └── repositories/           # Data access layer
│   │       ├── __init__.py
│   │       ├── benchmark_repository.py
│   │       ├── performance_repository.py
│   │       ├── question_repository.py
│   │       └── response_repository.py
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── commands.py             # CLI command definitions
│   │   └── formatting.py           # Output formatting utilities
│   ├── commands/                   # Command implementations
│   │   ├── __init__.py
│   │   ├── health.py
│   │   ├── models.py
│   │   ├── session.py
│   │   ├── benchmarks/
│   │   │   ├── __init__.py
│   │   │   ├── compare.py
│   │   │   ├── export.py
│   │   │   ├── history.py
│   │   │   ├── leaderboard.py
│   │   │   ├── list.py
│   │   │   ├── report.py
│   │   │   ├── run.py
│   │   │   └── status.py
│   │   ├── config/
│   │   │   ├── __init__.py
│   │   │   └── settings.py
│   │   └── data/
│   │       ├── __init__.py
│   │       ├── init.py
│   │       ├── sample.py
│   │       ├── stats.py
│   │       └── validate.py
│   ├── api/                        # Future web API
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   └── __init__.py
│   │   └── middleware/
│   │       └── __init__.py
│   ├── scripts/                    # Initialization and utility scripts
│   │   ├── __init__.py
│   │   └── init_data.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py              # Logging configuration
│       ├── async_helpers.py        # Async utility functions
│       └── validation.py           # Input validation utilities
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Pytest configuration
│   ├── unit/                       # Unit tests
│   │   ├── __init__.py
│   │   ├── test_data/
│   │   ├── test_evaluation/
│   │   ├── test_models/
│   │   ├── test_storage/
│   │   ├── test_config/
│   │   └── test_benchmarks/
│   ├── integration/                # Integration tests
│   │   ├── __init__.py
│   │   ├── test_benchmark_flow.py
│   │   ├── test_data_pipeline.py
│   │   ├── test_models_integration.py
│   │   └── test_persistence.py
│   ├── e2e/                        # End-to-end tests
│   │   ├── test_cli_commands.py
│   │   └── test_complete_workflow.py
│   ├── performance/                # Performance tests
│   │   └── test_benchmark_performance.py
│   └── fixtures/                   # Test fixtures
│       ├── __init__.py
│       ├── sample_questions.json
│       └── mock_responses.json
├── scripts/                        # Utility scripts
│   ├── demo.py                     # Interactive demo
│   ├── quick_test.sh              # Quick verification
│   ├── smoke_test.py              # Smoke testing
│   └── test_agents.py             # Component testing
├── examples/                       # Usage examples
│   ├── quick_start.sh             # Getting started script
│   ├── sample_benchmark.py        # Example benchmarks
│   ├── sample_config.yaml         # Example configuration
│   └── sample_output.md           # Example output
├── docs/                          # Documentation
│   ├── USER_GUIDE.md              # Complete user guide
│   ├── API_REFERENCE.md           # API documentation
│   └── TESTING.md                 # Testing documentation
├── data/                          # Local data directory
│   ├── cache/                     # Model metadata cache
│   │   └── models.json
│   ├── raw/                       # Raw Kaggle dataset
│   ├── processed/                 # Processed question samples
│   └── exports/                   # Exported benchmark results
├── logs/                          # Application logs
└── database/                      # SQLite database files
    └── benchmarks.db
```

## 2. Key Modules and Responsibilities

### 2.1 Core Modules

#### `src/core/`

- **Purpose**: Foundational components used across the application
- **Key Components**:
  - `config.py`: Centralized configuration management
  - `database.py`: Database connection and session management
  - `exceptions.py`: Application-specific exception classes

#### `src/data/`

- **Purpose**: Data ingestion, preprocessing, and sampling
- **Key Components**:
  - `ingestion.py`: Kaggle dataset loading using kagglehub
  - `preprocessing.py`: Data cleaning, normalization, and validation
  - `sampling.py`: Statistical sampling algorithms for question selection

#### `src/models/`

- **Purpose**: Language model abstraction and API clients
- **Key Components**:
  - `base.py`: Abstract base class for model interfaces
  - `openrouter.py`: OpenRouter API client implementation
  - `adapters/`: Plugin directory for future model providers

### 2.2 Processing Modules

#### `src/evaluation/`

- **Purpose**: Answer evaluation and performance measurement
- **Key Components**:
  - `matcher.py`: Fuzzy string matching for answer validation
  - `grader.py`: Answer grading logic and confidence scoring
  - `metrics.py`: Performance metrics calculation and aggregation

#### `src/benchmarks/`

- **Purpose**: Benchmark execution and management
- **Key Components**:
  - `runner.py`: Main benchmark execution engine
  - `scheduler.py`: Async task scheduling and queue management
  - `reporting.py`: Results analysis, visualization, and report generation

### 2.3 Data Layer

#### `src/storage/`

- **Purpose**: Data persistence and access
- **Key Components**:
  - `models.py`: SQLAlchemy ORM model definitions
  - `repositories.py`: Repository pattern for data access
  - `migrations/`: Database schema migration scripts

### 2.4 Interface Modules

#### `src/cli/`

- **Purpose**: Command-line interface
- **Key Components**:
  - `commands.py`: Click-based CLI command definitions
  - `formatting.py`: Output formatting and progress display

#### `src/api/` (Future)

- **Purpose**: Web API for future GUI integration
- **Key Components**:
  - `app.py`: FastAPI application setup
  - `routes/`: API endpoint definitions
  - `middleware/`: Authentication, logging, CORS

## 3. Technology Stack

### 3.1 Core Dependencies

#### Data Processing

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations for statistical sampling
- **kagglehub**: Kaggle dataset integration

#### Database & ORM

- **SQLAlchemy**: Database ORM and connection management
- **alembic**: Database migration management
- **sqlite3**: Built-in SQLite support (Python standard library)

#### API Integration

- **aiohttp**: Async HTTP client for OpenRouter API
- **asyncio**: Asynchronous programming support
- **tenacity**: Retry mechanisms and error handling

#### Text Processing & Evaluation

- **fuzzywuzzy**: Fuzzy string matching for answer evaluation
- **python-Levenshtein**: Fast string similarity calculations
- **nltk**: Natural language processing utilities
- **regex**: Advanced regular expression support

#### CLI & Interface

- **click**: Command-line interface framework
- **rich**: Enhanced terminal output and progress bars
- **tabulate**: Table formatting for results display

#### Configuration & Utilities

- **pyyaml**: YAML configuration file support
- **python-dotenv**: Environment variable management
- **pydantic**: Data validation and settings management

### 3.2 Development Dependencies

#### Testing

- **pytest**: Testing framework
- **pytest-asyncio**: Async testing support
- **pytest-cov**: Code coverage reporting
- **pytest-mock**: Mocking utilities
- **factory-boy**: Test data generation

#### Code Quality

- **black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting and style checking
- **mypy**: Type checking
- **pre-commit**: Git pre-commit hooks

#### Documentation

- **sphinx**: Documentation generation
- **mkdocs**: Markdown-based documentation
- **mkdocs-material**: Material theme for documentation

### 3.3 Future/Optional Dependencies

#### Web Interface (Phase 2)

- **fastapi**: Modern web API framework
- **uvicorn**: ASGI server
- **websockets**: Real-time benchmark progress updates
- **jinja2**: Template rendering

#### Advanced Analytics

- **matplotlib**: Data visualization
- **plotly**: Interactive charts
- **jupyter**: Notebook-based analysis
- **scikit-learn**: Statistical analysis utilities

#### Production Deployment

- **docker**: Containerization
- **redis**: Caching and task queuing
- **postgresql**: Production database (if scaling beyond SQLite)
- **prometheus**: Metrics collection

## 4. Configuration Management

### 4.1 Configuration Structure

```yaml
# config/default.yaml
app:
  name: "Jeopardy Benchmark"
  version: "1.0.0"
  debug: false

database:
  url: "sqlite:///database/benchmarks.db"
  echo: false
  pool_size: 5

openrouter:
  base_url: "https://openrouter.ai/api/v1"
  timeout: 30
  max_retries: 3
  rate_limit:
    requests_per_minute: 60

benchmarks:
  default_sample_size: 1000
  confidence_level: 0.95
  margin_of_error: 0.05
  answer_similarity_threshold: 0.7
  max_concurrent_requests: 5

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/benchmark.log"
  max_size: "10MB"
  backup_count: 5

kaggle:
  dataset: "aravindram11/jeopardy-dataset-updated"
  cache_dir: "data/raw"
```

### 4.2 Environment Variables

```bash
# .env.example
OPENROUTER_API_KEY=your_api_key_here
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key

# Optional overrides
DATABASE_URL=sqlite:///database/benchmarks.db
LOG_LEVEL=INFO
DEBUG=False
```

## 5. Installation and Setup

### 5.1 Requirements File

```txt
# requirements.txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
kagglehub>=0.2.0
SQLAlchemy>=2.0.0
alembic>=1.12.0

# API and async
aiohttp>=3.8.0
asyncio-throttle>=1.0.2
tenacity>=8.2.0

# Text processing
fuzzywuzzy>=0.18.0
python-Levenshtein>=0.21.0
nltk>=3.8.0

# CLI and output
click>=8.1.0
rich>=13.0.0
tabulate>=0.9.0

# Configuration
PyYAML>=6.0.0
python-dotenv>=1.0.0
pydantic>=2.0.0

# Development dependencies (dev requirements)
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0
black>=23.0.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.5.0
pre-commit>=3.4.0
```

### 5.2 Setup Script

```python
# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="alex-trebench",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Language Model Benchmarking System using Jeopardy Questions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/alex-trebench",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "alex-trebench=main:cli",
        ],
    },
)
```

## 6. Module Interaction Patterns

### 6.1 Dependency Injection Pattern

```python
# Dependency injection for testability
class BenchmarkRunner:
    def __init__(
        self,
        data_loader: DataLoader,
        model_client: ModelClient,
        evaluator: AnswerEvaluator,
        storage: ResultsStorage
    ):
        self.data_loader = data_loader
        self.model_client = model_client
        self.evaluator = evaluator
        self.storage = storage
```

### 6.2 Repository Pattern

```python
# Data access abstraction
class BenchmarkRepository:
    def __init__(self, session: Session):
        self.session = session

    def create_benchmark(self, benchmark: Benchmark) -> Benchmark:
        # Implementation
        pass

    def get_benchmark(self, benchmark_id: int) -> Optional[Benchmark]:
        # Implementation
        pass
```

### 6.3 Plugin Architecture

```python
# Extensible model adapter system
class ModelAdapter(ABC):
    @abstractmethod
    async def query(self, prompt: str) -> ModelResponse:
        pass

class OpenRouterAdapter(ModelAdapter):
    # Implementation
    pass
```

## 7. Testing Strategy Integration

### 7.1 Test Structure

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete benchmark workflows

### 7.2 Mock Strategy

- Mock external APIs (OpenRouter, Kaggle) for consistent testing
- Use factory-boy for generating test data
- Pytest fixtures for common test setup

### 7.3 Coverage Goals

- Target 90%+ code coverage
- Focus on critical path coverage (benchmark execution, answer evaluation)
- Include edge case testing for fuzzy matching algorithms

This project structure provides a solid foundation for the benchmarking system while maintaining flexibility for future enhancements and extensions.
