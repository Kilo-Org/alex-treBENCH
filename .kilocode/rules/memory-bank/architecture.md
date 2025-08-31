# System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CLI Interface (Click)                    │
├─────────────────────────────────────────────────────────────┤
│                    Benchmark Runner                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Sampling │  │  Model   │  │  Grader  │  │ Metrics  │  │
│  │  Engine  │  │  Client  │  │  System  │  │   Calc   │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    Data Access Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Question    │  │  Benchmark   │  │ Performance  │     │
│  │  Repository  │  │  Repository  │  │ Repository   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
├─────────────────────────────────────────────────────────────┤
│              SQLAlchemy ORM / Database Layer                 │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Data Layer (`src/data/`)

- **ingestion.py**: Kaggle dataset loader using kagglehub
- **preprocessing.py**: Data cleaning and normalization
- **sampling.py**: Statistical sampling algorithms
- **validation.py**: Data quality validation

### 2. Model Integration (`src/models/`)

- **base.py**: Abstract ModelAdapter interface
- **openrouter.py**: OpenRouter API client with rate limiting
- **model_registry.py**: Registry of 20+ supported models
- **prompt_formatter.py**: Multiple prompt templates (Jeopardy, CoT, Few-shot)
- **response_parser.py**: Response parsing and extraction
- **cost_calculator.py**: Real-time cost tracking

### 3. Evaluation System (`src/evaluation/`)

- **matcher.py**: Fuzzy string matching algorithms
- **grader.py**: Multi-mode grading system (strict/lenient/adaptive)
- **metrics.py**: Comprehensive metrics calculation

### 4. Benchmark Engine (`src/benchmark/`)

- **runner.py**: Main orchestration with async execution
- **scheduler.py**: Concurrent benchmark scheduling
- **reporting.py**: Multi-format report generation

### 5. Storage Layer (`src/storage/`)

- **models.py**: SQLAlchemy ORM models
- **repositories.py**: Repository pattern implementation
- **cache.py**: Response caching system
- **backup.py**: Database backup utilities

### 6. CLI Interface (`src/cli/`)

- **commands.py**: Click-based command definitions
- **formatting.py**: Rich terminal output formatting

## Database Schema

### Core Tables

1. **questions**: Jeopardy questions cache

   - id, question_text, correct_answer, category, value, difficulty_level

2. **benchmark_runs**: Benchmark execution records

   - id, name, status, models_tested, created_at, completed_at

3. **benchmark_results**: Individual question results

   - id, benchmark_run_id, question_id, model_name, response_text, is_correct

4. **model_performance**: Aggregated performance metrics
   - id, benchmark_run_id, model_name, accuracy_rate, avg_response_time, total_cost

## Key Design Patterns

### Repository Pattern

- Clean separation between business logic and data access
- All database operations through repository classes
- Testable and maintainable data layer

### Adapter Pattern

- ModelAdapter base class for different LLM providers
- Extensible model integration
- Consistent interface across different APIs

### Strategy Pattern

- Multiple grading strategies (strict, lenient, adaptive)
- Configurable prompt formatting templates
- Flexible sampling algorithms

### Async/Await Pattern

- Concurrent model querying with asyncio
- Rate limiting and throttling
- Efficient batch processing

## External Integrations

### OpenRouter API

- Unified access to 20+ language models
- Rate limiting: 60 requests/minute default
- Retry logic with exponential backoff
- Cost tracking per request

### Kaggle Dataset

- Dataset: "aravindram11/jeopardy-dataset-updated"
- Cached locally after first download
- Supports JSON and CSV formats
- ~200,000+ questions available

## Configuration System

### Configuration Hierarchy

1. Default configuration (`config/default.yaml`)
2. Environment-specific overrides
3. Environment variables
4. Runtime parameters

### Key Configuration Areas

- Database connection settings
- OpenRouter API configuration
- Benchmark execution parameters
- Grading and evaluation settings
- Logging and monitoring

## Performance Considerations

### Optimization Strategies

- Batch processing for API calls
- Connection pooling for database
- Async execution for I/O operations
- Response caching to avoid duplicate calls
- Efficient sampling algorithms

### Scalability

- Supports concurrent benchmarking of multiple models
- Database can be upgraded to PostgreSQL for production
- Docker deployment for containerization
- Horizontal scaling possible with queue system

## Security Considerations

### API Key Management

- Environment variables for sensitive data
- No hardcoded credentials
- Support for .env files in development

### Data Protection

- Input validation and sanitization
- SQL injection prevention via ORM
- Rate limiting to prevent abuse
