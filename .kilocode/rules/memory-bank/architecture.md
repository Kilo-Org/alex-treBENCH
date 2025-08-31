# System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CLI Interface (Click)                   │
├─────────────────────────────────────────────────────────────┤
│                    Benchmark Runner                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ Sampling │  │  Model   │  │  Grader  │  │ Metrics  │     │
│  │  Engine  │  │  Client  │  │  System  │  │   Calc   │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │
├─────────────────────────────────────────────────────────────┤
│                    Data Access Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Question    │  │  Benchmark   │  │ Performance  │       │
│  │  Repository  │  │  Repository  │  │ Repository   │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
├─────────────────────────────────────────────────────────────┤
│              SQLAlchemy ORM / Database Layer                │
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
- **openrouter.py**: Enhanced OpenRouter API client with rate limiting and improved error handling
- **model_registry.py**: Dynamic registry supporting 323+ models with real-time fetching from OpenRouter API
- **model_cache.py**: Intelligent caching system for model metadata with TTL and fallback mechanisms
- **prompt_formatter.py**: Multiple prompt templates (Jeopardy, CoT, Few-shot)
- **response_parser.py**: Response parsing and extraction
- **cost_calculator.py**: Real-time cost tracking

### 3. Evaluation System (`src/evaluation/`)

- **matcher.py**: Fuzzy string matching algorithms
- **grader.py**: Multi-mode grading system (strict/lenient/adaptive/jeopardy)
- **metrics.py**: Comprehensive metrics calculation

### 4. Benchmark Engine (`src/benchmark/`)

- **runner.py**: Main orchestration with async execution
- **scheduler.py**: Concurrent benchmark scheduling
- **reporting.py**: Multi-format report generation

### 5. Storage Layer (`src/storage/`)

- **models.py**: SQLAlchemy ORM models with backward compatibility
- **repositories.py**: Repository pattern implementation
- **cache.py**: Response caching system
- **backup.py**: Database backup utilities
- **state_manager.py**: State management for pause/resume

### 6. CLI Interface (`src/cli/`)

- **commands.py**: Click-based command definitions with enhanced model management commands:
  - `models refresh`: Update model cache from OpenRouter API
  - `models search <pattern>`: Search available models by name/provider
  - `models info <model_id>`: Display detailed model information
  - `models cache`: Manage model cache (status, clear, etc.)
- **formatting.py**: Rich terminal output formatting

### 7. Core Infrastructure (`src/core/`)

- **config.py**: Configuration management with dataclasses
- **database.py**: Database connection and session management
- **exceptions.py**: Custom exception hierarchy
- **session.py**: Benchmark session management with pause/resume

## Database Schema

### Core Tables

1. **questions**: Jeopardy questions cache

   - id, question_text, correct_answer, category, value, difficulty_level, air_date, show_number, round

2. **benchmark_runs**: Benchmark execution records (formerly "benchmarks")

   - id, name, status, models_tested, created_at, completed_at, benchmark_mode, sample_size
   - config_snapshot, environment, total_cost_usd, total_tokens, avg_response_time_ms

3. **benchmark_results**: Individual question results (formerly "model_responses")

   - id, benchmark_run_id, question_id, model_name, response_text, is_correct
   - confidence_score, response_time_ms, tokens_generated, cost_usd

4. **model_performance**: Aggregated performance metrics (formerly "model_performance_summary")
   - id, benchmark_run_id, model_name, accuracy_rate, avg_response_time_ms, total_cost_usd
   - category_performance, difficulty_performance, confidence_accuracy_correlation

### Backward Compatibility

The system maintains backward compatibility through aliases in `src/storage/models.py`:

- `Benchmark = BenchmarkRun`
- `BenchmarkQuestion = Question`
- `ModelResponse = BenchmarkResult`
- `ModelPerformanceSummary = ModelPerformance`

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

- Multiple grading strategies (strict, lenient, adaptive, jeopardy)
- Configurable prompt formatting templates
- Flexible sampling algorithms

### Async/Await Pattern

- Concurrent model querying with asyncio
- Rate limiting and throttling
- Efficient batch processing

### Session Management Pattern

- Pause/resume capability for long-running benchmarks
- State persistence and recovery
- Signal handling for graceful shutdown

### Dynamic Model System Pattern

- Three-tier fallback system: OpenRouter API → Local Cache → Static Backup
- Real-time model fetching with intelligent caching (24-hour TTL)
- Graceful degradation when API is unavailable
- Cache invalidation and refresh mechanisms

## External Integrations

### OpenRouter API

- Unified access to 323+ language models
- Rate limiting: 60 requests/minute default
- Retry logic with exponential backoff
- Cost tracking per request
- Support for streaming responses
- Dynamic model discovery and metadata fetching
- Three-tier fallback system for reliability

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
- Session management settings
- Model cache settings (TTL, storage location)

## Performance Considerations

### Optimization Strategies

- Batch processing for API calls
- Connection pooling for database
- Async execution for I/O operations
- Response caching to avoid duplicate calls
- Efficient sampling algorithms
- Session-based checkpointing
- Model metadata caching with intelligent fallbacks

### Scalability

- Supports concurrent benchmarking of multiple models
- Database can be upgraded to PostgreSQL for production
- Docker deployment for containerization
- Horizontal scaling possible with queue system
- Pause/resume for handling long-running benchmarks
- Dynamic model support scales with OpenRouter's model catalog

## Security Considerations

### API Key Management

- Environment variables for sensitive data
- No hardcoded credentials
- Support for .env files in development

### Data Protection

- Input validation and sanitization
- SQL injection prevention via ORM
- Rate limiting to prevent abuse
- Session state encryption (future enhancement)

## Testing Infrastructure

### Test Suites

- **Unit Tests**: Component-level testing
- **Integration Tests**: End-to-end workflows
- **Smoke Tests**: Quick system verification with simulation mode
- **Test Agents**: Specialized testing for different components
- **Performance Tests**: Load and stress testing
- **Dynamic Model Tests**: Validation of model fetching and caching

### Test Utilities

- Mock data generation
- API simulation mode for testing without costs
- Temporary database creation for isolated tests
- Comprehensive test coverage reporting
- Model cache testing with fallback scenarios
