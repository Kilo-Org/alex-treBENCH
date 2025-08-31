# Technologies & Development Setup

## Technology Stack

### Core Language

- **Python 3.8+**: Primary development language
- **Type Hints**: Full type annotation support
- **Async/Await**: Asynchronous programming for concurrent operations

### Web Framework & API

- **Click**: Command-line interface framework
- **Rich**: Enhanced terminal output with tables and progress bars
- **FastAPI** (optional): REST API for web interface
- **aiohttp**: Async HTTP client for API calls

### Database & ORM

- **SQLAlchemy 2.0+**: Object-Relational Mapping
- **SQLite**: Default database (development)
- **PostgreSQL**: Production database option
- **Alembic**: Database migration management

### Data Processing

- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **kagglehub**: Kaggle dataset integration
- **scikit-learn**: Statistical sampling algorithms

### Text Processing

- **FuzzyWuzzy**: Fuzzy string matching
- **python-Levenshtein**: Fast string similarity
- **NLTK**: Natural language processing utilities

### Testing

- **pytest**: Testing framework
- **pytest-asyncio**: Async test support
- **pytest-cov**: Code coverage reporting
- **pytest-mock**: Mocking utilities
- **factory-boy**: Test data generation

### Code Quality

- **black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Static type checking
- **pre-commit**: Git hooks for code quality

### Configuration

- **PyYAML**: YAML configuration files
- **python-dotenv**: Environment variable management
- **Pydantic**: Data validation and settings

### Containerization

- **Docker**: Container platform
- **docker-compose**: Multi-container orchestration

## Development Setup

### Prerequisites

```bash
# System requirements
Python 3.8 or higher
Git
SQLite3
Docker (optional)
```

### Installation Steps

```bash
# Clone repository
git clone <repository-url>
cd jeopardyBench

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Initialize database
python -m src.main init

# Initialize dataset
python -m src.scripts.init_data
```

### Configuration Files

#### `.env` Environment Variables

```
OPENROUTER_API_KEY=your_key_here
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_kaggle_key
DATABASE_URL=sqlite:///database/benchmarks.db
LOG_LEVEL=INFO
DEBUG=False
```

#### `config/default.yaml` Main Configuration

- Database settings
- OpenRouter API configuration
- Benchmark parameters
- Logging configuration
- Model-specific overrides

### Development Commands

#### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/ --cov-report=html

# Specific test types
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/e2e/ -v
```

#### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# All quality checks
make quality
```

#### Docker Operations

```bash
# Build image
docker build -t alex-trebench .

# Run with docker-compose
docker-compose up

# Development mode
docker-compose --profile dev up

# Run tests in Docker
docker-compose --profile test run --rm test
```

### Makefile Commands

```bash
make install        # Install dependencies
make install-dev    # Install dev dependencies
make test          # Run tests
make test-coverage # Run tests with coverage
make lint          # Run linters
make format        # Format code
make docker-build  # Build Docker image
make docker-run    # Run in Docker
make demo          # Run interactive demo
```

## API Integration

### OpenRouter API

- **Endpoint**: https://openrouter.ai/api/v1
- **Authentication**: Bearer token
- **Rate Limits**: 60 requests/minute (configurable)
- **Supported Models**: 20+ including GPT-4, Claude, Llama

### Kaggle Dataset API

- **Dataset**: aravindram11/jeopardy-dataset-updated
- **Format**: JSON/CSV
- **Size**: ~200,000 questions
- **Caching**: Local cache after first download

## Performance Optimizations

### Async Processing

- Concurrent model queries
- Batch processing for efficiency
- Async database operations

### Caching Strategy

- Response caching to avoid duplicates
- Dataset caching after download
- Configuration caching

### Database Optimizations

- Connection pooling
- Indexed queries
- Batch inserts/updates

## Monitoring & Logging

### Logging Configuration

- **Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Output**: Console and file
- **Rotation**: 10MB max size, 5 backups
- **Format**: Timestamp, module, level, message

### Metrics Tracking

- Response times
- API costs
- Error rates
- Success rates
- Token usage

## Deployment Considerations

### Development

- SQLite database
- Local file storage
- Debug logging enabled
- Mock data available

### Production

- PostgreSQL database
- Environment-based config
- Error tracking (Sentry optional)
- Performance monitoring
- Backup strategies
