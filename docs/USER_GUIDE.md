# Jeopardy Benchmarking System - User Guide

## Overview

The Jeopardy Benchmarking System is a comprehensive tool for evaluating and comparing language models using Jeopardy-style questions. It provides automated benchmarking, detailed analytics, and flexible reporting to help you understand model performance across different categories and difficulty levels.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [CLI Reference](#cli-reference)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Quick Start

### Prerequisites

- Python 3.8+
- SQLite (default) or PostgreSQL
- OpenRouter API key (for model access)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd alex-trebench

# Install using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .

# Optional: Install development dependencies
uv pip install -r requirements-dev.txt

# Initialize the system
alex init
```

### First Benchmark

```bash
# Run a quick benchmark with GPT-3.5-turbo
alex benchmark run --model openai/gpt-3.5-turbo --size quick

# View results
alex benchmark list
```

## Installation

### System Requirements

- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 1GB free space for database and logs
- **Network**: Internet connection for API access

### Dependencies

The system requires the following Python packages:

```
click>=8.0.0
rich>=10.0.0
sqlalchemy>=1.4.0
pandas>=1.3.0
pytest>=6.0.0
httpx>=0.20.0
pydantic>=1.8.0
```

### Installation Steps

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd alex-trebench
   ```

2. **Create virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the package**:

   ```bash
   # Using uv (recommended)
   uv pip install -e .
   
   # Or using pip
   pip install -e .
   ```

4. **Set up environment variables**:

   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

5. **Initialize the database**:

   ```bash
   alex init
   ```

6. **Verify installation**:
   ```bash
   alex --help
   ```

## Configuration

### Configuration Files

The system uses YAML configuration files located in the `config/` directory:

- `config/default.yaml` - Main configuration
- `config/models/` - Model-specific settings

### Environment Variables

Create a `.env` file in the project root:

```bash
# API Keys
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Database
DATABASE_URL=sqlite:///alex_trebench.db

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/benchmark.log

# Performance
MAX_CONCURRENT_REQUESTS=5
REQUEST_TIMEOUT=60
```

### Model Configuration

Configure available models in `config/models/`:

```yaml
# config/models/openai.yaml
models:
  gpt-3.5-turbo:
    name: "GPT-3.5 Turbo"
    provider: "openai"
    context_length: 4096
    pricing:
      input: 0.0015
      output: 0.002

  gpt-4:
    name: "GPT-4"
    provider: "openai"
    context_length: 8192
    pricing:
      input: 0.03
      output: 0.06
```

## Basic Usage

### Running Benchmarks

#### Single Model Benchmark

```bash
# Quick benchmark (50 questions)
alex benchmark run --model openai/gpt-3.5-turbo --size quick

# Standard benchmark (200 questions)
alex benchmark run --model openai/gpt-4 --size standard

# Comprehensive benchmark (1000 questions)
alex benchmark run --model anthropic/claude-3-opus --size comprehensive
```

#### Custom Benchmark

```bash
# Custom sample size and settings
alex benchmark run \
  --model openai/gpt-4 \
  --size custom \
  --sample-size 500 \
  --timeout 120 \
  --grading-mode lenient \
  --name "Custom GPT-4 Benchmark"
```

### Comparing Models

```bash
# Compare two models
alex benchmark compare \
  --models openai/gpt-3.5-turbo,openai/gpt-4 \
  --size standard

# Compare multiple models with concurrency control
alex benchmark compare \
  --models openai/gpt-3.5-turbo,openai/gpt-4,anthropic/claude-3-haiku \
  --size quick \
  --concurrent-limit 3
```

### Viewing Results

```bash
# List recent benchmarks
alex benchmark list

# View benchmark history for a specific model
alex benchmark history --model openai/gpt-4

# Generate detailed report
alex benchmark report --run-id 1 --format markdown
```

## Advanced Features

### Data Management

#### Initialize Dataset

```bash
# Initialize with default Jeopardy dataset
alex data init

# Initialize with custom dataset
alex data init --source path/to/custom/dataset.csv
```

#### Sample Questions

```bash
# Sample questions by category
alex data sample \
  --category "SCIENCE" \
  --size 100 \
  --output science_sample.json

# Sample by difficulty and value range
alex data sample \
  --difficulty "Hard" \
  --min-value 1000 \
  --max-value 2000 \
  --size 50
```

### Model Management

#### List Available Models

```bash
# List all configured models
alex models list

# List models by provider
alex models list --provider openai
```

#### Test Model Connectivity

```bash
# Test a specific model
alex models test --model openai/gpt-3.5-turbo --prompt "Hello, world!"

# Estimate costs
alex models costs --model openai/gpt-4 --questions 1000
```

### Report Formats

#### Terminal Output

```bash
alex benchmark run --model openai/gpt-4 --report-format terminal
```

#### Markdown Report

```bash
alex benchmark run \
  --model openai/gpt-4 \
  --report-format markdown \
  --output benchmark_report.md
```

#### JSON Export

```bash
alex benchmark run \
  --model openai/gpt-4 \
  --report-format json \
  --output benchmark_results.json
```

### Configuration Management

#### View Current Configuration

```bash
alex config show
```

#### Use Custom Config File

```bash
alex --config path/to/custom/config.yaml benchmark run --model openai/gpt-4
```

## CLI Reference

### Global Options

- `--config, -c`: Path to configuration file
- `--verbose, -v`: Enable verbose logging
- `--debug`: Enable debug mode
- `--help`: Show help message

### Benchmark Commands

#### `benchmark run`

Run a benchmark for a specific model.

```bash
alex benchmark run [OPTIONS] --model MODEL

Options:
  --size, -s: Benchmark size (quick/standard/comprehensive)
  --name, -n: Custom benchmark name
  --description, -d: Benchmark description
  --timeout: Timeout in seconds per question
  --grading-mode: Grading mode (strict/lenient/jeopardy/adaptive)
  --save-results/--no-save-results: Save results to database
  --report-format: Report output format (terminal/markdown/json)
  --output, -o: Save report to file
```

#### `benchmark compare`

Compare multiple models.

```bash
alex benchmark compare [OPTIONS] --models MODELS

Options:
  --size, -s: Benchmark size for all models
  --concurrent-limit: Maximum concurrent model benchmarks
  --report-format: Report output format
  --output, -o: Save comparison report to file
```

#### `benchmark history`

Show benchmark history for a model.

```bash
alex benchmark history [OPTIONS] --model MODEL

Options:
  --limit, -l: Number of recent benchmarks to show
  --detailed, -d: Show detailed information
```

#### `benchmark report`

Generate a report for a benchmark run.

```bash
alex benchmark report [OPTIONS] --run-id RUN_ID

Options:
  --format, -f: Report format (terminal/markdown/json/html)
  --output, -o: Save report to file
  --detailed, -d: Include detailed metrics
```

#### `benchmark list`

List recent benchmarks.

```bash
alex benchmark list [OPTIONS]

Options:
  --limit, -l: Number of benchmarks to show
  --status: Filter by status (pending/running/completed/failed)
  --model, -m: Filter by model name
```

### Data Commands

#### `data init`

Initialize the Jeopardy dataset.

```bash
alex data init [OPTIONS]

Options:
  --force: Force re-initialization
  --strict: Enable strict validation
```

#### `data stats`

Show dataset statistics.

```bash
alex data stats [OPTIONS]

Options:
  --benchmark-id, -b: Show stats for specific benchmark
  --detailed, -d: Include detailed breakdown
```

#### `data sample`

Sample questions from the dataset.

```bash
alex data sample [OPTIONS]

Options:
  --size, -s: Number of questions to sample
  --category, -c: Filter by category
  --difficulty, -d: Filter by difficulty level
  --min-value: Minimum question value
  --max-value: Maximum question value
  --method: Sampling method (random/stratified)
  --seed: Random seed for reproducibility
  --output, -o: Save sample to file
```

### Model Commands

#### `models list`

List available models.

```bash
alex models list [OPTIONS]

Options:
  --provider, -p: Filter by provider
```

#### `models test`

Test a specific model.

```bash
alex models test [OPTIONS] --model MODEL

Options:
  --prompt, -p: Test prompt to send
```

#### `models costs`

Estimate costs for running benchmarks.

```bash
alex models costs [OPTIONS] --model MODEL

Options:
  --questions, -q: Number of questions
  --input-tokens: Estimated input tokens per question
  --output-tokens: Estimated output tokens per question
```

### Utility Commands

#### `init`

Initialize the database and create tables.

```bash
alex init [OPTIONS]

Options:
  --force: Force re-initialization
```

#### `config show`

Show current configuration.

```bash
alex config show [OPTIONS]

Options:
  --format, -f: Output format (yaml/json)
```

#### `health`

Check system health.

```bash
alex health [OPTIONS]

Options:
  --check-db: Check database connectivity
  --check-api: Check API connectivity
```

## Troubleshooting

### Common Issues

#### API Connection Problems

**Problem**: Unable to connect to OpenRouter API

**Solutions**:

1. Verify API key is set in `.env` file
2. Check internet connection
3. Verify API key has sufficient credits
4. Check API rate limits

```bash
# Test API connectivity
alex health --check-api
```

#### Database Issues

**Problem**: Database connection errors

**Solutions**:

1. Ensure database file exists and is writable
2. Check database URL in configuration
3. Reinitialize database if corrupted

```bash
# Reinitialize database
alex init --force
```

#### Memory Issues

**Problem**: Out of memory errors during large benchmarks

**Solutions**:

1. Reduce concurrent requests
2. Use smaller batch sizes
3. Increase system memory
4. Use `--size quick` for testing

```bash
# Run with reduced concurrency
alex benchmark run --model openai/gpt-4 --concurrent-limit 2
```

#### Model Not Found

**Problem**: Specified model is not available

**Solutions**:

1. Check model name spelling
2. Verify model is supported by OpenRouter
3. Update model configuration

```bash
# List available models
alex models list
```

### Performance Optimization

#### Slow Benchmark Execution

1. **Reduce sample size**:

   ```bash
   alex benchmark run --model openai/gpt-4 --size quick
   ```

2. **Increase concurrency** (if API allows):

   ```bash
   alex benchmark run --model openai/gpt-4 --concurrent-limit 10
   ```

3. **Use faster models for testing**:
   ```bash
   alex benchmark run --model openai/gpt-3.5-turbo
   ```

#### High Memory Usage

1. **Process in batches**:

   ```bash
   alex benchmark run --model openai/gpt-4 --batch-size 10
   ```

2. **Disable result saving for testing**:
   ```bash
   alex benchmark run --model openai/gpt-4 --no-save-results
   ```

### Logging and Debugging

#### Enable Debug Logging

```bash
# Enable debug mode
alex --debug benchmark run --model openai/gpt-4

# Enable verbose logging
alex --verbose benchmark run --model openai/gpt-4
```

#### View Logs

```bash
# Check log files
tail -f logs/benchmark.log

# View recent errors
grep "ERROR" logs/benchmark.log | tail -10
```

### Getting Help

1. **Check command help**:

   ```bash
   alex --help
   alex benchmark run --help
   ```

2. **View system health**:

   ```bash
   alex health
   ```

3. **Check configuration**:
   ```bash
   alex config show
   ```

## Best Practices

### Benchmark Design

1. **Start Small**: Begin with quick benchmarks to verify setup
2. **Use Appropriate Sample Sizes**: Match sample size to your needs and resources
3. **Consider Categories**: Focus on relevant categories for your use case
4. **Account for Costs**: Estimate costs before running large benchmarks

### Model Selection

1. **Test Multiple Models**: Compare several models for comprehensive evaluation
2. **Consider Use Case**: Choose models based on your specific requirements
3. **Monitor Performance**: Track model performance over time
4. **Update Regularly**: Test with latest model versions

### Data Management

1. **Regular Updates**: Keep dataset current with latest questions
2. **Data Quality**: Ensure question quality and accuracy
3. **Backup Data**: Regularly backup database and results
4. **Monitor Storage**: Track database size and growth

### Performance Optimization

1. **Right-size Resources**: Match hardware to benchmark requirements
2. **Batch Processing**: Use appropriate batch sizes for your system
3. **Concurrent Limits**: Set concurrency based on API limits and system capacity
4. **Monitor Usage**: Track memory, CPU, and API usage during benchmarks

### Result Analysis

1. **Multiple Metrics**: Consider accuracy, speed, and cost together
2. **Statistical Significance**: Use appropriate sample sizes for reliable results
3. **Context Matters**: Consider question categories and difficulty levels
4. **Trend Analysis**: Track performance changes over time

### Maintenance

1. **Regular Testing**: Run periodic benchmarks to monitor model performance
2. **Update Dependencies**: Keep Python packages and system updated
3. **Monitor Logs**: Regularly review logs for issues
4. **Backup Strategy**: Maintain regular backups of database and configuration

---

For more advanced usage and development information, see the [API Reference](API_REFERENCE.md) and [Technical Documentation](../TECHNICAL_SPEC.md).
