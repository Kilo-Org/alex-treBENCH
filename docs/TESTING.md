# alex-treBENCH Testing Guide

This document provides comprehensive information about testing alex-treBENCH, including automated tests, smoke tests, and verification procedures.

## Testing Overview

alex-treBENCH includes a multi-layered testing infrastructure designed to verify system functionality at different levels:

- **Smoke Tests**: End-to-end verification of the complete system
- **Test Agents**: Focused tests for individual system components
- **Unit Tests**: Component-level testing with mocking
- **Integration Tests**: Cross-component functionality testing
- **Performance Tests**: System performance and scalability validation

## 🔥 Smoke Test

The smoke test provides complete end-to-end verification of alex-treBENCH functionality.

### What It Tests

The smoke test verifies:

- ✅ Database initialization and schema creation
- ✅ Sample data loading
- ✅ API connectivity (OpenRouter integration)
- ✅ Benchmark execution with real/simulated API calls
- ✅ Report generation and data verification
- ✅ System health checks

### Running the Smoke Test

#### Quick Run

```bash
# Run the smoke test directly
python scripts/smoke_test.py

# Or using Make
make smoke-test
```

#### With API Key (Real API Calls)

```bash
# Set up environment
export OPENROUTER_API_KEY="your_api_key_here"

# Run smoke test (will make minimal real API calls)
python scripts/smoke_test.py
```

#### Simulation Mode (No API Key Required)

```bash
# Run without API key (uses simulation)
unset OPENROUTER_API_KEY
python scripts/smoke_test.py
```

### Expected Output

**Successful Run:**

```
🔥 alex-treBENCH Smoke Test
Running complete end-to-end system verification

✅ Setting up test environment...
✅ Initializing database...
✅ Loading sample data...
✅ Running minimal benchmark...
✅ Generating report...
✅ Verifying system health...

🎉 Smoke Test PASSED
alex-treBENCH system is working correctly!
```

**Failed Run:**

```
❌ Smoke test failed: [Error details]
Check the logs above for details

❌ Smoke Test FAILED
alex-treBENCH system has issues that need attention
```

### Cost Considerations

- **With API Key**: ~$0.001-0.005 per run (3 questions with GPT-3.5-Turbo)
- **Simulation Mode**: $0.00 (no API calls)

## 🤖 Test Agents

Test agents provide focused testing of individual system components.

### Available Test Agents

1. **Database Initialization Agent**

   - Tests database schema creation
   - Verifies table structure and connectivity

2. **Data Loading Agent**

   - Tests sample question loading
   - Verifies data persistence and retrieval

3. **Minimal Benchmark Agent**

   - Tests real API calls with minimal data
   - Requires OPENROUTER_API_KEY

4. **Report Generation Agent**

   - Tests report generation from benchmark data
   - Verifies report format and content

5. **CLI Command Agent**
   - Tests CLI interface functionality
   - Verifies command parsing and execution

### Running Test Agents

#### All Test Agents

```bash
# Run all test agents
python scripts/test_agents.py

# Or using Make
make test-agents
```

#### Individual Test Agents

```bash
# Run specific test agent (modify script as needed)
python scripts/test_agents.py --agent database
python scripts/test_agents.py --agent data-loading
python scripts/test_agents.py --agent benchmark
```

### Expected Output

```
🤖 Test Agent Runner
Running comprehensive system component tests

✅ Database Initialization: PASSED (0.2s)
✅ Data Loading: PASSED (0.4s)
✅ Minimal Benchmark: PASSED (2.3s)
✅ Report Generation: PASSED (0.1s)
✅ CLI Command: PASSED (0.3s)

📊 Test Results Summary:
• Passed: 5/5 (100%)
• Total execution time: 3.3s
```

## 🧪 Unit Tests

Unit tests provide component-level verification with mocking.

### Running Unit Tests

```bash
# All unit tests
pytest tests/unit/ -v

# Specific component
pytest tests/unit/test_models/ -v
pytest tests/unit/test_data/ -v
pytest tests/unit/test_evaluation/ -v

# With coverage
pytest tests/unit/ --cov=src/ --cov-report=html

# Using Make
make test-unit
```

### Test Organization

```
tests/unit/
├── test_benchmarks/     # Benchmark execution tests
├── test_config/         # Configuration tests
├── test_data/           # Data processing tests
├── test_evaluation/     # Grading and metrics tests
├── test_models/         # Model adapter tests
└── test_storage/        # Database and storage tests
```

## 🔗 Integration Tests

Integration tests verify cross-component functionality.

### Running Integration Tests

```bash
# All integration tests
pytest tests/integration/ -v

# Specific integration test
pytest tests/integration/test_benchmark_flow.py -v
pytest tests/integration/test_data_pipeline.py -v

# Using Make
make test-integration
```

### What Integration Tests Cover

- **Benchmark Flow**: Complete benchmark execution pipeline
- **Data Pipeline**: Data ingestion to storage flow
- **Model Integration**: API client to response processing
- **Persistence**: Database operations across components

## ⚡ Performance Tests

Performance tests validate system scalability and efficiency.

### Running Performance Tests

```bash
# All performance tests
pytest tests/performance/ -v

# Benchmark performance specifically
pytest tests/performance/test_benchmark_performance.py -v

# Using Make
make test-performance
```

### Performance Metrics Tracked

- Response time under load
- Memory usage during execution
- Database query performance
- API call efficiency

## 🚀 End-to-End Tests

End-to-end tests verify complete user workflows.

### Running E2E Tests

```bash
# All E2E tests
pytest tests/e2e/ -v

# CLI workflow testing
pytest tests/e2e/test_cli_commands.py -v

# Complete workflow testing
pytest tests/e2e/test_complete_workflow.py -v

# Using Make
make test-e2e
```

## 🔧 Quick System Verification

For rapid system verification, use the quick test script:

```bash
# Quick system check
./scripts/quick_test.sh

# Or if not executable
bash scripts/quick_test.sh
```

This script:

1. Checks prerequisites (Python, dependencies)
2. Runs smoke test
3. Reports success/failure clearly
4. Provides troubleshooting hints

## 🎯 Test Categories by Use Case

### Before Committing Code

```bash
make quality          # Lint, type-check, unit tests
make test-integration  # Integration tests
```

### Before Deployment

```bash
make deploy-check     # All tests + validation
make smoke-test       # End-to-end verification
```

### Debugging Issues

```bash
make test-agents      # Component-specific testing
python scripts/validate_system.py  # System validation
```

### Performance Validation

```bash
make test-performance # Performance benchmarks
make simulate-fast    # Fast simulation testing
```

## 🚨 Troubleshooting Common Issues

### Database Issues

**Symptom**: "Database initialization failed"

```bash
# Solutions:
rm data/*.db          # Remove corrupted database
make db-init          # Reinitialize database
make smoke-test       # Verify fix
```

**Symptom**: "SQLite locked" or "Database is locked"

```bash
# Solutions:
pkill -f "python.*jeopardy"  # Kill hanging processes
rm data/*.db-wal data/*.db-shm  # Remove lock files
make db-init                     # Reinitialize
```

### API Connection Issues

**Symptom**: "API call failed" or "OpenRouter" errors

```bash
# Check API key
echo $OPENROUTER_API_KEY

# Test connectivity
curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
     https://openrouter.ai/api/v1/models

# Run in simulation mode
unset OPENROUTER_API_KEY
make smoke-test
```

### Import/Module Issues

**Symptom**: "ModuleNotFoundError" or import errors

```bash
# Reinstall dependencies
make install-dev

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Verify installation
python -c "from src.core.config import get_config; print('OK')"
```

### Permission Issues

**Symptom**: "Permission denied" errors

```bash
# Make scripts executable
chmod +x scripts/*.py
chmod +x scripts/*.sh

# Check file permissions
ls -la scripts/
```

## 💰 Cost Management

### API Call Costs

- **Smoke Test**: ~$0.001-0.005 per run
- **Test Agents (with API)**: ~$0.002-0.01 per run
- **Unit/Integration Tests**: $0.00 (mocked)
- **Performance Tests**: $0.00 (simulated)

### Cost Optimization

```bash
# Use simulation mode for development
export JEOPARDY_SIMULATION_MODE=true

# Run tests without API calls
make test-unit test-integration

# Use cheap models for testing
export JEOPARDY_TEST_MODEL="openai/gpt-3.5-turbo"
```

## 📊 Test Coverage Goals

- **Unit Tests**: >80% code coverage
- **Integration Tests**: All major workflows
- **E2E Tests**: All CLI commands and user paths
- **Performance Tests**: All bottleneck scenarios

### Checking Coverage

```bash
# Generate coverage report
make test-coverage

# View HTML report
open htmlcov/index.html

# Terminal coverage summary
pytest tests/ --cov=src/ --cov-report=term-missing
```

## 🔄 Continuous Testing

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Manual run
pre-commit run --all-files
```

### Automated Testing

Tests automatically run on:

- Git commits (pre-commit hooks)
- Pull requests (GitHub Actions)
- Main branch pushes (CI/CD)

See `.github/workflows/smoke-test.yml` for CI configuration.

## 📝 Writing New Tests

### Test Structure Guidelines

```python
# Unit test example
def test_component_functionality():
    """Test specific component behavior."""
    # Arrange
    setup_test_data()

    # Act
    result = component.process()

    # Assert
    assert result.success
    assert result.data == expected_data
```

### Test Naming Convention

- `test_[component]_[functionality].py` for files
- `test_[action]_[expected_outcome]()` for functions
- Use descriptive names that explain the test purpose

### Mocking Best Practices

```python
from unittest.mock import Mock, patch

@patch('src.models.openrouter.OpenRouterClient')
def test_benchmark_without_api_calls(mock_client):
    """Test benchmark logic without actual API calls."""
    mock_client.return_value.query.return_value = mock_response
    # Test logic here
```

## 🎯 Success Criteria

### Smoke Test Success

- All 6 smoke test steps pass
- No critical errors in output
- System health check passes
- Report generation successful

### Test Agent Success

- All 5 test agents pass
- Component-specific validation passes
- No timeout or connection errors

### Full Test Suite Success

- > 95% unit tests passing
- All integration tests passing
- No critical performance regressions
- Code coverage >80%

---

For more information, see:

- [User Guide](USER_GUIDE.md) for usage examples
- [API Reference](API_REFERENCE.md) for implementation details
- [README.md](../README.md) for project overview
