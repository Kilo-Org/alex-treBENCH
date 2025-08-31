# Installation Guide - alex-treBENCH CLI

This guide covers multiple ways to install and run the alex-treBENCH command-line tool.

## Option 1: Development Installation (Recommended)

For development and most users, install using `uv` (or `pip`) in editable mode:

```bash
# Clone the repository
git clone <repository-url>
cd alex-trebench

# Install in development mode using uv (recommended)
uv pip install -e .

# Or using pip if you prefer
pip install -e .
```

After installation, you can use the `alex` command directly:

```bash
alex --help
alex init
alex benchmark run --help
alex models list
```

### Verification

Test that the installation worked:

```bash
alex --help
alex models --help
alex benchmark --help
```

## Option 2: Production Installation

Install from PyPI (when published):

```bash
uv pip install alex-trebench
# or
pip install alex-trebench
```

## Option 3: Standalone Binary

For distribution without requiring Python installation on target systems:

### Building the Binary

```bash
# Run the build script
python build_binary.py
```

This will:

1. Install PyInstaller if not already installed
2. Create a standalone binary in the `dist/` directory
3. Test the binary to ensure it works

### Using the Binary

After building, you can run the standalone executable:

```bash
# On macOS/Linux
./dist/alex --help

# On Windows
.\dist\alex.exe --help
```

The binary includes all dependencies and can be distributed to systems without Python installed.

## Command Structure

The alex-treBENCH CLI provides the following main command groups:

### Core Commands

- `alex init` - Initialize the database and create tables
- `alex health` - Check system health and connectivity
- `alex config-show` - Show current configuration

### Benchmark Commands

- `alex benchmark run` - Run a benchmark for a specific model
- `alex benchmark compare` - Compare multiple models
- `alex benchmark history` - Show benchmark history
- `alex benchmark list` - List recent benchmarks
- `alex benchmark report` - Generate detailed reports

### Model Commands

- `alex models list` - List available models from OpenRouter
- `alex models search <query>` - Search for specific models
- `alex models info <model-id>` - Show detailed model information
- `alex models refresh` - Update model cache from OpenRouter API
- `alex models cache` - Manage model cache
- `alex models test` - Test a model with a prompt
- `alex models costs` - Estimate benchmark costs

### Data Commands

- `alex data init` - Initialize the Jeopardy dataset
- `alex data stats` - Show dataset statistics
- `alex data sample` - Test sampling functionality

## Quick Start Examples

### Initialize the System

```bash
# Initialize database
alex init

# Initialize Jeopardy dataset
alex data init
```

### Run Your First Benchmark

```bash
# List available models
alex models list

# Run a quick benchmark with the default model
alex benchmark run --size quick

# Run a benchmark with a specific model
alex benchmark run --model "anthropic/claude-3.5-sonnet" --size standard
```

### Compare Multiple Models

```bash
alex benchmark compare --models "openai/gpt-4,anthropic/claude-3.5-sonnet" --size quick
```

### Explore Models

```bash
# Search for GPT models
alex models search "gpt-4"

# Get detailed info about a model
alex models info "anthropic/claude-3.5-sonnet"

# Estimate costs for a benchmark
alex models costs --model "openai/gpt-4" --questions 100
```

## Configuration

The CLI supports various configuration options:

### Environment Variables

```bash
export OPENROUTER_API_KEY="your-api-key-here"
export DATABASE_URL="sqlite:///path/to/database.db"
```

### Configuration File

You can specify a custom configuration file:

```bash
alex --config custom-config.yaml benchmark run
```

### Global Options

All commands support these global options:

- `--config PATH` - Custom configuration file
- `--verbose` - Enable verbose logging
- `--debug` - Enable debug mode

## Troubleshooting

### Common Issues

1. **"alex command not found"**

   - Make sure you've installed the package with `uv pip install -e .`
   - Check that your virtual environment is activated

2. **"No module named 'src'"**

   - This should be fixed in the current version. Try reinstalling:

   ```bash
   uv pip uninstall alex-trebench
   uv pip install -e .
   ```

3. **OpenRouter API errors**

   - Ensure your `OPENROUTER_API_KEY` environment variable is set
   - Check your API key validity with `alex models list`

4. **Database initialization errors**
   - Try forcing reinitialization: `alex init --force`
   - Check file permissions in the database directory

### Getting Help

- Use `--help` with any command to see detailed options
- Check the logs with `--verbose` or `--debug` flags
- Visit the project documentation for more details

## Development

If you're contributing to the project:

```bash
# Install development dependencies
uv pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Format code
black src/ tests/
isort src/ tests/

# Build binary for testing
python build_binary.py
```

## Binary Distribution Notes

The standalone binary:

- Contains all Python dependencies
- Is typically 50-100MB in size
- Works on the target platform (macOS/Linux/Windows)
- Does not require Python installation on target system
- Includes configuration files and data templates

The binary is ideal for:

- Distribution to non-technical users
- Deployment in containerized environments
- Systems where Python installation is restricted
- Simplified deployment workflows
