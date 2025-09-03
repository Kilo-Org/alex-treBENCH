#!/bin/bash

# Jeopardy Benchmarking System - Quick Start Script
# This script sets up the environment and runs sample benchmarks

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "setup.py" ] || [ ! -d "src" ]; then
    print_error "Please run this script from the alex-trebench project root directory"
    exit 1
fi

echo "🧠 Jeopardy Benchmarking System - Quick Start"
echo "=============================================="

# Step 1: Check Python version
print_step "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
if [[ "$python_version" =~ ^3\.[8-9] ]]; then
    print_success "Python $python_version is compatible"
else
    print_error "Python 3.8+ is required. Current version: $python_version"
    exit 1
fi

# Step 2: Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_step "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Step 3: Activate virtual environment
print_step "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Step 4: Install package
print_step "Installing alex-trebench package..."
if command -v uv >/dev/null 2>&1; then
    print_success "Using uv for installation"
    uv pip install -e .
else
    print_warning "uv not found, using pip"
    pip install --upgrade pip
    pip install -e .
fi
print_success "Package installed"

# Step 5: Check for API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    print_warning "OPENROUTER_API_KEY environment variable not set"
    echo "Please set your OpenRouter API key:"
    echo "export OPENROUTER_API_KEY='your_api_key_here'"
    echo ""
    read -p "Do you want to continue with mock data only? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_error "Setup cancelled. Please set OPENROUTER_API_KEY and run again."
        exit 1
    fi
    USE_MOCK=true
else
    print_success "OpenRouter API key found"
    USE_MOCK=false
fi

# Step 6: Initialize database
print_step "Initializing database..."
alex init
print_success "Database initialized"

# Step 7: Run system health check
print_step "Running system health check..."
alex health
print_success "System health check completed"

# Step 8: Run sample benchmarks
if [ "$USE_MOCK" = true ]; then
    print_step "Running sample benchmark with mock data..."
    print_warning "Using mock data (no API calls will be made)"

    # Run a quick benchmark (this will use mock data in tests)
    python -c "
import asyncio
import sys
sys.path.insert(0, 'src')
from benchmark.runner import BenchmarkRunner, RunMode

async def mock_benchmark():
    runner = BenchmarkRunner()
    # This would normally make API calls, but we'll demonstrate the structure
    print('Mock benchmark completed successfully!')
    print('In a real scenario, this would test the model with Jeopardy questions')

asyncio.run(mock_benchmark())
"
else
    print_step "Running quick benchmark with GPT-3.5-turbo..."
    alex benchmark run --model openai/gpt-3.5-turbo --size quick
    print_success "Quick benchmark completed"
fi

# Step 9: Generate sample report
print_step "Generating sample report..."
if [ "$USE_MOCK" = false ]; then
    alex benchmark report --run-id 1 --format markdown --output sample_report.md
    print_success "Sample report generated: sample_report.md"
else
    # Create a mock report
    cat > sample_report.md << 'EOF'
# Sample Benchmark Report (Mock Data)

This is a sample report showing the expected output format.

## Summary
- **Model:** GPT-3.5-turbo (Mock)
- **Accuracy:** 75.2%
- **Response Time:** 1.2s average
- **Total Cost:** $0.45

## Recommendations
1. Use for quick evaluations
2. Consider GPT-4 for production use
3. Monitor API costs

*This is mock data for demonstration purposes.*
EOF
    print_success "Mock sample report generated: sample_report.md"
fi

# Step 10: Show available commands
print_step "Displaying available commands..."
echo ""
echo "📋 Available Commands:"
echo "======================"
alex --help

echo ""
echo "🔧 Useful Commands:"
echo "==================="
echo "# Run a comprehensive benchmark"
echo "alex benchmark run --model openai/gpt-4 --size comprehensive"
echo ""
echo "# Compare multiple models"
echo "alex benchmark compare --models 'openai/gpt-3.5-turbo,openai/gpt-4' --size standard"
echo ""
echo "# View benchmark history"
echo "alex benchmark history --model openai/gpt-4"
echo ""
echo "# Generate detailed report"
echo "alex benchmark report --run-id 1 --format markdown"
echo ""
echo "# List all benchmarks"
echo "alex benchmark list"

# Step 11: Create demo script
print_step "Creating demo script..."
cat > run_demo.sh << 'EOF'
#!/bin/bash
# Demo script for Jeopardy Benchmarking System

echo "🧠 Running Jeopardy Benchmark Demo"
echo "==================================="

# Activate virtual environment
source venv/bin/activate

echo "1. System Health Check:"
alex health

echo ""
echo "2. List Available Models:"
alex models list

echo ""
echo "3. Run Quick Benchmark:"
alex benchmark run --model openai/gpt-3.5-turbo --size quick

echo ""
echo "4. View Results:"
alex benchmark list

echo ""
echo "Demo completed! Check the generated reports for detailed results."
EOF

chmod +x run_demo.sh
print_success "Demo script created: run_demo.sh"

# Final instructions
echo ""
echo "🎉 Quick Start Setup Complete!"
echo "=============================="
print_success "Environment is ready for benchmarking"
echo ""
echo "Next steps:"
echo "1. Set your OpenRouter API key (if not already set):"
echo "   export OPENROUTER_API_KEY='your_key_here'"
echo ""
echo "2. Run your first real benchmark:"
echo "   alex benchmark run --model openai/gpt-3.5-turbo --size quick"
echo ""
echo "3. Explore the system:"
echo "   alex --help"
echo ""
echo "4. Run the demo:"
echo "   ./run_demo.sh"
echo ""
echo "📚 Documentation:"
echo "   User Guide: docs/USER_GUIDE.md"
echo "   API Reference: docs/API_REFERENCE.md"
echo "   README: README.md"
echo ""
print_warning "Remember to monitor your API usage and costs!"

# Deactivate virtual environment
deactivate