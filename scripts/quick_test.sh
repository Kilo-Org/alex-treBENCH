#!/bin/bash

# alex-treBENCH Quick Test Script
# Provides rapid system verification with clear success/failure reporting

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Emojis for better visual feedback
SUCCESS="✅"
FAILURE="❌"
WARNING="⚠️"
INFO="ℹ️"
ROCKET="🚀"
FIRE="🔥"

echo -e "${BLUE}${ROCKET} alex-treBENCH Quick Test${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Function to print status messages
print_status() {
    local status=$1
    local message=$2
    case $status in
        "success")
            echo -e "${GREEN}${SUCCESS} $message${NC}"
            ;;
        "failure")
            echo -e "${RED}${FAILURE} $message${NC}"
            ;;
        "warning")
            echo -e "${YELLOW}${WARNING} $message${NC}"
            ;;
        "info")
            echo -e "${BLUE}${INFO} $message${NC}"
            ;;
    esac
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_CMD="python"
    else
        print_status "failure" "Python not found in PATH"
        return 1
    fi

    # Check Python version
    local version=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    local major=$(echo $version | cut -d'.' -f1)
    local minor=$(echo $version | cut -d'.' -f2)
    
    if [ "$major" -lt 3 ] || ([ "$major" -eq 3 ] && [ "$minor" -lt 8 ]); then
        print_status "failure" "Python 3.8+ required, found $version"
        return 1
    fi
    
    print_status "success" "Python $version found"
    return 0
}

# Function to check if we're in the right directory
check_project_directory() {
    if [ ! -f "scripts/smoke_test.py" ] || [ ! -f "src/main.py" ] || [ ! -f "requirements.txt" ]; then
        print_status "failure" "Not in alex-treBENCH project directory"
        print_status "info" "Please run this script from the project root directory"
        return 1
    fi
    
    print_status "success" "Project directory verified"
    return 0
}

# Function to check basic dependencies
check_dependencies() {
    local missing_deps=0
    
    # Check if virtual environment is active (optional but recommended)
    if [ -z "$VIRTUAL_ENV" ]; then
        print_status "warning" "Virtual environment not detected (recommended but not required)"
    else
        print_status "success" "Virtual environment active: $(basename $VIRTUAL_ENV)"
    fi
    
    # Try to import key modules
    if ! $PYTHON_CMD -c "import src.core.config, src.core.database, src.benchmark.runner" 2>/dev/null; then
        print_status "failure" "Core modules not importable - run 'pip install -r requirements.txt'"
        missing_deps=1
    else
        print_status "success" "Core dependencies available"
    fi
    
    # Check if Rich is available for nice output
    if ! $PYTHON_CMD -c "import rich" 2>/dev/null; then
        print_status "warning" "Rich not available - install requirements-dev.txt for better output"
    fi
    
    return $missing_deps
}

# Function to run the smoke test
run_smoke_test() {
    print_status "info" "Running smoke test..."
    echo ""
    
    # Run the smoke test and capture exit code
    if $PYTHON_CMD scripts/smoke_test.py; then
        echo ""
        print_status "success" "Smoke test completed successfully"
        return 0
    else
        local exit_code=$?
        echo ""
        print_status "failure" "Smoke test failed (exit code: $exit_code)"
        return $exit_code
    fi
}

# Function to provide troubleshooting help
show_troubleshooting() {
    echo ""
    echo -e "${YELLOW}${WARNING} Troubleshooting Tips:${NC}"
    echo ""
    echo "1. Dependencies issue:"
    echo "   pip install -r requirements.txt"
    echo "   pip install -r requirements-dev.txt"
    echo ""
    echo "2. Database issues:"
    echo "   rm -f data/*.db"
    echo "   python -m src.main init"
    echo ""
    echo "3. Module import errors:"
    echo "   export PYTHONPATH=\$(pwd)"
    echo "   source venv/bin/activate  # if using virtual env"
    echo ""
    echo "4. API connectivity issues:"
    echo "   unset OPENROUTER_API_KEY  # to use simulation mode"
    echo "   export OPENROUTER_API_KEY=\"your_key_here\"  # for real API calls"
    echo ""
    echo "5. For detailed testing information:"
    echo "   cat docs/TESTING.md"
    echo ""
}

# Function to show success summary
show_success_summary() {
    echo ""
    echo -e "${GREEN}${FIRE} System Verification Complete! ${FIRE}${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    print_status "success" "alex-treBENCH is working correctly"
    print_status "success" "All core components are functional"
    print_status "success" "Ready for benchmarking operations"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "• Run your first benchmark: python -m src.main benchmark run --model openai/gpt-3.5-turbo --size quick"
    echo "• View available models: python -m src.main models list"
    echo "• Read the user guide: docs/USER_GUIDE.md"
    echo ""
}

# Main execution
main() {
    local overall_success=0
    
    # Step 1: Check prerequisites
    print_status "info" "Checking prerequisites..."
    
    if ! check_python_version; then
        overall_success=1
    fi
    
    if ! check_project_directory; then
        overall_success=1
    fi
    
    if ! check_dependencies; then
        overall_success=1
    fi
    
    # If prerequisites failed, exit early
    if [ $overall_success -eq 1 ]; then
        echo ""
        print_status "failure" "Prerequisites check failed"
        show_troubleshooting
        exit 1
    fi
    
    echo ""
    print_status "success" "All prerequisites satisfied"
    echo ""
    
    # Step 2: Run smoke test
    if ! run_smoke_test; then
        overall_success=1
    fi
    
    # Show final result
    if [ $overall_success -eq 0 ]; then
        show_success_summary
        exit 0
    else
        echo ""
        print_status "failure" "Quick test failed"
        show_troubleshooting
        exit 1
    fi
}

# Handle script interruption
trap 'echo -e "\n${YELLOW}Test interrupted by user${NC}"; exit 130' INT

# Run main function
main "$@"
