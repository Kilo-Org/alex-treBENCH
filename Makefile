# Jeopardy Benchmarking System - Makefile
# Common commands for development, testing, and deployment

.PHONY: help install install-dev test test-verbose test-coverage lint format clean build run dev demo validate simulate docker-build docker-run docker-stop docker-clean deploy docs

# Default target
help: ## Show this help message
	@echo "Jeopardy Benchmarking System - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# Installation & Setup
# =============================================================================

install: ## Install production dependencies
	@echo "📦 Installing production dependencies..."
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	@echo "📦 Installing development dependencies..."
	pip install -r requirements.txt -r requirements-dev.txt

setup: ## Complete setup (install deps, init database)
	@echo "🚀 Setting up Jeopardy Benchmarking System..."
	make install-dev
	make init-db
	@echo "✅ Setup complete! Run 'make demo' to see it in action."

init-db: ## Initialize the database
	@echo "🗄️  Initializing database..."
	alex init

# =============================================================================
# Testing
# =============================================================================

test: ## Run all tests
	@echo "🧪 Running tests..."
	pytest tests/ -v

test-verbose: ## Run tests with detailed output
	@echo "🧪 Running tests (verbose)..."
	pytest tests/ -v -s

test-coverage: ## Run tests with coverage report
	@echo "🧪 Running tests with coverage..."
	pytest tests/ --cov=src/ --cov-report=html --cov-report=term-missing
	@echo "📊 Coverage report generated: htmlcov/index.html"

test-unit: ## Run only unit tests
	@echo "🧪 Running unit tests..."
	pytest tests/unit/ -v

test-integration: ## Run only integration tests
	@echo "🧪 Running integration tests..."
	pytest tests/integration/ -v

test-e2e: ## Run only end-to-end tests
	@echo "🧪 Running end-to-end tests..."
	pytest tests/e2e/ -v

test-performance: ## Run only performance tests
	@echo "🧪 Running performance tests..."
	pytest tests/performance/ -v

smoke-test: ## Run complete end-to-end smoke test
	@echo "🔥 Running smoke test..."
	python scripts/smoke_test.py

test-agents: ## Run comprehensive test agents
	@echo "🤖 Running test agents..."
	python scripts/test_agents.py

# =============================================================================
# Code Quality
# =============================================================================

lint: ## Run linting checks
	@echo "🔍 Running linters..."
	@command -v flake8 >/dev/null 2>&1 || { echo "Installing flake8..."; pip install flake8; }
	@command -v black >/dev/null 2>&1 || { echo "Installing black..."; pip install black; }
	@command -v isort >/dev/null 2>&1 || { echo "Installing isort..."; pip install isort; }
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format: ## Format code with black and isort
	@echo "🎨 Formatting code..."
	@command -v black >/dev/null 2>&1 || { echo "Installing black..."; pip install black; }
	@command -v isort >/dev/null 2>&1 || { echo "Installing isort..."; pip install isort; }
	black src/ tests/
	isort src/ tests/

type-check: ## Run type checking with mypy
	@echo "🔍 Running type checking..."
	@command -v mypy >/dev/null 2>&1 || { echo "Installing mypy..."; pip install mypy; }
	mypy src/

quality: ## Run all code quality checks
	@echo "🔍 Running all code quality checks..."
	make lint
	make type-check
	make test

# =============================================================================
# Development
# =============================================================================

dev: ## Start development environment
	@echo "🚀 Starting development environment..."
	alex --help

run-quick: ## Run a quick benchmark for testing
	@echo "⚡ Running quick benchmark..."
	alex benchmark run --size quick

run-standard: ## Run a standard benchmark  
	@echo "📊 Running standard benchmark..."
	alex benchmark run --size standard

run-comparison: ## Run model comparison
	@echo "🔄 Running model comparison..."
	alex benchmark compare --size quick

list-models: ## List available models
	@echo "📋 Listing available models..."
	alex models list

# =============================================================================
# Validation & Simulation  
# =============================================================================

validate: ## Run system validation
	@echo "🔍 Running system validation..."
	python scripts/validate_system.py

simulate: ## Run benchmark simulation
	@echo "🎭 Running benchmark simulation..."
	python scripts/benchmark_simulator.py --models 3 --questions 50

simulate-fast: ## Run fast simulation (10x speed)
	@echo "⚡ Running fast simulation..."
	python scripts/benchmark_simulator.py --models 3 --questions 50 --speed 10

demo: ## Run interactive demo
	@echo "🎭 Running interactive demo..."
	python scripts/demo.py

# =============================================================================
# Docker
# =============================================================================

docker-build: ## Build Docker image
	@echo "🏗️  Building Docker image..."
	docker build -t alex-trebench .

docker-run: ## Run application in Docker
	@echo "🐳 Running in Docker..."
	docker run --rm -it \
		--env-file .env \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/logs:/app/logs \
		-v $(PWD)/reports:/app/reports \
		alex-trebench

docker-dev: ## Run development environment with Docker Compose
	@echo "🐳 Starting development environment..."
	docker-compose --profile dev up --build

docker-test: ## Run tests in Docker
	@echo "🐳 Running tests in Docker..."
	docker-compose --profile test run --rm test

docker-demo: ## Run demo in Docker
	@echo "🐳 Running demo in Docker..."
	docker-compose --profile demo run --rm demo

docker-stop: ## Stop all Docker services
	@echo "🐳 Stopping Docker services..."
	docker-compose down

docker-clean: ## Clean up Docker resources
	@echo "🧹 Cleaning up Docker resources..."
	docker-compose down -v
	docker system prune -f
	docker image rm alex-trebench 2>/dev/null || true

# =============================================================================
# Database
# =============================================================================

db-init: ## Initialize database
	@echo "🗄️  Initializing database..."
	alex init

db-reset: ## Reset database (WARNING: destroys all data)
	@echo "⚠️  WARNING: This will destroy all data!"
	@read -p "Are you sure? (y/N): " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		echo "🔄 Resetting database..."; \
		find . -name "*.db" -path "./data/*" -delete 2>/dev/null || true; \
		make db-init; \
	else \
		echo "❌ Database reset cancelled."; \
	fi

db-backup: ## Backup database
	@echo "💾 Backing up database..."
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	mkdir -p backups; \
	find . -name "*.db" -path "./data/*" -exec cp {} backups/jeopardy_backup_$$timestamp.db \; 2>/dev/null && \
		echo "✅ Backup created: backups/jeopardy_backup_$$timestamp.db" || \
		echo "❌ No database files found to backup"

db-restore: ## Restore database from backup  
	@echo "📁 Available backups:"
	@ls -la backups/ 2>/dev/null || echo "No backups directory found"
	@echo ""
	@read -p "Enter backup filename: " backup_file; \
	if [ -f "backups/$$backup_file" ]; then \
		echo "🔄 Restoring from $$backup_file..."; \
		mkdir -p data; \
		cp backups/$$backup_file data/alex_trebench.db; \
		echo "✅ Database restored."; \
	else \
		echo "❌ Backup file not found."; \
	fi

# =============================================================================
# Reporting & Analytics
# =============================================================================

reports-list: ## List all benchmark reports
	@echo "📋 Available benchmark reports:"
	alex benchmark list

reports-clean: ## Clean up old reports
	@echo "🧹 Cleaning up old reports..."
	find reports/ -name "*.md" -mtime +30 -delete 2>/dev/null || true
	find reports/ -name "*.json" -mtime +30 -delete 2>/dev/null || true
	echo "✅ Old reports cleaned up."

# =============================================================================
# Documentation
# =============================================================================

docs: ## Show documentation locations
	@echo "📚 Documentation available at:"
	@echo "  User Guide: docs/USER_GUIDE.md"
	@echo "  API Reference: docs/API_REFERENCE.md"
	@echo "  README: README.md"
	@echo "  Technical Spec: TECHNICAL_SPEC.md"

# =============================================================================
# Deployment
# =============================================================================

deploy-check: ## Pre-deployment checks
	@echo "🔍 Running pre-deployment checks..."
	make test
	make validate
	make lint
	@echo "✅ Pre-deployment checks passed"

# =============================================================================
# Cleanup
# =============================================================================

clean: ## Clean up temporary files
	@echo "🧹 Cleaning up temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.log" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/ .pytest_cache/ .mypy_cache/
	rm -rf simulation_results/ demo_results/
	rm -f system_validation_report.json
	@echo "✅ Cleanup complete"

clean-all: ## Clean up everything (including data)
	@echo "⚠️  WARNING: This will remove all data and logs!"
	@read -p "Are you sure? (y/N): " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		make clean; \
		rm -rf data/ logs/ reports/ backups/; \
		echo "✅ Full cleanup complete"; \
	else \
		echo "❌ Cleanup cancelled."; \
	fi

# =============================================================================
# Information
# =============================================================================

info: ## Show system information
	@echo "ℹ️  Jeopardy Benchmarking System Information"
	@echo "=========================================="
	@echo "Python Version: $$(python --version)"
	@echo "Working Directory: $$(pwd)"
	@echo "Database: $$(find data/ -name "*.db" 2>/dev/null | wc -l) file(s)"
	@echo "Reports: $$(find reports/ -name "*.md" 2>/dev/null | wc -l) markdown, $$(find reports/ -name "*.json" 2>/dev/null | wc -l) json"
	@echo "Logs: $$(find logs/ -name "*.log" 2>/dev/null | wc -l) file(s)"
	@echo ""
	@echo "Recent Benchmarks:"
	@python -c "\
try:\
    from src.core.database import get_db_session;\
    from src.storage.repositories import BenchmarkRepository;\
    with get_db_session() as session:\
        repo = BenchmarkRepository(session);\
        benchmarks = repo.list_benchmarks(limit=3);\
        if benchmarks:\
            for b in benchmarks:\
                print(f'  • {b.name} (ID: {b.id}, Status: {b.status})');\
        else:\
            print('  No benchmarks found');\
except:\
    print('  Database not initialized');\
" 2>/dev/null || echo "  Database not accessible"

version: ## Show version information
	@echo "alex-treBENCH v1.0.0"
	@echo "Built with Python 3.8+"
	@echo "Supports 323+ models via OpenRouter API"
	@echo ""
	@echo "Components:"
	@echo "  • Core Engine: ✅ Active"
	@echo "  • Database: ✅ SQLite/PostgreSQL"
	@echo "  • CLI Interface: ✅ Complete"
	@echo "  • Dynamic Models: ✅ 323+ via OpenRouter"
	@echo "  • Reporting: ✅ Markdown/JSON"
	@echo "  • Docker Support: ✅ Multi-stage"
	@echo "  • Testing: ✅ Comprehensive"

# =============================================================================
# Aliases for common tasks
# =============================================================================

quick-test: test-unit ## Quick test run (unit tests only)
full-test: test-coverage ## Full test suite with coverage
check: quality ## Run all quality checks
build: docker-build ## Build application
start: docker-dev ## Start development environment
stop: docker-stop ## Stop all services