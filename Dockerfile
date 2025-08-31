# Jeopardy Benchmarking System - Multi-stage Docker Build
# Optimized for production deployment with minimal image size

# =============================================================================
# Base Stage - Common dependencies and setup
# =============================================================================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# =============================================================================
# Builder Stage - Install Python dependencies
# =============================================================================
FROM base as builder

# Install Python build dependencies
RUN apt-get update && apt-get install -y \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# =============================================================================
# Development Stage - Full development environment
# =============================================================================
FROM base as development

# Copy installed dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data logs config && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port for potential web interface
EXPOSE 8000

# Default command
CMD ["python", "-m", "src.main", "--help"]

# =============================================================================
# Testing Stage - Environment for running tests
# =============================================================================
FROM development as testing

# Install test dependencies
RUN pip install --user --no-cache-dir -r requirements-dev.txt

# Copy test files
COPY tests/ ./tests/

# Run tests by default
CMD ["pytest", "tests/", "-v", "--cov=src/", "--cov-report=term-missing"]

# =============================================================================
# Production Stage - Optimized for production deployment
# =============================================================================
FROM base as production

# Copy installed dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code (excluding unnecessary files)
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY setup.py ./
COPY README.md ./

# Create necessary directories and set permissions
RUN mkdir -p data logs reports && \
    chown -R appuser:appuser /app

# Create volume mount points
VOLUME ["/app/data", "/app/logs", "/app/reports", "/app/config"]

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.core.config import get_config; print('Config loaded successfully')" || exit 1

# Default command
CMD ["python", "-m", "src.main", "--help"]

# =============================================================================
# CLI Stage - Optimized for CLI usage
# =============================================================================
FROM production as cli

# Set CLI-specific environment
ENV JEOPARDY_ENV=production \
    LOG_LEVEL=INFO

# Default command for CLI usage
CMD ["python", "-m", "src.main"]

# =============================================================================
# Demo Stage - Environment for running demonstrations
# =============================================================================
FROM production as demo

# Copy demo and example files
COPY examples/ ./examples/
COPY docs/ ./docs/

# Install additional demo dependencies if needed
RUN pip install --user --no-cache-dir rich matplotlib seaborn plotly

# Demo command
CMD ["python", "scripts/demo.py"]

# =============================================================================
# Labels for better container management
# =============================================================================
LABEL org.opencontainers.image.title="Jeopardy Benchmarking System" \
      org.opencontainers.image.description="Comprehensive benchmarking tool for language models using Jeopardy questions" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.authors="Jeopardy Bench Team" \
      org.opencontainers.image.source="https://github.com/your-org/jeopardy-benchmark" \
      org.opencontainers.image.licenses="MIT"

# =============================================================================
# Build Arguments
# =============================================================================
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.revision=$VCS_REF \
      org.opencontainers.image.version=$VERSION

# =============================================================================
# Default target
# =============================================================================
FROM production