"""
Pytest Configuration

Global test configuration, fixtures, and utilities for the
Jeopardy benchmarking system test suite.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import Mock
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add src to Python path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.config import AppConfig
from core.database import Base
from storage.models import BenchmarkRun, Question, BenchmarkResult
from data.ingestion import DataIngestionEngine
from data.preprocessing import DataPreprocessor
from models.openrouter import OpenRouterClient
from models.base import ModelConfig


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def test_config(temp_dir):
    """Provide test configuration."""
    return AppConfig(
        name="Test Jeopardy Benchmark",
        version="test",
        debug=True,
        database=AppConfig.DatabaseConfig(
            url=f"sqlite:///{temp_dir}/test.db",
            echo=False
        ),
        logging=AppConfig.LoggingConfig(
            level="DEBUG",
            file=str(temp_dir / "test.log")
        ),
        benchmarks=AppConfig.BenchmarkConfig(
            default_sample_size=10,
            max_concurrent_requests=2
        )
    )


@pytest.fixture
def test_db_session(test_config, temp_dir):
    """Provide a test database session."""
    # Create test database
    engine = create_engine(test_config.database.url, echo=False)
    Base.metadata.create_all(engine)
    
    # Create session
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    try:
        yield session
    finally:
        session.close()
        # Clean up
        Base.metadata.drop_all(engine)


@pytest.fixture
def sample_questions_data():
    """Provide sample Jeopardy questions data."""
    return [
        {
            "question_id": "1",
            "question": "This planet is known as the Red Planet",
            "answer": "What is Mars?",
            "category": "ASTRONOMY",
            "value": 200,
            "difficulty_level": "Easy"
        },
        {
            "question_id": "2", 
            "question": "The largest ocean on Earth",
            "answer": "What is the Pacific Ocean?",
            "category": "GEOGRAPHY",
            "value": 400,
            "difficulty_level": "Easy"
        },
        {
            "question_id": "3",
            "question": "The author of 'To Kill a Mockingbird'",
            "answer": "Who is Harper Lee?",
            "category": "LITERATURE",
            "value": 600,
            "difficulty_level": "Medium"
        },
        {
            "question_id": "4",
            "question": "The chemical symbol for gold",
            "answer": "What is Au?",
            "category": "SCIENCE",
            "value": 800,
            "difficulty_level": "Medium"
        },
        {
            "question_id": "5",
            "question": "The year the Berlin Wall fell",
            "answer": "What is 1989?",
            "category": "HISTORY",
            "value": 1000,
            "difficulty_level": "Hard"
        }
    ]


@pytest.fixture
def sample_questions_df(sample_questions_data):
    """Provide sample questions as a pandas DataFrame."""
    return pd.DataFrame(sample_questions_data)


@pytest.fixture
def test_benchmark(test_db_session, sample_questions_data):
    """Create a test benchmark with questions."""
    # Create benchmark
    benchmark = BenchmarkRun(
        name="Test Benchmark",
        description="A test benchmark for unit tests",
        question_count=len(sample_questions_data),
        status="pending"
    )
    test_db_session.add(benchmark)
    test_db_session.commit()
    test_db_session.refresh(benchmark)
    
    # Add questions
    questions = []
    for q_data in sample_questions_data:
        question = Question(
            id=f"q_{q_data['id']}",
            question_id=q_data["question_id"],
            question_text=q_data["question"],
            correct_answer=q_data["answer"],
            category=q_data["category"],
            value=q_data["value"],
            difficulty_level=q_data["difficulty_level"]
        )
        questions.append(question)
    
    test_db_session.add_all(questions)
    test_db_session.commit()
    
    benchmark.questions = questions
    return benchmark


@pytest.fixture
def mock_openrouter_client():
    """Provide a mock OpenRouter client."""
    client = Mock(spec=OpenRouterClient)
    
    # Mock successful responses
    async def mock_query(prompt, **kwargs):
        from models.base import ModelResponse
        return ModelResponse(
            text="What is Mars?",
            model_name="test-model",
            response_time_ms=100,
            tokens_generated=5,
            cost_usd=0.001
        )
    
    async def mock_batch_query(prompts, **kwargs):
        from models.base import ModelResponse
        return [
            ModelResponse(
                text=f"Test answer {i}",
                model_name="test-model", 
                response_time_ms=100,
                tokens_generated=5,
                cost_usd=0.001
            )
            for i in range(len(prompts))
        ]
    
    client.query = mock_query
    client.batch_query = mock_batch_query
    client.is_available.return_value = True
    client.get_pricing_info.return_value = {"input": 0.001, "output": 0.002}
    
    return client


@pytest.fixture
def mock_data_ingestion():
    """Provide a mock data ingestion engine."""
    ingestion = Mock(spec=DataIngestionEngine)
    
    def mock_load_dataset(file_name=None):
        return pd.DataFrame([
            {
                "question": "Test question 1",
                "answer": "Test answer 1", 
                "category": "TEST",
                "value": 200
            },
            {
                "question": "Test question 2",
                "answer": "Test answer 2",
                "category": "TEST", 
                "value": 400
            }
        ])
    
    ingestion.load_dataset = mock_load_dataset
    ingestion.get_dataset_info.return_value = {
        "total_questions": 2,
        "columns": ["question", "answer", "category", "value"],
        "unique_categories": 1
    }
    
    return ingestion


@pytest.fixture
def sample_model_responses(test_benchmark):
    """Provide sample model responses."""
    responses = []
    for i, question in enumerate(test_questions[:3]):
        response = BenchmarkResult(
            benchmark_run_id=test_benchmark.id,
            question_id=question.id,
            model_name="test-model",
            response_text=f"Test response {i}",
            response_time_ms=100 + i * 10,
            tokens_generated=5,
            cost_usd=0.001,
            is_correct=i % 2 == 0,  # Alternate correct/incorrect
            confidence_score=0.8 - (i * 0.1)
        )
        responses.append(response)
    
    return responses


# Test utilities

class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def create_question_data(**overrides) -> Dict[str, Any]:
        """Create question data with optional overrides."""
        default = {
            "question_id": "test_q_1",
            "question": "Test question text",
            "answer": "Test answer",
            "category": "TEST",
            "value": 200,
            "difficulty_level": "Easy"
        }
        default.update(overrides)
        return default
    
    @staticmethod
    def create_questions_df(count: int = 5) -> pd.DataFrame:
        """Create a DataFrame with test questions."""
        questions = []
        for i in range(count):
            questions.append(TestDataGenerator.create_question_data(
                question_id=f"test_q_{i+1}",
                question=f"Test question {i+1}",
                answer=f"Test answer {i+1}",
                value=200 * (i + 1),
                difficulty_level=["Easy", "Medium", "Hard"][i % 3]
            ))
        return pd.DataFrame(questions)


# Pytest markers for test categorization

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests that don't require external dependencies"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests that may require databases or APIs"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests that take more than a few seconds"
    )
    config.addinivalue_line(
        "markers", "api: Tests that require external API access"
    )


# Helper functions for async testing

async def run_async_test(coro):
    """Helper to run async tests in sync test functions."""
    return await coro


def make_async_test(async_func):
    """Decorator to make async functions testable in sync pytest."""
    def wrapper(*args, **kwargs):
        return asyncio.run(async_func(*args, **kwargs))
    return wrapper


# Mock factories

class MockModelResponseFactory:
    """Factory for creating mock model responses."""
    
    @staticmethod
    def create(text: str = "Mock response", is_correct: bool = True, **kwargs):
        """Create a mock model response."""
        from models.base import ModelResponse
        defaults = {
            "text": text,
            "model_name": "mock-model",
            "response_time_ms": 100,
            "tokens_generated": 10,
            "cost_usd": 0.001,
            "metadata": {"mock": True}
        }
        defaults.update(kwargs)
        return ModelResponse(**defaults)


# Test environment setup

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch, temp_dir):
    """Set up test environment variables."""
    # Mock environment variables
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_api_key")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DEBUG", "True")
    
    # Set test data directory
    monkeypatch.setenv("JEOPARDY_TEST_DATA_DIR", str(temp_dir))


@pytest.fixture(scope="session", autouse=True)
def suppress_warnings():
    """Suppress known warnings in test environment."""
    import warnings
    
    # Suppress specific warnings that are not relevant for tests
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*asyncio.*")
    warnings.filterwarnings("ignore", message=".*SQLAlchemy.*")