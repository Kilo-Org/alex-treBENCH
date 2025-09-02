"""
End-to-End Tests for Complete Benchmarking Workflow

Tests the complete workflow from data loading to report generation,
including error recovery and session management.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List
import json

from src.benchmark.runner import BenchmarkRunner, RunMode, BenchmarkConfig, BenchmarkRunResult
from src.benchmark.reporting import ReportGenerator, ReportFormat
from src.core.config import AppConfig
from src.core.database import init_database, get_session
from src.storage.repositories import BenchmarkRepository, QuestionRepository, ResponseRepository
from src.models.openrouter import OpenRouterClient
from src.models.base import ModelConfig, ModelResponse
from src.evaluation.grader import GradingMode
from src.evaluation.metrics import ComprehensiveMetrics


class TestCompleteWorkflow:
    """Test complete benchmarking workflow end-to-end."""

    @pytest.fixture
    def test_config(self, temp_dir):
        """Test configuration for e2e tests."""
        return AppConfig(
            name="E2E Test Jeopardy Benchmark",
            version="e2e-test",
            debug=True,
            database=AppConfig.DatabaseConfig(
                url=f"sqlite:///{temp_dir}/e2e_test.db",
                echo=False
            ),
            logging=AppConfig.LoggingConfig(
                level="DEBUG",
                file=str(temp_dir / "e2e_test.log")
            ),
            benchmarks=AppConfig.BenchmarkConfig(
                default_sample_size=20,
                max_concurrent_requests=3
            )
        )

    @pytest.fixture
    def mock_openrouter_responses(self):
        """Mock OpenRouter API responses for testing."""
        return [
            ModelResponse(
                text="What is Mars?",
                model_name="test-model",
                response_time_ms=150,
                tokens_generated=4,
                cost_usd=0.001,
                metadata={"mock": True}
            ),
            ModelResponse(
                text="What is the Pacific Ocean?",
                model_name="test-model",
                response_time_ms=120,
                tokens_generated=6,
                cost_usd=0.001,
                metadata={"mock": True}
            ),
            ModelResponse(
                text="Who is Harper Lee?",
                model_name="test-model",
                response_time_ms=180,
                tokens_generated=5,
                cost_usd=0.001,
                metadata={"mock": True}
            ),
            ModelResponse(
                text="What is Au?",
                model_name="test-model",
                response_time_ms=100,
                tokens_generated=3,
                cost_usd=0.001,
                metadata={"mock": True}
            ),
            ModelResponse(
                text="What is 1989?",
                model_name="test-model",
                response_time_ms=140,
                tokens_generated=4,
                cost_usd=0.001,
                metadata={"mock": True}
            )
        ]

    @pytest.fixture
    def mock_openrouter_client(self, mock_openrouter_responses):
        """Mock OpenRouter client with realistic responses."""
        client = Mock(spec=OpenRouterClient)

        async def mock_batch_query(prompts, **kwargs):
            # Return responses matching the number of prompts
            responses = []
            for i, prompt in enumerate(prompts):
                if i < len(mock_openrouter_responses):
                    response = mock_openrouter_responses[i]
                else:
                    # Generate additional mock responses if needed
                    response = ModelResponse(
                        text=f"Mock answer {i}",
                        model_name="test-model",
                        response_time_ms=100 + (i * 10),
                        tokens_generated=5,
                        cost_usd=0.001,
                        metadata={"mock": True}
                    )
                responses.append(response)
            return responses

        client.batch_query = AsyncMock(side_effect=mock_batch_query)
        client.is_available.return_value = True
        client.get_pricing_info.return_value = {"input": 0.001, "output": 0.002}

        return client

    def setup_method(self):
        """Set up test environment."""
        # Ensure clean state for each test
        pass

    def teardown_method(self):
        """Clean up after each test."""
        pass

    @pytest.mark.asyncio
    async def test_complete_benchmark_workflow(self, test_config, mock_openrouter_client):
        """Test complete benchmark workflow from start to finish."""
        # Initialize database
        init_database(test_config)

        # Patch the OpenRouter client
        with patch('src.models.openrouter.OpenRouterClient', return_value=mock_openrouter_client):
            # Create benchmark runner
            runner = BenchmarkRunner()

            # Configure benchmark
            config = BenchmarkConfig(
                mode=RunMode.QUICK,
                sample_size=5,
                timeout_seconds=30,
                grading_mode=GradingMode.LENIENT,
                save_results=True
            )

            # Run benchmark
            result = await runner.run_benchmark(
                model_name="openai/gpt-3.5-turbo",
                mode=RunMode.QUICK,
                custom_config=config,
                benchmark_name="e2e_test_benchmark"
            )

            # Verify result structure
            assert isinstance(result, BenchmarkRunResult)
            assert result.success is True
            assert result.benchmark_id > 0
            assert result.model_name == "openai/gpt-3.5-turbo"
            assert result.execution_time > 0
            assert len(result.questions) == 5
            assert len(result.responses) == 5
            assert len(result.graded_responses) == 5
            assert isinstance(result.metrics, ComprehensiveMetrics)

            # Verify progress tracking
            assert result.progress.total_questions == 5
            assert result.progress.completed_questions == 5
            assert result.progress.successful_responses >= 0
            assert result.progress.current_phase == "Complete"

            # Verify metrics
            assert result.metrics.overall_score >= 0.0
            assert result.metrics.overall_score <= 1.0
            assert result.metrics.accuracy.overall_accuracy >= 0.0
            assert result.metrics.accuracy.overall_accuracy <= 1.0

            # Verify metadata
            assert 'benchmark_name' in result.metadata
            assert 'completion_time' in result.metadata
            assert 'total_cost' in result.metadata
            assert 'avg_response_time' in result.metadata

    @pytest.mark.asyncio
    async def test_benchmark_with_error_recovery(self, test_config):
        """Test benchmark with simulated errors and recovery."""
        # Initialize database
        init_database(test_config)

        # Create mock client that fails on first attempt
        client = Mock(spec=OpenRouterClient)
        call_count = 0

        async def mock_batch_query_with_failure(prompts, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Simulate API failure on first call
                raise Exception("Simulated API failure")
            else:
                # Return successful responses on retry
                return [
                    ModelResponse(
                        text=f"Recovery answer {i}",
                        model_name="test-model",
                        response_time_ms=100,
                        tokens_generated=5,
                        cost_usd=0.001,
                        metadata={"mock": True, "retry": True}
                    )
                    for i in range(len(prompts))
                ]

        client.batch_query = AsyncMock(side_effect=mock_batch_query_with_failure)
        client.is_available.return_value = True

        with patch('src.models.openrouter.OpenRouterClient', return_value=client):
            runner = BenchmarkRunner()

            config = BenchmarkConfig(
                mode=RunMode.QUICK,
                sample_size=3,
                timeout_seconds=30,
                grading_mode=GradingMode.STRICT,
                save_results=True
            )

            # This should handle the error gracefully
            result = await runner.run_benchmark(
                model_name="openai/gpt-4",
                mode=RunMode.QUICK,
                custom_config=config,
                benchmark_name="error_recovery_test"
            )

            # Verify the benchmark completed despite initial failure
            assert isinstance(result, BenchmarkRunResult)
            assert result.is_successful is True  # Should succeed on retry
            assert result.benchmark_id > 0
            assert len(result.responses) == 3

    @pytest.mark.asyncio
    async def test_benchmark_with_different_modes(self, test_config, mock_openrouter_client):
        """Test benchmark with different run modes."""
        init_database(test_config)

        with patch('src.models.openrouter.OpenRouterClient', return_value=mock_openrouter_client):
            runner = BenchmarkRunner()

            # Test different modes
            modes_to_test = [RunMode.QUICK, RunMode.STANDARD]

            for mode in modes_to_test:
                config = runner.mode_configs[mode]

                result = await runner.run_benchmark(
                    model_name="anthropic/claude-3-haiku",
                    mode=mode,
                    benchmark_name=f"mode_test_{mode.value}"
                )

                assert result.success is True
                assert result.config.mode == mode
                assert len(result.questions) == config.sample_size
                assert len(result.responses) == config.sample_size

    @pytest.mark.asyncio
    async def test_benchmark_report_generation(self, test_config, mock_openrouter_client):
        """Test report generation after benchmark completion."""
        init_database(test_config)

        with patch('src.models.openrouter.OpenRouterClient', return_value=mock_openrouter_client):
            runner = BenchmarkRunner()

            # Run benchmark
            result = await runner.run_benchmark(
                model_name="openai/gpt-4",
                mode=RunMode.QUICK,
                benchmark_name="report_test"
            )

            assert result.success is True

            # Test report generation
            report_gen = ReportGenerator()

            # Test terminal report
            terminal_report = report_gen.generate_report(result, ReportFormat.TERMINAL)
            assert isinstance(terminal_report, str)
            assert len(terminal_report) > 0
            assert "Benchmark Results" in terminal_report

            # Test markdown report
            markdown_report = report_gen.generate_report(result, ReportFormat.MARKDOWN)
            assert isinstance(markdown_report, str)
            assert len(markdown_report) > 0
            assert "# Benchmark Report" in markdown_report

            # Test JSON report
            json_report = report_gen.generate_report(result, ReportFormat.JSON)
            assert isinstance(json_report, str)
            json_data = json.loads(json_report)
            assert "benchmark_id" in json_data
            assert "metrics" in json_data
            assert "model_name" in json_data

    @pytest.mark.asyncio
    async def test_benchmark_data_persistence(self, test_config, mock_openrouter_client):
        """Test that benchmark data is properly persisted to database."""
        init_database(test_config)

        with patch('src.models.openrouter.OpenRouterClient', return_value=mock_openrouter_client):
            runner = BenchmarkRunner()

            # Run benchmark
            result = await runner.run_benchmark(
                model_name="test-model",
                mode=RunMode.QUICK,
                benchmark_name="persistence_test"
            )

            assert result.success is True

            # Verify data persistence
            with get_session() as session:
                benchmark_repo = BenchmarkRepository(session)
                question_repo = QuestionRepository(session)
                response_repo = ResponseRepository(session)

                # Check benchmark record
                benchmark = benchmark_repo.get_benchmark_by_id(result.benchmark_id)
                assert benchmark is not None
                assert benchmark.name == "persistence_test"
                assert benchmark.status == "completed"
                assert benchmark.question_count == 5

                # Check questions
                questions = question_repo.get_questions_by_benchmark(result.benchmark_id)
                assert len(questions) == 5

                # Check responses
                responses = response_repo.get_responses_by_benchmark(result.benchmark_id)
                assert len(responses) == 5

                # Verify response data
                for response in responses:
                    assert response.model_name == "test-model"
                    assert response.response_text is not None
                    assert response.response_time_ms > 0
                    assert response.cost_usd >= 0

    @pytest.mark.asyncio
    async def test_concurrent_benchmark_execution(self, test_config, mock_openrouter_client):
        """Test running multiple benchmarks concurrently."""
        init_database(test_config)

        with patch('src.models.openrouter.OpenRouterClient', return_value=mock_openrouter_client):
            runner = BenchmarkRunner()

            # Run multiple benchmarks concurrently
            tasks = []
            model_names = ["model1", "model2", "model3"]

            for model in model_names:
                config = BenchmarkConfig(
                    mode=RunMode.QUICK,
                    sample_size=3,
                    timeout_seconds=30,
                    grading_mode=GradingMode.LENIENT,
                    save_results=True
                )

                task = runner.run_benchmark(
                    model_name=model,
                    mode=RunMode.QUICK,
                    custom_config=config,
                    benchmark_name=f"concurrent_test_{model}"
                )
                tasks.append(task)

            # Wait for all benchmarks to complete
            results = await asyncio.gather(*tasks)

            # Verify all completed successfully
            assert len(results) == 3
            for result in results:
                assert result.success is True
                assert len(result.questions) == 3
                assert len(result.responses) == 3

    @pytest.mark.asyncio
    async def test_benchmark_with_custom_grading_modes(self, test_config, mock_openrouter_client):
        """Test benchmark with different grading modes."""
        init_database(test_config)

        with patch('src.models.openrouter.OpenRouterClient', return_value=mock_openrouter_client):
            runner = BenchmarkRunner()

            grading_modes = [GradingMode.STRICT, GradingMode.LENIENT, GradingMode.JEOPARDY]

            for grading_mode in grading_modes:
                config = BenchmarkConfig(
                    mode=RunMode.QUICK,
                    sample_size=3,
                    timeout_seconds=30,
                    grading_mode=grading_mode,
                    save_results=True
                )

                result = await runner.run_benchmark(
                    model_name="test-model",
                    mode=RunMode.QUICK,
                    custom_config=config,
                    benchmark_name=f"grading_test_{grading_mode.value}"
                )

                assert result.success is True
                assert result.config.grading_mode == grading_mode
                assert len(result.graded_responses) == 3

                # Verify grading results are present
                for graded in result.graded_responses:
                    assert hasattr(graded, 'is_correct')
                    assert hasattr(graded, 'confidence')

    @pytest.mark.asyncio
    async def test_benchmark_workflow_with_large_dataset(self, test_config):
        """Test benchmark workflow with larger dataset simulation."""
        init_database(test_config)

        # Create mock client for larger dataset
        client = Mock(spec=OpenRouterClient)

        async def mock_batch_query_large(prompts, **kwargs):
            return [
                ModelResponse(
                    text=f"Large dataset answer {i}",
                    model_name="test-model",
                    response_time_ms=100 + (i % 10),
                    tokens_generated=5,
                    cost_usd=0.001,
                    metadata={"mock": True}
                )
                for i in range(len(prompts))
            ]

        client.batch_query = AsyncMock(side_effect=mock_batch_query_large)
        client.is_available.return_value = True

        with patch('src.models.openrouter.OpenRouterClient', return_value=client):
            runner = BenchmarkRunner()

            # Test with larger sample size
            config = BenchmarkConfig(
                mode=RunMode.CUSTOM,
                sample_size=50,  # Larger dataset
                timeout_seconds=60,
                grading_mode=GradingMode.LENIENT,
                save_results=True
            )

            result = await runner.run_benchmark(
                model_name="test-model",
                mode=RunMode.CUSTOM,
                custom_config=config,
                benchmark_name="large_dataset_test"
            )

            assert result.success is True
            assert len(result.questions) == 50
            assert len(result.responses) == 50
            assert len(result.graded_responses) == 50

            # Verify performance metrics
            assert result.execution_time > 0
            assert result.metrics is not None
            assert result.metadata['total_cost'] > 0

    @pytest.mark.asyncio
    async def test_benchmark_error_handling_comprehensive(self, test_config):
        """Comprehensive test of error handling scenarios."""
        init_database(test_config)

        # Test various error scenarios
        error_scenarios = [
            ("API Timeout", Exception("Request timeout")),
            ("API Rate Limit", Exception("Rate limit exceeded")),
            ("Invalid Model", Exception("Model not found")),
            ("Network Error", Exception("Connection failed")),
        ]

        for error_name, error in error_scenarios:
            client = Mock(spec=OpenRouterClient)
            client.batch_query = AsyncMock(side_effect=error)
            client.is_available.return_value = True

            with patch('src.models.openrouter.OpenRouterClient', return_value=client):
                runner = BenchmarkRunner()

                config = BenchmarkConfig(
                    mode=RunMode.QUICK,
                    sample_size=3,
                    timeout_seconds=30,
                    grading_mode=GradingMode.LENIENT,
                    save_results=False  # Don't save on expected failures
                )

                result = await runner.run_benchmark(
                    model_name="test-model",
                    mode=RunMode.QUICK,
                    custom_config=config,
                    benchmark_name=f"error_test_{error_name.lower().replace(' ', '_')}"
                )

                # Verify error handling
                assert result.success is False
                assert result.error_message is not None
                assert isinstance(result.error_message, str)
                assert result.execution_time > 0

                # Verify partial results are still captured
                assert result.benchmark_id > 0
                assert result.config is not None
                assert result.progress is not None