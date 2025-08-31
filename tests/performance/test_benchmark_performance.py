"""
Performance Tests for Benchmarking System

Tests system performance with large datasets, measures memory usage,
tests concurrent model benchmarking, and verifies rate limiting.
"""

import pytest
import asyncio
import time
import psutil
import os
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any
import gc
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
import threading

from src.benchmark.runner import BenchmarkRunner, RunMode, BenchmarkConfig
from src.benchmark.scheduler import BenchmarkScheduler
from src.core.config import AppConfig
from src.core.database import init_database
from src.models.openrouter import OpenRouterClient
from src.models.base import ModelResponse
from src.evaluation.metrics import ComprehensiveMetrics


class PerformanceMetrics:
    """Helper class to track performance metrics."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_usage = []
        self.cpu_usage = []

    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        tracemalloc.start()
        self._start_resource_monitoring()

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.end_time = time.time()
        self._stop_resource_monitoring()

    def _start_resource_monitoring(self):
        """Start monitoring system resources."""
        def monitor_resources():
            process = psutil.Process(os.getpid())
            while self.start_time and not self.end_time:
                try:
                    self.memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
                    self.cpu_usage.append(process.cpu_percent(interval=0.1))
                    time.sleep(0.5)
                except:
                    break

        self.monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        self.monitor_thread.start()

    def _stop_resource_monitoring(self):
        """Stop resource monitoring."""
        time.sleep(0.1)  # Allow final measurements

    @property
    def execution_time(self):
        """Get total execution time."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0

    @property
    def peak_memory_usage(self):
        """Get peak memory usage in MB."""
        return max(self.memory_usage) if self.memory_usage else 0

    @property
    def average_memory_usage(self):
        """Get average memory usage in MB."""
        return sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0

    @property
    def average_cpu_usage(self):
        """Get average CPU usage percentage."""
        return sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0


class TestBenchmarkPerformance:
    """Performance tests for the benchmarking system."""

    @pytest.fixture
    def test_config(self, temp_dir):
        """Test configuration for performance tests."""
        return AppConfig(
            name="Performance Test Jeopardy Benchmark",
            version="perf-test",
            debug=False,  # Disable debug for performance
            database=AppConfig.DatabaseConfig(
                url=f"sqlite:///{temp_dir}/perf_test.db",
                echo=False
            ),
            logging=AppConfig.LoggingConfig(
                level="WARNING",  # Reduce logging for performance
                file=str(temp_dir / "perf_test.log")
            ),
            benchmarks=AppConfig.BenchmarkConfig(
                default_sample_size=100,
                max_concurrent_requests=5
            )
        )

    @pytest.fixture
    def large_mock_responses(self):
        """Generate mock responses for large dataset testing."""
        return [
            ModelResponse(
                text=f"Performance test answer {i}",
                model_name="perf-test-model",
                response_time_ms=100 + (i % 50),  # Vary response times
                tokens_generated=10,
                cost_usd=0.001,
                metadata={"mock": True, "index": i}
            )
            for i in range(200)  # Large dataset
        ]

    @pytest.fixture
    def mock_openrouter_client(self, large_mock_responses):
        """Mock OpenRouter client with performance characteristics."""
        client = Mock(spec=OpenRouterClient)

        async def mock_batch_query(prompts, **kwargs):
            # Simulate realistic delays
            await asyncio.sleep(0.01)  # Small delay per batch
            responses = []
            for i, prompt in enumerate(prompts):
                if i < len(large_mock_responses):
                    response = large_mock_responses[i]
                else:
                    response = ModelResponse(
                        text=f"Extra answer {i}",
                        model_name="perf-test-model",
                        response_time_ms=100,
                        tokens_generated=10,
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
        # Force garbage collection before each test
        gc.collect()
        tracemalloc.reset_peak()

    def teardown_method(self):
        """Clean up after test."""
        gc.collect()

    @pytest.mark.performance
    @pytest.mark.slow
    def test_large_dataset_performance(self, test_config, mock_openrouter_client):
        """Test performance with large dataset (200 questions)."""
        init_database(test_config)

        metrics = PerformanceMetrics()
        metrics.start_monitoring()

        try:
            with patch('src.models.openrouter.OpenRouterClient', return_value=mock_openrouter_client):
                runner = BenchmarkRunner()

                config = BenchmarkConfig(
                    mode=RunMode.CUSTOM,
                    sample_size=200,
                    timeout_seconds=60,
                    grading_mode=GradingMode.LENIENT,
                    save_results=True
                )

                result = asyncio.run(runner.run_benchmark(
                    model_name="openai/gpt-4",
                    mode=RunMode.CUSTOM,
                    custom_config=config,
                    benchmark_name="large_dataset_perf_test"
                ))

                # Verify successful completion
                assert result.success is True
                assert len(result.questions) == 200
                assert len(result.responses) == 200
                assert len(result.graded_responses) == 200

                # Performance assertions
                assert result.execution_time < 30.0  # Should complete within 30 seconds
                assert result.execution_time > 0

                # Memory usage should be reasonable
                peak_memory = metrics.peak_memory_usage
                assert peak_memory < 500  # Less than 500MB peak memory

        finally:
            metrics.stop_monitoring()

        # Log performance results
        print(f"\nLarge Dataset Performance Results:")
        print(f"Execution time: {metrics.execution_time:.2f}s")
        print(f"Peak memory: {metrics.peak_memory_usage:.2f}MB")
        print(f"Average memory: {metrics.average_memory_usage:.2f}MB")
        print(f"Average CPU: {metrics.average_cpu_usage:.2f}%")

    @pytest.mark.performance
    def test_memory_usage_scaling(self, test_config, mock_openrouter_client):
        """Test how memory usage scales with dataset size."""
        init_database(test_config)

        dataset_sizes = [50, 100, 150]
        memory_usage = []

        for size in dataset_sizes:
            gc.collect()  # Clean up before each run
            tracemalloc.reset_peak()

            metrics = PerformanceMetrics()
            metrics.start_monitoring()

            try:
                with patch('src.models.openrouter.OpenRouterClient', return_value=mock_openrouter_client):
                    runner = BenchmarkRunner()

                    config = BenchmarkConfig(
                        mode=RunMode.CUSTOM,
                        sample_size=size,
                        timeout_seconds=30,
                        grading_mode=GradingMode.LENIENT,
                        save_results=False  # Don't save to reduce overhead
                    )

                    result = asyncio.run(runner.run_benchmark(
                        model_name="openai/gpt-3.5-turbo",
                        mode=RunMode.CUSTOM,
                        custom_config=config,
                        benchmark_name=f"memory_test_{size}"
                    ))

                    assert result.success is True
                    memory_usage.append(metrics.peak_memory_usage)

            finally:
                metrics.stop_monitoring()

        # Verify memory usage scales reasonably
        # Memory should increase but not exponentially
        assert memory_usage[1] < memory_usage[0] * 3  # Less than 3x increase for 2x data
        assert memory_usage[2] < memory_usage[1] * 2  # Less than 2x increase for 1.5x data

        print("
Memory Scaling Results:")
        for i, size in enumerate(dataset_sizes):
            print(f"Size {size}: {memory_usage[i]:.2f}MB")

    @pytest.mark.performance
    def test_concurrent_benchmark_performance(self, test_config):
        """Test performance of concurrent benchmark execution."""
        init_database(test_config)

        # Create multiple mock clients
        clients = []
        for i in range(5):
            client = Mock(spec=OpenRouterClient)

            async def mock_batch_query(prompts, client_id=i, **kwargs):
                await asyncio.sleep(0.005)  # Small delay
                return [
                    ModelResponse(
                        text=f"Concurrent answer {client_id}_{j}",
                        model_name=f"model_{client_id}",
                        response_time_ms=100,
                        tokens_generated=10,
                        cost_usd=0.001,
                        metadata={"mock": True, "client_id": client_id}
                    )
                    for j in range(len(prompts))
                ]

            client.batch_query = AsyncMock(side_effect=mock_batch_query)
            client.is_available.return_value = True
            clients.append(client)

        metrics = PerformanceMetrics()
        metrics.start_monitoring()

        try:
            # Run concurrent benchmarks
            tasks = []
            for i, client in enumerate(clients):
                with patch('src.models.openrouter.OpenRouterClient', return_value=client):
                    runner = BenchmarkRunner()

                    config = BenchmarkConfig(
                        mode=RunMode.QUICK,
                        sample_size=20,
                        timeout_seconds=30,
                        grading_mode=GradingMode.LENIENT,
                        save_results=True
                    )

                    task = runner.run_benchmark(
                        model_name=f"concurrent_model_{i}",
                        mode=RunMode.QUICK,
                        custom_config=config,
                        benchmark_name=f"concurrent_test_{i}"
                    )
                    tasks.append(task)

            # Execute all concurrently
            results = asyncio.run(asyncio.gather(*tasks))

            # Verify all completed successfully
            assert len(results) == 5
            for result in results:
                assert result.success is True
                assert len(result.questions) == 20

        finally:
            metrics.stop_monitoring()

        # Performance assertions for concurrent execution
        total_time = metrics.execution_time
        assert total_time < 15.0  # Should complete within 15 seconds

        print("
Concurrent Benchmark Performance:")
        print(f"Total execution time: {total_time:.2f}s")
        print(f"Average time per benchmark: {total_time/5:.2f}s")
        print(f"Peak memory: {metrics.peak_memory_usage:.2f}MB")

    @pytest.mark.performance
    def test_rate_limiting_performance(self, test_config):
        """Test performance with rate limiting."""
        init_database(test_config)

        # Mock client with rate limiting simulation
        client = Mock(spec=OpenRouterClient)
        request_count = 0
        rate_limit_hits = 0

        async def mock_batch_query_with_rate_limiting(prompts, **kwargs):
            nonlocal request_count, rate_limit_hits
            request_count += 1

            # Simulate rate limiting every 10 requests
            if request_count % 10 == 0:
                rate_limit_hits += 1
                await asyncio.sleep(0.1)  # Rate limit delay

            await asyncio.sleep(0.01)  # Normal processing delay
            return [
                ModelResponse(
                    text=f"Rate limit test answer {i}",
                    model_name="rate-limit-test-model",
                    response_time_ms=100 + (rate_limit_hits * 50),
                    tokens_generated=10,
                    cost_usd=0.001,
                    metadata={"mock": True, "rate_limited": rate_limit_hits > 0}
                )
                for i in range(len(prompts))
            ]

        client.batch_query = AsyncMock(side_effect=mock_batch_query_with_rate_limiting)
        client.is_available.return_value = True

        metrics = PerformanceMetrics()
        metrics.start_monitoring()

        try:
            with patch('src.models.openrouter.OpenRouterClient', return_value=client):
                runner = BenchmarkRunner()

                config = BenchmarkConfig(
                    mode=RunMode.CUSTOM,
                    sample_size=100,
                    timeout_seconds=120,  # Longer timeout for rate limiting
                    grading_mode=GradingMode.LENIENT,
                    save_results=True
                )

                result = asyncio.run(runner.run_benchmark(
                    model_name="openai/gpt-4",
                    mode=RunMode.CUSTOM,
                    custom_config=config,
                    benchmark_name="rate_limiting_test"
                ))

                assert result.success is True
                assert len(result.questions) == 100
                assert len(result.responses) == 100

                # Should have experienced some rate limiting
                assert rate_limit_hits > 0

        finally:
            metrics.stop_monitoring()

        print("
Rate Limiting Performance:")
        print(f"Total execution time: {metrics.execution_time:.2f}s")
        print(f"Rate limit hits: {rate_limit_hits}")
        print(f"Average response time: {sum(r.latency_ms or 0 for r in result.responses) / len(result.responses):.2f}ms")

    @pytest.mark.performance
    def test_database_performance_under_load(self, test_config, mock_openrouter_client):
        """Test database performance with high-frequency operations."""
        init_database(test_config)

        metrics = PerformanceMetrics()
        metrics.start_monitoring()

        try:
            with patch('src.models.openrouter.OpenRouterClient', return_value=mock_openrouter_client):
                # Run multiple benchmarks in sequence to stress the database
                runner = BenchmarkRunner()

                for i in range(10):
                    config = BenchmarkConfig(
                        mode=RunMode.QUICK,
                        sample_size=20,
                        timeout_seconds=30,
                        grading_mode=GradingMode.LENIENT,
                        save_results=True
                    )

                    result = asyncio.run(runner.run_benchmark(
                        model_name="openai/gpt-3.5-turbo",
                        mode=RunMode.QUICK,
                        custom_config=config,
                        benchmark_name=f"db_load_test_{i}"
                    ))

                    assert result.success is True

        finally:
            metrics.stop_monitoring()

        # Database should handle the load without excessive memory growth
        assert metrics.peak_memory_usage < 300  # MB
        assert metrics.execution_time < 60.0  # Should complete within 1 minute

        print("
Database Load Performance:")
        print(f"Total execution time: {metrics.execution_time:.2f}s")
        print(f"Peak memory: {metrics.peak_memory_usage:.2f}MB")
        print(f"Average memory: {metrics.average_memory_usage:.2f}MB")

    @pytest.mark.performance
    def test_resource_cleanup_performance(self, test_config, mock_openrouter_client):
        """Test that resources are properly cleaned up after benchmarks."""
        init_database(test_config)

        # Track memory before benchmark
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024

        with patch('src.models.openrouter.OpenRouterClient', return_value=mock_openrouter_client):
            runner = BenchmarkRunner()

            config = BenchmarkConfig(
                mode=RunMode.STANDARD,
                sample_size=50,
                timeout_seconds=60,
                grading_mode=GradingMode.LENIENT,
                save_results=True
            )

            result = asyncio.run(runner.run_benchmark(
                model_name="openai/gpt-4",
                mode=RunMode.STANDARD,
                custom_config=config,
                benchmark_name="cleanup_test"
            ))

            assert result.success is True

        # Force garbage collection
        gc.collect()

        # Check memory after cleanup
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_increase = memory_after - memory_before

        # Memory increase should be reasonable and not growing unbounded
        assert memory_increase < 50  # Less than 50MB permanent increase

        print("
Resource Cleanup Performance:")
        print(f"Memory before: {memory_before:.2f}MB")
        print(f"Memory after: {memory_after:.2f}MB")
        print(f"Memory increase: {memory_increase:.2f}MB")

    @pytest.mark.performance
    @pytest.mark.slow
    def test_endurance_performance(self, test_config):
        """Test system endurance with prolonged operation."""
        init_database(test_config)

        # Create a client that can handle many requests
        client = Mock(spec=OpenRouterClient)

        async def mock_batch_query_endurance(prompts, **kwargs):
            await asyncio.sleep(0.005)  # Very small delay
            return [
                ModelResponse(
                    text=f"Endurance answer {i}",
                    model_name="endurance-test-model",
                    response_time_ms=50,  # Fast responses
                    tokens_generated=8,
                    cost_usd=0.0005,
                    metadata={"mock": True}
                )
                for i in range(len(prompts))
            ]

        client.batch_query = AsyncMock(side_effect=mock_batch_query_endurance)
        client.is_available.return_value = True

        metrics = PerformanceMetrics()
        metrics.start_monitoring()

        try:
            with patch('src.models.openrouter.OpenRouterClient', return_value=client):
                runner = BenchmarkRunner()

                # Run a large benchmark to test endurance
                config = BenchmarkConfig(
                    mode=RunMode.CUSTOM,
                    sample_size=500,  # Large sample for endurance test
                    timeout_seconds=300,  # 5 minute timeout
                    grading_mode=GradingMode.LENIENT,
                    save_results=True
                )

                result = asyncio.run(runner.run_benchmark(
                    model_name="openai/gpt-4",
                    mode=RunMode.CUSTOM,
                    custom_config=config,
                    benchmark_name="endurance_test"
                ))

                assert result.success is True
                assert len(result.questions) == 500
                assert len(result.responses) == 500

        finally:
            metrics.stop_monitoring()

        # Endurance test should complete within reasonable time
        assert metrics.execution_time < 120.0  # Less than 2 minutes
        assert metrics.peak_memory_usage < 600  # Less than 600MB

        print("
Endurance Performance Results:")
        print(f"Total execution time: {metrics.execution_time:.2f}s")
        print(f"Peak memory: {metrics.peak_memory_usage:.2f}MB")
        print(f"Average CPU: {metrics.average_cpu_usage:.2f}%")
        print(f"Questions processed: 500")
        print(f"Throughput: {500 / metrics.execution_time:.2f} questions/second")

    @pytest.mark.performance
    def test_concurrent_scheduler_performance(self, test_config):
        """Test performance of the benchmark scheduler with concurrent execution."""
        init_database(test_config)

        # Create mock clients for different models
        model_configs = {}
        for i in range(3):
            client = Mock(spec=OpenRouterClient)

            async def mock_batch_query_scheduler(prompts, model_id=i, **kwargs):
                await asyncio.sleep(0.01)
                return [
                    ModelResponse(
                        text=f"Scheduler answer model_{model_id}_{j}",
                        model_name=f"scheduler_model_{model_id}",
                        response_time_ms=100,
                        tokens_generated=10,
                        cost_usd=0.001,
                        metadata={"mock": True, "model_id": model_id}
                    )
                    for j in range(len(prompts))
                ]

            client.batch_query = AsyncMock(side_effect=mock_batch_query_scheduler)
            client.is_available.return_value = True
            model_configs[f"model_{i}"] = client

        def get_client_for_model(model_name):
            model_id = int(model_name.split('_')[1])
            return model_configs[model_name]

        metrics = PerformanceMetrics()
        metrics.start_monitoring()

        try:
            # Mock the client creation to return appropriate client per model
            with patch('src.models.openrouter.OpenRouterClient') as mock_client_class:
                mock_client_class.side_effect = lambda config: get_client_for_model(config.model_name)

                scheduler = BenchmarkScheduler()

                # Schedule multiple models
                model_list = [f"model_{i}" for i in range(3)]
                scheduled_ids = scheduler.schedule_multiple(
                    models=model_list,
                    mode=RunMode.QUICK,
                    concurrent_limit=3,
                    benchmark_name_prefix="scheduler_perf_test"
                )

                assert len(scheduled_ids) == 3

                # Run all scheduled benchmarks
                results = asyncio.run(scheduler.run_scheduled_benchmarks())

                assert len(results) == 3
                for result in results.values():
                    assert result.success is True
                    assert len(result.questions) == 50  # Default quick mode size

        finally:
            metrics.stop_monitoring()

        print("
Scheduler Performance Results:")
        print(f"Total execution time: {metrics.execution_time:.2f}s")
        print(f"Models processed: 3")
        print(f"Average time per model: {metrics.execution_time/3:.2f}s")
        print(f"Peak memory: {metrics.peak_memory_usage:.2f}MB")