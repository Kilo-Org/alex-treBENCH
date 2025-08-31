"""
End-to-End Tests for CLI Commands

Tests all CLI commands work correctly, including command combinations,
workflows, output formats, and error handling for invalid inputs.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from click.testing import CliRunner
import json
import sys
import os

from src.main import cli, benchmark
from src.core.config import AppConfig
from src.core.database import init_database
from src.models.openrouter import OpenRouterClient
from src.models.base import ModelResponse


class TestCLICommands:
    """Test CLI commands end-to-end."""

    @pytest.fixture
    def runner(self):
        """CLI runner for testing."""
        return CliRunner()

    @pytest.fixture
    def test_config(self, temp_dir):
        """Test configuration for CLI tests."""
        return AppConfig(
            name="CLI Test Jeopardy Benchmark",
            version="cli-test",
            debug=True,
            database=AppConfig.DatabaseConfig(
                url=f"sqlite:///{temp_dir}/cli_test.db",
                echo=False
            ),
            logging=AppConfig.LoggingConfig(
                level="DEBUG",
                file=str(temp_dir / "cli_test.log")
            ),
            benchmarks=AppConfig.BenchmarkConfig(
                default_sample_size=10,
                max_concurrent_requests=2
            )
        )

    @pytest.fixture
    def mock_openrouter_client(self):
        """Mock OpenRouter client for CLI tests."""
        client = Mock(spec=OpenRouterClient)

        async def mock_batch_query(prompts, **kwargs):
            return [
                ModelResponse(
                    text=f"CLI test answer {i}",
                    model_name="cli-test-model",
                    response_time_ms=100,
                    tokens_generated=5,
                    cost_usd=0.001,
                    metadata={"mock": True}
                )
                for i in range(len(prompts))
            ]

        client.batch_query = AsyncMock(side_effect=mock_batch_query)
        client.is_available.return_value = True
        client.get_pricing_info.return_value = {"input": 0.001, "output": 0.002}

        return client

    def setup_method(self):
        """Set up test environment."""
        # Ensure clean state
        pass

    def teardown_method(self):
        """Clean up after test."""
        pass

    def test_cli_help_command(self, runner):
        """Test CLI help command displays correctly."""
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert "Jeopardy Benchmarking System" in result.output
        assert "benchmark" in result.output
        assert "--help" in result.output
        assert "--verbose" in result.output

    def test_cli_benchmark_help(self, runner):
        """Test benchmark subcommand help."""
        result = runner.invoke(cli, ['benchmark', '--help'])
        assert result.exit_code == 0
        assert "benchmark management commands" in result.output.lower()
        assert "run" in result.output
        assert "compare" in result.output
        assert "history" in result.output
        assert "report" in result.output
        assert "list" in result.output

    @patch('src.models.openrouter.OpenRouterClient')
    def test_benchmark_run_basic(self, mock_client_class, runner, test_config, mock_openrouter_client):
        """Test basic benchmark run command."""
        mock_client_class.return_value = mock_openrouter_client

        # Initialize database
        init_database(test_config)

        with patch('src.core.config.get_config', return_value=test_config):
            result = runner.invoke(cli, [
                'benchmark', 'run',
                '--model', 'openai/gpt-3.5-turbo',
                '--size', 'quick'
            ])

            assert result.exit_code == 0
            assert "Starting quick benchmark" in result.output
            assert "✓ Benchmark completed successfully" in result.output
            assert "Benchmark ID:" in result.output

    @patch('src.models.openrouter.OpenRouterClient')
    def test_benchmark_run_with_custom_options(self, mock_client_class, runner, test_config, mock_openrouter_client):
        """Test benchmark run with custom options."""
        mock_client_class.return_value = mock_openrouter_client

        # Initialize database
        init_database(test_config)

        with patch('src.core.config.get_config', return_value=test_config):
            result = runner.invoke(cli, [
                'benchmark', 'run',
                '--model', 'openai/gpt-4',
                '--size', 'standard',
                '--name', 'Custom Test Benchmark',
                '--description', 'A test benchmark with custom settings',
                '--timeout', '45',
                '--grading-mode', 'strict',
                '--report-format', 'json'
            ])

            assert result.exit_code == 0
            assert "Starting standard benchmark" in result.output
            assert "Custom Test Benchmark" in result.output
            assert "Benchmark completed successfully" in result.output

    @patch('src.models.openrouter.OpenRouterClient')
    def test_benchmark_run_terminal_report(self, mock_client_class, runner, test_config, mock_openrouter_client):
        """Test benchmark run with terminal report format."""
        mock_client_class.return_value = mock_openrouter_client

        # Initialize database
        init_database(test_config)

        with patch('src.core.config.get_config', return_value=test_config):
            result = runner.invoke(cli, [
                'benchmark', 'run',
                '--model', 'anthropic/claude-3-haiku',
                '--size', 'quick',
                '--report-format', 'terminal'
            ])

            assert result.exit_code == 0
            # Terminal report should contain formatted output
            assert "Benchmark Results" in result.output or "Overall Score" in result.output

    @patch('src.models.openrouter.OpenRouterClient')
    def test_benchmark_run_markdown_report(self, mock_client_class, runner, test_config, mock_openrouter_client, temp_dir):
        """Test benchmark run with markdown report format."""
        mock_client_class.return_value = mock_openrouter_client

        # Initialize database
        init_database(test_config)

        report_file = temp_dir / "test_report.md"

        with patch('src.core.config.get_config', return_value=test_config):
            result = runner.invoke(cli, [
                'benchmark', 'run',
                '--model', 'openai/gpt-3.5-turbo',
                '--size', 'quick',
                '--report-format', 'markdown',
                '--output', str(report_file)
            ])

            assert result.exit_code == 0
            assert report_file.exists()

            # Check markdown content
            content = report_file.read_text()
            assert "# Benchmark Report" in content
            assert "Overall Score" in content

    @patch('src.models.openrouter.OpenRouterClient')
    def test_benchmark_run_json_report(self, mock_client_class, runner, test_config, mock_openrouter_client, temp_dir):
        """Test benchmark run with JSON report format."""
        mock_client_class.return_value = mock_openrouter_client

        # Initialize database
        init_database(test_config)

        report_file = temp_dir / "test_report.json"

        with patch('src.core.config.get_config', return_value=test_config):
            result = runner.invoke(cli, [
                'benchmark', 'run',
                '--model', 'openai/gpt-4',
                '--size', 'quick',
                '--report-format', 'json',
                '--output', str(report_file)
            ])

            assert result.exit_code == 0
            assert report_file.exists()

            # Check JSON content
            content = report_file.read_text()
            json_data = json.loads(content)
            assert "benchmark_id" in json_data
            assert "metrics" in json_data
            assert "model_name" in json_data

    @patch('src.models.openrouter.OpenRouterClient')
    def test_benchmark_compare_two_models(self, mock_client_class, runner, test_config, mock_openrouter_client):
        """Test comparing two models."""
        mock_client_class.return_value = mock_openrouter_client

        # Initialize database
        init_database(test_config)

        with patch('src.core.config.get_config', return_value=test_config):
            result = runner.invoke(cli, [
                'benchmark', 'compare',
                '--models', 'openai/gpt-3.5-turbo,openai/gpt-4',
                '--size', 'quick',
                '--concurrent-limit', '2'
            ])

            assert result.exit_code == 0
            assert "Comparing 2 models" in result.output
            assert "✓ Comparison complete" in result.output

    @patch('src.models.openrouter.OpenRouterClient')
    def test_benchmark_compare_multiple_models(self, mock_client_class, runner, test_config, mock_openrouter_client):
        """Test comparing multiple models."""
        mock_client_class.return_value = mock_openrouter_client

        # Initialize database
        init_database(test_config)

        with patch('src.core.config.get_config', return_value=test_config):
            result = runner.invoke(cli, [
                'benchmark', 'compare',
                '--models', 'openai/gpt-3.5-turbo,openai/gpt-4,anthropic/claude-3-haiku',
                '--size', 'quick',
                '--concurrent-limit', '3',
                '--report-format', 'markdown'
            ])

            assert result.exit_code == 0
            assert "Comparing 3 models" in result.output
            assert "models succeeded" in result.output

    @patch('src.models.openrouter.OpenRouterClient')
    def test_benchmark_history_command(self, mock_client_class, runner, test_config, mock_openrouter_client):
        """Test benchmark history command."""
        mock_client_class.return_value = mock_openrouter_client

        # Initialize database and create some benchmark history
        init_database(test_config)

        with patch('src.core.config.get_config', return_value=test_config):
            # Create a few benchmarks first
            for i in range(3):
                runner.invoke(cli, [
                    'benchmark', 'run',
                    '--model', 'openai/gpt-3.5-turbo',
                    '--size', 'quick',
                    '--name', f'History Test {i+1}'
                ])

            # Now test history command
            result = runner.invoke(cli, [
                'benchmark', 'history',
                '--model', 'openai/gpt-3.5-turbo',
                '--limit', '5'
            ])

            assert result.exit_code == 0
            assert "Benchmark History" in result.output
            assert "History Test" in result.output

    @patch('src.models.openrouter.OpenRouterClient')
    def test_benchmark_history_detailed(self, mock_client_class, runner, test_config, mock_openrouter_client):
        """Test benchmark history with detailed output."""
        mock_client_class.return_value = mock_openrouter_client

        # Initialize database
        init_database(test_config)

        with patch('src.core.config.get_config', return_value=test_config):
            # Create a benchmark
            runner.invoke(cli, [
                'benchmark', 'run',
                '--model', 'openai/gpt-4',
                '--size', 'quick',
                '--name', 'Detailed History Test'
            ])

            # Test detailed history
            result = runner.invoke(cli, [
                'benchmark', 'history',
                '--model', 'openai/gpt-4',
                '--detailed'
            ])

            assert result.exit_code == 0
            assert "Detailed History Test" in result.output

    @patch('src.models.openrouter.OpenRouterClient')
    def test_benchmark_list_command(self, mock_client_class, runner, test_config, mock_openrouter_client):
        """Test benchmark list command."""
        mock_client_class.return_value = mock_openrouter_client

        # Initialize database
        init_database(test_config)

        with patch('src.core.config.get_config', return_value=test_config):
            # Create some benchmarks
            models = ['openai/gpt-3.5-turbo', 'openai/gpt-4', 'anthropic/claude-3-haiku']
            for model in models:
                runner.invoke(cli, [
                    'benchmark', 'run',
                    '--model', model,
                    '--size', 'quick'
                ])

            # Test list command
            result = runner.invoke(cli, [
                'benchmark', 'list',
                '--limit', '10'
            ])

            assert result.exit_code == 0
            assert "Recent Benchmarks" in result.output
            assert len(result.output.split('\n')) > 5  # Should have multiple lines

    @patch('src.models.openrouter.OpenRouterClient')
    def test_benchmark_list_with_filters(self, mock_client_class, runner, test_config, mock_openrouter_client):
        """Test benchmark list with status and model filters."""
        mock_client_class.return_value = mock_openrouter_client

        # Initialize database
        init_database(test_config)

        with patch('src.core.config.get_config', return_value=test_config):
            # Create benchmarks with different models
            runner.invoke(cli, [
                'benchmark', 'run',
                '--model', 'openai/gpt-3.5-turbo',
                '--size', 'quick'
            ])
            runner.invoke(cli, [
                'benchmark', 'run',
                '--model', 'openai/gpt-4',
                '--size', 'quick'
            ])

            # Test filtering by model
            result = runner.invoke(cli, [
                'benchmark', 'list',
                '--model', 'openai/gpt-4'
            ])

            assert result.exit_code == 0
            assert "openai/gpt-4" in result.output

    @patch('src.models.openrouter.OpenRouterClient')
    def test_benchmark_report_command(self, mock_client_class, runner, test_config, mock_openrouter_client, temp_dir):
        """Test benchmark report command."""
        mock_client_class.return_value = mock_openrouter_client

        # Initialize database
        init_database(test_config)

        with patch('src.core.config.get_config', return_value=test_config):
            # Create a benchmark first
            run_result = runner.invoke(cli, [
                'benchmark', 'run',
                '--model', 'openai/gpt-3.5-turbo',
                '--size', 'quick',
                '--name', 'Report Test Benchmark'
            ])

            # Extract benchmark ID from output (this is a bit fragile but works for testing)
            output_lines = run_result.output.split('\n')
            benchmark_id = None
            for line in output_lines:
                if "Benchmark ID:" in line:
                    benchmark_id = line.split(":")[1].strip()
                    break

            assert benchmark_id is not None

            # Test report command
            result = runner.invoke(cli, [
                'benchmark', 'report',
                '--run-id', benchmark_id,
                '--format', 'terminal'
            ])

            assert result.exit_code == 0
            assert "Benchmark Summary" in result.output
            assert "Report Test Benchmark" in result.output

    @patch('src.models.openrouter.OpenRouterClient')
    def test_benchmark_report_markdown_format(self, mock_client_class, runner, test_config, mock_openrouter_client, temp_dir):
        """Test benchmark report in markdown format."""
        mock_client_class.return_value = mock_openrouter_client

        # Initialize database
        init_database(test_config)

        report_file = temp_dir / "cli_report.md"

        with patch('src.core.config.get_config', return_value=test_config):
            # Create a benchmark
            run_result = runner.invoke(cli, [
                'benchmark', 'run',
                '--model', 'openai/gpt-4',
                '--size', 'quick'
            ])

            # Get benchmark ID
            output_lines = run_result.output.split('\n')
            benchmark_id = None
            for line in output_lines:
                if "Benchmark ID:" in line:
                    benchmark_id = line.split(":")[1].strip()
                    break

            # Generate markdown report
            result = runner.invoke(cli, [
                'benchmark', 'report',
                '--run-id', benchmark_id,
                '--format', 'markdown',
                '--output', str(report_file)
            ])

            assert result.exit_code == 0
            assert report_file.exists()

            content = report_file.read_text()
            assert len(content) > 0
            assert "Benchmark" in content

    def test_cli_error_handling_invalid_model(self, runner, test_config):
        """Test error handling for invalid model."""
        init_database(test_config)

        with patch('src.core.config.get_config', return_value=test_config):
            result = runner.invoke(cli, [
                'benchmark', 'run',
                '--model', 'invalid/model/name',
                '--size', 'quick'
            ])

            # Should handle error gracefully
            assert result.exit_code == 1
            assert "Error" in result.output or "failed" in result.output.lower()

    def test_cli_error_handling_missing_model(self, runner, test_config):
        """Test error handling for missing model parameter."""
        init_database(test_config)

        with patch('src.core.config.get_config', return_value=test_config):
            result = runner.invoke(cli, [
                'benchmark', 'run',
                '--size', 'quick'
            ])

            assert result.exit_code == 2  # Click error for missing required parameter
            assert "Missing option" in result.output
            assert "--model" in result.output

    def test_cli_error_handling_invalid_size(self, runner, test_config):
        """Test error handling for invalid size parameter."""
        init_database(test_config)

        with patch('src.core.config.get_config', return_value=test_config):
            result = runner.invoke(cli, [
                'benchmark', 'run',
                '--model', 'openai/gpt-3.5-turbo',
                '--size', 'invalid_size'
            ])

            assert result.exit_code == 2  # Click validation error
            assert "Invalid value" in result.output

    def test_cli_error_handling_invalid_grading_mode(self, runner, test_config):
        """Test error handling for invalid grading mode."""
        init_database(test_config)

        with patch('src.core.config.get_config', return_value=test_config):
            result = runner.invoke(cli, [
                'benchmark', 'run',
                '--model', 'openai/gpt-3.5-turbo',
                '--size', 'quick',
                '--grading-mode', 'invalid_mode'
            ])

            assert result.exit_code == 2
            assert "Invalid value" in result.output

    def test_cli_error_handling_invalid_report_format(self, runner, test_config):
        """Test error handling for invalid report format."""
        init_database(test_config)

        with patch('src.core.config.get_config', return_value=test_config):
            result = runner.invoke(cli, [
                'benchmark', 'run',
                '--model', 'openai/gpt-3.5-turbo',
                '--size', 'quick',
                '--report-format', 'invalid_format'
            ])

            assert result.exit_code == 2
            assert "Invalid value" in result.output

    @patch('src.models.openrouter.OpenRouterClient')
    def test_cli_verbose_output(self, mock_client_class, runner, test_config, mock_openrouter_client):
        """Test CLI with verbose output."""
        mock_client_class.return_value = mock_openrouter_client

        # Initialize database
        init_database(test_config)

        with patch('src.core.config.get_config', return_value=test_config):
            result = runner.invoke(cli, [
                'benchmark', 'run',
                '--model', 'openai/gpt-3.5-turbo',
                '--size', 'quick',
                '--verbose'
            ])

            assert result.exit_code == 0
            # Verbose output should contain more detailed information
            assert len(result.output) > 100  # Rough check for verbose output

    @patch('src.models.openrouter.OpenRouterClient')
    def test_cli_debug_mode(self, mock_client_class, runner, test_config, mock_openrouter_client):
        """Test CLI in debug mode."""
        mock_client_class.return_value = mock_openrouter_client

        # Initialize database
        init_database(test_config)

        with patch('src.core.config.get_config', return_value=test_config):
            result = runner.invoke(cli, [
                'benchmark', 'run',
                '--model', 'openai/gpt-3.5-turbo',
                '--size', 'quick',
                '--debug'
            ])

            assert result.exit_code == 0
            assert "✓ Benchmark completed successfully" in result.output

    @patch('src.models.openrouter.OpenRouterClient')
    def test_cli_with_config_file(self, mock_client_class, runner, test_config, mock_openrouter_client, temp_dir):
        """Test CLI with custom config file."""
        mock_client_class.return_value = mock_openrouter_client

        # Create a temporary config file
        config_file = temp_dir / "test_config.yaml"
        config_content = """
name: "CLI Config Test"
version: "1.0.0"
debug: true
database:
  url: "sqlite:///test.db"
  echo: false
logging:
  level: "INFO"
  file: "test.log"
benchmarks:
  default_sample_size: 5
  max_concurrent_requests: 2
"""
        config_file.write_text(config_content)

        # Initialize database
        init_database(test_config)

        with patch('src.core.config.reload_config') as mock_reload:
            mock_reload.return_value = test_config

            result = runner.invoke(cli, [
                'benchmark', 'run',
                '--config', str(config_file),
                '--model', 'openai/gpt-3.5-turbo',
                '--size', 'quick'
            ])

            assert result.exit_code == 0
            mock_reload.assert_called_once()

    def test_cli_version_display(self, runner):
        """Test CLI version display."""
        result = runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        # Version should be displayed
        assert len(result.output.strip()) > 0

    @patch('src.models.openrouter.OpenRouterClient')
    def test_cli_command_combination_workflow(self, mock_client_class, runner, test_config, mock_openrouter_client):
        """Test a complete workflow combining multiple CLI commands."""
        mock_client_class.return_value = mock_openrouter_client

        # Initialize database
        init_database(test_config)

        with patch('src.core.config.get_config', return_value=test_config):
            # Step 1: Run a benchmark
            run_result = runner.invoke(cli, [
                'benchmark', 'run',
                '--model', 'openai/gpt-3.5-turbo',
                '--size', 'quick',
                '--name', 'Workflow Test Benchmark'
            ])
            assert run_result.exit_code == 0

            # Step 2: List benchmarks to verify it was created
            list_result = runner.invoke(cli, ['benchmark', 'list'])
            assert list_result.exit_code == 0
            assert "Workflow Test Benchmark" in list_result.output

            # Step 3: Check history for the model
            history_result = runner.invoke(cli, [
                'benchmark', 'history',
                '--model', 'openai/gpt-3.5-turbo'
            ])
            assert history_result.exit_code == 0
            assert "Workflow Test Benchmark" in history_result.output

            # Step 4: Run another benchmark for comparison
            run2_result = runner.invoke(cli, [
                'benchmark', 'run',
                '--model', 'openai/gpt-4',
                '--size', 'quick',
                '--name', 'Workflow Comparison Benchmark'
            ])
            assert run2_result.exit_code == 0

            # Step 5: Compare the two models
            compare_result = runner.invoke(cli, [
                'benchmark', 'compare',
                '--models', 'openai/gpt-3.5-turbo,openai/gpt-4',
                '--size', 'quick'
            ])
            assert compare_result.exit_code == 0
            assert "Comparing 2 models" in compare_result.output

    @patch('src.models.openrouter.OpenRouterClient')
    def test_cli_output_formats_comprehensive(self, mock_client_class, runner, test_config, mock_openrouter_client, temp_dir):
        """Test all output formats work correctly."""
        mock_client_class.return_value = mock_openrouter_client

        # Initialize database
        init_database(test_config)

        formats = ['terminal', 'markdown', 'json']
        files = {}

        with patch('src.core.config.get_config', return_value=test_config):
            for fmt in formats:
                output_file = temp_dir / f"comprehensive_test.{fmt}"
                files[fmt] = output_file

                result = runner.invoke(cli, [
                    'benchmark', 'run',
                    '--model', 'openai/gpt-3.5-turbo',
                    '--size', 'quick',
                    '--report-format', fmt,
                    '--output', str(output_file)
                ])

                assert result.exit_code == 0
                assert output_file.exists()

                # Verify content based on format
                content = output_file.read_text()
                if fmt == 'json':
                    json_data = json.loads(content)
                    assert isinstance(json_data, dict)
                elif fmt == 'markdown':
                    assert '#' in content or '*' in content
                # Terminal format is just text output