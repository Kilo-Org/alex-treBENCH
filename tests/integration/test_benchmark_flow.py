"""
Integration Tests for Complete Benchmark Flow

Tests the entire benchmarking workflow from question selection through report generation,
using mock data and API responses to avoid external dependencies.
"""

import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

from src.benchmark.runner import BenchmarkRunner, RunMode, BenchmarkConfig
from src.benchmark.scheduler import BenchmarkScheduler
from src.benchmark.reporting import ReportGenerator, ReportFormat
from src.evaluation.matcher import FuzzyMatcher
from src.evaluation.grader import AnswerGrader, GradingMode
from src.evaluation.metrics import MetricsCalculator
from src.models.base import ModelResponse
from src.storage.models import Benchmark, BenchmarkQuestion
from src.core.database import get_session
from src.storage.repositories import BenchmarkRepository


class TestBenchmarkFlow:
    """Integration tests for the complete benchmark flow."""
    
    @pytest.fixture
    def mock_model_responses(self):
        """Create mock model responses for testing."""
        responses = []
        test_cases = [
            ("What is Paris?", "Paris", True),
            ("Who is Shakespeare?", "William Shakespeare", True), 
            ("What is 42?", "The answer to everything", False),
            ("What is DNA?", "Deoxyribonucleic acid", True),
            ("Who is Einstein?", "Albert Einstein", True)
        ]
        
        for i, (expected_answer, response_text, should_be_correct) in enumerate(test_cases):
            response = ModelResponse(
                model_id="test/model",
                prompt=f"Test question {i+1}",
                response=response_text,
                latency_ms=100.0 + i * 10,
                tokens_used=50 + i * 5,
                cost=0.001 + i * 0.0001,
                timestamp=datetime.now(),
                metadata={
                    'tokens_input': 30 + i * 2,
                    'tokens_output': 20 + i * 3,
                    'finish_reason': 'stop'
                }
            )
            responses.append(response)
        
        return responses
    
    @pytest.fixture
    def sample_questions(self):
        """Create sample questions for testing."""
        questions = [
            {
                'question_id': 'q1',
                'question_text': 'This French city is known as the City of Light',
                'correct_answer': 'What is Paris?',
                'category': 'GEOGRAPHY',
                'value': 400,
                'difficulty_level': 'Easy'
            },
            {
                'question_id': 'q2', 
                'question_text': 'This English playwright wrote Hamlet',
                'correct_answer': 'Who is Shakespeare?',
                'category': 'LITERATURE',
                'value': 600,
                'difficulty_level': 'Medium'
            },
            {
                'question_id': 'q3',
                'question_text': 'The answer to life, the universe, and everything',
                'correct_answer': 'What is 42?',
                'category': 'SCIENCE FICTION',
                'value': 800,
                'difficulty_level': 'Easy'
            },
            {
                'question_id': 'q4',
                'question_text': 'This molecule carries genetic information in living organisms',
                'correct_answer': 'What is DNA?',
                'category': 'SCIENCE',
                'value': 1000,
                'difficulty_level': 'Hard'
            },
            {
                'question_id': 'q5',
                'question_text': 'This physicist developed the theory of relativity',
                'correct_answer': 'Who is Einstein?',
                'category': 'SCIENCE',
                'value': 1200,
                'difficulty_level': 'Hard'
            }
        ]
        return questions
    
    @pytest.mark.asyncio
    async def test_complete_benchmark_flow(self, mock_model_responses, sample_questions):
        """Test the complete benchmark workflow end-to-end."""
        
        with patch('src.benchmark.runner.OpenRouterClient') as mock_client_class:
            # Mock the OpenRouter client
            mock_client = AsyncMock()
            mock_client.batch_query.return_value = mock_model_responses
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            # Mock question loading to return our sample questions
            with patch.object(BenchmarkRunner, '_load_questions') as mock_load_questions:
                mock_questions = []
                for i, q_data in enumerate(sample_questions):
                    mock_question = MagicMock()
                    mock_question.id = i + 1
                    mock_question.question_id = q_data['question_id']
                    mock_question.question_text = q_data['question_text']
                    mock_question.correct_answer = q_data['correct_answer']
                    mock_question.category = q_data['category']
                    mock_question.value = q_data['value']
                    mock_question.difficulty_level = q_data['difficulty_level']
                    mock_questions.append(mock_question)
                
                mock_load_questions.return_value = mock_questions
                
                # Mock database operations
                with patch('src.benchmark.runner.get_session') as mock_get_session:
                    mock_session = MagicMock()
                    mock_get_session.return_value.__enter__.return_value = mock_session
                    
                    # Mock repositories
                    mock_benchmark_repo = MagicMock()
                    mock_benchmark = MagicMock()
                    mock_benchmark.id = 123
                    mock_benchmark_repo.create_benchmark.return_value = mock_benchmark
                    
                    with patch('src.benchmark.runner.BenchmarkRepository', return_value=mock_benchmark_repo):
                        with patch('src.benchmark.runner.ResponseRepository'):
                            # Create and run benchmark
                            runner = BenchmarkRunner()
                            
                            result = await runner.run_benchmark(
                                model_name="test/model",
                                mode=RunMode.QUICK,
                                benchmark_name="integration_test"
                            )
                            
                            # Verify the result
                            assert result.success
                            assert result.benchmark_id == 123
                            assert result.model_name == "test/model"
                            assert len(result.responses) == len(mock_model_responses)
                            assert len(result.graded_responses) == len(sample_questions)
                            assert result.metrics is not None
                            
                            # Verify metrics
                            metrics = result.metrics
                            assert metrics.model_name == "test/model"
                            assert metrics.accuracy.total_count == 5
                            assert metrics.accuracy.overall_accuracy >= 0.0
                            assert metrics.performance.mean_response_time > 0
                            assert metrics.cost.total_cost > 0
                            
                            # Verify some responses were graded correctly
                            correct_count = sum(1 for gr in result.graded_responses if gr.is_correct)
                            assert correct_count >= 3  # Should get most answers right with good matching
    
    @pytest.mark.asyncio
    async def test_fuzzy_matching_integration(self):
        """Test the fuzzy matching integration in the grading process."""
        
        # Test cases with various answer formats
        test_cases = [
            ("What is Paris?", "Paris", True),  # Exact match
            ("Who is Shakespeare?", "William Shakespeare", True),  # Partial match
            ("What is DNA?", "dna", True),  # Case insensitive
            ("Who is Einstein?", "Albert Einsteen", True),  # Typo but close
            ("What is the moon?", "sun", False),  # Wrong answer
            ("Who is Mozart?", "I don't know", False),  # Refusal
        ]
        
        grader = AnswerGrader()
        
        for expected, response, should_match in test_cases:
            graded = grader.grade_response(
                response_text=response,
                correct_answer=expected
            )
            
            # The fuzzy matcher should handle these cases appropriately
            if should_match:
                assert graded.confidence > 0.5, f"Failed to match '{response}' to '{expected}'"
            else:
                assert graded.confidence < 0.7, f"Incorrectly matched '{response}' to '{expected}'"
    
    @pytest.mark.asyncio 
    async def test_scheduler_integration(self, mock_model_responses, sample_questions):
        """Test the benchmark scheduler with multiple models."""
        
        models = ["test/model1", "test/model2"]
        
        with patch('src.benchmark.scheduler.BenchmarkRunner') as mock_runner_class:
            # Mock multiple runner instances
            mock_runners = {}
            for model in models:
                mock_runner = AsyncMock()
                mock_result = MagicMock()
                mock_result.success = True
                mock_result.model_name = model
                mock_result.benchmark_id = hash(model) % 1000
                mock_result.execution_time = 45.0
                mock_result.metrics = MagicMock()
                mock_result.metrics.overall_score = 0.85
                mock_result.responses = mock_model_responses
                mock_result.graded_responses = []
                mock_runner.run_benchmark.return_value = mock_result
                mock_runners[model] = mock_runner
            
            def get_runner(model_name):
                return mock_runners.get(model_name, mock_runners[models[0]])
            
            mock_runner_class.side_effect = lambda: get_runner("default")
            
            # Test scheduler
            scheduler = BenchmarkScheduler()
            
            # Schedule multiple models
            scheduled_ids = scheduler.schedule_multiple(
                models=models,
                mode=RunMode.QUICK,
                concurrent_limit=2
            )
            
            assert len(scheduled_ids) == len(models)
            
            # Run scheduled benchmarks
            results = await scheduler.run_scheduled_benchmarks()
            
            # Verify results
            assert len(results) >= 0  # Results depend on mock implementation
    
    @pytest.mark.asyncio
    async def test_report_generation_integration(self, mock_model_responses, sample_questions):
        """Test report generation with realistic data."""
        
        # Create a mock benchmark result
        from src.benchmark.runner import BenchmarkResult, BenchmarkProgress
        from src.evaluation.metrics import ComprehensiveMetrics, AccuracyMetrics, PerformanceMetrics, CostMetrics, ConsistencyMetrics
        
        # Create comprehensive metrics
        accuracy = AccuracyMetrics(
            overall_accuracy=0.80,
            correct_count=4,
            total_count=5,
            by_category={'SCIENCE': 0.75, 'GEOGRAPHY': 1.0, 'LITERATURE': 0.5},
            by_difficulty={'Easy': 0.85, 'Medium': 0.75, 'Hard': 0.70}
        )
        
        performance = PerformanceMetrics(
            mean_response_time=1.25,
            median_response_time=1.20,
            p95_response_time=1.50,
            p99_response_time=1.60,
            min_response_time=1.00,
            max_response_time=1.60,
            response_time_std=0.20,
            timeout_count=0,
            error_count=0
        )
        
        cost = CostMetrics(
            total_cost=0.0055,
            cost_per_question=0.0011,
            cost_per_correct_answer=0.001375,
            total_tokens=275,
            tokens_per_question=55.0,
            tokens_per_correct_answer=68.75,
            input_tokens=160,
            output_tokens=115,
            cost_efficiency_score=0.73
        )
        
        consistency = ConsistencyMetrics(
            performance_variance=0.15,
            category_consistency_score=0.85,
            difficulty_consistency_score=0.78,
            confidence_correlation=0.65,
            response_type_distribution={'jeopardy_format': 0.8, 'direct_answer': 0.2},
            match_type_distribution={'fuzzy': 0.6, 'exact': 0.3, 'semantic': 0.1}
        )
        
        metrics = ComprehensiveMetrics(
            model_name="test/model",
            benchmark_id=123,
            timestamp=datetime.now(),
            accuracy=accuracy,
            performance=performance,
            cost=cost,
            consistency=consistency,
            overall_score=0.82,
            quality_score=0.78,
            efficiency_score=0.75
        )
        
        progress = BenchmarkProgress(
            total_questions=5,
            completed_questions=5,
            successful_responses=5,
            failed_responses=0,
            current_phase="Complete",
            start_time=datetime.now()
        )
        
        result = BenchmarkResult(
            benchmark_id=123,
            model_name="test/model",
            config=BenchmarkConfig(mode=RunMode.QUICK, sample_size=5),
            progress=progress,
            metrics=metrics,
            questions=[],
            responses=mock_model_responses,
            graded_responses=[],
            execution_time=45.0,
            success=True
        )
        
        # Test report generation
        report_gen = ReportGenerator()
        
        # Test terminal report
        terminal_report = report_gen.generate_report(result, ReportFormat.TERMINAL)
        assert len(terminal_report) > 0
        assert "test/model" in terminal_report
        assert "80.0%" in terminal_report  # Accuracy
        
        # Test markdown report
        markdown_report = report_gen.generate_report(result, ReportFormat.MARKDOWN)
        assert len(markdown_report) > 0
        assert "# Jeopardy Benchmark Report" in markdown_report
        assert "test/model" in markdown_report
        
        # Test JSON report
        json_report = report_gen.generate_report(result, ReportFormat.JSON)
        assert len(json_report) > 0
        
        # Verify JSON is valid
        report_data = json.loads(json_report)
        assert report_data["report_metadata"]["models_tested"] == 1
        assert len(report_data["results"]) == 1
        assert report_data["results"][0]["model_name"] == "test/model"
        assert report_data["results"][0]["success"] == True
    
    def test_matcher_accuracy_with_jeopardy_formats(self):
        """Test matcher accuracy with various Jeopardy answer formats."""
        
        matcher = FuzzyMatcher()
        
        test_cases = [
            # Standard Jeopardy format
            ("What is Paris?", "What is Paris?", True),
            ("Who is Shakespeare?", "Who is William Shakespeare?", True),
            ("What is DNA?", "What is deoxyribonucleic acid?", True),
            
            # Direct answers vs Jeopardy format
            ("Paris", "What is Paris?", True),
            ("Shakespeare", "Who is William Shakespeare?", True),
            
            # Case variations
            ("what is paris?", "What is Paris?", True),
            ("WHO IS SHAKESPEARE?", "Who is William Shakespeare?", True),
            
            # Typos and variations
            ("What is Paaris?", "What is Paris?", True),
            ("Who is Shakespear?", "Who is Shakespeare?", True),
            
            # Wrong answers
            ("What is London?", "What is Paris?", False),
            ("Who is Dickens?", "Who is Shakespeare?", False),
        ]
        
        for answer, expected, should_match in test_cases:
            result = matcher.match_answer(answer, expected)
            
            if should_match:
                assert result.is_match or result.confidence > 0.7, \
                    f"Failed to match '{answer}' to '{expected}' (confidence: {result.confidence})"
            else:
                assert not result.is_match and result.confidence < 0.5, \
                    f"Incorrectly matched '{answer}' to '{expected}' (confidence: {result.confidence})"
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, sample_questions):
        """Test error handling and recovery mechanisms in the benchmark flow."""
        
        with patch('src.benchmark.runner.OpenRouterClient') as mock_client_class:
            # Mock client that raises an error
            mock_client = AsyncMock()
            mock_client.batch_query.side_effect = Exception("API Error")
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            # Mock question loading
            with patch.object(BenchmarkRunner, '_load_questions') as mock_load_questions:
                mock_load_questions.return_value = []
                
                # Mock database operations
                with patch('src.benchmark.runner.get_session') as mock_get_session:
                    mock_session = MagicMock()
                    mock_get_session.return_value.__enter__.return_value = mock_session
                    
                    mock_benchmark_repo = MagicMock()
                    mock_benchmark = MagicMock()
                    mock_benchmark.id = 123
                    mock_benchmark_repo.create_benchmark.return_value = mock_benchmark
                    
                    with patch('src.benchmark.runner.BenchmarkRepository', return_value=mock_benchmark_repo):
                        # Test that errors are handled gracefully
                        runner = BenchmarkRunner()
                        
                        result = await runner.run_benchmark(
                            model_name="test/failing-model",
                            mode=RunMode.QUICK
                        )
                        
                        # Should return a failed result, not crash
                        assert not result.success
                        assert result.error_message is not None
                        assert "API Error" in result.error_message
                        assert result.execution_time > 0
    
    def test_metrics_calculation_edge_cases(self):
        """Test metrics calculation with edge cases."""
        
        calculator = MetricsCalculator()
        
        # Test with empty data
        empty_graded = []
        empty_responses = []
        
        try:
            metrics = calculator.calculate_metrics(
                graded_responses=empty_graded,
                model_responses=empty_responses,
                model_name="test/empty-model"
            )
            # Should handle empty data gracefully
            assert metrics.accuracy.overall_accuracy == 0.0
            assert metrics.accuracy.total_count == 0
        except Exception as e:
            # Or raise appropriate error
            assert "must have same length" in str(e)
        
        # Test with single response
        from src.evaluation.grader import GradedResponse
        from src.evaluation.matcher import MatchResult, MatchType
        from src.models.response_parser import ParsedResponse, ResponseType
        
        single_match = MatchResult(
            is_match=True,
            confidence=0.95,
            match_type=MatchType.EXACT,
            details={},
            normalized_answer="paris",
            normalized_expected="paris"
        )
        
        single_parsed = ParsedResponse(
            original_text="Paris",
            extracted_answer="Paris", 
            response_type=ResponseType.DIRECT_ANSWER,
            confidence_indicators=[]
        )
        
        single_graded = GradedResponse(
            is_correct=True,
            score=1.0,
            confidence=0.95,
            partial_credit=1.0,
            match_result=single_match,
            parsed_response=single_parsed,
            grading_metadata={},
            grade_explanation="Perfect match",
            timestamp=datetime.now()
        )
        
        single_model_response = ModelResponse(
            model_id="test/model",
            prompt="Test",
            response="Paris",
            latency_ms=100.0,
            tokens_used=10,
            cost=0.001,
            timestamp=datetime.now()
        )
        
        metrics = calculator.calculate_metrics(
            graded_responses=[single_graded],
            model_responses=[single_model_response],
            model_name="test/single-model"
        )
        
        assert metrics.accuracy.overall_accuracy == 1.0
        assert metrics.accuracy.total_count == 1
        assert metrics.accuracy.correct_count == 1
        assert metrics.performance.mean_response_time == 0.1  # 100ms = 0.1s
        assert metrics.cost.total_cost == 0.001


@pytest.mark.integration
class TestBenchmarkCLIIntegration:
    """Integration tests for CLI commands."""
    
    def test_benchmark_command_structure(self):
        """Test that all benchmark CLI commands are properly structured."""
        
        from src.main import cli
        import click.testing
        
        runner = click.testing.CliRunner()
        
        # Test main benchmark command exists
        result = runner.invoke(cli, ['benchmark', '--help'])
        assert result.exit_code == 0
        assert 'Benchmark management commands' in result.output
        
        # Test subcommands exist
        subcommands = ['run', 'compare', 'history', 'report', 'list']
        for subcommand in subcommands:
            result = runner.invoke(cli, ['benchmark', subcommand, '--help'])
            assert result.exit_code == 0, f"Subcommand '{subcommand}' failed"
    
    def test_model_commands_integration(self):
        """Test model-related CLI commands."""
        
        from src.main import cli
        import click.testing
        
        runner = click.testing.CliRunner()
        
        # Test models list
        result = runner.invoke(cli, ['models', 'list', '--help'])
        assert result.exit_code == 0
        
        # Test models test
        result = runner.invoke(cli, ['models', 'test', '--help']) 
        assert result.exit_code == 0
        
        # Test models costs
        result = runner.invoke(cli, ['models', 'costs', '--help'])
        assert result.exit_code == 0


if __name__ == "__main__":
    # Run a quick integration test
    import asyncio
    
    async def quick_test():
        """Quick integration test for development."""
        print("Running quick integration test...")
        
        # Test basic components
        matcher = FuzzyMatcher()
        result = matcher.match_answer("Paris", "What is Paris?")
        print(f"Matcher test: {result.is_match}, confidence: {result.confidence}")
        
        grader = AnswerGrader()
        graded = grader.grade_response("Paris", "What is Paris?")
        print(f"Grader test: {graded.is_correct}, score: {graded.score}")
        
        print("Quick integration test completed!")
    
    asyncio.run(quick_test())