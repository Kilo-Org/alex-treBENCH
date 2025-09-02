#!/usr/bin/env python3
"""
Benchmark Simulator

Simulates benchmark runs with realistic mock data for testing and demonstration.
Generates performance metrics and tests system behavior under various conditions.
"""

import asyncio
import sys
import time
import random
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import statistics

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmark.runner import BenchmarkRunner, RunMode, BenchmarkConfig, BenchmarkRunResult
from benchmark.reporting import ReportGenerator, ReportFormat
from evaluation.metrics import MetricsCalculator, ComprehensiveMetrics
from models.base import ModelResponse
from storage.models import BenchmarkQuestion, create_benchmark_question
from evaluation.grader import GradedResponse
from core.config import get_config


@dataclass
class SimulationConfig:
    """Configuration for benchmark simulation."""
    num_models: int = 3
    questions_per_model: int = 50
    simulation_speed: float = 1.0  # Multiplier for simulation speed
    include_errors: bool = True
    error_rate: float = 0.05
    variable_performance: bool = True
    save_reports: bool = True
    output_dir: str = "simulation_results"


class BenchmarkSimulator:
    """Simulates benchmark runs with realistic mock data."""

    def __init__(self, config: SimulationConfig = None):
        """Initialize the benchmark simulator."""
        self.config = config or SimulationConfig()
        self.metrics_calculator = MetricsCalculator()
        self.report_generator = ReportGenerator()

        # Realistic model configurations
        self.models = [
            {
                "name": "openai/gpt-4",
                "base_accuracy": 0.82,
                "response_time_mean": 1.8,
                "response_time_std": 0.5,
                "cost_per_token": 0.00006,
                "tokens_per_response": 85
            },
            {
                "name": "openai/gpt-3.5-turbo",
                "base_accuracy": 0.71,
                "response_time_mean": 0.9,
                "response_time_std": 0.3,
                "cost_per_token": 0.000002,
                "tokens_per_response": 92
            },
            {
                "name": "anthropic/claude-3-sonnet",
                "base_accuracy": 0.79,
                "response_time_mean": 1.2,
                "response_time_std": 0.4,
                "cost_per_token": 0.000015,
                "tokens_per_response": 78
            },
            {
                "name": "anthropic/claude-3-haiku",
                "base_accuracy": 0.68,
                "response_time_mean": 0.7,
                "response_time_std": 0.2,
                "cost_per_token": 0.00000025,
                "tokens_per_response": 65
            },
            {
                "name": "google/gemini-pro",
                "base_accuracy": 0.75,
                "response_time_mean": 1.1,
                "response_time_std": 0.6,
                "cost_per_token": 0.0000005,
                "tokens_per_response": 88
            }
        ]

        # Jeopardy-style questions for simulation
        self.question_templates = [
            "This planet is known as the Red Planet",
            "The largest ocean on Earth",
            "The author of 'To Kill a Mockingbird'",
            "The chemical symbol for gold",
            "The year the Berlin Wall fell",
            "This programming language was created by Guido van Rossum",
            "The first element on the periodic table",
            "This battle marked the end of Napoleon Bonaparte's rule",
            "The capital of Australia",
            "This scientist formulated the theory of relativity"
        ]

        self.correct_answers = [
            "What is Mars?",
            "What is the Pacific Ocean?",
            "Who is Harper Lee?",
            "What is Au?",
            "What is 1989?",
            "What is Python?",
            "What is hydrogen?",
            "What is the Battle of Waterloo?",
            "What is Canberra?",
            "Who is Albert Einstein?"
        ]

    def generate_mock_questions(self, count: int) -> List[BenchmarkQuestion]:
        """Generate mock Jeopardy questions."""
        questions = []
        categories = ["Science", "History", "Literature", "Geography", "Arts"]

        for i in range(count):
            template_idx = i % len(self.question_templates)
            question = create_benchmark_question(
                benchmark_id=9999,  # Simulation benchmark ID
                question_id=f"sim_q_{i+1:04d}",
                question_text=self.question_templates[template_idx],
                correct_answer=self.correct_answers[template_idx],
                category=random.choice(categories),
                value=((i % 5) + 1) * 200,  # $200, $400, $600, $800, $1000
                difficulty_level=["Easy", "Medium", "Hard"][i % 3]
            )
            questions.append(question)

        return questions

    def generate_mock_responses(self, model_config: Dict[str, Any], questions: List[BenchmarkQuestion]) -> List[ModelResponse]:
        """Generate mock model responses with realistic characteristics."""
        responses = []
        base_accuracy = model_config["base_accuracy"]
        time_mean = model_config["response_time_mean"]
        time_std = model_config["response_time_std"]
        cost_per_token = model_config["cost_per_token"]
        tokens_per_response = model_config["tokens_per_response"]

        for i, question in enumerate(questions):
            # Simulate variable accuracy based on difficulty
            difficulty_multiplier = {
                "Easy": 1.1,
                "Medium": 1.0,
                "Hard": 0.9
            }.get(question.difficulty_level, 1.0)

            accuracy = base_accuracy * difficulty_multiplier

            # Add some randomness
            if self.config.variable_performance:
                accuracy += random.uniform(-0.1, 0.1)
            accuracy = max(0.1, min(0.95, accuracy))

            # Determine if answer is correct
            is_correct = random.random() < accuracy

            # Generate response text
            if is_correct:
                response_text = question.correct_answer
            else:
                # Generate a plausible wrong answer
                wrong_answers = [
                    "What is Jupiter?", "What is the Atlantic Ocean?", "Who is Mark Twain?",
                    "What is Ag?", "What is 1991?", "What is Java?", "What is helium?",
                    "What is the Battle of Hastings?", "What is Sydney?", "Who is Isaac Newton?"
                ]
                response_text = random.choice(wrong_answers)

            # Simulate response time with some variability
            response_time = random.gauss(time_mean, time_std)
            response_time = max(0.1, response_time)  # Minimum 100ms

            # Apply simulation speed multiplier
            response_time /= self.config.simulation_speed

            # Calculate cost
            cost = tokens_per_response * cost_per_token

            # Simulate occasional errors
            metadata = {"mock": True, "simulation": True}
            if self.config.include_errors and random.random() < self.config.error_rate:
                metadata["error"] = "Simulated API error"
                response_time += random.uniform(1.0, 5.0)  # Add delay for errors

            response = ModelResponse(
                model_id=model_config["name"],
                prompt=question.question_text,
                response=response_text,
                latency_ms=response_time * 1000,
                tokens_used=tokens_per_response,
                cost=cost,
                timestamp=datetime.now(),
                metadata=metadata
            )

            responses.append(response)

            # Simulate processing delay
            time.sleep(response_time * 0.1)  # Brief pause for realism

        return responses

    def generate_graded_responses(self, responses: List[ModelResponse], questions: List[BenchmarkQuestion]) -> List[GradedResponse]:
        """Generate graded responses based on model responses."""
        graded_responses = []

        for response, question in zip(responses, questions):
            # Determine correctness
            is_correct = response.response.strip().lower() == question.correct_answer.strip().lower()

            # Calculate confidence (simplified)
            confidence = 0.8 if is_correct else 0.4
            if self.config.variable_performance:
                confidence += random.uniform(-0.2, 0.2)
            confidence = max(0.1, min(0.95, confidence))

            graded = GradedResponse(
                question_id=question.question_id,
                model_answer=response.response,
                correct_answer=question.correct_answer,
                is_correct=is_correct,
                confidence=confidence,
                match_result=None,  # Simplified for simulation
                partial_credit=0.0,
                grading_metadata={
                    "mode": "lenient",
                    "grading_time_ms": random.uniform(10, 50),
                    "simulation": True
                }
            )

            graded_responses.append(graded)

        return graded_responses

    async def simulate_single_model_benchmark(self, model_config: Dict[str, Any]) -> BenchmarkRunResult:
        """Simulate a benchmark for a single model."""
        print(f"🎯 Simulating benchmark for {model_config['name']}...")

        # Generate questions
        questions = self.generate_mock_questions(self.config.questions_per_model)

        # Generate responses
        responses = self.generate_mock_responses(model_config, questions)

        # Generate graded responses
        graded_responses = self.generate_graded_responses(responses, questions)

        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(
            graded_responses=graded_responses,
            model_responses=responses,
            model_name=model_config["name"],
            benchmark_id=9999
        )

        # Create benchmark result
        result = BenchmarkRunResult(
            benchmark_id=9999,
            model_name=model_config["name"],
            config=BenchmarkConfig(
                mode=RunMode.CUSTOM,
                sample_size=self.config.questions_per_model
            ),
            progress=None,
            metrics=metrics,
            questions=questions,
            responses=responses,
            graded_responses=graded_responses,
            execution_time=sum(r.latency_ms / 1000 for r in responses),
            success=True,
            metadata={
                "simulation": True,
                "model_config": model_config,
                "simulation_config": {
                    "speed": self.config.simulation_speed,
                    "include_errors": self.config.include_errors,
                    "error_rate": self.config.error_rate
                }
            }
        )

        print(".1%")

        return result

    async def simulate_model_comparison(self) -> List[BenchmarkRunResult]:
        """Simulate benchmark comparison across multiple models."""
        print(f"🔄 Simulating comparison of {self.config.num_models} models...")

        # Select models for simulation
        selected_models = self.models[:self.config.num_models]
        results = []

        for model_config in selected_models:
            result = await self.simulate_single_model_benchmark(model_config)
            results.append(result)

        return results

    def generate_simulation_report(self, results: List[BenchmarkRunResult]) -> Dict[str, Any]:
        """Generate a comprehensive simulation report."""
        report = {
            "simulation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "num_models": self.config.num_models,
                    "questions_per_model": self.config.questions_per_model,
                    "simulation_speed": self.config.simulation_speed,
                    "include_errors": self.config.include_errors,
                    "error_rate": self.config.error_rate
                },
                "total_questions": len(results) * self.config.questions_per_model,
                "total_responses": sum(len(r.responses) for r in results)
            },
            "model_results": [],
            "comparison_metrics": {},
            "performance_analysis": {}
        }

        # Individual model results
        for result in results:
            model_data = {
                "model_name": result.model_name,
                "accuracy": result.metrics.accuracy.overall_accuracy,
                "execution_time": result.execution_time,
                "total_cost": result.metrics.cost.total_cost,
                "cost_per_question": result.metrics.cost.cost_per_question,
                "mean_response_time": result.metrics.performance.mean_response_time,
                "consistency_score": result.metrics.consistency.confidence_correlation,
                "questions_processed": len(result.questions)
            }
            report["model_results"].append(model_data)

        # Comparison metrics
        if len(results) > 1:
            accuracies = [r.metrics.accuracy.overall_accuracy for r in results]
            costs = [r.metrics.cost.cost_per_question for r in results]
            times = [r.metrics.performance.mean_response_time for r in results]

            report["comparison_metrics"] = {
                "accuracy_range": {
                    "min": min(accuracies),
                    "max": max(accuracies),
                    "spread": max(accuracies) - min(accuracies)
                },
                "cost_range": {
                    "min": min(costs),
                    "max": max(costs),
                    "spread": max(costs) - min(costs)
                },
                "performance_range": {
                    "min": min(times),
                    "max": max(times),
                    "spread": max(times) - min(times)
                },
                "best_performers": {
                    "accuracy": results[accuracies.index(max(accuracies))].model_name,
                    "cost_efficiency": results[costs.index(min(costs))].model_name,
                    "speed": results[times.index(min(times))].model_name
                }
            }

        # Performance analysis
        all_response_times = []
        all_costs = []
        all_accuracies = []

        for result in results:
            all_response_times.extend([r.latency_ms for r in result.responses])
            all_costs.extend([r.cost for r in result.responses])
            all_accuracies.extend([1 if g.is_correct else 0 for g in result.graded_responses])

        if all_response_times:
            report["performance_analysis"] = {
                "response_time_stats": {
                    "mean": statistics.mean(all_response_times),
                    "median": statistics.median(all_response_times),
                    "std_dev": statistics.stdev(all_response_times) if len(all_response_times) > 1 else 0,
                    "min": min(all_response_times),
                    "max": max(all_response_times)
                },
                "overall_accuracy": sum(all_accuracies) / len(all_accuracies),
                "total_simulation_cost": sum(all_costs),
                "average_cost_per_response": statistics.mean(all_costs)
            }

        return report

    def save_simulation_results(self, results: List[BenchmarkRunResult], report: Dict[str, Any]):
        """Save simulation results to files."""
        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)

        # Save detailed report
        report_file = output_dir / "simulation_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Generate and save comparison report
        if len(results) > 1:
            comparison_report = self.report_generator.generate_comparison_report(
                results, ReportFormat.MARKDOWN
            )

            comparison_file = output_dir / "model_comparison.md"
            with open(comparison_file, "w") as f:
                f.write(comparison_report)

        # Save individual model reports
        for result in results:
            model_name_safe = result.model_name.replace("/", "_").replace("-", "_")
            individual_report = self.report_generator.generate_report(
                result, ReportFormat.MARKDOWN
            )

            individual_file = output_dir / f"{model_name_safe}_report.md"
            with open(individual_file, "w") as f:
                f.write(individual_report)

        print(f"\n📁 Simulation results saved to: {output_dir.absolute()}")

    def display_simulation_summary(self, results: List[BenchmarkRunResult], report: Dict[str, Any]):
        """Display a summary of the simulation results."""
        print("\n" + "="*70)
        print("📊 BENCHMARK SIMULATION SUMMARY")
        print("="*70)

        meta = report["simulation_metadata"]
        print(f"Models Tested: {meta['config']['num_models']}")
        print(f"Questions per Model: {meta['config']['questions_per_model']}")
        print(f"Total Questions: {meta['total_questions']}")
        print(f"Simulation Speed: {meta['config']['simulation_speed']}x")

        print("
🏆 MODEL PERFORMANCE RANKING:"        print("<25")
        print("-" * 50)

        sorted_results = sorted(results, key=lambda r: r.metrics.accuracy.overall_accuracy, reverse=True)
        for i, result in enumerate(sorted_results, 1):
            metrics = result.metrics
            print("<25")

        if "comparison_metrics" in report:
            comp = report["comparison_metrics"]
            print("
📈 PERFORMANCE SPREAD:"            print(".1%")
            print(".4f")
            print(".2f")

        if "performance_analysis" in report:
            perf = report["performance_analysis"]
            print("
⚡ OVERALL STATISTICS:"            print(".1%")
            print(".2f")
            print(".4f")

        print("\n" + "="*70)

    async def run_simulation(self):
        """Run the complete benchmark simulation."""
        print("🎭 Jeopardy Benchmarking System - Simulation")
        print("=" * 50)
        print(f"Running simulation with {self.config.num_models} models")
        print(f"Questions per model: {self.config.questions_per_model}")
        print(f"Simulation speed: {self.config.simulation_speed}x")
        print()

        start_time = time.time()

        # Run simulation
        results = await self.simulate_model_comparison()

        # Generate report
        report = self.generate_simulation_report(results)

        # Save results
        if self.config.save_reports:
            self.save_simulation_results(results, report)

        # Display summary
        self.display_simulation_summary(results, report)

        total_time = time.time() - start_time
        print(".2f"
        print("
✅ Simulation completed successfully!"        print("📄 Check the simulation_results/ directory for detailed reports."
        return results, report


async def main():
    """Main simulation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Simulator")
    parser.add_argument("--models", type=int, default=3, help="Number of models to simulate")
    parser.add_argument("--questions", type=int, default=50, help="Questions per model")
    parser.add_argument("--speed", type=float, default=1.0, help="Simulation speed multiplier")
    parser.add_argument("--errors", action="store_true", help="Include simulated errors")
    parser.add_argument("--error-rate", type=float, default=0.05, help="Error rate (0.0-1.0)")
    parser.add_argument("--output-dir", default="simulation_results", help="Output directory")

    args = parser.parse_args()

    # Configure simulation
    config = SimulationConfig(
        num_models=args.models,
        questions_per_model=args.questions,
        simulation_speed=args.speed,
        include_errors=args.errors,
        error_rate=args.error_rate,
        output_dir=args.output_dir
    )

    # Run simulation
    simulator = BenchmarkSimulator(config)
    results, report = await simulator.run_simulation()

    # Exit successfully
    sys.exit(0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error during simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)