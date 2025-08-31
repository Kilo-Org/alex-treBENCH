#!/usr/bin/env python3
"""
Interactive Demo Script for Jeopardy Benchmarking System

This script provides an interactive demonstration of the system's key features,
using mock data to showcase functionality without requiring API keys.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmark.runner import BenchmarkRunner, RunMode, BenchmarkConfig, BenchmarkResult
from benchmark.reporting import ReportGenerator, ReportFormat
from evaluation.metrics import MetricsCalculator, ComprehensiveMetrics
from models.base import ModelResponse
from core.config import AppConfig


class DemoSystem:
    """Interactive demo system for showcasing benchmarking features."""

    def __init__(self):
        """Initialize the demo system."""
        self.runner = None
        self.generator = ReportGenerator()
        self.calculator = MetricsCalculator()
        self.mock_responses = self._generate_mock_responses()

    def _generate_mock_responses(self) -> List[ModelResponse]:
        """Generate realistic mock responses for demonstration."""
        return [
            ModelResponse(
                text="What is Mars?",
                model_name="demo-model",
                response_time_ms=1200,
                tokens_generated=4,
                cost=0.001,
                metadata={"mock": True, "confidence": 0.95}
            ),
            ModelResponse(
                text="What is the Pacific Ocean?",
                model_name="demo-model",
                response_time_ms=980,
                tokens_generated=6,
                cost=0.001,
                metadata={"mock": True, "confidence": 0.92}
            ),
            ModelResponse(
                text="Who is Harper Lee?",
                model_name="demo-model",
                response_time_ms=1450,
                tokens_generated=5,
                cost=0.001,
                metadata={"mock": True, "confidence": 0.88}
            ),
            ModelResponse(
                text="What is Au?",
                model_name="demo-model",
                response_time_ms=850,
                tokens_generated=3,
                cost=0.001,
                metadata={"mock": True, "confidence": 0.96}
            ),
            ModelResponse(
                text="What is 1989?",
                model_name="demo-model",
                response_time_ms=720,
                tokens_generated=4,
                cost=0.001,
                metadata={"mock": True, "confidence": 0.89}
            )
        ]

    def print_header(self):
        """Print the demo header."""
        print("\n" + "="*60)
        print("🧠 Jeopardy Benchmarking System - Interactive Demo")
        print("="*60)
        print("This demo showcases the system's key features using mock data.")
        print("No API keys required - perfect for evaluation and testing!")
        print("="*60)

    def print_menu(self):
        """Print the main menu."""
        print("\n📋 Demo Menu:")
        print("1. 🚀 Run Quick Benchmark Demo")
        print("2. 🔄 Compare Multiple Models Demo")
        print("3. 📊 View Detailed Metrics Analysis")
        print("4. 📄 Generate Sample Reports")
        print("5. 🎯 Test Different Grading Modes")
        print("6. ⏱️ Performance Benchmark Demo")
        print("7. 📈 Advanced Analytics Demo")
        print("8. 🛠️ System Configuration Demo")
        print("9. 📚 Show Available Features")
        print("0. 👋 Exit Demo")
        print("\n" + "-"*30)

    async def run_quick_benchmark_demo(self):
        """Demonstrate a quick benchmark run."""
        print("\n🚀 Quick Benchmark Demo")
        print("-" * 30)

        print("Setting up benchmark configuration...")
        config = BenchmarkConfig(
            mode=RunMode.QUICK,
            sample_size=5,
            timeout_seconds=30,
            grading_mode="lenient",
            save_results=False
        )

        print("Initializing benchmark runner...")
        runner = BenchmarkRunner()

        print("Running benchmark with mock data...")
        print("(In real usage, this would call the OpenRouter API)")

        # Simulate benchmark execution with mock data
        start_time = time.time()

        # Create mock result
        mock_result = self._create_mock_benchmark_result(
            model_name="demo-gpt-3.5-turbo",
            sample_size=5,
            execution_time=3.45
        )

        print("
✅ Benchmark completed!"        print(f"📊 Model: {mock_result.model_name}")
        print(f"📏 Sample Size: {len(mock_result.questions)} questions")
        print(".1%")
        print(".2f")
        print(".4f")

        return mock_result

    async def run_model_comparison_demo(self):
        """Demonstrate comparing multiple models."""
        print("\n🔄 Model Comparison Demo")
        print("-" * 30)

        models = ["demo-gpt-3.5-turbo", "demo-gpt-4", "demo-claude-3-haiku"]
        results = []

        print(f"Comparing {len(models)} models...")
        print()

        for i, model in enumerate(models, 1):
            print(f"{i}. Testing {model}...")

            # Create mock result for each model
            accuracy = 0.75 + (i * 0.05)  # Increasing accuracy for demo
            cost = 0.001 * (i * 0.5)  # Varying costs

            mock_result = self._create_mock_benchmark_result(
                model_name=model,
                sample_size=10,
                execution_time=2.5 + (i * 0.3),
                accuracy=accuracy,
                cost_per_question=cost
            )

            results.append(mock_result)
            print(".1%")

        print("
📊 Comparison Summary:"        print("<10")
        print("-" * 40)
        for result in results:
            metrics = result.metrics
            print("<10")

        # Generate comparison report
        print("
📄 Generating comparison report..."        comparison_report = self.generator.generate_comparison_report(
            results, ReportFormat.MARKDOWN
        )

        with open("demo_comparison_report.md", "w") as f:
            f.write(comparison_report)

        print("💾 Comparison report saved to: demo_comparison_report.md")

        return results

    async def show_metrics_analysis_demo(self):
        """Demonstrate detailed metrics analysis."""
        print("\n📊 Detailed Metrics Analysis Demo")
        print("-" * 40)

        # Create a comprehensive mock result
        result = self._create_mock_benchmark_result(
            model_name="demo-gpt-4",
            sample_size=20,
            execution_time=8.92
        )

        metrics = result.metrics

        print("🎯 ACCURACY METRICS:")
        print(f"  Overall Accuracy: {metrics.accuracy.overall_accuracy:.1%}")
        print(f"  Correct Answers: {metrics.accuracy.correct_count}/{metrics.accuracy.total_count}")
        print(f"  Confidence Interval: ±{2.1:.1f}% (95% confidence)")

        print("
⏱️  PERFORMANCE METRICS:"        print(".2f")
        print(".2f")
        print(".2f")
        print(f"  Error Rate: {metrics.performance.error_count / metrics.accuracy.total_count:.1f}")

        print("
💰 COST ANALYSIS:"        print(".4f")
        print(".4f")
        print(".4f")
        print(f"  Cost Efficiency Score: {metrics.cost.cost_efficiency_score:.2f}")

        print("
📈 CATEGORY PERFORMANCE:"        categories = ["Science", "History", "Literature", "Geography"]
        for category in categories:
            accuracy = 0.70 + (hash(category) % 20) / 100  # Mock category accuracies
            print("<12")

        print("
🎯 CONSISTENCY METRICS:"        print(".3f")
        print(".3f")
        print(".3f")

        print("
🏆 COMPOSITE SCORES:"        print(".3f")
        print(".3f")
        print(".3f")

        return result

    async def generate_reports_demo(self):
        """Demonstrate report generation in different formats."""
        print("\n📄 Report Generation Demo")
        print("-" * 30)

        # Create mock result
        result = self._create_mock_benchmark_result(
            model_name="demo-gpt-4",
            sample_size=15,
            execution_time=6.78
        )

        formats = [
            (ReportFormat.TERMINAL, "demo_terminal_report.txt"),
            (ReportFormat.MARKDOWN, "demo_markdown_report.md"),
            (ReportFormat.JSON, "demo_json_report.json")
        ]

        print("Generating reports in multiple formats...")

        for format_type, filename in formats:
            print(f"📝 Creating {format_type.value} report...")

            report_content = self.generator.generate_report(result, format_type)

            with open(filename, "w") as f:
                f.write(report_content)

            print(f"💾 Saved to: {filename}")

        print("
📊 Sample Markdown Report Preview:"        print("-" * 40)
        markdown_report = self.generator.generate_report(result, ReportFormat.MARKDOWN)
        # Show first few lines
        lines = markdown_report.split('\n')[:10]
        for line in lines:
            print(f"  {line}")

        print("  ... (truncated)")

    async def test_grading_modes_demo(self):
        """Demonstrate different grading modes."""
        print("\n🎯 Grading Modes Demo")
        print("-" * 25)

        grading_modes = ["strict", "lenient", "jeopardy", "adaptive"]
        base_accuracy = 0.78

        print("Testing different grading modes on the same responses...")
        print()

        for mode in grading_modes:
            # Simulate different accuracies based on grading strictness
            if mode == "strict":
                accuracy = base_accuracy - 0.08
            elif mode == "lenient":
                accuracy = base_accuracy + 0.05
            elif mode == "jeopardy":
                accuracy = base_accuracy
            else:  # adaptive
                accuracy = base_accuracy + 0.02

            print("<10")

        print("
📝 Grading Mode Explanations:"        print("  • Strict: Exact matches only, no partial credit")
        print("  • Lenient: Allows minor variations and typos")
        print("  • Jeopardy: Follows Jeopardy answer format rules")
        print("  • Adaptive: Adjusts based on question difficulty")

    async def performance_benchmark_demo(self):
        """Demonstrate performance benchmarking."""
        print("\n⏱️ Performance Benchmark Demo")
        print("-" * 32)

        print("Testing system performance with different configurations...")

        configs = [
            ("Quick Mode", 5, 0.8),
            ("Standard Mode", 20, 2.1),
            ("Large Dataset", 100, 8.5)
        ]

        print("<15")
        print("-" * 50)

        for name, sample_size, exec_time in configs:
            memory_mb = sample_size * 0.5 + 10  # Mock memory usage
            throughput = sample_size / exec_time

            print("<15")

        print("
💡 Performance Insights:"        print("  • Memory usage scales linearly with dataset size")
        print("  • Throughput remains consistent across configurations")
        print("  • Quick mode: Best for rapid evaluation")
        print("  • Standard mode: Balanced performance and accuracy")
        print("  • Large datasets: Best for statistical significance")

    async def advanced_analytics_demo(self):
        """Demonstrate advanced analytics features."""
        print("\n📈 Advanced Analytics Demo")
        print("-" * 30)

        print("Analyzing performance patterns and trends...")

        # Mock analytics data
        print("
🔍 Performance by Difficulty Level:"        difficulties = ["Easy", "Medium", "Hard"]
        for diff in difficulties:
            accuracy = 0.85 - (difficulties.index(diff) * 0.08)
            avg_time = 1.0 + (difficulties.index(diff) * 0.3)
            print("<8")

        print("
📊 Response Time Distribution:"        print("  • Fast responses (< 1s): 35%")
        print("  • Normal responses (1-2s): 45%")
        print("  • Slow responses (2-5s): 18%")
        print("  • Very slow responses (> 5s): 2%")

        print("
🎯 Confidence vs Accuracy Correlation:"        print("  • High confidence (> 0.8): 89% accuracy")
        print("  • Medium confidence (0.6-0.8): 76% accuracy")
        print("  • Low confidence (< 0.6): 65% accuracy")
        print("  • Correlation coefficient: 0.82")

        print("
💡 Key Insights:"        print("  • Model performs best on easy questions")
        print("  • Response time increases with difficulty")
        print("  • Confidence scores are well-calibrated")
        print("  • Consider timeout adjustments for hard questions")

    async def system_configuration_demo(self):
        """Demonstrate system configuration options."""
        print("\n🛠️ System Configuration Demo")
        print("-" * 32)

        print("Exploring configuration options...")

        # Show sample configuration
        config = AppConfig()

        print("
📋 Current Configuration:"        print(f"  Database: {config.database.url}")
        print(f"  Debug Mode: {config.debug}")
        print(f"  Max Concurrent Requests: {config.benchmark.max_concurrent_requests}")
        print(f"  Default Sample Size: {config.benchmark.default_sample_size}")
        print(f"  Log Level: {config.logging.level}")

        print("
⚙️ Available Benchmark Modes:"        modes = config.benchmark.modes
        for mode_name, mode_config in modes.items():
            print("<12")

        print("
🔧 Configuration Files:"        print("  • config/default.yaml - Main configuration")
        print("  • config/models/ - Model-specific settings")
        print("  • .env - Environment variables")

        print("
💡 Configuration Tips:"        print("  • Use YAML files for complex configurations")
        print("  • Override with environment variables for secrets")
        print("  • Test configurations with quick benchmarks first")

    async def show_features_demo(self):
        """Show comprehensive list of available features."""
        print("\n📚 Available Features Overview")
        print("-" * 35)

        features = {
            "🏃 Benchmarking": [
                "Single model evaluation",
                "Multi-model comparison",
                "Custom benchmark configurations",
                "Real-time progress tracking"
            ],
            "📊 Analytics": [
                "Comprehensive metrics calculation",
                "Category-wise performance analysis",
                "Difficulty level breakdowns",
                "Statistical significance testing"
            ],
            "📋 Reporting": [
                "Terminal output formatting",
                "Markdown report generation",
                "JSON export for integration",
                "HTML reports (future)"
            ],
            "🔧 Configuration": [
                "YAML-based configuration",
                "Environment variable overrides",
                "Model-specific settings",
                "Runtime configuration reloading"
            ],
            "🗃️ Data Management": [
                "Kaggle dataset integration",
                "Statistical sampling",
                "Data preprocessing pipeline",
                "Question categorization"
            ],
            "🤖 Model Support": [
                "OpenAI GPT models",
                "Anthropic Claude models",
                "Google Gemini models",
                "Meta Llama models"
            ],
            "🧪 Testing": [
                "Unit test coverage",
                "Integration testing",
                "End-to-end workflow tests",
                "Performance benchmarking"
            ]
        }

        for category, feature_list in features.items():
            print(f"\n{category}:")
            for feature in feature_list:
                print(f"  • {feature}")

        print("
🚀 Getting Started:"        print("  1. Run the quick start script: ./examples/quick_start.sh")
        print("  2. Read the user guide: docs/USER_GUIDE.md")
        print("  3. Check the API reference: docs/API_REFERENCE.md")
        print("  4. Explore examples: examples/")

    def _create_mock_benchmark_result(self,
                                    model_name: str,
                                    sample_size: int,
                                    execution_time: float,
                                    accuracy: float = 0.782,
                                    cost_per_question: float = 0.0047) -> BenchmarkResult:
        """Create a mock benchmark result for demonstration."""
        from datetime import datetime
        from storage.models import BenchmarkQuestion, create_benchmark_question
        from evaluation.grader import GradedResponse

        # Create mock questions
        questions = []
        for i in range(sample_size):
            question = create_benchmark_question(
                benchmark_id=999,
                question_id=f"demo_q_{i+1}",
                question_text=f"Demo question {i+1}?",
                correct_answer=f"What is answer {i+1}?",
                category="DEMO",
                value=200,
                difficulty_level="Medium"
            )
            questions.append(question)

        # Create mock graded responses
        graded_responses = []
        for i in range(sample_size):
            is_correct = i < int(accuracy * sample_size)  # Mock correctness
            graded = GradedResponse(
                question_id=f"demo_q_{i+1}",
                model_answer=f"Demo answer {i+1}",
                correct_answer=f"What is answer {i+1}?",
                is_correct=is_correct,
                confidence=0.8 + (i % 3) * 0.1,
                match_result=None,
                partial_credit=0.0,
                grading_metadata={"mode": "lenient", "grading_time_ms": 50}
            )
            graded_responses.append(graded)

        # Create mock metrics
        metrics = self.calculator.calculate_metrics(
            graded_responses=graded_responses,
            model_responses=self.mock_responses[:sample_size],
            model_name=model_name,
            benchmark_id=999
        )

        # Override some metrics for demo purposes
        metrics.accuracy.overall_accuracy = accuracy
        metrics.cost.cost_per_question = cost_per_question
        metrics.cost.total_cost = cost_per_question * sample_size

        return BenchmarkResult(
            benchmark_id=999,
            model_name=model_name,
            config=BenchmarkConfig(mode=RunMode.QUICK, sample_size=sample_size),
            progress=None,
            metrics=metrics,
            questions=questions,
            responses=self.mock_responses[:sample_size],
            graded_responses=graded_responses,
            execution_time=execution_time,
            success=True,
            metadata={"demo": True, "timestamp": datetime.now().isoformat()}
        )

    async def run_demo(self):
        """Run the interactive demo."""
        self.print_header()

        while True:
            self.print_menu()

            try:
                choice = input("Enter your choice (0-9): ").strip()

                if choice == "0":
                    print("\n👋 Thank you for exploring the Jeopardy Benchmarking System!")
                    print("We hope this demo has given you a good understanding of our capabilities.")
                    print("\nFor more information:")
                    print("📖 User Guide: docs/USER_GUIDE.md")
                    print("🔧 API Reference: docs/API_REFERENCE.md")
                    print("📧 GitHub: [repository-url]")
                    break

                elif choice == "1":
                    await self.run_quick_benchmark_demo()
                elif choice == "2":
                    await self.run_model_comparison_demo()
                elif choice == "3":
                    await self.show_metrics_analysis_demo()
                elif choice == "4":
                    await self.generate_reports_demo()
                elif choice == "5":
                    await self.test_grading_modes_demo()
                elif choice == "6":
                    await self.performance_benchmark_demo()
                elif choice == "7":
                    await self.advanced_analytics_demo()
                elif choice == "8":
                    await self.system_configuration_demo()
                elif choice == "9":
                    await self.show_features_demo()
                else:
                    print("❌ Invalid choice. Please enter a number between 0-9.")

            except KeyboardInterrupt:
                print("\n\n👋 Demo interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please try again.")

            input("\nPress Enter to continue...")


async def main():
    """Main function to run the demo."""
    demo = DemoSystem()
    await demo.run_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo exited. Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)