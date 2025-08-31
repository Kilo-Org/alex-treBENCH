#!/usr/bin/env python3
"""
Sample Benchmark Script

This script demonstrates how to use the Jeopardy Benchmarking System
programmatically to run benchmarks and generate reports.
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmark.runner import BenchmarkRunner, RunMode, BenchmarkConfig
from benchmark.reporting import ReportGenerator, ReportFormat
from core.config import get_config
from models.base import ModelConfig


async def run_single_model_benchmark():
    """Run a benchmark for a single model."""
    print("🚀 Running single model benchmark...")

    runner = BenchmarkRunner()

    # Configure benchmark
    config = BenchmarkConfig(
        mode=RunMode.STANDARD,
        sample_size=100,
        timeout_seconds=60,
        grading_mode="lenient",
        save_results=True
    )

    # Run benchmark
    result = await runner.run_benchmark(
        model_name="openai/gpt-3.5-turbo",
        mode=RunMode.STANDARD,
        custom_config=config,
        benchmark_name="Sample Single Model Benchmark"
    )

    print("✅ Benchmark completed!"    print(f"📊 Accuracy: {result.metrics.accuracy.overall_accuracy:.1%}")
    print(f"⏱️  Execution Time: {result.execution_time:.2f}s")
    print(f"💰 Total Cost: ${result.metrics.cost.total_cost:.4f}")

    return result


async def run_model_comparison():
    """Compare multiple models."""
    print("🔄 Running model comparison...")

    runner = BenchmarkRunner()

    models = ["openai/gpt-3.5-turbo", "openai/gpt-4", "anthropic/claude-3-haiku"]
    results = []

    # Run benchmarks for each model
    for model in models:
        print(f"📝 Benchmarking {model}...")

        config = BenchmarkConfig(
            mode=RunMode.QUICK,
            sample_size=50,
            timeout_seconds=45,
            grading_mode="lenient",
            save_results=True
        )

        result = await runner.run_benchmark(
            model_name=model,
            mode=RunMode.QUICK,
            custom_config=config,
            benchmark_name=f"Comparison - {model}"
        )

        results.append(result)
        print(".1%")

    # Generate comparison report
    generator = ReportGenerator()
    comparison_report = generator.generate_comparison_report(
        results, ReportFormat.MARKDOWN
    )

    # Save comparison report
    with open("model_comparison_report.md", "w") as f:
        f.write(comparison_report)

    print("📄 Comparison report saved to model_comparison_report.md")
    return results


async def run_custom_benchmark():
    """Run a custom benchmark with specific settings."""
    print("⚙️ Running custom benchmark...")

    runner = BenchmarkRunner()

    # Custom configuration
    config = BenchmarkConfig(
        mode=RunMode.CUSTOM,
        sample_size=75,
        timeout_seconds=90,
        grading_mode="strict",
        save_results=True,
        max_concurrent_requests=3
    )

    # Custom model configuration
    model_config = ModelConfig(
        model_name="openai/gpt-4",
        temperature=0.2,
        max_tokens=200
    )

    result = await runner.run_benchmark(
        model_name="openai/gpt-4",
        mode=RunMode.CUSTOM,
        custom_config=config,
        benchmark_name="Custom GPT-4 Benchmark"
    )

    print("✅ Custom benchmark completed!"    print(f"📊 Accuracy: {result.metrics.accuracy.overall_accuracy:.1%}")
    print(f"🎯 Strict Grading Mode Used")
    print(f"📏 Sample Size: {len(result.questions)} questions")

    return result


async def generate_detailed_report(result):
    """Generate a detailed report for a benchmark result."""
    print("📊 Generating detailed report...")

    generator = ReportGenerator()

    # Generate different report formats
    formats = [
        (ReportFormat.TERMINAL, "terminal_report.txt"),
        (ReportFormat.MARKDOWN, "detailed_report.md"),
        (ReportFormat.JSON, "benchmark_data.json")
    ]

    for format_type, filename in formats:
        report_content = generator.generate_report(result, format_type)

        with open(filename, "w") as f:
            f.write(report_content)

        print(f"💾 {format_type.value.capitalize()} report saved to {filename}")


async def analyze_benchmark_metrics(result):
    """Analyze and display detailed metrics."""
    print("🔍 Analyzing benchmark metrics...")

    metrics = result.metrics

    print("\n📈 PERFORMANCE METRICS:"    print(f"  Overall Accuracy: {metrics.accuracy.overall_accuracy:.1%}")
    print(f"  Total Questions: {metrics.accuracy.total_count}")
    print(f"  Correct Answers: {metrics.accuracy.correct_count}")

    print("
⏱️  TIMING METRICS:"    print(f"  Mean Response Time: {metrics.performance.mean_response_time:.2f}s")
    print(f"  Median Response Time: {metrics.performance.median_response_time:.2f}s")
    print(f"  95th Percentile: {metrics.performance.p95_response_time:.2f}s")

    print("
💰 COST METRICS:"    print(f"  Total Cost: ${metrics.cost.total_cost:.4f}")
    print(f"  Cost per Question: ${metrics.cost.cost_per_question:.4f}")
    print(f"  Cost per Correct Answer: ${metrics.cost.cost_per_correct_answer:.4f}")

    print("
📊 CATEGORY PERFORMANCE:"    for category, accuracy in metrics.accuracy.by_category.items():
        print(f"  {category}: {accuracy:.1%}")

    print("
🎯 CONSISTENCY METRICS:"    print(f"  Confidence Correlation: {metrics.consistency.confidence_correlation:.3f}")
    print(f"  Category Consistency: {metrics.consistency.category_consistency_score:.3f}")

    print("
🏆 COMPOSITE SCORES:"    print(f"  Overall Score: {metrics.overall_score:.3f}")
    print(f"  Quality Score: {metrics.quality_score:.3f}")
    print(f"  Efficiency Score: {metrics.efficiency_score:.3f}")


async def main():
    """Main function to run sample benchmarks."""
    print("🧠 Jeopardy Benchmarking System - Sample Script")
    print("=" * 50)

    try:
        # Example 1: Single model benchmark
        print("\n" + "="*50)
        result1 = await run_single_model_benchmark()

        # Example 2: Model comparison
        print("\n" + "="*50)
        results = await run_model_comparison()

        # Example 3: Custom benchmark
        print("\n" + "="*50)
        result3 = await run_custom_benchmark()

        # Generate reports
        print("\n" + "="*50)
        await generate_detailed_report(result1)

        # Analyze metrics
        print("\n" + "="*50)
        await analyze_benchmark_metrics(result1)

        print("\n" + "="*50)
        print("🎉 All sample benchmarks completed successfully!")
        print("📁 Check the generated report files for detailed results.")

    except Exception as e:
        print(f"❌ Error running benchmarks: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    # Run the sample benchmarks
    exit_code = asyncio.run(main())
    sys.exit(exit_code)