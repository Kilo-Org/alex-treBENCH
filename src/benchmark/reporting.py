
"""
Benchmark Report Generator

Generates comprehensive reports from benchmark results in multiple formats
including terminal output, Markdown, JSON, and CSV with rich visualizations.
"""

import json
import csv
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from pathlib import Path
import io

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    from rich.layout import Layout
    from rich.text import Text
    from rich.tree import Tree
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None
    Table = None

from .runner import BenchmarkResult
from ..evaluation.metrics import ComprehensiveMetrics
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ReportFormat(str, Enum):
    """Available report formats."""
    TERMINAL = "terminal"
    MARKDOWN = "markdown"
    JSON = "json"
    CSV = "csv"
    HTML = "html"


@dataclass
class ReportSection:
    """A section of the report."""
    title: str
    content: Any
    level: int = 1  # Heading level for markdown
    metadata: Dict[str, Any] = None


@dataclass  
class ReportConfig:
    """Configuration for report generation."""
    include_summary: bool = True
    include_detailed_metrics: bool = True
    include_category_breakdown: bool = True
    include_cost_analysis: bool = True
    include_performance_charts: bool = True
    include_raw_data: bool = False
    max_items_per_section: int = 50
    decimal_places: int = 3
    show_confidence_intervals: bool = True


class ReportGenerator:
    """Generates comprehensive benchmark reports."""
    
    def __init__(self):
        """Initialize the report generator."""
        self.console = Console() if RICH_AVAILABLE else None
        
    def generate_report(self,
                       results: Union[BenchmarkResult, List[BenchmarkResult]],
                       format_type: ReportFormat = ReportFormat.TERMINAL,
                       config: Optional[ReportConfig] = None,
                       output_path: Optional[Path] = None) -> str:
        """
        Generate a comprehensive benchmark report.
        
        Args:
            results: Benchmark result(s) to generate report from
            format_type: Output format for the report
            config: Report configuration options
            output_path: Optional path to save the report
            
        Returns:
            Generated report as string
        """
        config = config or ReportConfig()
        
        # Normalize input to list
        if isinstance(results, BenchmarkResult):
            results = [results]
        
        logger.info(f"Generating {format_type.value} report for {len(results)} benchmark(s)")
        
        # Generate report content based on format
        if format_type == ReportFormat.TERMINAL:
            report_content = self._generate_terminal_report(results, config)
        elif format_type == ReportFormat.MARKDOWN:
            report_content = self._generate_markdown_report(results, config)
        elif format_type == ReportFormat.JSON:
            report_content = self._generate_json_report(results, config)
        elif format_type == ReportFormat.CSV:
            report_content = self._generate_csv_report(results, config)
        elif format_type == ReportFormat.HTML:
            report_content = self._generate_html_report(results, config)
        else:
            raise ValueError(f"Unsupported report format: {format_type}")
        
        # Save to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"Report saved to {output_path}")
        
        return report_content
    
    def _generate_terminal_report(self, results: List[BenchmarkResult], config: ReportConfig) -> str:
        """Generate a terminal report using Rich."""
        if not RICH_AVAILABLE:
            return self._generate_plain_text_report(results, config)
        
        # Create console for string capture
        console = Console(file=io.StringIO(), width=120, legacy_windows=False)
        
        # Executive Summary
        if config.include_summary:
            self._add_terminal_summary(console, results)
        
        # Individual model reports
        for result in results:
            if result.success and result.metrics:
                self._add_terminal_model_report(console, result, config)
            else:
                self._add_terminal_failure_report(console, result)
        
        # Comparison section if multiple results
        if len(results) > 1 and config.include_detailed_metrics:
            self._add_terminal_comparison(console, results, config)
        
        return console.file.getvalue()
    
    def _add_terminal_summary(self, console: Console, results: List[BenchmarkResult]):
        """Add executive summary section to terminal report."""
        successful_results = [r for r in results if r.success and r.metrics]
        
        # Summary table
        summary_table = Table(title="📊 Benchmark Executive Summary", box=box.ROUNDED)
        summary_table.add_column("Metric", style="bold blue")
        summary_table.add_column("Value", justify="right")
        
        if successful_results:
            best_accuracy = max(successful_results, key=lambda r: r.metrics.accuracy.overall_accuracy)
            best_cost = min(successful_results, key=lambda r: r.metrics.cost.cost_per_correct_answer)
            fastest = min(successful_results, key=lambda r: r.metrics.performance.mean_response_time)
            
            summary_table.add_row("Models Tested", str(len(results)))
            summary_table.add_row("Successful Runs", str(len(successful_results)))
            summary_table.add_row("Best Accuracy", f"{best_accuracy.model_name} ({best_accuracy.metrics.accuracy.overall_accuracy:.1%})")
            summary_table.add_row("Most Cost-Effective", f"{best_cost.model_name} (${best_cost.metrics.cost.cost_per_correct_answer:.4f})")
            summary_table.add_row("Fastest", f"{fastest.model_name} ({fastest.metrics.performance.mean_response_time:.2f}s)")
            
            total_cost = sum(r.metrics.cost.total_cost for r in successful_results)
            total_questions = sum(r.metrics.accuracy.total_count for r in successful_results)
            summary_table.add_row("Total Cost", f"${total_cost:.4f}")
            summary_table.add_row("Total Questions", str(total_questions))
        else:
            summary_table.add_row("Status", "[red]All benchmarks failed[/red]")
        
        console.print(summary_table)
        console.print()
    
    def _add_terminal_model_report(self, console: Console, result: BenchmarkResult, config: ReportConfig):
        """Add individual model report section."""
        metrics = result.metrics
        
        # Model header
        console.print(Panel(f"🤖 {result.model_name}", style="bold green"))
        
        # Key metrics table
        metrics_table = Table(title="Key Metrics", box=box.SIMPLE)
        metrics_table.add_column("Metric", style="bold")
        metrics_table.add_column("Value", justify="right")
        metrics_table.add_column("Details")
        
        # Accuracy
        accuracy = metrics.accuracy.overall_accuracy
        accuracy_color = "green" if accuracy >= 0.8 else "yellow" if accuracy >= 0.6 else "red"
        metrics_table.add_row(
            "Accuracy",
            f"[{accuracy_color}]{accuracy:.1%}[/{accuracy_color}]",
            f"{metrics.accuracy.correct_count}/{metrics.accuracy.total_count} correct"
        )
        
        # Performance
        response_time = metrics.performance.mean_response_time
        time_color = "green" if response_time <= 2.0 else "yellow" if response_time <= 5.0 else "red"
        metrics_table.add_row(
            "Avg Response Time",
            f"[{time_color}]{response_time:.2f}s[/{time_color}]",
            f"P95: {metrics.performance.p95_response_time:.2f}s"
        )
        
        # Cost
        cost_per_correct = metrics.cost.cost_per_correct_answer
        cost_color = "green" if cost_per_correct <= 0.01 else "yellow" if cost_per_correct <= 0.05 else "red"
        metrics_table.add_row(
            "Cost per Correct",
            f"[{cost_color}]${cost_per_correct:.4f}[/{cost_color}]",
            f"Total: ${metrics.cost.total_cost:.4f}"
        )
        
        # Overall Score
        overall = metrics.overall_score
        overall_color = "green" if overall >= 0.8 else "yellow" if overall >= 0.6 else "red"
        metrics_table.add_row(
            "Overall Score",
            f"[{overall_color}]{overall:.3f}[/{overall_color}]",
            "Composite performance score"
        )
        
        console.print(metrics_table)
        
        # Category breakdown if requested
        if config.include_category_breakdown and metrics.accuracy.by_category:
            category_table = Table(title="Performance by Category", box=box.SIMPLE)
            category_table.add_column("Category", style="bold")
            category_table.add_column("Accuracy", justify="right")
            category_table.add_column("Performance")
            
            for category, accuracy in sorted(metrics.accuracy.by_category.items()):
                bar_length = int(accuracy * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                category_table.add_row(category, f"{accuracy:.1%}", f"[green]{bar}[/green]")
            
            console.print(category_table)
        
        console.print()
    
    def _add_terminal_failure_report(self, console: Console, result: BenchmarkResult):
        """Add failure report for a benchmark that didn't complete successfully."""
        console.print(Panel(f"❌ {result.model_name} - FAILED", style="bold red"))
        
        failure_table = Table(box=box.SIMPLE)
        failure_table.add_column("Detail", style="bold")
        failure_table.add_column("Value")
        
        failure_table.add_row("Error", result.error_message or "Unknown error")
        failure_table.add_row("Execution Time", f"{result.execution_time:.2f}s")
        
        if result.progress:
            failure_table.add_row("Progress", f"{result.progress.completion_percentage:.1f}%")
            failure_table.add_row("Completed", f"{result.progress.completed_questions}/{result.progress.total_questions}")
        
        console.print(failure_table)
        console.print()
    
    def _add_terminal_comparison(self, console: Console, results: List[BenchmarkResult], config: ReportConfig):
        """Add model comparison section."""
        successful_results = [r for r in results if r.success and r.metrics]
        
        if len(successful_results) < 2:
            return
        
        console.print(Panel("📈 Model Comparison", style="bold blue"))
        
        # Comparison table
        comparison_table = Table(title="Detailed Comparison", box=box.ROUNDED)
        comparison_table.add_column("Model", style="bold")
        comparison_table.add_column("Accuracy", justify="right")
        comparison_table.add_column("Avg Time", justify="right")
        comparison_table.add_column("Cost/Correct", justify="right")
        comparison_table.add_column("Overall Score", justify="right")
        comparison_table.add_column("Rank", justify="center")
        
        # Sort by overall score
        sorted_results = sorted(successful_results, key=lambda r: r.metrics.overall_score, reverse=True)
        
        for i, result in enumerate(sorted_results):
            metrics = result.metrics
            
            # Color coding based on rank
            rank_style = "gold" if i == 0 else "silver" if i == 1 else "white"
            rank_medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}"
            
            comparison_table.add_row(
                f"[{rank_style}]{result.model_name}[/{rank_style}]",
                f"{metrics.accuracy.overall_accuracy:.1%}",
                f"{metrics.performance.mean_response_time:.2f}s",
                f"${metrics.cost.cost_per_correct_answer:.4f}",
                f"{metrics.overall_score:.3f}",
                rank_medal
            )
        
        console.print(comparison_table)
        console.print()
    
    def _generate_plain_text_report(self, results: List[BenchmarkResult], config: ReportConfig) -> str:
        """Generate a plain text report (fallback when Rich is not available)."""
        lines = []
        lines.append("=" * 80)
        lines.append("BENCHMARK REPORT")
        lines.append("=" * 80)
        lines.append()
        
        successful_results = [r for r in results if r.success and r.metrics]
        
        # Summary
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Models Tested: {len(results)}")
        lines.append(f"Successful Runs: {len(successful_results)}")
        lines.append()
        
        # Individual results
        for result in results:
            lines.append(f"MODEL: {result.model_name}")
            lines.append("-" * 40)
            
            if result.success and result.metrics:
                metrics = result.metrics
                lines.append(f"Accuracy: {metrics.accuracy.overall_accuracy:.1%} ({metrics.accuracy.correct_count}/{metrics.accuracy.total_count})")
                lines.append(f"Avg Response Time: {metrics.performance.mean_response_time:.2f}s")
                lines.append(f"Cost per Correct: ${metrics.cost.cost_per_correct_answer:.4f}")
                lines.append(f"Total Cost: ${metrics.cost.total_cost:.4f}")
                lines.append(f"Overall Score: {metrics.overall_score:.3f}")
            else:
                lines.append(f"FAILED: {result.error_message or 'Unknown error'}")
                lines.append(f"Execution Time: {result.execution_time:.2f}s")
            
            lines.append()
        
        return "\n".join(lines)
    
    def _generate_markdown_report(self, results: List[BenchmarkResult], config: ReportConfig) -> str:
        """Generate a Markdown report."""
        lines = []
        
        # Title
        lines.append("# Jeopardy Benchmark Report")
        lines.append()
        lines.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append()
        
        successful_results = [r for r in results if r.success and r.metrics]
        
        # Executive Summary
        if config.include_summary:
            lines.append("## Executive Summary")
            lines.append()
            lines.append(f"- **Models Tested**: {len(results)}")
            lines.append(f"- **Successful Runs**: {len(successful_results)}")
            
            if successful_results:
                best_accuracy = max(successful_results, key=lambda r: r.metrics.accuracy.overall_accuracy)
                best_cost = min(successful_results, key=lambda r: r.metrics.cost.cost_per_correct_answer)
                fastest = min(successful_results, key=lambda r: r.metrics.performance.mean_response_time)
                
                lines.append(f"- **Best Accuracy**: {best_accuracy.model_name} ({best_accuracy.metrics.accuracy.overall_accuracy:.1%})")
                lines.append(f"- **Most Cost-Effective**: {best_cost.model_name} (${best_cost.metrics.cost.cost_per_correct_answer:.4f})")
                lines.append(f"- **Fastest**: {fastest.model_name} ({fastest.metrics.performance.mean_response_time:.2f}s)")
            
            lines.append()
        
        # Individual model results
        for result in results:
            lines.append(f"## {result.model_name}")
            lines.append()
            
            if result.success and result.metrics:
                metrics = result.metrics
                
                # Key metrics table
                lines.append("### Key Metrics")
                lines.append()
                lines.append("| Metric | Value | Details |")
                lines.append("|--------|-------|---------|")
                lines.append(f"| Accuracy | {metrics.accuracy.overall_accuracy:.1%} | {metrics.accuracy.correct_count}/{metrics.accuracy.total_count} correct |")
                lines.append(f"| Avg Response Time | {metrics.performance.mean_response_time:.2f}s | P95: {metrics.performance.p95_response_time:.2f}s |")
                lines.append(f"| Cost per Correct | ${metrics.cost.cost_per_correct_answer:.4f} | Total: ${metrics.cost.total_cost:.4f} |")
                lines.append(f"| Overall Score | {metrics.overall_score:.3f} | Composite performance |")
                lines.append()
                
                # Category breakdown
                if config.include_category_breakdown and metrics.accuracy.by_category:
                    lines.append("### Performance by Category")
                    lines.append()
                    lines.append("| Category | Accuracy |")
                    lines.append("|----------|----------|")
                    
                    for category, accuracy in sorted(metrics.accuracy.by_category.items()):
                        lines.append(f"| {category} | {accuracy:.1%} |")
                    
                    lines.append()
                
                # Detailed metrics
                if config.include_detailed_metrics:
                    lines.append("### Detailed Metrics")
                    lines.append()
                    lines.append("**Performance:**")
                    lines.append(f"- Mean Response Time: {metrics.performance.mean_response_time:.3f}s")
                    lines.append(f"- Median Response Time: {metrics.performance.median_response_time:.3f}s")
                    lines.append(f"- 95th Percentile: {metrics.performance.p95_response_time:.3f}s")
                    lines.append(f"- Error Count: {metrics.performance.error_count}")
                    lines.append()
                    
                    lines.append("**Cost Analysis:**")
                    lines.append(f"- Total Cost: ${metrics.cost.total_cost:.4f}")
                    lines.append(f"- Cost per Question: ${metrics.cost.cost_per_question:.4f}")
                    lines.append(f"- Total Tokens: {metrics.cost.total_tokens:,}")
                    lines.append(f"- Tokens per Question: {metrics.cost.tokens_per_question:.1f}")
                    lines.append()
                    
                    lines.append("**Quality Metrics:**")
                    lines.append(f"- Quality Score: {metrics.quality_score:.3f}")
                    lines.append(f"- Consistency Score: {metrics.consistency.confidence_correlation:.3f}")
                    lines.append(f"- Performance Variance: {metrics.consistency.performance_variance:.3f}")
                    lines.append()
            
            else:
                lines.append("### ❌ Benchmark Failed")
                lines.append()
                lines.append(f"**Error**: {result.error_message or 'Unknown error'}")
                lines.append(f"**Execution Time**: {result.execution_time:.2f}s")
                lines.append()
                
                if result.progress:
                    lines.append(f"**Progress**: {result.progress.completion_percentage:.1f}%")
                    lines.append(f"**Completed**: {result.progress.completed_questions}/{result.progress.total_questions}")
                    lines.append()
        
        # Model comparison
        if len(successful_results) > 1:
            lines.append("## Model Comparison")
            lines.append()
            
            lines.append("| Model | Accuracy | Avg Time | Cost/Correct | Overall Score | Rank |")
            lines.append("|-------|----------|----------|--------------|---------------|------|")
            
            sorted_results = sorted(successful_results, key=lambda r: r.metrics.overall_score, reverse=True)
            
            for i, result in enumerate(sorted_results):
                metrics = result.metrics
                rank_medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}"
                
                lines.append(f"| {result.model_name} | {metrics.accuracy.overall_accuracy:.1%} | "
                           f"{metrics.performance.mean_response_time:.2f}s | "
                           f"${metrics.cost.cost_per_correct_answer:.4f} | "
                           f"{metrics.overall_score:.3f} | {rank_medal} |")
            
            lines.append()
        
        return "\n".join(lines)
    
    def _generate_json_report(self, results: List[BenchmarkResult], config: ReportConfig) -> str:
        """Generate a JSON report."""
        report_data = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "format": "json",
                "models_tested": len(results),
                "successful_runs": len([r for r in results if r.success])
            },
            "results": []
        }
        
        for result in results:
            result_data = {
                "model_name": result.model_name,
                "benchmark_id": result.benchmark_id,
                "success": result.success,
                "execution_time": result.execution_time,
                "config": {
                    "mode": result.config.mode.value,
                    "sample_size": result.config.sample_size,
                    "timeout_seconds": result.config.timeout_seconds
                }
            }
            
            if result.error_message:
                result_data["error_message"] = result.error_message
            
            if result.progress:
                result_data["progress"] = {
                    "completion_percentage": result.progress.completion_percentage,
                    "success_rate": result.progress.success_rate,
                    "total_questions": result.progress.total_questions,
                    "completed_questions": result.progress.completed_questions
                }
            
            if result.success and result.metrics:
                metrics = result.metrics
                result_data["metrics"] = {
                    "accuracy": {
                        "overall_accuracy": metrics.accuracy.overall_accuracy,
                        "correct_count": metrics.accuracy.correct_count,
                        "total_count": metrics.accuracy.total_count,
                        "by_category": metrics.accuracy.by_category,
                        "by_difficulty": metrics.accuracy.by_difficulty,
                        "by_value": metrics.accuracy.by_value
                    },
                    "performance": {
                        "mean_response_time": metrics.performance.mean_response_time,
                        "median_response_time": metrics.performance.median_response_time,
                        "p95_response_time": metrics.performance.p95_response_time,
                        "error_count": metrics.performance.error_count,
                        "timeout_count": metrics.performance.timeout_count
                    },
                    "cost": {
                        "total_cost": metrics.cost.total_cost,
                        "cost_per_question": metrics.cost.cost_per_question,
                        "cost_per_correct_answer": metrics.cost.cost_per_correct_answer,
                        "total_tokens": metrics.cost.total_tokens,
                        "tokens_per_question": metrics.cost.tokens_per_question
                    },
                    "scores": {
                        "overall_score": metrics.overall_score,
                        "quality_score": metrics.quality_score,
                        "efficiency_score": metrics.efficiency_score
                    }
                }
            
            if config.include_raw_data:
                result_data["raw_responses"] = len(result.responses)
                result_data["graded_responses"] = len(result.graded_responses)
            
            report_data["results"].append(result_data)
        
        return json.dumps(report_data, indent=2, default=str)
    
    def _generate_csv_report(self, results: List[BenchmarkResult], config: ReportConfig) -> str:
        """Generate a CSV report."""
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header row
        header = [
            "Model Name", "Success", "Execution Time (s)", "Sample Size",
            "Accuracy", "Correct Count", "Total Count",
            "Mean Response Time (s)", "P95 Response Time (s)", "Error Count",
            "Total Cost ($)", "Cost per Correct ($)", "Total Tokens",
            "Overall Score", "Quality Score", "Efficiency Score"
        ]
        
        if config.include_category_breakdown:
            # We'll add category columns dynamically
            pass
        
        writer.writerow(header)
        
        # Data rows
        for result in results:
            row = [
                result.model_name,
                result.success,
                f"{result.execution_time:.3f}",
                result.config.sample_size if result.config else ""
            ]
            
            if result.success and result.metrics:
                metrics = result.metrics
                row.extend([
                    f"{metrics.accuracy.overall_accuracy:.4f}",
                    metrics.accuracy.correct_count,
                    metrics.accuracy.total_count,
                    f"{metrics.performance.mean_response_time:.3f}",
                    f"{metrics.performance.p95_response_time:.3f}",
                    metrics.performance.error_count,
                    f"{metrics.cost.total_cost:.6f}",
                    f"{metrics.cost.cost_per_correct_answer:.6f}",
                    metrics.cost.total_tokens,
                    f"{metrics.overall_score:.4f}",
                    f"{metrics.quality_score:.4f}",
                    f"{metrics.efficiency_score:.4f}"
                ])
            else:
                # Fill with empty values for failed benchmarks
                row.extend([""] * (len(header) - len(row)))
            
            writer.writerow(row)
        
        return output.getvalue()
    
    def _generate_html_report(self, results: List[BenchmarkResult], config: ReportConfig) -> str:
        """Generate an HTML report."""
        successful_results = [r for r in results if r.success and r.metrics]
        
        html_parts = []
        
        # HTML head
        html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jeopardy Benchmark Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; line-height: 1.6; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1, h2, h3 { color: #333; }
        .summary { background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .model-section { border: 1px solid #ddd; margin: 20px 0; padding: 20px; border-radius: 8px; }
        .success { border-left: 4px solid #28a745; }
        .failure { border-left: 4px solid #dc3545; }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; font-weight: 600; }
        .metric-good { color: #28a745; font-weight: bold; }
        .metric-warning { color: #ffc107; font-weight: bold; }
        .metric-poor { color: #dc3545; font-weight: bold; }
        .progress-bar { width: 100%; height: 20px; background: #e9ecef; border-radius: 4px; overflow: hidden; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #28a745, #20c997); transition: width 0.3s; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Jeopardy Benchmark Report</h1>
        <p><em>Generated on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</em></p>
""")
        
        # Summary section
        if config.include_summary:
            html_parts.append('<div class="summary">')
            html_parts.append('<h2>Executive Summary</h2>')
            html_parts.append('<table>')
            html_parts.append('<tr><th>Metric</th><th>Value</th></tr>')
            html_parts.append(f'<tr><td>Models Tested</td><td>{len(results)}</td></tr>')
            html_parts.append(f'<tr><td>Successful Runs</td><td>{len(successful_results)}</td></tr>')
            
            if successful_results:
                best_accuracy = max(successful_results, key=lambda r: r.metrics.accuracy.overall_accuracy)
                best_cost = min(successful_results, key=lambda r: r.metrics.cost.cost_per_correct_answer)
                fastest = min(successful_results, key=lambda r: r.metrics.performance.mean_response_time)
                
                html_parts.append(f'<tr><td>Best Accuracy</td><td>{best_accuracy.model_name} ({best_accuracy.metrics.accuracy.overall_accuracy:.1%})</td></tr>')
                html_parts.append(f'<tr><td>Most Cost-Effective</td><td>{best_cost.model_name} (${best_cost.metrics.cost.cost_per_correct_answer:.4f})</td></tr>')
                html_parts.append(f'<tr><td>Fastest</td><td>{fastest.model_name} ({fastest.metrics.performance.mean_response_time:.2f}s)</td></tr>')
            
            html_parts.append('</table>')
            html_parts.append('</div>')
        
        # Individual model sections
        for result in results:
            css_class = "model-section success" if result.success else "model-section failure"
            html_parts.append(f'<div class="{css_class}">')
            
            if result.success and result.metrics:
                metrics = result.metrics
                html_parts.append(f'<h2>🤖 {result.model_name}</h2>')
                
                # Key metrics
                html_parts.append('<h3>Key Metrics</h3>')
                html_parts.append('<table>')
                html_parts.append('<tr><th>Metric</th><th>Value</th><th>Details</th></tr>')
                
                accuracy = metrics.accuracy.overall_accuracy
                accuracy_class = "metric-good" if accuracy >= 0.8 else "metric-warning" if accuracy >= 0.6 else "metric-poor"
                html_parts.append(f'<tr><td>Accuracy</td><td class="{accuracy_class}">{accuracy:.1%}</td><td>{metrics.accuracy.correct_count}/{metrics.accuracy.total_count} correct</td></tr>')
                
                response_time = metrics.performance.mean_response_time
                time_class = "metric-good" if response_time <= 2.0 else "metric-warning" if response_time <= 5.0 else "metric-poor"
                html_parts.append(f'<tr><td>Avg Response Time</td><td class="{time_class}">{response_time:.2f}s</td><td>P95: {metrics.performance.p95_response_time:.2f}s</td></tr>')
                
                cost = metrics.cost.cost_per_correct_answer
                cost_class = "metric-good" if cost <= 0.01 else "metric-warning" if cost <= 0.05 else "metric-poor"
                html_parts.append(f'<tr><td>Cost per Correct</td><td class="{cost_class}">${cost:.4f}</td><td>Total: ${metrics.cost.total_cost:.4f}</td></tr>')
                
                overall = metrics.overall_score
                overall_class = "metric-good" if overall >= 0.8 else "metric-warning" if overall >= 0.6 else "metric-poor"
                html_parts.append(f'<tr><td>Overall Score</td><td class="{overall_class}">{overall:.3f}</td><td>Composite performance</td></tr>')
                
                html_parts.append('</table>')
                
                # Category breakdown
                if config.include_category_breakdown and metrics.accuracy.by_category:
                    html_parts.append('<h3>Performance by Category</h3>')
                    html_parts.append('<table>')
                    html_parts.append('<tr><th>Category</th><th>Accuracy</th><th>Visual</th></tr>')
                    
                    for category, accuracy in sorted(metrics.accuracy.by_category.items()):
                        html_parts.append(f'<tr><td>{category}</td><td>{accuracy:.1%}</td>')
                        html_parts.append(f'<td><div class="progress-bar"><div class="progress-fill" style="width: {accuracy*100}%"></div></div></td></tr>')
                    
                    html_parts.append('</table>')
                
            else:
                html_parts.append(f'<h2>❌ {result.model_name} - FAILED</h2>')
                html_parts.append('<table>')
                html_parts.append('<tr><th>Detail</th><th>Value</th></tr>')
                html_parts.append(f'<tr><td>Error</td><td>{result.error_message or "Unknown error"}</td></tr>')
                html_parts.append(f'<tr><td>Execution Time</td><td>{result.execution_time:.2f}s</td></tr>')
                html_parts.append('</table>')
            
            html_parts.append('</div>')
        
        # Comparison section
        if len(successful_results) > 1:
            html_parts.append('<div class="summary">')
            html_parts.append('<h2>📈 Model Comparison</h2>')
            html_parts.append('<table>')
            html_parts.append('<tr><th>Model</th><th>Accuracy</th><th>Avg Time</th><th>Cost/Correct</th><th>Overall Score</th><th>Rank</th></tr>')
            
            sorted_results = sorted(successful_results, key=lambda r: r.metrics.overall_score, reverse=True)
            
            for i, result in enumerate(sorted_results):
                metrics = result.metrics
                rank_medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}"
                
                html_parts.append(f'<tr>')
                html_parts.append(f'<td><strong>{result.model_name}</strong></td>')
                html_parts.append(f'<td>{metrics.accuracy.overall_accuracy:.1%}</td>')
                html_parts.append(f'<td>{metrics.performance.mean_response_time:.2f}s</td>')
                html_parts.append(f'<td>${metrics.cost.cost_per_correct_answer:.4f}</td>')
                html_parts.append(f'<td>{metrics.overall_score:.3f}</td>')
                html_parts.append(f'<td>{rank_medal}</td>')
                html_parts.append('</tr>')
            
            html_parts.append('</table>')
            html_parts.append('</div>')
        
        # HTML footer
        html_parts.append("""
    </div>
</body>
</html>""")
        
        return "".join(html_parts)
    
    def display_terminal_report(self, results: Union[BenchmarkResult, List[BenchmarkResult]],
                               config: Optional[ReportConfig] = None):
        """Display report directly to terminal using Rich."""
        if not RICH_AVAILABLE or not self.console:
            # Fallback to plain print
            report = self.generate_report(results, ReportFormat.TERMINAL, config)
            print(report)
            return
        
        config = config or ReportConfig()
        
        # Normalize to list
        if isinstance(results, BenchmarkResult):
            results = [results]
        
        # Generate and display
        if config.include_summary:
            self._add_terminal_summary(self.console, results)
        
        for result in results:
            if result.success and result.metrics:
                self._add_terminal_model_report(self.console, result, config)
            else:
                self._add_terminal_failure_report(self.console, result)
        
        if len(results) > 1 and config.include_detailed_metrics:
            self._add_terminal_comparison(self.console, results, config)
    
    def generate_comparison_report(self, results: List[BenchmarkResult],
                                  format_type: ReportFormat = ReportFormat.MARKDOWN,
                                  output_path: Optional[Path] = None) -> str:
        """Generate a focused comparison report for multiple models."""
        successful_results = [r for r in results if r.success and r.metrics]
        
        if len(successful_results) < 2:
            raise ValueError("Need at least 2 successful results for comparison")
        
        # Sort by overall score
        sorted_results = sorted(successful_results, key=lambda r: r.metrics.overall_score, reverse=True)
        
        # Generate comparison-focused config
        comparison_config = ReportConfig(
            include_summary=True,
            include_detailed_metrics=False,
            include_category_breakdown=True,
            include_cost_analysis=True,
            include_performance_charts=False
        )
        
        return self.generate_report(sorted_results, format_type, comparison_config, output_path)
    
    def export_metrics_csv(self, results: List[BenchmarkResult], output_path: Path) -> Path:
        """Export detailed metrics to CSV for analysis."""
        csv_content = self._generate_csv_report(results, ReportConfig(include_raw_data=True))
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            f.write(csv_content)
        
        logger.info(f"Metrics exported to {output_path}")
        return output_path