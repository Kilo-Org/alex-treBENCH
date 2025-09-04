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

from src.evaluation.metrics import ComprehensiveMetrics
from src.utils.logging import get_logger

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
    """A section of a report."""
    title: str
    content: Any
    order: int = 0


@dataclass 
class ReportConfig:
    """Configuration for report generation."""
    include_detailed_metrics: bool = True
    include_category_breakdown: bool = True
    include_difficulty_analysis: bool = True
    include_cost_analysis: bool = True
    include_response_samples: bool = False
    show_jeopardy_scores: bool = True  # Show Jeopardy scoring prominently
    show_leaderboard: bool = True  # Show Jeopardy leaderboard for multiple models
    max_response_samples: int = 10
    output_path: Optional[Path] = None


class ReportGenerator:
    """Generates comprehensive benchmark reports in multiple formats."""
    
    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
        self.console = Console() if RICH_AVAILABLE and Console is not None else None
    
    def generate_report(self, result, format_type: ReportFormat, output_path: Optional[Path] = None) -> str:
        """Generate a report in the specified format."""
        # Use local import to avoid circular dependency
        from .runner import BenchmarkRunResult
        
        if format_type == ReportFormat.TERMINAL:
            return self._generate_terminal_report(result)
        elif format_type == ReportFormat.MARKDOWN:
            return self._generate_markdown_report(result)
        elif format_type == ReportFormat.JSON:
            return self._generate_json_report(result)
        else:
            return f"Report format {format_type} not yet implemented"
    
    def _generate_terminal_report(self, result) -> str:
        """Generate terminal report using Rich."""
        if not RICH_AVAILABLE:
            return self._generate_plain_terminal_report(result)
        
        # Create Rich layout
        layout = Layout()
        
        sections = []
        
        # Header section
        header_content = f"[bold blue]🎯 Benchmark Report: {getattr(result, 'model_name', 'Unknown Model')}[/bold blue]\n"
        header_content += f"Benchmark ID: {getattr(result, 'benchmark_id', 'N/A')}\n"
        header_content += f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        sections.append(Panel(header_content, title="Summary", box=box.ROUNDED))
        
        # Jeopardy Score section (prominent display)
        if self.config.show_jeopardy_scores and hasattr(result, 'metrics') and result.metrics:
            jeopardy_content = self._create_jeopardy_score_display(result.metrics)
            sections.append(Panel(jeopardy_content, title="🏆 Jeopardy Scores", box=box.DOUBLE))
        
        # Performance metrics section
        if hasattr(result, 'metrics') and result.metrics:
            perf_content = self._create_performance_summary(result.metrics)
            sections.append(Panel(perf_content, title="📊 Performance Metrics", box=box.ROUNDED))
        
        # Category breakdown
        if self.config.include_category_breakdown and hasattr(result, 'metrics') and result.metrics:
            category_content = self._create_category_breakdown(result.metrics)
            sections.append(Panel(category_content, title="📁 Category Performance", box=box.ROUNDED))
        
        # Combine all sections
        report = "\n\n".join([str(section) for section in sections])
        return report
    
    def _create_jeopardy_score_display(self, metrics) -> str:
        """Create Jeopardy score display section."""
        if not hasattr(metrics, 'jeopardy_score'):
            return "Jeopardy scoring not available"
        
        js = metrics.jeopardy_score
        content = []
        
        # Main score with formatting
        score = js.total_jeopardy_score
        score_color = "green" if score >= 0 else "red"
        content.append(f"[bold {score_color}]Total Score: ${score:,}[/bold {score_color}]")
        
        # Breakdown
        content.append(f"✅ Correct Answers: {js.positive_scores} questions")
        content.append(f"❌ Incorrect Answers: {js.negative_scores} questions")
        
        # Category scores (top 5)
        if js.category_scores:
            content.append("\n[bold]Top Category Scores:[/bold]")
            sorted_categories = sorted(js.category_scores.items(), key=lambda x: x[1], reverse=True)
            for cat, cat_score in sorted_categories[:5]:
                cat_color = "green" if cat_score >= 0 else "red"
                content.append(f"  • {cat}: [bold {cat_color}]${cat_score:,}[/bold {cat_color}]")
        
        # Value range performance
        if js.value_range_scores:
            content.append("\n[bold]Performance by Value Range:[/bold]")
            for range_name, range_score in js.value_range_scores.items():
                range_color = "green" if range_score >= 0 else "red"
                content.append(f"  • {range_name}: [bold {range_color}]${range_score:,}[/bold {range_color}]")
        
        return "\n".join(content)
    
    def _create_performance_summary(self, metrics) -> str:
        """Create performance summary section."""
        content = []
        
        if hasattr(metrics, 'accuracy'):
            content.append(f"🎯 Accuracy: {metrics.accuracy.overall_accuracy:.1%} ({metrics.accuracy.correct_count}/{metrics.accuracy.total_count})")
        
        if hasattr(metrics, 'performance'):
            content.append(f"⚡ Avg Response Time: {metrics.performance.mean_response_time:.2f}s")
        
        if hasattr(metrics, 'cost'):
            content.append(f"💰 Total Cost: ${metrics.cost.total_cost:.4f}")
            content.append(f"💸 Cost per Question: ${metrics.cost.cost_per_question:.6f}")
            content.append(f"🎪 Cost per Correct Answer: ${metrics.cost.cost_per_correct_answer:.6f}")
        
        if hasattr(metrics, 'overall_score'):
            content.append(f"⭐ Overall Score: {metrics.overall_score:.3f}/1.000")
        
        return "\n".join(content)
    
    def _create_category_breakdown(self, metrics) -> str:
        """Create category breakdown section."""
        if not hasattr(metrics, 'accuracy') or not hasattr(metrics.accuracy, 'by_category'):
            return "Category breakdown not available"
        
        content = []
        content.append("[bold]Accuracy by Category:[/bold]")
        
        for category, accuracy in metrics.accuracy.by_category.items():
            jeopardy_score = ""
            if hasattr(metrics, 'jeopardy_score') and metrics.jeopardy_score.category_scores:
                cat_score = metrics.jeopardy_score.category_scores.get(category, 0)
                score_color = "green" if cat_score >= 0 else "red"
                jeopardy_score = f" | [bold {score_color}]${cat_score:,}[/bold {score_color}]"
            
            content.append(f"  • {category}: {accuracy:.1%}{jeopardy_score}")
        
        return "\n".join(content)
    
    def _generate_plain_terminal_report(self, result) -> str:
        """Generate plain text terminal report for when Rich is not available."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"🎯 BENCHMARK REPORT: {getattr(result, 'model_name', 'Unknown Model')}")
        lines.append("=" * 60)
        lines.append(f"Benchmark ID: {getattr(result, 'benchmark_id', 'N/A')}")
        lines.append(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Jeopardy Scores
        if self.config.show_jeopardy_scores and hasattr(result, 'metrics') and result.metrics:
            lines.append("🏆 JEOPARDY SCORES")
            lines.append("-" * 20)
            js = result.metrics.jeopardy_score
            lines.append(f"Total Score: ${js.total_jeopardy_score:,}")
            lines.append(f"Correct Answers: {js.positive_scores} questions")
            lines.append(f"Incorrect Answers: {js.negative_scores} questions")
            lines.append("")
        
        # Performance metrics
        if hasattr(result, 'metrics') and result.metrics:
            lines.append("📊 PERFORMANCE METRICS")
            lines.append("-" * 25)
            m = result.metrics
            lines.append(f"Accuracy: {m.accuracy.overall_accuracy:.1%}")
            lines.append(f"Avg Response Time: {m.performance.mean_response_time:.2f}s")
            lines.append(f"Total Cost: ${m.cost.total_cost:.4f}")
            lines.append(f"Overall Score: {m.overall_score:.3f}/1.000")
            lines.append("")
        
        return "\n".join(lines)

    def generate_leaderboard_report(self, results: List, format_type: ReportFormat = ReportFormat.TERMINAL) -> str:
        """Generate Jeopardy leaderboard report for multiple models."""
        if not results or not self.config.show_leaderboard:
            return "No results available for leaderboard"
        
        # Sort results by Jeopardy score
        sorted_results = sorted(
            [r for r in results if hasattr(r, 'metrics') and r.metrics and hasattr(r.metrics, 'jeopardy_score')],
            key=lambda x: x.metrics.jeopardy_score.total_jeopardy_score,
            reverse=True
        )
        
        if format_type == ReportFormat.TERMINAL:
            return self._generate_terminal_leaderboard(sorted_results)
        elif format_type == ReportFormat.MARKDOWN:
            return self._generate_markdown_leaderboard(sorted_results)
        else:
            return self._generate_plain_leaderboard(sorted_results)
    
    def _generate_markdown_report(self, result) -> str:
        """Generate Markdown report."""
        lines = []
        lines.append(f"# 🎯 Benchmark Report: {getattr(result, 'model_name', 'Unknown Model')}")
        lines.append("")
        lines.append(f"**Benchmark ID:** {getattr(result, 'benchmark_id', 'N/A')}")
        lines.append(f"**Completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Jeopardy Scores Section
        if self.config.show_jeopardy_scores and hasattr(result, 'metrics') and result.metrics:
            lines.append("## 🏆 Jeopardy Scores")
            lines.append("")
            js = result.metrics.jeopardy_score
            lines.append(f"**Total Score:** ${js.total_jeopardy_score:,}")
            lines.append(f"**Correct Answers:** {js.positive_scores} questions")
            lines.append(f"**Incorrect Answers:** {js.negative_scores} questions")
            lines.append("")
            
            if js.category_scores:
                lines.append("### Category Scores")
                lines.append("")
                sorted_categories = sorted(js.category_scores.items(), key=lambda x: x[1], reverse=True)
                for cat, score in sorted_categories:
                    lines.append(f"- **{cat}:** ${score:,}")
                lines.append("")
        
        # Performance Metrics Section
        if hasattr(result, 'metrics') and result.metrics:
            lines.append("## 📊 Performance Metrics")
            lines.append("")
            m = result.metrics
            lines.append(f"- **Accuracy:** {m.accuracy.overall_accuracy:.1%} ({m.accuracy.correct_count}/{m.accuracy.total_count})")
            lines.append(f"- **Average Response Time:** {m.performance.mean_response_time:.2f}s")
            lines.append(f"- **Total Cost:** ${m.cost.total_cost:.4f}")
            lines.append(f"- **Cost per Question:** ${m.cost.cost_per_question:.6f}")
            lines.append(f"- **Overall Score:** {m.overall_score:.3f}/1.000")
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_json_report(self, result) -> str:
        """Generate JSON report."""
        report_data = {
            "benchmark_id": getattr(result, 'benchmark_id', 'N/A'),
            "model_name": getattr(result, 'model_name', 'Unknown Model'),
            "timestamp": datetime.now().isoformat(),
            "success": getattr(result, 'success', False),
            "execution_time": getattr(result, 'execution_time', 0)
        }
        
        # Add metrics if available
        if hasattr(result, 'metrics') and result.metrics:
            m = result.metrics
            report_data["metrics"] = {
                "accuracy": {
                    "overall_accuracy": m.accuracy.overall_accuracy,
                    "correct_count": m.accuracy.correct_count,
                    "total_count": m.accuracy.total_count,
                    "by_category": m.accuracy.by_category
                },
                "performance": {
                    "mean_response_time": m.performance.mean_response_time,
                    "median_response_time": m.performance.median_response_time,
                    "error_count": m.performance.error_count
                },
                "cost": {
                    "total_cost": m.cost.total_cost,
                    "cost_per_question": m.cost.cost_per_question,
                    "cost_per_correct_answer": m.cost.cost_per_correct_answer,
                    "total_tokens": m.cost.total_tokens
                },
                "overall_score": m.overall_score,
                "quality_score": m.quality_score,
                "efficiency_score": m.efficiency_score
            }
            
            # Add Jeopardy scores
            if hasattr(m, 'jeopardy_score'):
                report_data["metrics"]["jeopardy_score"] = {
                    "total_jeopardy_score": m.jeopardy_score.total_jeopardy_score,
                    "positive_scores": m.jeopardy_score.positive_scores,
                    "negative_scores": m.jeopardy_score.negative_scores,
                    "category_scores": m.jeopardy_score.category_scores,
                    "difficulty_scores": m.jeopardy_score.difficulty_scores,
                    "value_range_scores": m.jeopardy_score.value_range_scores
                }
        
        return json.dumps(report_data, indent=2)
    
    def _generate_terminal_leaderboard(self, sorted_results) -> str:
        """Generate Rich terminal leaderboard."""
        if not RICH_AVAILABLE or not sorted_results or Table is None:
            return self._generate_plain_leaderboard(sorted_results)
        
        # Create leaderboard table
        table = Table(title="🏆 Jeopardy Leaderboard", box=box.ROUNDED)
        table.add_column("Rank", style="cyan", no_wrap=True)
        table.add_column("Model", style="magenta")
        table.add_column("Jeopardy Score", style="green", justify="right")
        table.add_column("Accuracy", style="blue", justify="right")
        table.add_column("Correct", style="green", justify="right")
        table.add_column("Incorrect", style="red", justify="right")
        
        for i, result in enumerate(sorted_results[:10], 1):  # Top 10
            metrics = result.metrics
            js = metrics.jeopardy_score
            
            # Determine rank emoji
            rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else str(i)
            
            # Format score with color
            score = js.total_jeopardy_score
            score_text = f"${score:,}"
            if score < 0:
                score_text = f"[red]{score_text}[/red]"
            else:
                score_text = f"[green]{score_text}[/green]"
            
            table.add_row(
                f"{rank_emoji}",
                getattr(result, 'model_name', 'Unknown'),
                score_text,
                f"{metrics.accuracy.overall_accuracy:.1%}",
                str(js.positive_scores),
                str(js.negative_scores)
            )
        
        return table
    
    def _generate_plain_leaderboard(self, sorted_results) -> str:
        """Generate plain text leaderboard."""
        lines = []
        lines.append("🏆 JEOPARDY LEADERBOARD")
        lines.append("=" * 50)
        lines.append(f"{'Rank':<6} {'Model':<25} {'Score':<12} {'Accuracy':<10}")
        lines.append("-" * 50)
        
        for i, result in enumerate(sorted_results[:10], 1):
            metrics = result.metrics
            js = metrics.jeopardy_score
            
            rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else str(i)
            model_name = getattr(result, 'model_name', 'Unknown')[:25]
            score = f"${js.total_jeopardy_score:,}"
            accuracy = f"{metrics.accuracy.overall_accuracy:.1%}"
            
            lines.append(f"{rank_emoji:<6} {model_name:<25} {score:<12} {accuracy:<10}")
        
        return "\n".join(lines)
    
    def _generate_markdown_leaderboard(self, sorted_results) -> str:
        """Generate Markdown leaderboard."""
        lines = []
        lines.append("# 🏆 Jeopardy Leaderboard")
        lines.append("")
        lines.append("| Rank | Model | Jeopardy Score | Accuracy | Correct | Incorrect |")
        lines.append("|------|-------|----------------|----------|---------|-----------|")
        
        for i, result in enumerate(sorted_results[:10], 1):
            metrics = result.metrics
            js = metrics.jeopardy_score
            
            rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else str(i)
            model_name = getattr(result, 'model_name', 'Unknown')
            score = f"${js.total_jeopardy_score:,}"
            accuracy = f"{metrics.accuracy.overall_accuracy:.1%}"
            
            lines.append(f"| {rank_emoji} | {model_name} | {score} | {accuracy} | {js.positive_scores} | {js.negative_scores} |")
        
        lines.append("")
        return "\n".join(lines)
    
    def display_terminal_report(self, result):
        """Display report directly to terminal."""
        if self.console:
            report_content = self._generate_terminal_report(result)
            self.console.print(report_content)
        else:
            print(self._generate_terminal_report(result))
    
    def generate_comparison_report(self, results: List, format_type: ReportFormat, output_path: Optional[Path] = None) -> str:
        """Generate comparison report for multiple results."""
        if not results:
            return "No results provided for comparison"
        
        if format_type == ReportFormat.TERMINAL:
            # Generate leaderboard for terminal
            return self.generate_leaderboard_report(results, ReportFormat.TERMINAL)
        elif format_type == ReportFormat.MARKDOWN:
            # Generate markdown comparison
            lines = []
            lines.append("# 📊 Benchmark Comparison Report")
            lines.append("")
            lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"**Models Compared:** {len(results)}")
            lines.append("")
            
            # Add leaderboard
            lines.append(self.generate_leaderboard_report(results, ReportFormat.MARKDOWN))
            
            # Add detailed comparison
            lines.append("")
            lines.append("## Detailed Comparison")
            lines.append("")
            
            for result in results:
                if hasattr(result, 'metrics') and result.metrics:
                    model_name = getattr(result, 'model_name', 'Unknown')
                    m = result.metrics
                    lines.append(f"### {model_name}")
                    lines.append("")
                    lines.append(f"- **Jeopardy Score:** ${m.jeopardy_score.total_jeopardy_score:,}")
                    lines.append(f"- **Accuracy:** {m.accuracy.overall_accuracy:.1%}")
                    lines.append(f"- **Response Time:** {m.performance.mean_response_time:.2f}s")
                    lines.append(f"- **Total Cost:** ${m.cost.total_cost:.4f}")
                    lines.append("")
            
            return "\n".join(lines)
        else:
            return f"Comparison report format {format_type} not yet implemented"
