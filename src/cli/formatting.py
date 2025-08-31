"""
CLI Output Formatting

Rich text formatting utilities for CLI output including tables, progress bars,
and result formatting for the Jeopardy Benchmarking System.
"""

from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
import json

console = Console()


def format_table(data: List[Dict[str, Any]], title: str = "Results", headers: Optional[List[str]] = None) -> Table:
    """
    Format data as a Rich table.
    
    Args:
        data: List of dictionaries with row data
        title: Table title
        headers: Optional list of column headers (uses keys from first row if not provided)
    
    Returns:
        Rich Table object
    """
    if not data:
        table = Table(title=title)
        table.add_column("Message", style="dim")
        table.add_row("No data available")
        return table
    
    # Use provided headers or extract from first row
    if headers is None:
        headers = list(data[0].keys())
    
    # Create table
    table = Table(title=title, show_header=True, header_style="bold blue")
    
    # Add columns
    for header in headers:
        table.add_column(header, style="white", justify="left")
    
    # Add rows
    for row in data:
        row_values = [str(row.get(header, "N/A")) for header in headers]
        table.add_row(*row_values)
    
    return table


def format_progress(description: str = "Processing...", total: Optional[int] = None) -> Progress:
    """
    Create a progress bar with spinner.
    
    Args:
        description: Progress description text
        total: Total number of items (None for indeterminate progress)
    
    Returns:
        Rich Progress object
    """
    if total is None:
        # Indeterminate progress with spinner
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        )
    else:
        # Determinate progress with bar
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        )
    
    return progress


def format_results(results: Dict[str, Any], format_type: str = "summary") -> str:
    """
    Format benchmark results for display.
    
    Args:
        results: Results dictionary
        format_type: Type of formatting ("summary", "detailed", "json")
    
    Returns:
        Formatted results string
    """
    if format_type == "json":
        return json.dumps(results, indent=2, default=str)
    
    # Create formatted output
    if format_type == "detailed":
        return _format_detailed_results(results)
    else:
        return _format_summary_results(results)


def _format_summary_results(results: Dict[str, Any]) -> str:
    """Format results as a summary."""
    summary = []
    summary.append(f"Benchmark: {results.get('benchmark_name', 'N/A')}")
    summary.append(f"Model: {results.get('model_name', 'N/A')}")
    summary.append(f"Status: {results.get('status', 'N/A')}")
    
    if 'metrics' in results:
        metrics = results['metrics']
        summary.append(f"Accuracy: {metrics.get('accuracy', 0):.1%}")
        summary.append(f"Total Questions: {metrics.get('total_questions', 0)}")
        summary.append(f"Execution Time: {results.get('execution_time', 0):.2f}s")
    
    if 'cost' in results:
        cost = results['cost']
        summary.append(f"Total Cost: ${cost.get('total_cost', 0):.4f}")
    
    return "\n".join(summary)


def _format_detailed_results(results: Dict[str, Any]) -> str:
    """Format results with detailed information."""
    detailed = []
    detailed.append("=" * 50)
    detailed.append(f"BENCHMARK RESULTS: {results.get('benchmark_name', 'N/A')}")
    detailed.append("=" * 50)
    detailed.append(f"Model: {results.get('model_name', 'N/A')}")
    detailed.append(f"Status: {results.get('status', 'N/A')}")
    detailed.append(f"Started: {results.get('start_time', 'N/A')}")
    detailed.append(f"Completed: {results.get('end_time', 'N/A')}")
    detailed.append(f"Duration: {results.get('execution_time', 0):.2f} seconds")
    detailed.append("")
    
    # Metrics section
    if 'metrics' in results:
        metrics = results['metrics']
        detailed.append("METRICS:")
        detailed.append("-" * 20)
        detailed.append(f"Total Questions: {metrics.get('total_questions', 0)}")
        detailed.append(f"Correct Answers: {metrics.get('correct_answers', 0)}")
        detailed.append(f"Overall Accuracy: {metrics.get('accuracy', 0):.3f}")
        detailed.append(f"Average Response Time: {metrics.get('avg_response_time', 0):.3f}s")
        detailed.append("")
    
    # Cost section
    if 'cost' in results:
        cost = results['cost']
        detailed.append("COST BREAKDOWN:")
        detailed.append("-" * 20)
        detailed.append(f"Input Tokens: {cost.get('input_tokens', 0):,}")
        detailed.append(f"Output Tokens: {cost.get('output_tokens', 0):,}")
        detailed.append(f"Total Cost: ${cost.get('total_cost', 0):.6f}")
        detailed.append("")
    
    # Category breakdown
    if 'categories' in results:
        categories = results['categories']
        detailed.append("CATEGORY BREAKDOWN:")
        detailed.append("-" * 20)
        for category, data in categories.items():
            accuracy = data.get('accuracy', 0)
            count = data.get('count', 0)
            detailed.append(f"{category}: {accuracy:.1%} ({count} questions)")
        detailed.append("")
    
    detailed.append("=" * 50)
    return "\n".join(detailed)


def display_banner(title: str = "Jeopardy Benchmarking System", subtitle: str = "") -> None:
    """Display a formatted banner."""
    banner_text = f"[bold blue]{title}[/bold blue]"
    if subtitle:
        banner_text += f"\n[dim]{subtitle}[/dim]"
    
    banner = Panel.fit(
        banner_text,
        title="🧠 LLM Benchmarking",
        border_style="blue"
    )
    console.print(banner)


def display_error(message: str, error_type: str = "Error") -> None:
    """Display a formatted error message."""
    error_panel = Panel(
        f"[red]{message}[/red]",
        title=f"❌ {error_type}",
        border_style="red"
    )
    console.print(error_panel)


def display_success(message: str, title: str = "Success") -> None:
    """Display a formatted success message."""
    success_panel = Panel(
        f"[green]{message}[/green]",
        title=f"✅ {title}",
        border_style="green"
    )
    console.print(success_panel)


def display_info(message: str, title: str = "Info") -> None:
    """Display a formatted info message."""
    info_panel = Panel(
        f"[blue]{message}[/blue]",
        title=f"ℹ️ {title}",
        border_style="blue"
    )
    console.print(info_panel)