"""
Benchmark Status Commands

This module contains the benchmark status command implementation.
"""

import json
import sys
from collections import defaultdict
from datetime import datetime

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.table import Table

from core.database import get_db_session
from utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


def _get_status_display(status):
    """Get colored status display."""
    colors = {
        'pending': 'blue',
        'running': 'yellow',
        'completed': 'green',
        'failed': 'red'
    }
    color = colors.get(status, 'white')
    return f"[{color}]{status.upper()}[/{color}]"


@click.command()
@click.argument('benchmark_id', type=int)
@click.pass_context
def status(ctx, benchmark_id):
    """Show detailed status for a specific benchmark.
    
    \b
    🔍 EXAMPLES:
    
    alex benchmark status 1
    alex benchmark status 42
    
    \b
    📊 Shows comprehensive status including progress, metrics, category performance,
    and model comparisons for the specified benchmark run.
    """
    
    try:
        from storage.repositories import (
            BenchmarkRepository, 
            BenchmarkResultRepository, 
            ModelPerformanceRepository
        )
        from src.storage.models import BenchmarkRun
        
        with get_db_session() as session:
            # Get benchmark details
            benchmark_repo = BenchmarkRepository(session)
            benchmark = benchmark_repo.get_benchmark_by_id(benchmark_id)
            
            if not benchmark:
                console.print(f"[red]❌ Benchmark {benchmark_id} not found[/red]")
                return
            
            # Get related data
            result_repo = BenchmarkResultRepository(session)
            performance_repo = ModelPerformanceRepository(session)
            
            results = result_repo.get_results_by_benchmark(benchmark_id)
            performances = performance_repo.get_performances_by_benchmark(benchmark_id)
            
            # Parse models tested
            models_tested = []
            if hasattr(benchmark, 'models_tested') and benchmark.models_tested:
                try:
                    if isinstance(benchmark.models_tested, str):
                        models_tested = json.loads(benchmark.models_tested)
                    else:
                        models_tested = benchmark.models_tested
                    if not isinstance(models_tested, list):
                        models_tested = [models_tested]
                except (json.JSONDecodeError, TypeError):
                    models_tested = [benchmark.models_tested] if benchmark.models_tested else []
            
            # Create main info panel
            info_content = f"""[bold]ID:[/bold] {benchmark.id}
[bold]Name:[/bold] {getattr(benchmark, 'name', 'N/A')}
[bold]Description:[/bold] {getattr(benchmark, 'description', 'N/A') or 'N/A'}
[bold]Status:[/bold] {_get_status_display(getattr(benchmark, 'status', 'unknown'))}
[bold]Mode:[/bold] {getattr(benchmark, 'benchmark_mode', 'unknown').title()}
[bold]Sample Size:[/bold] {getattr(benchmark, 'sample_size', 0):,}
[bold]Models Tested:[/bold] {len(models_tested)}

[bold]Timeline:[/bold]"""

            # Handle datetime fields safely
            created_at = getattr(benchmark, 'created_at', None)
            started_at = getattr(benchmark, 'started_at', None)  
            completed_at = getattr(benchmark, 'completed_at', None)
            
            if created_at:
                info_content += f"\n  Created: {created_at.strftime('%Y-%m-%d %H:%M:%S') if hasattr(created_at, 'strftime') else str(created_at)}"
            else:
                info_content += "\n  Created: N/A"
                
            if started_at:
                info_content += f"\n  Started: {started_at.strftime('%Y-%m-%d %H:%M:%S') if hasattr(started_at, 'strftime') else str(started_at)}"
            else:
                info_content += "\n  Started: N/A"
                
            if completed_at:
                info_content += f"\n  Completed: {completed_at.strftime('%Y-%m-%d %H:%M:%S') if hasattr(completed_at, 'strftime') else str(completed_at)}"
            else:
                info_content += "\n  Completed: N/A"
            
            console.print(Panel(info_content, title=f"📊 Benchmark {benchmark_id} Details", border_style="blue"))
            
            # Show progress for running benchmarks
            benchmark_status = getattr(benchmark, 'status', 'unknown')
            if benchmark_status == 'running':
                completed_questions = getattr(benchmark, 'completed_questions', 0)
                total_questions = getattr(benchmark, 'total_questions', 0)
                error_count = getattr(benchmark, 'error_count', 0)
                avg_response_time_ms = getattr(benchmark, 'avg_response_time_ms', None)
                
                progress_percent = (completed_questions / total_questions * 100) if total_questions > 0 else 0
                
                progress_content = f"""[bold]Progress:[/bold] {completed_questions:,} / {total_questions:,} questions
[bold]Completion:[/bold] {progress_percent:.1f}%
[bold]Errors:[/bold] {error_count}"""
                
                if avg_response_time_ms:
                    estimated_remaining = (total_questions - completed_questions) * float(avg_response_time_ms) / 1000 / 60
                    progress_content += f"\n[bold]Est. Time Remaining:[/bold] {estimated_remaining:.1f} minutes"
                
                console.print(Panel(progress_content, title="⏳ Progress Status", border_style="yellow"))
                
                # Progress bar
                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    expand=True
                ) as progress_bar:
                    task = progress_bar.add_task("Processing questions...", total=total_questions)
                    progress_bar.update(task, completed=completed_questions)
                    import time
                    time.sleep(0.5)  # Brief display
            
            # Performance metrics (for completed benchmarks or those with results)
            if results or performances:
                console.print()
                
                if performances:
                    # Model performance table
                    perf_table = Table(title="🎯 Model Performance", show_header=True, header_style="bold magenta")
                    perf_table.add_column("Model", style="cyan")
                    perf_table.add_column("Accuracy", justify="right", style="green")
                    perf_table.add_column("Correct", justify="right")
                    perf_table.add_column("Total", justify="right")
                    perf_table.add_column("Avg Time (ms)", justify="right", style="blue")
                    perf_table.add_column("Cost ($)", justify="right", style="yellow")
                    
                    for perf in performances:
                        accuracy = f"{float(perf.accuracy_rate):.1%}" if perf.accuracy_rate else "N/A"
                        avg_time = f"{float(perf.avg_response_time_ms):.0f}" if perf.avg_response_time_ms else "N/A"
                        cost = f"${float(perf.total_cost_usd):.4f}" if perf.total_cost_usd else "N/A"
                        
                        perf_table.add_row(
                            str(getattr(perf, 'model_name', 'N/A')),
                            accuracy,
                            str(getattr(perf, 'correct_answers', 0)),
                            str(getattr(perf, 'total_questions', 0)),
                            avg_time,
                            cost
                        )
                    
                    console.print(perf_table)
                
                # Category breakdown
                if results:
                    category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
                    
                    for result in results:
                        if hasattr(result, 'question') and result.question and hasattr(result.question, 'category') and result.question.category:
                            category = result.question.category
                            category_stats[category]['total'] += 1
                            if getattr(result, 'is_correct', False):
                                category_stats[category]['correct'] += 1
                    
                    if category_stats:
                        console.print()
                        cat_table = Table(title="📚 Category Performance", show_header=True, header_style="bold cyan")
                        cat_table.add_column("Category", style="magenta")
                        cat_table.add_column("Correct", justify="right")
                        cat_table.add_column("Total", justify="right")
                        cat_table.add_column("Accuracy", justify="right", style="green")
                        
                        for category, stats in sorted(category_stats.items()):
                            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                            cat_table.add_row(
                                category,
                                str(stats['correct']),
                                str(stats['total']),
                                f"{accuracy:.1%}"
                            )
                        
                        console.print(cat_table)
            
            # Financial summary
            total_cost_usd = getattr(benchmark, 'total_cost_usd', None)
            total_tokens = getattr(benchmark, 'total_tokens', None)
            avg_response_time_ms = getattr(benchmark, 'avg_response_time_ms', None)
            
            if total_cost_usd or any(getattr(p, 'total_cost_usd', None) for p in performances):
                console.print()
                cost_content = f"""[bold]Total Cost:[/bold] ${float(total_cost_usd or 0):.4f}
[bold]Total Tokens:[/bold] {total_tokens or 0:,}
[bold]Average Response Time:[/bold] {float(avg_response_time_ms or 0):.0f}ms"""
                
                console.print(Panel(cost_content, title="💰 Resource Usage", border_style="green"))
            
            # Error information
            error_count = getattr(benchmark, 'error_count', None)
            if error_count and error_count > 0:
                console.print()
                error_content = f"[bold]Error Count:[/bold] {error_count}"
                
                error_details = getattr(benchmark, 'error_details', None)
                if error_details:
                    try:
                        if isinstance(error_details, str):
                            errors = json.loads(error_details)
                        else:
                            errors = error_details
                        if isinstance(errors, list) and errors:
                            error_content += f"\n[bold]Recent Error:[/bold] {errors[-1]}"
                    except:
                        error_content += f"\n[bold]Error Details:[/bold] Available"
                
                console.print(Panel(error_content, title="⚠️  Error Information", border_style="red"))
            
            # Configuration details
            config_snapshot = getattr(benchmark, 'config_snapshot', None)
            if config_snapshot:
                try:
                    if isinstance(config_snapshot, str):
                        config_data = json.loads(config_snapshot)
                    else:
                        config_data = config_snapshot
                        
                    console.print()
                    environment = getattr(benchmark, 'environment', None)
                    config_content = f"[bold]Environment:[/bold] {environment or 'production'}"
                    
                    if isinstance(config_data, dict):
                        if 'grading_mode' in config_data:
                            config_content += f"\n[bold]Grading Mode:[/bold] {config_data['grading_mode']}"
                        if 'timeout' in config_data:
                            config_content += f"\n[bold]Timeout:[/bold] {config_data['timeout']}s"
                    
                    console.print(Panel(config_content, title="⚙️  Configuration", border_style="dim"))
                except:
                    pass
            
    except Exception as e:
        console.print(f"[red]❌ Error retrieving benchmark status: {str(e)}[/red]")
        logger.exception("Benchmark status retrieval failed")
        if ctx.obj.get('config', {}).debug:
            import traceback
            console.print(f"[red]{traceback.format_exc()}[/red]")