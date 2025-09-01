"""
Benchmark List Commands

This module contains the benchmark list command implementation.
"""

import click
import json
from rich.console import Console
from rich.table import Table

from core.database import get_db_session
from utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.command()
@click.option('--limit', '-l', type=int, default=20, help='Number of benchmarks to show')
@click.option('--status', type=click.Choice(['pending', 'running', 'completed', 'failed']), help='Filter by status')
@click.option('--model', '-m', help='Filter by model name')
@click.pass_context
def list_benchmarks(ctx, limit, status, model):
    """List recent benchmarks.
    
    \b
    📋 EXAMPLES:
    
    alex benchmark list
    alex benchmark list --limit 10
    alex benchmark list --status completed
    alex benchmark list --model anthropic/claude-3-5-sonnet
    
    \b
    💡 Shows recent benchmark runs with their status and key metrics.
    """
    try:
        from src.storage.repositories import BenchmarkRepository
        from src.storage.models import BenchmarkRun
        
        with get_db_session() as session:
            benchmark_repo = BenchmarkRepository(session)
            
            # Get benchmarks with filters
            benchmarks = benchmark_repo.list_benchmarks(limit=limit)
            
            # Apply filters
            if status:
                benchmarks = [b for b in benchmarks if getattr(b, 'status', '') == status]
            
            if model:
                filtered_benchmarks = []
                for b in benchmarks:
                    models_tested = getattr(b, 'models_tested', '')
                    if isinstance(models_tested, str):
                        try:
                            models_list = json.loads(models_tested)
                            if isinstance(models_list, list) and model in models_list:
                                filtered_benchmarks.append(b)
                        except json.JSONDecodeError:
                            if model in models_tested:
                                filtered_benchmarks.append(b)
                    elif isinstance(models_tested, list) and model in models_tested:
                        filtered_benchmarks.append(b)
                benchmarks = filtered_benchmarks
            
            if not benchmarks:
                console.print("[yellow]No benchmarks found matching criteria[/yellow]")
                return
            
            # Create table
            table = Table(title="Recent Benchmarks")
            table.add_column("ID", justify="right", style="cyan")
            table.add_column("Name", style="magenta")
            table.add_column("Status", justify="center")
            table.add_column("Models", style="blue")
            table.add_column("Questions", justify="right", style="green")
            table.add_column("Created", style="dim")
            
            for benchmark in benchmarks:
                # Status coloring
                status_color = {
                    'completed': 'green',
                    'failed': 'red',
                    'running': 'yellow',
                    'pending': 'blue'
                }.get(getattr(benchmark, 'status', ''), 'white')
                
                status_text = f"[{status_color}]{getattr(benchmark, 'status', 'unknown')}[/{status_color}]"
                
                # Parse models tested
                models_tested = getattr(benchmark, 'models_tested', '')
                if isinstance(models_tested, str):
                    try:
                        models_list = json.loads(models_tested)
                        models_text = ', '.join(models_list[:2])  # Show first 2 models
                        if len(models_list) > 2:
                            models_text += f' (+{len(models_list)-2} more)'
                    except json.JSONDecodeError:
                        models_text = models_tested[:30] + "..." if len(models_tested) > 30 else models_tested
                else:
                    models_text = str(models_tested)[:30]
                
                created_at = getattr(benchmark, 'created_at', None)
                created_text = created_at.strftime('%Y-%m-%d %H:%M') if created_at else 'N/A'
                
                table.add_row(
                    str(getattr(benchmark, 'id', 'N/A')),
                    getattr(benchmark, 'name', 'N/A'),
                    status_text,
                    models_text,
                    str(getattr(benchmark, 'sample_size', 0)),
                    created_text
                )
            
            console.print(table)
            console.print(f"\n[dim]Showing {len(benchmarks)} benchmarks[/dim]")
            
    except Exception as e:
        console.print(f"[red]Error listing benchmarks: {str(e)}[/red]")
        logger.exception("Benchmark listing failed")
