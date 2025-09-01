"""
Benchmark History Commands

This module contains the benchmark history command implementation.
"""

import click
from rich.console import Console
from rich.table import Table

from core.database import get_db_session
from utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.command()
@click.option('--model', '-m', required=True, help='Model to show history for')
@click.option('--limit', '-l', type=int, default=10, help='Number of recent benchmarks to show')
@click.option('--detailed', '-d', is_flag=True, help='Show detailed information')
@click.pass_context
def history(ctx, model, limit, detailed):
    """Show benchmark history for a specific model.
    
    \b
    📈 EXAMPLES:
    
    alex benchmark history --model anthropic/claude-3-5-sonnet
    alex benchmark history --model openai/gpt-4 --limit 20
    alex benchmark history --model anthropic/claude-3-haiku --detailed
    alex benchmark history --model openai/gpt-3.5-turbo --limit 5 --detailed
    
    \b
    💡 TIP: Use --detailed to see more information about each benchmark run.
    """
    
    try:
        from storage.repositories import BenchmarkRepository
        
        with get_db_session() as session:
            repo = BenchmarkRepository(session)
            benchmarks = repo.get_benchmark_history(model, limit)
            
            if not benchmarks:
                console.print(f"[yellow]No benchmark history found for {model}[/yellow]")
                return
            
            # Create history table
            table = Table(title=f"Benchmark History: {model}")
            table.add_column("ID", justify="right", style="cyan")
            table.add_column("Name", style="magenta")
            table.add_column("Status", justify="center")
            table.add_column("Questions", justify="right", style="green")
            table.add_column("Created", style="dim")
            
            if detailed:
                table.add_column("Completed", style="dim")
                table.add_column("Models", style="blue")
            
            for benchmark in benchmarks:
                # Determine status color
                status_color = {
                    'completed': 'green',
                    'failed': 'red',
                    'running': 'yellow',
                    'pending': 'blue'
                }.get(benchmark.status, 'white')
                
                status_text = f"[{status_color}]{benchmark.status}[/{status_color}]"
                
                row = [
                    str(benchmark.id),
                    benchmark.name,
                    status_text,
                    str(benchmark.sample_size),
                    benchmark.created_at.strftime('%Y-%m-%d %H:%M') if benchmark.created_at else 'N/A'
                ]
                
                if detailed:
                    completed_text = benchmark.completed_at.strftime('%Y-%m-%d %H:%M') if benchmark.completed_at else 'N/A'
                    models_text = ', '.join(benchmark.models_tested_list) if benchmark.models_tested_list else 'N/A'
                    row.extend([completed_text, models_text])
                
                table.add_row(*row)
            
            console.print(table)
            console.print(f"\n[dim]Showing {len(benchmarks)} most recent benchmarks[/dim]")
            
    except Exception as e:
        console.print(f"[red]Error retrieving history: {str(e)}[/red]")
        logger.exception("History retrieval failed")