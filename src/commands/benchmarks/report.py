"""
Benchmark Report Commands

This module contains the benchmark report generation command implementation.
"""

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from core.database import get_db_session
from utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.command()
@click.option('--run-id', '-r', type=int, required=True, help='Benchmark run ID')
@click.option('--format', '-f', type=click.Choice(['terminal', 'markdown', 'json', 'html']),
              default='terminal', help='Report format')
@click.option('--output', '-o', type=click.Path(), help='Save report to file')
@click.option('--detailed', '-d', is_flag=True, help='Include detailed metrics')
@click.pass_context
def report(ctx, run_id, format, output, detailed):
    """Generate a report for a specific benchmark run.
    
    \b
    📄 EXAMPLES:
    
    alex benchmark report --run-id 1
    alex benchmark report --run-id 1 --format markdown --output report.md
    alex benchmark report --run-id 1 --format json --detailed --output results.json
    alex benchmark report --run-id 1 --detailed
    
    \b
    💡 TIP: Use 'alex benchmark list' to find run IDs.
    """
    
    try:
        from storage.repositories import BenchmarkRepository
        from benchmark.reporting import ReportGenerator, ReportFormat, ReportConfig
        
        with get_db_session() as session:
            repo = BenchmarkRepository(session)
            
            # Get benchmark summary
            stats = repo.get_benchmark_summary_stats(run_id)
            
            if not stats:
                console.print(f"[red]Benchmark {run_id} not found[/red]")
                return
            
            console.print(f"[blue]Generating {format} report for benchmark {run_id}[/blue]")
            
            # Create a mock BenchmarkResult for the report generator
            # In a complete implementation, you'd reconstruct the full result from database
            console.print(f"[yellow]Note: Full report generation requires complete result reconstruction[/yellow]")
            
            # Display basic stats
            table = Table(title=f"Benchmark {run_id} Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Name", str(stats.get('name', 'N/A')))
            table.add_row("Status", str(stats.get('status', 'N/A')))
            
            # Handle datetime formatting safely
            created_at = stats.get('created_at')
            if created_at:
                table.add_row("Created", created_at.strftime('%Y-%m-%d %H:%M:%S'))
            else:
                table.add_row("Created", 'N/A')
            
            completed_at = stats.get('completed_at')
            if completed_at:
                table.add_row("Completed", completed_at.strftime('%Y-%m-%d %H:%M:%S'))
            else:
                table.add_row("Completed", 'N/A')
            
            models_tested = stats.get('models_tested', [])
            if isinstance(models_tested, list):
                table.add_row("Models Tested", ', '.join(models_tested))
            else:
                table.add_row("Models Tested", str(models_tested))
            
            table.add_row("Total Questions", str(stats.get('total_questions', 0)))
            table.add_row("Total Responses", str(stats.get('total_responses', 0)))
            
            overall_accuracy = stats.get('overall_accuracy', 0)
            if isinstance(overall_accuracy, (int, float)):
                table.add_row("Overall Accuracy", f"{overall_accuracy:.1%}")
            else:
                table.add_row("Overall Accuracy", str(overall_accuracy))
            
            categories = stats.get('categories', [])
            table.add_row("Categories", str(len(categories)) if categories else '0')
            
            console.print(table)
            
            if detailed:
                # Show category breakdown
                if categories:
                    if isinstance(categories, list):
                        console.print(f"\n[bold]Categories:[/bold] {', '.join(categories)}")
                    else:
                        console.print(f"\n[bold]Categories:[/bold] {categories}")
                
                difficulty_levels = stats.get('difficulty_levels', [])
                if difficulty_levels:
                    if isinstance(difficulty_levels, list):
                        console.print(f"[bold]Difficulty Levels:[/bold] {', '.join(difficulty_levels)}")
                    else:
                        console.print(f"[bold]Difficulty Levels:[/bold] {difficulty_levels}")
                
                value_range = stats.get('value_range', {})
                if value_range and isinstance(value_range, dict):
                    min_val = value_range.get('min', 0)
                    max_val = value_range.get('max', 0)
                    console.print(f"[bold]Value Range:[/bold] ${min_val} - ${max_val}")
            
            if output:
                # Save the basic stats as JSON
                output_path = Path(output)
                
                # Convert stats to JSON-serializable format
                json_stats = {}
                for key, value in stats.items():
                    if hasattr(value, 'isoformat'):  # datetime objects
                        json_stats[key] = value.isoformat()
                    else:
                        json_stats[key] = value
                
                with open(output_path, 'w') as f:
                    json.dump(json_stats, f, indent=2, default=str)
                console.print(f"\n[green]✓ Report saved to {output}[/green]")
                
    except Exception as e:
        console.print(f"[red]Error generating report: {str(e)}[/red]")
        logger.exception("Report generation failed")