"""
CLI Entry Point

Main command-line interface for the alex-treBENCH Jeopardy Benchmarking System
using Click framework with rich output formatting.
"""

import warnings
import sys
import os
from pathlib import Path
from typing import Optional

# Suppress GIL-related warnings from pandas and other libraries  
# These warnings occur in Python 3.13+ with free-threading when libraries
# haven't declared GIL-safety yet. This is harmless for our use case.
warnings.filterwarnings("ignore", 
                       message=".*has been enabled to load module.*which has not declared that it can run safely without the GIL.*",
                       category=RuntimeWarning)

warnings.filterwarnings("ignore", 
                       message=".*pandas.*GIL.*",
                       category=RuntimeWarning)

# Set environment variable programmatically if not already set
if 'PYTHONWARNINGS' not in os.environ:
    os.environ['PYTHONWARNINGS'] = 'ignore::RuntimeWarning:importlib._bootstrap'

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Create src namespace mapping for installed package
import types
if 'src' not in sys.modules:
    src_module = types.ModuleType('src')
    sys.modules['src'] = src_module
    src_module.__path__ = [str(Path(__file__).parent)]

# Import storage to populate src namespace
import storage

from core.config import get_config, reload_config
from core.database import init_database, check_database_connection
from core.exceptions import AlexTreBenchException
from utils.logging import setup_logging, get_logger

# Import all command modules
from commands.health import health
from commands.models import models
from commands.benchmarks import run, compare, history, report, status, leaderboard
from commands.data import init as data_init, stats, sample, validate
from commands.config import show as config_show, validate as config_validate, export as config_export
from commands.session import session

console = Console()
logger = get_logger(__name__)


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.pass_context
def cli(ctx, config, verbose, debug):
    """alex-treBENCH - Jeopardy Benchmarking System for Language Models
    
    \b
    🚀 QUICK START EXAMPLES:
    
    alex benchmark run --model anthropic/claude-3-5-sonnet --size quick
    alex benchmark compare --models "openai/gpt-4,anthropic/claude-3-5-sonnet"
    alex models list
    alex benchmark report --run-id 1 --format markdown
    alex data init
    alex health
    
    \b
    💡 TIP: Use 'alex COMMAND --help' for detailed options on any command.
    
    📚 For complete documentation, see: docs/USER_GUIDE.md
    """
    
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    try:
        # Load configuration
        if config:
            app_config = reload_config(Path(config))
        else:
            app_config = get_config()
        
        # Override debug mode if specified
        if debug:
            app_config.debug = debug
        
        # Setup logging
        log_level = 'DEBUG' if (verbose or debug) else app_config.logging.level
        app_config.logging.level = log_level
        setup_logging(app_config)
        
        # Store config in context
        ctx.obj['config'] = app_config
        
        # Display banner if no subcommand
        if not ctx.invoked_subcommand:
            _display_banner()
            
    except Exception as e:
        console.print(f"[red]Error initializing application: {str(e)}[/red]")
        sys.exit(1)


def _display_banner():
    """Display application banner."""
    banner = Panel.fit(
        "[bold blue]alex-treBENCH[/bold blue]\n"
        "[dim]Jeopardy Benchmarking System for Language Models[/dim]\n\n"
        "Use --help for available commands",
        title="🧠 LLM Benchmarking",
        border_style="blue"
    )
    console.print(banner)


# ===== BENCHMARK COMMANDS =====

@cli.group()
def benchmark():
    """Benchmark management commands."""
    pass


# Register benchmark subcommands
benchmark.add_command(run)
benchmark.add_command(compare)
benchmark.add_command(history)
benchmark.add_command(report)
benchmark.add_command(status)
benchmark.add_command(leaderboard)


@benchmark.command('list')
@click.option('--limit', '-l', type=int, default=20, help='Number of benchmarks to show')
@click.option('--status', type=click.Choice(['pending', 'running', 'completed', 'failed']), help='Filter by status')
@click.option('--model', '-m', help='Filter by model name')
@click.pass_context
def benchmark_list(ctx, limit, status, model):
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
        from core.database import get_db_session
        from storage.repositories import BenchmarkRepository, ModelPerformanceRepository
        from storage.models import BenchmarkRun
        from rich.table import Table
        import json
        
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


# ===== DATA COMMANDS =====

@cli.group()
def data():
    """Data management commands."""
    pass


# Register data subcommands
data.add_command(data_init)
data.add_command(stats)
data.add_command(sample)
data.add_command(validate)


# ===== CONFIG COMMANDS =====

@cli.group()
def config():
    """Configuration management commands."""
    pass


# Register config subcommands  
config.add_command(config_show)
config.add_command(config_validate)
config.add_command(config_export)


# ===== TOP-LEVEL COMMANDS =====

# Register top-level commands
cli.add_command(health)
cli.add_command(models)
cli.add_command(session)


@cli.command()
@click.option('--force', is_flag=True, help='Force database recreation')
@click.pass_context
def init(ctx, force):
    """Initialize the database and create tables.
    
    \b
    🏗️ EXAMPLES:
    
    alex init
    alex init --force
    
    \b
    💡 Sets up the database schema required for benchmarking.
    Use --force to recreate existing tables.
    """
    
    try:
        config = ctx.obj['config']
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            if force:
                progress.add_task("Recreating database...", total=None)
                console.print("[yellow]Database reset not yet implemented[/yellow]")
            else:
                progress.add_task("Initializing database...", total=None)
            
            # Initialize database
            init_database()
            
            # Check connection
            if check_database_connection():
                console.print("[green]✓ Database initialized successfully[/green]")
            else:
                console.print("[red]✗ Database connection failed[/red]")
                sys.exit(1)
                
    except Exception as e:
        console.print(f"[red]Database initialization failed: {str(e)}[/red]")
        logger.exception("Database initialization failed")
        sys.exit(1)


@cli.command()
@click.option('--benchmark-id', '-b', type=int, help='Export specific benchmark results')
@click.option('--format', type=click.Choice(['json', 'csv', 'html']), default='json', help='Export format')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.pass_context
def export(ctx, benchmark_id, format, output):
    """Export benchmark results.
    
    \b
    📤 EXAMPLES:
    
    alex export --benchmark-id 1
    alex export --benchmark-id 1 --format csv --output results.csv
    alex export --benchmark-id 1 --format html --output report.html
    
    \b
    💡 Exports benchmark data in various formats for analysis or reporting.
    """
    
    console.print(f"[yellow]⚠️ Export functionality not yet implemented[/yellow]")
    console.print(f"[dim]Would export benchmark {benchmark_id} to {format} format[/dim]")
    
    if output:
        console.print(f"[dim]Would save to: {output}[/dim]")
    else:
        console.print(f"[dim]Would save to: benchmark_{benchmark_id}.{format}[/dim]")

def main():
    """Main entry point with comprehensive error handling."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except AlexTreBenchException as e:
        logger.error(f"Application error: {str(e)}")
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error")
        console.print(f"[red]Unexpected error: {str(e)}[/red]")
        if '--debug' in sys.argv:
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        sys.exit(1)


if __name__ == "__main__":
    main()

