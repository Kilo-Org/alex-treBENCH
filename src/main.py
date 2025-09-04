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

from core.config import get_config, reload_config
from core.exceptions import AlexTreBenchException
from utils.logging import setup_logging, get_logger
from utils.help_text import show_help_with_markdown

# Import all command modules
from commands.health import health
from commands.models import models
from commands.benchmarks import run, compare, history, report, status, leaderboard, list_benchmarks, export as benchmark_export
from commands.data import init as data_init, stats, sample, validate
from commands.config import show as config_show, validate as config_validate, export as config_export
from commands.session import session
from commands.database import init as database_init

console = Console()
logger = get_logger(__name__)


@click.group(invoke_without_command=True)
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.option('--help', '-h', is_flag=True, expose_value=False, is_eager=True,
              callback=show_help_with_markdown, help='Show this message and exit')
@click.pass_context
def cli(ctx, config, verbose, debug):
    """alex-treBENCH - Jeopardy Benchmarking System for Language Models"""
    
    # If no subcommand is provided, show the Rich markdown help
    if ctx.invoked_subcommand is None:
        show_help_with_markdown(ctx, None, True)
    
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
benchmark.add_command(list_benchmarks, name='list')
benchmark.add_command(benchmark_export, name='export')


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


# ===== DATABASE COMMANDS =====

@cli.group()
def database():
    """Database management commands."""
    pass

# Register database subcommands
database.add_command(database_init)


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
