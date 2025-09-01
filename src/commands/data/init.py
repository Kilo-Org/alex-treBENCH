"""
Data Initialization Commands

This module contains the data initialization command implementation.
"""

import sys

import click
from rich.console import Console

from utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.command()
@click.option('--force', is_flag=True, help='Force re-download of dataset')
@click.option('--strict', is_flag=True, help='Use strict validation rules')
@click.pass_context
def init(ctx, force, strict):
    """Initialize the Jeopardy dataset.
    
    \b
    📊 EXAMPLES:
    
    alex data init
    alex data init --force
    alex data init --strict
    alex data init --force --strict
    
    \b
    💡 Downloads and processes the Jeopardy dataset from Kaggle.
    Use --force to re-download even if dataset already exists.
    Use --strict for enhanced validation rules.
    """
    try:
        from scripts.init_data import main as init_main
        
        console.print("[blue]Starting dataset initialization...[/blue]")
        result = init_main(force_download=force, strict_validation=strict)
        
        if result:
            console.print(f"[green]✓ Dataset initialization completed successfully![/green]")
            console.print(f"[dim]Benchmark ID: {result['benchmark_id']}[/dim]")
            console.print(f"[dim]Questions loaded: {result['questions_saved']:,}[/dim]")
        else:
            console.print("[red]✗ Dataset initialization failed[/red]")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[red]Error during dataset initialization: {str(e)}[/red]")
        logger.exception("Dataset initialization failed")
        sys.exit(1)