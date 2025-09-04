"""
Benchmark Compare Commands

This module contains the benchmark comparison command implementation.
"""

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, TextColumn

from utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.command()
@click.option('--models', '-m', required=True, help='Comma-separated list of models to compare')
@click.option('--size', type=click.Choice(['small', 'medium', 'large', 'quick', 'standard', 'comprehensive']),
              default='standard', help='Benchmark size for all models')
@click.option('--concurrent-limit', type=int, default=2, help='Maximum concurrent model benchmarks')
@click.option('--report-format', type=click.Choice(['terminal', 'markdown', 'json']),
              default='terminal', help='Report output format')
@click.option('--output', '-o', type=click.Path(), help='Save report to file')
@click.pass_context
def compare(ctx, models, size, concurrent_limit, report_format, output):
    """Compare multiple models using benchmarks.
    
    \b
    📊 EXAMPLES:
    
    alex benchmark compare --models "openai/gpt-4,anthropic/claude-3-5-sonnet"
    alex benchmark compare --models "openai/gpt-3.5-turbo,anthropic/claude-3-haiku" --size quick
    alex benchmark compare --models "openai/gpt-4,anthropic/claude-3-5-sonnet,google/gemini-pro" --size comprehensive --output comparison.md
    alex benchmark compare --models "openai/gpt-3.5-turbo,anthropic/claude-3-haiku" --concurrent-limit 4
    
    \b
    💡 TIP: Use quotes around the model list to avoid shell parsing issues.
    """
    
    async def run_comparison_async():
        try:
            from benchmark.scheduler import BenchmarkScheduler
            from benchmark.runner import RunMode
            from benchmark.reporting import ReportGenerator, ReportFormat
            
            # Parse model list
            model_list = [m.strip() for m in models.split(',')]
            
            if len(model_list) < 2:
                console.print("[red]Need at least 2 models to compare[/red]")
                return
            
            # Map size to run mode
            size_map = {
                'small': RunMode.QUICK, 'quick': RunMode.QUICK,
                'medium': RunMode.STANDARD, 'standard': RunMode.STANDARD,
                'large': RunMode.COMPREHENSIVE, 'comprehensive': RunMode.COMPREHENSIVE
            }
            run_mode = size_map.get(size, RunMode.STANDARD)
            
            console.print(f"[blue]Comparing {len(model_list)} models: {', '.join(model_list)}[/blue]")
            console.print(f"[dim]Mode: {run_mode.value}, Concurrent limit: {concurrent_limit}[/dim]\n")
            
            # Set up scheduler
            scheduler = BenchmarkScheduler()
            
            # Schedule benchmarks for all models
            scheduled_ids = scheduler.schedule_multiple(
                models=model_list,
                mode=run_mode,
                concurrent_limit=concurrent_limit,
                benchmark_name_prefix=f"compare_{run_mode.value}"
            )
            
            console.print(f"[green]Scheduled {len(scheduled_ids)} benchmarks[/green]")
            
            # Show progress
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Running benchmarks...", total=None)
                
                # Run all benchmarks
                results = await scheduler.run_scheduled_benchmarks()
                
                progress.update(task, description="Generating comparison report...")
                
                # Convert to list of BenchmarkResult objects
                result_list = list(results.values())
                
                if not result_list:
                    console.print("[red]No benchmarks completed successfully[/red]")
                    return
                
                # Generate comparison report
                report_gen = ReportGenerator()
                format_enum = {
                    'terminal': ReportFormat.TERMINAL,
                    'markdown': ReportFormat.MARKDOWN,
                    'json': ReportFormat.JSON
                }[report_format]
                
                output_path = Path(output) if output else None
                report_content = report_gen.generate_comparison_report(
                    result_list, format_enum, output_path
                )
                
                console.print(report_content)
                
                progress.update(task, description="Complete!")
                progress.stop()
            
            successful_count = len([r for r in result_list if r.is_successful])
            console.print(f"\n[green]✓ Comparison complete: {successful_count}/{len(model_list)} models succeeded[/green]")
            
            if output:
                console.print(f"[dim]Report saved to: {output}[/dim]")
                
        except Exception as e:
            console.print(f"[red]Error running comparison: {str(e)}[/red]")
            logger.exception("Comparison failed")
            sys.exit(1)
    
    asyncio.run(run_comparison_async())