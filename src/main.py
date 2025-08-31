"""
CLI Entry Point

Main command-line interface for the Jeopardy Benchmarking System
using Click framework with rich output formatting.
"""

import warnings
import sys
import os

# Suppress GIL-related warnings from pandas and other libraries
# These warnings occur in Python 3.13+ with free-threading when libraries
# haven't declared GIL-safety yet. This is harmless for our use case.
warnings.filterwarnings("ignore", 
                       message=".*has been enabled to load module.*which has not declared that it can run safely without the GIL.*",
                       category=RuntimeWarning)

# Also suppress any pandas-specific warnings that might be related
warnings.filterwarnings("ignore", 
                       message=".*pandas.*GIL.*",
                       category=RuntimeWarning)

# Alternative approach: Set environment variable programmatically if not already set
if 'PYTHONWARNINGS' not in os.environ:
    os.environ['PYTHONWARNINGS'] = 'ignore::RuntimeWarning:importlib._bootstrap'

import asyncio
from pathlib import Path
from typing import Optional, List
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel


# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Create src namespace mapping for installed package
import sys
import types
if 'src' not in sys.modules:
    src_module = types.ModuleType('src')
    sys.modules['src'] = src_module
    # Add current directory as src for submodule imports
    src_module.__path__ = [str(Path(__file__).parent)]


# Import storage to populate src namespace
import storage

from core.config import get_config, reload_config
from core.database import init_database, check_database_connection, get_db_session
from core.exceptions import AlexTreBenchException
from utils.logging import setup_logging, get_logger

console = Console()
logger = get_logger(__name__)

@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.pass_context
def cli(ctx, config, verbose, debug):
    """Jeopardy Benchmarking System - Benchmark language models using Jeopardy questions.
    
    \b
    🚀 QUICK START EXAMPLES:
    
    alex benchmark run --model anthropic/claude-3-5-sonnet --size quick
    alex benchmark compare --models "openai/gpt-4,anthropic/claude-3-5-sonnet"
    alex models list
    alex benchmark report --run-id 1 --format markdown
    alex benchmark history --model anthropic/claude-3-5-sonnet
    
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
            app_config.debug = True
        
        # Setup logging
        log_level = 'DEBUG' if (verbose or debug) else app_config.logging.level
        app_config.logging.level = log_level
        setup_logging(app_config)
        
        # Store config in context
        ctx.obj['config'] = app_config
        
        # Display banner
        if not ctx.invoked_subcommand:
            _display_banner()
            
    except Exception as e:
        console.print(f"[red]Error initializing application: {str(e)}[/red]")
        sys.exit(1)


def _display_banner():
    """Display application banner."""
    banner = Panel.fit(
        "[bold blue]Jeopardy Benchmarking System[/bold blue]\n"
        "[dim]Benchmark language models using Jeopardy questions[/dim]\n\n"
        "Use --help for available commands",
        title="🧠 LLM Benchmarking",
        border_style="blue"
    )
    console.print(banner)


@cli.group()
def benchmark():
    """Benchmark management commands."""
    pass


@benchmark.command('run')
@click.option('--model', '-m', default=None, help='Model to benchmark (e.g., openai/gpt-4). Uses anthropic/claude-3.5-sonnet if not specified')
@click.option('--size', type=click.Choice(['small', 'medium', 'large', 'quick', 'standard', 'comprehensive']),
              default='standard', help='Benchmark size')
@click.option('--name', '-n', help='Custom benchmark name')
@click.option('--description', '-d', help='Benchmark description')
@click.option('--timeout', type=int, help='Timeout in seconds per question')
@click.option('--grading-mode', type=click.Choice(['strict', 'lenient', 'jeopardy', 'adaptive']),
              default='lenient', help='Grading mode')
@click.option('--save-results/--no-save-results', default=True, help='Save results to database')
@click.option('--report-format', type=click.Choice(['terminal', 'markdown', 'json']),
              default='terminal', help='Report output format')
@click.option('--show-jeopardy-scores/--no-jeopardy-scores', default=True, help='Show Jeopardy scores in reports')
@click.option('--list-models', is_flag=True, help='Show available models and exit')
@click.pass_context
def benchmark_run(ctx, model, size, name, description, timeout, grading_mode, save_results, report_format, show_jeopardy_scores, list_models):
    """Run a benchmark for a specific model.
    
    \b
    📋 EXAMPLES:
    
    alex benchmark run --size quick
    alex benchmark run --model openai/gpt-4 --size standard
    alex benchmark run --model anthropic/claude-3-5-sonnet --size comprehensive --name "Claude Production Test"
    alex benchmark run --model anthropic/claude-3-haiku --size quick --grading-mode lenient
    alex benchmark run --model openai/gpt-4 --size standard --show-jeopardy-scores
    alex benchmark run --model anthropic/claude-3-5-sonnet --size quick --no-jeopardy-scores
    alex benchmark run --list-models
    
    \b
    💡 SIZES: quick=10 questions, standard=50 questions, comprehensive=200 questions
    💰 JEOPARDY SCORES: Shows winnings/losses based on question values (enabled by default)
    """
    
    async def run_benchmark_async():
        try:
            from benchmark.runner import BenchmarkRunner, RunMode, BenchmarkConfig
            from benchmark.reporting import ReportGenerator, ReportFormat
            from evaluation.grader import GradingMode
            from models.model_registry import model_registry, get_default_model
            
            # Handle --list-models option
            if list_models:
                console.print("[blue]Loading available models...[/blue]")
                models = await model_registry.get_available_models()
                
                # Create table of available models
                table = Table(title="Available Models for Benchmarking")
                table.add_column("Provider", style="cyan")
                table.add_column("Model ID", style="magenta") 
                table.add_column("Display Name", style="blue")
                table.add_column("Context", justify="right", style="green")
                table.add_column("Cost", style="yellow")
                table.add_column("Recommended", justify="center", style="dim")
                
                # Sort models by provider, then name
                sorted_models = sorted(models, key=lambda x: (x.get('provider', ''), x.get('name', '')))
                
                for m in sorted_models:
                    # Skip models that aren't available
                    if not m.get('available', True):
                        continue
                        
                    pricing = m.get('pricing', {})
                    input_cost = pricing.get('input_cost_per_1m_tokens', 0)
                    output_cost = pricing.get('output_cost_per_1m_tokens', 0)
                    
                    # Mark recommended models
                    recommended = ""
                    if m.get('id') == get_default_model():
                        recommended = "⭐ Default"
                    elif 'claude-3' in m.get('id', '').lower() or 'gpt-4' in m.get('id', '').lower():
                        recommended = "✓ Popular"
                    
                    table.add_row(
                        (m.get('provider', 'Unknown')).title(),
                        m.get('id', 'N/A'),
                        m.get('name', 'N/A'),
                        f"{m.get('context_length', 0):,}",
                        f"${input_cost:.2f}/${output_cost:.2f}",
                        recommended
                    )
                
                console.print(table)
                console.print(f"\n[dim]Total available models: {len([m for m in models if m.get('available', True)])}[/dim]")
                console.print(f"[dim]Use --model MODEL_ID to specify a model for benchmarking[/dim]")
                console.print(f"[dim]Default model: {get_default_model()}[/dim]");
                return
            
            # Use default model if none specified
            actual_model = model
            if not actual_model:
                actual_model = get_default_model()
                console.print(f"[blue]No model specified, using default: {actual_model}[/blue]")
            
            # Validate model using dynamic system
            console.print("[dim]Validating model availability...[/dim]")
            models = await model_registry.get_available_models()
            model_info = None
            
            for m in models:
                if m.get('id', '').lower() == actual_model.lower():
                    model_info = m
                    break
            
            if not model_info:
                console.print(f"[red]Model not found: {actual_model}[/red]")
                console.print("[dim]Use 'benchmark run --list-models' to see available models[/dim]")
                
                # Show similar models as suggestions
                similar_models = model_registry.search_models(actual_model.split('/')[-1], models)[:5]
                if similar_models:
                    console.print(f"\n[yellow]Similar models you might want to try:[/yellow]")
                    for sim in similar_models:
                        if sim.get('available', True):
                            console.print(f"  • [cyan]{sim.get('id', 'N/A')}[/cyan] - {sim.get('name', 'N/A')}")
                else:
                    console.print(f"\n[yellow]💡 Popular models to try:[/yellow]")
                    console.print(f"  • [cyan]anthropic/claude-3.5-sonnet[/cyan] (Default)")
                    console.print(f"  • [cyan]openai/gpt-4-turbo[/cyan]")
                    console.print(f"  • [cyan]anthropic/claude-3-haiku[/cyan] (Fast & cheap)")
                return
            
            # Check if model is available
            if not model_info.get('available', True):
                console.print(f"[red]Model is currently unavailable: {actual_model}[/red]")
                console.print("[dim]Try a different model or check OpenRouter status[/dim]")
                return
            
            # Map size to run mode
            size_map = {
                'small': RunMode.QUICK, 'quick': RunMode.QUICK,
                'medium': RunMode.STANDARD, 'standard': RunMode.STANDARD,
                'large': RunMode.COMPREHENSIVE, 'comprehensive': RunMode.COMPREHENSIVE
            }
            run_mode = size_map.get(size, RunMode.STANDARD)
            
            # Map grading mode
            grading_mode_map = {
                'strict': GradingMode.STRICT,
                'lenient': GradingMode.LENIENT,
                'jeopardy': GradingMode.JEOPARDY,
                'adaptive': GradingMode.ADAPTIVE
            }
            grading_mode_enum = grading_mode_map.get(grading_mode, GradingMode.LENIENT)
            
            runner = BenchmarkRunner()
            config = runner.get_default_config(run_mode)
            
            # Apply custom settings
            if timeout:
                config.timeout_seconds = timeout
            config.grading_mode = grading_mode_enum
            config.save_results = save_results
            
            # Display benchmark info with model details
            console.print(f"[blue]Starting {run_mode.value} benchmark[/blue]")
            console.print(f"[dim]Model: {model_info.get('name', actual_model)} ({model_info.get('provider', 'Unknown')})[/dim]")
            console.print(f"[dim]Sample size: {config.sample_size}, Grading: {grading_mode}[/dim]")
            
            # Show cost estimate if available
            pricing = model_info.get('pricing', {})
            if pricing.get('input_cost_per_1m_tokens', 0) > 0:
                estimated_cost = ((100 * config.sample_size) / 1_000_000) * (pricing.get('input_cost_per_1m_tokens', 0) + pricing.get('output_cost_per_1m_tokens', 0))
                console.print(f"[dim]Estimated cost: ~${estimated_cost:.4f}[/dim]")
            
            console.print()
            
            # Show progress
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Running benchmark...", total=None)
                
                # Run the benchmark
                result = await runner.run_benchmark(
                    model_name=actual_model,
                    mode=run_mode,
                    custom_config=config,
                    benchmark_name=name
                )
                
                progress.update(task, description="Generating report...")
                
                # Generate and display report
                from benchmark.reporting import ReportConfig
                report_config = ReportConfig(show_jeopardy_scores=show_jeopardy_scores)
                report_gen = ReportGenerator(config=report_config)
                
                if report_format == 'terminal':
                    report_gen.display_terminal_report(result)
                else:
                    format_enum = ReportFormat.MARKDOWN if report_format == 'markdown' else ReportFormat.JSON
                    report_content = report_gen.generate_report(result, format_enum)
                    console.print(report_content)
                
                progress.update(task, description="Complete!")
                progress.stop()
            
            if result.success:
                console.print(f"\n[green]✓ Benchmark completed successfully in {result.execution_time:.2f}s[/green]")
                console.print(f"[dim]Benchmark ID: {result.benchmark_id}[/dim]")
                
                if result.metrics:
                    console.print(f"[dim]Overall Score: {result.metrics.overall_score:.3f}[/dim]")
                    console.print(f"[dim]Accuracy: {result.metrics.accuracy.overall_accuracy:.1%}[/dim]")
                    if hasattr(result.metrics, 'jeopardy_score') and show_jeopardy_scores:
                        console.print(f"[dim]Jeopardy Score: ${result.metrics.jeopardy_score.total_jeopardy_score:,}[/dim]")
                    if hasattr(result.metrics, 'cost'):
                        console.print(f"[dim]Total Cost: ${result.metrics.cost.total_cost:.4f}[/dim]")
            else:
                console.print(f"\n[red]✗ Benchmark failed: {result.error_message}[/red]")
                console.print(f"[dim]Execution time: {result.execution_time:.2f}s[/dim]")
                sys.exit(1)
                
        except Exception as e:
            console.print(f"[red]Error running benchmark: {str(e)}[/red]")
            logger.exception("Benchmark execution failed")
            sys.exit(1)
    
    asyncio.run(run_benchmark_async())


@benchmark.command('compare')
@click.option('--models', '-m', required=True, help='Comma-separated list of models to compare')
@click.option('--size', type=click.Choice(['small', 'medium', 'large', 'quick', 'standard', 'comprehensive']),
              default='standard', help='Benchmark size for all models')
@click.option('--concurrent-limit', type=int, default=2, help='Maximum concurrent model benchmarks')
@click.option('--report-format', type=click.Choice(['terminal', 'markdown', 'json']),
              default='terminal', help='Report output format')
@click.option('--output', '-o', type=click.Path(), help='Save report to file')
@click.pass_context
def benchmark_compare(ctx, models, size, concurrent_limit, report_format, output):
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
                
                if report_format == 'terminal':
                    report_gen.display_terminal_report(result_list)
                else:
                    console.print(report_content)
                
                progress.update(task, description="Complete!")
                progress.stop()
            
            successful_count = len([r for r in result_list if r.success])
            console.print(f"\n[green]✓ Comparison complete: {successful_count}/{len(model_list)} models succeeded[/green]")
            
            if output:
                console.print(f"[dim]Report saved to: {output}[/dim]")
                
        except Exception as e:
            console.print(f"[red]Error running comparison: {str(e)}[/red]")
            logger.exception("Comparison failed")
            sys.exit(1)
    
    asyncio.run(run_comparison_async())


@benchmark.command('history')
@click.option('--model', '-m', required=True, help='Model to show history for')
@click.option('--limit', '-l', type=int, default=10, help='Number of recent benchmarks to show')
@click.option('--detailed', '-d', is_flag=True, help='Show detailed information')
@click.pass_context
def benchmark_history(ctx, model, limit, detailed):
    """Show benchmark history for a specific model."""
    
    try:
        from core.database import get_db_session
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


@benchmark.command('report')
@click.option('--run-id', '-r', type=int, required=True, help='Benchmark run ID')
@click.option('--format', '-f', type=click.Choice(['terminal', 'markdown', 'json', 'html']),
              default='terminal', help='Report format')
@click.option('--output', '-o', type=click.Path(), help='Save report to file')
@click.option('--detailed', '-d', is_flag=True, help='Include detailed metrics')
@click.pass_context
def benchmark_report(ctx, run_id, format, output, detailed):
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
        from core.database import get_db_session
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
            
            table.add_row("Name", stats['name'])
            table.add_row("Status", stats['status'])
            table.add_row("Created", stats['created_at'].strftime('%Y-%m-%d %H:%M:%S') if stats['created_at'] else 'N/A')
            table.add_row("Completed", stats['completed_at'].strftime('%Y-%m-%d %H:%M:%S') if stats['completed_at'] else 'N/A')
            table.add_row("Models Tested", ', '.join(stats['models_tested']))
            table.add_row("Total Questions", str(stats['total_questions']))
            table.add_row("Total Responses", str(stats['total_responses']))
            table.add_row("Overall Accuracy", f"{stats['overall_accuracy']:.1%}")
            table.add_row("Categories", str(len(stats['categories'])))
            
            console.print(table)
            
            if detailed:
                # Show category breakdown
                if stats['categories']:
                    console.print(f"\n[bold]Categories:[/bold] {', '.join(stats['categories'])}")
                
                if stats['difficulty_levels']:
                    console.print(f"[bold]Difficulty Levels:[/bold] {', '.join(stats['difficulty_levels'])}")
                
                console.print(f"[bold]Value Range:[/bold] ${stats['value_range']['min']} - ${stats['value_range']['max']}")
            
            if output:
                # For now, just save the basic stats as JSON
                import json
                with open(output, 'w') as f:
                    json.dump(stats, f, indent=2, default=str)
                console.print(f"\n[green]✓ Report saved to {output}[/green]")
                
    except Exception as e:
        console.print(f"[red]Error generating report: {str(e)}[/red]")
        logger.exception("Report generation failed")


@benchmark.command('list')
@click.option('--limit', '-l', type=int, default=20, help='Number of benchmarks to show')
@click.option('--status', type=click.Choice(['pending', 'running', 'completed', 'failed']), help='Filter by status')
@click.option('--model', '-m', help='Filter by model name')
@click.pass_context
def benchmark_list(ctx, limit, status, model):
    """List recent benchmarks."""
    
    try:
        from core.database import get_db_session
        from storage.repositories import BenchmarkRepository
        
        with get_db_session() as session:
            repo = BenchmarkRepository(session)
            
            # Get benchmarks (basic implementation - could be enhanced with filters)
            benchmarks = repo.list_benchmarks(limit=limit)
            
            # Apply filters
            if status:
                benchmarks = [b for b in benchmarks if b.status == status]
            
            if model:
                benchmarks = [b for b in benchmarks if model in b.models_tested_list]
            
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
                }.get(benchmark.status, 'white')
                
                status_text = f"[{status_color}]{benchmark.status}[/{status_color}]"
                models_text = ', '.join(benchmark.models_tested_list[:2])  # Show first 2 models
                if len(benchmark.models_tested_list) > 2:
                    models_text += f' (+{len(benchmark.models_tested_list)-2} more)'
                
                table.add_row(
                    str(benchmark.id),
                    benchmark.name,
                    status_text,
                    models_text,
                    str(benchmark.sample_size),
                    benchmark.created_at.strftime('%Y-%m-%d %H:%M') if benchmark.created_at else 'N/A'
                )
            
            console.print(table)
            console.print(f"\n[dim]Showing {len(benchmarks)} benchmarks[/dim]")
            
    except Exception as e:
        console.print(f"[red]Error listing benchmarks: {str(e)}[/red]")
        logger.exception("Benchmark listing failed")


@benchmark.command('leaderboard')
@click.option('--limit', '-l', type=int, default=10, help='Number of models to show in leaderboard')
@click.option('--report-format', type=click.Choice(['terminal', 'markdown', 'json']),
              default='terminal', help='Report output format')
@click.option('--output', '-o', type=click.Path(), help='Save leaderboard to file')
@click.pass_context
def benchmark_leaderboard(ctx, limit, report_format, output):
    """Show Jeopardy leaderboard of model performances.
    
    \b
    🏆 EXAMPLES:
    
    alex benchmark leaderboard
    alex benchmark leaderboard --limit 20
    alex benchmark leaderboard --report-format markdown --output leaderboard.md
    alex benchmark leaderboard --report-format json --output leaderboard.json
    
    \b
    📊 Shows models ranked by their total Jeopardy winnings/losses with detailed breakdown.
    """
    
    async def show_leaderboard_async():
        try:
            from storage.repositories import BenchmarkRepository, PerformanceRepository
            from benchmark.reporting import ReportGenerator, ReportConfig, ReportFormat
            
            console.print("[blue]Loading benchmark results for leaderboard...[/blue]")
            
            with get_db_session() as session:
                # Get recent benchmark results with Jeopardy scores
                benchmark_repo = BenchmarkRepository(session)
                perf_repo = PerformanceRepository(session)
                
                # Get all completed benchmarks with performance data
                recent_benchmarks = benchmark_repo.list_benchmarks(limit=100)
                completed_benchmarks = [b for b in recent_benchmarks if b.status == 'completed']
                
                if not completed_benchmarks:
                    console.print("[yellow]No completed benchmarks found. Run some benchmarks first![/yellow]")
                    return
                
                # Get performance records with Jeopardy scores
                leaderboard_data = []
                for benchmark in completed_benchmarks:
                    performances = perf_repo.get_performances_by_benchmark(benchmark.id)
                    for perf in performances:
                        if hasattr(perf, 'jeopardy_score') and perf.jeopardy_score is not None:
                            # Create a mock result object for the leaderboard
                            mock_result = type('MockResult', (), {
                                'model_name': perf.model_name,
                                'benchmark_id': benchmark.id,
                                'metrics': type('MockMetrics', (), {
                                    'jeopardy_score': type('MockJeopardyScore', (), {
                                        'total_jeopardy_score': perf.jeopardy_score,
                                        'positive_scores': perf.correct_answers,
                                        'negative_scores': perf.total_questions - perf.correct_answers,
                                        'category_scores': perf.category_jeopardy_scores_dict
                                    })(),
                                    'accuracy': type('MockAccuracy', (), {
                                        'overall_accuracy': float(perf.accuracy_rate) if perf.accuracy_rate else 0
                                    })()
                                })()
                            })()
                            leaderboard_data.append(mock_result)
                
                if not leaderboard_data:
                    console.print("[yellow]No Jeopardy scores found in benchmarks. Run benchmarks with Jeopardy scoring enabled![/yellow]")
                    return
                
                # Generate leaderboard report
                report_config = ReportConfig(show_jeopardy_scores=True, show_leaderboard=True)
                report_gen = ReportGenerator(config=report_config)
                
                if report_format == 'terminal':
                    leaderboard_report = report_gen.generate_leaderboard_report(leaderboard_data[:limit])
                    console.print(leaderboard_report)
                else:
                    format_enum = ReportFormat.MARKDOWN if report_format == 'markdown' else ReportFormat.JSON
                    leaderboard_report = report_gen.generate_leaderboard_report(leaderboard_data[:limit], format_enum)
                    
                    if output:
                        with open(output, 'w') as f:
                            f.write(leaderboard_report)
                        console.print(f"[green]Leaderboard saved to {output}[/green]")
                    else:
                        console.print(leaderboard_report)
                
                console.print(f"\n[dim]Showing top {min(limit, len(leaderboard_data))} models from {len(completed_benchmarks)} completed benchmarks[/dim]")
                
        except Exception as e:
            console.print(f"[red]Error generating leaderboard: {str(e)}[/red]")
            logger.exception("Leaderboard generation failed")
            sys.exit(1)
    
    asyncio.run(show_leaderboard_async())


# Keep the old run command as an alias to benchmark run for backward compatibility
@cli.command()
@click.option('--sample-size', '-s', type=int, help='Number of questions to sample (deprecated: use benchmark run)')
@click.option('--models', '-m', multiple=True, help='Model names to benchmark (deprecated: use benchmark run)')
@click.option('--name', '-n', help='Benchmark name (deprecated: use benchmark run)')
@click.option('--description', '-d', help='Benchmark description (deprecated: use benchmark run)')
@click.option('--category', '-cat', help='Filter by category (deprecated: use benchmark run)')
@click.option('--difficulty', type=click.Choice(['Easy', 'Medium', 'Hard']), help='Filter by difficulty (deprecated: use benchmark run)')
@click.option('--dry-run', is_flag=True, help='Show what would be done without executing (deprecated: use benchmark run)')
@click.pass_context
def run(ctx, sample_size, models, name, description, category, difficulty, dry_run):
    """[DEPRECATED] Run a new benchmark - use 'benchmark run' instead."""
    
    console.print("[yellow]⚠️  The 'run' command is deprecated. Use 'benchmark run --model <model>' instead.[/yellow]")
    console.print("[dim]Example: benchmark run --model openai/gpt-4 --size standard[/dim]\n")
    
    if models:
        first_model = models[0]
        console.print(f"[blue]Tip: To run with {first_model}:[/blue]")
        console.print(f"[dim]benchmark run --model {first_model} --size standard[/dim]\n")


@cli.command()
@click.option('--limit', '-l', type=int, default=10, help='Number of benchmarks to show')
@click.option('--status', type=click.Choice(['pending', 'running', 'completed', 'failed']), help='Filter by status')
@click.pass_context
def list_benchmarks(ctx, limit, status):
    """List existing benchmarks."""
    
    # TODO: Implement benchmark listing
    console.print("[yellow]⚠️  Benchmark listing not yet implemented[/yellow]")
    
    # Mock data for demonstration
    table = Table(title="Benchmarks")
    table.add_column("ID", justify="right", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Status", justify="center")
    table.add_column("Models", style="blue")
    table.add_column("Questions", justify="right", style="green")
    table.add_column("Created", style="dim")
    
    # Mock entries
    table.add_row("1", "test-benchmark", "[green]Completed[/green]", "gpt-3.5-turbo", "1000", "2024-01-15")
    table.add_row("2", "comparison-test", "[yellow]Running[/yellow]", "claude-3-haiku, gpt-4", "500", "2024-01-16")
    
    console.print(table)
    console.print(f"\n[dim]Showing mock data - implement database queries to show real benchmarks[/dim]")


@cli.command()
@click.argument('benchmark_id', type=int)
@click.option('--detailed', '-d', is_flag=True, help='Show detailed results')
@click.pass_context  
def status(ctx, benchmark_id, detailed):
    """Show status and results for a specific benchmark."""
    
    # TODO: Implement status checking
    console.print(f"[yellow]⚠️  Status checking for benchmark {benchmark_id} not yet implemented[/yellow]")
    
    # Mock status display
    panel_content = f"""
    [bold]Benchmark ID:[/bold] {benchmark_id}
    [bold]Name:[/bold] test-benchmark
    [bold]Status:[/bold] [green]Completed[/green]
    [bold]Progress:[/bold] 1000/1000 questions processed
    [bold]Duration:[/bold] 45 minutes
    [bold]Models:[/bold] openai/gpt-3.5-turbo
    [bold]Average Accuracy:[/bold] 78.5%
    [bold]Total Cost:[/bold] $2.34
    """
    
    console.print(Panel(panel_content, title=f"Benchmark {benchmark_id} Status", border_style="green"))
    
    if detailed:
        console.print("\n[dim]Detailed results would be shown here[/dim]")


@cli.command()
@click.option('--force', is_flag=True, help='Force database recreation')
@click.pass_context
def init(ctx, force):
    """Initialize the database and create tables."""
    
    try:
        config = ctx.obj['config']
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            if force:
                progress.add_task("Recreating database...", total=None)
                # TODO: Implement database reset
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
        sys.exit(1)


@cli.command()
@click.option('--format', type=click.Choice(['table', 'json', 'yaml']), default='table', help='Output format')
@click.pass_context
def config_show(ctx, format):
    """Show current configuration."""
    
    config = ctx.obj['config']
    
    if format == 'table':
        table = Table(title="Configuration")
        table.add_column("Section", style="cyan")
        table.add_column("Setting", style="magenta")
        table.add_column("Value", style="green")
        
        # App settings
        table.add_row("app", "name", config.name)
        table.add_row("app", "version", config.version)
        table.add_row("app", "debug", str(config.debug))
        
        # Database settings
        table.add_row("database", "url", config.database.url)
        table.add_row("database", "echo", str(config.database.echo))
        
        # Benchmark settings
        table.add_row("benchmark", "default_sample_size", str(config.benchmark.default_sample_size))
        table.add_row("benchmark", "max_concurrent_requests", str(config.benchmark.max_concurrent_requests))
        
        console.print(table)
        
    elif format == 'json':
        # TODO: Implement JSON output
        console.print("[yellow]JSON format not yet implemented[/yellow]")
        
    elif format == 'yaml':
        # TODO: Implement YAML output  
        console.print("[yellow]YAML format not yet implemented[/yellow]")


@cli.group()
def models():
    """Model management commands."""
    pass


@models.command('list')
@click.option('--provider', '-p', help='Filter by provider (e.g., openai, anthropic)')
@click.option('--refresh', '-r', is_flag=True, help='Refresh model cache from OpenRouter API')
@click.option('--search', '-s', help='Search models by name or description')
@click.pass_context
def models_list(ctx, provider, refresh, search):
    """List available models from OpenRouter.
    
    \b
    🔍 EXAMPLES:
    
    alex models list
    alex models list --provider anthropic
    alex models list --provider openai
    
    alex models list --search gpt-4
    alex models list --search claude
    
    alex models list --refresh
    
    \b
    💡 TIP: Models are cached for 24 hours. Use --refresh to get the latest list.
    """
    
    async def list_models_async():
        try:
            from models.model_registry import model_registry
            from models.model_cache import get_model_cache
            
            console.print("[blue]Loading available models...[/blue]")
            
            # Get models using dynamic system
            if refresh:
                # Force refresh from API
                models = await model_registry.fetch_models()
                if not models:
                    console.print("[red]Failed to fetch models from API[/red]")
                    return
                console.print("[green]✓ Models refreshed from OpenRouter API[/green]")
            else:
                models = await model_registry.get_available_models()
            
            if not models:
                console.print("[yellow]No models available[/yellow]")
                return
            
            # Apply search filter
            if search:
                models = model_registry.search_models(search, models)
                if not models:
                    console.print(f"[yellow]No models found matching '{search}'[/yellow]")
                    return
            
            # Apply provider filter
            if provider:
                models = [m for m in models if m.get('provider', '').lower() == provider.lower()]
                if not models:
                    available_providers = sorted(set(m.get('provider', '') for m in models if m.get('provider')))
                    console.print(f"[red]No models found for provider: {provider}[/red]")
                    console.print(f"Available providers: {', '.join(available_providers)}")
                    return
            
            # Create table
            table = Table(title="Available Models")
            table.add_column("Provider", style="cyan")
            table.add_column("Model ID", style="magenta") 
            table.add_column("Display Name", style="blue")
            table.add_column("Context", justify="right", style="green")
            table.add_column("Cost (Input/Output per 1M)", style="yellow")
            table.add_column("Available", justify="center", style="dim")
            
            # Sort models by provider, then name
            sorted_models = sorted(models, key=lambda x: (x.get('provider', ''), x.get('name', '')))
            
            for model in sorted_models:
                # Extract pricing info
                pricing = model.get('pricing', {})
                input_cost = pricing.get('input_cost_per_1m_tokens', 0)
                output_cost = pricing.get('output_cost_per_1m_tokens', 0)
                
                table.add_row(
                    (model.get('provider', 'Unknown')).title(),
                    model.get('id', 'N/A'),
                    model.get('name', 'N/A'),
                    f"{model.get('context_length', 0):,}",
                    f"${input_cost:.2f}/${output_cost:.2f}",
                    "✓" if model.get('available', True) else "✗"
                )
            
            console.print(table)
            
            # Show summary and cache status
            console.print(f"\n[dim]Total models: {len(models)}[/dim]")
            
            if search:
                console.print(f"[dim]Filtered by search: '{search}'[/dim]")
            if provider:
                console.print(f"[dim]Filtered by provider: '{provider}'[/dim)")
            
            # Show cache status
            cache = get_model_cache()
            cache_info = cache.get_cache_info()
            if cache_info['exists']:
                status = "valid" if cache_info['valid'] else "expired"
                age_mins = cache_info['age_seconds'] / 60 if cache_info['age_seconds'] else 0
                console.print(f"[dim]Cache: {cache_info['model_count']} models, {status} (age: {age_mins:.1f} mins)[/dim]")
            else:
                console.print("[dim]Cache: No cached data[/dim]")
                
        except Exception as e:
            console.print(f"[red]Error listing models: {str(e)}[/red]")
            logger.exception("Model listing failed")
    
    asyncio.run(list_models_async())


@models.command('search')
@click.argument('query', required=True)
@click.option('--limit', '-l', type=int, default=20, help='Maximum number of results to show')
@click.pass_context
def models_search(ctx, query, limit):
    """Search for models by name, provider, or capabilities."""
    
    async def search_models_async():
        try:
            from models.model_registry import model_registry
            
            console.print(f"[blue]Searching for models matching '{query}'...[/blue]")
            
            # Get all available models and search
            models = await model_registry.get_available_models()
            matching_models = model_registry.search_models(query, models)
            
            if not matching_models:
                console.print(f"[yellow]No models found matching '{query}'[/yellow]")
                console.print("[dim]Try searching by provider (e.g., 'anthropic'), model family (e.g., 'gpt'), or capability[/dim]")
                return
            
            # Limit results
            if len(matching_models) > limit:
                matching_models = matching_models[:limit]
                console.print(f"[dim]Showing first {limit} results (use --limit to see more)[/dim]\n")
            
            # Create results table
            table = Table(title=f"Search Results: '{query}'")
            table.add_column("Provider", style="cyan")
            table.add_column("Model ID", style="magenta")
            table.add_column("Display Name", style="blue") 
            table.add_column("Context", justify="right", style="green")
            table.add_column("Cost", style="yellow")
            
            for model in matching_models:
                pricing = model.get('pricing', {})
                input_cost = pricing.get('input_cost_per_1m_tokens', 0)
                output_cost = pricing.get('output_cost_per_1m_tokens', 0)
                
                table.add_row(
                    (model.get('provider', 'Unknown')).title(),
                    model.get('id', 'N/A'),
                    model.get('name', 'N/A'),
                    f"{model.get('context_length', 0):,}",
                    f"${input_cost:.2f}/${output_cost:.2f}"
                )
            
            console.print(table)
            console.print(f"\n[green]Found {len(matching_models)} models matching '{query}'[/green]")
            
        except Exception as e:
            console.print(f"[red]Error searching models: {str(e)}[/red]")
            logger.exception("Model search failed")
    
    asyncio.run(search_models_async())


@models.command('info')
@click.argument('model_id', required=True)
@click.pass_context
def models_info(ctx, model_id):
    """Show detailed information about a specific model."""
    
    async def show_model_info_async():
        try:
            from models.model_registry import model_registry
            
            console.print(f"[blue]Getting information for model: {model_id}[/blue]")
            
            # Get all models and find the specific one
            models = await model_registry.get_available_models()
            model_info = None
            
            for model in models:
                if model.get('id', '').lower() == model_id.lower():
                    model_info = model
                    break
            
            if not model_info:
                console.print(f"[red]Model not found: {model_id}[/red]")
                console.print("[dim]Use 'models list' or 'models search' to find available models[/dim]")
                
                # Show similar models
                similar = model_registry.search_models(model_id.split('/')[-1], models)[:5]
                if similar:
                    console.print(f"\n[yellow]Similar models:[/yellow]")
                    for sim in similar:
                        console.print(f"  • {sim.get('id', 'N/A')}")
                return
            
            # Display detailed information
            console.print(Panel.fit(
                f"[bold blue]{model_info.get('name', 'N/A')}[/bold blue]\n"
                f"[dim]{model_info.get('description', 'No description available')}[/dim]",
                title="Model Information",
                border_style="blue"
            ))
            
            # Basic details table
            details_table = Table(title="Model Details")
            details_table.add_column("Property", style="cyan")
            details_table.add_column("Value", style="green")
            
            details_table.add_row("Model ID", model_info.get('id', 'N/A'))
            details_table.add_row("Provider", (model_info.get('provider', 'Unknown')).title())
            details_table.add_row("Context Length", f"{model_info.get('context_length', 0):,} tokens")
            details_table.add_row("Available", "✓ Yes" if model_info.get('available', True) else "✗ No")
            details_table.add_row("Modality", (model_info.get('modality', 'text')).title())
            
            # Add architecture info if available
            architecture = model_info.get('architecture', {})
            if architecture:
                if 'tokenizer' in architecture:
                    details_table.add_row("Tokenizer", architecture['tokenizer'])
                if 'instruct_type' in architecture:
                    details_table.add_row("Instruction Type", architecture['instruct_type'])
            
            console.print(details_table)
            
            # Pricing table
            pricing = model_info.get('pricing', {})
            if pricing:
                pricing_table = Table(title="Pricing Information")
                pricing_table.add_column("Type", style="cyan")
                pricing_table.add_column("Cost per 1M tokens", style="yellow")
                
                input_cost = pricing.get('input_cost_per_1m_tokens', 0)
                output_cost = pricing.get('output_cost_per_1m_tokens', 0)
                
                # Format costs properly, handling scientific notation
                def format_cost(cost):
                    price_per_million = cost * 1_000_000
                    if cost == 0:
                        return "$0.0000"
                    else:
                        return f"${price_per_million:,.0f}"
                
                pricing_table.add_row("Input", format_cost(input_cost))
                pricing_table.add_row("Output", format_cost(output_cost))
                pricing_table.add_row("Combined", format_cost(input_cost + output_cost))
                
                console.print(pricing_table)
            
            # Top provider info
            top_provider = model_info.get('top_provider', {})
            if top_provider:
                console.print(f"\n[bold]Top Provider:[/bold]")
                console.print(f"• Max completion tokens: {top_provider.get('max_completion_tokens', 'N/A')}")
                console.print(f"• Max throughput: {top_provider.get('max_throughput_tokens_per_minute', 'N/A')} tokens/min")
            
            # Per-request limits
            limits = model_info.get('per_request_limits', {})
            if limits:
                console.print(f"\n[bold]Request Limits:[/bold]")
                if 'prompt_tokens' in limits:
                    console.print(f"• Max prompt tokens: {limits['prompt_tokens']:,}")
                if 'completion_tokens' in limits:
                    console.print(f"• Max completion tokens: {limits['completion_tokens']:,}")
            
        except Exception as e:
            console.print(f"[red]Error getting model info: {str(e)}[/red]")
            logger.exception("Model info retrieval failed")
    
    asyncio.run(show_model_info_async())


@models.command('refresh')
@click.pass_context  
def models_refresh(ctx):
    """Force refresh the model cache from OpenRouter API."""
    
    async def refresh_models_async():
        try:
            from models.model_registry import model_registry
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Fetching models from OpenRouter API...", total=None)
                
                models = await model_registry.fetch_models()
                
                if models:
                    progress.update(task, description="Complete!")
                    progress.stop()
                    
                    console.print(f"[green]✓ Successfully refreshed {len(models)} models from OpenRouter API[/green]")
                    console.print("[dim]Use 'models list' to see the updated model list[/dim]")
                else:
                    progress.update(task, description="Failed!")
                    progress.stop()
                    console.print("[red]✗ Failed to fetch models from OpenRouter API[/red]")
                    console.print("[dim]Check your API key and network connection[/dim]")
                    
        except Exception as e:
            console.print(f"[red]Error refreshing models: {str(e)}[/red]")
            logger.exception("Model refresh failed")
    
    asyncio.run(refresh_models_async())


@models.command('cache')
@click.option('--clear', is_flag=True, help='Clear the model cache')
@click.option('--info', is_flag=True, help='Show detailed cache information', default=True)
@click.pass_context
def models_cache(ctx, clear, info):
    """Manage model cache."""
    try:
        from models.model_cache import get_model_cache
        
        cache = get_model_cache()
        
        if clear:
            if cache.clear_cache():
                console.print("[green]✓ Model cache cleared[/green]")
            else:
                console.print("[red]✗ Failed to clear cache[/red]")
            return
        
        if info:
            cache_info = cache.get_cache_info()
            
            # Cache status table
            status_table = Table(title="Model Cache Status")
            status_table.add_column("Property", style="cyan")
            status_table.add_column("Value", style="green")
            
            status_table.add_row("Cache Path", cache_info['cache_path'])
            status_table.add_row("Exists", "✓ Yes" if cache_info['exists'] else "✗ No")
            status_table.add_row("Valid", "✓ Yes" if cache_info['valid'] else "✗ No")
            status_table.add_row("TTL", f"{cache_info['ttl_seconds']} seconds")
            
            if cache_info['exists']:
                status_table.add_row("Size", f"{cache_info['size_bytes']:,} bytes")
                status_table.add_row("Model Count", str(cache_info['model_count']))
                
                if cache_info['cached_at']:
                    status_table.add_row("Cached At", cache_info['cached_at'])
                
                if cache_info['age_seconds'] is not None:
                    age_mins = cache_info['age_seconds'] / 60
                    age_hours = age_mins / 60
                    if age_hours > 1:
                        age_str = f"{age_hours:.1f} hours"
                    else:
                        age_str = f"{age_mins:.1f} minutes"
                    status_table.add_row("Age", age_str)
            
            console.print(status_table)
            
            # Cache recommendations
            if not cache_info['exists']:
                console.print("\n[yellow]💡 Run 'models refresh' to populate the cache[/yellow]")
            elif not cache_info['valid']:
                console.print("\n[yellow]💡 Cache has expired. Run 'models refresh' to update[/yellow]")
            else:
                console.print("\n[green]💡 Cache is up to date[/green]")
            
    except Exception as e:
        console.print(f"[red]Error managing cache: {str(e)}[/red]")
        logger.exception("Cache management failed")


@models.command('test')
@click.option('--model', '-m', required=True, help='Model ID to test')
@click.option('--prompt', '-p', default="What is the capital of France?", help='Test prompt')
@click.pass_context
def models_test(ctx, model, prompt):
    """Test a specific model with a prompt."""
    
    async def run_test():
        try:
            from models.model_registry import model_registry
            from models.openrouter import OpenRouterClient
            from models.base import ModelConfig
            
            # Validate model using dynamic system
            models = await model_registry.get_available_models()
            model_info = None
            
            for m in models:
                if m.get('id', '').lower() == model.lower():
                    model_info = m
                    break
            
            if not model_info:
                console.print(f"[red]Model not found: {model}[/red]")
                console.print("[dim]Use 'models list' or 'models search' to find available models[/dim]")
                
                # Show source of models being used
                cache_info = model_registry._get_cache().get_cache_info()
                if cache_info['valid']:
                    console.print("[dim]Using cached models from API[/dim]")
                else:
                    console.print("[dim]Using static fallback models[/dim]")
                return
            
            console.print(f"[blue]Testing model: {model_info.get('name', model)}[/blue]")
            console.print(f"[dim]Provider: {model_info.get('provider', 'Unknown')}[/dim]")
            console.print(f"[dim]Source: {'API/Cache' if model_info.get('available', True) else 'Static Fallback'}[/dim]")
            console.print(f"[dim]Prompt: {prompt}[/dim]\n")
            
            # Create client and test
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Querying model...", total=None)
                
                # Create client with proper model configuration
                model_config = ModelConfig(model_name=model)
                client = OpenRouterClient(config=model_config)
                
                # Use the correct method name 'query'
                response = await client.query(prompt)
                
                progress.update(task, description="Complete!")
                progress.stop()
                
                # Display results  
                result_table = Table(title="Test Results")
                result_table.add_column("Metric", style="cyan")
                result_table.add_column("Value", style="green")
                
                result_table.add_row("Response", response.response)
                result_table.add_row("Latency", f"{response.latency_ms:.0f} ms")
                result_table.add_row("Tokens Used", str(response.tokens_used))
                result_table.add_row("Cost", f"${response.cost:.6f}")
                result_table.add_row("Model", response.model_id)
                
                console.print(result_table)
                
                # Clean up
                await client.close()
                
        except Exception as e:
            console.print(f"[red]Test failed: {str(e)}[/red]")
            logger.exception("Model test failed")
    
    asyncio.run(run_test())


@models.command('costs')
@click.option('--model', '-m', required=True, help='Model ID to estimate costs for')
@click.option('--questions', '-q', type=int, default=100, help='Number of questions')
@click.option('--input-tokens', type=int, help='Average input tokens per question')
@click.option('--output-tokens', type=int, help='Average output tokens per question')
@click.pass_context
def models_costs(ctx, model, questions, input_tokens, output_tokens):
    """Estimate costs for running benchmarks with a model."""
    
    async def calculate_costs_async():
        try:
            from models.model_registry import model_registry
            from models.cost_calculator import CostCalculator
            
            # Validate model using dynamic system
            models = await model_registry.get_available_models()
            model_info = None
            
            for m in models:
                if m.get('id', '').lower() == model.lower():
                    model_info = m
                    break
            
            if not model_info:
                console.print(f"[red]Model not found: {model}[/red]")
                console.print("[dim]Use 'models list' or 'models search' to find available models[/dim]")
                return
            
            # Use defaults if not specified - fix variable scoping
            default_input_tokens = 100
            default_output_tokens = 50
            
            config = ctx.obj.get('config')
            if config and hasattr(config, 'costs') and hasattr(config.costs, 'estimation'):
                try:
                    default_input_tokens = getattr(config.costs.estimation, 'default_input_tokens_per_question', 100)
                    default_output_tokens = getattr(config.costs.estimation, 'default_output_tokens_per_question', 50)
                except AttributeError:
                    pass  # Use defaults
            
            # Apply the values - use different variable names to avoid shadowing
            actual_input_tokens = input_tokens if input_tokens is not None else default_input_tokens
            actual_output_tokens = output_tokens if output_tokens is not None else default_output_tokens
            
            # Get pricing from model info
            pricing = model_info.get('pricing', {})
            input_cost_per_1m = pricing.get('input_cost_per_1m_tokens', 0)
            output_cost_per_1m = pricing.get('output_cost_per_1m_tokens', 0)
            
            # Calculate costs
            total_input_tokens = questions * actual_input_tokens
            total_output_tokens = questions * actual_output_tokens
            total_tokens = total_input_tokens + total_output_tokens
            
            input_cost = (total_input_tokens / 1_000_000) * input_cost_per_1m
            output_cost = (total_output_tokens / 1_000_000) * output_cost_per_1m
            total_cost = input_cost + output_cost
            cost_per_question = total_cost / questions if questions > 0 else 0
            
            # Display estimate
            table = Table(title=f"Cost Estimate: {model_info.get('name', model)}")
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Model ID", model)
            table.add_row("Model Name", model_info.get('name', 'N/A'))
            table.add_row("Provider", (model_info.get('provider', 'Unknown')).title())
            table.add_row("Questions", f"{questions:,}")
            table.add_row("Input Tokens per Question", f"{actual_input_tokens:,}")
            table.add_row("Output Tokens per Question", f"{actual_output_tokens:,}")
            table.add_row("Total Input Tokens", f"{total_input_tokens:,}")
            table.add_row("Total Output Tokens", f"{total_output_tokens:,}")
            table.add_row("Total Tokens", f"{total_tokens:,}")
            table.add_row("Input Cost", f"${input_cost:.6f}")
            table.add_row("Output Cost", f"${output_cost:.6f}")
            table.add_row("Total Cost", f"${total_cost:.4f}")
            table.add_row("Cost per Question", f"${cost_per_question:.6f}")
            
            console.print(table)
            
            # Add context about pricing
            if input_cost_per_1m == 0 and output_cost_per_1m == 0:
                console.print("\n[yellow]⚠️  No pricing information available for this model[/yellow]")
            else:
                console.print(f"\n[dim]Based on: ${input_cost_per_1m:.2f}/${output_cost_per_1m:.2f} per 1M input/output tokens[/dim]")
            
        except Exception as e:
            console.print(f"[red]Error calculating costs: {str(e)}[/red]")
            logger.exception("Cost calculation failed")
    
    asyncio.run(calculate_costs_async())


@cli.command()
@click.option('--check-db', is_flag=True, help='Check database connection')
@click.option('--check-api', is_flag=True, help='Check API connections')
@click.pass_context
def health(ctx, check_db, check_api):
    """Check system health and connectivity."""
    
    config = ctx.obj['config']
    all_good = True
    
    # Check database
    if check_db or not (check_api):
        try:
            if check_database_connection():
                console.print("[green]✓ Database connection: OK[/green]")
            else:
                console.print("[red]✗ Database connection: FAILED[/red]")
                all_good = False
        except Exception as e:
            console.print(f"[red]✗ Database connection: ERROR - {str(e)}[/red]")
            all_good = False
    
    # Check API connections  
    if check_API or not (check_db):
        # TODO: Implement API health checks
        console.print("[yellow]⚠️  API health checks not yet implemented[/yellow]")
        console.print("[dim]Would check: OpenRouter API, Kaggle API[/dim]")
    
    if all_good:
        console.print("\n[green]System health: OK[/green]")
    else:
        console.print("\n[red]System health: ISSUES DETECTED[/red]")
        sys.exit(1)


@cli.command()
@click.argument('benchmark_id', type=int)
@click.option('--format', type=click.Choice(['json', 'csv', 'html']), default='json', help='Export format')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.pass_context
def export(ctx, benchmark_id, format, output):
    """Export benchmark results."""
    
    # TODO: Implement result export
    console.print(f"[yellow]⚠️  Export Export benchmark {benchmark_id} to {format} not yet implemented[/yellow]")
    
    if output:
        console.print(f"[dim]Would export to: {output}[/dim]")
    else:
        console.print(f"[dim]Would export to: benchmark_{benchmark_id}.{format}[/dim]")


@cli.group()
def data():
    """Data management commands."""
    pass


@data.command('init')
@click.option('--force', is_flag=True, help='Force re-download of dataset')
@click.option('--strict', is_flag=True, help='Use strict validation rules')
@click.pass_context
def data_init(ctx, force, strict):
    """Initialize the Jeopardy dataset."""
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
        sys.exit(1)


@data.command('stats')
@click.option('--benchmark-id', '-b', type=int, help='Show stats for specific benchmark')
@click.option('--detailed', '-d', is_flag=True, help='Show detailed statistics')
@click.pass_context
def data_stats(ctx, benchmark_id, detailed):
    """Show dataset statistics."""
    try:
        from storage.repositories import QuestionRepository
        from core.database import get_db_session
        
        with get_db_session() as session:
            repo = QuestionRepository(session)
            stats = repo.get_question_statistics(benchmark_id)
            
            if not stats or stats['total_questions'] == 0:
                if benchmark_id:
                    console.print(f"[yellow]No questions found for benchmark {benchmark_id}[/yellow]")
                else:
                    console.print("[yellow]No questions found in database. Run 'data init' first.[/yellow]")
                return
            
            # Display basic statistics
            table = Table(title="Dataset Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Total Questions", f"{stats['total_questions']:,}")
            table.add_row("Unique Categories", f"{stats['unique_categories']:,}");
            
            if stats['value_range']:
                table.add_row("Value Range", f"${stats['value_range']['min']} - ${stats['value_range']['max']}")
                table.add_row("Average Value", f"${stats['value_range']['average']}")
                table.add_row("Questions with Values", f"{stats['value_range']['count']:,}");
            
            console.print(table)
            
            if detailed:
                # Show category distribution
                if stats['category_distribution']:
                    cat_table = Table(title="Top 10 Categories")
                    cat_table.add_column("Category", style="cyan")
                    cat_table.add_column("Count", justify="right", style="green")
                    cat_table.add_column("Percentage", justify="right", style="yellow")
                    
                    sorted_cats = sorted(stats['category_distribution'].items(),
                                       key=lambda x: x[1], reverse=True)[:10]
                    
                    for category, count in sorted_cats:
                        percentage = (count / stats['total_questions']) * 100
                        cat_table.add_row(category, str(count), f"{percentage:.1f}%")
                    
                    console.print(cat_table)
                
                # Show difficulty distribution
                if stats['difficulty_distribution']:
                    diff_table = Table(title="Difficulty Distribution")
                    diff_table.add_column("Difficulty", style="cyan")
                    diff_table.add_column("Count", justify="right", style="green")
                    diff_table.add_column("Percentage", justify="right", style="yellow")
                    
                    for difficulty, count in stats['difficulty_distribution'].items():
                        percentage = (count / stats['total_questions']) * 100
                        diff_table.add_row(difficulty, str(count), f"{percentage:.1f}%")
                    
                    console.print(diff_table)
    
    except Exception as e:
        console.print(f"[red]Error retrieving statistics: {str(e)}[/red]")
        sys.exit(1)


@data.command('sample')
@click.option('--size', '-s', type=int, default=100, help='Sample size')
@click.option('--category', '-c', help='Filter by category')
@click.option('--difficulty', '-d', type=click.Choice(['Easy', 'Medium', 'Hard']), help='Filter by difficulty')
@click.option('--min-value', type=int, help='Minimum dollar value')
@click.option('--max-value', type=int, help='Maximum dollar value')
@click.option('--method', type=click.Choice(['random', 'stratified', 'balanced']),
              default='stratified', help='Sampling method')
@click.option('--seed', type=int, help='Random seed for reproducibility')
@click.option('--output', '-o', type=click.Path(), help='Save sample to CSV file')
@click.pass_context
def data_sample(ctx, size, category, difficulty, min_value, max_value, method, seed, output):
    """Test sampling functionality with the dataset."""
    try:
        from storage.repositories import QuestionRepository
        from data.sampling import StatisticalSampler
        from data.preprocessing import DataPreprocessor
        from core.database import get_db_session
        import pandas as pd
        
        # Get questions from database
        with get_db_session() as session:
            repo = QuestionRepository(session)
            
            # Build filters
            filters = {}
            if category:
                filters['categories'] = [category]
            if difficulty:
                filters['difficulty_levels'] = [difficulty]
            if min_value is not None:
                filters['min_value'] = min_value
            if max_value is not None:
                filters['max_value'] = max_value
            
            # Get questions
            questions = repo.get_questions(filters)
            
            if not questions:
                console.print("[yellow]No questions found matching the specified criteria[/yellow]")
                return
            
            # Convert to DataFrame
            data = []
            for q in questions:
                data.append({
                    'question_id': q.id,
                    'question': q.question_text,
                    'answer': q.correct_answer,
                    'category': q.category,
                    'value': q.value,
                    'difficulty_level': q.difficulty_level
                })
            
            df = pd.DataFrame(data)
            console.print(f"Found {len(df)} questions matching criteria")
        
        # Initialize sampler
        sampler = StatisticalSampler()
        
        # Apply sampling method
        if method == 'random':
            sample_df = sampler.random_sample(df, size, seed)
        elif method == 'stratified':
            sample_df = sampler.stratified_sample(df, size, seed=seed)
        elif method == 'balanced':
            sample_df = sampler.balanced_difficulty_sample(df, size, seed=seed)
        else:
            sample_df = sampler.stratified_sample(df, size, seed=seed)
        
        # Display sample information
        console.print(f"[green]✓ Generated {method} sample of {len(sample_df)} questions[/green]")
        
        # Show sample statistics
        stats_table = Table(title="Sample Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Sample Size", str(len(sample_df)))
        stats_table.add_row("Sampling Method", method.title())
        
        if 'category' in sample_df.columns:
            stats_table.add_row("Unique Categories", str(sample_df['category'].nunique()))
        
        if 'difficulty_level' in sample_df.columns:
            diff_counts = sample_df['difficulty_level'].value_counts()
            stats_table.add_row("Difficulty Breakdown",
                              ", ".join([f"{k}: {v}" for k, v in diff_counts.items()]))
        
        if 'value' in sample_df.columns and sample_df['value'].notna().any():
            stats_table.add_row("Value Range",
                              f"${sample_df['value'].min():.0f} - ${sample_df['value'].max():.0f}")
        
        console.print(stats_table)
        
        # Show sample preview
        if len(sample_df) > 0:
            preview_table = Table(title="Sample Preview (first 5 questions)")
            preview_table.add_column("Category", style="cyan", max_width=20)
            preview_table.add_column("Question", style="white", max_width=50)
            preview_table.add_column("Answer", style="green", max_width=30)
            preview_table.add_column("Value", justify="right", style="yellow")
            
            for _, row in sample_df.head(5).iterrows():
                question_preview = (row['question'][:47] + "...") if len(str(row['question'])) > 50 else str(row['question'])
                answer_preview = (row['answer'][:27] + "...") if len(str(row['answer'])) > 30 else str(row['answer'])
                
                preview_table.add_row(
                    str(row.get('category', 'N/A'))[:20],
                    question_preview,
                    answer_preview,
                    f"${row.get('value', 0):.0f}" if pd.notna(row.get('value')) else "N/A"
                )
            
            console.print(preview_table)
        
        # Save to file if requested
        if output:
            sample_df.to_csv(output, index=False)
            console.print(f"[green]✓ Sample saved to {output}[/green]")
    
    except Exception as e:
        console.print(f"[red]Error during sampling: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
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
