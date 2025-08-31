"""
CLI Entry Point

Main command-line interface for the Jeopardy Benchmarking System
using Click framework with rich output formatting.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional, List
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.config import get_config, reload_config
from core.database import init_database, check_database_connection, get_db_session
from core.exceptions import JeopardyBenchException
from utils.logging import setup_logging, get_logger

console = Console()
logger = get_logger(__name__)

@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.pass_context
def cli(ctx, config, verbose, debug):
    """Jeopardy Benchmarking System - Benchmark language models using Jeopardy questions."""
    
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
@click.option('--model', '-m', required=True, help='Model to benchmark (e.g., openai/gpt-4)')
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
@click.pass_context
def benchmark_run(ctx, model, size, name, description, timeout, grading_mode, save_results, report_format):
    """Run a benchmark for a specific model."""
    
    async def run_benchmark_async():
        try:
            from benchmark.runner import BenchmarkRunner, RunMode, BenchmarkConfig
            from benchmark.reporting import ReportGenerator, ReportFormat
            from evaluation.grader import GradingMode
            
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
            
            console.print(f"[blue]Starting {run_mode.value} benchmark for {model}[/blue]")
            console.print(f"[dim]Sample size: {config.sample_size}, Grading: {grading_mode}[/dim]\n")
            
            # Show progress
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Running benchmark...", total=None)
                
                # Run the benchmark
                result = await runner.run_benchmark(
                    model_name=model,
                    mode=run_mode,
                    custom_config=config,
                    benchmark_name=name
                )
                
                progress.update(task, description="Generating report...")
                
                # Generate and display report
                report_gen = ReportGenerator()
                
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
    """Compare multiple models using benchmarks."""
    
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
                    str(benchmark.question_count),
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
    """Generate a report for a specific benchmark run."""
    
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
                    str(benchmark.question_count),
                    benchmark.created_at.strftime('%Y-%m-%d %H:%M') if benchmark.created_at else 'N/A'
                )
            
            console.print(table)
            console.print(f"\n[dim]Showing {len(benchmarks)} benchmarks[/dim]")
            
    except Exception as e:
        console.print(f"[red]Error listing benchmarks: {str(e)}[/red]")
        logger.exception("Benchmark listing failed")


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
    table.add_row("1", "test-benchmark", "[green]completed[/green]", "gpt-3.5-turbo", "1000", "2024-01-15")
    table.add_row("2", "comparison-test", "[yellow]running[/yellow]", "claude-3-haiku, gpt-4", "500", "2024-01-16")
    
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
@click.option('--provider', help='Filter by provider (e.g., openai, anthropic)')
@click.pass_context
def models_list(ctx, provider):
    """List available models."""
    try:
        from models.model_registry import ModelRegistry
        from models.model_registry import ModelProvider
        
        # Get models from registry
        all_models = list(ModelRegistry.MODELS.values())
        
        # Filter by provider if specified
        if provider:
            provider_enum = None
            for p in ModelProvider:
                if p.value.lower() == provider.lower():
                    provider_enum = p
                    break
            
            if provider_enum:
                all_models = [m for m in all_models if m.provider == provider_enum]
            else:
                console.print(f"[red]Unknown provider: {provider}[/red]")
                console.print(f"Available providers: {', '.join([p.value for p in ModelProvider])}")
                return
        
        if not all_models:
            console.print("[yellow]No models found matching the criteria[/yellow]")
            return
        
        # Create table
        table = Table(title="Available Models")
        table.add_column("Provider", style="cyan")
        table.add_column("Model", style="magenta")
        table.add_column("Display Name", style="blue")
        table.add_column("Context", justify="right", style="green")
        table.add_column("Cost (Input/Output per 1M)", style="yellow")
        table.add_column("Streaming", justify="center", style="dim")
        
        for model_config in sorted(all_models, key=lambda x: (x.provider.value, x.display_name)):
            table.add_row(
                model_config.provider.value.title(),
                model_config.model_id,
                model_config.display_name,
                f"{model_config.context_window:,}",
                f"${model_config.input_cost_per_1m_tokens:.2f}/${model_config.output_cost_per_1m_tokens:.2f}",
                "✓" if model_config.supports_streaming else "✗"
            )
        
        console.print(table)
        console.print(f"\n[dim]Total models: {len(all_models)}[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error listing models: {str(e)}[/red]")


@models.command('test')
@click.option('--model', '-m', required=True, help='Model ID to test')
@click.option('--prompt', '-p', default="What is the capital of France?", help='Test prompt')
@click.pass_context
def models_test(ctx, model, prompt):
    """Test a specific model with a prompt."""
    async def run_test():
        try:
            from models.model_registry import ModelRegistry
            from models.openrouter import OpenRouterClient
            from models.base import ModelConfig
            
            # Validate model
            if not ModelRegistry.validate_model_availability(model):
                console.print(f"[red]Unknown model: {model}[/red]")
                console.print("Use 'models list' to see available models")
                return
            
            model_config = ModelRegistry.get_model_config(model)
            console.print(f"[blue]Testing model: {model_config.display_name}[/blue]")
            console.print(f"[dim]Provider: {model_config.provider.value}[/dim]")
            console.print(f"[dim]Prompt: {prompt}[/dim]\n")
            
            # Create client
            config = ModelConfig(model_name=model)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Querying model...", total=None)
                
                async with OpenRouterClient(config=config) as client:
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
                    
                    console.print(result_table)
                    
        except Exception as e:
            console.print(f"[red]Test failed: {str(e)}[/red]")
    
    asyncio.run(run_test())


@models.command('costs')
@click.option('--model', '-m', required=True, help='Model ID to estimate costs for')
@click.option('--questions', '-q', type=int, default=100, help='Number of questions')
@click.option('--input-tokens', type=int, help='Average input tokens per question')
@click.option('--output-tokens', type=int, help='Average output tokens per question')
@click.pass_context
def models_costs(ctx, model, questions, input_tokens, output_tokens):
    """Estimate costs for running benchmarks with a model."""
    try:
        from models.model_registry import ModelRegistry
        from models.cost_calculator import CostCalculator
        
        # Validate model
        if not ModelRegistry.validate_model_availability(model):
            console.print(f"[red]Unknown model: {model}[/red]")
            console.print("Use 'models list' to see available models")
            return
        
        # Use defaults if not specified
        config = ctx.obj['config']
        input_tokens = input_tokens or config.costs.estimation.default_input_tokens_per_question
        output_tokens = output_tokens or config.costs.estimation.default_output_tokens_per_question
        
        # Calculate costs
        calculator = CostCalculator()
        estimate = calculator.estimate_batch_cost(
            model,
            ["dummy"] * questions,  # Just for count
            input_tokens,
            output_tokens
        )
        
        # Display estimate
        table = Table(title=f"Cost Estimate: {estimate['model_name']}")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Model", estimate['model_name'])
        table.add_row("Questions", f"{questions:,}")
        table.add_row("Est. Input Tokens", f"{estimate['estimated_input_tokens']:,}")
        table.add_row("Est. Output Tokens", f"{estimate['estimated_output_tokens']:,}")
        table.add_row("Total Tokens", f"{estimate['total_tokens']:,}")
        table.add_row("Total Cost", f"${estimate['estimated_total_cost']:.4f}")
        table.add_row("Cost per Question", f"${estimate['cost_per_question']:.6f}")
        table.add_row("Billing Tier", estimate['billing_tier'].title())
        
        if estimate['tier_discount'] > 0:
            table.add_row("Discount", f"{estimate['tier_discount']:.1f}%")
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error calculating costs: {str(e)}[/red]")


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
    if check_api or not (check_db):
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
    console.print(f"[yellow]⚠️  Exporting benchmark {benchmark_id} to {format} not yet implemented[/yellow]")
    
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
            table.add_row("Unique Categories", f"{stats['unique_categories']:,}")
            
            if stats['value_range']:
                table.add_row("Value Range", f"${stats['value_range']['min']} - ${stats['value_range']['max']}")
                table.add_row("Average Value", f"${stats['value_range']['average']}")
                table.add_row("Questions with Values", f"{stats['value_range']['count']:,}")
            
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
                    'question_id': q.question_id,
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
            sample_df = sampler.stratified_sample(df, size)
        elif method == 'balanced':
            sample_df = sampler.balanced_difficulty_sample(df, size)
        else:
            sample_df = sampler.stratified_sample(df, size)
        
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
    except JeopardyBenchException as e:
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