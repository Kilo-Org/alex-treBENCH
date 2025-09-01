"""
Benchmark Run Commands

This module contains the benchmark run command implementation.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from core.database import get_db_session
from utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.command()
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
def run(ctx, model, size, name, description, timeout, grading_mode, save_results, report_format, show_jeopardy_scores, list_models):
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