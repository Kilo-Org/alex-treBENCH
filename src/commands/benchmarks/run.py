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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from src.cli.formatting import format_progress

from models.model_registry import model_registry, get_default_model
from benchmark.runner import BenchmarkRunner, RunMode, BenchmarkConfig
from benchmark.reporting import ReportGenerator, ReportFormat
from src.core.database import get_db_session
from evaluation.grader import GradingMode
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
@click.option('--debug/--no-debug', default=False, help='Enable detailed debug logging of prompts and responses')
@click.option('--debug-errors-only', is_flag=True, help='Only log incorrect answers and errors (requires --debug)')
@click.option('--list-models', is_flag=True, help='Show available models and exit')
@click.pass_context
def run(ctx, model, size, name, description, timeout, grading_mode, save_results, report_format, show_jeopardy_scores, debug, debug_errors_only, list_models):
    """Run a benchmark for a specific model.
    
    \b
    📋 EXAMPLES:
    
    alex benchmark run --size quick 
    
    alex benchmark run --model openai/gpt-4 --size standard

    alex benchmark run --model anthropic/claude-3.5-sonnet --size comprehensive --name "Claude Production Test"

    alex benchmark run --model anthropic/claude-3-haiku --size quick --grading-mode lenient
    
    alex benchmark run --model openai/gpt-4 --size standard --show-jeopardy-scores
    
    alex benchmark run --model anthropic/claude-3.5-sonnet --size quick --no-jeopardy-scores
    
    alex benchmark run --debug --model openai/gpt-4 --size quick
    
    alex benchmark run --debug --debug-errors-only --model anthropic/claude-3-haiku --size standard
    
    alex benchmark run --list-models
    
    \b
    💡 SIZES: quick=10 questions, standard=50 questions, comprehensive=200 questions
    💰 JEOPARDY SCORES: Shows winnings/losses based on question values (enabled by default)
    🐛 DEBUG MODE: Logs all prompts, responses, and grading details to files in logs/debug/
    """
    
    async def run_benchmark_async():
        try:
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
                console.print(f"[dim]Default model: {get_default_model()}[/dim]")
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
            
            # Handle debug configuration
            if debug:
                # Override global config for debug mode
                from src.core.config import get_config
                config_instance = get_config()
                
                # Temporarily override debug settings
                original_debug_enabled = config_instance.logging.debug.enabled
                original_debug_errors_only = config_instance.logging.debug.log_errors_only
                
                config_instance.logging.debug.enabled = True
                config_instance.logging.debug.log_errors_only = debug_errors_only
                
                console.print("[yellow]🐛 Debug mode enabled - detailed logs will be saved to logs/debug/[/yellow]")
                if debug_errors_only:
                    console.print("[dim]Only logging incorrect answers and errors[/dim]")
            
            # Initialize runner (will pick up debug config)
            runner = BenchmarkRunner()
            
            # Set display name using validated model_info
            if model_info:
                # Use the name directly as it already includes provider prefix
                model_name = model_info.get('name', actual_model)
                model_id_short = model_info.get('id', actual_model).split('/')[-1]
                
                display_name = f"{model_name} ({model_id_short})"
                pricing = model_info.get('pricing', {})
            else:
                display_name = f"Unknown Model: {actual_model}"
                pricing = {}
            
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
            
            # Get configuration based on mode
            config = None
            if run_mode:
                if run_mode.upper() == 'QUICK':
                    config = BenchmarkConfig(mode=RunMode.QUICK, sample_size=50)
                elif run_mode.upper() == 'STANDARD':
                    config = BenchmarkConfig(mode=RunMode.STANDARD, sample_size=200)
                elif run_mode.upper() == 'COMPREHENSIVE':
                    config = BenchmarkConfig(mode=RunMode.COMPREHENSIVE, sample_size=1000)
                else:
                    config = BenchmarkConfig(mode=RunMode.STANDARD, sample_size=200)
            else:
                config = BenchmarkConfig(mode=RunMode.STANDARD, sample_size=200)
            
            # Apply additional configuration
            if timeout:
                config.timeout_seconds = timeout
            config.grading_mode = grading_mode_enum
            config.save_results = save_results
            
            # Display benchmark info
            console.print(f"\n[cyan]Starting {config.mode.value} benchmark[/cyan]")
            console.print(f"[cyan]Model: {display_name}[/cyan]")
            console.print(f"[dim]Sample size: {config.sample_size}, Grading: {grading_mode}[/dim]")
            
            # Estimate cost using ModelRegistry
            from src.models.model_registry import ModelRegistry
            
            # Estimate tokens per question (input + output)
            estimated_input_tokens_per_q = 100  # Average prompt size
            estimated_output_tokens_per_q = 50   # Average response size
            
            total_input_tokens = config.sample_size * estimated_input_tokens_per_q
            total_output_tokens = config.sample_size * estimated_output_tokens_per_q
            
            estimated_cost = ModelRegistry.estimate_cost(actual_model, total_input_tokens, total_output_tokens)
            
            if estimated_cost > 0:
                console.print(f"[dim]Estimated cost: ~${estimated_cost:.4f}[/dim]")
            else:
                # Fall back to basic calculation if no pricing found
                if pricing and (pricing.get('input_cost_per_1m_tokens', 0) > 0 or pricing.get('output_cost_per_1m_tokens', 0) > 0):
                    fallback_cost = ((estimated_input_tokens_per_q + estimated_output_tokens_per_q) * config.sample_size / 1_000_000) * (pricing.get('input_cost_per_1m_tokens', 0) + pricing.get('output_cost_per_1m_tokens', 0))
                    console.print(f"[dim]Estimated cost: ~${fallback_cost:.4f}[/dim]")
            
            console.print()
            
            # Run benchmark with progress tracking
            progress_instance = format_progress("Running benchmark...", config.sample_size, show_count=True)
            with progress_instance as progress:
                task = progress.add_task("[green]Running benchmark...", total=config.sample_size)
                
                # Execute benchmark
                result = await runner.run_benchmark(
                    model_name=actual_model,
                    mode=RunMode[config.mode.value.upper()],
                    custom_config=config,
                    progress=progress,
                    task_id=task
                )
                
                progress.update(task, completed=config.sample_size, description="[green]Complete!")
            
            # Display results
            if result.is_successful:
                console.print(f"\n[green]✓ Benchmark completed successfully in {result.execution_time_seconds:.2f}s[/green]")
                
                # Basic results display
                if result.progress:
                    accuracy = result.progress.success_rate
                    console.print(f"[green]Accuracy: {accuracy:.1f}%[/green]")
                    console.print(f"[dim]Questions completed: {result.progress.completed_questions}/{result.progress.total_questions}[/dim]")
                
                if result.total_cost > 0:
                    console.print(f"[dim]Total cost: ${result.total_cost:.6f}[/dim]")
                
                # Show debug log location if debug was enabled
                if debug:
                    console.print(f"\n[yellow]🐛 Debug logs saved to: logs/debug/[/yellow]")
                    console.print("[dim]Files: model_interactions_*.jsonl (structured data) and debug_summary_*.log (readable)[/dim]")
                    
            else:
                error_msg = result.errors[0] if result.errors else "Unknown error"
                console.print(f"\n[red]✗ Benchmark failed: {error_msg}[/red]")
                console.print(f"[dim]Execution time: {result.execution_time_seconds:.2f}s[/dim]")
                
                # Show debug log location if debug was enabled
                if debug:
                    console.print(f"\n[yellow]🐛 Debug logs saved to: logs/debug/[/yellow]")
                    console.print("[dim]Check debug logs for error details[/dim]")
            
            # Restore original debug config if it was overridden
            if debug:
                try:
                    config_instance.logging.debug.enabled = original_debug_enabled
                    config_instance.logging.debug.log_errors_only = original_debug_errors_only
                except:
                    pass  # Ignore errors restoring config
            
        except Exception as e:
            console.print(f"[red]Error running benchmark: {str(e)}[/red]")
            logger.exception("Benchmark execution failed")
            sys.exit(1)
    
    asyncio.run(run_benchmark_async())
