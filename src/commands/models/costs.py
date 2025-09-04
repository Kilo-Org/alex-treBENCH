"""
Models Costs Command

Estimate costs for running benchmarks with a model.
"""

import asyncio
import click
from rich.console import Console
from rich.table import Table

from src.utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.command()
@click.option('--model', '-m', required=True, help='Model ID to estimate costs for')
@click.option('--questions', '-q', type=int, default=100, help='Number of questions')
@click.option('--input-tokens', type=int, help='Average input tokens per question')
@click.option('--output-tokens', type=int, help='Average output tokens per question')
@click.pass_context
def models_costs(ctx, model, questions, input_tokens, output_tokens):
    """Estimate costs for running benchmarks with a model."""
    
    async def calculate_costs_async():
        try:
            from src.models.model_registry import model_registry
            from src.models.cost_calculator import CostCalculator
            
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
            
            config = ctx.obj.get('config') if ctx.obj else None
            if config and hasattr(config, 'costs') and hasattr(config.costs, 'estimation'):
                try:
                    default_input_tokens = getattr(config.costs.estimation, 'default_input_tokens_per_question', 100)
                    default_output_tokens = getattr(config.costs.estimation, 'default_output_tokens_per_question', 50)
                except AttributeError:
                    pass  # Use defaults
            
            # Apply the values - use different variable names to avoid shadowing
            actual_input_tokens = input_tokens if input_tokens is not None else default_input_tokens
            actual_output_tokens = output_tokens if output_tokens is not None else default_output_tokens
            
            # Calculate costs using the proper ModelRegistry method
            total_input_tokens = questions * actual_input_tokens
            total_output_tokens = questions * actual_output_tokens
            total_tokens = total_input_tokens + total_output_tokens
            
            # Use ModelRegistry.estimate_cost for proper cost calculation
            from src.models.model_registry import ModelRegistry
            total_cost = ModelRegistry.estimate_cost(model, total_input_tokens, total_output_tokens)
            input_cost = ModelRegistry.estimate_cost(model, total_input_tokens, 0)
            output_cost = ModelRegistry.estimate_cost(model, 0, total_output_tokens)
            cost_per_question = total_cost / questions if questions > 0 else 0
            
            # Get pricing information for display purposes
            pricing = model_info.get('pricing', {})
            input_cost_per_1m = pricing.get('input_cost_per_1m_tokens', 0)
            output_cost_per_1m = pricing.get('output_cost_per_1m_tokens', 0)
            
            # If not found in dynamic model info, try static config
            if input_cost_per_1m == 0 and output_cost_per_1m == 0:
                static_config = ModelRegistry.get_model_config(model)
                if static_config:
                    input_cost_per_1m = static_config.input_cost_per_1m_tokens
                    output_cost_per_1m = static_config.output_cost_per_1m_tokens
            
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