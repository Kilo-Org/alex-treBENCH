"""
Models Info Command

Show detailed information about a specific model.
"""

import asyncio
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.command()
@click.argument('model_id', required=True)
@click.pass_context
def models_info(ctx, model_id):
    """Show detailed information about a specific model."""
    
    async def show_model_info_async():
        try:
            from src.models.model_registry import model_registry
            
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
                    if cost == 0:
                        return "$0"
                    # Check if values are already per-million-tokens (larger values) or per-token (very small values)
                    if cost < 0.01:
                        # Values are per-token, convert to per-million-tokens
                        price_per_million = cost * 1_000_000
                    else:
                        # Values are already per-million-tokens
                        price_per_million = cost
                    
                    if price_per_million < 0.01:
                        # For very small values, show more decimal places
                        return f"${price_per_million:.4f}"
                    elif price_per_million < 1:
                        return f"${price_per_million:.2f}"
                    else:
                        return f"${price_per_million:.0f}"
                
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
                console.print(f"\n[bold]Request Limits:[/bold]");
                if 'prompt_tokens' in limits:
                    console.print(f"• Max prompt tokens: {limits['prompt_tokens']:,}")
                if 'completion_tokens' in limits:
                    console.print(f"• Max completion tokens: {limits['completion_tokens']:,}")
            
        except Exception as e:
            console.print(f"[red]Error getting model info: {str(e)}[/red]")
            logger.exception("Model info retrieval failed")
    
    asyncio.run(show_model_info_async())