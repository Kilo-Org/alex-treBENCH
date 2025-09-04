"""
Models List Command

List available models from OpenRouter.
"""

import asyncio
import click
from rich.console import Console
from rich.table import Table

from src.utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.command()
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
            from src.models.model_registry import model_registry
            from src.models.model_cache import get_model_cache
            
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
                input_cost_per_token = pricing.get('input_cost_per_1m_tokens', 0)
                output_cost_per_token = pricing.get('output_cost_per_1m_tokens', 0)
                
                # The cached values can be either per-token or per-million-tokens
                # Very small values (< 0.01) are per-token, larger values are per-million-tokens
                if input_cost_per_token > 0 and input_cost_per_token < 0.01:
                    # Values are per-token, convert to per-million-tokens
                    input_cost_per_million = input_cost_per_token * 1_000_000
                else:
                    # Values are already per-million-tokens
                    input_cost_per_million = input_cost_per_token
                    
                if output_cost_per_token > 0 and output_cost_per_token < 0.01:
                    # Values are per-token, convert to per-million-tokens
                    output_cost_per_million = output_cost_per_token * 1_000_000
                else:
                    # Values are already per-million-tokens
                    output_cost_per_million = output_cost_per_token
                
                # Format costs
                def format_list_cost(cost):
                    if cost == 0:
                        return "$0"
                    elif cost < 0.01:
                        return f"${cost:.4f}"
                    elif cost < 1:
                        return f"${cost:.2f}"
                    else:
                        return f"${cost:.0f}"
                
                cost_display = f"{format_list_cost(input_cost_per_million)}/{format_list_cost(output_cost_per_million)}"
                
                table.add_row(
                    (model.get('provider', 'Unknown')).title(),
                    model.get('id', 'N/A'),
                    model.get('name', 'N/A'),
                    f"{model.get('context_length', 0):,}",
                    cost_display,
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