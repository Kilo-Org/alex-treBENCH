"""
Models Search Command

Search for models by name, provider, or capabilities.
"""

import asyncio
import click
from rich.console import Console
from rich.table import Table

from src.utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.command()
@click.argument('query', required=True)
@click.option('--limit', '-l', type=int, default=20, help='Maximum number of results to show')
@click.pass_context
def models_search(ctx, query, limit):
    """Search for models by name, provider, or capabilities."""
    
    async def search_models_async():
        try:
            from src.models.model_registry import model_registry
            
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
                input_cost_per_token = pricing.get('input_cost_per_1m_tokens', 0)
                output_cost_per_token = pricing.get('output_cost_per_1m_tokens', 0)
                
                # Convert per-token costs to per-million-tokens for display
                # Very small values (< 0.01) are per-token, larger values are per-million-tokens
                if input_cost_per_token > 0 and input_cost_per_token < 0.01:
                    input_cost = input_cost_per_token * 1_000_000
                else:
                    input_cost = input_cost_per_token
                    
                if output_cost_per_token > 0 and output_cost_per_token < 0.01:
                    output_cost = output_cost_per_token * 1_000_000
                else:
                    output_cost = output_cost_per_token
                
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