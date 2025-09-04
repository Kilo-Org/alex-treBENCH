"""
Models Refresh Command

Force refresh the model cache from OpenRouter API.
"""

import asyncio
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.command()
@click.pass_context  
def models_refresh(ctx):
    """Force refresh the model cache from OpenRouter API."""
    
    async def refresh_models_async():
        try:
            from src.models.model_registry import model_registry
            
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