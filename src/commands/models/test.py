"""
Models Test Command

Test a specific model with a prompt.
"""

import asyncio
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.command()
@click.option('--model', '-m', required=True, help='Model ID to test')
@click.option('--prompt', '-p', default="What is the capital of France?", help='Test prompt')
@click.pass_context
def models_test(ctx, model, prompt):
    """Test a specific model with a prompt."""
    
    async def run_test():
        try:
            from src.models.model_registry import model_registry
            from src.models.openrouter import OpenRouterClient
            from src.models.base import ModelConfig
            
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