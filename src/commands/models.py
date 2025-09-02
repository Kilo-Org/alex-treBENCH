"""
Models Command Group

Model management commands for alex-treBENCH.
"""

import asyncio
import sys
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.group()
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
                input_cost_per_token = pricing.get('input_cost_per_1m_tokens', 0)
                output_cost_per_token = pricing.get('output_cost_per_1m_tokens', 0)
                
                # Convert from per-token to per-million-tokens for display
                input_cost_per_million = input_cost_per_token * 1_000_000
                output_cost_per_million = output_cost_per_token * 1_000_000
                
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
                    # OpenRouter API returns cost per token, so multiply by 1M to get cost per 1M tokens
                    price_per_million = cost * 1_000_000
                    if cost == 0:
                        return "$0"
                    elif price_per_million < 0.01:
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