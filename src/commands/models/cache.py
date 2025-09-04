"""
Models Cache Command

Manage model cache.
"""

import click
from rich.console import Console
from rich.table import Table

from src.utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.command()
@click.option('--clear', is_flag=True, help='Clear the model cache')
@click.option('--info', is_flag=True, help='Show detailed cache information', default=True)
@click.pass_context
def models_cache(ctx, clear, info):
    """Manage model cache."""
    try:
        from src.models.model_cache import get_model_cache
        
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