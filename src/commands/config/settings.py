"""
Configuration Settings Commands

This module contains the configuration management command implementation.
"""

import json
import yaml
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.command()
@click.option('--format', type=click.Choice(['table', 'json', 'yaml']), default='table', help='Output format')
@click.pass_context
def show(ctx, format):
    """Show current configuration.
    
    \b
    📋 EXAMPLES:
    
    alex config show
    alex config show --format json
    alex config show --format yaml
    
    \b
    💡 Displays the current application configuration including database settings,
    benchmark parameters, and other system settings.
    """
    
    try:
        config = ctx.obj.get('config')
        if not config:
            console.print("[red]Configuration not available[/red]")
            return
        
        if format == 'table':
            table = Table(title="Configuration")
            table.add_column("Section", style="cyan")
            table.add_column("Setting", style="magenta")
            table.add_column("Value", style="green")
            
            # App settings
            table.add_row("app", "name", getattr(config, 'name', 'alex-trebench'))
            table.add_row("app", "version", getattr(config, 'version', '1.0.0'))
            table.add_row("app", "debug", str(getattr(config, 'debug', False)))
            
            # Database settings
            if hasattr(config, 'database'):
                table.add_row("database", "url", getattr(config.database, 'url', 'N/A'))
                table.add_row("database", "echo", str(getattr(config.database, 'echo', False)))
            
            # Benchmark settings
            if hasattr(config, 'benchmark'):
                table.add_row("benchmark", "default_sample_size", str(getattr(config.benchmark, 'default_sample_size', 50)))
                table.add_row("benchmark", "max_concurrent_requests", str(getattr(config.benchmark, 'max_concurrent_requests', 5)))
            
            # OpenRouter settings
            if hasattr(config, 'openrouter'):
                table.add_row("openrouter", "api_url", getattr(config.openrouter, 'api_url', 'N/A'))
                table.add_row("openrouter", "rate_limit", str(getattr(config.openrouter, 'rate_limit', 60)))
            
            # Logging settings
            if hasattr(config, 'logging'):
                table.add_row("logging", "level", getattr(config.logging, 'level', 'INFO'))
                table.add_row("logging", "format", getattr(config.logging, 'format', 'standard'))
            
            console.print(table)
            
        elif format == 'json':
            config_dict = _config_to_dict(config)
            console.print_json(json.dumps(config_dict, indent=2))
            
        elif format == 'yaml':
            config_dict = _config_to_dict(config)
            yaml_output = yaml.dump(config_dict, default_flow_style=False, indent=2)
            console.print(yaml_output)
            
    except Exception as e:
        console.print(f"[red]Error displaying configuration: {str(e)}[/red]")
        logger.exception("Configuration display failed")


@click.command()
@click.pass_context
def validate(ctx):
    """Validate current configuration.
    
    \b
    🔍 EXAMPLES:
    
    alex config validate
    
    \b
    💡 Validates the current configuration for completeness and correctness.
    Checks for required settings, valid values, and potential issues.
    """
    
    try:
        from core.config_validator import ConfigValidator
        
        config = ctx.obj.get('config')
        if not config:
            console.print("[red]Configuration not available for validation[/red]")
            return
        
        console.print("[blue]Validating configuration...[/blue]")
        
        validator = ConfigValidator()
        validation_results = validator.validate(config)
        
        # Display validation results
        results_table = Table(title="Configuration Validation")
        results_table.add_column("Check", style="cyan")
        results_table.add_column("Status", justify="center")
        results_table.add_column("Message", style="dim")
        
        all_passed = True
        
        for check_name, result in validation_results.items():
            status = "✓ PASS" if result['valid'] else "✗ FAIL"
            status_color = "green" if result['valid'] else "red"
            
            if not result['valid']:
                all_passed = False
            
            results_table.add_row(
                check_name.replace('_', ' ').title(),
                f"[{status_color}]{status}[/{status_color}]",
                result.get('message', '')
            )
        
        console.print(results_table)
        
        # Overall status
        if all_passed:
            console.print(Panel(
                "[green]✓ Configuration validation passed![/green]\n"
                "All required settings are present and valid.",
                title="🔍 Validation Results",
                border_style="green"
            ))
        else:
            console.print(Panel(
                "[red]✗ Configuration validation failed![/red]\n"
                "Please review and fix the issues above.",
                title="🔍 Validation Results", 
                border_style="red"
            ))
            
    except Exception as e:
        console.print(f"[red]Error validating configuration: {str(e)}[/red]")
        logger.exception("Configuration validation failed")


@click.command()
@click.option('--format', type=click.Choice(['json', 'yaml']), default='yaml', help='Export format')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.pass_context  
def export(ctx, format, output):
    """Export current configuration to file.
    
    \b
    📤 EXAMPLES:
    
    alex config export
    alex config export --format json --output config.json
    alex config export --output my_config.yaml
    
    \b
    💡 Exports the current configuration to a file for backup or sharing.
    """
    
    try:
        config = ctx.obj.get('config')
        if not config:
            console.print("[red]Configuration not available for export[/red]")
            return
        
        # Convert config to dictionary
        config_dict = _config_to_dict(config)
        
        # Determine output path
        if output:
            output_path = Path(output)
        else:
            timestamp = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
            extension = 'json' if format == 'json' else 'yaml'
            output_path = Path(f"config_export_{timestamp}.{extension}")
        
        # Export configuration
        with open(output_path, 'w') as f:
            if format == 'json':
                json.dump(config_dict, f, indent=2)
            else:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        console.print(f"[green]✓ Configuration exported to {output_path}[/green]")
        console.print(f"[dim]Format: {format.upper()}[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error exporting configuration: {str(e)}[/red]")
        logger.exception("Configuration export failed")


def _config_to_dict(config):
    """Convert configuration object to dictionary."""
    config_dict = {}
    
    for attr_name in dir(config):
        if not attr_name.startswith('_'):
            attr_value = getattr(config, attr_name)
            
            # Skip methods and functions
            if callable(attr_value):
                continue
            
            # Handle nested configuration objects
            if hasattr(attr_value, '__dict__') and not isinstance(attr_value, (str, int, float, bool, list, dict)):
                nested_dict = {}
                for nested_attr in dir(attr_value):
                    if not nested_attr.startswith('_'):
                        nested_value = getattr(attr_value, nested_attr)
                        if not callable(nested_value):
                            nested_dict[nested_attr] = nested_value
                config_dict[attr_name] = nested_dict
            else:
                config_dict[attr_name] = attr_value
    
    return config_dict