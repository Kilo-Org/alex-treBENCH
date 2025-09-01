"""
Health Command

System health and connectivity checks for alex-treBENCH.
"""

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from core.database import check_database_connection
from utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.command()
@click.option('--check-db', is_flag=True, help='Check database connection')
@click.option('--check-api', is_flag=True, help='Check API connections')
@click.option('--check-files', is_flag=True, help='Check file system and directories')
@click.option('--check-config', is_flag=True, help='Check configuration validity')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed information')
@click.pass_context
def health(ctx, check_db, check_api, check_files, check_config, verbose):
    """Check system health and connectivity.
    
    \b
    🏥 EXAMPLES:
    
    alex health
    alex health --check-db
    alex health --check-api
    alex health --verbose
    alex health --check-files --check-config
    
    \b
    💡 Performs comprehensive health checks including database connectivity,
    API availability, file system status, and configuration validation.
    """
    
    async def run_health_checks():
        config = ctx.obj.get('config')
        all_checks_passed = True
        
        # If no specific checks requested, run all
        run_all = not any([check_db, check_api, check_files, check_config])
        
        health_results = []
        
        # Database health check
        if check_db or run_all:
            console.print("[blue]Checking database connection...[/blue]")
            try:
                if check_database_connection():
                    health_results.append(("Database Connection", True, "Connected successfully"))
                    if verbose:
                        console.print("[green]✓ Database connection: OK[/green]")
                else:
                    health_results.append(("Database Connection", False, "Connection failed"))
                    console.print("[red]✗ Database connection: FAILED[/red]")
                    all_checks_passed = False
            except Exception as e:
                health_results.append(("Database Connection", False, f"Error: {str(e)}"))
                console.print(f"[red]✗ Database connection: ERROR - {str(e)}[/red]")
                logger.exception("Database health check failed")
                all_checks_passed = False
        
        # API health checks  
        if check_api or run_all:
            console.print("[blue]Checking API connections...[/blue]")
            
            # OpenRouter API check
            try:
                from models.openrouter import OpenRouterClient
                from models.base import ModelConfig
                
                model_config = ModelConfig(model_name="anthropic/claude-3-haiku")
                client = OpenRouterClient(config=model_config)
                
                # Test basic API connectivity
                is_healthy = await client.health_check()
                
                if is_healthy:
                    health_results.append(("OpenRouter API", True, "API accessible"))
                    if verbose:
                        console.print("[green]✓ OpenRouter API: OK[/green]")
                else:
                    health_results.append(("OpenRouter API", False, "API not accessible"))
                    console.print("[red]✗ OpenRouter API: FAILED[/red]")
                    all_checks_passed = False
                
                await client.close()
                
            except Exception as e:
                health_results.append(("OpenRouter API", False, f"Error: {str(e)}"))
                console.print(f"[red]✗ OpenRouter API: ERROR - {str(e)}[/red]")
                logger.exception("OpenRouter API health check failed")
                all_checks_passed = False
            
            # Model registry check
            try:
                from models.model_registry import model_registry
                
                models = await model_registry.get_available_models()
                if models and len(models) > 0:
                    health_results.append(("Model Registry", True, f"{len(models)} models available"))
                    if verbose:
                        console.print(f"[green]✓ Model Registry: {len(models)} models available[/green]")
                else:
                    health_results.append(("Model Registry", False, "No models available"))
                    console.print("[red]✗ Model Registry: No models available[/red]")
                    all_checks_passed = False
                    
            except Exception as e:
                health_results.append(("Model Registry", False, f"Error: {str(e)}"))
                console.print(f"[red]✗ Model Registry: ERROR - {str(e)}[/red]")
                logger.exception("Model registry health check failed")
                all_checks_passed = False
        
        # File system checks
        if check_files or run_all:
            console.print("[blue]Checking file system...[/blue]")
            
            # Check critical directories
            critical_dirs = [
                ("data", "data"),
                ("logs", "logs"),  
                ("config", "config"),
                ("cache", "data/cache")
            ]
            
            for name, path in critical_dirs:
                try:
                    dir_path = Path(path)
                    if dir_path.exists():
                        if dir_path.is_dir():
                            # Check if writable
                            test_file = dir_path / ".health_check"
                            try:
                                test_file.touch()
                                test_file.unlink()
                                health_results.append((f"{name} Directory", True, f"Exists and writable: {path}"))
                                if verbose:
                                    console.print(f"[green]✓ {name} directory: OK ({path})[/green]")
                            except Exception:
                                health_results.append((f"{name} Directory", False, f"Exists but not writable: {path}"))
                                console.print(f"[red]✗ {name} directory: Not writable ({path})[/red]")
                                all_checks_passed = False
                        else:
                            health_results.append((f"{name} Directory", False, f"Path exists but is not a directory: {path}"))
                            console.print(f"[red]✗ {name} directory: Not a directory ({path})[/red]")
                            all_checks_passed = False
                    else:
                        # Try to create the directory
                        try:
                            dir_path.mkdir(parents=True, exist_ok=True)
                            health_results.append((f"{name} Directory", True, f"Created: {path}"))
                            if verbose:
                                console.print(f"[green]✓ {name} directory: Created ({path})[/green]")
                        except Exception as e:
                            health_results.append((f"{name} Directory", False, f"Cannot create: {path} - {str(e)}"))
                            console.print(f"[red]✗ {name} directory: Cannot create ({path})[/red]")
                            all_checks_passed = False
                            
                except Exception as e:
                    health_results.append((f"{name} Directory", False, f"Error checking: {str(e)}"))
                    console.print(f"[red]✗ {name} directory: ERROR - {str(e)}[/red]")
                    all_checks_passed = False
        
        # Configuration checks
        if check_config or run_all:
            console.print("[blue]Checking configuration...[/blue]")
            
            try:
                if not config:
                    health_results.append(("Configuration", False, "Configuration not loaded"))
                    console.print("[red]✗ Configuration: Not loaded[/red]")
                    all_checks_passed = False
                else:
                    # Basic config validation
                    config_checks = []
                    
                    # Check required attributes
                    required_attrs = ['database', 'openrouter', 'benchmark']
                    for attr in required_attrs:
                        if hasattr(config, attr):
                            config_checks.append(f"{attr}: OK")
                        else:
                            config_checks.append(f"{attr}: Missing")
                            all_checks_passed = False
                    
                    # Check environment variables
                    import os
                    if 'OPENROUTER_API_KEY' in os.environ:
                        config_checks.append("OPENROUTER_API_KEY: Set")
                    else:
                        config_checks.append("OPENROUTER_API_KEY: Missing")
                        all_checks_passed = False
                    
                    status = "Valid" if all_checks_passed else "Issues detected"
                    health_results.append(("Configuration", all_checks_passed, status))
                    
                    if verbose:
                        for check in config_checks:
                            console.print(f"[dim]  {check}[/dim]")
                            
            except Exception as e:
                health_results.append(("Configuration", False, f"Error: {str(e)}"))
                console.print(f"[red]✗ Configuration: ERROR - {str(e)}[/red]")
                logger.exception("Configuration health check failed")
                all_checks_passed = False
        
        # Display summary table
        console.print()
        
        summary_table = Table(title="Health Check Results")
        summary_table.add_column("Component", style="cyan")
        summary_table.add_column("Status", justify="center")
        summary_table.add_column("Details", style="dim")
        
        for component, passed, details in health_results:
            status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
            summary_table.add_row(component, status, details)
        
        console.print(summary_table)
        
        # Overall status
        if all_checks_passed:
            console.print(Panel(
                "[green]✓ All health checks passed![/green]\n"
                "System is healthy and ready for benchmarking.",
                title="🏥 System Health",
                border_style="green"
            ))
        else:
            console.print(Panel(
                "[red]✗ Some health checks failed![/red]\n"
                "Please review and fix the issues above before running benchmarks.",
                title="🏥 System Health", 
                border_style="red"
            ))
            sys.exit(1)
    
    asyncio.run(run_health_checks())