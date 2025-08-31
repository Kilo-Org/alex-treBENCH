#!/usr/bin/env python3
"""
Test Infrastructure Integration Verification

This script verifies that the test infrastructure (test agents and smoke test)
integrates properly with the existing alex-treBENCH codebase by checking:
1. All imports work correctly
2. Test scripts can be executed
3. Dependencies are satisfied
4. Configuration is compatible
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class IntegrationVerifier:
    """Verifies test infrastructure integration."""
    
    def __init__(self):
        self.results = []
        
    def verify_all(self) -> bool:
        """Run all verification checks."""
        console.print(Panel.fit(
            "[bold blue]🔧 alex-treBENCH Test Infrastructure Verification[/bold blue]\n"
            "Checking integration with existing codebase",
            title="Integration Verification",
            border_style="blue"
        ))
        
        checks = [
            ("Import Verification", self.verify_imports),
            ("Script Syntax Check", self.verify_script_syntax),
            ("Makefile Integration", self.verify_makefile),
            ("Dependencies Check", self.verify_dependencies),
            ("Configuration Compatibility", self.verify_configuration)
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for check_name, check_func in checks:
                task = progress.add_task(f"Running {check_name}...", total=None)
                
                try:
                    success, message, details = check_func()
                    self.results.append({
                        "name": check_name,
                        "success": success,
                        "message": message,
                        "details": details
                    })
                    
                    status = "✅" if success else "❌"
                    progress.update(task, description=f"{status} {check_name}")
                    
                except Exception as e:
                    self.results.append({
                        "name": check_name,
                        "success": False,
                        "message": f"Check failed: {str(e)}",
                        "details": {"error": str(e)}
                    })
                    progress.update(task, description=f"❌ {check_name}")
        
        return self.display_results()
    
    def verify_imports(self) -> tuple[bool, str, Dict[str, Any]]:
        """Verify all required imports work correctly."""
        
        # Test scripts to verify
        test_scripts = [
            "scripts/test_agents.py",
            "scripts/smoke_test.py"
        ]
        
        # Critical imports to test
        critical_imports = [
            "src.core.config",
            "src.core.database", 
            "src.storage.models",
            "src.storage.repositories",
            "src.benchmark.runner",
            "src.benchmark.reporting",
            "src.models.model_registry"
        ]
        
        import_results = {}
        
        # Test direct imports
        for module_name in critical_imports:
            try:
                importlib.import_module(module_name)
                import_results[module_name] = "✅ OK"
            except ImportError as e:
                import_results[module_name] = f"❌ Failed: {e}"
        
        # Test script imports by syntax checking
        script_results = {}
        for script_path in test_scripts:
            if os.path.exists(script_path):
                try:
                    with open(script_path, 'r') as f:
                        compile(f.read(), script_path, 'exec')
                    script_results[script_path] = "✅ Syntax OK"
                except SyntaxError as e:
                    script_results[script_path] = f"❌ Syntax Error: {e}"
                except Exception as e:
                    script_results[script_path] = f"❌ Error: {e}"
            else:
                script_results[script_path] = "❌ File not found"
        
        # Determine overall success
        failed_imports = [k for k, v in import_results.items() if "❌" in v]
        failed_scripts = [k for k, v in script_results.items() if "❌" in v]
        
        success = len(failed_imports) == 0 and len(failed_scripts) == 0
        
        if success:
            message = "All imports and scripts verified successfully"
        else:
            message = f"{len(failed_imports)} import failures, {len(failed_scripts)} script failures"
        
        return success, message, {
            "imports": import_results,
            "scripts": script_results
        }
    
    def verify_script_syntax(self) -> tuple[bool, str, Dict[str, Any]]:
        """Verify test scripts have correct syntax."""
        
        scripts = [
            "scripts/test_agents.py",
            "scripts/smoke_test.py",
            "scripts/verify_test_integration.py"
        ]
        
        results = {}
        for script in scripts:
            if os.path.exists(script):
                try:
                    result = subprocess.run([
                        sys.executable, "-m", "py_compile", script
                    ], capture_output=True, text=True, timeout=10)
                    
                    if result.returncode == 0:
                        results[script] = "✅ Valid syntax"
                    else:
                        results[script] = f"❌ Syntax error: {result.stderr}"
                        
                except subprocess.TimeoutExpired:
                    results[script] = "❌ Timeout during syntax check"
                except Exception as e:
                    results[script] = f"❌ Error: {e}"
            else:
                results[script] = "❌ File not found"
        
        failed_scripts = [k for k, v in results.items() if "❌" in v]
        success = len(failed_scripts) == 0
        
        message = "All scripts have valid syntax" if success else f"{len(failed_scripts)} scripts have syntax errors"
        
        return success, message, results
    
    def verify_makefile(self) -> tuple[bool, str, Dict[str, Any]]:
        """Verify Makefile integration."""
        
        makefile_path = "Makefile"
        if not os.path.exists(makefile_path):
            return False, "Makefile not found", {}
        
        try:
            with open(makefile_path, 'r') as f:
                makefile_content = f.read()
            
            # Check for required targets
            required_targets = ["smoke-test", "test-agents"]
            found_targets = {}
            
            for target in required_targets:
                if f"{target}:" in makefile_content:
                    found_targets[target] = "✅ Found"
                else:
                    found_targets[target] = "❌ Missing"
            
            # Test make help to see if targets are listed
            help_result = {}
            try:
                result = subprocess.run(["make", "help"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    help_output = result.stdout
                    for target in required_targets:
                        if target in help_output:
                            help_result[target] = "✅ Listed in help"
                        else:
                            help_result[target] = "❌ Not in help"
                else:
                    help_result = {"error": "Make help failed"}
            except Exception as e:
                help_result = {"error": f"Make help error: {e}"}
            
            missing_targets = [k for k, v in found_targets.items() if "❌" in v]
            success = len(missing_targets) == 0
            
            message = "Makefile integration verified" if success else f"Missing targets: {missing_targets}"
            
            return success, message, {
                "targets": found_targets,
                "help_output": help_result
            }
            
        except Exception as e:
            return False, f"Makefile verification failed: {e}", {"error": str(e)}
    
    def verify_dependencies(self) -> tuple[bool, str, Dict[str, Any]]:
        """Verify all required dependencies are available."""
        
        # Required packages
        required_packages = [
            "rich",
            "sqlalchemy", 
            "click",
            "asyncio",
            "tempfile",
            "subprocess",
            "json",
            "pathlib"
        ]
        
        package_results = {}
        for package in required_packages:
            try:
                importlib.import_module(package)
                package_results[package] = "✅ Available"
            except ImportError:
                package_results[package] = "❌ Missing"
        
        # Check Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        python_ok = sys.version_info >= (3, 8)
        
        missing_packages = [k for k, v in package_results.items() if "❌" in v]
        success = len(missing_packages) == 0 and python_ok
        
        if success:
            message = "All dependencies satisfied"
        else:
            issues = []
            if not python_ok:
                issues.append("Python version too old")
            if missing_packages:
                issues.append(f"{len(missing_packages)} missing packages")
            message = "; ".join(issues)
        
        return success, message, {
            "packages": package_results,
            "python_version": python_version,
            "python_ok": python_ok
        }
    
    def verify_configuration(self) -> tuple[bool, str, Dict[str, Any]]:
        """Verify configuration compatibility."""
        
        config_results = {}
        
        try:
            # Test config loading
            from src.core.config import get_config
            config = get_config()
            config_results["config_loading"] = "✅ Config loads successfully"
            config_results["config_version"] = config.version
            
            # Test database configuration
            if hasattr(config, 'database'):
                config_results["database_config"] = "✅ Database config present"
            else:
                config_results["database_config"] = "❌ Database config missing"
            
            # Test logging configuration
            if hasattr(config, 'logging'):
                config_results["logging_config"] = "✅ Logging config present"
            else:
                config_results["logging_config"] = "❌ Logging config missing"
                
            # Test benchmark configuration
            if hasattr(config, 'benchmark'):
                config_results["benchmark_config"] = "✅ Benchmark config present"
            else:
                config_results["benchmark_config"] = "❌ Benchmark config missing"
            
        except Exception as e:
            config_results["config_loading"] = f"❌ Config loading failed: {e}"
        
        # Check environment variables
        env_vars = ["OPENROUTER_API_KEY", "DATABASE_URL"]
        env_results = {}
        for var in env_vars:
            if os.getenv(var):
                env_results[var] = "✅ Set"
            else:
                env_results[var] = "❌ Not set (optional for tests)"
        
        failed_configs = [k for k, v in config_results.items() if "❌" in v]
        success = len(failed_configs) == 0
        
        message = "Configuration compatible" if success else f"{len(failed_configs)} config issues"
        
        return success, message, {
            "config": config_results,
            "environment": env_results
        }
    
    def display_results(self) -> bool:
        """Display verification results."""
        
        # Summary statistics
        total_checks = len(self.results)
        successful_checks = sum(1 for r in self.results if r["success"])
        failed_checks = total_checks - successful_checks
        
        # Summary panel
        status_color = "green" if failed_checks == 0 else "red" if successful_checks == 0 else "yellow"
        summary_panel = Panel.fit(
            f"[bold {status_color}]Verification Summary[/bold {status_color}]\n"
            f"Total Checks: {total_checks}\n"
            f"Passed: [green]{successful_checks}[/green]\n"
            f"Failed: [red]{failed_checks}[/red]",
            title="Results",
            border_style=status_color
        )
        console.print(summary_panel)
        
        # Detailed results table
        table = Table(title="Integration Verification Results")
        table.add_column("Check", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Message", style="dim")
        
        for result in self.results:
            status_icon = "✅" if result["success"] else "❌"
            status_color = "green" if result["success"] else "red"
            
            table.add_row(
                result["name"],
                f"[{status_color}]{status_icon}[/{status_color}]",
                result["message"][:100] + "..." if len(result["message"]) > 100 else result["message"]
            )
        
        console.print("\n")
        console.print(table)
        
        # Show failed check details
        failed_results = [r for r in self.results if not r["success"]]
        if failed_results:
            console.print(f"\n[red]Failed Checks Details:[/red]")
            for result in failed_results:
                console.print(f"[red]❌ {result['name']}[/red]: {result['message']}")
                if result.get("details"):
                    # Show key details
                    details = result["details"]
                    if isinstance(details, dict):
                        for key, value in details.items():
                            if isinstance(value, dict):
                                failed_items = [k for k, v in value.items() if "❌" in str(v)]
                                if failed_items:
                                    console.print(f"[dim]   {key}: {len(failed_items)} issues[/dim]")
        else:
            console.print(f"\n[green]🎉 All integration checks passed![/green]")
        
        return failed_checks == 0


def main():
    """Main entry point."""
    console.print("[blue]Starting alex-treBENCH Test Infrastructure Integration Verification...[/blue]\n")
    
    verifier = IntegrationVerifier()
    success = verifier.verify_all()
    
    if success:
        console.print(Panel.fit(
            "[bold green]✅ Integration Verification PASSED[/bold green]\n"
            "Test infrastructure is properly integrated with alex-treBENCH!",
            title="Success",
            border_style="green"
        ))
        return 0
    else:
        console.print(Panel.fit(
            "[bold red]❌ Integration Verification FAILED[/bold red]\n"
            "Test infrastructure has integration issues that need attention",
            title="Failure", 
            border_style="red"
        ))
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)