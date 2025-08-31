#!/usr/bin/env python3
"""
System Validation Script

Comprehensive system health check and validation for the Jeopardy Benchmarking System.
Verifies all dependencies, database connectivity, configuration, and API access.
"""

import sys
import os
import importlib
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import json
import asyncio
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.config import get_config, AppConfig
from core.database import init_database, get_session, check_database_connection
from models.openrouter import OpenRouterClient
from models.base import ModelConfig
from benchmark.runner import BenchmarkRunner
from utils.logging import setup_logging


class SystemValidator:
    """Comprehensive system validation and health checking."""

    def __init__(self):
        """Initialize the system validator."""
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "checks": {},
            "summary": {},
            "recommendations": []
        }
        self.passed_checks = 0
        self.failed_checks = 0

    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information."""
        return {
            "python_version": sys.version,
            "platform": sys.platform,
            "working_directory": str(Path.cwd()),
            "user": os.getenv("USER", "unknown")
        }

    def log_check(self, name: str, status: bool, message: str,
                  details: Optional[Dict[str, Any]] = None):
        """Log a validation check result."""
        self.results["checks"][name] = {
            "status": "PASSED" if status else "FAILED",
            "message": message,
            "details": details or {}
        }

        if status:
            self.passed_checks += 1
            print(f"✅ {name}: {message}")
        else:
            self.failed_checks += 1
            print(f"❌ {name}: {message}")

    def add_recommendation(self, recommendation: str):
        """Add a recommendation for fixing issues."""
        self.results["recommendations"].append(recommendation)

    async def validate_python_environment(self) -> bool:
        """Validate Python environment and version."""
        try:
            version = sys.version_info
            version_str = f"{version.major}.{version.minor}.{version.micro}"

            if version.major >= 3 and version.minor >= 8:
                self.log_check(
                    "Python Version",
                    True,
                    f"Python {version_str} is compatible"
                )
                return True
            else:
                self.log_check(
                    "Python Version",
                    False,
                    f"Python {version_str} is not supported. Requires Python 3.8+"
                )
                self.add_recommendation("Upgrade to Python 3.8 or higher")
                return False

        except Exception as e:
            self.log_check("Python Version", False, f"Failed to check Python version: {e}")
            return False

    def validate_dependencies(self) -> bool:
        """Validate that all required dependencies are installed."""
        required_packages = [
            "click", "rich", "sqlalchemy", "pandas", "pytest",
            "httpx", "pydantic", "pyyaml", "numpy", "scipy"
        ]

        optional_packages = [
            "matplotlib", "seaborn", "plotly"
        ]

        missing_required = []
        missing_optional = []

        for package in required_packages:
            try:
                importlib.import_module(package.replace("-", "_"))
            except ImportError:
                missing_required.append(package)

        for package in optional_packages:
            try:
                importlib.import_module(package.replace("-", "_"))
            except ImportError:
                missing_optional.append(package)

        if missing_required:
            self.log_check(
                "Required Dependencies",
                False,
                f"Missing required packages: {', '.join(missing_required)}"
            )
            self.add_recommendation(f"Install missing packages: pip install {' '.join(missing_required)}")
            return False
        else:
            self.log_check(
                "Required Dependencies",
                True,
                "All required dependencies are installed"
            )

        if missing_optional:
            self.log_check(
                "Optional Dependencies",
                False,
                f"Missing optional packages: {', '.join(missing_optional)}",
                {"missing": missing_optional}
            )
            self.add_recommendation(f"Consider installing optional packages: pip install {' '.join(missing_optional)}")

        return len(missing_required) == 0

    def validate_project_structure(self) -> bool:
        """Validate that the project structure is correct."""
        required_dirs = ["src", "tests", "config", "docs"]
        required_files = ["setup.py", "requirements.txt", "README.md"]

        missing_dirs = []
        missing_files = []

        for dir_name in required_dirs:
            if not Path(dir_name).exists():
                missing_dirs.append(dir_name)

        for file_name in required_files:
            if not Path(file_name).exists():
                missing_files.append(file_name)

        if missing_dirs or missing_files:
            details = {}
            if missing_dirs:
                details["missing_directories"] = missing_dirs
            if missing_files:
                details["missing_files"] = missing_files

            self.log_check(
                "Project Structure",
                False,
                "Project structure is incomplete",
                details
            )
            return False
        else:
            self.log_check(
                "Project Structure",
                True,
                "Project structure is complete"
            )
            return True

    def validate_configuration(self) -> bool:
        """Validate configuration files and settings."""
        try:
            # Try to load configuration
            config = get_config()

            # Check required configuration sections
            required_sections = ["database", "logging", "benchmark"]
            missing_sections = []

            for section in required_sections:
                if not hasattr(config, section):
                    missing_sections.append(section)

            if missing_sections:
                self.log_check(
                    "Configuration",
                    False,
                    f"Missing configuration sections: {', '.join(missing_sections)}"
                )
                return False

            # Validate database URL
            if not config.database.url:
                self.log_check(
                    "Configuration",
                    False,
                    "Database URL is not configured"
                )
                self.add_recommendation("Set DATABASE_URL in configuration or environment variables")
                return False

            self.log_check(
                "Configuration",
                True,
                "Configuration is valid",
                {
                    "database_url": config.database.url,
                    "debug_mode": config.debug,
                    "sample_size": config.benchmark.default_sample_size
                }
            )
            return True

        except Exception as e:
            self.log_check(
                "Configuration",
                False,
                f"Configuration validation failed: {e}"
            )
            self.add_recommendation("Check configuration files in config/ directory")
            return False

    async def validate_database(self) -> bool:
        """Validate database connectivity and setup."""
        try:
            # Initialize database
            config = get_config()
            init_database(config)

            # Test connection
            if check_database_connection():
                self.log_check(
                    "Database Connection",
                    True,
                    "Database connection successful"
                )

                # Check if tables exist by trying to create a session
                try:
                    with get_session() as session:
                        self.log_check(
                            "Database Tables",
                            True,
                            "Database tables are accessible"
                        )
                    return True
                except Exception as e:
                    self.log_check(
                        "Database Tables",
                        False,
                        f"Database tables issue: {e}"
                    )
                    self.add_recommendation("Run database initialization: python -m src.main init")
                    return False
            else:
                self.log_check(
                    "Database Connection",
                    False,
                    "Database connection failed"
                )
                self.add_recommendation("Check database URL and ensure database server is running")
                return False

        except Exception as e:
            self.log_check(
                "Database",
                False,
                f"Database validation failed: {e}"
            )
            self.add_recommendation("Check database configuration and connectivity")
            return False

    async def validate_api_connectivity(self) -> bool:
        """Validate API connectivity (OpenRouter)."""
        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            self.log_check(
                "API Key",
                False,
                "OPENROUTER_API_KEY environment variable not set"
            )
            self.add_recommendation("Set OPENROUTER_API_KEY environment variable or create .env file")
            return False

        try:
            # Test API connectivity with a simple model
            model_config = ModelConfig(
                model_name="openai/gpt-3.5-turbo",
                max_tokens=10,
                timeout_seconds=10
            )

            async with OpenRouterClient(model_config) as client:
                # Try a simple health check
                health = await client.health_check()

                if health.get("status") == "healthy":
                    self.log_check(
                        "API Connectivity",
                        True,
                        "OpenRouter API connection successful",
                        {
                            "response_time_ms": health.get("response_time_ms"),
                            "model": health.get("model_name")
                        }
                    )
                    return True
                else:
                    self.log_check(
                        "API Connectivity",
                        False,
                        f"API health check failed: {health.get('error', 'Unknown error')}"
                    )
                    return False

        except Exception as e:
            self.log_check(
                "API Connectivity",
                False,
                f"API connectivity test failed: {e}"
            )
            self.add_recommendation("Check API key validity and network connectivity")
            return False

    async def validate_benchmark_system(self) -> bool:
        """Validate that the benchmark system can run."""
        try:
            runner = BenchmarkRunner()

            # Test configuration loading
            config = runner.get_default_config("quick")

            if config and config.sample_size > 0:
                self.log_check(
                    "Benchmark System",
                    True,
                    "Benchmark system initialized successfully",
                    {
                        "default_sample_size": config.sample_size,
                        "timeout_seconds": config.timeout_seconds
                    }
                )
                return True
            else:
                self.log_check(
                    "Benchmark System",
                    False,
                    "Benchmark system configuration is invalid"
                )
                return False

        except Exception as e:
            self.log_check(
                "Benchmark System",
                False,
                f"Benchmark system validation failed: {e}"
            )
            return False

    def validate_file_permissions(self) -> bool:
        """Validate file and directory permissions."""
        paths_to_check = [
            "config/",
            "data/",
            "logs/",
            "src/",
            "tests/"
        ]

        permission_issues = []

        for path_str in paths_to_check:
            path = Path(path_str)
            if path.exists():
                if not os.access(path, os.R_OK):
                    permission_issues.append(f"No read access: {path_str}")
                if path.is_dir() and not os.access(path, os.W_OK):
                    permission_issues.append(f"No write access: {path_str}")
            else:
                # Try to create directory to test permissions
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except PermissionError:
                    permission_issues.append(f"Cannot create: {path_str}")

        if permission_issues:
            self.log_check(
                "File Permissions",
                False,
                "File permission issues detected",
                {"issues": permission_issues}
            )
            self.add_recommendation("Fix file permissions for the listed directories")
            return False
        else:
            self.log_check(
                "File Permissions",
                True,
                "File permissions are correct"
            )
            return True

    def check_disk_space(self) -> bool:
        """Check available disk space."""
        try:
            stat = os.statvfs('.')
            # Get available space in GB
            available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)

            if available_gb < 1.0:
                self.log_check(
                    "Disk Space",
                    False,
                    ".1f"
                )
                self.add_recommendation("Free up disk space (at least 1GB recommended)")
                return False
            else:
                self.log_check(
                    "Disk Space",
                    True,
                    ".1f"
                )
                return True

        except Exception as e:
            self.log_check(
                "Disk Space",
                False,
                f"Could not check disk space: {e}"
            )
            return False

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive validation report."""
        total_checks = self.passed_checks + self.failed_checks

        self.results["summary"] = {
            "total_checks": total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "success_rate": (self.passed_checks / total_checks * 100) if total_checks > 0 else 0,
            "overall_status": "HEALTHY" if self.failed_checks == 0 else "ISSUES_FOUND"
        }

        return self.results

    def save_report(self, filename: str = "system_validation_report.json"):
        """Save validation report to file."""
        report = self.generate_report()

        with open(filename, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 Validation report saved to: {filename}")

    def print_summary(self):
        """Print a summary of the validation results."""
        report = self.generate_report()

        print("\n" + "="*60)
        print("📊 SYSTEM VALIDATION SUMMARY")
        print("="*60)

        summary = report["summary"]
        print(f"Total Checks: {summary['total_checks']}")
        print(f"Passed: {summary['passed_checks']}")
        print(f"Failed: {summary['failed_checks']}")
        print(".1f"
        if summary["overall_status"] == "HEALTHY":
            print("🎉 Overall Status: HEALTHY")
        else:
            print("⚠️  Overall Status: ISSUES FOUND")

        if report["recommendations"]:
            print("
💡 Recommendations:"            for rec in report["recommendations"]:
                print(f"  • {rec}")

        print("\n" + "="*60)


async def main():
    """Main validation function."""
    print("🔍 Jeopardy Benchmarking System - Validation")
    print("=" * 50)

    validator = SystemValidator()

    # Run all validation checks
    checks = [
        ("Python Environment", validator.validate_python_environment()),
        ("Dependencies", validator.validate_dependencies()),
        ("Project Structure", validator.validate_project_structure()),
        ("Configuration", validator.validate_configuration()),
        ("Database", validator.validate_database()),
        ("API Connectivity", validator.validate_api_connectivity()),
        ("Benchmark System", validator.validate_benchmark_system()),
        ("File Permissions", validator.validate_file_permissions()),
        ("Disk Space", validator.check_disk_space())
    ]

    print("\nRunning validation checks...")
    print("-" * 30)

    for check_name, check_coro in checks:
        try:
            if asyncio.iscoroutine(check_coro):
                result = await check_coro
            else:
                result = check_coro
        except Exception as e:
            print(f"❌ {check_name}: Exception during validation - {e}")
            validator.failed_checks += 1

    # Generate and display summary
    validator.print_summary()

    # Save detailed report
    validator.save_report()

    # Exit with appropriate code
    if validator.failed_checks > 0:
        print("
❌ System validation found issues. Please address the recommendations above."        sys.exit(1)
    else:
        print("
✅ System validation passed! The Jeopardy Benchmarking System is ready to use."        sys.exit(0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Validation interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error during validation: {e}")
        sys.exit(1)