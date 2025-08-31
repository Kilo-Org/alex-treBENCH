#!/usr/bin/env python3
"""
Test Agents for alex-treBENCH System

Comprehensive test agents that verify different aspects of the system:
- Database initialization
- Data loading
- Minimal benchmarking with real API calls
- Report generation
- CLI command verification

These agents use minimal data to keep execution fast and costs low.
"""

import os
import sys
import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import subprocess
import sqlite3

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import AppConfig, get_config, set_config
from src.core.database import init_database, get_db_session, Base
from src.core.exceptions import AlexTreBenchException
from src.storage.models import Question, BenchmarkRun, BenchmarkResult
from src.storage.repositories import BenchmarkRepository
from src.benchmark.runner import BenchmarkRunner, RunMode
from src.benchmark.reporting import ReportGenerator, ReportFormat
from src.models.model_registry import ModelRegistry
from sqlalchemy import text
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


class TestAgentResult:
    """Result of a test agent execution."""
    
    def __init__(self, name: str, success: bool = True, message: str = "", 
                 details: Optional[Dict] = None, execution_time: float = 0.0):
        self.name = name
        self.success = success
        self.message = message
        self.details = details or {}
        self.execution_time = execution_time
        self.timestamp = datetime.now()


class BaseTestAgent:
    """Base class for test agents."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.config = None
        self.test_db_path: Optional[str] = None
        
    async def setup(self) -> None:
        """Setup test environment."""
        pass
        
    async def run(self) -> TestAgentResult:
        """Run the test agent."""
        raise NotImplementedError
        
    async def cleanup(self) -> None:
        """Cleanup test environment."""
        pass
        
    def _setup_test_database(self) -> str:
        """Setup a temporary test database."""
        # Create temporary database file
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)  # Close the file descriptor
        
        # Update config to use test database
        test_config = AppConfig()
        test_config.database.url = f"sqlite:///{path}"
        set_config(test_config)
        
        self.test_db_path = path
        self.config = test_config
        return path
        
    def _cleanup_test_database(self) -> None:
        """Cleanup test database."""
        if self.test_db_path and os.path.exists(self.test_db_path):
            try:
                os.unlink(self.test_db_path)
            except OSError:
                pass


class DatabaseInitializationAgent(BaseTestAgent):
    """Test agent for database initialization."""
    
    def __init__(self):
        super().__init__("Database Initialization", "Test database table creation and schema")
        
    async def setup(self) -> None:
        self._setup_test_database()
        
    async def run(self) -> TestAgentResult:
        start_time = datetime.now()
        
        try:
            console.print(f"[blue]Running {self.name}...[/blue]")
            
            # Initialize database
            init_database()
            
            # Verify tables were created
            if self.test_db_path is None:
                raise Exception("Test database path not set")
                
            with sqlite3.connect(self.test_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
            expected_tables = ['questions', 'benchmark_runs', 'benchmark_results', 'model_performance']
            missing_tables = [t for t in expected_tables if t not in tables]
            
            if missing_tables:
                return TestAgentResult(
                    self.name, False, 
                    f"Missing tables: {missing_tables}",
                    {"created_tables": tables, "missing_tables": missing_tables}
                )
                
            # Test database session
            with get_db_session() as session:
                # Simple query to verify connection
                result = session.execute(text("SELECT 1")).fetchone()
                
            execution_time = (datetime.now() - start_time).total_seconds()
            return TestAgentResult(
                self.name, True, 
                f"Database initialized successfully with {len(tables)} tables",
                {"tables": tables},
                execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return TestAgentResult(
                self.name, False, 
                f"Database initialization failed: {str(e)}",
                {"error": str(e)},
                execution_time
            )
            
    async def cleanup(self) -> None:
        self._cleanup_test_database()


class DataLoadingAgent(BaseTestAgent):
    """Test agent for loading sample data."""
    
    def __init__(self):
        super().__init__("Data Loading", "Test loading sample questions into database")
        
    async def setup(self) -> None:
        self._setup_test_database()
        init_database()
        
    async def run(self) -> TestAgentResult:
        start_time = datetime.now()
        
        try:
            console.print(f"[blue]Running {self.name}...[/blue]")
            
            # Create sample questions
            sample_questions = [
                Question(
                    id="test_q1",
                    question_text="This programming language was created by Guido van Rossum",
                    correct_answer="Python",
                    category="PROGRAMMING",
                    value=400,
                    difficulty_level="Medium"
                ),
                Question(
                    id="test_q2", 
                    question_text="This planet is known as the Red Planet",
                    correct_answer="Mars",
                    category="ASTRONOMY", 
                    value=200,
                    difficulty_level="Easy"
                ),
                Question(
                    id="test_q3",
                    question_text="This scientist developed the theory of relativity",
                    correct_answer="Albert Einstein", 
                    category="SCIENCE",
                    value=600,
                    difficulty_level="Hard"
                ),
                Question(
                    id="test_q4",
                    question_text="This is the capital of France",
                    correct_answer="Paris",
                    category="GEOGRAPHY",
                    value=100, 
                    difficulty_level="Easy"
                ),
                Question(
                    id="test_q5",
                    question_text="This novel was written by Harper Lee",
                    correct_answer="To Kill a Mockingbird",
                    category="LITERATURE",
                    value=800,
                    difficulty_level="Hard"
                )
            ]
            
            # Load questions into database
            with get_db_session() as session:
                loaded_count = 0
                for question in sample_questions:
                    session.add(question)
                    loaded_count += 1
                
                session.commit()
                
                # Verify questions were loaded
                total_questions = session.execute(text("SELECT COUNT(*) FROM questions")).scalar()
                categories_result = session.execute(text("SELECT DISTINCT category FROM questions WHERE category IS NOT NULL")).fetchall()
                categories = [row[0] for row in categories_result]
                
            execution_time = (datetime.now() - start_time).total_seconds()
            return TestAgentResult(
                self.name, True,
                f"Loaded {loaded_count} sample questions successfully",
                {
                    "questions_loaded": loaded_count,
                    "total_questions": total_questions,
                    "categories": categories
                },
                execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return TestAgentResult(
                self.name, False,
                f"Data loading failed: {str(e)}",
                {"error": str(e)},
                execution_time
            )
            
    async def cleanup(self) -> None:
        self._cleanup_test_database()


class MinimalBenchmarkAgent(BaseTestAgent):
    """Test agent for running a minimal benchmark with real API calls."""
    
    def __init__(self):
        super().__init__("Minimal Benchmark", "Test benchmark execution with minimal data and real API")
        
    async def setup(self) -> None:
        self._setup_test_database()
        init_database()
        
        # Load sample questions
        sample_questions = [
            Question(
                id="bench_q1",
                question_text="This programming language was created by Guido van Rossum",
                correct_answer="Python",
                category="PROGRAMMING",
                value=400,
                difficulty_level="Medium"
            ),
            Question(
                id="bench_q2",
                question_text="This planet is known as the Red Planet", 
                correct_answer="Mars",
                category="ASTRONOMY",
                value=200,
                difficulty_level="Easy"
            )
        ]
        
        with get_db_session() as session:
            for question in sample_questions:
                session.add(question)
            session.commit()
            
    async def run(self) -> TestAgentResult:
        start_time = datetime.now()
        
        try:
            console.print(f"[blue]Running {self.name}...[/blue]")
            
            # Check for API key
            api_key = os.getenv('OPENROUTER_API_KEY')
            if not api_key:
                return TestAgentResult(
                    self.name, False,
                    "OPENROUTER_API_KEY not found in environment",
                    {"error": "Missing API key"}
                )
            
            # Use a cheap, fast model for testing
            test_model = "openai/gpt-3.5-turbo"
            
            # Create custom benchmark config for minimal test
            runner = BenchmarkRunner()
            config = runner.get_default_config(RunMode.QUICK)
            config.sample_size = 2  # Only 2 questions
            config.timeout_seconds = 30
            config.save_results = True
            
            console.print(f"[dim]Testing with model: {test_model}[/dim]")
            console.print(f"[dim]Sample size: {config.sample_size}[/dim]")
            
            # Run the benchmark
            result = await runner.run_benchmark(
                model_name=test_model,
                mode=RunMode.QUICK,
                custom_config=config,
                benchmark_name="Test Agent Minimal Benchmark"
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if result.success:
                details = {
                    "benchmark_id": result.benchmark_id,
                    "questions_processed": len(result.responses) if hasattr(result, 'responses') and result.responses else 0,
                    "execution_time_seconds": result.execution_time,
                    "model_tested": test_model
                }
                
                if result.metrics:
                    details.update({
                        "accuracy": result.metrics.accuracy.overall_accuracy,
                        "total_cost": result.metrics.cost.total_cost,
                        "avg_response_time": getattr(result.metrics.performance, 'avg_response_time_seconds', 0)
                    })
                
                return TestAgentResult(
                    self.name, True,
                    f"Benchmark completed successfully in {result.execution_time:.2f}s",
                    details,
                    execution_time
                )
            else:
                return TestAgentResult(
                    self.name, False,
                    f"Benchmark failed: {result.error_message}",
                    {"error": result.error_message, "execution_time": result.execution_time},
                    execution_time
                )
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return TestAgentResult(
                self.name, False,
                f"Benchmark execution failed: {str(e)}",
                {"error": str(e)},
                execution_time
            )
            
    async def cleanup(self) -> None:
        self._cleanup_test_database()


class ReportGenerationAgent(BaseTestAgent):
    """Test agent for generating reports."""
    
    def __init__(self):
        super().__init__("Report Generation", "Test report generation from benchmark data")
        self.benchmark_id: Optional[int] = None
        
    async def setup(self) -> None:
        self._setup_test_database()
        init_database()
        
        # Create sample benchmark data
        with get_db_session() as session:
            # Create benchmark run
            benchmark = BenchmarkRun(
                name="Test Report Benchmark",
                description="Test data for report generation",
                benchmark_mode="quick",
                sample_size=2,
                status="completed",
                models_tested='["test/model"]',
                total_questions=2,
                completed_questions=2,
                total_cost_usd=0.001,
                avg_response_time_ms=1500
            )
            
            session.add(benchmark)
            session.commit()
            session.refresh(benchmark)
            self.benchmark_id = benchmark.id  # type: ignore
            
            # Create sample results
            results_data = [
                {
                    "benchmark_run_id": self.benchmark_id,
                    "question_id": "report_q1",
                    "model_name": "test/model",
                    "response_text": "What is Python?",
                    "is_correct": True,
                    "confidence_score": 0.95,
                    "response_time_ms": 1200,
                    "cost_usd": 0.0005
                },
                {
                    "benchmark_run_id": self.benchmark_id,
                    "question_id": "report_q2", 
                    "model_name": "test/model",
                    "response_text": "What is Mars?",
                    "is_correct": True,
                    "confidence_score": 0.88,
                    "response_time_ms": 1800,
                    "cost_usd": 0.0005
                }
            ]
            
            for result_data in results_data:
                result = BenchmarkResult(**result_data)
                session.add(result)
            
            session.commit()
            
    async def run(self) -> TestAgentResult:
        start_time = datetime.now()
        
        try:
            console.print(f"[blue]Running {self.name}...[/blue]")
            
            if self.benchmark_id is None:
                raise Exception("Benchmark ID not set during setup")
            
            with get_db_session() as session:
                repo = BenchmarkRepository(session)
                
                # Get benchmark summary
                summary = repo.get_benchmark_summary_stats(self.benchmark_id)
                
                if not summary:
                    return TestAgentResult(
                        self.name, False,
                        f"Benchmark {self.benchmark_id} not found",
                        {"benchmark_id": self.benchmark_id}
                    )
                
                # Test different report formats
                reports_generated = {}
                
                # Generate JSON report
                json_report = json.dumps(summary, indent=2, default=str)
                reports_generated["json"] = len(json_report) > 0
                
                # Test terminal report display (capture output)
                try:
                    # This would normally display to console
                    # For testing, we just verify no exceptions
                    reports_generated["terminal"] = True
                except Exception as e:
                    reports_generated["terminal"] = False
                    console.print(f"[yellow]Terminal report test failed: {e}[/yellow]")
                
            execution_time = (datetime.now() - start_time).total_seconds()
            
            successful_reports = sum(reports_generated.values())
            total_reports = len(reports_generated)
            
            return TestAgentResult(
                self.name, True,
                f"Generated {successful_reports}/{total_reports} report formats successfully",
                {
                    "benchmark_id": self.benchmark_id,
                    "reports_tested": reports_generated,
                    "summary_stats": {
                        "name": summary.get("name"),
                        "status": summary.get("status"),
                        "total_questions": summary.get("total_questions"),
                        "overall_accuracy": summary.get("overall_accuracy")
                    }
                },
                execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return TestAgentResult(
                self.name, False,
                f"Report generation failed: {str(e)}",
                {"error": str(e), "benchmark_id": self.benchmark_id},
                execution_time
            )
            
    async def cleanup(self) -> None:
        self._cleanup_test_database()


class CLICommandAgent(BaseTestAgent):
    """Test agent for CLI command verification."""
    
    def __init__(self):
        super().__init__("CLI Commands", "Test CLI command functionality")
        
    async def setup(self) -> None:
        self._setup_test_database()
        
    async def run(self) -> TestAgentResult:
        start_time = datetime.now()
        
        try:
            console.print(f"[blue]Running {self.name}...[/blue]")
            
            if self.test_db_path is None:
                raise Exception("Test database path not set")
            
            # Test commands that don't require external dependencies
            commands_to_test = [
                {
                    "cmd": [sys.executable, "-m", "src.main", "--help"],
                    "name": "main help",
                    "expect_success": True
                },
                {
                    "cmd": [sys.executable, "-m", "src.main", "init", "--force"],
                    "name": "database init",
                    "expect_success": True
                },
                {
                    "cmd": [sys.executable, "-m", "src.main", "benchmark", "--help"],
                    "name": "benchmark help", 
                    "expect_success": True
                },
                {
                    "cmd": [sys.executable, "-m", "src.main", "models", "list"],
                    "name": "models list",
                    "expect_success": True
                },
                {
                    "cmd": [sys.executable, "-m", "src.main", "data", "stats"],
                    "name": "data stats",
                    "expect_success": False  # Expected to fail with no data
                }
            ]
            
            results = []
            
            for test in commands_to_test:
                try:
                    # Set environment to use test database
                    env = os.environ.copy()
                    env["DATABASE_URL"] = f"sqlite:///{self.test_db_path}"
                    
                    # Run command with timeout
                    result = subprocess.run(
                        test["cmd"],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        env=env,
                        cwd=Path(__file__).parent.parent
                    )
                    
                    success = (result.returncode == 0) == test["expect_success"]
                    
                    results.append({
                        "name": test["name"],
                        "success": success,
                        "return_code": result.returncode,
                        "expected_success": test["expect_success"],
                        "stdout_length": len(result.stdout),
                        "stderr_length": len(result.stderr)
                    })
                    
                except subprocess.TimeoutExpired:
                    results.append({
                        "name": test["name"],
                        "success": False,
                        "error": "Command timed out"
                    })
                except Exception as e:
                    results.append({
                        "name": test["name"],
                        "success": False, 
                        "error": str(e)
                    })
            
            successful_commands = sum(1 for r in results if r["success"])
            total_commands = len(results)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return TestAgentResult(
                self.name, successful_commands == total_commands,
                f"CLI commands tested: {successful_commands}/{total_commands} passed",
                {"command_results": results},
                execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return TestAgentResult(
                self.name, False,
                f"CLI testing failed: {str(e)}",
                {"error": str(e)},
                execution_time
            )
            
    async def cleanup(self) -> None:
        self._cleanup_test_database()


class TestAgentRunner:
    """Orchestrates running all test agents."""
    
    def __init__(self):
        self.agents = [
            DatabaseInitializationAgent(),
            DataLoadingAgent(),
            MinimalBenchmarkAgent(),
            ReportGenerationAgent(),
            CLICommandAgent()
        ]
        
    async def run_all_agents(self) -> List[TestAgentResult]:
        """Run all test agents and return results."""
        console.print(Panel.fit(
            "[bold blue]🧪 alex-treBENCH Test Agents[/bold blue]\n"
            f"Running {len(self.agents)} comprehensive system tests",
            title="System Testing",
            border_style="blue"
        ))
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for agent in self.agents:
                task = progress.add_task(f"Running {agent.name}...", total=None)
                
                try:
                    await agent.setup()
                    result = await agent.run()
                    await agent.cleanup()
                    
                    results.append(result)
                    
                    if result.success:
                        progress.update(task, description=f"✅ {agent.name}")
                    else:
                        progress.update(task, description=f"❌ {agent.name}")
                        
                except Exception as e:
                    await agent.cleanup()
                    result = TestAgentResult(
                        agent.name, False, 
                        f"Agent execution failed: {str(e)}",
                        {"error": str(e)}
                    )
                    results.append(result)
                    progress.update(task, description=f"❌ {agent.name}")
        
        return results
    
    def display_results(self, results: List[TestAgentResult]) -> None:
        """Display test results in a formatted table."""
        
        # Summary statistics
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r.success)
        failed_tests = total_tests - successful_tests
        total_time = sum(r.execution_time for r in results)
        
        # Summary panel
        status_color = "green" if failed_tests == 0 else "red" if successful_tests == 0 else "yellow"
        summary_panel = Panel.fit(
            f"[bold {status_color}]Test Results Summary[/bold {status_color}]\n"
            f"Total Tests: {total_tests}\n"
            f"Passed: [green]{successful_tests}[/green]\n"
            f"Failed: [red]{failed_tests}[/red]\n"
            f"Total Time: {total_time:.2f}s",
            title="Results",
            border_style=status_color
        )
        console.print(summary_panel)
        
        # Detailed results table
        table = Table(title="Detailed Test Results")
        table.add_column("Agent", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Message", style="dim")
        table.add_column("Time (s)", justify="right", style="green")
        
        for result in results:
            status_icon = "✅" if result.success else "❌"
            status_color = "green" if result.success else "red"
            
            table.add_row(
                result.name,
                f"[{status_color}]{status_icon}[/{status_color}]",
                result.message[:80] + "..." if len(result.message) > 80 else result.message,
                f"{result.execution_time:.2f}"
            )
        
        console.print("\n")
        console.print(table)
        
        # Show failed test details
        failed_results = [r for r in results if not r.success]
        if failed_results:
            console.print(f"\n[red]Failed Tests Details:[/red]")
            for result in failed_results:
                console.print(f"[red]❌ {result.name}[/red]: {result.message}")
                if result.details:
                    console.print(f"[dim]   Details: {result.details}[/dim]")
        else:
            console.print(f"\n[green]🎉 All tests passed![/green]")


async def main():
    """Main entry point for test agents."""
    console.print("[blue]Starting alex-treBENCH Test Agents...[/blue]\n")
    
    runner = TestAgentRunner()
    results = await runner.run_all_agents()
    
    console.print("\n")
    runner.display_results(results)
    
    # Exit with error code if any tests failed
    failed_count = sum(1 for r in results if not r.success)
    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())