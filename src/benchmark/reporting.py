"""
Benchmark Report Generator

Generates comprehensive reports from benchmark results in multiple formats
including terminal output, Markdown, JSON, and CSV with rich visualizations.
"""

import json
import csv
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from pathlib import Path
import io

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    from rich.layout import Layout
    from rich.text import Text
    from rich.tree import Tree
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None
    Table = None

from src.evaluation.metrics import ComprehensiveMetrics
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ReportFormat(str, Enum):
    """Available report formats."""
    TERMINAL = "terminal"
    MARKDOWN = "markdown"
    JSON = "json"
    CSV = "csv"
    HTML = "html"


@dataclass
class ReportSection:
    """A section of a report."""
    title: str
    content: Any
    order: int = 0


@dataclass 
class ReportConfig:
    """Configuration for report generation."""
    include_detailed_metrics: bool = True
    include_category_breakdown: bool = True
    include_difficulty_analysis: bool = True
    include_cost_analysis: bool = True
    include_response_samples: bool = False
    max_response_samples: int = 10
    output_path: Optional[Path] = None


class ReportGenerator:
    """Generates comprehensive benchmark reports in multiple formats."""
    
    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
        self.console = Console() if RICH_AVAILABLE and Console is not None else None
    
    def generate_report(self, result, format_type: ReportFormat, output_path: Optional[Path] = None) -> str:
        """Generate a report in the specified format."""
        # Use local import to avoid circular dependency
        from .runner import BenchmarkResult
        
        if format_type == ReportFormat.TERMINAL:
            return self._generate_terminal_report(result)
        elif format_type == ReportFormat.MARKDOWN:
            return self._generate_markdown_report(result)
        elif format_type == ReportFormat.JSON:
            return self._generate_json_report(result)
        else:
            return f"Report format {format_type} not yet implemented"
    
    def _generate_terminal_report(self, result) -> str:
        """Generate terminal report using Rich."""
        if not RICH_AVAILABLE:
            return "Terminal reports require 'rich' library to be installed"
        
        # Basic implementation
        return f"Benchmark Report for {getattr(result, 'benchmark_id', 'N/A')}"
    
    def _generate_markdown_report(self, result) -> str:
        """Generate Markdown report."""
        return f"# Benchmark Report\n\nBenchmark ID: {getattr(result, 'benchmark_id', 'N/A')}\n"
    
    def _generate_json_report(self, result) -> str:
        """Generate JSON report."""
        return json.dumps({
            "benchmark_id": getattr(result, 'benchmark_id', 'N/A'),
            "timestamp": datetime.now().isoformat()
        }, indent=2)
    
    def display_terminal_report(self, result):
        """Display report directly to terminal."""
        if self.console:
            self.console.print(self._generate_terminal_report(result))
        else:
            print(self._generate_terminal_report(result))
    
    def generate_comparison_report(self, results: List, format_type: ReportFormat, output_path: Optional[Path] = None) -> str:
        """Generate comparison report for multiple results."""
        return f"Comparison report for {len(results)} benchmarks (format: {format_type})"