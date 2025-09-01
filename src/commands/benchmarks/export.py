"""
Benchmark Export Commands

This module contains the benchmark export command implementation for exporting
benchmark results in various formats (JSON, CSV, HTML).
"""

import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

import click
from rich.console import Console
from rich.table import Table

from core.database import get_db_session
from utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.command()
@click.option('--benchmark-id', '-b', type=int, help='Export specific benchmark results')
@click.option('--format', type=click.Choice(['json', 'csv', 'html']), default='json', help='Export format')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.pass_context
def export(ctx, benchmark_id, format, output):
    """Export benchmark results.
    
    \b
    📤 EXAMPLES:
    
    alex benchmark export --benchmark-id 1
    alex benchmark export --benchmark-id 1 --format csv --output results.csv
    alex benchmark export --benchmark-id 1 --format html --output report.html
    alex benchmark export --format json  # Export all benchmarks
    
    \b
    💡 Exports benchmark data in various formats for analysis or reporting.
    """
    
    try:
        from storage.repositories import BenchmarkRepository
        
        with get_db_session() as session:
            repo = BenchmarkRepository(session)
            
            if benchmark_id:
                # Export specific benchmark
                benchmark = repo.get_benchmark(benchmark_id)
                if not benchmark:
                    console.print(f"[red]Benchmark {benchmark_id} not found[/red]")
                    return
                
                benchmarks = [benchmark]
                filename_suffix = f"benchmark_{benchmark_id}"
            else:
                # Export all benchmarks
                benchmarks = repo.list_benchmarks()
                if not benchmarks:
                    console.print("[yellow]No benchmark data found to export[/yellow]")
                    return
                
                filename_suffix = "all_benchmarks"
            
            # Generate output filename if not provided
            if not output:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output = f"export_{filename_suffix}_{timestamp}.{format}"
            
            output_path = Path(output)
            
            # Convert benchmarks to export data
            export_data = _prepare_export_data(benchmarks)
            
            # Export based on format
            if format == 'json':
                _export_json(export_data, output_path)
            elif format == 'csv':
                _export_csv(export_data, output_path)
            elif format == 'html':
                _export_html(export_data, output_path)
            
            console.print(f"\n[green]✓ Exported {len(export_data)} benchmark(s) to {output_path}[/green]")
            console.print(f"[dim]Format: {format.upper()}, Size: {output_path.stat().st_size} bytes[/dim]")
            
    except Exception as e:
        console.print(f"[red]Error exporting data: {str(e)}[/red]")
        logger.exception("Export failed")


def _prepare_export_data(benchmarks) -> List[Dict[str, Any]]:
    """Convert benchmark objects to export-ready data."""
    export_data = []
    
    for benchmark in benchmarks:
        # Convert benchmark to dict with safe value extraction
        benchmark_data = {
            'id': benchmark.id,
            'name': benchmark.name,
            'status': benchmark.status,
            'created_at': benchmark.created_at.isoformat() if benchmark.created_at else None,
            'completed_at': benchmark.completed_at.isoformat() if benchmark.completed_at else None,
            'sample_size': benchmark.sample_size,
            'benchmark_mode': benchmark.benchmark_mode,
            'models_tested': benchmark.models_tested_list if hasattr(benchmark, 'models_tested_list') else [],
            'total_cost_usd': float(benchmark.total_cost_usd) if benchmark.total_cost_usd else 0.0,
            'total_tokens': benchmark.total_tokens if benchmark.total_tokens else 0,
            'avg_response_time_ms': float(benchmark.avg_response_time_ms) if benchmark.avg_response_time_ms else 0.0,
        }
        
        export_data.append(benchmark_data)
    
    return export_data


def _export_json(data: List[Dict[str, Any]], output_path: Path) -> None:
    """Export data as JSON."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'exported_at': datetime.now().isoformat(),
            'total_benchmarks': len(data),
            'benchmarks': data
        }, f, indent=2, ensure_ascii=False)


def _export_csv(data: List[Dict[str, Any]], output_path: Path) -> None:
    """Export data as CSV."""
    if not data:
        return
    
    # Flatten nested data for CSV format
    flattened_data = []
    for benchmark in data:
        flat_record = _flatten_dict(benchmark)
        flattened_data.append(flat_record)
    
    # Get all unique keys for CSV headers
    all_keys = set()
    for record in flattened_data:
        all_keys.update(record.keys())
    
    fieldnames = sorted(all_keys)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened_data)


def _export_html(data: List[Dict[str, Any]], output_path: Path) -> None:
    """Export data as HTML."""
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Export Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .header {{ color: #333; margin-bottom: 20px; }}
        .summary {{ background-color: #f9f9f9; padding: 10px; margin: 20px 0; }}
        .status-completed {{ color: green; font-weight: bold; }}
        .status-failed {{ color: red; font-weight: bold; }}
        .status-running {{ color: orange; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Benchmark Export Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Total Benchmarks: {len(data)}</p>
    </div>
    
    <table>
        <tr>
            <th>ID</th>
            <th>Name</th>
            <th>Status</th>
            <th>Created</th>
            <th>Sample Size</th>
            <th>Models Tested</th>
            <th>Total Cost</th>
        </tr>
"""
    
    for benchmark in data:
        status_class = f"status-{benchmark.get('status', '').lower()}"
        models_tested = benchmark.get('models_tested', [])
        models_str = ', '.join(models_tested) if models_tested else 'N/A'
        
        html_content += f"""
        <tr>
            <td>{benchmark.get('id', 'N/A')}</td>
            <td>{benchmark.get('name', 'N/A')}</td>
            <td class="{status_class}">{benchmark.get('status', 'N/A')}</td>
            <td>{_format_datetime(benchmark.get('created_at'))}</td>
            <td>{benchmark.get('sample_size', 0)}</td>
            <td>{models_str}</td>
            <td>${benchmark.get('total_cost_usd', 0):.4f}</td>
        </tr>
"""
    
    html_content += """
    </table>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Convert lists to comma-separated strings
            items.append((new_key, ', '.join(str(item) for item in v)))
        else:
            items.append((new_key, v))
    return dict(items)


def _format_datetime(dt_str: Any) -> str:
    """Format datetime for display."""
    if not dt_str:
        return 'N/A'
    # Already converted to ISO string in _prepare_export_data
    return str(dt_str)