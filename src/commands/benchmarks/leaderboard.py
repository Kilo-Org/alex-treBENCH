"""
Benchmark Leaderboard Commands

This module contains the benchmark leaderboard command implementation.
"""

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from core.database import get_db_session
from utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.command()
@click.option('--limit', '-l', type=int, default=10, help='Number of models to show in leaderboard')
@click.option('--report-format', type=click.Choice(['terminal', 'markdown', 'json']),
              default='terminal', help='Report output format')
@click.option('--output', '-o', type=click.Path(), help='Save leaderboard to file')
@click.pass_context
def leaderboard(ctx, limit, report_format, output):
    """Show Jeopardy leaderboard of model performances.
    
    \b
    🏆 EXAMPLES:
    
    alex benchmark leaderboard
    alex benchmark leaderboard --limit 20
    alex benchmark leaderboard --report-format markdown --output leaderboard.md
    alex benchmark leaderboard --report-format json --output leaderboard.json
    
    \b
    📊 Shows models ranked by their total Jeopardy winnings/losses with detailed breakdown.
    """
    
    async def show_leaderboard_async():
        try:
            from storage.repositories import BenchmarkRepository, PerformanceRepository
            from benchmark.reporting import ReportGenerator, ReportConfig, ReportFormat
            
            console.print("[blue]Loading benchmark results for leaderboard...[/blue]")
            
            with get_db_session() as session:
                # Get recent benchmark results with Jeopardy scores
                benchmark_repo = BenchmarkRepository(session)
                perf_repo = PerformanceRepository(session)
                
                # Get all completed benchmarks with performance data
                recent_benchmarks = benchmark_repo.list_benchmarks(limit=100)
                completed_benchmarks = [b for b in recent_benchmarks if getattr(b, 'status', '') == 'completed']
                
                if not completed_benchmarks:
                    console.print("[yellow]No completed benchmarks found. Run some benchmarks first![/yellow]")
                    return
                
                # Get performance records with Jeopardy scores
                leaderboard_data = []
                for benchmark in completed_benchmarks:
                    performances = perf_repo.get_performances_by_benchmark(benchmark.id)
                    for perf in performances:
                        if hasattr(perf, 'jeopardy_score') and perf.jeopardy_score is not None:
                            # Create a mock result object for the leaderboard
                            mock_result = type('MockResult', (), {
                                'model_name': getattr(perf, 'model_name', 'Unknown'),
                                'benchmark_id': benchmark.id,
                                'metrics': type('MockMetrics', (), {
                                    'jeopardy_score': type('MockJeopardyScore', (), {
                                        'total_jeopardy_score': perf.jeopardy_score,
                                        'positive_scores': getattr(perf, 'correct_answers', 0),
                                        'negative_scores': getattr(perf, 'total_questions', 0) - getattr(perf, 'correct_answers', 0),
                                        'category_scores': getattr(perf, 'category_jeopardy_scores_dict', {})
                                    })(),
                                    'accuracy': type('MockAccuracy', (), {
                                        'overall_accuracy': float(perf.accuracy_rate) if perf.accuracy_rate else 0
                                    })()
                                })()
                            })()
                            leaderboard_data.append(mock_result)
                
                if not leaderboard_data:
                    console.print("[yellow]No Jeopardy scores found in benchmarks. Run benchmarks with Jeopardy scoring enabled![/yellow]")
                    return
                
                # Generate leaderboard report
                report_config = ReportConfig(show_jeopardy_scores=True, show_leaderboard=True)
                report_gen = ReportGenerator(config=report_config)
                
                if report_format == 'terminal':
                    leaderboard_report = report_gen.generate_leaderboard_report(leaderboard_data[:limit])
                    console.print(leaderboard_report)
                else:
                    format_enum = ReportFormat.MARKDOWN if report_format == 'markdown' else ReportFormat.JSON
                    leaderboard_report = report_gen.generate_leaderboard_report(leaderboard_data[:limit], format_enum)
                    
                    if output:
                        output_path = Path(output)
                        with open(output_path, 'w') as f:
                            f.write(leaderboard_report)
                        console.print(f"[green]Leaderboard saved to {output}[/green]")
                    else:
                        console.print(leaderboard_report)
                
                console.print(f"\n[dim]Showing top {min(limit, len(leaderboard_data))} models from {len(completed_benchmarks)} completed benchmarks[/dim]")
                
        except Exception as e:
            console.print(f"[red]Error generating leaderboard: {str(e)}[/red]")
            logger.exception("Leaderboard generation failed")
            sys.exit(1)
    
    asyncio.run(show_leaderboard_async())