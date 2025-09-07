"""
Benchmark Samples Commands

This module contains the benchmark samples extraction command implementation.
Extracts sample questions/answers from benchmark runs for analysis.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from core.database import get_db_session
from storage.repositories.benchmark_repository import BenchmarkRepository
from storage.models import BenchmarkResult, Question
from utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.command()
@click.option('--run-id', '-r', type=int, required=True, help='Benchmark run ID')
@click.option('--count', '-c', type=int, default=10, help='Number of samples to show (default: 10)')
@click.option('--correct', type=int, help='Number of correct samples to include')
@click.option('--incorrect', type=int, help='Number of incorrect samples to include')
@click.option('--model', '-m', type=str, help='Filter samples by specific model')
@click.option('--category', type=str, help='Filter samples by question category')
@click.option('--format', '-f', type=click.Choice(['terminal', 'json', 'csv']),
              default='terminal', help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Save samples to file')
@click.option('--seed', type=int, help='Random seed for reproducible sampling')
@click.pass_context
def samples(ctx, run_id, count, correct, incorrect, model, category, format, output, seed):
    """Extract sample questions/answers from a benchmark run.
    
    Shows a mix of correct and incorrect answers for analysis and debugging.
    
    \b
    📄 EXAMPLES:
    
    alex benchmark samples --run-id 1
    alex benchmark samples --run-id 1 --count 20
    alex benchmark samples --run-id 1 --correct 5 --incorrect 5
    alex benchmark samples --run-id 1 --model "gpt-4" --category "SCIENCE"
    alex benchmark samples --run-id 1 --format json --output samples.json
    alex benchmark samples --run-id 1 --seed 42  # Reproducible sampling
    
    \b
    💡 TIP: Use 'alex benchmark list' to find run IDs.
    """
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    try:
        with get_db_session() as session:
            repo = BenchmarkRepository(session)
            
            # Verify benchmark exists
            benchmark = repo.get_benchmark_by_id(run_id)
            if not benchmark:
                console.print(f"[red]Benchmark {run_id} not found[/red]")
                return
            
            # Build query for benchmark results
            query = session.query(BenchmarkResult, Question).join(
                Question, BenchmarkResult.question_id == Question.id
            ).filter(BenchmarkResult.benchmark_run_id == run_id)
            
            # Apply filters
            if model:
                query = query.filter(BenchmarkResult.model_name.ilike(f'%{model}%'))
            
            if category:
                query = query.filter(Question.category.ilike(f'%{category}%'))
            
            # Get all results
            all_results = query.all()
            
            if not all_results:
                console.print(f"[yellow]No results found for benchmark {run_id} with given filters[/yellow]")
                return
            
            # Separate correct and incorrect results
            correct_results = [(result, question) for result, question in all_results if result.is_correct]
            incorrect_results = [(result, question) for result, question in all_results if not result.is_correct]
            
            console.print(f"[blue]Found {len(correct_results)} correct and {len(incorrect_results)} incorrect results[/blue]")
            
            # Determine sampling counts
            if correct is not None and incorrect is not None:
                correct_count = min(correct, len(correct_results))
                incorrect_count = min(incorrect, len(incorrect_results))
            else:
                # Default: try for 50/50 split
                total_available = len(all_results)
                if correct is not None:
                    correct_count = min(correct, len(correct_results))
                    incorrect_count = min(count - correct_count, len(incorrect_results))
                elif incorrect is not None:
                    incorrect_count = min(incorrect, len(incorrect_results))
                    correct_count = min(count - incorrect_count, len(correct_results))
                else:
                    # Try for balanced split
                    target_each = count // 2
                    correct_count = min(target_each, len(correct_results))
                    incorrect_count = min(count - correct_count, len(incorrect_results))
                    
                    # If we don't have enough incorrect, take more correct
                    if incorrect_count < target_each and len(correct_results) > correct_count:
                        remaining = count - incorrect_count
                        correct_count = min(remaining, len(correct_results))
            
            # Sample results
            sampled_correct = random.sample(correct_results, correct_count) if correct_count > 0 else []
            sampled_incorrect = random.sample(incorrect_results, incorrect_count) if incorrect_count > 0 else []
            
            # Combine and shuffle samples
            samples = sampled_correct + sampled_incorrect
            random.shuffle(samples)
            
            console.print(f"[green]Showing {len(samples)} samples ({len(sampled_correct)} correct, {len(sampled_incorrect)} incorrect)[/green]\n")
            
            # Display based on format
            if format == 'terminal':
                _display_terminal_format(samples, benchmark)
            elif format == 'json':
                samples_data = _prepare_json_format(samples, benchmark)
                if output:
                    _save_to_file(samples_data, output, 'json')
                else:
                    console.print_json(json.dumps(samples_data, indent=2))
            elif format == 'csv':
                samples_data = _prepare_csv_format(samples, benchmark)
                if output:
                    _save_to_file(samples_data, output, 'csv')
                else:
                    console.print(samples_data)
            
            if output and format == 'terminal':
                # Save terminal output as markdown
                samples_data = _prepare_markdown_format(samples, benchmark)
                _save_to_file(samples_data, output, 'md')
                
    except Exception as e:
        console.print(f"[red]Error extracting samples: {str(e)}[/red]")
        logger.exception("Samples extraction failed")


def _display_terminal_format(samples: List[tuple], benchmark):
    """Display samples in rich terminal format."""
    
    for i, (result, question) in enumerate(samples, 1):
        # Create status indicator
        status = "✅ CORRECT" if result.is_correct else "❌ INCORRECT"
        status_color = "green" if result.is_correct else "red"
        
        # Create panel title
        title = f"Sample {i}: {status}"
        
        # Create content
        content_parts = []
        
        # Question info
        content_parts.append(f"[bold cyan]Category:[/bold cyan] {question.category or 'Unknown'}")
        content_parts.append(f"[bold cyan]Value:[/bold cyan] ${question.value or 'Unknown'}")
        content_parts.append(f"[bold cyan]Model:[/bold cyan] {result.model_name}")
        
        if question.air_date:
            content_parts.append(f"[bold cyan]Air Date:[/bold cyan] {question.air_date}")
        
        content_parts.append("")  # Blank line
        
        # Question text
        content_parts.append(f"[bold yellow]Question:[/bold yellow]")
        content_parts.append(f"{question.question_text}")
        content_parts.append("")
        
        # Correct answer
        content_parts.append(f"[bold green]Correct Answer:[/bold green]")
        content_parts.append(f"{question.correct_answer}")
        content_parts.append("")
        
        # Model response
        content_parts.append(f"[bold blue]Model Response:[/bold blue]")
        content_parts.append(f"{result.response_text or 'No response'}")
        
        # Performance metrics
        if result.response_time_ms or result.cost_usd:
            content_parts.append("")
            metrics = []
            if result.response_time_ms:
                metrics.append(f"Time: {result.response_time_ms}ms")
            if result.cost_usd:
                metrics.append(f"Cost: ${result.cost_usd:.6f}")
            if result.confidence_score:
                metrics.append(f"Confidence: {result.confidence_score:.2f}")
            
            content_parts.append(f"[dim]{' | '.join(metrics)}[/dim]")
        
        # Create and display panel
        panel = Panel(
            "\n".join(content_parts),
            title=title,
            border_style=status_color,
            expand=False
        )
        
        console.print(panel)
        console.print()  # Extra spacing


def _prepare_json_format(samples: List[tuple], benchmark) -> Dict[str, Any]:
    """Prepare samples data in JSON format."""
    
    samples_data = {
        "benchmark_info": {
            "run_id": benchmark.id,
            "name": benchmark.name,
            "status": benchmark.status,
            "created_at": benchmark.created_at.isoformat() if benchmark.created_at else None
        },
        "samples": []
    }
    
    for i, (result, question) in enumerate(samples, 1):
        sample = {
            "sample_number": i,
            "is_correct": result.is_correct,
            "question": {
                "id": question.id,
                "text": question.question_text,
                "correct_answer": question.correct_answer,
                "category": question.category,
                "value": question.value,
                "air_date": question.air_date,
                "difficulty_level": question.difficulty_level,
                "round": question.round
            },
            "result": {
                "model_name": result.model_name,
                "response_text": result.response_text,
                "confidence_score": float(result.confidence_score) if result.confidence_score else None,
                "response_time_ms": result.response_time_ms,
                "cost_usd": float(result.cost_usd) if result.cost_usd else None,
                "tokens_generated": result.tokens_generated,
                "retry_count": result.retry_count,
                "error_message": result.error_message
            }
        }
        samples_data["samples"].append(sample)
    
    return samples_data


def _prepare_csv_format(samples: List[tuple], benchmark) -> str:
    """Prepare samples data in CSV format."""
    
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    headers = [
        "sample_number", "is_correct", "model_name", "category", "value", "air_date",
        "question_text", "correct_answer", "model_response", "confidence_score",
        "response_time_ms", "cost_usd", "tokens_generated", "error_message"
    ]
    writer.writerow(headers)
    
    # Data rows
    for i, (result, question) in enumerate(samples, 1):
        row = [
            i,
            result.is_correct,
            result.model_name,
            question.category,
            question.value,
            question.air_date,
            question.question_text,
            question.correct_answer,
            result.response_text,
            result.confidence_score,
            result.response_time_ms,
            result.cost_usd,
            result.tokens_generated,
            result.error_message
        ]
        writer.writerow(row)
    
    return output.getvalue()


def _prepare_markdown_format(samples: List[tuple], benchmark) -> str:
    """Prepare samples data in Markdown format."""
    
    lines = [
        f"# Benchmark Samples - {benchmark.name}",
        "",
        f"**Run ID:** {benchmark.id}",
        f"**Status:** {benchmark.status}",
        f"**Created:** {benchmark.created_at}",
        "",
        f"**Total Samples:** {len(samples)}",
        ""
    ]
    
    for i, (result, question) in enumerate(samples, 1):
        status = "✅ CORRECT" if result.is_correct else "❌ INCORRECT"
        
        lines.extend([
            f"## Sample {i}: {status}",
            "",
            f"**Category:** {question.category or 'Unknown'}  ",
            f"**Value:** ${question.value or 'Unknown'}  ",
            f"**Model:** {result.model_name}  ",
            f"**Air Date:** {question.air_date or 'Unknown'}",
            "",
            f"**Question:** {question.question_text}",
            "",
            f"**Correct Answer:** {question.correct_answer}",
            "",
            f"**Model Response:** {result.response_text or 'No response'}",
            ""
        ])
        
        # Add metrics if available
        metrics = []
        if result.response_time_ms:
            metrics.append(f"Time: {result.response_time_ms}ms")
        if result.cost_usd:
            metrics.append(f"Cost: ${result.cost_usd:.6f}")
        if result.confidence_score:
            metrics.append(f"Confidence: {result.confidence_score:.2f}")
        
        if metrics:
            lines.extend([
                f"**Metrics:** {' | '.join(metrics)}",
                ""
            ])
        
        lines.append("---")
        lines.append("")
    
    return "\n".join(lines)


def _save_to_file(data: Any, output_path: str, file_format: str):
    """Save data to file in specified format."""
    
    output_file = Path(output_path)
    
    try:
        if file_format == 'json':
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            with open(output_file, 'w') as f:
                f.write(str(data))
        
        console.print(f"[green]✓ Samples saved to {output_path}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error saving to {output_path}: {str(e)}[/red]")