"""
Data Sampling Commands

This module contains the data sampling command implementation.
"""

import pandas as pd
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from core.database import get_db_session
from utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.command()
@click.option('--benchmark-id', '-b', type=int, help='Sample from specific benchmark')
@click.option('--size', '-s', type=int, default=100, help='Sample size (default: 100)')
@click.option('--method', type=click.Choice(['random', 'stratified', 'systematic']), 
              default='stratified', help='Sampling method')
@click.option('--seed', type=int, default=123, help='Random seed for reproducibility')
@click.option('--output', '-o', type=click.Path(), help='Save sample to CSV file')
@click.pass_context
def sample(ctx, benchmark_id, size, method, seed, output):
    """Generate dataset samples using various sampling methods.
    
    \b
    📊 EXAMPLES:
    
    alex data sample
    alex data sample --size 200 --method random
    alex data sample --benchmark-id 1 --size 50
    alex data sample --method stratified --seed 456 --output sample.csv
    
    \b
    💡 Generate statistical samples from the Jeopardy dataset for testing or analysis.
    Stratified sampling ensures representative category distribution.
    """
    try:
        from src.storage.repositories import QuestionRepository
        from src.data.sampling import StatisticalSampler
        
        # Get questions from database
        with get_db_session() as session:
            repo = QuestionRepository(session)
            
            # Build filters
            filters = {}
            if benchmark_id:
                filters['benchmark_id'] = benchmark_id
            
            # Get questions
            questions = repo.get_questions(filters)
            
            if not questions:
                console.print("[yellow]No questions found matching the specified criteria[/yellow]")
                return
            
            # Convert to DataFrame
            data = []
            for q in questions:
                data.append({
                    'question_id': q.id,
                    'question': q.question_text,
                    'answer': q.correct_answer,
                    'category': q.category,
                    'value': q.value,
                    'difficulty_level': q.difficulty_level
                })
            
            df = pd.DataFrame(data)
            console.print(f"Found {len(df)} questions matching criteria")
        
        # Initialize sampler
        sampler = StatisticalSampler()
        
        # Apply sampling method
        if method == 'stratified':
            sample_df = sampler.stratified_sample(df, size, seed)
        elif method == 'random':
            sample_df = sampler.random_sample(df, size, seed)
        elif method == 'systematic':
            sample_df = sampler.systematic_sample(df, size, seed)
        else:
            sample_df = sampler.stratified_sample(df, size, seed)
        
        # Display sample information
        console.print(f"[green]✓ Generated {method} sample of {len(sample_df)} questions[/green]")
        
        # Show sample statistics
        stats_table = Table(title="Sample Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Sample Size", str(len(sample_df)))
        stats_table.add_row("Sampling Method", method.title())
        stats_table.add_row("Random Seed", str(seed))
        
        if 'category' in sample_df.columns:
            stats_table.add_row("Unique Categories", str(sample_df['category'].nunique()))
        
        if 'difficulty_level' in sample_df.columns:
            diff_counts = sample_df['difficulty_level'].value_counts()
            stats_table.add_row("Difficulty Breakdown", ", ".join([f"{k}: {v}" for k, v in diff_counts.items()]))
        
        if 'value' in sample_df.columns and sample_df['value'].notna().any():
            stats_table.add_row("Value Range", f"${sample_df['value'].min():.0f} - ${sample_df['value'].max():.0f}")
        
        console.print(stats_table)
        
        # Show sample preview
        if len(sample_df) > 0:
            preview_table = Table(title="Sample Preview (first 5 questions)")
            preview_table.add_column("Category", style="cyan", max_width=20)
            preview_table.add_column("Question", style="white", max_width=50)
            preview_table.add_column("Answer", style="green", max_width=30)
            preview_table.add_column("Value", justify="right", style="yellow")
            
            for _, row in sample_df.head(5).iterrows():
                question_preview = (row['question'][:47] + "...") if len(str(row['question'])) > 50 else str(row['question'])
                answer_preview = (row['answer'][:27] + "...") if len(str(row['answer'])) > 30 else str(row['answer'])
                
                preview_table.add_row(
                    str(row.get('category', 'N/A'))[:20],
                    question_preview,
                    answer_preview,
                    f"${row.get('value', 0):.0f}" if pd.notna(row.get('value')) else "N/A"
                )
            
            console.print(preview_table)
        
        # Save to file if requested
        if output:
            output_path = Path(output)
            sample_df.to_csv(output_path, index=False)
            console.print(f"[green]✓ Sample saved to {output_path}[/green]")
        elif benchmark_id:
            # Auto-save if benchmark_id specified
            output_dir = Path("data/sample")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"benchmark_{benchmark_id}_{method}_{size}.csv"
            sample_df.to_csv(output_path, index=False)
            console.print(f"[green]✓ Sample saved to {output_path}[/green]")
    
    except Exception as e:
        console.print(f"[red]Error during sampling: {str(e)}[/red]")
        logger.exception("Sampling failed")