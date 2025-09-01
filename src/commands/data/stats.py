"""
Data Statistics Commands

This module contains the data statistics command implementation.
"""

import click
from rich.console import Console
from rich.table import Table

from core.database import get_db_session
from utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.command()
@click.option('--benchmark-id', '-b', type=int, help='Show stats for specific benchmark')
@click.option('--detailed', '-d', is_flag=True, help='Show detailed statistics')
@click.pass_context
def stats(ctx, benchmark_id, detailed):
    """Show dataset statistics.
    
    \b
    📊 EXAMPLES:
    
    alex data stats
    alex data stats --detailed
    alex data stats --benchmark-id 1
    alex data stats --benchmark-id 1 --detailed
    
    \b
    💡 Shows comprehensive statistics about the Jeopardy dataset including
    question counts, categories, difficulty distribution, and value ranges.
    """
    try:
        from src.storage.repositories import QuestionRepository
        
        with get_db_session() as session:
            repo = QuestionRepository(session)
            stats = repo.get_question_statistics(benchmark_id)
            
            if not stats or stats.get('total_questions', 0) == 0:
                if benchmark_id:
                    console.print(f"[yellow]No questions found for benchmark {benchmark_id}[/yellow]")
                else:
                    console.print("[yellow]No questions found in database. Run 'alex data init' first.[/yellow]")
                return
            
            # Display basic statistics
            table = Table(title="Dataset Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Total Questions", f"{stats['total_questions']:,}")
            table.add_row("Unique Categories", f"{stats.get('unique_categories', 0):,}")
            
            value_range = stats.get('value_range')
            if value_range:
                table.add_row("Value Range", f"${value_range['min']} - ${value_range['max']}")
                table.add_row("Average Value", f"${value_range.get('average', 0)}")
                table.add_row("Questions with Values", f"{value_range.get('count', 0):,}")
            
            console.print(table)
            
            if detailed:
                # Show category distribution
                category_distribution = stats.get('category_distribution')
                if category_distribution:
                    cat_table = Table(title="Top 10 Categories")
                    cat_table.add_column("Category", style="cyan")
                    cat_table.add_column("Count", justify="right", style="green")
                    cat_table.add_column("Percentage", justify="right", style="yellow")
                    
                    sorted_cats = sorted(category_distribution.items(),
                                       key=lambda x: x[1], reverse=True)[:10]
                    
                    for category, count in sorted_cats:
                        percentage = (count / stats['total_questions']) * 100
                        cat_table.add_row(category, str(count), f"{percentage:.1f}%")
                    
                    console.print(cat_table)
                
                # Show difficulty distribution
                difficulty_distribution = stats.get('difficulty_distribution')
                if difficulty_distribution:
                    diff_table = Table(title="Difficulty Distribution")
                    diff_table.add_column("Difficulty", style="cyan")
                    diff_table.add_column("Count", justify="right", style="green")
                    diff_table.add_column("Percentage", justify="right", style="yellow")
                    
                    for difficulty, count in difficulty_distribution.items():
                        percentage = (count / stats['total_questions']) * 100
                        diff_table.add_row(difficulty, str(count), f"{percentage:.1f}%")
                    
                    console.print(diff_table)
    
    except Exception as e:
        console.print(f"[red]Error retrieving statistics: {str(e)}[/red]")
        logger.exception("Statistics retrieval failed")