"""
Data Initialization Script

Script to download, preprocess, and load the Jeopardy dataset into the database.
Runnable via CLI: python -m src.main data init
"""

import sys
from pathlib import Path
from typing import Optional
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ingestion import KaggleDatasetLoader
from data.preprocessing import DataPreprocessor
from data.validation import DataValidator
# Use the clean database system to avoid registry conflicts
from storage.clean_db_system import (
    Question, BenchmarkRun,
    get_clean_db_session, init_clean_database
)
from core.config import get_config
from core.exceptions import DataIngestionError, ValidationError, DatabaseError
from utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


class DataInitializer:
    """Handles the complete data initialization process."""
    
    def __init__(self, force_download: bool = False, validate_strict: bool = False):
        """
        Initialize the data loader.
        
        Args:
            force_download: Force re-download of dataset
            validate_strict: Use strict validation rules
        """
        self.force_download = force_download
        self.validate_strict = validate_strict
        self.config = get_config()
        
        # Initialize components
        self.loader = KaggleDatasetLoader()
        self.preprocessor = DataPreprocessor()
        self.validator = DataValidator(strict_mode=validate_strict)
    
    def initialize_database(self) -> None:
        """Initialize the database schema."""
        try:
            console.print("[yellow]Initializing database schema...[/yellow]")
            init_clean_database()
            console.print("[green]✓ Database schema initialized[/green]")
        except Exception as e:
            console.print(f"[red]✗ Database initialization failed: {str(e)}[/red]")
            raise
    
    def download_dataset(self) -> pd.DataFrame:
        """Download and load the raw dataset."""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Downloading Jeopardy dataset...", total=None)
                
                # Download and load dataset
                df = self.loader.load_dataset(force_download=self.force_download)
                
                progress.update(task, description=f"Downloaded {len(df)} questions")
            
            console.print(f"[green]✓ Dataset loaded: {len(df)} questions[/green]")
            return df
            
        except DataIngestionError as e:
            console.print(f"[red]✗ Dataset download failed: {str(e)}[/red]")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the raw dataset."""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=console
            ) as progress:
                
                initial_count = len(df)
                task = progress.add_task("Preprocessing data...", total=5)
                
                # Step 1: Standardize columns
                progress.update(task, advance=1, description="Standardizing columns...")
                df = self.preprocessor._standardize_columns(df)
                
                # Step 2: Clean text
                progress.update(task, advance=1, description="Cleaning text...")
                df = self.preprocessor._clean_text_columns(df)
                
                # Step 3: Normalize values
                progress.update(task, advance=1, description="Normalizing values...")
                df = self.preprocessor._normalize_values(df)
                df = self.preprocessor._parse_dates(df)
                
                # Step 4: Filter invalid records
                progress.update(task, advance=1, description="Filtering invalid records...")
                df = self.preprocessor._filter_invalid_records(df)
                
                # Step 5: Add derived columns
                progress.update(task, advance=1, description="Adding metadata...")
                df = self.preprocessor._add_difficulty_levels(df)
                df = self.preprocessor._add_metadata_columns(df)
            
            filtered_count = len(df)
            retention_rate = (filtered_count / initial_count) * 100 if initial_count > 0 else 0
            
            console.print(f"[green]✓ Preprocessing complete: {filtered_count}/{initial_count} questions retained ({retention_rate:.1f}%)[/green]")
            return df
            
        except Exception as e:
            console.print(f"[red]✗ Preprocessing failed: {str(e)}[/red]")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate the processed dataset."""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Validating data quality...", total=None)
                
                validation_results = self.validator.validate_dataframe(df)
                
                progress.update(task, description="Validation complete")
            
            # Display validation summary
            valid_rate = (validation_results['valid_questions'] / validation_results['total_questions']) * 100
            
            if valid_rate < 80 and self.validate_strict:
                console.print(f"[red]✗ Validation failed: Only {valid_rate:.1f}% of questions are valid[/red]")
                raise ValidationError(f"Data quality too low: {valid_rate:.1f}% valid")
            else:
                console.print(f"[green]✓ Data validation complete: {valid_rate:.1f}% questions valid[/green]")
            
            return df
            
        except ValidationError:
            raise
        except Exception as e:
            console.print(f"[red]✗ Validation failed: {str(e)}[/red]")
            raise
    
    def save_to_database(self, df: pd.DataFrame) -> dict:
        """Save the processed dataset to the database."""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                
                # Create benchmark record
                task = progress.add_task("Saving to database...", total=3)
                
                with get_clean_db_session() as session:
                    # Step 1: Create benchmark
                    progress.update(task, advance=1, description="Creating benchmark record...")
                    
                    benchmark = BenchmarkRun(
                        name=f"jeopardy_dataset_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
                        description="Complete Jeopardy dataset from Kaggle",
                        benchmark_mode="data_ingestion",
                        sample_size=len(df),
                        status="running"
                    )
                    session.add(benchmark)
                    session.flush()  # Get ID
                    
                    # Step 2: Convert DataFrame to Question objects and save
                    progress.update(task, advance=1, description="Saving questions...")
                    
                    # Convert DataFrame rows to Question objects
                    questions = []
                    for idx, row in df.iterrows():
                        # Handle air_date conversion safely
                        air_date_str = None
                        air_date_value = row.get('air_date')
                        if pd.notna(air_date_value) and air_date_value is not None:
                            try:
                                air_date_parsed = pd.to_datetime(air_date_value)
                                air_date_str = str(air_date_parsed.date())
                            except (ValueError, TypeError):
                                air_date_str = str(air_date_value)
                        
                        # Generate unique ID using row index and hash for uniqueness
                        unique_id = row.get('id', f"q_{idx}_{hash(f"{row.get('question', '')}{row.get('answer', '')}")}")
                        question = Question(
                            id=str(unique_id),
                            question_text=row.get('question', ''),
                            correct_answer=row.get('answer', ''),
                            category=row.get('category', ''),
                            value=int(row.get('value', 0)) if pd.notna(row.get('value')) else 0,
                            air_date=air_date_str,
                            show_number=int(row.get('show_number', 0)) if pd.notna(row.get('show_number')) else None,
                            round=row.get('round', ''),
                            difficulty_level=row.get('difficulty_level', 'Medium')
                        )
                        questions.append(question)
                    
                    # Save questions in batch using SQLAlchemy directly
                    session.add_all(questions)
                    session.flush()
                    saved_questions = questions
                    
                    # Step 3: Generate basic statistics
                    progress.update(task, advance=1, description="Generating statistics...")
                    
                    # Create basic statistics using direct SQLAlchemy queries
                    total_questions = session.query(Question).count()
                    unique_categories = session.query(Question.category).distinct().count()
                    
                    stats = {
                        'total_questions': total_questions,
                        'unique_categories': unique_categories,
                        'saved_questions': len(saved_questions),
                        'value_range': {
                            'min': df['value'].min() if 'value' in df.columns else 0,
                            'max': df['value'].max() if 'value' in df.columns else 0,
                            'average': df['value'].mean() if 'value' in df.columns else 0
                        }
                    }
                    
                    # Update benchmark - use direct update approach
                    benchmark.status = 'completed'  # type: ignore
                    benchmark.total_questions = len(saved_questions)  # type: ignore
                    benchmark.completed_questions = len(saved_questions)  # type: ignore
                    
                    # Capture ID and other values while session is active
                    benchmark_id = benchmark.id
                    questions_count = len(saved_questions)
                    session.commit()
            
            console.print(f"[green]✓ Data saved to database: {questions_count} questions in benchmark {benchmark_id}[/green]")
            
            return {
                'benchmark_id': benchmark_id,
                'questions_saved': questions_count,
                'statistics': stats
            }
            
        except DatabaseError as e:
            console.print(f"[red]✗ Database save failed: {str(e)}[/red]")
            raise
        except Exception as e:
            console.print(f"[red]✗ Unexpected error during save: {str(e)}[/red]")
            raise
    
    def display_statistics(self, stats: dict) -> None:
        """Display comprehensive dataset statistics."""
        
        # Main statistics panel
        main_stats = f"""
        [bold]Total Questions:[/bold] {stats['total_questions']:,}
        [bold]Unique Categories:[/bold] {stats['unique_categories']:,}
        [bold]Value Range:[/bold] ${stats['value_range'].get('min', 'N/A')} - ${stats['value_range'].get('max', 'N/A')}
        [bold]Average Value:[/bold] ${stats['value_range'].get('average', 'N/A')}
        """
        
        console.print(Panel(main_stats, title="📊 Dataset Statistics", border_style="green"))
        
        # Category distribution table
        if stats.get('category_distribution'):
            table = Table(title="Top 10 Categories")
            table.add_column("Category", style="cyan")
            table.add_column("Count", justify="right", style="green")
            table.add_column("Percentage", justify="right", style="yellow")
            
            total_questions = stats['total_questions']
            sorted_categories = sorted(
                stats['category_distribution'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for category, count in sorted_categories[:10]:
                percentage = (count / total_questions) * 100 if total_questions > 0 else 0
                table.add_row(category, str(count), f"{percentage:.1f}%")
            
            console.print(table)
        
        # Difficulty distribution table
        if stats.get('difficulty_distribution'):
            table = Table(title="Difficulty Distribution")
            table.add_column("Difficulty", style="cyan")
            table.add_column("Count", justify="right", style="green")
            table.add_column("Percentage", justify="right", style="yellow")
            
            total_questions = stats['total_questions']
            for difficulty, count in stats['difficulty_distribution'].items():
                percentage = (count / total_questions) * 100 if total_questions > 0 else 0
                table.add_row(difficulty, str(count), f"{percentage:.1f}%")
            
            console.print(table)
        
        # Value distribution table
        if stats.get('value_distribution'):
            table = Table(title="Value Range Distribution")
            table.add_column("Value Range", style="cyan")
            table.add_column("Count", justify="right", style="green")
            table.add_column("Percentage", justify="right", style="yellow")
            
            total_questions = stats['total_questions']
            for value_range, count in stats['value_distribution'].items():
                percentage = (count / total_questions) * 100 if total_questions > 0 else 0
                table.add_row(value_range, str(count), f"{percentage:.1f}%")
            
            console.print(table)
    
    def run(self) -> dict:
        """Execute the complete data initialization process."""
        try:
            console.print(Panel.fit(
                "[bold blue]Jeopardy Dataset Initialization[/bold blue]\n"
                "[dim]Downloading, processing, and loading Jeopardy questions[/dim]",
                border_style="blue"
            ))
            
            # Step 1: Initialize database
            self.initialize_database()
            
            # Step 2: Download dataset
            raw_df = self.download_dataset()
            
            # Step 3: Preprocess data
            processed_df = self.preprocess_data(raw_df)
            
            # Step 4: Validate data
            validated_df = self.validate_data(processed_df)
            
            # Step 5: Save to database
            result = self.save_to_database(validated_df)
            
            # Step 6: Display statistics
            self.display_statistics(result['statistics'])
            
            console.print(Panel.fit(
                f"[bold green]✓ Data initialization complete![/bold green]\n"
                f"[dim]Benchmark ID: {result['benchmark_id']}[/dim]\n"
                f"[dim]Questions loaded: {result['questions_saved']:,}[/dim]",
                border_style="green"
            ))
            
            return result
            
        except Exception as e:
            console.print(Panel.fit(
                f"[bold red]✗ Data initialization failed![/bold red]\n"
                f"[dim]Error: {str(e)}[/dim]",
                border_style="red"
            ))
            raise


def main(force_download: bool = False, strict_validation: bool = False) -> Optional[dict]:
    """
    Main function for data initialization.
    
    Args:
        force_download: Force re-download of dataset
        strict_validation: Use strict validation rules
        
    Returns:
        Dictionary with initialization results or None if failed
    """
    try:
        initializer = DataInitializer(
            force_download=force_download,
            validate_strict=strict_validation
        )
        return initializer.run()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Initialization cancelled by user[/yellow]")
        return None
    except Exception as e:
        logger.exception("Data initialization failed")
        console.print(f"[red]Initialization failed: {str(e)}[/red]")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize Jeopardy dataset")
    parser.add_argument("--force", action="store_true", help="Force re-download of dataset")
    parser.add_argument("--strict", action="store_true", help="Use strict validation rules")
    
    args = parser.parse_args()
    
    result = main(force_download=args.force, strict_validation=args.strict)
    
    if result is None:
        sys.exit(1)
    else:
        sys.exit(0)