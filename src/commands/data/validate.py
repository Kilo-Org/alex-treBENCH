"""
Data Validation Commands

This module contains the data validation command implementation.
"""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from core.database import get_db_session
from utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.command()
@click.option('--benchmark-id', '-b', type=int, help='Validate specific benchmark data')
@click.option('--strict', is_flag=True, help='Use strict validation rules')
@click.option('--fix', is_flag=True, help='Attempt to fix validation issues')
@click.pass_context
def validate(ctx, benchmark_id, strict, fix):
    """Validate dataset integrity and quality.
    
    \b
    🔍 EXAMPLES:
    
    alex data validate
    alex data validate --strict
    alex data validate --benchmark-id 1
    alex data validate --strict --fix
    
    \b
    💡 Performs comprehensive validation of the dataset including:
    - Missing data checks
    - Format validation
    - Category consistency
    - Value range validation
    - Duplicate detection
    """
    try:
        from src.storage.repositories import QuestionRepository
        from src.data.validation import DataValidator
        
        console.print("[blue]Starting data validation...[/blue]")
        
        with get_db_session() as session:
            repo = QuestionRepository(session)
            
            # Build filters
            filters = {}
            if benchmark_id:
                filters['benchmark_id'] = benchmark_id
            
            # Get questions
            questions = repo.get_questions(filters)
            
            if not questions:
                console.print("[yellow]No questions found for validation[/yellow]")
                return
            
            console.print(f"[dim]Validating {len(questions)} questions...[/dim]")
            
            # Initialize validator
            validator = DataValidator(strict_mode=strict)
            
            # Run validation
            validation_results = validator.validate_questions(questions)
            
            # Display validation summary
            summary_table = Table(title="Validation Summary")
            summary_table.add_column("Check", style="cyan")
            summary_table.add_column("Status", justify="center")
            summary_table.add_column("Issues", justify="right", style="red")
            summary_table.add_column("Passed", justify="right", style="green")
            
            total_issues = 0
            total_passed = 0
            
            for check_name, result in validation_results.items():
                status = "✓ PASS" if result['passed'] else "✗ FAIL"
                status_color = "green" if result['passed'] else "red"
                
                issues = result.get('issues', 0)
                passed = result.get('passed_count', 0)
                
                total_issues += issues
                total_passed += passed
                
                summary_table.add_row(
                    check_name.replace('_', ' ').title(),
                    f"[{status_color}]{status}[/{status_color}]",
                    str(issues) if issues > 0 else "-",
                    str(passed) if passed > 0 else "-"
                )
            
            console.print(summary_table)
            
            # Overall status
            overall_status = "PASSED" if total_issues == 0 else "FAILED"
            status_color = "green" if total_issues == 0 else "red"
            
            status_content = f"""[bold]Overall Status:[/bold] [{status_color}]{overall_status}[/{status_color}]
[bold]Total Questions:[/bold] {len(questions):,}
[bold]Issues Found:[/bold] {total_issues:,}
[bold]Validation Checks Passed:[/bold] {total_passed:,}"""
            
            console.print(Panel(status_content, title="🔍 Validation Results", border_style=status_color))
            
            # Show detailed issues if any
            if total_issues > 0:
                console.print("\n[yellow]Detailed Issues:[/yellow]")
                
                for check_name, result in validation_results.items():
                    if not result['passed'] and result.get('details'):
                        console.print(f"\n[red]• {check_name.replace('_', ' ').title()}:[/red]")
                        for detail in result['details'][:5]:  # Show first 5 issues
                            console.print(f"  - {detail}")
                        
                        if len(result['details']) > 5:
                            console.print(f"  - ... and {len(result['details']) - 5} more issues")
                
                if fix:
                    console.print("\n[blue]Attempting to fix validation issues...[/blue]")
                    
                    # Apply fixes
                    fix_results = validator.fix_validation_issues(questions, validation_results)
                    
                    if fix_results['fixed_count'] > 0:
                        console.print(f"[green]✓ Fixed {fix_results['fixed_count']} issues[/green]")
                        
                        # Save fixes back to database
                        repo.update_questions(fix_results['fixed_questions'])
                        console.print(f"[green]✓ Updated {len(fix_results['fixed_questions'])} questions in database[/green]")
                    else:
                        console.print("[yellow]No issues could be automatically fixed[/yellow]")
            else:
                console.print(f"\n[green]✓ All validation checks passed! Dataset is healthy.[/green]")
            
            # Validation recommendations
            if strict and total_issues == 0:
                console.print(f"\n[green]🏆 Excellent! Dataset passes strict validation standards.[/green]")
            elif not strict and total_issues > 0:
                console.print(f"\n[yellow]💡 Try running with --strict for more comprehensive validation.[/yellow]")
    
    except Exception as e:
        console.print(f"[red]Error during validation: {str(e)}[/red]")
        logger.exception("Data validation failed")