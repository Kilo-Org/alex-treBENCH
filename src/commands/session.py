"""
Session Management Commands

This module contains the session management command implementation.
"""

import json
import sys
from datetime import datetime

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from core.database import get_db_session
from utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.group()
def session():
    """Session management commands."""
    pass


@session.command('list')
@click.option('--status', type=click.Choice(['active', 'paused', 'completed', 'failed']), 
              help='Filter by session status')
@click.option('--limit', '-l', type=int, default=10, help='Number of sessions to show')
@click.pass_context
def list_sessions(ctx, status, limit):
    """List benchmark sessions.
    
    \b
    📋 EXAMPLES:
    
    alex session list
    alex session list --status active
    alex session list --limit 20
    
    \b
    💡 Shows all benchmark sessions with their current status and progress.
    """
    try:
        from core.session import SessionManager
        
        session_manager = SessionManager()
        sessions = session_manager.list_sessions(status_filter=status, limit=limit)
        
        if not sessions:
            console.print("[yellow]No sessions found matching criteria[/yellow]")
            return
        
        # Create sessions table
        table = Table(title="Benchmark Sessions")
        table.add_column("ID", justify="right", style="cyan")
        table.add_column("Name", style="magenta")
        table.add_column("Status", justify="center")
        table.add_column("Progress", justify="right", style="blue")
        table.add_column("Started", style="dim")
        table.add_column("Duration", style="green")
        
        for session_info in sessions:
            # Status coloring
            status_color = {
                'active': 'green',
                'paused': 'yellow',
                'completed': 'blue',
                'failed': 'red'
            }.get(session_info.status, 'white')
            
            status_display = f"[{status_color}]{session_info.status.upper()}[/{status_color}]"
            
            # Progress calculation
            if session_info.total_questions > 0:
                progress_pct = (session_info.completed_questions / session_info.total_questions) * 100
                progress_display = f"{progress_pct:.1f}%"
            else:
                progress_display = "N/A"
            
            # Duration calculation
            if session_info.started_at:
                if session_info.completed_at:
                    duration = session_info.completed_at - session_info.started_at
                else:
                    duration = datetime.now() - session_info.started_at
                duration_display = str(duration).split('.')[0]  # Remove microseconds
            else:
                duration_display = "N/A"
            
            table.add_row(
                str(session_info.id),
                session_info.name,
                status_display,
                progress_display,
                session_info.started_at.strftime('%Y-%m-%d %H:%M') if session_info.started_at else 'N/A',
                duration_display
            )
        
        console.print(table)
        console.print(f"\n[dim]Showing {len(sessions)} sessions[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error listing sessions: {str(e)}[/red]")
        logger.exception("Session listing failed")


@session.command('pause')
@click.argument('session_id', type=int)
@click.pass_context
def pause_session(ctx, session_id):
    """Pause an active benchmark session.
    
    \b
    ⏸️ EXAMPLES:
    
    alex session pause 1
    alex session pause 42
    
    \b
    💡 Pauses a running benchmark session, saving current progress.
    The session can be resumed later from the same point.
    """
    try:
        from core.session import SessionManager
        
        session_manager = SessionManager()
        
        # Check if session exists and is active
        session_info = session_manager.get_session(session_id)
        if not session_info:
            console.print(f"[red]Session {session_id} not found[/red]")
            return
        
        if session_info.status != 'active':
            console.print(f"[yellow]Session {session_id} is not active (status: {session_info.status})[/yellow]")
            return
        
        # Pause the session
        success = session_manager.pause_session(session_id)
        
        if success:
            console.print(f"[green]✓ Session {session_id} paused successfully[/green]")
            console.print(f"[dim]Progress saved: {session_info.completed_questions}/{session_info.total_questions} questions[/dim]")
        else:
            console.print(f"[red]Failed to pause session {session_id}[/red]")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[red]Error pausing session: {str(e)}[/red]")
        logger.exception("Session pause failed")
        sys.exit(1)


@session.command('resume')
@click.argument('session_id', type=int)
@click.pass_context
def resume_session(ctx, session_id):
    """Resume a paused benchmark session.
    
    \b
    ▶️ EXAMPLES:
    
    alex session resume 1
    alex session resume 42
    
    \b
    💡 Resumes a paused benchmark session from where it left off.
    """
    try:
        from core.session import SessionManager
        
        session_manager = SessionManager()
        
        # Check if session exists and is paused
        session_info = session_manager.get_session(session_id)
        if not session_info:
            console.print(f"[red]Session {session_id} not found[/red]")
            return
        
        if session_info.status != 'paused':
            console.print(f"[yellow]Session {session_id} is not paused (status: {session_info.status})[/yellow]")
            return
        
        console.print(f"[blue]Resuming session {session_id}: {session_info.name}[/blue]")
        console.print(f"[dim]Resuming from {session_info.completed_questions}/{session_info.total_questions} questions[/dim]")
        
        # Resume the session
        success = session_manager.resume_session(session_id)
        
        if success:
            console.print(f"[green]✓ Session {session_id} resumed successfully[/green]")
        else:
            console.print(f"[red]Failed to resume session {session_id}[/red]")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[red]Error resuming session: {str(e)}[/red]")
        logger.exception("Session resume failed")
        sys.exit(1)


@session.command('status')
@click.argument('session_id', type=int)
@click.pass_context
def session_status(ctx, session_id):
    """Show detailed session status.
    
    \b
    🔍 EXAMPLES:
    
    alex session status 1
    alex session status 42
    
    \b
    📊 Shows comprehensive session information including progress, timing,
    and performance metrics.
    """
    try:
        from core.session import SessionManager
        
        session_manager = SessionManager()
        session_info = session_manager.get_session(session_id)
        
        if not session_info:
            console.print(f"[red]Session {session_id} not found[/red]")
            return
        
        # Status color
        status_color = {
            'active': 'green',
            'paused': 'yellow', 
            'completed': 'blue',
            'failed': 'red'
        }.get(session_info.status, 'white')
        
        # Build session info content
        info_content = f"""[bold]Session ID:[/bold] {session_info.id}
[bold]Name:[/bold] {session_info.name}
[bold]Status:[/bold] [{status_color}]{session_info.status.upper()}[/{status_color}]
[bold]Benchmark ID:[/bold] {session_info.benchmark_id}
[bold]Model:[/bold] {session_info.model_name}

[bold]Progress:[/bold]
  Questions Completed: {session_info.completed_questions:,}
  Total Questions: {session_info.total_questions:,}
  Progress: {(session_info.completed_questions / session_info.total_questions * 100) if session_info.total_questions > 0 else 0:.1f}%

[bold]Timing:[/bold]"""
        
        if session_info.started_at:
            info_content += f"\n  Started: {session_info.started_at.strftime('%Y-%m-%d %H:%M:%S')}"
        
        if session_info.paused_at:
            info_content += f"\n  Paused: {session_info.paused_at.strftime('%Y-%m-%d %H:%M:%S')}"
            
        if session_info.completed_at:
            info_content += f"\n  Completed: {session_info.completed_at.strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Calculate duration
        if session_info.started_at:
            if session_info.completed_at:
                duration = session_info.completed_at - session_info.started_at
            elif session_info.paused_at:
                duration = session_info.paused_at - session_info.started_at
            else:
                duration = datetime.now() - session_info.started_at
            info_content += f"\n  Duration: {str(duration).split('.')[0]}"
        
        console.print(Panel(info_content, title=f"📊 Session {session_id} Details", border_style=status_color))
        
        # Show session state if available
        if hasattr(session_info, 'session_state') and session_info.session_state:
            try:
                state_data = json.loads(session_info.session_state) if isinstance(session_info.session_state, str) else session_info.session_state
                
                console.print("\n[bold]Session State:[/bold]")
                
                if isinstance(state_data, dict):
                    for key, value in state_data.items():
                        if key not in ['questions', 'responses']:  # Skip large data structures
                            console.print(f"  {key}: {value}")
                            
            except (json.JSONDecodeError, TypeError):
                pass
        
    except Exception as e:
        console.print(f"[red]Error retrieving session status: {str(e)}[/red]")
        logger.exception("Session status retrieval failed")


@session.command('cleanup')
@click.option('--older-than', type=int, default=30, help='Remove sessions older than N days')
@click.option('--status', type=click.Choice(['completed', 'failed']), 
              help='Only cleanup sessions with this status')
@click.option('--dry-run', is_flag=True, help='Show what would be removed without actually removing')
@click.pass_context
def cleanup_sessions(ctx, older_than, status, dry_run):
    """Clean up old benchmark sessions.
    
    \b
    🧹 EXAMPLES:
    
    alex session cleanup
    alex session cleanup --older-than 7
    alex session cleanup --status completed --dry-run
    
    \b
    💡 Removes old benchmark sessions to free up database space.
    Use --dry-run to preview what would be removed.
    """
    try:
        from core.session import SessionManager
        
        session_manager = SessionManager()
        
        # Find sessions to cleanup
        sessions_to_cleanup = session_manager.find_sessions_for_cleanup(
            older_than_days=older_than,
            status_filter=status
        )
        
        if not sessions_to_cleanup:
            console.print("[green]No sessions found for cleanup[/green]")
            return
        
        console.print(f"[yellow]Found {len(sessions_to_cleanup)} sessions for cleanup:[/yellow]")
        
        # Show what would be cleaned up
        cleanup_table = Table(title="Sessions to Cleanup")
        cleanup_table.add_column("ID", justify="right", style="cyan")
        cleanup_table.add_column("Name", style="magenta")
        cleanup_table.add_column("Status", justify="center")
        cleanup_table.add_column("Age (days)", justify="right", style="yellow")
        
        for session_info in sessions_to_cleanup:
            age_days = (datetime.now() - session_info.created_at).days if session_info.created_at else 0
            
            cleanup_table.add_row(
                str(session_info.id),
                session_info.name,
                session_info.status,
                str(age_days)
            )
        
        console.print(cleanup_table)
        
        if dry_run:
            console.print(f"\n[blue]Dry run: Would remove {len(sessions_to_cleanup)} sessions[/blue]")
            return
        
        # Confirm cleanup
        if not click.confirm(f"Remove {len(sessions_to_cleanup)} sessions?"):
            console.print("[yellow]Cleanup cancelled[/yellow]")
            return
        
        # Perform cleanup
        removed_count = session_manager.cleanup_sessions(sessions_to_cleanup)
        
        console.print(f"[green]✓ Cleaned up {removed_count} sessions[/green]")
        
    except Exception as e:
        console.print(f"[red]Error during session cleanup: {str(e)}[/red]")
        logger.exception("Session cleanup failed")