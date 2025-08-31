"""
Benchmark Session Management

Manages benchmark execution sessions with pause/resume functionality,
state persistence, and recovery from interruptions.
"""

import json
import signal
import atexit
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading

from ..core.config import get_config
from ..core.exceptions import SessionError
from ..storage.models import BenchmarkRun, BenchmarkResult

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Benchmark session states."""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass
class SessionProgress:
    """Session progress tracking."""
    total_questions: int = 0
    completed_questions: int = 0
    current_model: Optional[str] = None
    current_question_index: int = 0
    start_time: Optional[datetime] = None
    pause_time: Optional[datetime] = None
    resume_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None

    @property
    def progress_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_questions == 0:
            return 0.0
        return (self.completed_questions / self.total_questions) * 100

    @property
    def elapsed_time(self) -> Optional[timedelta]:
        """Calculate elapsed time."""
        if self.start_time is None:
            return None

        end_time = self.pause_time or datetime.now()
        return end_time - self.start_time

    @property
    def remaining_time(self) -> Optional[timedelta]:
        """Estimate remaining time."""
        if self.elapsed_time is None or self.completed_questions == 0:
            return None

        avg_time_per_question = self.elapsed_time / self.completed_questions
        remaining_questions = self.total_questions - self.completed_questions
        return avg_time_per_question * remaining_questions


@dataclass
class SessionMetadata:
    """Session metadata."""
    session_id: str
    benchmark_run_id: int
    created_at: datetime
    updated_at: datetime
    state: SessionState
    config_snapshot: Dict[str, Any]
    error_message: Optional[str] = None
    retry_count: int = 0


class BenchmarkSession:
    """Manages benchmark execution session with persistence."""

    def __init__(self, benchmark_run_id: int, session_dir: Optional[Path] = None):
        self.benchmark_run_id = benchmark_run_id
        self.config = get_config()

        # Session directories
        self.session_dir = session_dir or Path("sessions")
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Session state
        self.session_id = f"session_{benchmark_run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.state = SessionState.CREATED
        self.progress = SessionProgress()
        self.metadata = SessionMetadata(
            session_id=self.session_id,
            benchmark_run_id=benchmark_run_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            state=self.state,
            config_snapshot=self._create_config_snapshot()
        )

        # Callbacks
        self.on_pause: Optional[Callable] = None
        self.on_resume: Optional[Callable] = None
        self.on_complete: Optional[Callable] = None
        self.on_error: Optional[Callable] = None

        # Threading
        self._running = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # Start unpaused

        # Signal handlers
        self._setup_signal_handlers()

        # Auto-save
        self._auto_save_enabled = True
        self._last_save = datetime.now()

        # Save initial state
        self._save_session()

        logger.info(f"Created benchmark session: {self.session_id}")

    def start(self) -> None:
        """Start the benchmark session."""
        if self.state != SessionState.CREATED and self.state != SessionState.PAUSED:
            raise SessionError(f"Cannot start session in state: {self.state}")

        self.state = SessionState.RUNNING
        self.progress.start_time = datetime.now()
        self.metadata.updated_at = datetime.now()
        self._running = True

        self._save_session()

        logger.info(f"Started benchmark session: {self.session_id}")

    def pause(self) -> None:
        """Pause the benchmark session."""
        if self.state != SessionState.RUNNING:
            raise SessionError(f"Cannot pause session in state: {self.state}")

        self.state = SessionState.PAUSED
        self.progress.pause_time = datetime.now()
        self.metadata.updated_at = datetime.now()
        self._pause_event.clear()  # Block execution

        self._save_session()

        if self.on_pause:
            self.on_pause()

        logger.info(f"Paused benchmark session: {self.session_id}")

    def resume(self) -> None:
        """Resume the benchmark session."""
        if self.state != SessionState.PAUSED:
            raise SessionError(f"Cannot resume session in state: {self.state}")

        self.state = SessionState.RUNNING
        self.progress.resume_time = datetime.now()
        self.metadata.updated_at = datetime.now()
        self._pause_event.set()  # Allow execution

        self._save_session()

        if self.on_resume:
            self.on_resume()

        logger.info(f"Resumed benchmark session: {self.session_id}")

    def complete(self) -> None:
        """Mark session as completed."""
        self.state = SessionState.COMPLETED
        self.metadata.updated_at = datetime.now()
        self._running = False

        self._save_session()

        if self.on_complete:
            self.on_complete()

        logger.info(f"Completed benchmark session: {self.session_id}")

    def fail(self, error_message: str) -> None:
        """Mark session as failed."""
        self.state = SessionState.FAILED
        self.metadata.error_message = error_message
        self.metadata.updated_at = datetime.now()
        self._running = False

        self._save_session()

        if self.on_error:
            self.on_error(error_message)

        logger.error(f"Failed benchmark session: {self.session_id} - {error_message}")

    def interrupt(self) -> None:
        """Handle interruption (e.g., SIGINT)."""
        if self.state == SessionState.RUNNING:
            self.state = SessionState.INTERRUPTED
            self.metadata.updated_at = datetime.now()
            self._running = False

            self._save_session()

            logger.warning(f"Interrupted benchmark session: {self.session_id}")

    def update_progress(self, completed_questions: int, current_model: Optional[str] = None,
                       current_question_index: int = 0) -> None:
        """Update session progress."""
        self.progress.completed_questions = completed_questions
        self.progress.current_model = current_model
        self.progress.current_question_index = current_question_index
        self.metadata.updated_at = datetime.now()

        # Auto-save periodically
        if self._auto_save_enabled and (datetime.now() - self._last_save) > timedelta(minutes=5):
            self._save_session()

    def wait_if_paused(self) -> None:
        """Wait if session is paused."""
        self._pause_event.wait()

    def is_running(self) -> bool:
        """Check if session is running."""
        return self._running and self.state == SessionState.RUNNING

    def can_resume(self) -> bool:
        """Check if session can be resumed."""
        return self.state in [SessionState.PAUSED, SessionState.INTERRUPTED]

    def get_status(self) -> Dict[str, Any]:
        """Get session status."""
        return {
            "session_id": self.session_id,
            "benchmark_run_id": self.benchmark_run_id,
            "state": self.state.value,
            "progress": {
                "percentage": self.progress.progress_percentage,
                "completed": self.progress.completed_questions,
                "total": self.progress.total_questions,
                "current_model": self.progress.current_model,
                "current_question": self.progress.current_question_index,
                "elapsed_time": str(self.progress.elapsed_time) if self.progress.elapsed_time else None,
                "remaining_time": str(self.progress.remaining_time) if self.progress.remaining_time else None
            },
            "created_at": self.metadata.created_at.isoformat(),
            "updated_at": self.metadata.updated_at.isoformat(),
            "error_message": self.metadata.error_message
        }

    def _create_config_snapshot(self) -> Dict[str, Any]:
        """Create snapshot of current configuration."""
        config_dict = {}
        for key, value in self.config.__dict__.items():
            if not key.startswith('_'):
                try:
                    config_dict[key] = value
                except:
                    config_dict[key] = str(value)
        return config_dict

    def _save_session(self) -> None:
        """Save session state to disk."""
        try:
            session_file = self.session_dir / f"{self.session_id}.json"

            session_data = {
                "metadata": {
                    "session_id": self.metadata.session_id,
                    "benchmark_run_id": self.metadata.benchmark_run_id,
                    "created_at": self.metadata.created_at.isoformat(),
                    "updated_at": self.metadata.updated_at.isoformat(),
                    "state": self.metadata.state.value,
                    "config_snapshot": self.metadata.config_snapshot,
                    "error_message": self.metadata.error_message,
                    "retry_count": self.metadata.retry_count
                },
                "progress": {
                    "total_questions": self.progress.total_questions,
                    "completed_questions": self.progress.completed_questions,
                    "current_model": self.progress.current_model,
                    "current_question_index": self.progress.current_question_index,
                    "start_time": self.progress.start_time.isoformat() if self.progress.start_time else None,
                    "pause_time": self.progress.pause_time.isoformat() if self.progress.pause_time else None,
                    "resume_time": self.progress.resume_time.isoformat() if self.progress.resume_time else None,
                    "estimated_completion": self.progress.estimated_completion.isoformat() if self.progress.estimated_completion else None
                }
            }

            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)

            self._last_save = datetime.now()

        except Exception as e:
            logger.error(f"Failed to save session {self.session_id}: {e}")

    def _load_session(self, session_file: Path) -> None:
        """Load session state from disk."""
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)

            # Load metadata
            meta = session_data["metadata"]
            self.metadata = SessionMetadata(
                session_id=meta["session_id"],
                benchmark_run_id=meta["benchmark_run_id"],
                created_at=datetime.fromisoformat(meta["created_at"]),
                updated_at=datetime.fromisoformat(meta["updated_at"]),
                state=SessionState(meta["state"]),
                config_snapshot=meta["config_snapshot"],
                error_message=meta.get("error_message"),
                retry_count=meta.get("retry_count", 0)
            )

            # Load progress
            prog = session_data["progress"]
            self.progress = SessionProgress(
                total_questions=prog["total_questions"],
                completed_questions=prog["completed_questions"],
                current_model=prog["current_model"],
                current_question_index=prog["current_question_index"],
                start_time=datetime.fromisoformat(prog["start_time"]) if prog.get("start_time") else None,
                pause_time=datetime.fromisoformat(prog["pause_time"]) if prog.get("pause_time") else None,
                resume_time=datetime.fromisoformat(prog["resume_time"]) if prog.get("resume_time") else None,
                estimated_completion=datetime.fromisoformat(prog["estimated_completion"]) if prog.get("estimated_completion") else None
            )

            self.state = self.metadata.state

        except Exception as e:
            logger.error(f"Failed to load session from {session_file}: {e}")
            raise SessionError(f"Session loading failed: {e}")

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, interrupting session")
            self.interrupt()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Register cleanup on exit
        atexit.register(self._cleanup)

    def _cleanup(self) -> None:
        """Cleanup session on exit."""
        if self.state == SessionState.RUNNING:
            self.interrupt()
        self._save_session()


def create_session(benchmark_run_id: int) -> BenchmarkSession:
    """Create a new benchmark session."""
    return BenchmarkSession(benchmark_run_id)


def load_session(session_id: str, session_dir: Optional[Path] = None) -> BenchmarkSession:
    """Load existing session from disk."""
    session_dir = session_dir or Path("sessions")
    session_file = session_dir / f"{session_id}.json"

    if not session_file.exists():
        raise SessionError(f"Session file not found: {session_file}")

    # Create session instance
    session = BenchmarkSession(0)  # Dummy ID, will be overwritten
    session._load_session(session_file)

    return session


def list_sessions(session_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """List all available sessions."""
    session_dir = session_dir or Path("sessions")
    sessions = []

    if not session_dir.exists():
        return sessions

    for session_file in session_dir.glob("*.json"):
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)

            meta = session_data["metadata"]
            prog = session_data["progress"]

            sessions.append({
                "session_id": meta["session_id"],
                "benchmark_run_id": meta["benchmark_run_id"],
                "state": meta["state"],
                "created_at": meta["created_at"],
                "progress_percentage": (prog["completed_questions"] / prog["total_questions"] * 100) if prog["total_questions"] > 0 else 0,
                "current_model": prog["current_model"]
            })

        except Exception as e:
            logger.warning(f"Failed to read session file {session_file}: {e}")

    return sorted(sessions, key=lambda x: x["created_at"], reverse=True)


def cleanup_old_sessions(days: int = 30, session_dir: Optional[Path] = None) -> int:
    """Clean up old completed sessions."""
    session_dir = session_dir or Path("sessions")
    cutoff_date = datetime.now() - timedelta(days=days)
    cleaned_count = 0

    if not session_dir.exists():
        return 0

    for session_file in session_dir.glob("*.json"):
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)

            meta = session_data["metadata"]
            state = meta["state"]
            created_at = datetime.fromisoformat(meta["created_at"])

            # Remove completed sessions older than cutoff
            if state in ["completed", "failed"] and created_at < cutoff_date:
                session_file.unlink()
                cleaned_count += 1

        except Exception as e:
            logger.warning(f"Failed to process session file {session_file}: {e}")

    logger.info(f"Cleaned up {cleaned_count} old sessions")
    return cleaned_count


# CLI Helper Functions
def cli_list_sessions() -> None:
    """CLI command to list sessions."""
    sessions = list_sessions()
    if not sessions:
        print("No sessions found")
        return

    print("Available Sessions:")
    for session in sessions:
        print(f"  {session['session_id']}: {session['state']} ({session['progress_percentage']:.1f}%) - {session['current_model'] or 'No model'}")


def cli_session_status(session_id: str) -> None:
    """CLI command to show session status."""
    try:
        session = load_session(session_id)
        status = session.get_status()

        print(f"Session: {status['session_id']}")
        print(f"State: {status['state']}")
        print(f"Progress: {status['progress']['percentage']:.1f}%")
        print(f"Completed: {status['progress']['completed']}/{status['progress']['total']}")
        print(f"Current Model: {status['progress']['current_model'] or 'None'}")
        print(f"Elapsed Time: {status['progress']['elapsed_time'] or 'N/A'}")
        print(f"Remaining Time: {status['progress']['remaining_time'] or 'N/A'}")

    except SessionError as e:
        print(f"Error: {e}")


def cli_pause_session(session_id: str) -> None:
    """CLI command to pause session."""
    try:
        session = load_session(session_id)
        session.pause()
        print(f"Paused session: {session_id}")
    except SessionError as e:
        print(f"Error: {e}")


def cli_resume_session(session_id: str) -> None:
    """CLI command to resume session."""
    try:
        session = load_session(session_id)
        session.resume()
        print(f"Resumed session: {session_id}")
    except SessionError as e:
        print(f"Error: {e}")


def cli_cleanup_sessions(days: int = 30) -> None:
    """CLI command to cleanup old sessions."""
    cleaned = cleanup_old_sessions(days)
    print(f"Cleaned up {cleaned} sessions older than {days} days")