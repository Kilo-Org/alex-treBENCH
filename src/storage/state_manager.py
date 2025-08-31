"""
State Manager

Manages persistent state for benchmarks including progress tracking,
partial results, checkpoints, and cleanup of old state files.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging
import hashlib

from core.config import get_config
from core.exceptions import StateError
from storage.models import BenchmarkRun, BenchmarkResult, ModelPerformance

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """Benchmark checkpoint data."""
    checkpoint_id: str
    benchmark_run_id: int
    timestamp: datetime
    progress_data: Dict[str, Any]
    partial_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def age_days(self) -> float:
        """Get age of checkpoint in days."""
        return (datetime.now() - self.timestamp).total_seconds() / (24 * 3600)


@dataclass
class StateMetadata:
    """State file metadata."""
    state_id: str
    benchmark_run_id: int
    created_at: datetime
    updated_at: datetime
    checkpoints: List[str] = field(default_factory=list)
    total_size_bytes: int = 0
    compression_enabled: bool = False


class StateManager:
    """Manages persistent state for benchmark runs."""

    def __init__(self, state_dir: Optional[Path] = None):
        self.config = get_config()
        self.state_dir = state_dir or Path("state")
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.checkpoints_dir = self.state_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.results_dir = self.state_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, benchmark_run_id: int, progress_data: Dict[str, Any],
                       partial_results: Optional[List[Dict[str, Any]]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a checkpoint for the benchmark run.

        Args:
            benchmark_run_id: Benchmark run ID
            progress_data: Current progress data
            partial_results: Partial results to save
            metadata: Additional metadata

        Returns:
            Checkpoint ID
        """
        timestamp = datetime.now()
        checkpoint_id = f"checkpoint_{benchmark_run_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            benchmark_run_id=benchmark_run_id,
            timestamp=timestamp,
            progress_data=progress_data,
            partial_results=partial_results or [],
            metadata=metadata or {}
        )

        try:
            # Save checkpoint to file
            checkpoint_file = self.checkpoints_dir / f"{checkpoint_id}.json"
            checkpoint_data = {
                "checkpoint_id": checkpoint.checkpoint_id,
                "benchmark_run_id": checkpoint.benchmark_run_id,
                "timestamp": checkpoint.timestamp.isoformat(),
                "progress_data": checkpoint.progress_data,
                "partial_results": checkpoint.partial_results,
                "metadata": checkpoint.metadata
            }

            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)

            # Update state metadata
            self._update_state_metadata(benchmark_run_id, checkpoint_id)

            logger.info(f"Saved checkpoint: {checkpoint_id}")
            return checkpoint_id

        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint_id}: {e}")
            raise StateError(f"Checkpoint save failed: {e}")

    def load_checkpoint(self, checkpoint_id: str) -> Checkpoint:
        """
        Load a checkpoint from disk.

        Args:
            checkpoint_id: Checkpoint ID

        Returns:
            Checkpoint object
        """
        checkpoint_file = self.checkpoints_dir / f"{checkpoint_id}.json"

        if not checkpoint_file.exists():
            raise StateError(f"Checkpoint not found: {checkpoint_id}")

        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)

            checkpoint = Checkpoint(
                checkpoint_id=data["checkpoint_id"],
                benchmark_run_id=data["benchmark_run_id"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                progress_data=data["progress_data"],
                partial_results=data.get("partial_results", []),
                metadata=data.get("metadata", {})
            )

            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            raise StateError(f"Checkpoint load failed: {e}")

    def get_latest_checkpoint(self, benchmark_run_id: int) -> Optional[Checkpoint]:
        """
        Get the latest checkpoint for a benchmark run.

        Args:
            benchmark_run_id: Benchmark run ID

        Returns:
            Latest checkpoint or None
        """
        checkpoints = self.list_checkpoints(benchmark_run_id)
        if not checkpoints:
            return None

        # Get most recent checkpoint
        latest = max(checkpoints, key=lambda c: c.timestamp)
        return self.load_checkpoint(latest.checkpoint_id)

    def list_checkpoints(self, benchmark_run_id: int) -> List[Checkpoint]:
        """
        List all checkpoints for a benchmark run.

        Args:
            benchmark_run_id: Benchmark run ID

        Returns:
            List of checkpoints
        """
        checkpoints = []

        for checkpoint_file in self.checkpoints_dir.glob(f"checkpoint_{benchmark_run_id}_*.json"):
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)

                checkpoint = Checkpoint(
                    checkpoint_id=data["checkpoint_id"],
                    benchmark_run_id=data["benchmark_run_id"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    progress_data=data["progress_data"],
                    partial_results=data.get("partial_results", []),
                    metadata=data.get("metadata", {})
                )

                checkpoints.append(checkpoint)

            except Exception as e:
                logger.warning(f"Failed to read checkpoint file {checkpoint_file}: {e}")

        return sorted(checkpoints, key=lambda c: c.timestamp, reverse=True)

    def save_partial_results(self, benchmark_run_id: int, results: List[Dict[str, Any]]) -> str:
        """
        Save partial results to disk.

        Args:
            benchmark_run_id: Benchmark run ID
            results: Partial results

        Returns:
            Results file ID
        """
        timestamp = datetime.now()
        results_id = f"results_{benchmark_run_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

        try:
            results_file = self.results_dir / f"{results_id}.json"
            results_data = {
                "results_id": results_id,
                "benchmark_run_id": benchmark_run_id,
                "timestamp": timestamp.isoformat(),
                "results": results
            }

            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)

            logger.info(f"Saved partial results: {results_id}")
            return results_id

        except Exception as e:
            logger.error(f"Failed to save partial results {results_id}: {e}")
            raise StateError(f"Partial results save failed: {e}")

    def load_partial_results(self, results_id: str) -> List[Dict[str, Any]]:
        """
        Load partial results from disk.

        Args:
            results_id: Results file ID

        Returns:
            Partial results
        """
        results_file = self.results_dir / f"{results_id}.json"

        if not results_file.exists():
            raise StateError(f"Partial results not found: {results_id}")

        try:
            with open(results_file, 'r') as f:
                data = json.load(f)

            return data["results"]

        except Exception as e:
            logger.error(f"Failed to load partial results {results_id}: {e}")
            raise StateError(f"Partial results load failed: {e}")

    def restore_from_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Restore benchmark state from checkpoint.

        Args:
            checkpoint_id: Checkpoint ID

        Returns:
            Restored state data
        """
        checkpoint = self.load_checkpoint(checkpoint_id)

        restored_state = {
            "benchmark_run_id": checkpoint.benchmark_run_id,
            "progress_data": checkpoint.progress_data,
            "partial_results": checkpoint.partial_results,
            "metadata": checkpoint.metadata,
            "checkpoint_timestamp": checkpoint.timestamp.isoformat()
        }

        logger.info(f"Restored state from checkpoint: {checkpoint_id}")
        return restored_state

    def cleanup_old_checkpoints(self, benchmark_run_id: int, keep_last: int = 5) -> int:
        """
        Clean up old checkpoints for a benchmark run.

        Args:
            benchmark_run_id: Benchmark run ID
            keep_last: Number of recent checkpoints to keep

        Returns:
            Number of checkpoints cleaned up
        """
        checkpoints = self.list_checkpoints(benchmark_run_id)
        cleaned_count = 0

        if len(checkpoints) > keep_last:
            # Sort by timestamp (oldest first)
            checkpoints.sort(key=lambda c: c.timestamp)

            # Remove old checkpoints
            for checkpoint in checkpoints[:-keep_last]:
                try:
                    checkpoint_file = self.checkpoints_dir / f"{checkpoint.checkpoint_id}.json"
                    if checkpoint_file.exists():
                        checkpoint_file.unlink()
                        cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {checkpoint.checkpoint_id}: {e}")

        logger.info(f"Cleaned up {cleaned_count} old checkpoints for benchmark {benchmark_run_id}")
        return cleaned_count

    def cleanup_old_states(self, days: int = 30) -> int:
        """
        Clean up old state files.

        Args:
            days: Remove states older than this many days

        Returns:
            Number of state files cleaned up
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        cleaned_count = 0

        # Clean up old checkpoints
        for checkpoint_file in self.checkpoints_dir.glob("*.json"):
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)

                timestamp = datetime.fromisoformat(data["timestamp"])
                if timestamp < cutoff_date:
                    checkpoint_file.unlink()
                    cleaned_count += 1

            except Exception as e:
                logger.warning(f"Failed to process checkpoint file {checkpoint_file}: {e}")

        # Clean up old results
        for results_file in self.results_dir.glob("*.json"):
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)

                timestamp = datetime.fromisoformat(data["timestamp"])
                if timestamp < cutoff_date:
                    results_file.unlink()
                    cleaned_count += 1

            except Exception as e:
                logger.warning(f"Failed to process results file {results_file}: {e}")

        logger.info(f"Cleaned up {cleaned_count} old state files")
        return cleaned_count

    def get_state_stats(self, benchmark_run_id: int) -> Dict[str, Any]:
        """
        Get statistics for benchmark state.

        Args:
            benchmark_run_id: Benchmark run ID

        Returns:
            State statistics
        """
        checkpoints = self.list_checkpoints(benchmark_run_id)

        if not checkpoints:
            return {"checkpoints_count": 0, "total_size": 0, "oldest_checkpoint": None, "newest_checkpoint": None}

        total_size = 0
        for checkpoint in checkpoints:
            checkpoint_file = self.checkpoints_dir / f"{checkpoint.checkpoint_id}.json"
            if checkpoint_file.exists():
                total_size += checkpoint_file.stat().st_size

        return {
            "checkpoints_count": len(checkpoints),
            "total_size": total_size,
            "oldest_checkpoint": min(checkpoints, key=lambda c: c.timestamp).timestamp.isoformat(),
            "newest_checkpoint": max(checkpoints, key=lambda c: c.timestamp).timestamp.isoformat()
        }

    def _update_state_metadata(self, benchmark_run_id: int, checkpoint_id: str) -> None:
        """Update state metadata file."""
        try:
            metadata_file = self.state_dir / f"state_{benchmark_run_id}.json"

            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    state_meta = StateMetadata(**metadata)
            else:
                state_meta = StateMetadata(
                    state_id=f"state_{benchmark_run_id}",
                    benchmark_run_id=benchmark_run_id,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )

            if checkpoint_id not in state_meta.checkpoints:
                state_meta.checkpoints.append(checkpoint_id)

            state_meta.updated_at = datetime.now()

            # Calculate total size
            total_size = 0
            for cp_id in state_meta.checkpoints:
                cp_file = self.checkpoints_dir / f"{cp_id}.json"
                if cp_file.exists():
                    total_size += cp_file.stat().st_size
            state_meta.total_size_bytes = total_size

            with open(metadata_file, 'w') as f:
                json.dump({
                    "state_id": state_meta.state_id,
                    "benchmark_run_id": state_meta.benchmark_run_id,
                    "created_at": state_meta.created_at.isoformat(),
                    "updated_at": state_meta.updated_at.isoformat(),
                    "checkpoints": state_meta.checkpoints,
                    "total_size_bytes": state_meta.total_size_bytes,
                    "compression_enabled": state_meta.compression_enabled
                }, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to update state metadata for {benchmark_run_id}: {e}")


# Global state manager instance
_state_manager = None

def get_state_manager() -> StateManager:
    """Get global state manager instance."""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager


# CLI Helper Functions
def cli_list_checkpoints(benchmark_run_id: int) -> None:
    """CLI command to list checkpoints."""
    state_manager = get_state_manager()
    checkpoints = state_manager.list_checkpoints(benchmark_run_id)

    if not checkpoints:
        print(f"No checkpoints found for benchmark {benchmark_run_id}")
        return

    print(f"Checkpoints for benchmark {benchmark_run_id}:")
    for checkpoint in checkpoints:
        age = checkpoint.age_days
        print(f"  {checkpoint.checkpoint_id}: {checkpoint.timestamp.strftime('%Y-%m-%d %H:%M:%S')} ({age:.1f} days old)")


def cli_save_checkpoint(benchmark_run_id: int, progress_data: Dict[str, Any]) -> None:
    """CLI command to save checkpoint."""
    state_manager = get_state_manager()
    checkpoint_id = state_manager.save_checkpoint(benchmark_run_id, progress_data)
    print(f"Saved checkpoint: {checkpoint_id}")


def cli_restore_checkpoint(checkpoint_id: str) -> None:
    """CLI command to restore from checkpoint."""
    state_manager = get_state_manager()
    state_data = state_manager.restore_from_checkpoint(checkpoint_id)
    print(f"Restored state from checkpoint: {checkpoint_id}")
    print(f"Benchmark Run ID: {state_data['benchmark_run_id']}")


def cli_cleanup_checkpoints(benchmark_run_id: int, keep_last: int = 5) -> None:
    """CLI command to cleanup old checkpoints."""
    state_manager = get_state_manager()
    cleaned = state_manager.cleanup_old_checkpoints(benchmark_run_id, keep_last)
    print(f"Cleaned up {cleaned} old checkpoints, kept last {keep_last}")


def cli_state_stats(benchmark_run_id: int) -> None:
    """CLI command to show state statistics."""
    state_manager = get_state_manager()
    stats = state_manager.get_state_stats(benchmark_run_id)
    print(f"State statistics for benchmark {benchmark_run_id}:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def cli_cleanup_old_states(days: int = 30) -> None:
    """CLI command to cleanup old state files."""
    state_manager = get_state_manager()
    cleaned = state_manager.cleanup_old_states(days)
    print(f"Cleaned up {cleaned} state files older than {days} days")