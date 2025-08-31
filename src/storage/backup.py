"""
Database Backup System

Handles database backups, exports to various formats (JSON, CSV),
imports from backup files, and manages backup rotation.
"""

import os
import json
import csv
import gzip
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

from .models import BenchmarkRun, BenchmarkResult, ModelPerformance, Question
from src.core.database import get_db_url, get_db_session
from src.core.config import get_config
from src.core.exceptions import DatabaseError

logger = logging.getLogger(__name__)


@dataclass
class BackupMetadata:
    """Metadata for backup files."""
    timestamp: datetime
    version: str
    record_counts: Dict[str, int]
    tables: List[str]
    format: str
    compressed: bool


class DatabaseBackup:
    """Database backup and export manager."""

    def __init__(self, backup_dir: Optional[Path] = None):
        self.config = get_config()
        self.backup_dir = backup_dir or Path(self.config.database.backup.path)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.max_backups = self.config.database.backup.rotation

    def create_backup(self, name: Optional[str] = None, compress: bool = True) -> Path:
        """
        Create a full database backup.

        Args:
            name: Optional backup name
            compress: Whether to compress the backup

        Returns:
            Path to the created backup file
        """
        timestamp = datetime.now()
        backup_name = name or f"backup_{timestamp.strftime('%Y%m%d_%H%M%S')}"

        try:
            # Create backup data
            backup_data = self._export_all_tables()

            # Create metadata
            metadata = BackupMetadata(
                timestamp=timestamp,
                version=self.config.app.version,
                record_counts={table: len(data) for table, data in backup_data.items()},
                tables=list(backup_data.keys()),
                format="json",
                compressed=compress
            )

            # Save backup
            backup_path = self._save_backup(backup_data, metadata, backup_name, compress)

            # Rotate backups
            self._rotate_backups()

            logger.info(f"Backup created: {backup_path}")
            return backup_path

        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise DatabaseError(f"Backup creation failed: {e}")

    def restore_backup(self, backup_path: Path, drop_existing: bool = False) -> None:
        """
        Restore database from backup.

        Args:
            backup_path: Path to backup file
            drop_existing: Whether to drop existing data first
        """
        if not backup_path.exists():
            raise DatabaseError(f"Backup file not found: {backup_path}")

        try:
            # Load backup data
            backup_data, metadata = self._load_backup(backup_path)

            # Validate backup
            self._validate_backup(metadata)

            # Restore data
            self._restore_data(backup_data, drop_existing)

            logger.info(f"Backup restored from: {backup_path}")

        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            raise DatabaseError(f"Backup restoration failed: {e}")

    def export_to_csv(self, output_dir: Path, tables: Optional[List[str]] = None) -> List[Path]:
        """
        Export database tables to CSV files.

        Args:
            output_dir: Directory to save CSV files
            tables: List of tables to export (None for all)

        Returns:
            List of created CSV file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        exported_files = []

        try:
            data = self._export_all_tables()

            for table_name, records in data.items():
                if tables and table_name not in tables:
                    continue

                csv_path = output_dir / f"{table_name}.csv"
                self._export_table_to_csv(records, csv_path)
                exported_files.append(csv_path)
                logger.info(f"Exported {table_name} to {csv_path}")

            return exported_files

        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
            raise DatabaseError(f"CSV export failed: {e}")

    def export_to_json(self, output_file: Path, tables: Optional[List[str]] = None) -> Path:
        """
        Export database tables to a single JSON file.

        Args:
            output_file: Path for the JSON file
            tables: List of tables to export (None for all)

        Returns:
            Path to the created JSON file
        """
        try:
            data = self._export_all_tables()

            if tables:
                data = {table: records for table, records in data.items() if table in tables}

            # Add metadata
            export_data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "version": self.config.app.version,
                    "tables": list(data.keys()),
                    "record_counts": {table: len(records) for table, records in data.items()}
                },
                "data": data
            }

            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Exported to JSON: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Failed to export to JSON: {e}")
            raise DatabaseError(f"JSON export failed: {e}")

    def import_from_json(self, json_file: Path) -> None:
        """
        Import data from JSON file.

        Args:
            json_file: Path to JSON file
        """
        if not json_file.exists():
            raise DatabaseError(f"JSON file not found: {json_file}")

        try:
            with open(json_file, 'r') as f:
                import_data = json.load(f)

            if "data" not in import_data:
                raise DatabaseError("Invalid JSON format: missing 'data' key")

            self._restore_data(import_data["data"], drop_existing=False)
            logger.info(f"Imported from JSON: {json_file}")

        except Exception as e:
            logger.error(f"Failed to import from JSON: {e}")
            raise DatabaseError(f"JSON import failed: {e}")

    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups with metadata."""
        backups = []

        for backup_file in self.backup_dir.glob("*.json*"):
            try:
                if backup_file.suffix == ".gz":
                    base_name = backup_file.stem
                else:
                    base_name = backup_file.name

                # Try to load metadata
                metadata = None
                if backup_file.exists():
                    try:
                        _, metadata = self._load_backup(backup_file)
                    except:
                        pass

                backups.append({
                    "name": base_name,
                    "path": backup_file,
                    "size": backup_file.stat().st_size,
                    "created": datetime.fromtimestamp(backup_file.stat().st_ctime),
                    "metadata": metadata
                })

            except Exception as e:
                logger.warning(f"Failed to read backup metadata for {backup_file}: {e}")

        return sorted(backups, key=lambda x: x["created"], reverse=True)

    def _export_all_tables(self) -> Dict[str, List[Dict[str, Any]]]:
        """Export all tables to dictionary format."""
        with get_db_session() as session:
            try:
                data = {}

                # Export questions
                questions = session.query(Question).all()
                data["questions"] = [self._model_to_dict(q) for q in questions]

            # Export benchmark runs
            runs = session.query(BenchmarkRun).all()
            data["benchmark_runs"] = [self._model_to_dict(r) for r in runs]

            # Export results
            results = session.query(BenchmarkResult).all()
            data["benchmark_results"] = [self._model_to_dict(r) for r in results]

            # Export performances
            performances = session.query(ModelPerformance).all()
            data["model_performance"] = [self._model_to_dict(p) for p in performances]

            return data

        finally:
            session.close()

    def _model_to_dict(self, model) -> Dict[str, Any]:
        """Convert SQLAlchemy model to dictionary."""
        result = {}
        for column in model.__table__.columns:
            value = getattr(model, column.name)
            if isinstance(value, datetime):
                result[column.name] = value.isoformat()
            else:
                result[column.name] = value
        return result

    def _save_backup(self, data: Dict[str, Any], metadata: BackupMetadata,
                    name: str, compress: bool) -> Path:
        """Save backup data to file."""
        backup_data = {
            "metadata": {
                "timestamp": metadata.timestamp.isoformat(),
                "version": metadata.version,
                "record_counts": metadata.record_counts,
                "tables": metadata.tables,
                "format": metadata.format,
                "compressed": metadata.compressed
            },
            "data": data
        }

        if compress:
            backup_path = self.backup_dir / f"{name}.json.gz"
            with gzip.open(backup_path, 'wt', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, default=str)
        else:
            backup_path = self.backup_dir / f"{name}.json"
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)

        return backup_path

    def _load_backup(self, backup_path: Path) -> tuple:
        """Load backup data from file."""
        if backup_path.suffix == ".gz":
            with gzip.open(backup_path, 'rt', encoding='utf-8') as f:
                backup_data = json.load(f)
        else:
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)

        # Parse metadata
        meta = backup_data["metadata"]
        metadata = BackupMetadata(
            timestamp=datetime.fromisoformat(meta["timestamp"]),
            version=meta["version"],
            record_counts=meta["record_counts"],
            tables=meta["tables"],
            format=meta["format"],
            compressed=meta.get("compressed", False)
        )

        return backup_data["data"], metadata

    def _validate_backup(self, metadata: BackupMetadata) -> None:
        """Validate backup metadata."""
        if metadata.version != self.config.app.version:
            logger.warning(f"Backup version mismatch: {metadata.version} vs {self.config.app.version}")

        if not metadata.tables:
            raise DatabaseError("Backup contains no tables")

    def _restore_data(self, data: Dict[str, List[Dict[str, Any]]], drop_existing: bool) -> None:
        """Restore data to database."""
        with get_db_session() as session:
            try:
                # Clear existing data if requested
                if drop_existing:
                    self._clear_all_tables(session)

                # Restore in order (respecting foreign keys)
                if "questions" in data:
                    self._restore_questions(session, data["questions"])

            if "benchmark_runs" in data:
                self._restore_benchmark_runs(session, data["benchmark_runs"])

            if "benchmark_results" in data:
                self._restore_results(session, data["benchmark_results"])

            if "model_performance" in data:
                self._restore_performances(session, data["model_performance"])

            session.commit()

        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def _clear_all_tables(self, session: Session) -> None:
        """Clear all tables."""
        session.query(ModelPerformance).delete()
        session.query(BenchmarkResult).delete()
        session.query(BenchmarkRun).delete()
        session.query(Question).delete()
        session.commit()

    def _restore_questions(self, session: Session, questions: List[Dict[str, Any]]) -> None:
        """Restore questions."""
        for q_data in questions:
            question = Question(**q_data)
            session.merge(question)

    def _restore_benchmark_runs(self, session: Session, runs: List[Dict[str, Any]]) -> None:
        """Restore benchmark runs."""
        for r_data in runs:
            run = BenchmarkRun(**r_data)
            session.merge(run)

    def _restore_results(self, session: Session, results: List[Dict[str, Any]]) -> None:
        """Restore benchmark results."""
        for r_data in results:
            result = BenchmarkResult(**r_data)
            session.merge(result)

    def _restore_performances(self, session: Session, performances: List[Dict[str, Any]]) -> None:
        """Restore model performances."""
        for p_data in performances:
            performance = ModelPerformance(**p_data)
            session.merge(performance)

    def _export_table_to_csv(self, records: List[Dict[str, Any]], csv_path: Path) -> None:
        """Export table records to CSV."""
        if not records:
            # Create empty CSV with headers
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([])
            return

        fieldnames = records[0].keys()

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    def _rotate_backups(self) -> None:
        """Rotate backups to keep only the most recent ones."""
        backups = self.list_backups()

        if len(backups) > self.max_backups:
            # Sort by creation time (oldest first)
            backups.sort(key=lambda x: x["created"])

            # Remove oldest backups
            for backup in backups[:len(backups) - self.max_backups]:
                try:
                    os.remove(backup["path"])
                    logger.info(f"Removed old backup: {backup['path']}")
                except Exception as e:
                    logger.error(f"Failed to remove backup {backup['path']}: {e}")


# CLI Helper Functions
def cli_create_backup(name: Optional[str] = None, compress: bool = True) -> None:
    """CLI command to create database backup."""
    backup = DatabaseBackup()
    backup_path = backup.create_backup(name, compress)
    print(f"Backup created: {backup_path}")


def cli_restore_backup(backup_path: str, drop_existing: bool = False) -> None:
    """CLI command to restore from backup."""
    backup = DatabaseBackup()
    backup.restore_backup(Path(backup_path), drop_existing)
    print(f"Backup restored from: {backup_path}")


def cli_list_backups() -> None:
    """CLI command to list available backups."""
    backup = DatabaseBackup()
    backups = backup.list_backups()

    if not backups:
        print("No backups found")
        return

    print("Available backups:")
    for b in backups:
        size_mb = b["size"] / (1024 * 1024)
        print(f"  {b['name']}: {size_mb:.2f} MB ({b['created'].strftime('%Y-%m-%d %H:%M:%S')})")


def cli_export_csv(output_dir: str, tables: Optional[List[str]] = None) -> None:
    """CLI command to export to CSV."""
    backup = DatabaseBackup()
    files = backup.export_to_csv(Path(output_dir), tables)
    print(f"Exported {len(files)} CSV files to {output_dir}")


def cli_export_json(output_file: str, tables: Optional[List[str]] = None) -> None:
    """CLI command to export to JSON."""
    backup = DatabaseBackup()
    file_path = backup.export_to_json(Path(output_file), tables)
    print(f"Exported to JSON: {file_path}")


def cli_import_json(json_file: str) -> None:
    """CLI command to import from JSON."""
    backup = DatabaseBackup()
    backup.import_from_json(Path(json_file))
    print(f"Imported from JSON: {json_file}")