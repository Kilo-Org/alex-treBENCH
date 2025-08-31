"""
Unit tests for DatabaseBackup.
"""

import pytest
import json
import gzip
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
from datetime import datetime

from storage.backup import DatabaseBackup, BackupMetadata
from core.exceptions import DatabaseError


class TestDatabaseBackup:
    """Test cases for DatabaseBackup."""

    def setup_method(self):
        """Setup test fixtures."""
        self.backup_dir = Path("test_backups")
        self.backup = DatabaseBackup(backup_dir=self.backup_dir)
        self.test_backup_path = self.backup_dir / "test_backup.json"

    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up test files
        if self.backup_dir.exists():
            for file in self.backup_dir.glob("*"):
                file.unlink()
            self.backup_dir.rmdir()

    def test_init_creates_backup_dir(self):
        """Test that backup directory is created on initialization."""
        assert self.backup_dir.exists()

    @patch('storage.backup.get_session')
    def test_create_backup_success(self, mock_get_session):
        """Test successful backup creation."""
        # Mock session and data
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        # Mock query results
        mock_questions = [MagicMock(id='q1', question_text='Test?', correct_answer='Answer')]
        mock_runs = [MagicMock(id=1, name='Test Run')]
        mock_results = [MagicMock(id=1, benchmark_run_id=1)]
        mock_performances = [MagicMock(id=1, benchmark_run_id=1)]

        mock_session.query.return_value.all.side_effect = [
            mock_questions, mock_runs, mock_results, mock_performances
        ]

        # Create backup
        backup_path = self.backup.create_backup("test_backup")

        # Verify backup was created
        assert backup_path.exists()
        assert backup_path.name == "test_backup.json"

        # Verify backup contents
        with open(backup_path, 'r') as f:
            backup_data = json.load(f)

        assert "metadata" in backup_data
        assert "data" in backup_data
        assert "questions" in backup_data["data"]
        assert "benchmark_runs" in backup_data["data"]

    @patch('storage.backup.get_session')
    def test_create_backup_with_compression(self, mock_get_session):
        """Test backup creation with compression."""
        # Mock session
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        mock_session.query.return_value.all.return_value = []

        # Create compressed backup
        backup_path = self.backup.create_backup("compressed_backup", compress=True)

        # Verify compressed file was created
        assert backup_path.exists()
        assert backup_path.suffix == ".gz"

        # Verify it's actually compressed
        with gzip.open(backup_path, 'rt') as f:
            backup_data = json.load(f)

        assert "metadata" in backup_data
        assert "data" in backup_data

    def test_restore_backup_success(self):
        """Test successful backup restoration."""
        # Create a test backup file
        test_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "record_counts": {"questions": 1},
                "tables": ["questions"],
                "format": "json",
                "compressed": False
            },
            "data": {
                "questions": [{"id": "q1", "question_text": "Test?", "correct_answer": "Answer"}],
                "benchmark_runs": [],
                "benchmark_results": [],
                "model_performance": []
            }
        }

        with open(self.test_backup_path, 'w') as f:
            json.dump(test_data, f)

        # Mock the restore methods
        with patch.object(self.backup, '_restore_questions') as mock_restore_questions, \
             patch.object(self.backup, '_restore_benchmark_runs') as mock_restore_runs, \
             patch.object(self.backup, '_restore_results') as mock_restore_results, \
             patch.object(self.backup, '_restore_performances') as mock_restore_performances, \
             patch('storage.backup.get_session') as mock_get_session:

            mock_session = MagicMock()
            mock_get_session.return_value = mock_session

            # Restore backup
            self.backup.restore_backup(self.test_backup_path)

            # Verify restore methods were called
            mock_restore_questions.assert_called_once()
            mock_restore_runs.assert_called_once()
            mock_restore_results.assert_called_once()
            mock_restore_performances.assert_called_once()

    def test_restore_backup_file_not_found(self):
        """Test restoration when backup file doesn't exist."""
        nonexistent_path = Path("nonexistent_backup.json")

        with pytest.raises(DatabaseError, match="Backup file not found"):
            self.backup.restore_backup(nonexistent_path)

    def test_list_backups(self):
        """Test listing available backups."""
        # Create some test backup files
        backup1 = self.backup_dir / "backup1.json"
        backup2 = self.backup_dir / "backup2.json.gz"

        # Create backup1
        with open(backup1, 'w') as f:
            json.dump({
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0.0",
                    "record_counts": {"questions": 1},
                    "tables": ["questions"],
                    "format": "json",
                    "compressed": False
                },
                "data": {"questions": []}
            }, f)

        # Create backup2 (compressed)
        with gzip.open(backup2, 'wt') as f:
            json.dump({
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0.0",
                    "record_counts": {"questions": 2},
                    "tables": ["questions"],
                    "format": "json",
                    "compressed": True
                },
                "data": {"questions": []}
            }, f)

        # List backups
        backups = self.backup.list_backups()

        assert len(backups) == 2
        assert any(b["name"] == "backup1" for b in backups)
        assert any(b["name"] == "backup2" for b in backups)

    @patch('storage.backup.get_session')
    def test_export_to_csv(self, mock_get_session):
        """Test exporting to CSV format."""
        # Mock session and data
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        mock_session.query.return_value.all.return_value = [
            MagicMock(id='q1', question_text='Test?', correct_answer='Answer',
                     category='Test', value=100, air_date=datetime.now(),
                     show_number=1, round='Jeopardy', difficulty_level='Easy',
                     created_at=datetime.now(), updated_at=datetime.now())
        ]

        output_dir = Path("test_csv_output")
        csv_files = self.backup.export_to_csv(output_dir)

        # Verify CSV files were created
        assert len(csv_files) == 4  # questions, benchmark_runs, results, performance
        assert all(f.exists() for f in csv_files)
        assert all(f.suffix == ".csv" for f in csv_files)

        # Clean up
        for f in csv_files:
            f.unlink()
        output_dir.rmdir()

    @patch('storage.backup.get_session')
    def test_export_to_json(self, mock_get_session):
        """Test exporting to JSON format."""
        # Mock session and data
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        mock_session.query.return_value.all.return_value = []

        output_file = Path("test_export.json")
        result_file = self.backup.export_to_json(output_file)

        # Verify JSON file was created
        assert result_file == output_file
        assert output_file.exists()

        # Verify contents
        with open(output_file, 'r') as f:
            data = json.load(f)

        assert "metadata" in data
        assert "data" in data
        assert "timestamp" in data["metadata"]

        # Clean up
        output_file.unlink()

    def test_import_from_json(self):
        """Test importing from JSON file."""
        # Create test JSON file
        test_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "record_counts": {"questions": 1},
                "tables": ["questions"]
            },
            "data": {
                "questions": [{"id": "q1", "question_text": "Test?", "correct_answer": "Answer"}],
                "benchmark_runs": [],
                "benchmark_results": [],
                "model_performance": []
            }
        }

        json_file = Path("test_import.json")
        with open(json_file, 'w') as f:
            json.dump(test_data, f)

        # Mock restore methods
        with patch.object(self.backup, '_restore_questions') as mock_restore, \
             patch('storage.backup.get_session') as mock_get_session:

            mock_session = MagicMock()
            mock_get_session.return_value = mock_session

            # Import data
            self.backup.import_from_json(json_file)

            # Verify restore was called
            mock_restore.assert_called_once()

        # Clean up
        json_file.unlink()

    def test_import_from_json_file_not_found(self):
        """Test importing from non-existent JSON file."""
        nonexistent_file = Path("nonexistent.json")

        with pytest.raises(DatabaseError, match="JSON file not found"):
            self.backup.import_from_json(nonexistent_file)

    def test_backup_metadata_creation(self):
        """Test backup metadata creation."""
        timestamp = datetime.now()
        metadata = BackupMetadata(
            timestamp=timestamp,
            version="1.0.0",
            record_counts={"questions": 10, "runs": 5},
            tables=["questions", "runs"],
            format="json",
            compressed=True
        )

        assert metadata.timestamp == timestamp
        assert metadata.version == "1.0.0"
        assert metadata.record_counts["questions"] == 10
        assert metadata.compressed is True
        assert not metadata.is_expired  # Should not be expired immediately

    def test_backup_rotation(self):
        """Test backup rotation functionality."""
        # Create multiple backup files
        for i in range(7):  # More than max_backups (5)
            backup_file = self.backup_dir / f"old_backup_{i}.json"
            with open(backup_file, 'w') as f:
                json.dump({"test": "data"}, f)

        # Trigger rotation by creating a new backup
        with patch.object(self.backup, '_export_all_tables', return_value={}), \
             patch('storage.backup.get_session'):
            self.backup.create_backup("new_backup")

        # Count remaining files
        remaining_files = list(self.backup_dir.glob("*.json"))
        # Should keep max_backups (5) + the new one, but rotation removes old ones
        assert len(remaining_files) <= 6  # Allow some flexibility


class TestBackupMetadata:
    """Test cases for BackupMetadata."""

    def test_age_calculation(self):
        """Test age calculation in days."""
        past_time = datetime.now().replace(day=datetime.now().day - 1)
        metadata = BackupMetadata(
            checkpoint_id="test",
            benchmark_run_id=1,
            timestamp=past_time,
            progress_data={},
            tables=[],
            format="json",
            compressed=False
        )

        assert metadata.age_days >= 0.9  # Approximately 1 day
        assert metadata.age_days < 2

    def test_is_expired_false(self):
        """Test that fresh metadata is not expired."""
        metadata = BackupMetadata(
            checkpoint_id="test",
            benchmark_run_id=1,
            timestamp=datetime.now(),
            progress_data={},
            tables=[],
            format="json",
            compressed=False
        )

        assert not metadata.is_expired