"""
Integration tests for persistence layer components.

Tests the interaction between database models, cache, backup, and state management.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch
from datetime import datetime

from core.database import init_database, get_session
from storage.models import Question, BenchmarkRun, BenchmarkResult, ModelPerformance
from storage.cache import CacheManager
from storage.backup import DatabaseBackup
from storage.state_manager import StateManager
from core.config import get_config


class TestPersistenceIntegration:
    """Integration tests for persistence components."""

    def setup_method(self):
        """Setup test database and fixtures."""
        # Use in-memory SQLite for testing
        self.db_url = "sqlite:///:memory:"
        self.config = get_config()

        # Override database URL for testing
        with patch('core.database.get_db_url', return_value=self.db_url):
            init_database()

        # Create test data
        self._create_test_data()

    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up would happen automatically with in-memory DB
        pass

    def _create_test_data(self):
        """Create test data for integration tests."""
        session = get_session()

        try:
            # Create test question
            question = Question(
                id="test_q1",
                question_text="What is the capital of France?",
                correct_answer="Paris",
                category="Geography",
                value=200,
                air_date=datetime(2023, 1, 1),
                show_number=9001,
                round="Jeopardy",
                difficulty_level="Easy"
            )
            session.add(question)

            # Create test benchmark run
            benchmark_run = BenchmarkRun(
                name="Integration Test Run",
                description="Test benchmark for integration testing",
                benchmark_mode="standard",
                sample_size=10,
                status="completed",
                models_tested='["openai/gpt-4", "anthropic/claude-3"]',
                total_questions=10,
                completed_questions=10,
                config_snapshot='{"test": "config"}',
                environment="test"
            )
            session.add(benchmark_run)
            session.flush()  # Get ID

            # Create test result
            result = BenchmarkResult(
                benchmark_run_id=benchmark_run.id,
                question_id="test_q1",
                model_name="openai/gpt-4",
                response_text="Paris",
                is_correct=True,
                confidence_score=0.95,
                response_time_ms=1500,
                tokens_generated=10,
                cost_usd=0.002
            )
            session.add(result)

            # Create test performance
            performance = ModelPerformance(
                benchmark_run_id=benchmark_run.id,
                model_name="openai/gpt-4",
                total_questions=10,
                correct_answers=9,
                accuracy_rate=0.9,
                avg_response_time_ms=1200.0,
                total_cost_usd=0.02,
                category_performance='{"Geography": {"correct": 9, "total": 10}}',
                difficulty_performance='{"Easy": {"correct": 9, "total": 10}}'
            )
            session.add(performance)

            session.commit()

            # Store IDs for later use
            self.question_id = question.id
            self.benchmark_run_id = benchmark_run.id
            self.result_id = result.id
            self.performance_id = performance.id

        finally:
            session.close()

    def test_database_models_integration(self):
        """Test integration between database models."""
        session = get_session()

        try:
            # Test relationships
            benchmark_run = session.query(BenchmarkRun).filter_by(id=self.benchmark_run_id).first()
            assert benchmark_run is not None
            assert len(benchmark_run.results) == 1
            assert len(benchmark_run.performances) == 1

            # Test result relationships
            result = session.query(BenchmarkResult).filter_by(id=self.result_id).first()
            assert result is not None
            assert result.benchmark_run.id == self.benchmark_run_id
            assert result.question.id == self.question_id

            # Test performance relationships
            performance = session.query(ModelPerformance).filter_by(id=self.performance_id).first()
            assert performance is not None
            assert performance.benchmark_run.id == self.benchmark_run_id

            # Test question relationships
            question = session.query(Question).filter_by(id=self.question_id).first()
            assert question is not None
            assert len(question.benchmark_results) == 1

        finally:
            session.close()

    def test_cache_database_integration(self):
        """Test integration between cache and database."""
        cache = CacheManager(max_size=10, default_ttl=300)

        try:
            # Cache some database data
            session = get_session()
            try:
                question = session.query(Question).filter_by(id=self.question_id).first()
                cache.set(f"question:{question.id}", {
                    "id": question.id,
                    "text": question.question_text,
                    "answer": question.correct_answer
                })
            finally:
                session.close()

            # Verify data is cached
            cached_data = cache.get(f"question:{self.question_id}")
            assert cached_data is not None
            assert cached_data["id"] == self.question_id
            assert cached_data["text"] == "What is the capital of France?"
            assert cached_data["answer"] == "Paris"

            # Test cache invalidation
            cache.invalidate_pattern("question:")
            assert cache.get(f"question:{self.question_id}") is None

        finally:
            cache.clear()

    def test_backup_database_integration(self):
        """Test integration between backup and database."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backup_dir = Path(temp_dir) / "backups"
            backup = DatabaseBackup(backup_dir=backup_dir)

            # Create backup
            backup_path = backup.create_backup("integration_test")

            # Verify backup was created
            assert backup_path.exists()

            # List backups
            backups = backup.list_backups()
            assert len(backups) > 0
            assert any(b["name"] == "integration_test" for b in backups)

            # Test backup restoration to a new database
            # Note: In a real scenario, you'd restore to a fresh DB
            # Here we just test the restoration logic
            backup.restore_backup(backup_path)

    def test_state_manager_database_integration(self):
        """Test integration between state manager and database."""
        with tempfile.TemporaryDirectory() as temp_dir:
            state_dir = Path(temp_dir) / "state"
            state_manager = StateManager(state_dir=state_dir)

            # Create a checkpoint
            progress_data = {
                "completed_questions": 5,
                "total_questions": 10,
                "current_model": "openai/gpt-4"
            }

            checkpoint_id = state_manager.save_checkpoint(
                benchmark_run_id=self.benchmark_run_id,
                progress_data=progress_data
            )

            # Verify checkpoint was created
            assert checkpoint_id is not None

            # Load checkpoint
            checkpoint = state_manager.load_checkpoint(checkpoint_id)
            assert checkpoint.benchmark_run_id == self.benchmark_run_id
            assert checkpoint.progress_data["completed_questions"] == 5

            # List checkpoints
            checkpoints = state_manager.list_checkpoints(self.benchmark_run_id)
            assert len(checkpoints) == 1
            assert checkpoints[0].checkpoint_id == checkpoint_id

    def test_full_persistence_workflow(self):
        """Test a complete persistence workflow."""
        # 1. Database operations
        session = get_session()
        try:
            # Query data
            benchmark_run = session.query(BenchmarkRun).filter_by(id=self.benchmark_run_id).first()
            assert benchmark_run.status == "completed"
        finally:
            session.close()

        # 2. Cache operations
        cache = CacheManager(max_size=10, default_ttl=300)
        try:
            # Cache benchmark data
            cache.set(f"benchmark:{self.benchmark_run_id}", {
                "id": benchmark_run.id,
                "name": benchmark_run.name,
                "status": benchmark_run.status
            })

            cached_benchmark = cache.get(f"benchmark:{self.benchmark_run_id}")
            assert cached_benchmark["status"] == "completed"
        finally:
            cache.clear()

        # 3. State management
        with tempfile.TemporaryDirectory() as temp_dir:
            state_dir = Path(temp_dir) / "state"
            state_manager = StateManager(state_dir=state_dir)

            # Save state
            state_data = {
                "benchmark_id": self.benchmark_run_id,
                "status": "completed",
                "progress": 100
            }
            checkpoint_id = state_manager.save_checkpoint(self.benchmark_run_id, state_data)

            # Restore state
            restored_state = state_manager.restore_from_checkpoint(checkpoint_id)
            assert restored_state["benchmark_run_id"] == self.benchmark_run_id
            assert restored_state["progress_data"]["status"] == "completed"

        # 4. Backup operations
        with tempfile.TemporaryDirectory() as temp_dir:
            backup_dir = Path(temp_dir) / "backups"
            backup = DatabaseBackup(backup_dir=backup_dir)

            # Create and verify backup
            backup_path = backup.create_backup("workflow_test")
            assert backup_path.exists()

            # Export to different formats
            csv_files = backup.export_to_csv(backup_dir / "csv")
            assert len(csv_files) > 0

            json_file = backup.export_to_json(backup_dir / "export.json")
            assert json_file.exists()

    def test_concurrent_access_simulation(self):
        """Test simulated concurrent access to persistence layer."""
        import threading
        import time

        results = []
        errors = []

        def worker(worker_id):
            """Worker function for concurrent access."""
            try:
                # Simulate database access
                session = get_session()
                try:
                    count = session.query(BenchmarkRun).count()
                    results.append(f"Worker {worker_id}: {count} runs")
                finally:
                    session.close()

                # Simulate cache access
                cache = CacheManager(max_size=10, default_ttl=300)
                try:
                    cache.set(f"worker:{worker_id}", f"data_{worker_id}")
                    cached_data = cache.get(f"worker:{worker_id}")
                    assert cached_data == f"data_{worker_id}"
                finally:
                    cache.clear()

            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")

        # Create and start threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Verify results
        assert len(results) == 5
        assert len(errors) == 0
        assert all("runs" in result for result in results)

    def test_data_consistency_across_components(self):
        """Test data consistency across all persistence components."""
        # Create a comprehensive test scenario
        test_data = {
            "question_id": "consistency_q1",
            "benchmark_name": "Consistency Test",
            "model_name": "openai/gpt-4",
            "response_text": "Test Answer",
            "is_correct": True
        }

        # 1. Store in database
        session = get_session()
        try:
            # Create question
            question = Question(
                id=test_data["question_id"],
                question_text="Consistency test question?",
                correct_answer="Test Answer",
                category="Test"
            )
            session.add(question)

            # Create benchmark run
            benchmark_run = BenchmarkRun(
                name=test_data["benchmark_name"],
                sample_size=1,
                status="completed"
            )
            session.add(benchmark_run)
            session.flush()

            # Create result
            result = BenchmarkResult(
                benchmark_run_id=benchmark_run.id,
                question_id=test_data["question_id"],
                model_name=test_data["model_name"],
                response_text=test_data["response_text"],
                is_correct=test_data["is_correct"],
                confidence_score=1.0
            )
            session.add(result)

            session.commit()

            benchmark_id = benchmark_run.id

        finally:
            session.close()

        # 2. Verify in database
        session = get_session()
        try:
            db_result = session.query(BenchmarkResult).filter_by(benchmark_run_id=benchmark_id).first()
            assert db_result.response_text == test_data["response_text"]
            assert db_result.is_correct == test_data["is_correct"]
        finally:
            session.close()

        # 3. Cache the data
        cache = CacheManager(max_size=10, default_ttl=300)
        try:
            cache.set(f"result:{benchmark_id}", {
                "response_text": test_data["response_text"],
                "is_correct": test_data["is_correct"]
            })

            cached_result = cache.get(f"result:{benchmark_id}")
            assert cached_result["response_text"] == test_data["response_text"]
            assert cached_result["is_correct"] == test_data["is_correct"]
        finally:
            cache.clear()

        # 4. Include in backup
        with tempfile.TemporaryDirectory() as temp_dir:
            backup_dir = Path(temp_dir) / "backups"
            backup = DatabaseBackup(backup_dir=backup_dir)

            backup_path = backup.create_backup("consistency_test")

            # Verify backup contains our data
            with open(backup_path, 'r') as f:
                backup_content = f.read()
                assert test_data["response_text"] in backup_content
                assert test_data["benchmark_name"] in backup_content

    def test_error_handling_integration(self):
        """Test error handling across persistence components."""
        # Test database connection errors
        with patch('core.database.get_session') as mock_session:
            mock_session.side_effect = Exception("Database connection failed")

            with pytest.raises(Exception, match="Database connection failed"):
                session = get_session()
                session.close()

        # Test cache errors
        cache = CacheManager(max_size=10, default_ttl=300)
        try:
            # Test with invalid data
            cache.set("invalid_key", None)
            result = cache.get("invalid_key")
            assert result is None
        finally:
            cache.clear()

        # Test backup errors
        with tempfile.TemporaryDirectory() as temp_dir:
            backup_dir = Path(temp_dir) / "backups"
            backup = DatabaseBackup(backup_dir=backup_dir)

            # Test restoring non-existent backup
            with pytest.raises(Exception):  # Should raise DatabaseError
                backup.restore_backup(Path("nonexistent.json"))

    def test_performance_under_load(self):
        """Test performance characteristics under load."""
        import time

        cache = CacheManager(max_size=100, default_ttl=300)

        try:
            # Test cache performance with multiple entries
            start_time = time.time()

            for i in range(50):
                cache.set(f"perf_key_{i}", f"perf_value_{i}")

            cache_time = time.time() - start_time

            # Should be fast (< 1 second for 50 operations)
            assert cache_time < 1.0

            # Test retrieval performance
            start_time = time.time()

            for i in range(50):
                value = cache.get(f"perf_key_{i}")
                assert value == f"perf_value_{i}"

            retrieval_time = time.time() - start_time

            # Should be very fast (< 0.5 seconds for 50 retrievals)
            assert retrieval_time < 0.5

        finally:
            cache.clear()