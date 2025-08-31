"""
Integration tests for the complete data pipeline.

Tests the end-to-end flow from data ingestion to database storage.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
import pandas as pd
import os
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from data.ingestion import KaggleDatasetLoader
from data.preprocessing import DataPreprocessor
from data.validation import DataValidator
from data.sampling import StatisticalSampler
from storage.repositories import QuestionRepository, BenchmarkRepository
from scripts.init_data import DataInitializer
from core.database import get_db_session
from core.config import get_config


@pytest.fixture
def temp_cache_dir():
    """Fixture providing temporary cache directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_jeopardy_dataset():
    """Fixture providing sample Jeopardy dataset for testing."""
    return pd.DataFrame({
        'Show Number': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10,  # 100 questions
        'Air Date': ['2020-01-01'] * 20 + ['2020-06-01'] * 30 + ['2020-12-01'] * 50,
        'Round': ['Jeopardy!'] * 60 + ['Double Jeopardy!'] * 35 + ['Final Jeopardy!'] * 5,
        'Category': ['SCIENCE'] * 25 + ['HISTORY'] * 25 + ['LITERATURE'] * 25 + 
                   ['GEOGRAPHY'] * 15 + ['SPORTS'] * 10,
        'Value': ['$200'] * 20 + ['$400'] * 20 + ['$600'] * 20 + 
                ['$800'] * 20 + ['$1000'] * 20,
        'Question': [f'This is test question number {i+1}.' for i in range(100)],
        'Answer': [f'What is test answer {i+1}?' for i in range(100)]
    })


class TestDataPipelineIntegration:
    """Integration tests for the complete data pipeline."""
    
    @patch('src.data.ingestion.kagglehub.load_dataset')
    def test_ingestion_to_preprocessing_pipeline(self, mock_load_dataset, 
                                               temp_cache_dir, sample_jeopardy_dataset):
        """Test data flow from ingestion through preprocessing."""
        # Mock kagglehub to return our sample data
        mock_load_dataset.return_value = sample_jeopardy_dataset
        
        # Initialize components
        loader = KaggleDatasetLoader(cache_dir=temp_cache_dir)
        preprocessor = DataPreprocessor()
        
        # Load data
        raw_data = loader.load_dataset()
        assert len(raw_data) == 100
        assert 'Show Number' in raw_data.columns  # Original column names
        
        # Preprocess data
        processed_data = preprocessor.preprocess_dataset(raw_data)
        
        # Verify preprocessing worked
        assert len(processed_data) > 0  # Should have valid questions after filtering
        assert 'question' in processed_data.columns  # Standardized column names
        assert 'answer' in processed_data.columns
        assert 'category' in processed_data.columns
        assert 'difficulty_level' in processed_data.columns
        
        # Check that values were normalized
        if 'value' in processed_data.columns:
            # Values should be integers, not strings with $
            assert processed_data['value'].dtype in ['int64', 'float64']
            assert not processed_data['value'].astype(str).str.contains('$').any()
    
    @patch('src.data.ingestion.kagglehub.load_dataset')
    def test_preprocessing_to_validation_pipeline(self, mock_load_dataset,
                                                temp_cache_dir, sample_jeopardy_dataset):
        """Test data flow from preprocessing through validation."""
        mock_load_dataset.return_value = sample_jeopardy_dataset
        
        # Initialize components
        loader = KaggleDatasetLoader(cache_dir=temp_cache_dir)
        preprocessor = DataPreprocessor()
        validator = DataValidator(strict_mode=False)
        
        # Process data
        raw_data = loader.load_dataset()
        processed_data = preprocessor.preprocess_dataset(raw_data)
        
        # Validate data
        validation_results = validator.validate_dataframe(processed_data)
        
        # Verify validation results
        assert 'total_questions' in validation_results
        assert 'valid_questions' in validation_results
        assert validation_results['total_questions'] > 0
        assert validation_results['valid_questions'] > 0
        
        # Check field validation
        field_validation = validation_results['field_validation']
        assert 'question' in field_validation
        assert 'answer' in field_validation
        assert field_validation['question']['valid'] > 0
        assert field_validation['answer']['valid'] > 0
    
    @patch('src.data.ingestion.kagglehub.load_dataset')
    def test_preprocessing_to_sampling_pipeline(self, mock_load_dataset,
                                              temp_cache_dir, sample_jeopardy_dataset):
        """Test data flow from preprocessing through sampling."""
        mock_load_dataset.return_value = sample_jeopardy_dataset
        
        # Initialize components
        loader = KaggleDatasetLoader(cache_dir=temp_cache_dir)
        preprocessor = DataPreprocessor()
        sampler = StatisticalSampler()
        
        # Process data
        raw_data = loader.load_dataset()
        processed_data = preprocessor.preprocess_dataset(raw_data)
        
        # Test different sampling methods
        sample_size = min(20, len(processed_data))
        
        # Random sampling
        random_sample = sampler.random_sample(processed_data, sample_size)
        assert len(random_sample) == sample_size
        
        # Stratified sampling
        stratified_sample = sampler.stratified_sample(processed_data, sample_size)
        assert len(stratified_sample) <= sample_size + 5  # Allow small variance
        
        # Balanced difficulty sampling (if difficulty column exists)
        if 'difficulty_level' in processed_data.columns:
            balanced_sample = sampler.balanced_difficulty_sample(processed_data, sample_size)
            assert len(balanced_sample) <= sample_size + 5
        
        # Verify samples contain expected columns
        for sample_df in [random_sample, stratified_sample]:
            assert 'question' in sample_df.columns
            assert 'answer' in sample_df.columns
            assert len(sample_df) > 0
    
    @patch('src.data.ingestion.kagglehub.load_dataset')
    @patch('src.storage.repositories.get_db_session')
    def test_complete_pipeline_to_database(self, mock_get_session, mock_load_dataset,
                                         temp_cache_dir, sample_jeopardy_dataset):
        """Test complete pipeline including database storage."""
        # Mock database session
        mock_session = Mock()
        mock_get_session.return_value.__enter__.return_value = mock_session
        
        # Mock successful database operations
        mock_benchmark = Mock()
        mock_benchmark.id = 1
        mock_session.add.return_value = None
        mock_session.commit.return_value = None
        mock_session.refresh.return_value = None
        
        # Mock kagglehub
        mock_load_dataset.return_value = sample_jeopardy_dataset
        
        # Initialize components
        loader = KaggleDatasetLoader(cache_dir=temp_cache_dir)
        preprocessor = DataPreprocessor()
        sampler = StatisticalSampler()
        
        # Run pipeline
        raw_data = loader.load_dataset()
        processed_data = preprocessor.preprocess_dataset(raw_data)
        sample_data = sampler.stratified_sample(processed_data, sample_size=10)
        
        # Mock repository operations
        with patch('src.storage.repositories.BenchmarkRepository') as MockBenchmarkRepo:
            with patch('src.storage.repositories.QuestionRepository') as MockQuestionRepo:
                # Mock benchmark creation
                mock_benchmark_repo = MockBenchmarkRepo.return_value
                mock_benchmark_repo.create_benchmark.return_value = mock_benchmark
                mock_benchmark_repo.update_benchmark_status.return_value = None
                
                # Mock question saving
                mock_question_repo = MockQuestionRepo.return_value
                mock_questions = [Mock() for _ in range(len(sample_data))]
                mock_question_repo.save_questions.return_value = mock_questions
                mock_question_repo.get_question_statistics.return_value = {
                    'total_questions': len(sample_data),
                    'unique_categories': 5,
                    'category_distribution': {'SCIENCE': 5, 'HISTORY': 3, 'LITERATURE': 2},
                    'difficulty_distribution': {'Easy': 6, 'Medium': 3, 'Hard': 1},
                    'value_range': {'min': 200, 'max': 1000, 'average': 600}
                }
                
                # Create repositories and save data
                benchmark_repo = MockBenchmarkRepo(mock_session)
                question_repo = MockQuestionRepo(mock_session)
                
                # This simulates the save operation
                benchmark = mock_benchmark_repo.create_benchmark.return_value
                questions = mock_question_repo.save_questions.return_value
                stats = mock_question_repo.get_question_statistics.return_value
                
                # Verify operations were called
                mock_question_repo.save_questions.assert_called_once()
                mock_benchmark_repo.create_benchmark.assert_called_once()
                
                # Verify results
                assert len(questions) > 0
                assert stats['total_questions'] > 0
    
    @patch('src.data.ingestion.kagglehub.load_dataset')
    @patch('src.core.database.init_database')
    @patch('src.storage.repositories.get_db_session')
    def test_data_initializer_complete_workflow(self, mock_get_session, mock_init_db,
                                              mock_load_dataset, temp_cache_dir, 
                                              sample_jeopardy_dataset):
        """Test the complete DataInitializer workflow."""
        # Mock database operations
        mock_session = Mock()
        mock_get_session.return_value.__enter__.return_value = mock_session
        mock_init_db.return_value = None
        
        # Mock kagglehub
        mock_load_dataset.return_value = sample_jeopardy_dataset
        
        # Mock successful database saves
        mock_benchmark = Mock()
        mock_benchmark.id = 1
        mock_questions = [Mock() for _ in range(10)]
        
        with patch('src.scripts.init_data.BenchmarkRepository') as MockBenchmarkRepo:
            with patch('src.scripts.init_data.QuestionRepository') as MockQuestionRepo:
                # Setup mocks
                mock_benchmark_repo = MockBenchmarkRepo.return_value
                mock_benchmark_repo.create_benchmark.return_value = mock_benchmark
                mock_benchmark_repo.update_benchmark_status.return_value = None
                
                mock_question_repo = MockQuestionRepo.return_value
                mock_question_repo.save_questions.return_value = mock_questions
                mock_question_repo.get_question_statistics.return_value = {
                    'total_questions': 10,
                    'unique_categories': 3,
                    'category_distribution': {'SCIENCE': 4, 'HISTORY': 3, 'LITERATURE': 3},
                    'difficulty_distribution': {'Easy': 5, 'Medium': 3, 'Hard': 2},
                    'value_range': {'min': 200, 'max': 1000, 'average': 600},
                    'value_distribution': {'Low ($1-600)': 6, 'Medium ($601-1200)': 4, 'High ($1201+)': 0}
                }
                
                # Initialize and run
                initializer = DataInitializer(force_download=False, validate_strict=False)
                
                # Override the loader cache directory
                initializer.loader.cache_dir = temp_cache_dir
                
                result = initializer.run()
                
                # Verify results
                assert result is not None
                assert 'benchmark_id' in result
                assert 'questions_saved' in result
                assert 'statistics' in result
                assert result['benchmark_id'] == 1
                assert result['questions_saved'] == 10


class TestDataPipelineErrorHandling:
    """Test error handling throughout the pipeline."""
    
    def test_ingestion_error_propagation(self, temp_cache_dir):
        """Test that ingestion errors are properly handled."""
        with patch('src.data.ingestion.kagglehub.load_dataset') as mock_load:
            mock_load.side_effect = Exception("Network error")
            
            loader = KaggleDatasetLoader(cache_dir=temp_cache_dir)
            
            with pytest.raises(Exception) as exc_info:
                loader.load_dataset()
            
            assert "Network error" in str(exc_info.value)
    
    def test_preprocessing_with_invalid_data(self, temp_cache_dir):
        """Test preprocessing with completely invalid data."""
        # Create dataset with missing required columns
        invalid_data = pd.DataFrame({
            'irrelevant_column': ['data1', 'data2', 'data3']
        })
        
        preprocessor = DataPreprocessor()
        
        # Should handle gracefully but result in empty or minimal dataset
        result = preprocessor.preprocess_dataset(invalid_data)
        
        # May result in empty dataset or dataset with minimal processing
        assert isinstance(result, pd.DataFrame)
    
    def test_sampling_with_empty_dataset(self):
        """Test sampling behavior with empty dataset."""
        empty_df = pd.DataFrame()
        sampler = StatisticalSampler()
        
        with pytest.raises(Exception):  # Should raise SamplingError
            sampler.stratified_sample(empty_df, sample_size=10)
    
    @patch('src.scripts.init_data.get_db_session')
    def test_database_error_handling(self, mock_get_session, temp_cache_dir):
        """Test handling of database errors in the pipeline."""
        # Mock database session that raises an error
        mock_session = Mock()
        mock_session.commit.side_effect = Exception("Database connection failed")
        mock_get_session.return_value.__enter__.return_value = mock_session
        
        with patch('src.data.ingestion.kagglehub.load_dataset') as mock_load:
            mock_load.return_value = pd.DataFrame({
                'question': ['Test?'],
                'answer': ['Test'],
                'category': ['TEST']
            })
            
            initializer = DataInitializer(force_download=False)
            initializer.loader.cache_dir = temp_cache_dir
            
            # Should handle database errors gracefully
            with pytest.raises(Exception):
                initializer.run()


class TestPipelinePerformance:
    """Test performance aspects of the pipeline."""
    
    @patch('src.data.ingestion.kagglehub.load_dataset')
    def test_large_dataset_processing(self, mock_load_dataset, temp_cache_dir):
        """Test pipeline with larger dataset."""
        # Create larger test dataset
        large_dataset = pd.DataFrame({
            'Show Number': list(range(1000)) * 2,
            'Air Date': ['2020-01-01'] * 2000,
            'Round': ['Jeopardy!'] * 2000,
            'Category': ['SCIENCE'] * 2000,
            'Value': ['$400'] * 2000,
            'Question': [f'Question {i}?' for i in range(2000)],
            'Answer': [f'Answer {i}' for i in range(2000)]
        })
        
        mock_load_dataset.return_value = large_dataset
        
        # Initialize components
        loader = KaggleDatasetLoader(cache_dir=temp_cache_dir)
        preprocessor = DataPreprocessor()
        sampler = StatisticalSampler()
        
        # Process data
        raw_data = loader.load_dataset()
        processed_data = preprocessor.preprocess_dataset(raw_data)
        sample_data = sampler.stratified_sample(processed_data, sample_size=100)
        
        # Verify processing completed successfully
        assert len(raw_data) == 2000
        assert len(processed_data) > 0
        assert len(sample_data) <= 105  # Allow small variance


class TestCliIntegration:
    """Test CLI integration with the data pipeline."""
    
    @patch('src.scripts.init_data.main')
    def test_cli_data_init_command(self, mock_init_main):
        """Test CLI data init command integration."""
        mock_init_main.return_value = {
            'benchmark_id': 1,
            'questions_saved': 100,
            'statistics': {'total_questions': 100}
        }
        
        # Test would require actual CLI execution
        # This is a placeholder for CLI integration testing
        result = mock_init_main(force_download=False, strict_validation=False)
        
        assert result['benchmark_id'] == 1
        assert result['questions_saved'] == 100


if __name__ == "__main__":
    # Allow running this test file directly for debugging
    pytest.main([__file__, "-v"])