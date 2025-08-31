"""
Unit tests for data ingestion module.
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from src.data.ingestion import KaggleDatasetLoader, DataIngestionEngine
from src.core.exceptions import DataIngestionError


class TestKaggleDatasetLoader:
    """Test cases for KaggleDatasetLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.loader = KaggleDatasetLoader(cache_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_init_creates_cache_directory(self):
        """Test that initialization creates cache directory."""
        assert self.loader.cache_dir.exists()
        assert self.loader.dataset_name == "aravindram11/jeopardy-dataset-updated"
    
    def test_get_cached_dataset_path(self):
        """Test cached dataset path generation."""
        expected_path = self.temp_dir / "aravindram11_jeopardy-dataset-updated"
        assert self.loader._get_cached_dataset_path() == expected_path
    
    def test_validate_dataset_requirements_valid_data(self):
        """Test dataset validation with valid data."""
        df = pd.DataFrame({
            'question': ['Test question?'],
            'answer': ['Test answer'],
            'category': ['TEST']
        })
        
        # Should not raise exception
        self.loader._validate_dataset_requirements(df)
    
    def test_validate_dataset_requirements_missing_columns(self):
        """Test dataset validation with missing required columns."""
        df = pd.DataFrame({
            'some_column': ['data']
        })
        
        with pytest.raises(DataIngestionError) as exc_info:
            self.loader._validate_dataset_requirements(df)
        
        assert "missing required columns" in str(exc_info.value)
    
    def test_validate_dataset_requirements_column_variations(self):
        """Test dataset validation with column name variations."""
        # Test with 'clue' instead of 'question'
        df = pd.DataFrame({
            'clue': ['Test clue?'],
            'response': ['Test response'],
        })
        
        # Should not raise exception
        self.loader._validate_dataset_requirements(df)
    
    @patch('src.data.ingestion.kagglehub.dataset_download')
    @patch('src.data.ingestion.kagglehub.load_dataset')
    def test_load_dataset_success(self, mock_load_dataset, mock_download):
        """Test successful dataset loading."""
        # Mock the kagglehub responses
        test_df = pd.DataFrame({
            'question': ['What is the capital of France?'],
            'answer': ['Paris'],
            'category': ['GEOGRAPHY']
        })
        mock_load_dataset.return_value = test_df
        
        result = self.loader.load_dataset('test.csv')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]['question'] == 'What is the capital of France?'
        mock_load_dataset.assert_called_once()
    
    @patch('src.data.ingestion.kagglehub.load_dataset')
    def test_load_dataset_validation_failure(self, mock_load_dataset):
        """Test dataset loading with validation failure."""
        # Mock invalid data
        invalid_df = pd.DataFrame({
            'invalid_column': ['data']
        })
        mock_load_dataset.return_value = invalid_df
        
        with pytest.raises(DataIngestionError):
            self.loader.load_dataset('test.csv')
    
    @patch('src.data.ingestion.kagglehub.dataset_download')
    def test_download_and_cache_dataset_success(self, mock_download):
        """Test successful dataset download and caching."""
        # Mock successful download
        mock_download_path = self.temp_dir / "downloaded_dataset"
        mock_download_path.mkdir()
        (mock_download_path / "test.csv").write_text("test,data\n1,2")
        
        mock_download.return_value = str(mock_download_path)
        
        result = self.loader.download_and_cache_dataset()
        
        assert result.exists()
        mock_download.assert_called_once()
    
    def test_download_and_cache_dataset_uses_cache(self):
        """Test that cached dataset is used when available."""
        # Create fake cached dataset
        cached_path = self.loader._get_cached_dataset_path()
        cached_path.mkdir(parents=True)
        (cached_path / "test.csv").write_text("cached,data\n1,2")
        
        with patch('src.data.ingestion.kagglehub.dataset_download') as mock_download:
            result = self.loader.download_and_cache_dataset()
            
            assert result == cached_path
            # Should not call download since cache exists
            mock_download.assert_not_called()
    
    @patch('src.data.ingestion.kagglehub.dataset_download')
    def test_download_and_cache_dataset_network_error(self, mock_download):
        """Test handling of network errors during download."""
        mock_download.side_effect = Exception("Network error")
        
        with pytest.raises(DataIngestionError) as exc_info:
            self.loader.download_and_cache_dataset()
        
        assert "Network error" in str(exc_info.value)


class TestDataIngestionEngine:
    """Test cases for legacy DataIngestionEngine class."""
    
    def test_legacy_class_inherits_from_loader(self):
        """Test that DataIngestionEngine inherits from KaggleDatasetLoader."""
        engine = DataIngestionEngine()
        assert isinstance(engine, KaggleDatasetLoader)
    
    def test_legacy_download_dataset_method(self):
        """Test legacy download_dataset method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = DataIngestionEngine(cache_dir=Path(temp_dir))
            
            with patch.object(engine, 'download_and_cache_dataset') as mock_method:
                mock_method.return_value = Path(temp_dir)
                
                result = engine.download_dataset(force_download=True)
                
                mock_method.assert_called_once_with(True)
                assert result == Path(temp_dir)


@pytest.fixture
def sample_jeopardy_data():
    """Fixture providing sample Jeopardy data."""
    return pd.DataFrame({
        'question': [
            'This European capital is known as the City of Light',
            'This planet is closest to the Sun',
            'This Shakespeare play features Romeo and Juliet'
        ],
        'answer': [
            'What is Paris?',
            'What is Mercury?', 
            'What is Romeo and Juliet?'
        ],
        'category': [
            'WORLD CAPITALS',
            'ASTRONOMY',
            'LITERATURE'
        ],
        'value': [200, 400, 600],
        'air_date': ['2020-01-01', '2020-01-02', '2020-01-03'],
        'difficulty_level': ['Easy', 'Medium', 'Hard']
    })


class TestIntegration:
    """Integration tests for data ingestion."""
    
    @patch('src.data.ingestion.kagglehub.load_dataset')
    def test_full_loading_pipeline(self, mock_load_dataset, sample_jeopardy_data):
        """Test complete data loading pipeline."""
        mock_load_dataset.return_value = sample_jeopardy_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = KaggleDatasetLoader(cache_dir=Path(temp_dir))
            
            result = loader.load_dataset()
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert all(col in result.columns for col in ['question', 'answer', 'category'])