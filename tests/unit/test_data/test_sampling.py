"""
Unit tests for data sampling module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.data.sampling import StatisticalSampler
from src.core.exceptions import SamplingError


@pytest.fixture
def sampler():
    """Fixture providing StatisticalSampler instance."""
    return StatisticalSampler()


@pytest.fixture
def sample_dataset():
    """Fixture providing sample dataset for testing."""
    np.random.seed(42)
    data = []
    
    categories = ['SCIENCE', 'HISTORY', 'LITERATURE', 'GEOGRAPHY', 'SPORTS']
    difficulties = ['Easy', 'Medium', 'Hard']
    
    for i in range(100):
        data.append({
            'question_id': f'q_{i}',
            'question': f'Question {i}?',
            'answer': f'Answer {i}',
            'category': np.random.choice(categories),
            'difficulty_level': np.random.choice(difficulties),
            'value': np.random.choice([200, 400, 600, 800, 1000]),
            'air_date': pd.Timestamp('2020-01-01') + pd.Timedelta(days=i)
        })
    
    return pd.DataFrame(data)


class TestStatisticalSampler:
    """Test cases for StatisticalSampler class."""
    
    def test_init_default_values(self):
        """Test sampler initialization with default values."""
        sampler = StatisticalSampler()
        
        assert sampler.confidence_level == 0.95
        assert sampler.margin_of_error == 0.05
        assert 0.95 in sampler.z_scores
        assert sampler.z_scores[0.95] == 1.96
    
    def test_init_custom_values(self):
        """Test sampler initialization with custom values."""
        sampler = StatisticalSampler(confidence_level=0.99, margin_of_error=0.03)
        
        assert sampler.confidence_level == 0.99
        assert sampler.margin_of_error == 0.03
    
    def test_get_sample_size_default(self, sampler):
        """Test sample size calculation with defaults."""
        sample_size = sampler.get_sample_size()
        
        # For 95% confidence and 5% margin of error, should be ~384
        assert sample_size >= 384
        assert sample_size <= 400  # Should be reasonable
    
    def test_get_sample_size_custom(self, sampler):
        """Test sample size calculation with custom parameters."""
        sample_size = sampler.get_sample_size(confidence_level=0.90, margin_of_error=0.10)
        
        # Lower confidence and higher margin should give smaller sample
        assert sample_size < 100
        assert sample_size >= 30  # Minimum sample size
    
    def test_calculate_sample_size_small_population(self, sampler):
        """Test sample size calculation with small population."""
        sample_size = sampler.calculate_sample_size(population_size=200)
        
        # Should apply finite population correction
        assert sample_size < 200
        assert sample_size >= 30
    
    def test_calculate_sample_size_large_population(self, sampler):
        """Test sample size calculation with large population."""
        sample_size = sampler.calculate_sample_size(population_size=100000)
        
        # Should not apply finite population correction
        expected_size = sampler.get_sample_size()
        assert sample_size == expected_size
    
    def test_calculate_sample_size_error_handling(self, sampler):
        """Test error handling in sample size calculation."""
        with pytest.raises(SamplingError):
            # This should trigger an error in the calculation
            sampler.calculate_sample_size(population_size=-1)
    
    def test_random_sample_basic(self, sampler, sample_dataset):
        """Test basic random sampling."""
        sample_size = 10
        result = sampler.random_sample(sample_dataset, sample_size)
        
        assert len(result) == sample_size
        assert isinstance(result, pd.DataFrame)
        assert all(col in result.columns for col in sample_dataset.columns)
    
    def test_random_sample_with_seed(self, sampler, sample_dataset):
        """Test random sampling with seed for reproducibility."""
        sample_size = 10
        seed = 123
        
        result1 = sampler.random_sample(sample_dataset, sample_size, seed=seed)
        result2 = sampler.random_sample(sample_dataset, sample_size, seed=seed)
        
        # Should get identical samples with same seed
        pd.testing.assert_frame_equal(result1.reset_index(drop=True), 
                                    result2.reset_index(drop=True))
    
    def test_random_sample_larger_than_population(self, sampler):
        """Test random sampling when requested size exceeds population."""
        small_dataset = pd.DataFrame({
            'question': ['Q1', 'Q2', 'Q3'],
            'answer': ['A1', 'A2', 'A3']
        })
        
        result = sampler.random_sample(small_dataset, n=10)
        
        # Should return all available records
        assert len(result) == 3
    
    def test_random_sample_empty_dataset(self, sampler):
        """Test random sampling with empty dataset."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(SamplingError) as exc_info:
            sampler.random_sample(empty_df, n=10)
        
        assert "Cannot sample from empty dataset" in str(exc_info.value)
    
    def test_stratified_sample_basic(self, sampler, sample_dataset):
        """Test basic stratified sampling."""
        sample_size = 20
        result = sampler.stratified_sample(sample_dataset, sample_size)
        
        assert len(result) <= sample_size + 5  # Allow some variance due to rounding
        assert isinstance(result, pd.DataFrame)
        
        # Check that we have representation from multiple categories
        unique_categories = result['category'].nunique()
        assert unique_categories > 1
    
    def test_stratified_sample_with_columns(self, sampler, sample_dataset):
        """Test stratified sampling with specified columns."""
        sample_size = 20
        stratify_columns = ['difficulty_level']
        
        result = sampler.stratified_sample(
            sample_dataset, 
            sample_size, 
            stratify_columns=stratify_columns
        )
        
        assert len(result) <= sample_size + 5
        
        # Check that we have representation from different difficulties
        unique_difficulties = result['difficulty_level'].nunique()
        assert unique_difficulties > 1
    
    def test_stratified_sample_no_stratification_columns(self, sampler):
        """Test stratified sampling with no valid stratification columns."""
        df = pd.DataFrame({
            'question': ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
            'answer': ['A1', 'A2', 'A3', 'A4', 'A5']
        })
        
        result = sampler.stratified_sample(df, sample_size=3)
        
        # Should fall back to random sampling
        assert len(result) == 3
    
    def test_stratified_sample_auto_size_calculation(self, sampler, sample_dataset):
        """Test stratified sampling with automatic sample size calculation."""
        result = sampler.stratified_sample(sample_dataset)  # No sample_size specified
        
        # Should calculate appropriate sample size
        expected_size = sampler.calculate_sample_size(len(sample_dataset))
        assert len(result) <= expected_size + 10  # Allow some variance
    
    def test_temporal_stratified_sample(self, sampler, sample_dataset):
        """Test temporal stratified sampling."""
        sample_size = 20
        result = sampler.temporal_stratified_sample(
            sample_dataset, 
            sample_size=sample_size,
            date_column='air_date'
        )
        
        assert len(result) <= sample_size + 5
        
        # Check that we have questions from different time periods
        date_range = result['air_date'].max() - result['air_date'].min()
        assert date_range.days > 10  # Should span more than 10 days
    
    def test_temporal_stratified_sample_missing_date_column(self, sampler, sample_dataset):
        """Test temporal stratified sampling with missing date column."""
        df_no_date = sample_dataset.drop('air_date', axis=1)
        
        # Should fall back to regular stratified sampling
        result = sampler.temporal_stratified_sample(df_no_date, sample_size=10)
        
        assert len(result) == 10
    
    def test_balanced_difficulty_sample(self, sampler, sample_dataset):
        """Test balanced difficulty sampling."""
        sample_size = 30
        result = sampler.balanced_difficulty_sample(sample_dataset, sample_size)
        
        assert len(result) <= sample_size + 5
        
        # Check that we have reasonable distribution across difficulties
        difficulty_counts = result['difficulty_level'].value_counts()
        assert len(difficulty_counts) > 1  # Multiple difficulties represented
    
    def test_balanced_difficulty_sample_custom_distribution(self, sampler, sample_dataset):
        """Test balanced difficulty sampling with custom distribution."""
        sample_size = 20
        custom_distribution = {
            'Easy': 0.6,
            'Medium': 0.3,
            'Hard': 0.1
        }
        
        result = sampler.balanced_difficulty_sample(
            sample_dataset, 
            sample_size=sample_size,
            difficulty_distribution=custom_distribution
        )
        
        assert len(result) <= sample_size + 5
        
        # Check approximate distribution (allow some variance)
        difficulty_counts = result['difficulty_level'].value_counts(normalize=True)
        
        if 'Easy' in difficulty_counts:
            assert difficulty_counts['Easy'] > difficulty_counts.get('Hard', 0)
    
    def test_balanced_difficulty_sample_no_difficulty_column(self, sampler):
        """Test balanced difficulty sampling without difficulty column."""
        df = pd.DataFrame({
            'question': ['Q1', 'Q2', 'Q3'],
            'answer': ['A1', 'A2', 'A3']
        })
        
        # Should fall back to regular sampling
        result = sampler.balanced_difficulty_sample(df, sample_size=2)
        assert len(result) == 2
    
    def test_get_sampling_statistics(self, sampler, sample_dataset):
        """Test sampling statistics generation."""
        # Create a sample
        sample = sampler.stratified_sample(sample_dataset, sample_size=20)
        
        stats = sampler.get_sampling_statistics(sample_dataset, sample)
        
        assert 'original_size' in stats
        assert 'sample_size' in stats
        assert 'sampling_ratio' in stats
        assert 'representativeness' in stats
        
        assert stats['original_size'] == len(sample_dataset)
        assert stats['sample_size'] == len(sample)
        assert 0 < stats['sampling_ratio'] < 1
    
    def test_get_sampling_statistics_representativeness(self, sampler, sample_dataset):
        """Test representativeness calculation in sampling statistics."""
        # Create a sample
        sample = sampler.stratified_sample(sample_dataset, sample_size=30)
        
        stats = sampler.get_sampling_statistics(sample_dataset, sample)
        
        # Check that representativeness includes key columns
        repr_stats = stats['representativeness']
        
        if 'category' in sample_dataset.columns:
            assert 'category' in repr_stats
            assert 'mean_difference' in repr_stats['category']
            assert 'max_difference' in repr_stats['category']


class TestSamplingErrorHandling:
    """Test error handling in sampling."""
    
    def test_stratified_sample_empty_dataset(self, sampler):
        """Test stratified sampling with empty dataset."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(SamplingError) as exc_info:
            sampler.stratified_sample(empty_df, sample_size=10)
        
        assert "Cannot sample from empty dataset" in str(exc_info.value)
    
    def test_balanced_difficulty_sample_error(self, sampler):
        """Test balanced difficulty sampling error handling."""
        df = pd.DataFrame({
            'question': ['Q1'],
            'difficulty_level': ['Easy']
        })
        
        # Request sample larger than available data for specific difficulty
        result = sampler.balanced_difficulty_sample(df, sample_size=10)
        
        # Should handle gracefully and return what's available
        assert len(result) <= 1


class TestSamplingPerformance:
    """Test performance aspects of sampling."""
    
    def test_large_dataset_sampling(self, sampler):
        """Test sampling with large dataset."""
        # Create large dataset
        large_df = pd.DataFrame({
            'question': [f'Question {i}' for i in range(10000)],
            'answer': [f'Answer {i}' for i in range(10000)],
            'category': ['TEST'] * 10000,
            'difficulty_level': ['Medium'] * 10000
        })
        
        result = sampler.stratified_sample(large_df, sample_size=100)
        
        assert len(result) <= 105  # Allow small variance
        assert isinstance(result, pd.DataFrame)
    
    def test_sampling_preserves_data_types(self, sampler, sample_dataset):
        """Test that sampling preserves original data types."""
        original_dtypes = sample_dataset.dtypes
        
        result = sampler.stratified_sample(sample_dataset, sample_size=20)
        
        # Check that dtypes are preserved for common columns
        for col in result.columns:
            if col in original_dtypes:
                assert result[col].dtype == original_dtypes[col]


class TestSamplingReproducibility:
    """Test reproducibility of sampling methods."""
    
    def test_stratified_sampling_consistency(self, sample_dataset):
        """Test that stratified sampling is consistent across runs."""
        sampler1 = StatisticalSampler()
        sampler2 = StatisticalSampler()
        
        # Both should use the same random_state internally
        result1 = sampler1.stratified_sample(sample_dataset, sample_size=20)
        result2 = sampler2.stratified_sample(sample_dataset, sample_size=20)
        
        # Results should be identical since we use fixed random_state=42
        pd.testing.assert_frame_equal(
            result1.sort_values('question_id').reset_index(drop=True),
            result2.sort_values('question_id').reset_index(drop=True)
        )