"""
Unit tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch

from src.data.preprocessing import DataPreprocessor
from src.core.exceptions import DataIngestionError, ValidationError


@pytest.fixture
def preprocessor():
    """Fixture providing DataPreprocessor instance."""
    return DataPreprocessor()


@pytest.fixture
def raw_jeopardy_data():
    """Fixture providing raw, unprocessed Jeopardy data."""
    return pd.DataFrame({
        'Clue': [
            'This European capital is <i>known</i> as the "City of Light"',
            '  This   planet    is closest to the Sun  ',
            'This Shakespeare play features Romeo and Juliet',
            '',  # Empty question
            'Very short'  # Too short
        ],
        'Response': [
            'What is Paris?',
            'What is Mercury?',
            'Romeo and Juliet',  # Missing "What is"
            'Answer without question',
            'Short'
        ],
        'Subject': [
            'WORLD CAPITALS',
            'ASTRONOMY  ',  # Extra whitespace
            'literature',   # Wrong case
            'CATEGORY',
            'TEST'
        ],
        'Worth': [
            '$200',
            '400',
            'None',  # Invalid value
            '$600',
            '800'
        ],
        'Date': [
            '2020-01-01',
            '01/15/2020',
            'January 20, 2020',
            'invalid-date',
            '2020-12-31'
        ],
        'Episode': [1, 2, 3, 4, 5]
    })


@pytest.fixture
def clean_jeopardy_data():
    """Fixture providing clean, processed Jeopardy data."""
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
        'difficulty_level': ['Easy', 'Easy', 'Medium']
    })


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    def test_standardize_columns(self, preprocessor, raw_jeopardy_data):
        """Test column name standardization."""
        result = preprocessor._standardize_columns(raw_jeopardy_data)
        
        # Check that columns were mapped correctly
        assert 'question' in result.columns  # Clue -> question
        assert 'answer' in result.columns    # Response -> answer
        assert 'category' in result.columns  # Subject -> category
        assert 'value' in result.columns     # Worth -> value
        assert 'air_date' in result.columns  # Date -> air_date
        
        # Original column names should be gone
        assert 'Clue' not in result.columns
        assert 'Response' not in result.columns
        assert 'Subject' not in result.columns
    
    def test_clean_text_columns(self, preprocessor):
        """Test text cleaning functionality."""
        df = pd.DataFrame({
            'question': [
                'This has <i>HTML</i> tags',
                'This   has    extra   spaces',
                'This has &quot;quotes&quot; and &amp; symbols'
            ],
            'answer': ['Clean answer', 'Another &lt;test&gt;', 'Final answer'],
            'category': ['TEST', 'ANOTHER  TEST', 'FINAL']
        })
        
        result = preprocessor._clean_text_columns(df)
        
        # Check HTML tags removed
        assert '<i>' not in result.iloc[0]['question']
        assert '</i>' not in result.iloc[0]['question']
        
        # Check whitespace normalized
        assert result.iloc[1]['question'] == 'This has extra spaces'
        
        # Check HTML entities decoded
        assert '"quotes"' in result.iloc[2]['question']
        assert '&' in result.iloc[2]['question']
        assert '&lt;' not in result.iloc[1]['answer']
    
    def test_normalize_values(self, preprocessor):
        """Test monetary value normalization."""
        df = pd.DataFrame({
            'value': ['$200', '400', '$1,000', 'None', '', None, 'invalid']
        })
        
        result = preprocessor._normalize_values(df)
        
        assert result.iloc[0]['value'] == 200
        assert result.iloc[1]['value'] == 400
        assert result.iloc[2]['value'] == 1000
        assert pd.isna(result.iloc[3]['value'])  # 'None' -> NaN
        assert pd.isna(result.iloc[4]['value'])  # '' -> NaN
        assert pd.isna(result.iloc[5]['value'])  # None -> NaN
        assert pd.isna(result.iloc[6]['value'])  # 'invalid' -> NaN
    
    def test_parse_dates(self, preprocessor):
        """Test date parsing functionality."""
        df = pd.DataFrame({
            'air_date': [
                '2020-01-01',
                '01/15/2020', 
                'January 20, 2020',
                'invalid-date',
                None
            ]
        })
        
        result = preprocessor._parse_dates(df)
        
        # Valid dates should be parsed
        assert result.iloc[0]['air_date'] == datetime(2020, 1, 1).date()
        assert result.iloc[1]['air_date'] == datetime(2020, 1, 15).date()
        assert result.iloc[2]['air_date'] == datetime(2020, 1, 20).date()
        
        # Invalid date should be None
        assert pd.isna(result.iloc[3]['air_date'])
        assert pd.isna(result.iloc[4]['air_date'])
    
    def test_filter_invalid_records(self, preprocessor):
        """Test filtering of invalid records."""
        df = pd.DataFrame({
            'question': ['Good question', '', 'Short', 'Another good question'],
            'answer': ['Good answer', 'Good answer', '', 'Another answer']
        })
        
        result = preprocessor._filter_invalid_records(df)
        
        # Should keep only records with valid question and answer
        assert len(result) == 2
        assert 'Good question' in result['question'].values
        assert 'Another good question' in result['question'].values
    
    def test_add_difficulty_levels(self, preprocessor):
        """Test difficulty level assignment."""
        df = pd.DataFrame({
            'value': [200, 800, 1500, None]
        })
        
        result = preprocessor._add_difficulty_levels(df)
        
        assert result.iloc[0]['difficulty_level'] == 'Easy'    # 200 <= 600
        assert result.iloc[1]['difficulty_level'] == 'Medium'  # 800 <= 1200
        assert result.iloc[2]['difficulty_level'] == 'Hard'    # 1500 > 1200
        assert result.iloc[3]['difficulty_level'] == 'Unknown' # None
    
    def test_add_metadata_columns(self, preprocessor):
        """Test addition of metadata columns."""
        df = pd.DataFrame({
            'question': ['Test question'],
            'answer': ['Test answer']
        })
        
        result = preprocessor._add_metadata_columns(df)
        
        assert 'question_id' in result.columns
        assert 'processed_at' in result.columns
        assert 'question_length' in result.columns
        assert 'answer_length' in result.columns
        
        assert result.iloc[0]['question_length'] == len('Test question')
        assert result.iloc[0]['answer_length'] == len('Test answer')
    
    def test_preprocess_dataset_complete_pipeline(self, preprocessor, raw_jeopardy_data):
        """Test complete preprocessing pipeline."""
        result = preprocessor.preprocess_dataset(raw_jeopardy_data)
        
        # Check that all processing steps were applied
        assert 'question' in result.columns
        assert 'difficulty_level' in result.columns
        assert 'processed_at' in result.columns
        
        # Check that invalid records were filtered out
        assert len(result) < len(raw_jeopardy_data)
        
        # Check that text was cleaned
        assert '<i>' not in result['question'].str.cat()
    
    def test_validate_dataset(self, preprocessor, clean_jeopardy_data):
        """Test dataset validation."""
        validation_results = preprocessor.validate_dataset(clean_jeopardy_data)
        
        assert validation_results['total_records'] == len(clean_jeopardy_data)
        assert 'required_columns_present' in validation_results
        assert validation_results['required_columns_present']['question'] is True
        assert validation_results['required_columns_present']['answer'] is True
    
    def test_validate_dataset_missing_required_column(self, preprocessor):
        """Test validation failure with missing required column."""
        df = pd.DataFrame({
            'question': ['Test question']
            # Missing 'answer' column
        })
        
        with pytest.raises(ValidationError) as exc_info:
            preprocessor.validate_dataset(df)
        
        assert "Required column 'answer' missing" in str(exc_info.value)
    
    def test_filter_by_category(self, preprocessor, clean_jeopardy_data):
        """Test category filtering."""
        result = preprocessor.filter_by_category(clean_jeopardy_data, ['ASTRONOMY'])
        
        assert len(result) == 1
        assert result.iloc[0]['category'] == 'ASTRONOMY'
    
    def test_filter_by_difficulty(self, preprocessor, clean_jeopardy_data):
        """Test difficulty filtering."""
        result = preprocessor.filter_by_difficulty(clean_jeopardy_data, ['Easy'])
        
        # Should get records with 'Easy' difficulty
        assert len(result) == 2
        assert all(result['difficulty_level'] == 'Easy')
    
    def test_filter_by_value_range(self, preprocessor, clean_jeopardy_data):
        """Test value range filtering."""
        result = preprocessor.filter_by_value_range(clean_jeopardy_data, min_value=300, max_value=700)
        
        # Should get records with values between 300 and 700
        assert len(result) == 2
        assert all((result['value'] >= 300) & (result['value'] <= 700))
    
    def test_filter_by_date_range(self, preprocessor):
        """Test date range filtering."""
        df = pd.DataFrame({
            'question': ['Q1', 'Q2', 'Q3'],
            'air_date': pd.to_datetime(['2020-01-01', '2020-06-01', '2020-12-01'])
        })
        
        result = preprocessor.filter_by_date_range(df, start_date='2020-03-01', end_date='2020-09-01')
        
        assert len(result) == 1
        assert result.iloc[0]['question'] == 'Q2'
    
    def test_filter_by_round(self, preprocessor):
        """Test round filtering."""
        df = pd.DataFrame({
            'question': ['Q1', 'Q2', 'Q3'],
            'round': ['Jeopardy!', 'Double Jeopardy!', 'Final Jeopardy!']
        })
        
        result = preprocessor.filter_by_round(df, ['jeopardy', 'double'])
        
        assert len(result) == 2
        assert 'Final Jeopardy!' not in result['round'].values
    
    def test_apply_filters_multiple(self, preprocessor, clean_jeopardy_data):
        """Test applying multiple filters together."""
        filters = {
            'categories': ['ASTRONOMY', 'LITERATURE'],
            'difficulty_levels': ['Medium'],
            'min_value': 500
        }
        
        result = preprocessor.apply_filters(clean_jeopardy_data, filters)
        
        # Should only get LITERATURE record (value=600, difficulty=Medium)
        assert len(result) == 1
        assert result.iloc[0]['category'] == 'LITERATURE'
    
    def test_empty_dataframe_handling(self, preprocessor):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame()
        
        # Should not raise exception
        result = preprocessor._filter_invalid_records(empty_df)
        assert len(result) == 0
        
        # Should handle missing columns gracefully
        result = preprocessor._clean_text_columns(empty_df)
        assert len(result) == 0


class TestErrorHandling:
    """Test error handling in preprocessing."""
    
    def test_preprocess_dataset_error_handling(self, preprocessor):
        """Test error handling in preprocessing pipeline."""
        # Create DataFrame that will cause processing errors
        problematic_df = pd.DataFrame({
            'question': [None, None, None],
            'answer': [None, None, None]
        })
        
        with pytest.raises(DataIngestionError):
            preprocessor.preprocess_dataset(problematic_df)
    
    def test_validation_error_handling(self, preprocessor):
        """Test validation error handling."""
        df = pd.DataFrame({
            'random_column': ['data']
        })
        
        with pytest.raises(ValidationError):
            preprocessor.validate_dataset(df)


class TestPerformance:
    """Test performance aspects of preprocessing."""
    
    def test_large_dataset_processing(self, preprocessor):
        """Test preprocessing with larger dataset."""
        # Create larger dataset
        large_df = pd.DataFrame({
            'question': [f'Question {i}' for i in range(1000)],
            'answer': [f'Answer {i}' for i in range(1000)],
            'category': ['TEST'] * 1000,
            'value': [200] * 1000
        })
        
        result = preprocessor.preprocess_dataset(large_df)
        
        assert len(result) == 1000
        assert all(col in result.columns for col in ['question', 'answer', 'difficulty_level'])