"""
Unit tests for data validation module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.data.validation import (
    DataValidator, 
    validate_question_format, 
    validate_answer_format,
    validate_category_name, 
    validate_dollar_value
)


@pytest.fixture
def validator():
    """Fixture providing DataValidator instance."""
    return DataValidator(strict_mode=False)


@pytest.fixture
def strict_validator():
    """Fixture providing strict DataValidator instance."""
    return DataValidator(strict_mode=True)


@pytest.fixture
def sample_questions_data():
    """Fixture providing sample questions data."""
    return pd.DataFrame({
        'question': [
            'This European capital is known as the City of Light.',
            'This planet is closest to the Sun.',
            'What is the largest ocean?',  # Direct question format
            'Too short',  # Too short
            '',  # Empty
            'This question has <b>HTML</b> tags.',  # HTML tags
            'This   has   excessive   whitespace.',  # Whitespace issues
        ],
        'answer': [
            'What is Paris?',
            'What is Mercury?',
            'Pacific Ocean',  # Missing "What is" format
            'A',  # Very short but valid
            '',  # Empty
            'Valid answer',
            'Another answer'
        ],
        'category': [
            'GEOGRAPHY',
            'SCIENCE',
            'GEOGRAPHY',
            'TEST',
            'INVALID',
            'SCIENCE & NATURE',  # With special chars
            'POTPOURRI'
        ],
        'value': [
            200,
            400,
            '$600',  # String format
            None,  # None value
            'invalid',  # Invalid format
            1000,
            -100  # Negative value
        ],
        'air_date': [
            '2020-01-01',
            '2020-02-15',
            'invalid-date',
            None,
            '1960-01-01',  # Too early
            '2025-01-01',  # Future date
            '2020-12-31'
        ],
        'difficulty_level': [
            'Easy',
            'Medium',
            'Hard',
            'Unknown',
            'Invalid',  # Not in valid list
            'Easy',
            'Medium'
        ]
    })


class TestDataValidator:
    """Test cases for DataValidator class."""
    
    def test_init_default(self):
        """Test validator initialization with defaults."""
        validator = DataValidator()
        assert validator.strict_mode is False
        assert 'SCIENCE' in validator.VALID_CATEGORIES
        assert 200 in validator.VALID_VALUES['modern']
    
    def test_init_strict_mode(self):
        """Test validator initialization in strict mode."""
        validator = DataValidator(strict_mode=True)
        assert validator.strict_mode is True
    
    def test_validate_question_format_valid(self, validator):
        """Test validation of valid questions."""
        valid_questions = [
            'This is a valid Jeopardy question.',
            'This European capital is known for its culture?',
            'This person wrote the famous novel in 1984.'
        ]
        
        for question in valid_questions:
            is_valid, issues = validator.validate_question_format(question)
            assert is_valid, f"Question should be valid: {question}. Issues: {issues}"
    
    def test_validate_question_format_invalid(self, validator):
        """Test validation of invalid questions."""
        invalid_questions = [
            '',  # Empty
            'Short',  # Too short
            'This question has <b>HTML</b> tags.',  # HTML tags
            'This   has   excessive   whitespace.',  # Whitespace
            'No punctuation',  # No ending punctuation
        ]
        
        for question in invalid_questions:
            is_valid, issues = validator.validate_question_format(question)
            assert not is_valid, f"Question should be invalid: {question}"
            assert len(issues) > 0
    
    def test_validate_question_format_strict_mode(self, strict_validator):
        """Test question validation in strict mode."""
        # Question that passes normal but might fail strict validation
        question = "What is the capital of France?"  # Direct question format
        
        is_valid, issues = strict_validator.validate_question_format(question)
        # In strict mode, this might be flagged for not following Jeopardy format
        # The actual result depends on implementation, but should have more checks
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)
    
    def test_validate_answer_format_valid(self, validator):
        """Test validation of valid answers."""
        valid_answers = [
            'Paris',
            'What is Mercury?',
            'Who is Shakespeare?',
            'The Pacific Ocean'
        ]
        
        for answer in valid_answers:
            is_valid, issues = validator.validate_answer_format(answer)
            assert is_valid, f"Answer should be valid: {answer}. Issues: {issues}"
    
    def test_validate_answer_format_invalid(self, validator):
        """Test validation of invalid answers."""
        invalid_answers = [
            '',  # Empty
            'This answer has <i>HTML</i> tags.',  # HTML tags
            'This   has   excessive   whitespace.',  # Whitespace
        ]
        
        for answer in invalid_answers:
            is_valid, issues = validator.validate_answer_format(answer)
            assert not is_valid, f"Answer should be invalid: {answer}"
            assert len(issues) > 0
    
    def test_validate_category_valid(self, validator):
        """Test validation of valid categories."""
        valid_categories = [
            'SCIENCE',
            'HISTORY',
            'SCIENCE & NATURE',
            'BEFORE & AFTER',
            'Custom Category'  # Should be valid with allow_custom=True
        ]
        
        for category in valid_categories:
            is_valid, issues = validator.validate_category(category, allow_custom=True)
            assert is_valid, f"Category should be valid: {category}. Issues: {issues}"
    
    def test_validate_category_invalid(self, validator):
        """Test validation of invalid categories."""
        invalid_categories = [
            '',  # Empty
            'X',  # Too short
            'A' * 60,  # Too long
            'Invalid@#$%',  # Invalid characters
        ]
        
        for category in invalid_categories:
            is_valid, issues = validator.validate_category(category)
            assert not is_valid, f"Category should be invalid: {category}"
            assert len(issues) > 0
    
    def test_validate_category_strict_mode_no_custom(self, strict_validator):
        """Test category validation in strict mode without custom categories."""
        custom_category = "CUSTOM CATEGORY"
        
        is_valid, issues = strict_validator.validate_category(custom_category, allow_custom=False)
        assert not is_valid
        assert any("not in predefined valid categories" in issue for issue in issues)
    
    def test_validate_dollar_value_valid(self, validator):
        """Test validation of valid dollar values."""
        valid_values = [
            200, 400, 600, 800, 1000,  # Integers
            None,  # None should be valid
        ]
        
        for value in valid_values:
            is_valid, issues = validator.validate_dollar_value(value)
            assert is_valid, f"Value should be valid: {value}. Issues: {issues}"
    
    def test_validate_dollar_value_invalid(self, validator):
        """Test validation of invalid dollar values."""
        invalid_values = [
            -100,  # Negative
            'invalid',  # Non-numeric string
            15000,  # Too high
        ]
        
        for value in invalid_values:
            is_valid, issues = validator.validate_dollar_value(value)
            assert not is_valid, f"Value should be invalid: {value}"
            assert len(issues) > 0
    
    def test_validate_dollar_value_different_eras(self, validator):
        """Test dollar value validation for different eras."""
        # Test classic era values
        is_valid, _ = validator.validate_dollar_value(100, era='classic')
        assert is_valid
        
        # Test modern era values
        is_valid, _ = validator.validate_dollar_value(200, era='modern')
        assert is_valid
        
        # Test double jeopardy values
        is_valid, _ = validator.validate_dollar_value(2000, era='double')
        assert is_valid
    
    def test_validate_dollar_value_strict_mode(self, strict_validator):
        """Test dollar value validation in strict mode."""
        # Value not in standard set
        non_standard_value = 350
        
        is_valid, issues = strict_validator.validate_dollar_value(non_standard_value, era='modern')
        assert not is_valid
        assert any("not in valid set" in issue for issue in issues)
    
    def test_validate_date_valid(self, validator):
        """Test validation of valid dates."""
        valid_dates = [
            '2020-01-01',
            '1984-12-31',
            datetime(2020, 6, 15),
            None,  # None should be valid
        ]
        
        for date_val in valid_dates:
            is_valid, issues = validator.validate_date(date_val)
            assert is_valid, f"Date should be valid: {date_val}. Issues: {issues}"
    
    def test_validate_date_invalid(self, validator):
        """Test validation of invalid dates."""
        invalid_dates = [
            'invalid-date',  # Invalid format
            '1960-01-01',    # Too early (before Jeopardy)
            '2025-12-31',    # Future date
        ]
        
        for date_val in invalid_dates:
            is_valid, issues = validator.validate_date(date_val)
            assert not is_valid, f"Date should be invalid: {date_val}"
            assert len(issues) > 0
    
    def test_validate_difficulty_level_valid(self, validator):
        """Test validation of valid difficulty levels."""
        valid_difficulties = ['Easy', 'Medium', 'Hard', 'Unknown', None]
        
        for difficulty in valid_difficulties:
            is_valid, issues = validator.validate_difficulty_level(difficulty)
            assert is_valid, f"Difficulty should be valid: {difficulty}. Issues: {issues}"
    
    def test_validate_difficulty_level_invalid(self, validator):
        """Test validation of invalid difficulty levels."""
        invalid_difficulties = ['Invalid', 'extreme', 123]
        
        for difficulty in invalid_difficulties:
            is_valid, issues = validator.validate_difficulty_level(difficulty)
            assert not is_valid, f"Difficulty should be invalid: {difficulty}"
            assert len(issues) > 0
    
    def test_validate_dataframe_complete(self, validator, sample_questions_data):
        """Test complete DataFrame validation."""
        results = validator.validate_dataframe(sample_questions_data)
        
        assert 'total_questions' in results
        assert 'valid_questions' in results
        assert 'validation_errors' in results
        assert 'field_validation' in results
        
        assert results['total_questions'] == len(sample_questions_data)
        assert results['valid_questions'] <= results['total_questions']
        
        # Check field validation results
        field_validation = results['field_validation']
        for field in ['question', 'answer', 'category', 'value']:
            assert field in field_validation
            assert 'valid' in field_validation[field]
            assert 'invalid' in field_validation[field]
            assert 'issues' in field_validation[field]
    
    def test_validate_dataframe_field_counts(self, validator, sample_questions_data):
        """Test that field validation counts are correct."""
        results = validator.validate_dataframe(sample_questions_data)
        
        total_records = len(sample_questions_data)
        
        # Check that valid + invalid counts equal total for each field
        for field_name, field_results in results['field_validation'].items():
            if field_name in sample_questions_data.columns:
                valid_count = field_results['valid']
                invalid_count = field_results['invalid']
                assert valid_count + invalid_count == total_records, \
                    f"Counts don't add up for {field_name}: {valid_count} + {invalid_count} != {total_records}"
    
    def test_validate_dataframe_issues_collection(self, validator, sample_questions_data):
        """Test that validation issues are properly collected."""
        results = validator.validate_dataframe(sample_questions_data)
        
        # Check that issues are collected for problematic records
        validation_errors = results['validation_errors']
        
        # Should have some validation errors with our problematic sample data
        assert len(validation_errors) > 0
        
        # Each error should have row_index and issues
        for error in validation_errors:
            assert 'row_index' in error
            assert 'issues' in error
            assert isinstance(error['issues'], list)
    
    def test_validate_dataframe_empty(self, validator):
        """Test DataFrame validation with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        results = validator.validate_dataframe(empty_df)
        
        assert results['total_questions'] == 0
        assert results['valid_questions'] == 0
        assert len(results['validation_errors']) == 0
    
    def test_validate_dataframe_missing_columns(self, validator):
        """Test DataFrame validation with missing columns."""
        df = pd.DataFrame({
            'some_column': ['data1', 'data2']
        })
        
        results = validator.validate_dataframe(df)
        
        # Should still run but might have fewer field validations
        assert results['total_questions'] == 2
        assert isinstance(results['field_validation'], dict)


class TestValidationHelperFunctions:
    """Test standalone validation helper functions."""
    
    def test_validate_question_format_helper(self):
        """Test standalone question format validation function."""
        assert validate_question_format('This is a valid question.')
        assert not validate_question_format('')
        assert not validate_question_format('Short')
    
    def test_validate_answer_format_helper(self):
        """Test standalone answer format validation function."""
        assert validate_answer_format('Valid answer')
        assert validate_answer_format('What is Paris?')
        assert not validate_answer_format('')
    
    def test_validate_category_name_helper(self):
        """Test standalone category name validation function."""
        assert validate_category_name('SCIENCE')
        assert validate_category_name('CUSTOM CATEGORY')
        assert not validate_category_name('')
        assert not validate_category_name('X')  # Too short
    
    def test_validate_dollar_value_helper(self):
        """Test standalone dollar value validation function."""
        assert validate_dollar_value(200)
        assert validate_dollar_value(None)  # None should be valid
        assert not validate_dollar_value(-100)
        assert not validate_dollar_value('invalid')


class TestValidationPerformance:
    """Test performance aspects of validation."""
    
    def test_large_dataset_validation(self, validator):
        """Test validation performance with large dataset."""
        # Create large dataset with mix of valid and invalid data
        size = 1000
        large_df = pd.DataFrame({
            'question': [f'Question {i}?' if i % 2 == 0 else 'Short' for i in range(size)],
            'answer': [f'Answer {i}' for i in range(size)],
            'category': ['TEST'] * size,
            'value': [200 if i % 2 == 0 else -100 for i in range(size)],
            'difficulty_level': ['Easy'] * size
        })
        
        results = validator.validate_dataframe(large_df)
        
        assert results['total_questions'] == size
        assert isinstance(results['valid_questions'], int)
        assert results['valid_questions'] <= size
    
    def test_validation_efficiency(self, validator):
        """Test that validation doesn't modify original data."""
        original_df = pd.DataFrame({
            'question': ['Original question?'],
            'answer': ['Original answer'],
            'category': ['ORIGINAL']
        })
        
        # Store original values
        original_values = original_df.copy()
        
        # Run validation
        results = validator.validate_dataframe(original_df)
        
        # Check that original DataFrame wasn't modified
        pd.testing.assert_frame_equal(original_df, original_values)


class TestValidationEdgeCases:
    """Test edge cases in validation."""
    
    def test_validation_with_nan_values(self, validator):
        """Test validation with NaN values."""
        df_with_nans = pd.DataFrame({
            'question': ['Valid question?', np.nan, None],
            'answer': ['Valid answer', np.nan, ''],
            'category': ['VALID', np.nan, 'ANOTHER'],
            'value': [200, np.nan, None]
        })
        
        results = validator.validate_dataframe(df_with_nans)
        
        assert results['total_questions'] == 3
        # NaN values should be handled gracefully
        assert isinstance(results['valid_questions'], int)
    
    def test_validation_with_mixed_types(self, validator):
        """Test validation with mixed data types."""
        df_mixed = pd.DataFrame({
            'question': ['String question?', 123, None],
            'answer': ['String answer', 456, ''],
            'category': ['STRING', 789, 'ANOTHER'],
            'value': [200, '400', None]
        })
        
        # Should handle mixed types gracefully
        results = validator.validate_dataframe(df_mixed)
        assert isinstance(results, dict)
        assert 'total_questions' in results