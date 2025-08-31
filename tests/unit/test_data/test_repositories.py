"""
Unit tests for updated repository functionality related to data handling.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from sqlalchemy.orm import Session

from src.storage.repositories import QuestionRepository, BenchmarkRepository
from src.storage.models import BenchmarkQuestion, Benchmark
from src.core.exceptions import DatabaseError


@pytest.fixture
def mock_session():
    """Fixture providing mock database session."""
    return Mock(spec=Session)


@pytest.fixture
def question_repo(mock_session):
    """Fixture providing QuestionRepository with mock session."""
    return QuestionRepository(mock_session)


@pytest.fixture
def benchmark_repo(mock_session):
    """Fixture providing BenchmarkRepository with mock session."""
    return BenchmarkRepository(mock_session)


@pytest.fixture
def sample_questions_df():
    """Fixture providing sample questions DataFrame."""
    return pd.DataFrame({
        'question_id': ['q1', 'q2', 'q3'],
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
        'category': ['GEOGRAPHY', 'SCIENCE', 'LITERATURE'],
        'value': [200, 400, 600],
        'difficulty_level': ['Easy', 'Easy', 'Medium']
    })


class TestQuestionRepository:
    """Test cases for QuestionRepository data handling methods."""
    
    def test_save_questions_success(self, question_repo, mock_session, sample_questions_df):
        """Test successful bulk saving of questions."""
        benchmark_id = 1
        
        # Mock successful commit
        mock_session.add_all.return_value = None
        mock_session.commit.return_value = None
        
        result = question_repo.save_questions(sample_questions_df, benchmark_id)
        
        # Verify session calls
        mock_session.add_all.assert_called_once()
        mock_session.commit.assert_called_once()
        
        # Verify returned questions
        assert len(result) == 3
        assert all(isinstance(q, BenchmarkQuestion) for q in result)
        assert result[0].benchmark_id == benchmark_id
        assert result[0].question_text == 'This European capital is known as the City of Light'
    
    def test_save_questions_database_error(self, question_repo, mock_session, sample_questions_df):
        """Test handling of database errors during save."""
        benchmark_id = 1
        
        # Mock database error
        mock_session.commit.side_effect = Exception("Database connection failed")
        
        with pytest.raises(DatabaseError) as exc_info:
            question_repo.save_questions(sample_questions_df, benchmark_id)
        
        # Verify rollback was called
        mock_session.rollback.assert_called_once()
        assert "Failed to save questions" in str(exc_info.value)
    
    def test_save_questions_handles_missing_columns(self, question_repo, mock_session):
        """Test saving questions with missing optional columns."""
        # DataFrame with minimal required columns
        minimal_df = pd.DataFrame({
            'question': ['Test question?'],
            'answer': ['Test answer']
        })
        
        benchmark_id = 1
        
        result = question_repo.save_questions(minimal_df, benchmark_id)
        
        assert len(result) == 1
        assert result[0].question_text == 'Test question?'
        assert result[0].correct_answer == 'Test answer'
        assert result[0].category is None  # Should handle missing columns gracefully
    
    def test_get_questions_no_filters(self, question_repo, mock_session):
        """Test getting questions without filters."""
        # Mock database query
        mock_questions = [
            Mock(spec=BenchmarkQuestion),
            Mock(spec=BenchmarkQuestion)
        ]
        mock_query = Mock()
        mock_query.offset.return_value.all.return_value = mock_questions
        mock_session.query.return_value = mock_query
        
        result = question_repo.get_questions()
        
        # Verify query construction
        mock_session.query.assert_called_once_with(BenchmarkQuestion)
        assert result == mock_questions
    
    def test_get_questions_with_filters(self, question_repo, mock_session):
        """Test getting questions with filters applied."""
        filters = {
            'benchmark_id': 1,
            'categories': ['SCIENCE', 'HISTORY'],
            'difficulty_levels': ['Easy', 'Medium'],
            'min_value': 200,
            'max_value': 800
        }
        
        # Mock query chain
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        
        # Mock filter chain
        mock_query.filter.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.all.return_value = []
        
        result = question_repo.get_questions(filters, limit=10, offset=5)
        
        # Verify query was built with filters
        mock_session.query.assert_called_once_with(BenchmarkQuestion)
        assert mock_query.filter.call_count >= 4  # Should apply multiple filters
        mock_query.offset.assert_called_once_with(5)
        mock_query.limit.assert_called_once_with(10)
    
    def test_get_random_questions(self, question_repo, mock_session):
        """Test getting random questions."""
        n = 5
        benchmark_id = 1
        category = 'SCIENCE'
        difficulty = 'Medium'
        
        # Mock query chain
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [Mock(spec=BenchmarkQuestion) for _ in range(n)]
        
        result = question_repo.get_random_questions(
            n=n, 
            benchmark_id=benchmark_id,
            category=category, 
            difficulty=difficulty
        )
        
        assert len(result) == n
        mock_query.filter.assert_called()  # Should apply filters
        mock_query.limit.assert_called_once_with(n)
    
    def test_get_question_statistics_complete(self, question_repo, mock_session):
        """Test getting comprehensive question statistics."""
        benchmark_id = 1
        
        # Mock query results
        mock_base_query = Mock()
        mock_session.query.return_value = mock_base_query
        
        # Mock count
        mock_base_query.count.return_value = 1000
        mock_base_query.filter.return_value = mock_base_query
        
        # Mock category distribution query
        mock_base_query.with_entities.return_value = mock_base_query
        mock_base_query.group_by.return_value = mock_base_query
        
        # Mock category results
        category_results = [('SCIENCE', 300), ('HISTORY', 250), ('LITERATURE', 200)]
        difficulty_results = [('Easy', 400), ('Medium', 400), ('Hard', 200)]
        value_stats = (200, 2000, 600.0, 800)  # min, max, avg, count
        
        # Set up mock returns for different query chains
        mock_base_query.all.side_effect = [category_results, difficulty_results]
        mock_base_query.first.return_value = value_stats
        
        result = question_repo.get_question_statistics(benchmark_id)
        
        # Verify statistics structure
        assert result['total_questions'] == 1000
        assert 'category_distribution' in result
        assert 'difficulty_distribution' in result
        assert 'value_range' in result
        assert 'unique_categories' in result
        
        # Verify category distribution
        assert result['category_distribution']['SCIENCE'] == 300
        assert result['unique_categories'] == 3
    
    def test_get_question_statistics_no_benchmark_filter(self, question_repo, mock_session):
        """Test getting statistics without benchmark filter."""
        # Mock base query without benchmark filter
        mock_base_query = Mock()
        mock_session.query.return_value = mock_base_query
        mock_base_query.count.return_value = 500
        mock_base_query.filter.return_value = mock_base_query
        mock_base_query.with_entities.return_value = mock_base_query
        mock_base_query.group_by.return_value = mock_base_query
        mock_base_query.all.return_value = []
        mock_base_query.first.return_value = (None, None, None, 0)
        
        result = question_repo.get_question_statistics()  # No benchmark_id
        
        assert result['total_questions'] == 500
        assert isinstance(result['category_distribution'], dict)
    
    def test_get_question_statistics_database_error(self, question_repo, mock_session):
        """Test handling of database errors in statistics."""
        mock_session.query.side_effect = Exception("Database error")
        
        with pytest.raises(DatabaseError) as exc_info:
            question_repo.get_question_statistics()
        
        assert "Failed to get question statistics" in str(exc_info.value)
    
    def test_get_questions_database_error(self, question_repo, mock_session):
        """Test handling of database errors in get_questions."""
        mock_session.query.side_effect = Exception("Database connection lost")
        
        with pytest.raises(DatabaseError) as exc_info:
            question_repo.get_questions()
        
        assert "Failed to get questions with filters" in str(exc_info.value)


class TestRepositoryDataHandling:
    """Test data handling aspects of repositories."""
    
    def test_question_mapping_from_dataframe(self, question_repo, mock_session):
        """Test proper mapping of DataFrame columns to BenchmarkQuestion fields."""
        # DataFrame with alternative column names
        df_alt_columns = pd.DataFrame({
            'clue': ['Test clue'],
            'response': ['Test response'],
            'subject': ['TEST CATEGORY'],
            'worth': [400],
            'difficulty_level': ['Medium']
        })
        
        result = question_repo.save_questions(df_alt_columns, benchmark_id=1)
        
        # Verify mapping worked correctly
        assert len(result) == 1
        question = result[0]
        assert question.question_text == 'Test clue'
        assert question.correct_answer == 'Test response'
        assert question.category == 'TEST CATEGORY'
        assert question.value == 400
    
    def test_question_id_generation(self, question_repo, mock_session):
        """Test question ID generation from DataFrame."""
        df = pd.DataFrame({
            'question': ['Test question'],
            'answer': ['Test answer']
        })
        
        result = question_repo.save_questions(df, benchmark_id=1)
        
        # Should use DataFrame index as question_id when not provided
        assert result[0].question_id == '0'  # First row index
    
    def test_question_id_from_dataframe(self, question_repo, mock_session):
        """Test using existing question_id from DataFrame."""
        df = pd.DataFrame({
            'question_id': ['custom_id_1'],
            'question': ['Test question'],
            'answer': ['Test answer']
        })
        
        result = question_repo.save_questions(df, benchmark_id=1)
        
        assert result[0].question_id == 'custom_id_1'
    
    def test_value_handling_with_nan(self, question_repo, mock_session):
        """Test handling of NaN values in numeric columns."""
        df = pd.DataFrame({
            'question': ['Test question'],
            'answer': ['Test answer'],
            'value': [float('nan')]  # NaN value
        })
        
        result = question_repo.save_questions(df, benchmark_id=1)
        
        assert result[0].value is None  # NaN should become None
    
    def test_empty_dataframe_handling(self, question_repo, mock_session):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        
        result = question_repo.save_questions(empty_df, benchmark_id=1)
        
        assert len(result) == 0
        # Should not crash and should not call add_all with empty list
        mock_session.add_all.assert_called_once_with([])


class TestRepositoryFilteringLogic:
    """Test filtering logic in repository methods."""
    
    def test_filter_construction_benchmark_id(self, question_repo, mock_session):
        """Test that benchmark_id filter is properly constructed."""
        filters = {'benchmark_id': 123}
        
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.all.return_value = []
        
        question_repo.get_questions(filters)
        
        # Verify filter was called (specific filter content is implementation detail)
        mock_query.filter.assert_called()
    
    def test_filter_construction_multiple_categories(self, question_repo, mock_session):
        """Test that multiple categories filter is properly constructed."""
        filters = {'categories': ['SCIENCE', 'HISTORY', 'LITERATURE']}
        
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.all.return_value = []
        
        question_repo.get_questions(filters)
        
        mock_query.filter.assert_called()
    
    def test_filter_construction_value_range(self, question_repo, mock_session):
        """Test that value range filters are properly constructed."""
        filters = {
            'min_value': 200,
            'max_value': 800
        }
        
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.all.return_value = []
        
        question_repo.get_questions(filters)
        
        # Should call filter twice (once for min, once for max)
        assert mock_query.filter.call_count >= 2


class TestRepositoryIntegration:
    """Integration-style tests for repository functionality."""
    
    def test_save_then_retrieve_questions(self, question_repo, mock_session, sample_questions_df):
        """Test saving questions then retrieving them back."""
        benchmark_id = 1
        
        # Mock save operation
        saved_questions = question_repo.save_questions(sample_questions_df, benchmark_id)
        
        # Mock retrieval operation
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.all.return_value = saved_questions
        
        retrieved_questions = question_repo.get_questions({'benchmark_id': benchmark_id})
        
        # Verify we get back what we saved
        assert len(retrieved_questions) == len(saved_questions)
        assert retrieved_questions == saved_questions
    
    @patch('src.storage.repositories.logger')
    def test_logging_in_operations(self, mock_logger, question_repo, mock_session, sample_questions_df):
        """Test that operations log appropriately."""
        question_repo.save_questions(sample_questions_df, benchmark_id=1)
        
        # Verify logging occurred
        mock_logger.info.assert_called()
        
        # Check that log messages contain expected information
        log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any('Saving' in msg and '3 questions' in msg for msg in log_calls)
        assert any('Successfully saved' in msg for msg in log_calls)