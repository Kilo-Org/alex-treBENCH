"""
Data Preprocessing

Data cleaning, normalization, and validation for Jeopardy questions
with standardized column mapping and quality checks.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime

from src.core.exceptions import DataIngestionError, ValidationError
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """Handles data cleaning and normalization for Jeopardy datasets."""
    
    # Standard column mapping for different dataset formats
    COLUMN_MAPPINGS = {
        'question': ['question', 'clue', 'prompt', 'text'],
        'answer': ['answer', 'response', 'correct_answer', 'solution'],
        'category': ['category', 'subject', 'topic', 'theme'],
        'value': ['value', 'points', 'dollar_value', 'amount', 'worth'],
        'air_date': ['air_date', 'date', 'broadcast_date', 'show_date'],
        'show_number': ['show_number', 'episode', 'episode_number', 'show_id'],
        'round': ['round', 'game_round', 'round_type']
    }
    
    # Valid Jeopardy rounds
    VALID_ROUNDS = ['Jeopardy!', 'Double Jeopardy!', 'Final Jeopardy!', 'Tiebreaker']
    
    def __init__(self):
        """Initialize the data preprocessor."""
        pass
    
    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for Jeopardy dataset.
        
        Args:
            df: Raw dataset DataFrame
            
        Returns:
            Cleaned and normalized DataFrame
            
        Raises:
            DataIngestionError: If preprocessing fails
        """
        try:
            logger.info(f"Starting preprocessing of {len(df)} questions")
            
            # Step 1: Standardize column names
            df = self._standardize_columns(df)
            
            # Step 2: Clean and validate required columns
            df = self._clean_text_columns(df)
            df = self._normalize_values(df)
            df = self._parse_dates(df)
            
            # Step 3: Filter out invalid records
            df = self._filter_invalid_records(df)
            
            # Step 4: Add derived columns
            df = self._add_difficulty_levels(df)
            df = self._add_metadata_columns(df)
            
            logger.info(f"Preprocessing complete. {len(df)} valid questions remain")
            return df
            
        except Exception as e:
            raise DataIngestionError(
                f"Data preprocessing failed: {str(e)}"
            ) from e
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to expected format."""
        logger.info("Standardizing column names")
        
        # Create mapping from current columns to standard names
        column_map = {}
        current_columns = [col.lower().strip() for col in df.columns]
        
        for standard_name, possible_names in self.COLUMN_MAPPINGS.items():
            for current_col, original_col in zip(current_columns, df.columns):
                if current_col in possible_names:
                    column_map[original_col] = standard_name
                    break
        
        # Rename columns
        df = df.rename(columns=column_map)
        
        # Log what columns were found/missing
        found_columns = set(column_map.values())
        expected_columns = set(self.COLUMN_MAPPINGS.keys())
        missing_columns = expected_columns - found_columns
        
        if missing_columns:
            logger.warning(f"Missing expected columns: {missing_columns}")
        
        logger.info(f"Standardized columns: {list(found_columns)}")
        return df
    
    def _clean_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text in question and answer columns."""
        logger.info("Cleaning text columns")
        
        for col in ['question', 'answer', 'category']:
            if col in df.columns:
                # Remove HTML tags
                df[col] = df[col].astype(str).str.replace(r'<[^>]+>', '', regex=True)
                
                # Normalize whitespace
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True).str.strip()
                
                # Handle common encoding issues
                df[col] = df[col].str.replace(r'&quot;', '"', regex=False)
                df[col] = df[col].str.replace(r'&amp;', '&', regex=False)
                df[col] = df[col].str.replace(r'&lt;', '<', regex=False)
                df[col] = df[col].str.replace(r'&gt;', '>', regex=False)
        
        return df
    
    def _normalize_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize monetary values."""
        if 'value' not in df.columns:
            return df
        
        logger.info("Normalizing value column")
        
        def parse_value(val):
            """Parse various value formats to numeric."""
            if pd.isna(val):
                return None
                
            val_str = str(val).strip()
            
            # Remove currency symbols and commas
            val_str = re.sub(r'[\$,]', '', val_str)
            
            # Handle "None" or empty strings
            if val_str.lower() in ['none', 'null', '']:
                return None
            
            try:
                return int(val_str)
            except (ValueError, TypeError):
                return None
        
        df['value'] = df['value'].apply(parse_value)
        return df
    
    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse air_date column to datetime."""
        if 'air_date' not in df.columns:
            return df
        
        logger.info("Parsing air_date column")
        
        def parse_date(date_str):
            """Parse various date formats."""
            if pd.isna(date_str):
                return None
            
            date_str = str(date_str).strip()
            
            # Common date formats
            formats = [
                '%Y-%m-%d',
                '%m/%d/%Y',
                '%d/%m/%Y',
                '%Y-%m-%d %H:%M:%S',
                '%B %d, %Y'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt).date()
                except ValueError:
                    continue
            
            return None
        
        df['air_date'] = df['air_date'].apply(parse_date)
        return df
    
    def _filter_invalid_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out records with missing critical data."""
        logger.info("Filtering invalid records")
        
        initial_count = len(df)
        
        # Must have question and answer
        if 'question' in df.columns:
            df = df.dropna(subset=['question'])
            df = df[df['question'].str.len() > 0]
        
        if 'answer' in df.columns:
            df = df.dropna(subset=['answer'])
            df = df[df['answer'].str.len() > 0]
        
        # Filter out common invalid patterns
        if 'question' in df.columns:
            # Remove questions that are too short or clearly invalid
            df = df[df['question'].str.len() >= 10]
            
        if 'answer' in df.columns:
            # Remove answers that are too short
            df = df[df['answer'].str.len() >= 1]
        
        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} invalid records")
        
        return df
    
    def _add_difficulty_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add difficulty level based on monetary value."""
        if 'value' not in df.columns:
            return df
        
        logger.info("Adding difficulty levels")
        
        def categorize_difficulty(value):
            """Categorize difficulty based on dollar value."""
            if pd.isna(value) or value is None:
                return 'Unknown'
            
            if value <= 600:
                return 'Easy'
            elif value <= 1200:
                return 'Medium'
            else:
                return 'Hard'
        
        df['difficulty_level'] = df['value'].apply(categorize_difficulty)
        return df
    
    def _add_metadata_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add metadata columns for tracking."""
        logger.info("Adding metadata columns")
        
        # Add unique question ID if not present
        if 'question_id' not in df.columns:
            df['question_id'] = df.reset_index().index.astype(str)
        
        # Add processing timestamp
        df['processed_at'] = datetime.now()
        
        # Add text length metrics
        if 'question' in df.columns:
            df['question_length'] = df['question'].str.len()
        
        if 'answer' in df.columns:
            df['answer_length'] = df['answer'].str.len()
        
        return df
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the preprocessed dataset and return quality metrics.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Dictionary with validation metrics
            
        Raises:
            ValidationError: If critical validation fails
        """
        try:
            logger.info("Validating preprocessed dataset")
            
            validation_results = {
                'total_records': len(df),
                'required_columns_present': {},
                'data_quality_issues': [],
                'completeness_metrics': {},
                'distribution_stats': {}
            }
            
            # Check required columns
            required_columns = ['question', 'answer']
            for col in required_columns:
                validation_results['required_columns_present'][col] = col in df.columns
                if col not in df.columns:
                    raise ValidationError(
                        f"Required column '{col}' missing from dataset",
                        field_name=col
                    )
            
            # Check data completeness
            for col in df.columns:
                null_count = df[col].isnull().sum()
                null_percentage = (null_count / len(df)) * 100
                validation_results['completeness_metrics'][col] = {
                    'null_count': int(null_count),
                    'null_percentage': round(null_percentage, 2)
                }
            
            # Check for potential quality issues
            if 'question' in df.columns:
                short_questions = len(df[df['question'].str.len() < 20])
                if short_questions > 0:
                    validation_results['data_quality_issues'].append(
                        f"{short_questions} questions are very short (<20 chars)"
                    )
            
            # Distribution statistics
            if 'category' in df.columns:
                validation_results['distribution_stats']['categories'] = {
                    'unique_count': int(df['category'].nunique()),
                    'top_5': df['category'].value_counts().head(5).to_dict()
                }
            
            if 'difficulty_level' in df.columns:
                validation_results['distribution_stats']['difficulty'] = \
                    df['difficulty_level'].value_counts().to_dict()
            
            logger.info("Dataset validation complete")
            return validation_results
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(
                f"Dataset validation failed: {str(e)}"
            ) from e
    
    def filter_by_category(self, df: pd.DataFrame, categories: List[str]) -> pd.DataFrame:
        """
        Filter questions by category.
        
        Args:
            df: DataFrame to filter
            categories: List of categories to include
            
        Returns:
            Filtered DataFrame
        """
        if 'category' not in df.columns:
            logger.warning("No category column found for filtering")
            return df
        
        # Case-insensitive category matching
        categories_lower = [cat.lower() for cat in categories]
        mask = df['category'].str.lower().isin(categories_lower)
        
        filtered_df = df[mask]
        logger.info(f"Filtered by category: {len(filtered_df)} questions remain from {len(df)} original")
        return filtered_df
    
    def filter_by_difficulty(self, df: pd.DataFrame, difficulty_levels: List[str]) -> pd.DataFrame:
        """
        Filter questions by difficulty level.
        
        Args:
            df: DataFrame to filter
            difficulty_levels: List of difficulty levels to include ('Easy', 'Medium', 'Hard')
            
        Returns:
            Filtered DataFrame
        """
        if 'difficulty_level' not in df.columns:
            logger.warning("No difficulty_level column found for filtering")
            return df
        
        mask = df['difficulty_level'].isin(difficulty_levels)
        filtered_df = df[mask]
        logger.info(f"Filtered by difficulty: {len(filtered_df)} questions remain from {len(df)} original")
        return filtered_df
    
    def filter_by_value_range(self, df: pd.DataFrame, min_value: int = None,
                             max_value: int = None) -> pd.DataFrame:
        """
        Filter questions by dollar value range.
        
        Args:
            df: DataFrame to filter
            min_value: Minimum dollar value (inclusive)
            max_value: Maximum dollar value (inclusive)
            
        Returns:
            Filtered DataFrame
        """
        if 'value' not in df.columns:
            logger.warning("No value column found for filtering")
            return df
        
        mask = pd.Series(True, index=df.index)
        
        if min_value is not None:
            mask &= (df['value'] >= min_value)
        if max_value is not None:
            mask &= (df['value'] <= max_value)
        
        filtered_df = df[mask]
        range_str = f"${min_value or 'min'}-${max_value or 'max'}"
        logger.info(f"Filtered by value range {range_str}: {len(filtered_df)} questions remain from {len(df)} original")
        return filtered_df
    
    def filter_by_date_range(self, df: pd.DataFrame, start_date: str = None,
                            end_date: str = None) -> pd.DataFrame:
        """
        Filter questions by air date range.
        
        Args:
            df: DataFrame to filter
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            
        Returns:
            Filtered DataFrame
        """
        if 'air_date' not in df.columns:
            logger.warning("No air_date column found for filtering")
            return df
        
        # Ensure air_date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['air_date']):
            df['air_date'] = pd.to_datetime(df['air_date'], errors='coerce')
        
        mask = pd.Series(True, index=df.index)
        
        if start_date:
            start_dt = pd.to_datetime(start_date)
            mask &= (df['air_date'] >= start_dt)
        if end_date:
            end_dt = pd.to_datetime(end_date)
            mask &= (df['air_date'] <= end_dt)
        
        filtered_df = df[mask]
        range_str = f"{start_date or 'earliest'} to {end_date or 'latest'}"
        logger.info(f"Filtered by date range {range_str}: {len(filtered_df)} questions remain from {len(df)} original")
        return filtered_df
    
    def filter_by_round(self, df: pd.DataFrame, rounds: List[str]) -> pd.DataFrame:
        """
        Filter questions by Jeopardy round.
        
        Args:
            df: DataFrame to filter
            rounds: List of rounds to include (e.g., ['Jeopardy!', 'Double Jeopardy!'])
            
        Returns:
            Filtered DataFrame
        """
        if 'round' not in df.columns:
            logger.warning("No round column found for filtering")
            return df
        
        # Normalize round names
        rounds_normalized = []
        for round_name in rounds:
            if round_name.lower() in ['jeopardy', 'single']:
                rounds_normalized.append('Jeopardy!')
            elif round_name.lower() in ['double', 'double jeopardy']:
                rounds_normalized.append('Double Jeopardy!')
            elif round_name.lower() in ['final', 'final jeopardy']:
                rounds_normalized.append('Final Jeopardy!')
            else:
                rounds_normalized.append(round_name)
        
        mask = df['round'].isin(rounds_normalized)
        filtered_df = df[mask]
        logger.info(f"Filtered by rounds {rounds_normalized}: {len(filtered_df)} questions remain from {len(df)} original")
        return filtered_df
    
    def apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply multiple filters to the dataset.
        
        Args:
            df: DataFrame to filter
            filters: Dictionary of filter criteria
                - categories: List[str] - Categories to include
                - difficulty_levels: List[str] - Difficulty levels to include
                - min_value: int - Minimum dollar value
                - max_value: int - Maximum dollar value
                - start_date: str - Start date (YYYY-MM-DD)
                - end_date: str - End date (YYYY-MM-DD)
                - rounds: List[str] - Jeopardy rounds to include
                
        Returns:
            Filtered DataFrame
        """
        logger.info(f"Applying filters to {len(df)} questions")
        original_count = len(df)
        
        if filters.get('categories'):
            df = self.filter_by_category(df, filters['categories'])
        
        if filters.get('difficulty_levels'):
            df = self.filter_by_difficulty(df, filters['difficulty_levels'])
        
        if filters.get('min_value') is not None or filters.get('max_value') is not None:
            df = self.filter_by_value_range(df, filters.get('min_value'), filters.get('max_value'))
        
        if filters.get('start_date') or filters.get('end_date'):
            df = self.filter_by_date_range(df, filters.get('start_date'), filters.get('end_date'))
        
        if filters.get('rounds'):
            df = self.filter_by_round(df, filters['rounds'])
        
        filtered_count = len(df)
        filter_rate = (filtered_count / original_count) * 100 if original_count > 0 else 0
        logger.info(f"Filtering complete: {filtered_count}/{original_count} questions ({filter_rate:.1f}%) passed filters")
        
        return df