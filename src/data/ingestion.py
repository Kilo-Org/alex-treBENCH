"""
Data Ingestion Engine

Kaggle dataset loading using kagglehub with caching and preprocessing
for Jeopardy questions dataset.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..core.config import get_config
from ..core.exceptions import DataIngestionError
from ..utils.logging import get_logger

logger = get_logger(__name__)


class KaggleDatasetLoader:
    """Handles loading and caching of Kaggle datasets using the new kagglehub API."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the Kaggle dataset loader.
        
        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.config = get_config()
        self.cache_dir = cache_dir or Path(self.config.kaggle.cache_dir)
        self.dataset_name = self.config.kaggle.dataset
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cached_dataset_path(self) -> Path:
        """Get the expected path for cached dataset."""
        return self.cache_dir / self.dataset_name.replace('/', '_')
        
    def _validate_dataset_requirements(self, df: pd.DataFrame) -> None:
        """
        Validate that the dataset has required columns.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            DataIngestionError: If required columns are missing
        """
        required_columns = ['question', 'answer']
        column_names_lower = [col.lower() for col in df.columns]
        
        missing_columns = []
        for req_col in required_columns:
            # Check for exact match or common variations
            variations = {
                'question': ['question', 'clue', 'prompt', 'text'],
                'answer': ['answer', 'response', 'correct_answer', 'solution']
            }
            
            if not any(var in column_names_lower for var in variations[req_col]):
                missing_columns.append(req_col)
        
        if missing_columns:
            available_cols = ', '.join(df.columns)
            raise DataIngestionError(
                f"Dataset missing required columns: {missing_columns}. "
                f"Available columns: {available_cols}",
                dataset_name=self.dataset_name
            )
        
        logger.info(f"Dataset validation passed. Found {len(df)} records with columns: {list(df.columns)}")
    
    def download_and_cache_dataset(self, force_download: bool = False) -> Path:
        """
        Download Jeopardy dataset from Kaggle with caching.
        
        Args:
            force_download: Force re-download even if cached
            
        Returns:
            Path to cached dataset directory
            
        Raises:
            DataIngestionError: If download fails
        """
        try:
            cached_path = self._get_cached_dataset_path()
            
            # Check if dataset already exists and is valid
            if cached_path.exists() and not force_download:
                # Verify cache integrity by checking if any CSV files exist
                csv_files = list(cached_path.glob("*.csv"))
                if csv_files:
                    logger.info(f"Using cached dataset at: {cached_path}")
                    return cached_path
                else:
                    logger.warning("Cached dataset directory exists but no CSV files found. Re-downloading...")
                    shutil.rmtree(cached_path)
            
            logger.info(f"Downloading dataset: {self.dataset_name}")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
            ) as progress:
                progress.add_task(f"Downloading {self.dataset_name}...", total=None)
                
                # Download using kagglehub
                dataset_path = kagglehub.dataset_download(self.dataset_name)
                dataset_path = Path(dataset_path)
            
            # Move to our cache directory if not already there
            if dataset_path != cached_path:
                if cached_path.exists():
                    shutil.rmtree(cached_path)
                shutil.move(str(dataset_path), str(cached_path))
                dataset_path = cached_path
            
            logger.info(f"Dataset cached successfully at: {dataset_path}")
            return dataset_path
            
        except Exception as e:
            raise DataIngestionError(
                f"Failed to download dataset '{self.dataset_name}': {str(e)}",
                dataset_name=self.dataset_name
            ) from e
    
    def load_dataset(self, file_name: Optional[str] = None,
                    force_download: bool = False) -> pd.DataFrame:
        """
        Load the Jeopardy dataset into a pandas DataFrame using kagglehub.
        
        Args:
            file_name: Specific file to load (if None, auto-detect)
            force_download: Force re-download even if cached
            
        Returns:
            DataFrame with Jeopardy questions and answers
            
        Raises:
            DataIngestionError: If loading fails
        """
        try:
            logger.info(f"Loading dataset: {self.dataset_name}")
            
            if file_name:
                # Load specific file using kagglehub
                logger.info(f"Loading specific file: {file_name}")
                df = kagglehub.load_dataset(
                    KaggleDatasetAdapter.PANDAS,
                    self.dataset_name,
                    file_name
                )
            else:
                # Try to auto-detect the main dataset file
                dataset_path = self.download_and_cache_dataset(force_download)
                
                # Auto-detect common Jeopardy file names
                possible_files = [
                    "jeopardy_questions.csv",
                    "jeopardy.csv",
                    "questions.csv",
                    "dataset.csv",
                    "data.csv"
                ]
                
                data_file = None
                for filename in possible_files:
                    candidate = dataset_path / filename
                    if candidate.exists():
                        data_file = candidate
                        break
                
                if data_file is None:
                    # Use first CSV file found
                    csv_files = list(dataset_path.glob("*.csv"))
                    if csv_files:
                        data_file = csv_files[0]
                    else:
                        raise DataIngestionError(
                            "No CSV files found in dataset",
                            dataset_name=self.dataset_name,
                            file_path=str(dataset_path)
                        )
                
                logger.info(f"Loading data from: {data_file.name}")
                
                # Use kagglehub to load the detected file
                df = kagglehub.load_dataset(
                    KaggleDatasetAdapter.PANDAS,
                    self.dataset_name,
                    data_file.name
                )
            
            # Validate dataset requirements
            self._validate_dataset_requirements(df)
            
            logger.info(f"Successfully loaded {len(df)} questions from dataset")
            return df
            
        except Exception as e:
            if isinstance(e, DataIngestionError):
                raise
            raise DataIngestionError(
                f"Failed to load dataset: {str(e)}",
                dataset_name=self.dataset_name
            ) from e


# Legacy class for backward compatibility
class DataIngestionEngine(KaggleDatasetLoader):
    """Legacy class - use KaggleDatasetLoader instead."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        logger.warning("DataIngestionEngine is deprecated. Use KaggleDatasetLoader instead.")
        super().__init__(cache_dir)
    
    def download_dataset(self, force_download: bool = False) -> Path:
        """Legacy method - use download_and_cache_dataset instead."""
        return self.download_and_cache_dataset(force_download)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get metadata about the loaded dataset.
        
        Returns:
            Dictionary with dataset information
        """
        try:
            df = self.load_dataset()
            
            info = {
                "total_questions": len(df),
                "columns": list(df.columns),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "null_counts": df.isnull().sum().to_dict(),
                "data_types": df.dtypes.to_dict()
            }
            
            # Add specific Jeopardy dataset info if available
            if 'category' in df.columns:
                info["unique_categories"] = df['category'].nunique()
                info["top_categories"] = df['category'].value_counts().head(10).to_dict()
            
            if 'value' in df.columns:
                info["value_range"] = {
                    "min": df['value'].min(),
                    "max": df['value'].max(),
                    "mean": df['value'].mean()
                }
            
            return info
            
        except Exception as e:
            raise DataIngestionError(
                f"Failed to get dataset info: {str(e)}",
                dataset_name=self.dataset_name
            ) from e
    
    def clear_cache(self) -> None:
        """Clear the dataset cache."""
        try:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Dataset cache cleared")
        except Exception as e:
            raise DataIngestionError(
                f"Failed to clear cache: {str(e)}",
                file_path=str(self.cache_dir)
            ) from e