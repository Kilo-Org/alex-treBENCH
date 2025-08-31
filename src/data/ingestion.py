"""
Data Ingestion Engine

Kaggle dataset loading using kagglehub with caching and preprocessing
for Jeopardy questions dataset.
"""

import os
import shutil
import json
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.core.config import get_config
from src.core.exceptions import DataIngestionError
from src.utils.logging import get_logger

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
        required_columns = ['category', 'question', 'answer']
        
        # Check for common alternative column names
        column_mapping = {
            'clue': 'question',
            'response': 'answer',
            'correct_answer': 'answer',
            'question_text': 'question'
        }
        
        # Apply column name mapping
        df_columns = df.columns.str.lower()
        for old_col, new_col in column_mapping.items():
            if old_col in df_columns and new_col not in df_columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        
        missing_columns = [col for col in required_columns if col not in df.columns.str.lower()]
        
        if missing_columns:
            available_columns = list(df.columns)
            raise DataIngestionError(
                f"Missing required columns: {missing_columns}. Available: {available_columns}",
                dataset_name=self.dataset_name
            )

    def _load_json_file(self, json_path: Path) -> pd.DataFrame:
        """
        Load JSON file and convert to DataFrame.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading JSON data from: {json_path.name}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # Array of objects
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Check if it's a nested structure
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                elif 'questions' in data:
                    df = pd.DataFrame(data['questions'])
                else:
                    # Assume it's a flat dictionary
                    df = pd.DataFrame([data])
            else:
                raise DataIngestionError(
                    f"Unsupported JSON structure in {json_path}",
                    dataset_name=self.dataset_name,
                    file_path=str(json_path)
                )
            
            logger.info(f"Successfully loaded {len(df)} records from JSON")
            return df
            
        except json.JSONDecodeError as e:
            raise DataIngestionError(
                f"Failed to parse JSON file {json_path}: {str(e)}",
                dataset_name=self.dataset_name,
                file_path=str(json_path)
            ) from e
        except Exception as e:
            raise DataIngestionError(
                f"Failed to load JSON file {json_path}: {str(e)}",
                dataset_name=self.dataset_name,
                file_path=str(json_path)
            ) from e

    def download_and_cache_dataset(self, force_download: bool = False) -> Path:
        """
        Download dataset from Kaggle and cache locally.
        
        Args:
            force_download: Force re-download even if cached
            
        Returns:
            Path to cached dataset directory
        """
        cached_path = self._get_cached_dataset_path()
        
        if cached_path.exists() and not force_download:
            logger.info(f"Using cached dataset: {cached_path}")
            return cached_path
        
        try:
            logger.info(f"Downloading dataset: {self.dataset_name}")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task("Downloading dataset...", total=None)
                
                # Download using kagglehub
                download_path = kagglehub.dataset_download(self.dataset_name)
                
                progress.update(task, description="Caching dataset...")
                
                # Move to our cache directory
                if cached_path.exists():
                    shutil.rmtree(cached_path)
                
                shutil.move(download_path, cached_path)
            
            logger.info(f"Dataset cached to: {cached_path}")
            return cached_path
            
        except Exception as e:
            raise DataIngestionError(
                f"Failed to download dataset {self.dataset_name}: {str(e)}",
                dataset_name=self.dataset_name
            ) from e

    def load_dataset(self, data_file: Optional[str] = None, 
                    force_download: bool = False) -> pd.DataFrame:
        """
        Load dataset from Kaggle with automatic caching.
        
        Args:
            data_file: Specific file to load (auto-detected if None)
            force_download: Force re-download even if cached
            
        Returns:
            DataFrame with loaded data
            
        Raises:
            DataIngestionError: If dataset loading fails
        """
        try:
            if data_file:
                # Load specific file
                dataset_path = self.download_and_cache_dataset(force_download)
                file_path = dataset_path / data_file
                
                if not file_path.exists():
                    raise DataIngestionError(
                        f"Data file not found: {data_file}",
                        dataset_name=self.dataset_name,
                        file_path=str(file_path)
                    )
                
                if file_path.suffix.lower() == '.json':
                    df = self._load_json_file(file_path)
                else:
                    df = pd.read_csv(file_path)
                    
            else:
                # Try to auto-detect the main dataset file
                dataset_path = self.download_and_cache_dataset(force_download)
                
                # Auto-detect common Jeopardy file names - now including JSON files
                possible_files = [
                    "jeopardy_questions.json",
                    "jeopardy.json", 
                    "questions.json",
                    "dataset.json",
                    "data.json",
                    "jeopardy_questions.csv",
                    "jeopardy.csv",
                    "questions.csv",
                    "dataset.csv",
                    "data.csv"
                ]
                
                data_file_path: Optional[Path] = None
                for filename in possible_files:
                    candidate = dataset_path / filename
                    if candidate.exists():
                        data_file_path = candidate
                        break
                
                if data_file_path is None:
                    # Try JSON files first, then CSV files
                    json_files = list(dataset_path.glob("*.json"))
                    csv_files = list(dataset_path.glob("*.csv"))
                    
                    if json_files:
                        data_file_path = json_files[0]
                    elif csv_files:
                        data_file_path = csv_files[0]
                    else:
                        available_files = list(dataset_path.glob("*"))
                        raise DataIngestionError(
                            f"No CSV or JSON files found in dataset. Available files: {[f.name for f in available_files]}",
                            dataset_name=self.dataset_name,
                            file_path=str(dataset_path)
                        )
                
                logger.info(f"Loading data from: {data_file_path.name}")
                
                # Load based on file extension
                if data_file_path.suffix.lower() == '.json':
                    df = self._load_json_file(data_file_path)
                else:
                    df = pd.read_csv(data_file_path)
            
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
        """Initialize legacy data ingestion engine."""
        super().__init__(cache_dir)
        logger.warning("DataIngestionEngine is deprecated, use KaggleDatasetLoader instead")
