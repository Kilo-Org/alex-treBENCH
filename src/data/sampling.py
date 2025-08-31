"""
Statistical Sampling

Statistical sampling algorithms for question selection with stratified sampling
by category, difficulty, and temporal distribution for benchmark reliability.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math
from collections import defaultdict

from src.core.config import get_config
from src.core.exceptions import SamplingError
from src.utils.logging import get_logger

logger = get_logger(__name__)


class StatisticalSampler:
    """Implements statistical sampling strategies for benchmark question selection."""
    
    def __init__(self, confidence_level: float = 0.95, margin_of_error: float = 0.05):
        """
        Initialize the statistical sampler.
        
        Args:
            confidence_level: Statistical confidence level (default: 0.95)
            margin_of_error: Acceptable margin of error (default: 0.05)
        """
        self.confidence_level = confidence_level
        self.margin_of_error = margin_of_error
        self.config = get_config()
        
        # Z-scores for common confidence levels
        self.z_scores = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576
        }
        
    def get_sample_size(self, confidence_level: float = None,
                       margin_of_error: float = None) -> int:
        """
        Get the sample size for given confidence level and margin of error.
        
        Args:
            confidence_level: Statistical confidence level (uses instance default if None)
            margin_of_error: Acceptable margin of error (uses instance default if None)
            
        Returns:
            Minimum required sample size
        """
        # Use provided values or fall back to instance defaults
        conf_level = confidence_level or self.confidence_level
        margin = margin_of_error or self.margin_of_error
        
        # Get Z-score for confidence level
        z = self.z_scores.get(conf_level, 1.96)
        
        # Calculate sample size using formula for maximum variance (p=0.5)
        # n = (Z²ₐ/₂ × p × (1-p)) / E²
        numerator = (z ** 2) * 0.5 * 0.5  # p=0.5 for maximum variance
        denominator = margin ** 2
        
        sample_size = int(math.ceil(numerator / denominator))
        
        # Ensure minimum sample size
        return max(sample_size, 30)
    
    def calculate_sample_size(self, population_size: int,
                            expected_proportion: float = 0.5) -> int:
        """
        Calculate statistically valid sample size with finite population correction.
        
        Args:
            population_size: Total population size
            expected_proportion: Expected proportion (0.5 gives max variance)
            
        Returns:
            Minimum required sample size
            
        Raises:
            SamplingError: If calculation fails
        """
        try:
            # Get Z-score for confidence level
            z = self.z_scores.get(self.confidence_level, 1.96)
            
            # Calculate sample size using formula:
            # n = (Z²ₐ/₂ × p × (1-p)) / E²
            numerator = (z ** 2) * expected_proportion * (1 - expected_proportion)
            denominator = self.margin_of_error ** 2
            
            # Base sample size
            n = numerator / denominator
            
            # Apply finite population correction if needed
            if population_size < 100000:  # Apply correction for smaller populations
                n = n / (1 + (n - 1) / population_size)
            
            # Ensure minimum sample size
            min_sample_size = max(int(math.ceil(n)), 30)
            
            # Cap at population size
            final_sample_size = min(min_sample_size, population_size)
            
            logger.info(f"Calculated sample size: {final_sample_size} "
                       f"(population: {population_size}, confidence: {self.confidence_level})")
            
            return final_sample_size
            
        except Exception as e:
            raise SamplingError(
                f"Failed to calculate sample size: {str(e)}",
                sample_size=None,
                population_size=population_size
            ) from e
    
    def stratified_sample(self, df: pd.DataFrame, 
                         sample_size: Optional[int] = None,
                         stratify_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Perform stratified sampling to maintain representativeness.
        
        Args:
            df: Source DataFrame
            sample_size: Desired sample size (calculated if None)
            stratify_columns: Columns to stratify by
            
        Returns:
            Stratified sample DataFrame
            
        Raises:
            SamplingError: If sampling fails
        """
        try:
            if len(df) == 0:
                raise SamplingError("Cannot sample from empty dataset")
            
            # Calculate sample size if not provided
            if sample_size is None:
                sample_size = self.calculate_sample_size(len(df))
            
            # Default stratification columns
            if stratify_columns is None:
                stratify_columns = []
                if 'difficulty_level' in df.columns:
                    stratify_columns.append('difficulty_level')
                if 'category' in df.columns and df['category'].nunique() <= 50:
                    stratify_columns.append('category')
            
            logger.info(f"Performing stratified sampling with columns: {stratify_columns}")
            
            # If no stratification columns, use simple random sampling
            if not stratify_columns:
                return self._simple_random_sample(df, sample_size)
            
            # Create stratification groups
            df_with_strata = df.copy()
            if stratify_columns:
                # Combine stratification columns into a single stratum identifier
                df_with_strata['stratum'] = df_with_strata[stratify_columns].apply(
                    lambda x: '|'.join(x.astype(str)), axis=1
                )
            else:
                df_with_strata['stratum'] = 'all'
            
            # Calculate strata proportions
            strata_counts = df_with_strata['stratum'].value_counts()
            strata_proportions = strata_counts / len(df)
            
            # Allocate sample sizes to strata proportionally
            samples_per_stratum = {}
            allocated_samples = 0
            
            for stratum, proportion in strata_proportions.items():
                stratum_sample_size = max(1, int(sample_size * proportion))
                stratum_population = strata_counts[stratum]
                
                # Don't sample more than available
                stratum_sample_size = min(stratum_sample_size, stratum_population)
                samples_per_stratum[stratum] = stratum_sample_size
                allocated_samples += stratum_sample_size
            
            # Handle rounding differences by adjusting largest strata
            if allocated_samples < sample_size:
                remaining = sample_size - allocated_samples
                largest_strata = strata_counts.head(remaining).index
                
                for stratum in largest_strata:
                    if samples_per_stratum[stratum] < strata_counts[stratum]:
                        samples_per_stratum[stratum] += 1
                        remaining -= 1
                        if remaining == 0:
                            break
            
            # Sample from each stratum
            sampled_dfs = []
            for stratum, n_samples in samples_per_stratum.items():
                stratum_data = df_with_strata[df_with_strata['stratum'] == stratum]
                
                if len(stratum_data) >= n_samples:
                    sampled_stratum = stratum_data.sample(n=n_samples, random_state=42)
                else:
                    # Take all available if stratum is smaller than required sample
                    sampled_stratum = stratum_data
                
                sampled_dfs.append(sampled_stratum)
            
            # Combine samples
            result_df = pd.concat(sampled_dfs, ignore_index=True)
            result_df = result_df.drop('stratum', axis=1)
            
            logger.info(f"Stratified sampling complete: {len(result_df)} questions selected")
            return result_df
            
        except Exception as e:
            if isinstance(e, SamplingError):
                raise
            raise SamplingError(
                f"Stratified sampling failed: {str(e)}",
                sample_size=sample_size,
                population_size=len(df)
            ) from e
    
    def random_sample(self, df: pd.DataFrame, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        """
        Perform simple random sampling.
        
        Args:
            df: Source DataFrame
            n: Sample size
            seed: Random seed for reproducibility
            
        Returns:
            Randomly sampled DataFrame
            
        Raises:
            SamplingError: If sampling fails
        """
        try:
            if len(df) == 0:
                raise SamplingError("Cannot sample from empty dataset")
            
            # Use provided seed or default
            random_state = seed if seed is not None else 42
            
            # Cap sample size at population size
            actual_sample_size = min(n, len(df))
            
            if actual_sample_size != n:
                logger.warning(f"Requested sample size {n} reduced to {actual_sample_size} (population limit)")
            
            sampled_df = df.sample(n=actual_sample_size, random_state=random_state)
            logger.info(f"Random sampling complete: {len(sampled_df)} questions selected")
            
            return sampled_df
            
        except Exception as e:
            if isinstance(e, SamplingError):
                raise
            raise SamplingError(
                f"Random sampling failed: {str(e)}",
                sample_size=n,
                population_size=len(df)
            ) from e
    
    def _simple_random_sample(self, df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Internal method for simple random sampling."""
        return self.random_sample(df, sample_size)
    
    def temporal_stratified_sample(self, df: pd.DataFrame,
                                 sample_size: Optional[int] = None,
                                 date_column: str = 'air_date',
                                 time_periods: int = 5) -> pd.DataFrame:
        """
        Perform temporal stratified sampling to ensure representation across time periods.
        
        Args:
            df: Source DataFrame with date column
            sample_size: Desired sample size
            date_column: Name of the date column
            time_periods: Number of time periods to create
            
        Returns:
            Temporally stratified sample
        """
        try:
            if date_column not in df.columns:
                logger.warning(f"Date column '{date_column}' not found, using regular stratified sampling")
                return self.stratified_sample(df, sample_size)
            
            # Remove records without valid dates
            df_with_dates = df.dropna(subset=[date_column])
            
            if len(df_with_dates) == 0:
                logger.warning("No valid dates found, using regular stratified sampling")
                return self.stratified_sample(df, sample_size)
            
            # Create time period bins
            df_with_dates = df_with_dates.sort_values(date_column)
            df_with_dates['time_period'] = pd.cut(
                range(len(df_with_dates)), 
                bins=time_periods, 
                labels=[f"period_{i}" for i in range(time_periods)]
            )
            
            # Add time period to stratification
            other_columns = []
            if 'difficulty_level' in df_with_dates.columns:
                other_columns.append('difficulty_level')
            
            stratify_columns = ['time_period'] + other_columns
            
            result = self.stratified_sample(df_with_dates, sample_size, stratify_columns)
            result = result.drop('time_period', axis=1)
            
            return result
            
        except Exception as e:
            raise SamplingError(
                f"Temporal stratified sampling failed: {str(e)}",
                sample_size=sample_size,
                population_size=len(df)
            ) from e
    
    def balanced_difficulty_sample(self, df: pd.DataFrame,
                                 sample_size: Optional[int] = None,
                                 difficulty_distribution: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Sample with specified difficulty distribution.
        
        Args:
            df: Source DataFrame
            sample_size: Desired sample size
            difficulty_distribution: Target distribution by difficulty
            
        Returns:
            Difficulty-balanced sample
        """
        try:
            if 'difficulty_level' not in df.columns:
                logger.warning("No difficulty_level column found, using regular sampling")
                return self.stratified_sample(df, sample_size)
            
            if sample_size is None:
                sample_size = self.calculate_sample_size(len(df))
            
            # Default balanced distribution
            if difficulty_distribution is None:
                difficulty_distribution = {
                    'Easy': 0.4,
                    'Medium': 0.4, 
                    'Hard': 0.2
                }
            
            # Sample by difficulty level
            sampled_dfs = []
            for difficulty, proportion in difficulty_distribution.items():
                difficulty_data = df[df['difficulty_level'] == difficulty]
                
                if len(difficulty_data) == 0:
                    continue
                
                target_count = int(sample_size * proportion)
                actual_count = min(target_count, len(difficulty_data))
                
                if actual_count > 0:
                    sampled_dfs.append(
                        difficulty_data.sample(n=actual_count, random_state=42)
                    )
            
            result_df = pd.concat(sampled_dfs, ignore_index=True)
            
            logger.info(f"Balanced difficulty sampling complete: {len(result_df)} questions")
            return result_df
            
        except Exception as e:
            raise SamplingError(
                f"Balanced difficulty sampling failed: {str(e)}",
                sample_size=sample_size,
                population_size=len(df)
            ) from e
    
    def get_sampling_statistics(self, original_df: pd.DataFrame, 
                              sampled_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate statistics comparing original and sampled datasets.
        
        Args:
            original_df: Original dataset
            sampled_df: Sampled dataset
            
        Returns:
            Dictionary with sampling statistics
        """
        stats = {
            'original_size': len(original_df),
            'sample_size': len(sampled_df),
            'sampling_ratio': len(sampled_df) / len(original_df) if len(original_df) > 0 else 0,
            'representativeness': {}
        }
        
        # Check representativeness for categorical columns
        for col in ['category', 'difficulty_level', 'round']:
            if col in original_df.columns and col in sampled_df.columns:
                original_dist = original_df[col].value_counts(normalize=True).sort_index()
                sample_dist = sampled_df[col].value_counts(normalize=True).sort_index()
                
                # Calculate distribution differences
                common_values = set(original_dist.index) & set(sample_dist.index)
                if common_values:
                    differences = []
                    for value in common_values:
                        diff = abs(original_dist.get(value, 0) - sample_dist.get(value, 0))
                        differences.append(diff)
                    
                    stats['representativeness'][col] = {
                        'mean_difference': np.mean(differences),
                        'max_difference': np.max(differences),
                        'original_unique': len(original_dist),
                        'sample_unique': len(sample_dist)
                    }
        
        return stats