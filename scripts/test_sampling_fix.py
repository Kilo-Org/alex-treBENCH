#!/usr/bin/env python3
"""
Comprehensive Test Script for Sampling Methodology Fixes

This script verifies that the sampling methodology fixes are working correctly by testing:
1. Random sampling without a seed (should produce different results each run)
2. Random sampling with a seed (should produce same results each run)
3. Stratified sampling without a seed (should produce different results each run)
4. Stratified sampling with a seed (should produce same results each run)
5. Balanced sampling without a seed (should produce different results each run)
6. Balanced sampling with a seed (should produce same results each run)

Usage:
    python scripts/test_sampling_fix.py
    python scripts/test_sampling_fix.py --verbose
    python scripts/test_sampling_fix.py --sample-size 50
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import json

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.sampling import StatisticalSampler
from src.core.exceptions import SamplingError

# Color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(60)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}")


def print_test_result(test_name: str, passed: bool, details: str = "") -> None:
    """Print a test result with color formatting."""
    status = f"{Colors.GREEN}✓ PASS{Colors.RESET}" if passed else f"{Colors.RED}✗ FAIL{Colors.RESET}"
    print(f"{status} {Colors.BOLD}{test_name}{Colors.RESET}")
    if details:
        print(f"      {Colors.YELLOW}{details}{Colors.RESET}")


def print_info(text: str) -> None:
    """Print informational text."""
    print(f"{Colors.BLUE}ℹ  {text}{Colors.RESET}")


def print_warning(text: str) -> None:
    """Print warning text."""
    print(f"{Colors.YELLOW}⚠  {text}{Colors.RESET}")


def create_test_dataset(size: int = 1000) -> pd.DataFrame:
    """Create a comprehensive test dataset with various categories and difficulty levels."""
    np.random.seed(42)  # Fixed seed for consistent test data creation
    
    categories = [
        'SCIENCE', 'HISTORY', 'LITERATURE', 'GEOGRAPHY', 'SPORTS',
        'ENTERTAINMENT', 'ARTS', 'MATHEMATICS', 'POLITICS', 'TECHNOLOGY'
    ]
    
    difficulties = ['Easy', 'Medium', 'Hard']
    rounds = ['Jeopardy!', 'Double Jeopardy!', 'Final Jeopardy!']
    values = [200, 400, 600, 800, 1000]
    
    data = []
    for i in range(size):
        data.append({
            'question_id': f'test_q_{i:04d}',
            'question': f'This is test question number {i}?',
            'answer': f'What is test answer {i}?',
            'category': np.random.choice(categories),
            'difficulty_level': np.random.choice(difficulties),
            'value': np.random.choice(values),
            'round': np.random.choice(rounds),
            'air_date': pd.Timestamp('2020-01-01') + pd.Timedelta(days=np.random.randint(0, 365*4)),
            'show_number': f'{8000 + np.random.randint(0, 1000)}'
        })
    
    df = pd.DataFrame(data)
    print_info(f"Created test dataset with {len(df)} questions")
    print_info(f"Categories: {df['category'].nunique()}, Difficulties: {df['difficulty_level'].nunique()}")
    
    return df


def are_dataframes_identical(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    """Check if two DataFrames are identical (same rows, same order)."""
    try:
        pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))
        return True
    except AssertionError:
        return False


def are_dataframes_different(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    """Check if two DataFrames contain different rows (allowing for different order)."""
    # Sort both DataFrames by question_id to compare content regardless of order
    df1_sorted = df1.sort_values('question_id').reset_index(drop=True)
    df2_sorted = df2.sort_values('question_id').reset_index(drop=True)
    
    # If they have different lengths, they're definitely different
    if len(df1_sorted) != len(df2_sorted):
        return True
    
    # Compare the actual question IDs selected
    return not df1_sorted['question_id'].equals(df2_sorted['question_id'])


def test_random_sampling_randomness(dataset: pd.DataFrame, sample_size: int, verbose: bool = False) -> bool:
    """Test that random sampling without seed produces different results."""
    print_info("Testing random sampling randomness (no seed)...")
    
    sampler = StatisticalSampler()
    
    # Take multiple samples without seed
    samples = []
    for i in range(5):
        try:
            sample = sampler.random_sample(dataset, sample_size, seed=None)
            samples.append(sample)
            if verbose:
                print(f"      Sample {i+1}: {len(sample)} questions, first ID: {sample.iloc[0]['question_id']}")
        except Exception as e:
            print_test_result(f"Random Sampling Randomness (Run {i+1})", False, f"Exception: {str(e)}")
            return False
    
    # Check that at least some samples are different
    different_pairs = 0
    total_pairs = 0
    
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            total_pairs += 1
            if are_dataframes_different(samples[i], samples[j]):
                different_pairs += 1
    
    success = different_pairs > 0
    
    if verbose:
        print(f"      {different_pairs}/{total_pairs} sample pairs were different")
    
    return success


def test_random_sampling_reproducibility(dataset: pd.DataFrame, sample_size: int, verbose: bool = False) -> bool:
    """Test that random sampling with seed produces identical results."""
    print_info("Testing random sampling reproducibility (with seed)...")
    
    sampler = StatisticalSampler()
    seed = 12345
    
    # Take multiple samples with same seed
    samples = []
    for i in range(3):
        try:
            sample = sampler.random_sample(dataset, sample_size, seed=seed)
            samples.append(sample)
            if verbose:
                print(f"      Sample {i+1}: {len(sample)} questions, first ID: {sample.iloc[0]['question_id']}")
        except Exception as e:
            print_test_result(f"Random Sampling Reproducibility (Run {i+1})", False, f"Exception: {str(e)}")
            return False
    
    # Check that all samples are identical
    all_identical = True
    for i in range(1, len(samples)):
        if not are_dataframes_identical(samples[0], samples[i]):
            all_identical = False
            break
    
    return all_identical


def test_stratified_sampling_randomness(dataset: pd.DataFrame, sample_size: int, verbose: bool = False) -> bool:
    """Test that stratified sampling without seed produces different results."""
    print_info("Testing stratified sampling randomness (no seed)...")
    
    sampler = StatisticalSampler()
    
    # Take multiple samples without seed
    samples = []
    for i in range(5):
        try:
            sample = sampler.stratified_sample(dataset, sample_size, seed=None)
            samples.append(sample)
            if verbose:
                categories = sample['category'].value_counts().to_dict()
                print(f"      Sample {i+1}: {len(sample)} questions, categories: {list(categories.keys())[:3]}...")
        except Exception as e:
            print_test_result(f"Stratified Sampling Randomness (Run {i+1})", False, f"Exception: {str(e)}")
            return False
    
    # Check that at least some samples are different
    different_pairs = 0
    total_pairs = 0
    
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            total_pairs += 1
            if are_dataframes_different(samples[i], samples[j]):
                different_pairs += 1
    
    success = different_pairs > 0
    
    if verbose:
        print(f"      {different_pairs}/{total_pairs} sample pairs were different")
    
    return success


def test_stratified_sampling_reproducibility(dataset: pd.DataFrame, sample_size: int, verbose: bool = False) -> bool:
    """Test that stratified sampling with seed produces identical results."""
    print_info("Testing stratified sampling reproducibility (with seed)...")
    
    sampler = StatisticalSampler()
    seed = 12345
    
    # Take multiple samples with same seed
    samples = []
    for i in range(3):
        try:
            sample = sampler.stratified_sample(dataset, sample_size, seed=seed)
            samples.append(sample)
            if verbose:
                categories = sample['category'].value_counts().to_dict()
                print(f"      Sample {i+1}: {len(sample)} questions, categories: {list(categories.keys())[:3]}...")
        except Exception as e:
            print_test_result(f"Stratified Sampling Reproducibility (Run {i+1})", False, f"Exception: {str(e)}")
            return False
    
    # Check that all samples are identical
    all_identical = True
    for i in range(1, len(samples)):
        if not are_dataframes_identical(samples[0], samples[i]):
            all_identical = False
            break
    
    return all_identical


def test_balanced_sampling_randomness(dataset: pd.DataFrame, sample_size: int, verbose: bool = False) -> bool:
    """Test that balanced difficulty sampling without seed produces different results."""
    print_info("Testing balanced difficulty sampling randomness (no seed)...")
    
    sampler = StatisticalSampler()
    
    # Take multiple samples without seed
    samples = []
    for i in range(5):
        try:
            sample = sampler.balanced_difficulty_sample(dataset, sample_size, seed=None)
            samples.append(sample)
            if verbose:
                difficulties = sample['difficulty_level'].value_counts().to_dict()
                print(f"      Sample {i+1}: {len(sample)} questions, difficulties: {difficulties}")
        except Exception as e:
            print_test_result(f"Balanced Sampling Randomness (Run {i+1})", False, f"Exception: {str(e)}")
            return False
    
    # Check that at least some samples are different
    different_pairs = 0
    total_pairs = 0
    
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            total_pairs += 1
            if are_dataframes_different(samples[i], samples[j]):
                different_pairs += 1
    
    success = different_pairs > 0
    
    if verbose:
        print(f"      {different_pairs}/{total_pairs} sample pairs were different")
    
    return success


def test_balanced_sampling_reproducibility(dataset: pd.DataFrame, sample_size: int, verbose: bool = False) -> bool:
    """Test that balanced difficulty sampling with seed produces identical results."""
    print_info("Testing balanced difficulty sampling reproducibility (with seed)...")
    
    sampler = StatisticalSampler()
    seed = 12345
    
    # Take multiple samples with same seed
    samples = []
    for i in range(3):
        try:
            sample = sampler.balanced_difficulty_sample(dataset, sample_size, seed=seed)
            samples.append(sample)
            if verbose:
                difficulties = sample['difficulty_level'].value_counts().to_dict()
                print(f"      Sample {i+1}: {len(sample)} questions, difficulties: {difficulties}")
        except Exception as e:
            print_test_result(f"Balanced Sampling Reproducibility (Run {i+1})", False, f"Exception: {str(e)}")
            return False
    
    # Check that all samples are identical
    all_identical = True
    for i in range(1, len(samples)):
        if not are_dataframes_identical(samples[0], samples[i]):
            all_identical = False
            break
    
    return all_identical


def run_all_tests(sample_size: int = 50, verbose: bool = False) -> Dict[str, bool]:
    """Run all sampling tests and return results."""
    print_header("SAMPLING METHODOLOGY FIX VERIFICATION")
    print_info(f"Sample size for tests: {sample_size}")
    print_info(f"Verbose mode: {'ON' if verbose else 'OFF'}")
    
    # Create test dataset
    print_info("\n" + "="*40)
    dataset = create_test_dataset(size=1000)
    
    # Results tracking
    results = {}
    
    # Test Random Sampling
    print_header("RANDOM SAMPLING TESTS")
    
    results['random_randomness'] = test_random_sampling_randomness(dataset, sample_size, verbose)
    print_test_result(
        "Random Sampling - Randomness (No Seed)",
        results['random_randomness'],
        "Different results each run" if results['random_randomness'] else "Results were identical (UNEXPECTED)"
    )
    
    results['random_reproducibility'] = test_random_sampling_reproducibility(dataset, sample_size, verbose)
    print_test_result(
        "Random Sampling - Reproducibility (With Seed)",
        results['random_reproducibility'],
        "Same results each run" if results['random_reproducibility'] else "Results were different (UNEXPECTED)"
    )
    
    # Test Stratified Sampling
    print_header("STRATIFIED SAMPLING TESTS")
    
    results['stratified_randomness'] = test_stratified_sampling_randomness(dataset, sample_size, verbose)
    print_test_result(
        "Stratified Sampling - Randomness (No Seed)",
        results['stratified_randomness'],
        "Different results each run" if results['stratified_randomness'] else "Results were identical (UNEXPECTED)"
    )
    
    results['stratified_reproducibility'] = test_stratified_sampling_reproducibility(dataset, sample_size, verbose)
    print_test_result(
        "Stratified Sampling - Reproducibility (With Seed)",
        results['stratified_reproducibility'],
        "Same results each run" if results['stratified_reproducibility'] else "Results were different (UNEXPECTED)"
    )
    
    # Test Balanced Difficulty Sampling
    print_header("BALANCED DIFFICULTY SAMPLING TESTS")
    
    results['balanced_randomness'] = test_balanced_sampling_randomness(dataset, sample_size, verbose)
    print_test_result(
        "Balanced Sampling - Randomness (No Seed)",
        results['balanced_randomness'],
        "Different results each run" if results['balanced_randomness'] else "Results were identical (UNEXPECTED)"
    )
    
    results['balanced_reproducibility'] = test_balanced_sampling_reproducibility(dataset, sample_size, verbose)
    print_test_result(
        "Balanced Sampling - Reproducibility (With Seed)",
        results['balanced_reproducibility'],
        "Same results each run" if results['balanced_reproducibility'] else "Results were different (UNEXPECTED)"
    )
    
    return results


def print_summary(results: Dict[str, bool]) -> None:
    """Print a summary of all test results."""
    print_header("TEST SUMMARY")
    
    passed_tests = sum(1 for result in results.values() if result)
    total_tests = len(results)
    
    print(f"{Colors.BOLD}Total Tests: {total_tests}{Colors.RESET}")
    print(f"{Colors.GREEN}Passed: {passed_tests}{Colors.RESET}")
    print(f"{Colors.RED}Failed: {total_tests - passed_tests}{Colors.RESET}")
    
    success_rate = (passed_tests / total_tests) * 100
    print(f"{Colors.BOLD}Success Rate: {success_rate:.1f}%{Colors.RESET}")
    
    if passed_tests == total_tests:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED! 🎉{Colors.RESET}")
        print(f"{Colors.GREEN}Sampling methodology fixes are working correctly!{Colors.RESET}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ SOME TESTS FAILED ❌{Colors.RESET}")
        print(f"{Colors.RED}Sampling methodology may need additional fixes.{Colors.RESET}")
        
        # Show which tests failed
        print(f"\n{Colors.YELLOW}Failed Tests:{Colors.RESET}")
        for test_name, result in results.items():
            if not result:
                print(f"  {Colors.RED}• {test_name}{Colors.RESET}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test script to verify sampling methodology fixes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_sampling_fix.py
  python scripts/test_sampling_fix.py --verbose
  python scripts/test_sampling_fix.py --sample-size 100 --verbose
        """
    )
    
    parser.add_argument(
        '--sample-size', 
        type=int, 
        default=50,
        help='Sample size to use for testing (default: 50)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output with detailed information'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.sample_size < 10:
        print_warning("Sample size too small, using minimum of 10")
        args.sample_size = 10
    elif args.sample_size > 200:
        print_warning("Sample size very large, consider using smaller size for faster testing")
    
    try:
        results = run_all_tests(sample_size=args.sample_size, verbose=args.verbose)
        print_summary(results)
        
        # Exit with appropriate code
        if all(results.values()):
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure
            
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Test interrupted by user{Colors.RESET}")
        sys.exit(2)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {str(e)}{Colors.RESET}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()