#!/usr/bin/env python3
"""
Test script for Jeopardy score functionality
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that all the enhanced modules import correctly."""
    print("🔍 Testing imports...")
    
    try:
        from evaluation.metrics import MetricsCalculator, JeopardyScoreMetrics, ComprehensiveMetrics
        print("✓ Enhanced metrics module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import metrics: {e}")
        return False
    
    try:
        from storage.models import BenchmarkResult, ModelPerformance
        print("✓ Enhanced storage models imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import storage models: {e}")
        return False
    
    try:
        from benchmark.reporting import ReportGenerator, ReportConfig
        print("✓ Enhanced reporting module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import reporting: {e}")
        return False
        
    try:
        from storage.repositories import PerformanceRepository
        print("✓ Enhanced repository module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import repositories: {e}")
        return False
    
    return True

def test_jeopardy_score_calculation():
    """Test Jeopardy score calculation logic."""
    print("\n🎯 Testing Jeopardy score calculation...")
    
    try:
        from evaluation.metrics import JeopardyScoreMetrics
        
        # Test basic Jeopardy score metrics creation
        jeopardy_metrics = JeopardyScoreMetrics(
            total_jeopardy_score=1500,
            positive_scores=3,
            negative_scores=2,
            category_scores={"SCIENCE": 800, "HISTORY": 700},
            difficulty_scores={"Easy": 600, "Medium": 900}
        )
        
        print(f"✓ JeopardyScoreMetrics created: Total Score = ${jeopardy_metrics.total_jeopardy_score:,}")
        print(f"  - Positive answers: {jeopardy_metrics.positive_scores}")
        print(f"  - Negative answers: {jeopardy_metrics.negative_scores}")
        print(f"  - Category scores: {jeopardy_metrics.category_scores}")
        
        return True
        
    except Exception as e:
        print(f"✗ Jeopardy score calculation test failed: {e}")
        return False

def test_database_models():
    """Test that database models have the new Jeopardy score fields."""
    print("\n🗄️ Testing database model enhancements...")
    
    try:
        from storage.models import BenchmarkResult, ModelPerformance
        
        # Check that BenchmarkResult has jeopardy_score field
        if hasattr(BenchmarkResult, 'jeopardy_score'):
            print("✓ BenchmarkResult has jeopardy_score field")
        else:
            print("✗ BenchmarkResult missing jeopardy_score field")
            return False
            
        # Check that ModelPerformance has jeopardy_score field
        if hasattr(ModelPerformance, 'jeopardy_score'):
            print("✓ ModelPerformance has jeopardy_score field")
        else:
            print("✗ ModelPerformance missing jeopardy_score field")
            return False
            
        # Check that ModelPerformance has category_jeopardy_scores field
        if hasattr(ModelPerformance, 'category_jeopardy_scores'):
            print("✓ ModelPerformance has category_jeopardy_scores field")
        else:
            print("✗ ModelPerformance missing category_jeopardy_scores field")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Database model test failed: {e}")
        return False

def test_reporting_enhancements():
    """Test that reporting has been enhanced for Jeopardy scores."""
    print("\n📊 Testing reporting enhancements...")
    
    try:
        from benchmark.reporting import ReportConfig, ReportGenerator
        
        # Test ReportConfig has show_jeopardy_scores option
        config = ReportConfig(show_jeopardy_scores=True)
        if hasattr(config, 'show_jeopardy_scores') and config.show_jeopardy_scores:
            print("✓ ReportConfig supports show_jeopardy_scores option")
        else:
            print("✗ ReportConfig missing show_jeopardy_scores option")
            return False
            
        # Test ReportGenerator can be created with the config
        generator = ReportGenerator(config=config)
        if hasattr(generator, 'generate_leaderboard_report'):
            print("✓ ReportGenerator has generate_leaderboard_report method")
        else:
            print("✗ ReportGenerator missing generate_leaderboard_report method")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Reporting enhancements test failed: {e}")
        return False

def test_repository_enhancements():
    """Test repository enhancements for Jeopardy scores."""
    print("\n🏪 Testing repository enhancements...")
    
    try:
        from storage.repositories import PerformanceRepository
        
        # Check for new methods
        repo_methods = dir(PerformanceRepository)
        
        expected_methods = [
            'save_performance',
            'get_performances_by_benchmark',
            'get_jeopardy_leaderboard',
            'get_performance_by_model_and_benchmark'
        ]
        
        for method in expected_methods:
            if method in repo_methods:
                print(f"✓ PerformanceRepository has {method} method")
            else:
                print(f"✗ PerformanceRepository missing {method} method")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Repository enhancements test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🎪 Testing Jeopardy Score Functionality Integration")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Jeopardy Score Calculation", test_jeopardy_score_calculation),
        ("Database Models", test_database_models),
        ("Reporting Enhancements", test_reporting_enhancements),
        ("Repository Enhancements", test_repository_enhancements),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("🏁 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! Jeopardy score functionality is properly integrated.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())