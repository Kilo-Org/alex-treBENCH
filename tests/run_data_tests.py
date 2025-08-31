"""
Test runner for data handling module tests.

Run this script to execute all data-related tests and verify the implementation.
"""

import sys
import subprocess
from pathlib import Path

def run_tests():
    """Run all data handling tests."""
    
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    test_files = [
        "tests/unit/test_data/test_ingestion.py",
        "tests/unit/test_data/test_preprocessing.py", 
        "tests/unit/test_data/test_sampling.py",
        "tests/unit/test_data/test_validation.py",
        "tests/unit/test_data/test_repositories.py",
        "tests/integration/test_data_pipeline.py"
    ]
    
    print("🧪 Running Data Handling Tests")
    print("=" * 50)
    
    all_passed = True
    
    for test_file in test_files:
        print(f"\n📋 Running {test_file}...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                str(project_root / test_file),
                "-v", "--tb=short"
            ], capture_output=True, text=True, cwd=project_root)
            
            if result.returncode == 0:
                print(f"✅ {test_file} - PASSED")
            else:
                print(f"❌ {test_file} - FAILED")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                all_passed = False
                
        except Exception as e:
            print(f"❌ {test_file} - ERROR: {str(e)}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All data handling tests passed!")
    else:
        print("⚠️  Some tests failed. Check output above.")
    
    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)