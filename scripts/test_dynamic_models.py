#!/usr/bin/env python3
"""
Dynamic Model System Test Script

Comprehensive testing script that validates:
- API fetching works and returns 300+ models
- Cache system saves and loads correctly
- Three-tier fallback works (API → Cache → Static)
- Default model (Claude 3.5 Sonnet) is correctly set
- Model search functionality works
- All CLI commands work with dynamic models
- Edge cases: API unavailable, cache expiration, invalid models

Usage:
    python scripts/test_dynamic_models.py [--skip-api] [--verbose]
"""

import asyncio
import sys
import os
import time
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import patch
import click

# Add project root to path for imports  
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.models.model_cache import ModelCache, get_model_cache
    from src.models.model_registry import ModelRegistry, model_registry, get_default_model
    from src.models.openrouter import OpenRouterClient
    from src.core.config import get_config
    from src.utils.logging import setup_logging, get_logger
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Make sure you're running from the project root directory")
    print(f"Current directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    print(f"sys.path: {sys.path[:5]}")
    sys.exit(1)

# Test configuration
TEST_RESULTS = {
    'passed': 0,
    'failed': 0,
    'errors': []
}

logger = None  # Initialize later

def print_result(test_name: str, success: bool, message: str = "", verbose: bool = False):
    """Print test result with consistent formatting."""
    icon = "✓" if success else "✗"
    color_start = "\033[92m" if success else "\033[91m"  # Green or Red
    color_end = "\033[0m"
    
    if success:
        TEST_RESULTS['passed'] += 1
    else:
        TEST_RESULTS['failed'] += 1
        TEST_RESULTS['errors'].append(f"{test_name}: {message}")
    
    print(f"{color_start}{icon}{color_end} {test_name}", end="")
    if message and (verbose or not success):
        print(f" - {message}")
    else:
        print("")

def print_section(title: str):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

async def test_basic_cache_functionality(verbose: bool = False):
    """Test basic model cache operations."""
    print_section("Testing Basic Cache Functionality")
    
    # Create temporary cache for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = os.path.join(temp_dir, "test_cache.json")
        cache = ModelCache(cache_path, ttl_seconds=60)
        
        # Test 1: Cache doesn't exist initially
        try:
            cache_info = cache.get_cache_info()
            success = not cache_info['exists']
            print_result("Cache initially empty", success, 
                        f"Expected False, got {cache_info['exists']}", verbose)
        except Exception as e:
            print_result("Cache initially empty", False, f"Error: {e}", verbose)
        
        # Test 2: Save cache with sample data
        try:
            test_models = [
                {
                    'id': 'test/model-1',
                    'name': 'Test Model 1',
                    'provider': 'test',
                    'available': True,
                    'pricing': {'input_cost_per_1m_tokens': 1.0, 'output_cost_per_1m_tokens': 2.0}
                },
                {
                    'id': 'test/model-2', 
                    'name': 'Test Model 2',
                    'provider': 'test',
                    'available': True,
                    'pricing': {'input_cost_per_1m_tokens': 0.5, 'output_cost_per_1m_tokens': 1.0}
                }
            ]
            
            success = cache.save_cache(test_models)
            print_result("Save cache with sample data", success, 
                        "Failed to save cache" if not success else "", verbose)
        except Exception as e:
            print_result("Save cache with sample data", False, f"Error: {e}", verbose)
        
        # Test 3: Load cache and verify data
        try:
            loaded_models = cache.load_cache()
            success = (loaded_models is not None and 
                      len(loaded_models) == 2 and
                      loaded_models[0]['id'] == 'test/model-1')
            print_result("Load cache and verify data", success,
                        f"Expected 2 models, got {len(loaded_models) if loaded_models else 0}", verbose)
        except Exception as e:
            print_result("Load cache and verify data", False, f"Error: {e}", verbose)
        
        # Test 4: Cache info shows correct details
        try:
            cache_info = cache.get_cache_info()
            success = (cache_info['exists'] and 
                      cache_info['valid'] and 
                      cache_info['model_count'] == 2)
            print_result("Cache info shows correct details", success,
                        f"Info: exists={cache_info['exists']}, valid={cache_info['valid']}, count={cache_info['model_count']}", verbose)
        except Exception as e:
            print_result("Cache info shows correct details", False, f"Error: {e}", verbose)
        
        # Test 5: Test cache expiration
        try:
            # Create cache with very short TTL
            short_cache = ModelCache(cache_path, ttl_seconds=1)
            time.sleep(1.1)  # Wait for expiration
            expired_models = short_cache.load_cache()
            success = expired_models is None
            print_result("Cache expiration works", success,
                        "Cache should have expired but didn't" if not success else "", verbose)
        except Exception as e:
            print_result("Cache expiration works", False, f"Error: {e}", verbose)
        
        # Test 6: Clear cache
        try:
            success = cache.clear_cache()
            cache_info = cache.get_cache_info()
            success = success and not cache_info['exists']
            print_result("Clear cache works", success,
                        "Cache should be cleared but still exists" if not success else "", verbose)
        except Exception as e:
            print_result("Clear cache works", False, f"Error: {e}", verbose)

async def test_api_fetching(skip_api: bool = False, verbose: bool = False):
    """Test OpenRouter API fetching functionality."""
    print_section("Testing API Fetching")
    
    if skip_api:
        print("⏭️  Skipping API tests (--skip-api specified)")
        return
    
    # Test 1: Test OpenRouter client creation
    try:
        client = OpenRouterClient()
        success = client is not None
        print_result("OpenRouter client creation", success, "", verbose)
    except Exception as e:
        print_result("OpenRouter client creation", False, f"Error: {e}", verbose)
        return
    
    # Test 2: Fetch models from API
    try:
        models = await client.list_available_models()
        success = models is not None and len(models) > 300
        print_result("Fetch models from API (300+ models)", success,
                    f"Got {len(models) if models else 0} models", verbose)
        
        if models and verbose:
            providers = set(m.get('provider', 'unknown') for m in models)
            print(f"  Found providers: {', '.join(sorted(providers))}")
    except Exception as e:
        print_result("Fetch models from API (300+ models)", False, f"Error: {e}", verbose)
        models = None
    
    # Test 3: Verify model structure
    if models:
        try:
            sample_model = models[0]
            required_fields = ['id', 'name', 'pricing']
            has_required = all(field in sample_model for field in required_fields)
            success = has_required
            print_result("API models have required structure", success,
                        f"Missing fields: {[f for f in required_fields if f not in sample_model]}" if not success else "", verbose)
        except Exception as e:
            print_result("API models have required structure", False, f"Error: {e}", verbose)
    
    # Clean up
    try:
        await client.close()
    except:
        pass

async def test_three_tier_fallback(skip_api: bool = False, verbose: bool = False):
    """Test three-tier fallback system: API → Cache → Static."""
    print_section("Testing Three-Tier Fallback System")
    
    # Test 1: Test model registry fetching (should try API first)
    try:
        registry = ModelRegistry()
        models = await registry.get_available_models()
        success = models is not None and len(models) > 10
        
        # Determine source
        source = "Unknown"
        if not skip_api and len(models) > 300:
            source = "API (fresh)"
        elif 50 < len(models) < 300:
            source = "Cache or mixed"
        elif len(models) <= 20:
            source = "Static fallback"
        else:
            source = f"API or Cache ({len(models)} models)"
            
        print_result("Model registry fetching works", success,
                    f"Got {len(models)} models from {source}", verbose)
    except Exception as e:
        print_result("Model registry fetching works", False, f"Error: {e}", verbose)
        return
    
    # Test 2: Test static fallback by simulating API failure
    try:
        registry = ModelRegistry()
        
        # Mock the fetch_models to fail
        async def mock_fetch_fail():
            return None
        
        with patch.object(registry, 'fetch_models', side_effect=mock_fetch_fail):
            # Also mock cache to fail
            with patch.object(registry._get_cache(), 'load_cache', return_value=None):
                fallback_models = await registry.get_available_models()
                success = (fallback_models is not None and 
                          len(fallback_models) >= 10 and 
                          len(fallback_models) <= 20)  # Should be static models only
                print_result("Static fallback when API/cache fail", success,
                            f"Got {len(fallback_models)} static models", verbose)
    except Exception as e:
        print_result("Static fallback when API/cache fail", False, f"Error: {e}", verbose)
    
    # Test 3: Test cache fallback by simulating API failure but valid cache
    try:
        registry = ModelRegistry()
        
        # Create some mock cached data
        mock_cached_models = [
            {'id': 'cached/model-1', 'name': 'Cached Model 1', 'provider': 'cached', 'available': True},
            {'id': 'cached/model-2', 'name': 'Cached Model 2', 'provider': 'cached', 'available': True}
        ]
        
        async def mock_fetch_fail():
            return None
        
        with patch.object(registry, 'fetch_models', side_effect=mock_fetch_fail):
            with patch.object(registry._get_cache(), 'load_cache', return_value=mock_cached_models):
                cached_models = await registry.get_available_models()
                success = (cached_models is not None and 
                          len(cached_models) == 2 and
                          cached_models[0]['id'] == 'cached/model-1')
                print_result("Cache fallback when API fails", success,
                            f"Got {len(cached_models)} cached models", verbose)
    except Exception as e:
        print_result("Cache fallback when API fails", False, f"Error: {e}", verbose)

async def test_model_search_functionality(verbose: bool = False):
    """Test model search functionality."""
    print_section("Testing Model Search Functionality")
    
    try:
        registry = ModelRegistry()
        
        # Get models to search through
        models = await registry.get_available_models()
        if not models:
            print_result("Model search setup", False, "No models available for testing", verbose)
            return
        
        # Test 1: Search by provider (claude)
        try:
            claude_models = registry.search_models("claude", models)
            success = len(claude_models) > 0
            print_result("Search by provider (claude)", success,
                        f"Found {len(claude_models)} Claude models", verbose)
        except Exception as e:
            print_result("Search by provider (claude)", False, f"Error: {e}", verbose)
        
        # Test 2: Search by model name (gpt)
        try:
            gpt_models = registry.search_models("gpt", models)
            success = len(gpt_models) > 0
            print_result("Search by model name (gpt)", success,
                        f"Found {len(gpt_models)} GPT models", verbose)
        except Exception as e:
            print_result("Search by model name (gpt)", False, f"Error: {e}", verbose)
        
        # Test 3: Search case insensitive
        try:
            upper_models = registry.search_models("CLAUDE", models)
            lower_models = registry.search_models("claude", models)
            success = len(upper_models) == len(lower_models) and len(upper_models) > 0
            print_result("Search case insensitive", success,
                        f"Upper: {len(upper_models)}, Lower: {len(lower_models)}", verbose)
        except Exception as e:
            print_result("Search case insensitive", False, f"Error: {e}", verbose)
        
        # Test 4: Search with no matches
        try:
            no_models = registry.search_models("nonexistent-model-xyz", models)
            success = len(no_models) == 0
            print_result("Search with no matches", success,
                        f"Expected 0, got {len(no_models)}", verbose)
        except Exception as e:
            print_result("Search with no matches", False, f"Error: {e}", verbose)
        
        # Test 5: Empty search returns all models
        try:
            all_models = registry.search_models("", models)
            success = len(all_models) == len(models)
            print_result("Empty search returns all models", success,
                        f"Expected {len(models)}, got {len(all_models)}", verbose)
        except Exception as e:
            print_result("Empty search returns all models", False, f"Error: {e}", verbose)
    
    except Exception as e:
        print_result("Model search setup", False, f"Error: {e}", verbose)

async def test_default_model(verbose: bool = False):
    """Test default model functionality."""
    print_section("Testing Default Model")
    
    # Test 1: Default model is Claude 3.5 Sonnet
    try:
        default_model = get_default_model()
        expected = "anthropic/claude-3.5-sonnet"
        success = default_model == expected
        print_result("Default model is Claude 3.5 Sonnet", success,
                    f"Expected {expected}, got {default_model}", verbose)
    except Exception as e:
        print_result("Default model is Claude 3.5 Sonnet", False, f"Error: {e}", verbose)
    
    # Test 2: Default model exists in available models
    try:
        registry = ModelRegistry()
        models = await registry.get_available_models()
        default_model = get_default_model()
        
        default_found = any(m.get('id', '').lower() == default_model.lower() for m in models)
        success = default_found
        print_result("Default model exists in available models", success,
                    f"Default model {default_model} not found in available models" if not success else "", verbose)
    except Exception as e:
        print_result("Default model exists in available models", False, f"Error: {e}", verbose)

async def test_cli_commands(verbose: bool = False):
    """Test CLI commands work with dynamic models."""
    print_section("Testing CLI Commands")
    
    import subprocess
    import json
    
    def run_cli_command(cmd_args: List[str]) -> tuple[bool, str]:
        """Run CLI command and return success, output."""
        try:
            result = subprocess.run(
                ["python", "-m", "src.main"] + cmd_args,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, f"Command failed: {e}"
    
    # Test 1: models list command
    try:
        success, output = run_cli_command(["models", "list"])
        # Should show models and not crash
        has_models = "model" in output.lower() or "claude" in output.lower() or "gpt" in output.lower()
        success = success and has_models
        print_result("CLI 'models list' command", success,
                    "No recognizable model output" if not has_models else "", verbose)
        if verbose and output:
            print(f"  Output preview: {output[:200]}...")
    except Exception as e:
        print_result("CLI 'models list' command", False, f"Error: {e}", verbose)
    
    # Test 2: models search command
    try:
        success, output = run_cli_command(["models", "search", "claude"])
        has_results = "claude" in output.lower()
        success = success and has_results
        print_result("CLI 'models search claude' command", success,
                    "No Claude models found" if not has_results else "", verbose)
    except Exception as e:
        print_result("CLI 'models search claude' command", False, f"Error: {e}", verbose)
    
    # Test 3: models info command (for default model)
    try:
        success, output = run_cli_command(["models", "info", "anthropic/claude-3.5-sonnet"])
        has_info = "claude" in output.lower() and ("anthropic" in output.lower() or "cost" in output.lower())
        success = success and has_info
        print_result("CLI 'models info' command", success,
                    "No model info found" if not has_info else "", verbose)
    except Exception as e:
        print_result("CLI 'models info' command", False, f"Error: {e}", verbose)
    
    # Test 4: models cache command
    try:
        success, output = run_cli_command(["models", "cache"])
        has_cache_info = "cache" in output.lower() or "ttl" in output.lower() or "models" in output.lower()
        success = success and has_cache_info
        print_result("CLI 'models cache' command", success,
                    "No cache info found" if not has_cache_info else "", verbose)
    except Exception as e:
        print_result("CLI 'models cache' command", False, f"Error: {e}", verbose)
    
    # Test 5: benchmark run --list-models command
    try:
        success, output = run_cli_command(["benchmark", "run", "--list-models"])
        has_model_list = "model" in output.lower() and ("claude" in output.lower() or "gpt" in output.lower())
        success = success and has_model_list
        print_result("CLI 'benchmark run --list-models' command", success,
                    "No model list found" if not has_model_list else "", verbose)
    except Exception as e:
        print_result("CLI 'benchmark run --list-models' command", False, f"Error: {e}", verbose)

async def test_edge_cases(verbose: bool = False):
    """Test edge cases and error handling."""
    print_section("Testing Edge Cases")
    
    # Test 1: Invalid model ID validation
    try:
        registry = ModelRegistry()
        models = await registry.get_available_models()
        
        # Search for completely invalid model
        invalid_results = registry.search_models("definitely-not-a-real-model-12345", models)
        success = len(invalid_results) == 0
        print_result("Invalid model search returns empty", success,
                    f"Expected 0 results, got {len(invalid_results)}", verbose)
    except Exception as e:
        print_result("Invalid model search returns empty", False, f"Error: {e}", verbose)
    
    # Test 2: Test with corrupted cache
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = os.path.join(temp_dir, "corrupted.json")
            
            # Write invalid JSON
            with open(cache_path, 'w') as f:
                f.write("{invalid json content")
            
            cache = ModelCache(cache_path)
            loaded_models = cache.load_cache()
            success = loaded_models is None  # Should gracefully handle corruption
            print_result("Corrupted cache handled gracefully", success,
                        "Should return None for corrupted cache" if not success else "", verbose)
    except Exception as e:
        print_result("Corrupted cache handled gracefully", False, f"Error: {e}", verbose)
    
    # Test 3: Test cache with missing required fields
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = os.path.join(temp_dir, "incomplete.json")
            
            # Write JSON without required 'models' field
            incomplete_data = {"timestamp": time.time()}
            with open(cache_path, 'w') as f:
                json.dump(incomplete_data, f)
            
            cache = ModelCache(cache_path)
            loaded_models = cache.load_cache()
            success = loaded_models is None  # Should reject incomplete cache
            print_result("Incomplete cache structure rejected", success,
                        "Should return None for incomplete cache" if not success else "", verbose)
    except Exception as e:
        print_result("Incomplete cache structure rejected", False, f"Error: {e}", verbose)
    
    # Test 4: Test API failure simulation
    try:
        registry = ModelRegistry()
        
        # Mock OpenRouterClient to raise exception
        async def mock_failing_client():
            raise Exception("Simulated API failure")
        
        with patch('src.models.model_registry.OpenRouterClient') as mock_client:
            mock_client.return_value.list_available_models = mock_failing_client
            
            # Should fall back to cache or static models
            models = await registry.get_available_models()
            success = models is not None and len(models) > 0
            print_result("API failure fallback works", success,
                        f"Got {len(models) if models else 0} fallback models", verbose)
    except Exception as e:
        print_result("API failure fallback works", False, f"Error: {e}", verbose)

async def run_full_test_suite(skip_api: bool = False, verbose: bool = False):
    """Run the complete test suite."""
    print("Dynamic Model System Test Suite")
    print(f"Skip API tests: {skip_api}, Verbose: {verbose}\n")
    
    # Initialize logging
    global logger
    try:
        config = get_config()
        setup_logging(config)
        logger = get_logger(__name__)
    except Exception as e:
        print(f"Warning: Could not initialize logging: {e}")
    
    # Run all test categories
    await test_basic_cache_functionality(verbose)
    await test_api_fetching(skip_api, verbose)
    await test_three_tier_fallback(skip_api, verbose)
    await test_model_search_functionality(verbose)
    await test_default_model(verbose)
    await test_cli_commands(verbose)
    await test_edge_cases(verbose)
    
    # Print summary
    print_section("Test Results Summary")
    total_tests = TEST_RESULTS['passed'] + TEST_RESULTS['failed']
    success_rate = (TEST_RESULTS['passed'] / total_tests * 100) if total_tests > 0 else 0
    
    print(f"✅ Passed: {TEST_RESULTS['passed']}")
    print(f"❌ Failed: {TEST_RESULTS['failed']}")
    print(f"📊 Total: {total_tests}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    if TEST_RESULTS['errors']:
        print(f"\n❌ Failed Tests:")
        for error in TEST_RESULTS['errors']:
            print(f"  • {error}")
    
    # Return overall success
    return TEST_RESULTS['failed'] == 0

@click.command()
@click.option('--skip-api', is_flag=True, help='Skip API-dependent tests')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed output')
def main(skip_api: bool, verbose: bool):
    """Run comprehensive tests for the dynamic model system."""
    # Run the test suite
    success = asyncio.run(run_full_test_suite(skip_api, verbose))
    
    if success:
        print(f"\n🎉 All tests passed! Dynamic model system is working correctly.")
        sys.exit(0)
    else:
        print(f"\n❌ Some tests failed. Please review the results above.")
        sys.exit(1)

if __name__ == "__main__":
    main()