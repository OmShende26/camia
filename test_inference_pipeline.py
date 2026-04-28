"""
test_inference_pipeline.py

Quick test to verify the inference pipeline works with sample data.
Run this to validate setup before running on full dataset.
"""

import os
import json
import sys

def test_imports():
    """Test that all required modules can be imported."""
    print("[TEST 1] Checking imports...")
    try:
        import torch
        import numpy as np
        import pandas as pd
        from util_features import collect_all_features
        from mimir.utils import fix_seed
        from mimir.models_without_debugging import LanguageModel
        from mimir.config import ExperimentConfig
        print("✓ All required imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_config_loading():
    """Test that config can be loaded."""
    print("\n[TEST 2] Checking config file...")
    config_path = "conf/config.yaml"
    if os.path.exists(config_path):
        print(f"✓ Config file exists at {config_path}")
        return True
    else:
        print(f"✗ Config file not found at {config_path}")
        return False

def test_jsonl_loading():
    """Test JSONL loading with sample data."""
    print("\n[TEST 3] Testing JSONL loading...")
    
    # Create a small test JSONL file
    test_file = "test_sample.jsonl"
    test_data = [
        {"source_file": "Book1.txt", "text": "This is a test text sample for the first book."},
        {"source_file": "Book1.txt", "text": "Another test text from the same book."},
        {"source_file": "Book2.txt", "text": "This is text from a different book."},
    ]
    
    try:
        with open(test_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        # Try to load it
        from run_inference_detect_books import load_jsonl_with_books
        texts, books = load_jsonl_with_books(test_file)
        
        if len(texts) == 3 and len(books) == 3:
            print(f"✓ Successfully loaded {len(texts)} texts from {len(set(books))} books")
            print(f"  Books: {set(books)}")
            
            # Cleanup
            os.remove(test_file)
            return True
        else:
            print(f"✗ Unexpected data shape: {len(texts)} texts, {len(set(books))} books")
            return False
    except Exception as e:
        print(f"✗ JSONL loading test failed: {e}")
        return False

def test_script_structure():
    """Test that the main script has required functions."""
    print("\n[TEST 4] Checking script structure...")
    try:
        from run_inference_detect_books import (
            load_jsonl_with_books,
            extract_features_with_repeated,
            extract_rep_half_split,
            run_inference,
            aggregate_results_by_book,
            print_results,
        )
        print("✓ All required functions found in script")
        return True
    except ImportError as e:
        print(f"✗ Function import failed: {e}")
        return False

def main():
    print("="*70)
    print("INFERENCE PIPELINE TEST SUITE")
    print("="*70)
    
    tests = [
        test_imports,
        test_config_loading,
        test_jsonl_loading,
        test_script_structure,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "="*70)
    print(f"SUMMARY: {sum(results)}/{len(results)} tests passed")
    print("="*70)
    
    if all(results):
        print("\n✓ All tests passed! Ready to run inference.")
        print("\nUsage:")
        print("  python run_inference_detect_books.py \\")
        print("    --dataset_jsonl <path/to/dataset.jsonl> \\")
        print("    --model_file <path/to/model.pkl> \\")
        print("    --config_path conf/config.yaml")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
