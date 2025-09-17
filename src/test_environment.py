#!/usr/bin/env python3
"""Test script to verify the Spite AI environment is working."""

import sys
import os

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.spite_ai import Config
        print("✅ Config imported successfully")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from src.spite_ai import SpiteAI
        print("✅ SpiteAI imported successfully (lazy load)")
    except Exception as e:
        print(f"❌ SpiteAI import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration functionality."""
    print("\nTesting configuration...")
    
    try:
        from src.spite_ai import Config
        config = Config()
        
        # Test basic config properties
        assert config.DEFAULT_K == 8
        assert config.SIMILARITY_THRESHOLD == 0.65
        assert config.MODEL_NAME == "all-mpnet-base-v2"
        
        print("✅ Configuration works correctly")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_environment():
    """Test that required files exist."""
    print("\nTesting environment...")
    
    required_files = [
        "spite_corpus.json",
        "spite_embeddings.npy", 
        "spite_style_profile.json",
        "spite_system_prompt.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  Missing required files: {missing_files}")
        print("   The AI system will not work without these files.")
        return False
    else:
        print("✅ All required files present")
        return True

def main():
    """Run all tests."""
    print("🧪 Testing Spite AI Environment")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_config,
        test_environment
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Environment is ready.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
