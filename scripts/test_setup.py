#!/usr/bin/env python3
"""
Test Setup Script

Verifies that all components are properly installed and configured.

Usage:
    python scripts/test_setup.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    modules = [
        ("numpy", None),
        ("pandas", None),
        ("torch", None),
        ("xgboost", None),
        ("yaml", None),
        ("aiohttp", None),
    ]

    for module_name, min_version in modules:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"  [OK] {module_name} ({version})")
        except ImportError as e:
            print(f"  [FAIL] {module_name}: {e}")
            return False

    return True


def test_project_structure():
    """Test that project structure exists."""
    print("\nTesting project structure...")

    project_root = Path(__file__).parent.parent
    required_dirs = [
        "src",
        "src/data",
        "src/features",
        "src/models",
        "src/risk",
        "src/backtest",
        "src/execution",
        "src/utils",
        "configs",
        "data",
        "data/raw",
        "data/processed",
        "data/cache",
        "scripts",
        "Kronos",
    ]

    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"  [OK] {dir_name}/")
        else:
            print(f"  [FAIL] {dir_name}/ missing")
            return False

    return True


def test_kronos_model():
    """Test that Kronos model exists."""
    print("\nTesting Kronos model...")

    project_root = Path(__file__).parent.parent
    model_path = project_root / "Kronos" / "model" / "base" / "model.safetensors"

    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  [OK] model.safetensors ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  [FAIL] model.safetensors not found at {model_path}")
        return False


def test_config():
    """Test that config loads correctly."""
    print("\nTesting configuration...")

    try:
        from src.utils.config import load_config

        config = load_config()
        print(f"  [OK] Config loaded")
        print(f"       - Symbols: {config['data']['symbols']}")
        print(f"       - Initial capital: ${config['risk']['initial_capital']}")
        return True
    except Exception as e:
        print(f"  [FAIL] Config error: {e}")
        return False


def test_src_modules():
    """Test that source modules can be imported."""
    print("\nTesting source modules...")

    modules = [
        "src.data.collector",
        "src.data.processor",
        "src.data.storage",
        "src.features.engineer",
        "src.models.signal_model",
        "src.models.trainer",
        "src.risk.engine",
        "src.risk.position",
        "src.backtest.engine",
        "src.backtest.metrics",
        "src.utils.config",
        "src.utils.logger",
    ]

    for module_name in modules:
        try:
            __import__(module_name)
            print(f"  [OK] {module_name}")
        except ImportError as e:
            print(f"  [FAIL] {module_name}: {e}")
            return False

    return True


def test_gpu():
    """Test GPU availability."""
    print("\nTesting GPU...")

    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  [OK] CUDA available: {device_name} ({memory_gb:.1f} GB)")
        else:
            print(f"  [WARN] CUDA not available, will use CPU")
        return True
    except Exception as e:
        print(f"  [WARN] GPU test failed: {e}")
        return True  # Not critical


def main():
    print("=" * 60)
    print("Kronos Trading System - Setup Test")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Project Structure", test_project_structure),
        ("Kronos Model", test_kronos_model),
        ("Configuration", test_config),
        ("Source Modules", test_src_modules),
        ("GPU", test_gpu),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[ERROR] {name} test failed with exception: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\nSetup is complete! You can now run:")
        print("  1. python scripts/collect_data.py --help")
        print("  2. python scripts/run_pipeline.py --help")
    else:
        print("\nSome tests failed. Please fix the issues above.")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
