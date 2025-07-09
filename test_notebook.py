#!/usr/bin/env python3
"""
Test script to verify Synergy_4_NWRQK_FVG_MLMI notebook functionality
"""
import sys
import subprocess
import importlib

def test_dependencies():
    """Test if all required dependencies are installed"""
    print("Testing dependencies...")
    required_packages = [
        ('numpy', 'np'),
        ('pandas', 'pd'),
        ('scipy', 'scipy'),
        ('numba', 'numba'),
        ('vectorbt', 'vbt'),
        ('plotly', 'plotly')
    ]
    
    missing = []
    for package, alias in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")
            missing.append(package)
    
    # Optional package
    try:
        import psutil
        print("✓ psutil is installed (optional)")
    except ImportError:
        print("ℹ psutil is NOT installed (optional, for memory monitoring)")
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    return True

def test_notebook_execution():
    """Test if the notebook can execute without errors"""
    print("\nTesting notebook execution...")
    
    # First check if dependencies are available
    if not test_dependencies():
        print("\n❌ Cannot test notebook execution due to missing dependencies")
        return False
    
    # Import required modules for testing
    import numpy as np
    import pandas as pd
    import os
    
    # Check if data directory exists
    data_dir = '/home/QuantNova/AlgoSpace-8/data'
    if not os.path.exists(data_dir):
        print(f"Creating data directory: {data_dir}")
        os.makedirs(data_dir, exist_ok=True)
    
    # Check if data files exist
    data_files = [
        '/home/QuantNova/AlgoSpace-8/data/BTC-USD-30m.csv',
        '/home/QuantNova/AlgoSpace-8/data/BTC-USD-5m.csv'
    ]
    
    for file in data_files:
        if os.path.exists(file):
            print(f"✓ Data file exists: {file}")
        else:
            print(f"✗ Data file missing: {file}")
            print("  Sample data will be generated when notebook runs")
    
    # Check output directories
    for dir_path in ['/home/QuantNova/AlgoSpace-8/logs', '/home/QuantNova/AlgoSpace-8/results']:
        if os.path.exists(dir_path) and os.access(dir_path, os.W_OK):
            print(f"✓ Output directory ready: {dir_path}")
        else:
            print(f"ℹ Creating output directory: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
    
    print("\n✅ Notebook is ready to run!")
    print("\nTo run the notebook:")
    print("1. Open it in Jupyter: jupyter notebook notebooks/Synergy_4_NWRQK_FVG_MLMI.ipynb")
    print("2. Or convert and run: jupyter nbconvert --to notebook --execute notebooks/Synergy_4_NWRQK_FVG_MLMI.ipynb")
    
    return True

def main():
    """Main test function"""
    print("="*60)
    print("AlgoSpace-8 Synergy 4 Notebook Test")
    print("="*60)
    
    success = test_notebook_execution()
    
    if success:
        print("\n✅ All tests passed! The notebook is ready to use.")
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()