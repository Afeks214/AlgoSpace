#!/usr/bin/env python3
"""
Test script to verify Synergy_2_MLMI_NWRQK_FVG notebook runs successfully
"""

import subprocess
import sys
import json
import time
from pathlib import Path

def test_notebook_execution():
    """Test the notebook executes without errors"""
    notebook_path = Path("/home/QuantNova/AlgoSpace-8/notebooks/Synergy_2_MLMI_NWRQK_FVG.ipynb")
    
    if not notebook_path.exists():
        print(f"❌ Notebook not found: {notebook_path}")
        return False
    
    print(f"🔄 Testing notebook execution: {notebook_path.name}")
    print("=" * 60)
    
    # Run the notebook using nbconvert
    cmd = [
        sys.executable, "-m", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=600",  # 10 minute timeout
        "--ExecutePreprocessor.kernel_name=python3",
        "--output", "test_output.ipynb",
        str(notebook_path)
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        execution_time = time.time() - start_time
        
        print(f"✅ Notebook executed successfully!")
        print(f"⏱️  Execution time: {execution_time:.2f} seconds")
        
        # Clean up output file
        output_file = Path("test_output.ipynb")
        if output_file.exists():
            output_file.unlink()
        
        return True
        
    except subprocess.CalledProcessError as e:
        execution_time = time.time() - start_time
        
        print(f"❌ Notebook execution failed after {execution_time:.2f} seconds")
        print(f"Return code: {e.returncode}")
        
        if e.stdout:
            print("\nStdout:")
            print(e.stdout)
        
        if e.stderr:
            print("\nStderr:")
            print(e.stderr)
            
            # Try to extract the specific error
            if "CellExecutionError" in e.stderr:
                lines = e.stderr.split('\n')
                for i, line in enumerate(lines):
                    if "CellExecutionError" in line:
                        # Print context around the error
                        start = max(0, i - 5)
                        end = min(len(lines), i + 10)
                        print("\nError context:")
                        for j in range(start, end):
                            print(lines[j])
                        break
        
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")
        return False

def quick_syntax_check():
    """Quick check for basic Python syntax errors in notebook cells"""
    notebook_path = Path("/home/QuantNova/AlgoSpace-8/notebooks/Synergy_2_MLMI_NWRQK_FVG.ipynb")
    
    print("🔍 Running quick syntax check...")
    
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    errors = []
    
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            code = ''.join(cell['source'])
            
            # Skip empty cells
            if not code.strip():
                continue
                
            try:
                compile(code, f'cell_{i}', 'exec')
            except SyntaxError as e:
                errors.append({
                    'cell': i,
                    'error': str(e),
                    'line': e.lineno,
                    'text': e.text
                })
    
    if errors:
        print(f"❌ Found {len(errors)} syntax errors:")
        for err in errors:
            print(f"\nCell {err['cell']}:")
            print(f"  Error: {err['error']}")
            if err['text']:
                print(f"  Line: {err['text'].strip()}")
    else:
        print("✅ No syntax errors found")
    
    return len(errors) == 0

def main():
    """Main test function"""
    print("🚀 Synergy_2_MLMI_NWRQK_FVG Notebook Test Runner")
    print("=" * 60)
    
    # First do a quick syntax check
    syntax_ok = quick_syntax_check()
    
    if not syntax_ok:
        print("\n⚠️  Fix syntax errors before running full execution test")
        return 1
    
    print("\n" + "=" * 60)
    
    # Then test full execution
    success = test_notebook_execution()
    
    print("\n" + "=" * 60)
    
    if success:
        print("✅ All tests passed! The notebook is ready for production use.")
        print("\nNext steps:")
        print("1. Run the notebook in Jupyter to see visualizations")
        print("2. Monitor memory usage during execution")
        print("3. Review the backtest results and performance metrics")
        return 0
    else:
        print("❌ Tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())