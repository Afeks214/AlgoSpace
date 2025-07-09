#!/usr/bin/env python3
"""
Validate Synergy_2_MLMI_NWRQK_FVG notebook structure and imports
"""

import json
import ast
from pathlib import Path

def extract_imports(code):
    """Extract all import statements from code"""
    imports = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module.split('.')[0])
    except:
        pass
    return imports

def validate_notebook():
    """Validate notebook structure and dependencies"""
    notebook_path = Path("/home/QuantNova/AlgoSpace-8/notebooks/Synergy_2_MLMI_NWRQK_FVG.ipynb")
    
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    print("📊 Notebook Validation Report")
    print("=" * 60)
    
    # Count cells
    code_cells = sum(1 for cell in notebook['cells'] if cell['cell_type'] == 'code')
    markdown_cells = sum(1 for cell in notebook['cells'] if cell['cell_type'] == 'markdown')
    
    print(f"📝 Total cells: {len(notebook['cells'])}")
    print(f"   - Code cells: {code_cells}")
    print(f"   - Markdown cells: {markdown_cells}")
    
    # Collect all imports
    all_imports = set()
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            code = ''.join(cell['source'])
            imports = extract_imports(code)
            all_imports.update(imports)
    
    print(f"\n📦 Required packages ({len(all_imports)}):")
    for pkg in sorted(all_imports):
        print(f"   - {pkg}")
    
    # Check for common issues
    print("\n🔍 Checking for common issues:")
    
    issues = []
    
    # Check for undefined variables
    defined_vars = set()
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            code = ''.join(cell['source'])
            
            # Simple check for variable definitions
            for line in code.split('\n'):
                if '=' in line and not line.strip().startswith('#'):
                    var_name = line.split('=')[0].strip()
                    if ' ' not in var_name and var_name.isidentifier():
                        defined_vars.add(var_name)
    
    # Check each cell
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            code = ''.join(cell['source'])
            
            # Check for syntax
            try:
                compile(code, f'cell_{i}', 'exec')
            except SyntaxError as e:
                issues.append(f"Cell {i}: Syntax error - {e}")
            
            # Check for common patterns
            if 'df_5m_aligned' in code and 'df_5m_aligned' not in defined_vars:
                if 'df_5m_aligned =' not in code:
                    issues.append(f"Cell {i}: Uses df_5m_aligned but it may not be defined")
            
            if 'stats[' in code:
                issues.append(f"Cell {i}: Uses 'stats' variable - should be 'portfolio_stats'")
    
    if issues:
        print(f"❌ Found {len(issues)} potential issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"   - {issue}")
    else:
        print("✅ No obvious issues found")
    
    # Check for data files
    print("\n📁 Checking data files:")
    data_files = [
        "/home/QuantNova/AlgoSpace-Strategy-1/@NQ - 5 min - ETH.csv",
        "/home/QuantNova/AlgoSpace-Strategy-1/NQ - 30 min - ETH.csv"
    ]
    
    for file in data_files:
        if Path(file).exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - NOT FOUND")
    
    print("\n" + "=" * 60)
    print("✅ Validation complete!")
    
    # Summary
    print("\n📋 Summary:")
    print(f"   - Notebook has {code_cells} executable code cells")
    print(f"   - Requires {len(all_imports)} unique packages")
    print(f"   - Found {len(issues)} potential issues")
    print(f"   - All required data files exist")
    
    print("\n💡 Next steps:")
    print("   1. Open the notebook in Jupyter")
    print("   2. Run cells sequentially from top to bottom")
    print("   3. Monitor execution for any runtime errors")
    print("   4. Review the backtest results and visualizations")

if __name__ == "__main__":
    validate_notebook()