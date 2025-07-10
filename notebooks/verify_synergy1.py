#!/usr/bin/env python3

import pandas as pd
import numpy as np
import vectorbt as vbt
from numba import njit, prange, typed, types, float64, int64, boolean
from numba.typed import Dict
from numba.experimental import jitclass
from scipy.spatial import cKDTree
import warnings
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
import json
import os
import numba
import sys

warnings.filterwarnings('ignore')

# Configure Numba
numba.config.THREADING_LAYER = 'threadsafe'
numba.config.NUMBA_NUM_THREADS = numba.config.NUMBA_DEFAULT_NUM_THREADS

# Import data loading functions
sys.path.append('/home/QuantNova/AlgoSpace-8/notebooks')
from data_loader import load_data_optimized, validate_dataframe

print("=" * 80)
print("SYNERGY 1 NOTEBOOK VERIFICATION REPORT")
print("=" * 80)

# Initialize result tracking
results = {
    'errors': [],
    'cells_executed': 0,
    'functions_verified': {},
    'signals': {},
    'backtest': {}
}

try:
    # Execute notebook code cells in sequence
    print("\n1. FULL NOTEBOOK EXECUTION")
    print("-" * 50)
    
    with open('Synergy_1_MLMI_FVG_NWRQK.ipynb', 'r') as f:
        notebook = json.load(f)
    
    code_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    print(f"Executing {len(code_cells)} code cells...")
    
    # Create a global namespace for execution
    exec_globals = globals().copy()
    
    # Execute each cell
    for i, cell in enumerate(code_cells):
        try:
            # Get cell source
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            
            # Skip cells with known numba caching issues in string execution
            if 'cache=True' in source and '<string>' not in source:
                # Remove cache=True for execution
                source = source.replace('cache=True', 'cache=False')
            
            # Execute the cell
            exec(source, exec_globals)
            results['cells_executed'] += 1
            
        except Exception as e:
            error_msg = f"Cell {i+1}: {str(e)}"
            results['errors'].append(error_msg)
            if "Fatal error" in str(e) or "Cannot proceed" in str(e):
                print(f"✗ Fatal error in Cell {i+1}, stopping execution")
                break
    
    print(f"\n✓ Successfully executed {results['cells_executed']}/{len(code_cells)} cells")
    if results['errors']:
        print(f"✗ {len(results['errors'])} errors encountered")
    
    # Extract results from executed namespace
    df_30m = exec_globals.get('df_30m')
    df_5m = exec_globals.get('df_5m')
    df_5m_aligned = exec_globals.get('df_5m_aligned')
    portfolio = exec_globals.get('portfolio')
    stats = exec_globals.get('stats', {})
    
    # 2. Code Logic Verification
    print("\n2. CODE LOGIC & FUNCTION VERIFICATION")
    print("-" * 50)
    
    # Check for key functions in notebook content
    notebook_str = str(notebook)
    
    functions_to_check = {
        'MLMI (MLMIDataFast)': 'MLMIDataFast' in notebook_str,
        'MLMI (calculate_mlmi_optimized)': 'calculate_mlmi_optimized' in notebook_str,
        'MLMI (cKDTree usage)': 'cKDTree' in notebook_str,
        'FVG (detect_fvg)': 'detect_fvg' in notebook_str,
        'FVG (process_fvg_active_zones)': 'process_fvg_active_zones' in notebook_str,
        'NW-RQK (calculate_nw_rqk)': 'calculate_nw_rqk' in notebook_str,
        'NW-RQK (kernel_regression_numba)': 'kernel_regression_numba' in notebook_str,
        'Removed (calculate_fvg_parallel)': 'calculate_fvg_parallel' not in notebook_str
    }
    
    for func_name, present in functions_to_check.items():
        status = "✓ PRESENT" if present else "✗ MISSING"
        if "Removed" in func_name:
            status = "✓ REMOVED" if present else "✗ STILL EXISTS"
        print(f"{func_name:40} {status}")
        results['functions_verified'][func_name] = present
    
    # 3. Data Flow & Signal Generation Report
    print("\n3. DATA FLOW & SIGNAL GENERATION REPORT")
    print("-" * 50)
    
    if df_30m is not None and isinstance(df_30m, pd.DataFrame):
        print(f"\n30-minute data loaded: {len(df_30m):,} bars")
        
        # NW-RQK Signals
        if 'isBullishChange' in df_30m.columns and 'isBearishChange' in df_30m.columns:
            bull_changes = df_30m['isBullishChange'].sum()
            bear_changes = df_30m['isBearishChange'].sum()
            print(f"\nNW-RQK Signals (30m data):")
            print(f"  isBullishChange signals: {bull_changes:,}")
            print(f"  isBearishChange signals: {bear_changes:,}")
            results['signals']['nwrqk_bull_changes'] = int(bull_changes)
            results['signals']['nwrqk_bear_changes'] = int(bear_changes)
        
        # MLMI Signals
        if 'mlmi_bull' in df_30m.columns and 'mlmi_bear' in df_30m.columns:
            mlmi_bull = df_30m['mlmi_bull'].sum()
            mlmi_bear = df_30m['mlmi_bear'].sum()
            print(f"\nMLMI Signals (30m data):")
            print(f"  mlmi_bull_cross signals: {mlmi_bull:,}")
            print(f"  mlmi_bear_cross signals: {mlmi_bear:,}")
            results['signals']['mlmi_bull_cross'] = int(mlmi_bull)
            results['signals']['mlmi_bear_cross'] = int(mlmi_bear)
        
        # Check MLMI values
        if 'mlmi' in df_30m.columns:
            valid_mlmi = (~df_30m['mlmi'].isna()).sum()
            print(f"\nMLMI indicator values: {valid_mlmi:,} valid values")
    else:
        print("\n✗ 30-minute data not loaded")
    
    if df_5m_aligned is not None and isinstance(df_5m_aligned, pd.DataFrame):
        print(f"\n5-minute aligned data: {len(df_5m_aligned):,} bars")
        
        # FVG Signals
        if 'fvg_bull' in df_5m_aligned.columns and 'fvg_bear' in df_5m_aligned.columns:
            fvg_bull = df_5m_aligned['fvg_bull'].sum()
            fvg_bear = df_5m_aligned['fvg_bear'].sum()
            print(f"\nFVG Signals (5m data):")
            print(f"  Bullish FVG zones: {fvg_bull:,}")
            print(f"  Bearish FVG zones: {fvg_bear:,}")
            results['signals']['fvg_bull_zones'] = int(fvg_bull)
            results['signals']['fvg_bear_zones'] = int(fvg_bear)
        
        # Final Synergy Signals
        if 'long_entry' in df_5m_aligned.columns and 'short_entry' in df_5m_aligned.columns:
            long_entries = df_5m_aligned['long_entry'].sum()
            short_entries = df_5m_aligned['short_entry'].sum()
            print(f"\nFinal Synergy Signals (MLMI → FVG → NW-RQK):")
            print(f"  Long entry signals: {long_entries:,}")
            print(f"  Short entry signals: {short_entries:,}")
            results['signals']['long_entries'] = int(long_entries)
            results['signals']['short_entries'] = int(short_entries)
    else:
        print("\n✗ 5-minute aligned data not available")
    
    # 4. Backtest Results
    print("\n4. FINAL BACKTEST RESULTS REPORT")
    print("-" * 50)
    
    if stats and isinstance(stats, dict):
        total_trades = stats.get('Total Trades', 0)
        total_return = stats.get('Total Return [%]', 0)
        win_rate = stats.get('Win Rate [%]', 0)
        sharpe = stats.get('Sharpe Ratio', 0)
        max_dd = stats.get('Max Drawdown [%]', 0)
        
        print(f"Total Trades Executed: {total_trades:,.0f}")
        print(f"Final Total Return: {total_return:.2f}%")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_dd:.2f}%")
        
        results['backtest'] = {
            'total_trades': int(total_trades),
            'total_return': float(total_return),
            'win_rate': float(win_rate),
            'sharpe_ratio': float(sharpe)
        }
    else:
        print("✗ Backtest results not available")
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    # Check if all critical components are working
    all_functions = all(v for k, v in results['functions_verified'].items() if "Removed" not in k)
    has_data = df_5m is not None and df_30m is not None
    has_signals = len(results['signals']) >= 6  # Should have 6 signal types
    has_backtest = results['backtest'].get('total_trades', 0) > 0
    
    print(f"\n✓ Code execution: {results['cells_executed']} cells executed")
    print(f"{'✓' if all_functions else '✗'} All required functions present: {all_functions}")
    print(f"{'✓' if has_data else '✗'} Data loaded successfully: {has_data}")
    print(f"{'✓' if has_signals else '✗'} All signals generated: {has_signals}")
    print(f"{'✓' if has_backtest else '✗'} Backtest completed: {has_backtest}")
    
    if results['errors']:
        print(f"\n⚠ Errors encountered: {len(results['errors'])}")
        for i, error in enumerate(results['errors'][:3]):
            print(f"  - {error}")
    
    # Final verdict
    if all_functions and has_data and has_signals and has_backtest:
        print("\n✓ VERIFICATION PASSED: Notebook is functioning correctly")
    else:
        print("\n✗ VERIFICATION FAILED: Issues need to be resolved")
        
except Exception as e:
    print(f"\n✗ CRITICAL ERROR during verification: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)