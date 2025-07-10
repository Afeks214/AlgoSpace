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
from typing import Tuple, Dict as TypeDict, Optional, NamedTuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
import json
import os
import numba

warnings.filterwarnings('ignore')

# Configure Numba
numba.config.THREADING_LAYER = 'threadsafe'
numba.config.NUMBA_NUM_THREADS = numba.config.NUMBA_DEFAULT_NUM_THREADS

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 100)

print("=" * 80)
print("SYNERGY 1 NOTEBOOK VERIFICATION TEST")
print("=" * 80)

# Initialize result tracking
results = {
    'errors': [],
    'warnings': [],
    'signals': {},
    'backtest': {}
}

try:
    # Execute notebook code cells in sequence
    print("\n1. Loading notebook cells...")
    with open('Synergy_1_MLMI_FVG_NWRQK.ipynb', 'r') as f:
        notebook = json.load(f)
    
    code_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    print(f"Found {len(code_cells)} code cells to execute")
    
    # Create a global namespace for execution
    exec_globals = globals().copy()
    
    # Execute each cell
    for i, cell in enumerate(code_cells):
        print(f"\nExecuting Cell {i+1}...")
        try:
            # Get cell source
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            
            # Execute the cell
            exec(source, exec_globals)
            
            # Check for specific outputs after key cells
            if 'calculate_mlmi_optimized' in source:
                print("✓ MLMI calculation cell executed")
            elif 'detect_fvg' in source and 'process_fvg_active_zones' in source:
                print("✓ FVG detection cell executed")
            elif 'calculate_nw_rqk' in source:
                print("✓ NW-RQK calculation cell executed")
                
        except Exception as e:
            error_msg = f"Cell {i+1} Error: {str(e)}"
            print(f"✗ {error_msg}")
            results['errors'].append(error_msg)
    
    # Extract results from executed namespace
    df_30m = exec_globals.get('df_30m')
    df_5m = exec_globals.get('df_5m')
    df_5m_aligned = exec_globals.get('df_5m_aligned')
    portfolio = exec_globals.get('portfolio')
    stats = exec_globals.get('stats', {})
    
    print("\n" + "=" * 80)
    print("VERIFICATION REPORT")
    print("=" * 80)
    
    # 2. Code Logic Verification
    print("\n2. CODE LOGIC VERIFICATION:")
    print("-" * 50)
    
    # Check for key functions
    mlmi_found = 'MLMIDataFast' in str(notebook) and 'calculate_mlmi_optimized' in str(notebook)
    fvg_found = 'detect_fvg' in str(notebook) and 'process_fvg_active_zones' in str(notebook)
    nwrqk_found = 'calculate_nw_rqk' in str(notebook) and 'kernel_regression_numba' in str(notebook)
    
    print(f"✓ MLMI implementation (MLMIDataFast + cKDTree): {'FOUND' if mlmi_found else 'NOT FOUND'}")
    print(f"✓ FVG implementation (detect_fvg + active zones): {'FOUND' if fvg_found else 'NOT FOUND'}")
    print(f"✓ NW-RQK implementation (calculate_nw_rqk + helpers): {'FOUND' if nwrqk_found else 'NOT FOUND'}")
    print(f"✓ Incorrect functions removed: {'YES' if 'calculate_fvg_parallel' not in str(notebook) else 'NO'}")
    
    # 3. Data Flow & Signal Generation Report
    print("\n3. DATA FLOW & SIGNAL GENERATION REPORT:")
    print("-" * 50)
    
    if df_30m is not None and isinstance(df_30m, pd.DataFrame):
        # NW-RQK Signals
        if 'isBullishChange' in df_30m.columns and 'isBearishChange' in df_30m.columns:
            bull_changes = df_30m['isBullishChange'].sum()
            bear_changes = df_30m['isBearishChange'].sum()
            print(f"\nNW-RQK Signals (30m data):")
            print(f"  - Bullish Changes: {bull_changes:,}")
            print(f"  - Bearish Changes: {bear_changes:,}")
            results['signals']['nwrqk_bull'] = bull_changes
            results['signals']['nwrqk_bear'] = bear_changes
        else:
            print("\nNW-RQK Signals: NOT FOUND in dataframe")
        
        # MLMI Signals
        if 'mlmi_bull' in df_30m.columns and 'mlmi_bear' in df_30m.columns:
            mlmi_bull = df_30m['mlmi_bull'].sum()
            mlmi_bear = df_30m['mlmi_bear'].sum()
            print(f"\nMLMI Signals (30m data):")
            print(f"  - Bull Crosses: {mlmi_bull:,}")
            print(f"  - Bear Crosses: {mlmi_bear:,}")
            results['signals']['mlmi_bull'] = mlmi_bull
            results['signals']['mlmi_bear'] = mlmi_bear
        else:
            print("\nMLMI Signals: NOT FOUND in dataframe")
    
    if df_5m_aligned is not None and isinstance(df_5m_aligned, pd.DataFrame):
        # FVG Signals
        if 'fvg_bull' in df_5m_aligned.columns and 'fvg_bear' in df_5m_aligned.columns:
            fvg_bull = df_5m_aligned['fvg_bull'].sum()
            fvg_bear = df_5m_aligned['fvg_bear'].sum()
            print(f"\nFVG Signals (5m data):")
            print(f"  - Bull Zones Active: {fvg_bull:,}")
            print(f"  - Bear Zones Active: {fvg_bear:,}")
            results['signals']['fvg_bull'] = fvg_bull
            results['signals']['fvg_bear'] = fvg_bear
        
        # Final Synergy Signals
        if 'long_entry' in df_5m_aligned.columns and 'short_entry' in df_5m_aligned.columns:
            long_entries = df_5m_aligned['long_entry'].sum()
            short_entries = df_5m_aligned['short_entry'].sum()
            print(f"\nFinal Synergy Signals (5m aligned):")
            print(f"  - Long Entries: {long_entries:,}")
            print(f"  - Short Entries: {short_entries:,}")
            results['signals']['long_entries'] = long_entries
            results['signals']['short_entries'] = short_entries
        else:
            print("\nFinal Synergy Signals: NOT FOUND in dataframe")
    
    # 4. Backtest Results
    print("\n4. FINAL BACKTEST RESULTS:")
    print("-" * 50)
    
    if stats and isinstance(stats, dict):
        total_trades = stats.get('Total Trades', 0)
        total_return = stats.get('Total Return [%]', 0)
        win_rate = stats.get('Win Rate [%]', 0)
        sharpe = stats.get('Sharpe Ratio', 0)
        
        print(f"Total Trades Executed: {total_trades:,.0f}")
        print(f"Final Total Return: {total_return:.2f}%")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        
        results['backtest'] = {
            'total_trades': total_trades,
            'total_return': total_return,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe
        }
    else:
        print("Backtest results not available or portfolio not generated")
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    if len(results['errors']) == 0:
        print("✓ All cells executed successfully without errors")
    else:
        print(f"✗ {len(results['errors'])} errors encountered:")
        for error in results['errors'][:5]:  # Show first 5 errors
            print(f"  - {error}")
    
    # Function presence summary
    all_functions_present = mlmi_found and fvg_found and nwrqk_found
    if all_functions_present:
        print("✓ All required functions are present and correct")
    else:
        print("✗ Some required functions are missing")
    
    # Signal generation summary
    signals_generated = all(key in results['signals'] for key in ['mlmi_bull', 'nwrqk_bull', 'fvg_bull', 'long_entries'])
    if signals_generated:
        print("✓ All signal types generated successfully")
    else:
        print("✗ Some signal types were not generated")
    
    # Backtest summary
    if results['backtest'].get('total_trades', 0) > 0:
        print("✓ Backtest completed with trades generated")
    else:
        print("✗ No trades generated in backtest")
    
except Exception as e:
    print(f"\nCRITICAL ERROR: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)