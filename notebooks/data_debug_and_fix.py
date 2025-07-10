#!/usr/bin/env python3
"""
Data Debug and Fix Script for Synergy Trading Notebooks
This script helps identify and fix column naming issues in CSV data files
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def check_csv_columns(file_path):
    """Check the actual columns in a CSV file"""
    try:
        # Read only first 5 rows for inspection
        df = pd.read_csv(file_path, nrows=5)
        
        print(f"\n{'='*60}")
        print(f"File: {os.path.basename(file_path)}")
        print(f"Full path: {file_path}")
        print(f"{'='*60}")
        print(f"\nNumber of columns: {len(df.columns)}")
        print(f"\nColumn names (raw):")
        for i, col in enumerate(df.columns):
            print(f"  [{i}] '{col}' (type: {type(col).__name__})")
        
        print(f"\nColumn names (normalized - lowercase, stripped):")
        for i, col in enumerate(df.columns):
            normalized = col.lower().strip() if isinstance(col, str) else str(col).lower().strip()
            print(f"  [{i}] '{col}' -> '{normalized}'")
        
        print(f"\nFirst row data sample:")
        for col in df.columns:
            print(f"  {col}: {df[col].iloc[0]}")
        
        print(f"\nData types:")
        print(df.dtypes)
        
        return df.columns.tolist()
        
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None

def create_column_mapping(actual_columns):
    """Create a mapping from actual columns to expected columns"""
    # Expected column names
    expected = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    # Common variations for each expected column
    variations = {
        'timestamp': ['timestamp', 'datetime', 'date', 'gmt time', 'time', 'gmt_time', 
                     'date_time', 'utc_time', 'utc', 'gmt', 'date time'],
        'open': ['open', 'o', 'open_price', 'opening', 'open price', 'opening price'],
        'high': ['high', 'h', 'high_price', 'highest', 'max', 'high price', 'maximum'],
        'low': ['low', 'l', 'low_price', 'lowest', 'min', 'low price', 'minimum'],
        'close': ['close', 'c', 'close_price', 'closing', 'close price', 'closing price', 'last'],
        'volume': ['volume', 'v', 'vol', 'volume_btc', 'volume_usd', 'volume btc', 
                  'volume usd', 'qty', 'quantity', 'amount']
    }
    
    mapping = {}
    found = {col: False for col in expected}
    
    # Try to find matches
    for actual_col in actual_columns:
        actual_normalized = actual_col.lower().strip() if isinstance(actual_col, str) else str(actual_col).lower().strip()
        
        for expected_col, var_list in variations.items():
            if not found[expected_col]:  # Only map if not already found
                for variation in var_list:
                    if variation == actual_normalized or variation.replace(' ', '_') == actual_normalized:
                        mapping[actual_col] = expected_col
                        found[expected_col] = True
                        break
    
    # Report findings
    print(f"\n{'='*60}")
    print("Column Mapping Results:")
    print(f"{'='*60}")
    for expected_col in expected:
        if found[expected_col]:
            actual = [k for k, v in mapping.items() if v == expected_col][0]
            print(f"✓ {expected_col}: Found as '{actual}'")
        else:
            print(f"✗ {expected_col}: NOT FOUND")
    
    return mapping, found

def fix_csv_columns(input_path, output_path=None, create_missing=True):
    """Fix column names and optionally create missing columns"""
    try:
        # Read the CSV
        df = pd.read_csv(input_path)
        original_columns = df.columns.tolist()
        
        print(f"\nProcessing: {os.path.basename(input_path)}")
        
        # Get column mapping
        mapping, found = create_column_mapping(original_columns)
        
        # Apply column renaming
        df.rename(columns=mapping, inplace=True)
        
        # Handle missing columns if requested
        if create_missing:
            print(f"\nHandling missing columns...")
            
            # Create missing price columns from available data
            price_cols = ['open', 'high', 'low', 'close']
            available_price_cols = [col for col in price_cols if col in df.columns]
            
            if available_price_cols:
                # If we have at least one price column
                base_price_col = available_price_cols[0]
                
                for col in price_cols:
                    if col not in df.columns:
                        if col == 'high' and 'close' in df.columns and 'open' in df.columns:
                            # High is typically the max of open and close (with small adjustment)
                            df['high'] = df[['open', 'close']].max(axis=1) * 1.001
                            print(f"  Created 'high' from max(open, close) * 1.001")
                        elif col == 'low' and 'close' in df.columns and 'open' in df.columns:
                            # Low is typically the min of open and close (with small adjustment)
                            df['low'] = df[['open', 'close']].min(axis=1) * 0.999
                            print(f"  Created 'low' from min(open, close) * 0.999")
                        else:
                            # Use the first available price column
                            df[col] = df[base_price_col]
                            print(f"  Created '{col}' from '{base_price_col}'")
            
            # Handle missing volume
            if 'volume' not in df.columns:
                df['volume'] = 1000000  # Default volume
                print(f"  Created 'volume' with default value 1000000")
            
            # Handle missing timestamp
            if 'timestamp' not in df.columns:
                # Try to create from index or use sequential timestamps
                if df.index.name and 'time' in df.index.name.lower():
                    df['timestamp'] = df.index
                    print(f"  Created 'timestamp' from index")
                else:
                    # Create sequential timestamps
                    start_date = datetime(2023, 1, 1)
                    df['timestamp'] = pd.date_range(start=start_date, periods=len(df), freq='30min')
                    print(f"  Created 'timestamp' with sequential dates starting from {start_date}")
        
        # Validate final columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        final_missing = [col for col in required_cols if col not in df.columns]
        
        if final_missing:
            print(f"\n⚠️  Warning: Still missing columns after fix: {final_missing}")
        else:
            print(f"\n✓ All required columns present!")
        
        # Save if output path provided
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"\n✓ Fixed data saved to: {output_path}")
        
        return df
        
    except Exception as e:
        print(f"Error fixing CSV: {str(e)}")
        return None

def create_fixed_data_loader():
    """Create a fixed version of the data loader function"""
    code = '''
def load_single_timeframe_fixed(file_path: str, timeframe: str) -> pd.DataFrame:
    """Load data for a single timeframe with enhanced error handling and column mapping"""
    try:
        # Load CSV with error handling
        df = pd.read_csv(file_path, na_values=['', 'null', 'NULL', 'NaN'])
        
        # Log the actual columns in the file for debugging
        logger.info(f"Original columns in {timeframe} data: {list(df.columns)}")
        
        # Convert column names to lowercase and strip whitespace
        df.columns = df.columns.str.lower().str.strip()
        
        # COMPREHENSIVE COLUMN MAPPING
        column_map = {
            # Open variations
            'open': 'open', 'o': 'open', 'open_price': 'open', 'opening': 'open',
            'open price': 'open', 'opening price': 'open',
            # High variations
            'high': 'high', 'h': 'high', 'high_price': 'high', 'highest': 'high', 
            'max': 'high', 'high price': 'high', 'maximum': 'high',
            # Low variations
            'low': 'low', 'l': 'low', 'low_price': 'low', 'lowest': 'low', 
            'min': 'low', 'low price': 'low', 'minimum': 'low',
            # Close variations
            'close': 'close', 'c': 'close', 'close_price': 'close', 'closing': 'close',
            'close price': 'close', 'closing price': 'close', 'last': 'close',
            # Volume variations
            'volume': 'volume', 'v': 'volume', 'vol': 'volume', 'volume_btc': 'volume', 
            'volume_usd': 'volume', 'volume btc': 'volume', 'volume usd': 'volume',
            'qty': 'volume', 'quantity': 'volume', 'amount': 'volume',
            # Timestamp variations
            'timestamp': 'timestamp', 'datetime': 'timestamp', 'date': 'timestamp', 
            'gmt time': 'timestamp', 'time': 'timestamp', 'gmt_time': 'timestamp',
            'date_time': 'timestamp', 'utc_time': 'timestamp', 'date time': 'timestamp',
            'utc': 'timestamp', 'gmt': 'timestamp'
        }
        
        # Apply column mapping
        df.rename(columns=lambda c: column_map.get(c.lower().strip(), c), inplace=True)
        
        # Log columns after mapping
        logger.info(f"Columns after mapping: {list(df.columns)}")
        
        # Check for required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        # Create missing columns if needed
        if missing_columns:
            logger.warning(f"Missing columns in {timeframe} data: {missing_columns}")
            
            # Handle missing price columns
            price_cols = ['open', 'high', 'low', 'close']
            available_price_cols = [col for col in price_cols if col in df.columns]
            
            if available_price_cols:
                # Create missing high
                if 'high' in missing_columns and 'close' in df.columns and 'open' in df.columns:
                    df['high'] = df[['open', 'close']].max(axis=1) * 1.001
                    logger.info(f"Created 'high' column from max(open, close)")
                    missing_columns.remove('high')
                
                # Create missing low
                if 'low' in missing_columns and 'close' in df.columns and 'open' in df.columns:
                    df['low'] = df[['open', 'close']].min(axis=1) * 0.999
                    logger.info(f"Created 'low' column from min(open, close)")
                    missing_columns.remove('low')
                
                # Use first available price for any remaining missing price columns
                base_price = available_price_cols[0]
                for col in ['open', 'high', 'low', 'close']:
                    if col in missing_columns:
                        df[col] = df[base_price]
                        logger.info(f"Created '{col}' column from '{base_price}'")
                        missing_columns.remove(col)
            
            # Create missing volume
            if 'volume' in missing_columns:
                df['volume'] = 1000000  # Default volume
                logger.info(f"Created 'volume' column with default value")
                missing_columns.remove('volume')
        
        # Final validation
        critical_missing = [col for col in ['timestamp', 'open', 'close'] if col not in df.columns]
        if critical_missing:
            raise ValueError(f"Critical columns still missing: {critical_missing}")
        
        # Continue with rest of the processing...
        # Parse timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, format='mixed', errors='coerce')
        
        # Remove invalid timestamps
        df = df[df['timestamp'].notna()]
        
        # Set index and sort
        df = df.set_index('timestamp').sort_index()
        
        # Remove duplicates
        df = df[~df.index.duplicated()]
        
        # Ensure numeric data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"Successfully loaded {len(df)} rows of {timeframe} data")
        return df
        
    except Exception as e:
        logger.error(f"Error loading {timeframe} data: {str(e)}")
        raise
'''
    return code

def main():
    """Main function to debug and fix data issues"""
    print("Data Debug and Fix Tool for Synergy Trading Notebooks")
    print("="*60)
    
    # Define file paths
    data_dir = "/home/QuantNova/AlgoSpace-8/notebooks/notebook data"
    file_30m = os.path.join(data_dir, "@CL - 30 min - ETH.csv")
    file_5m = os.path.join(data_dir, "@CL - 5 min - ETH.csv")
    
    # Check both files
    print("\n1. Checking CSV column names...")
    columns_30m = check_csv_columns(file_30m)
    columns_5m = check_csv_columns(file_5m)
    
    # Option to fix the files
    print("\n2. Column Analysis Complete!")
    print("\nOptions:")
    print("a) The debug info above shows what columns are in your files")
    print("b) You can use fix_csv_columns() to create fixed versions")
    print("c) Or update your notebook's load_single_timeframe() function")
    
    # Generate fixed loader code
    print("\n3. Fixed Data Loader Function:")
    print("Copy this into your notebook to replace load_single_timeframe():")
    print("-"*60)
    print(create_fixed_data_loader())
    
    return columns_30m, columns_5m

if __name__ == "__main__":
    columns_30m, columns_5m = main()