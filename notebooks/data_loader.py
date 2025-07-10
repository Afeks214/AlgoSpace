import pandas as pd
import numpy as np
import time
import os

def validate_dataframe(df, name, min_data_points=100):
    """Validate dataframe has required columns and data types"""
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Check for required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: {name} missing columns: {missing_cols}")
        return False
    
    # Check for sufficient data
    if len(df) < min_data_points:
        print(f"Warning: {name} has insufficient data: {len(df)} rows (minimum: {min_data_points})")
        return False
    
    # Check for valid price data
    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        if col in df.columns:
            if df[col].isna().all():
                print(f"Warning: {name} column '{col}' contains only NaN values")
                return False
            if (df[col] <= 0).any():
                print(f"Warning: {name} column '{col}' contains non-positive values")
                return False
    
    return True

def load_data_optimized(file_path, timeframe='5m'):
    """Load and prepare data with comprehensive error handling"""
    start_time = time.time()
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read CSV with modern pandas datetime parsing
        df = pd.read_csv(file_path, 
                         parse_dates=['Timestamp'],
                         index_col='Timestamp',
                         low_memory=False)
        
        # If Timestamp parsing failed, try alternative approaches
        if not isinstance(df.index, pd.DatetimeIndex):
            # Reset index and try to parse Timestamp column
            df = df.reset_index()
            if 'Timestamp' in df.columns:
                # Try different date formats
                for date_format in ['%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S']:
                    try:
                        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format=date_format)
                        df.set_index('Timestamp', inplace=True)
                        break
                    except:
                        continue
                
                # If specific formats fail, use pandas intelligent parsing
                if not isinstance(df.index, pd.DatetimeIndex):
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True, errors='coerce')
                    df.set_index('Timestamp', inplace=True)
        
        # Check if dataframe is empty
        if df.empty:
            raise ValueError(f"Empty dataframe loaded from {file_path}")
        
        # Ensure numeric types for fast operations
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float64)
        
        # Remove any NaN values in critical columns
        critical_cols = ['Open', 'High', 'Low', 'Close']
        existing_critical = [col for col in critical_cols if col in df.columns]
        
        if existing_critical:
            initial_len = len(df)
            df.dropna(subset=existing_critical, inplace=True)
            dropped = initial_len - len(df)
            if dropped > 0:
                print(f"Dropped {dropped} rows with NaN values")
        
        # Validate OHLC relationships
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            invalid_ohlc = ((df['High'] < df['Low']) | 
                           (df['High'] < df['Open']) | 
                           (df['High'] < df['Close']) | 
                           (df['Low'] > df['Open']) | 
                           (df['Low'] > df['Close']))
            if invalid_ohlc.any():
                print(f"Warning: Found {invalid_ohlc.sum()} rows with invalid OHLC relationships")
                df = df[~invalid_ohlc]
        
        # Sort index for faster operations
        df.sort_index(inplace=True)
        
        # Check for duplicate timestamps
        if df.index.duplicated().any():
            print(f"Warning: Found {df.index.duplicated().sum()} duplicate timestamps, keeping first")
            df = df[~df.index.duplicated(keep='first')]
        
        load_time = time.time() - start_time
        print(f"Loaded {len(df):,} rows in {load_time:.2f} seconds from {timeframe} file")
        
        # Validate the loaded data
        if not validate_dataframe(df, f"{timeframe} data"):
            print(f"Warning: Data validation failed for {timeframe} data")
        
        return df
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print(f"Please ensure the file exists at: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        print(f"Error: File {file_path} is empty")
        raise
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {str(e)}")
        raise
    except Exception as e:
        print(f"Unexpected error loading {file_path}: {str(e)}")
        raise