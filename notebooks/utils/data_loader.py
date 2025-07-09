"""
Data Loader Module for AlgoSpace Strategy
Handles data loading, validation, and preprocessing
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading and preprocessing for AlgoSpace strategy"""
    
    def __init__(self, config_path: str = "config/strategy_config.yaml"):
        """Initialize DataLoader with configuration"""
        self.config = self._load_config(config_path)
        self.data_config = self.config.get('data', {})
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return {}
    
    def load_data(self, file_path: str, timeframe: str = '5m') -> pd.DataFrame:
        """
        Load and preprocess data from CSV file
        
        Args:
            file_path: Path to CSV file
            timeframe: Timeframe of data ('5m' or '30m')
            
        Returns:
            Preprocessed DataFrame with OHLCV data
        """
        # Validate file exists
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        logger.info(f"Loading data from {file_path}")
        
        # Read CSV
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")
        
        # Validate minimum rows
        min_rows = self.data_config.get('min_required_rows', 1000)
        if len(df) < min_rows:
            raise ValueError(f"Insufficient data: {len(df)} rows (minimum: {min_rows})")
        
        # Process datetime
        df = self._process_datetime(df)
        
        # Standardize columns
        df = self._standardize_columns(df)
        
        # Validate data quality
        df = self._validate_data_quality(df)
        
        # Add derived features
        df = self._add_derived_features(df)
        
        logger.info(f"Loaded {len(df):,} rows from {df.index[0]} to {df.index[-1]}")
        
        return df
    
    def _process_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and set datetime index"""
        datetime_cols = ['Timestamp', 'timestamp', 'Date', 'date', 
                        'Time', 'time', 'Datetime', 'datetime']
        
        # Find datetime column
        datetime_col = None
        for col in datetime_cols:
            if col in df.columns:
                datetime_col = col
                break
        
        if not datetime_col:
            raise ValueError("No datetime column found in data")
        
        # Try different datetime formats
        formats = self.data_config.get('datetime_formats', [])
        parsed = False
        
        for fmt in formats:
            try:
                df['Datetime'] = pd.to_datetime(df[datetime_col], format=fmt)
                if df['Datetime'].notna().sum() > len(df) * 0.8:
                    parsed = True
                    break
            except:
                continue
        
        # Try automatic parsing if formats didn't work
        if not parsed:
            try:
                df['Datetime'] = pd.to_datetime(df[datetime_col])
                parsed = True
            except:
                pass
        
        if not parsed:
            raise ValueError(f"Could not parse datetime column: {datetime_col}")
        
        # Set index
        df.set_index('Datetime', inplace=True)
        df.sort_index(inplace=True)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to OHLCV format"""
        # Column mapping
        column_map = {
            'open': 'Open', 'o': 'Open', 'OPEN': 'Open',
            'high': 'High', 'h': 'High', 'HIGH': 'High',
            'low': 'Low', 'l': 'Low', 'LOW': 'Low',
            'close': 'Close', 'c': 'Close', 'CLOSE': 'Close',
            'volume': 'Volume', 'v': 'Volume', 'VOLUME': 'Volume',
            'vol': 'Volume', 'VOL': 'Volume'
        }
        
        # Rename columns
        for old_col, new_col in column_map.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        
        # Ensure numeric types
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    
    def _validate_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean data"""
        initial_len = len(df)
        
        # Check for missing values
        ohlc_cols = ['Open', 'High', 'Low', 'Close']
        missing_ratio = df[ohlc_cols].isna().sum().sum() / (len(df) * len(ohlc_cols))
        max_missing = self.data_config.get('max_missing_ratio', 0.05)
        
        if missing_ratio > max_missing:
            logger.warning(f"High missing data ratio: {missing_ratio:.2%}")
        
        # Remove rows with missing OHLC data
        df.dropna(subset=ohlc_cols, inplace=True)
        
        # Validate OHLC relationships
        invalid_mask = (
            (df['High'] < df['Low']) |
            (df['High'] < df['Open']) |
            (df['High'] < df['Close']) |
            (df['Low'] > df['Open']) |
            (df['Low'] > df['Close'])
        )
        
        if invalid_mask.any():
            logger.warning(f"Removing {invalid_mask.sum()} rows with invalid OHLC relationships")
            df = df[~invalid_mask]
        
        # Remove outliers (prices more than 10 std from mean)
        for col in ohlc_cols:
            mean = df[col].mean()
            std = df[col].std()
            outlier_mask = np.abs(df[col] - mean) > 10 * std
            if outlier_mask.any():
                logger.warning(f"Removing {outlier_mask.sum()} outliers from {col}")
                df = df[~outlier_mask]
        
        logger.info(f"Data validation complete: {initial_len} -> {len(df)} rows")
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to the DataFrame"""
        # Returns
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Volatility
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_50'] = df['returns'].rolling(50).std()
        
        # Price ranges
        df['high_low_range'] = df['High'] - df['Low']
        df['close_open_range'] = df['Close'] - df['Open']
        
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = true_range.rolling(14).mean()
        df['atr_20'] = true_range.rolling(20).mean()
        
        # Volume features
        if 'Volume' in df.columns:
            df['volume_sma_20'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
        
        return df
    
    def load_multiple_timeframes(self, config_override: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
        """Load data for multiple timeframes based on configuration"""
        if config_override:
            paths = config_override.get('paths', {})
        else:
            paths = self.data_config.get('paths', {})
        
        data = {}
        
        # Load 5-minute data
        if 'data_5min' in paths:
            try:
                data['5m'] = self.load_data(paths['data_5min'], '5m')
            except Exception as e:
                logger.error(f"Error loading 5-minute data: {e}")
                data['5m'] = pd.DataFrame()
        
        # Load 30-minute data
        if 'data_30min' in paths:
            try:
                data['30m'] = self.load_data(paths['data_30min'], '30m')
            except Exception as e:
                logger.error(f"Error loading 30-minute data: {e}")
                data['30m'] = pd.DataFrame()
        
        return data
    
    def align_timeframes(self, df_5m: pd.DataFrame, df_30m: pd.DataFrame, 
                        indicators_30m: List[str]) -> pd.DataFrame:
        """Align 30-minute indicators to 5-minute timeframe"""
        if df_5m.empty or df_30m.empty:
            raise ValueError("Cannot align empty DataFrames")
        
        # Start with 5-minute data
        df_aligned = df_5m.copy()
        
        # Align each 30-minute indicator
        for indicator in indicators_30m:
            if indicator in df_30m.columns:
                # Use forward fill to propagate 30-minute values
                aligned_indicator = df_30m[indicator].reindex(df_aligned.index, method='ffill')
                df_aligned[f'{indicator}_30m'] = aligned_indicator
            else:
                logger.warning(f"Indicator {indicator} not found in 30-minute data")
        
        # Drop rows where we don't have 30-minute data
        initial_len = len(df_aligned)
        df_aligned.dropna(subset=[f'{ind}_30m' for ind in indicators_30m 
                                  if f'{ind}_30m' in df_aligned.columns], inplace=True)
        
        logger.info(f"Timeframe alignment complete: {initial_len} -> {len(df_aligned)} rows")
        
        return df_aligned
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """Save processed data to file"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.endswith('.csv'):
            df.to_csv(output_path)
        elif output_path.endswith('.parquet'):
            df.to_parquet(output_path)
        elif output_path.endswith('.pkl'):
            df.to_pickle(output_path)
        else:
            raise ValueError(f"Unsupported file format: {output_path}")
        
        logger.info(f"Saved processed data to {output_path}")