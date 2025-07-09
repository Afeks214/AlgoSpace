"""
Robust Data Loading Module for AlgoSpace Strategy
Handles data loading, validation, preprocessing with comprehensive error handling
"""

import os
import pandas as pd
import numpy as np
import yaml
from typing import Tuple, List, Optional, Dict, Any
import logging
from datetime import datetime

class DataLoader:
    """Production-ready data loader with validation and error handling"""
    
    def __init__(self, config_path: str):
        """Initialize data loader with configuration"""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        self.data_cache = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        
        logger = logging.getLogger('DataLoader')
        logger.setLevel(getattr(logging, log_config.get('level', 'INFO')))
        
        # Console handler
        if log_config.get('console', True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter(log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            )
            logger.addHandler(console_handler)
        
        # File handler
        if 'file' in log_config:
            os.makedirs(os.path.dirname(log_config['file']), exist_ok=True)
            file_handler = logging.FileHandler(log_config['file'])
            file_handler.setFormatter(
                logging.Formatter(log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            )
            logger.addHandler(file_handler)
            
        return logger
    
    def load_data(self, validate: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and validate both 30m and 5m data
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 30m and 5m dataframes
        """
        self.logger.info("Starting data loading process...")
        
        # Load 30m data
        df_30m = self._load_single_timeframe('btc_30m_file', '30m')
        
        # Load 5m data
        df_5m = self._load_single_timeframe('btc_5m_file', '5m')
        
        if validate:
            df_30m = self._validate_data(df_30m, '30m')
            df_5m = self._validate_data(df_5m, '5m')
        
        # Add derived features
        df_30m = self._add_features(df_30m)
        df_5m = self._add_features(df_5m)
        
        self.logger.info("Data loading completed successfully")
        return df_30m, df_5m
    
    def _load_single_timeframe(self, file_key: str, timeframe: str) -> pd.DataFrame:
        """Load data for a single timeframe with error handling"""
        data_config = self.config['data']
        file_path = os.path.join(data_config['base_path'], data_config[file_key])
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        self.logger.info(f"Loading {timeframe} data from {file_path}")
        
        try:
            # Load CSV
            df = pd.read_csv(file_path)
            
            # Parse datetime with multiple format attempts
            datetime_parsed = False
            for fmt in data_config['datetime_formats']:
                try:
                    df['datetime'] = pd.to_datetime(df['datetime'], format=fmt)
                    datetime_parsed = True
                    self.logger.debug(f"Successfully parsed datetime with format: {fmt}")
                    break
                except Exception:
                    continue
            
            if not datetime_parsed:
                # Fallback to pandas automatic parsing
                df['datetime'] = pd.to_datetime(df['datetime'])
                self.logger.warning(f"Used automatic datetime parsing for {timeframe} data")
            
            # Set index and sort
            df = df.set_index('datetime').sort_index()
            
            # Check for required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            self.logger.info(f"Loaded {len(df)} rows for {timeframe} data")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading {timeframe} data: {str(e)}")
            raise
    
    def _validate_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Validate data quality and handle issues"""
        validation_config = self.config['data']['validation']
        self.logger.info(f"Validating {timeframe} data...")
        
        # Check for missing values
        if validation_config['check_missing']:
            missing_count = df.isnull().sum()
            total_missing = missing_count.sum()
            
            if total_missing > 0:
                missing_pct = total_missing / (len(df) * len(df.columns))
                self.logger.warning(f"Found {total_missing} missing values ({missing_pct:.2%}) in {timeframe} data")
                
                if missing_pct > validation_config['max_missing_pct']:
                    raise ValueError(f"Too many missing values: {missing_pct:.2%}")
                
                # Forward fill missing values
                df = df.fillna(method='ffill').fillna(method='bfill')
                self.logger.info(f"Filled missing values in {timeframe} data")
        
        # Check for outliers
        if validation_config['check_outliers']:
            outlier_threshold = validation_config['outlier_std_threshold']
            
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    # Calculate rolling statistics
                    rolling_mean = df[col].rolling(window=100, min_periods=1).mean()
                    rolling_std = df[col].rolling(window=100, min_periods=1).std()
                    
                    # Identify outliers
                    outliers = np.abs((df[col] - rolling_mean) / rolling_std) > outlier_threshold
                    n_outliers = outliers.sum()
                    
                    if n_outliers > 0:
                        self.logger.warning(f"Found {n_outliers} outliers in {col} for {timeframe} data")
                        
                        # Cap outliers at threshold
                        df.loc[outliers, col] = rolling_mean[outliers] + np.sign(
                            df.loc[outliers, col] - rolling_mean[outliers]
                        ) * outlier_threshold * rolling_std[outliers]
        
        # Validate price relationships
        invalid_candles = (df['high'] < df['low']) | (df['high'] < df['close']) | (df['low'] > df['close'])
        if invalid_candles.any():
            n_invalid = invalid_candles.sum()
            self.logger.warning(f"Found {n_invalid} invalid candles in {timeframe} data")
            
            # Fix invalid candles
            df.loc[invalid_candles, 'high'] = df.loc[invalid_candles, ['open', 'close']].max(axis=1)
            df.loc[invalid_candles, 'low'] = df.loc[invalid_candles, ['open', 'close']].min(axis=1)
        
        # Check for duplicate timestamps
        duplicates = df.index.duplicated()
        if duplicates.any():
            n_duplicates = duplicates.sum()
            self.logger.warning(f"Found {n_duplicates} duplicate timestamps in {timeframe} data")
            df = df[~duplicates]
        
        # Ensure data is sorted
        if not df.index.is_monotonic_increasing:
            self.logger.warning(f"Data not properly sorted for {timeframe}, sorting now")
            df = df.sort_index()
        
        self.logger.info(f"Validation completed for {timeframe} data")
        return df
    
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to dataframe"""
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Calculate log returns
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Calculate volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Calculate volume metrics
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Calculate price ranges
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['close_open_range'] = (df['close'] - df['open']) / df['open']
        
        # VWAP
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        return df
    
    def get_aligned_data(self, start_date: Optional[str] = None, 
                        end_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get time-aligned data for both timeframes"""
        df_30m, df_5m = self.load_data()
        
        # Apply date filters if provided
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df_30m = df_30m[df_30m.index >= start_dt]
            df_5m = df_5m[df_5m.index >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            df_30m = df_30m[df_30m.index <= end_dt]
            df_5m = df_5m[df_5m.index <= end_dt]
        
        # Ensure 5m data covers the same period as 30m data
        df_5m = df_5m[(df_5m.index >= df_30m.index[0]) & (df_5m.index <= df_30m.index[-1])]
        
        self.logger.info(f"Aligned data: 30m has {len(df_30m)} rows, 5m has {len(df_5m)} rows")
        
        return df_30m, df_5m