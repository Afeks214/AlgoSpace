"""
Technical Indicators for AlgoSpace Training
"""

import numpy as np
import pandas as pd
from typing import Union, List

def sma(data: Union[pd.Series, np.ndarray], window: int) -> np.ndarray:
    """Simple Moving Average"""
    return pd.Series(data).rolling(window).mean().values

def ema(data: Union[pd.Series, np.ndarray], window: int) -> np.ndarray:
    """Exponential Moving Average"""
    return pd.Series(data).ewm(span=window).mean().values

def rsi(data: Union[pd.Series, np.ndarray], window: int = 14) -> np.ndarray:
    """Relative Strength Index"""
    delta = pd.Series(data).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return (100 - (100 / (1 + rs))).values

def bollinger_bands(data: Union[pd.Series, np.ndarray], 
                   window: int = 20, 
                   num_std: float = 2) -> tuple:
    """Bollinger Bands"""
    series = pd.Series(data)
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    
    return rolling_mean.values, upper_band.values, lower_band.values

def macd(data: Union[pd.Series, np.ndarray], 
         fast: int = 12, 
         slow: int = 26, 
         signal: int = 9) -> tuple:
    """MACD Indicator"""
    series = pd.Series(data)
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    return macd_line.values, signal_line.values, histogram.values

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all indicators for training data"""
    result = df.copy()
    
    # Price-based indicators
    result['sma_10'] = sma(df['close'], 10)
    result['sma_20'] = sma(df['close'], 20)
    result['ema_10'] = ema(df['close'], 10)
    result['ema_20'] = ema(df['close'], 20)
    result['rsi'] = rsi(df['close'])
    
    # Bollinger Bands
    bb_middle, bb_upper, bb_lower = bollinger_bands(df['close'])
    result['bb_middle'] = bb_middle
    result['bb_upper'] = bb_upper
    result['bb_lower'] = bb_lower
    
    # MACD
    macd_line, signal_line, histogram = macd(df['close'])
    result['macd'] = macd_line
    result['macd_signal'] = signal_line
    result['macd_histogram'] = histogram
    
    return result