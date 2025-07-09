"""
Optimized Indicator Implementations for AlgoSpace Strategy
Includes FVG, MLMI, and NW-RQK with caching and validation
"""

import numpy as np
import pandas as pd
from numba import jit, njit, prange, typed, types
from numba.experimental import jitclass
import logging
from typing import Dict, Tuple, Optional
from functools import lru_cache
import hashlib

logger = logging.getLogger(__name__)


class IndicatorCache:
    """Simple cache for expensive indicator calculations"""
    
    def __init__(self):
        self._cache = {}
    
    def _get_key(self, name: str, data: np.ndarray, **params) -> str:
        """Generate cache key from indicator name, data hash, and parameters"""
        data_hash = hashlib.md5(data.tobytes()).hexdigest()[:8]
        param_str = "_".join([f"{k}={v}" for k, v in sorted(params.items())])
        return f"{name}_{data_hash}_{param_str}"
    
    def get(self, name: str, data: np.ndarray, **params):
        """Get cached result if available"""
        key = self._get_key(name, data, **params)
        return self._cache.get(key)
    
    def set(self, name: str, data: np.ndarray, result, **params):
        """Cache result"""
        key = self._get_key(name, data, **params)
        self._cache[key] = result
    
    def clear(self):
        """Clear cache"""
        self._cache.clear()


# Global cache instance
_indicator_cache = IndicatorCache()


# === FVG (Fair Value Gap) Implementation ===

@njit(parallel=True, fastmath=True, cache=True)
def detect_fvg_vectorized(high: np.ndarray, low: np.ndarray, 
                          lookback: int = 3, validity: int = 20) -> Tuple:
    """
    Vectorized Fair Value Gap detection with validation
    
    Args:
        high: High prices array
        low: Low prices array
        lookback: Lookback period for gap detection
        validity: Number of bars the gap remains valid
        
    Returns:
        Tuple of (bull_fvg, bear_fvg, bull_active, bear_active)
    """
    n = len(high)
    
    # Validate inputs
    if n < lookback + 1:
        return (np.zeros(n, dtype=np.bool_), np.zeros(n, dtype=np.bool_),
                np.zeros(n, dtype=np.bool_), np.zeros(n, dtype=np.bool_))
    
    # Initialize arrays
    bull_fvg = np.zeros(n, dtype=np.bool_)
    bear_fvg = np.zeros(n, dtype=np.bool_)
    bull_active = np.zeros(n, dtype=np.bool_)
    bear_active = np.zeros(n, dtype=np.bool_)
    
    # Detect FVGs
    for i in prange(lookback, n):
        # Bullish FVG: current low > previous high
        if low[i] > high[i-lookback]:
            bull_fvg[i] = True
            
            # Mark active period
            end_idx = min(i + validity, n)
            for j in range(i, end_idx):
                if low[j] >= high[i-lookback]:
                    bull_active[j] = True
                else:
                    break
        
        # Bearish FVG: current high < previous low
        if high[i] < low[i-lookback]:
            bear_fvg[i] = True
            
            # Mark active period
            end_idx = min(i + validity, n)
            for j in range(i, end_idx):
                if high[j] <= low[i-lookback]:
                    bear_active[j] = True
                else:
                    break
    
    return bull_fvg, bear_fvg, bull_active, bear_active


def calculate_fvg(df: pd.DataFrame, lookback: int = 3, validity: int = 20,
                  use_cache: bool = True) -> pd.DataFrame:
    """
    Calculate Fair Value Gap indicators with caching
    
    Args:
        df: DataFrame with OHLC data
        lookback: Lookback period for gap detection
        validity: Number of bars the gap remains valid
        use_cache: Whether to use caching
        
    Returns:
        DataFrame with FVG indicators added
    """
    logger.info(f"Calculating FVG indicators (lookback={lookback}, validity={validity})")
    
    # Validate inputs
    if 'High' not in df.columns or 'Low' not in df.columns:
        raise ValueError("DataFrame must contain 'High' and 'Low' columns")
    
    high = df['High'].values
    low = df['Low'].values
    
    # Check cache
    if use_cache:
        cached = _indicator_cache.get('fvg', high, low=low, 
                                     lookback=lookback, validity=validity)
        if cached is not None:
            logger.info("Using cached FVG results")
            bull_fvg, bear_fvg, bull_active, bear_active = cached
        else:
            bull_fvg, bear_fvg, bull_active, bear_active = detect_fvg_vectorized(
                high, low, lookback, validity)
            _indicator_cache.set('fvg', high, 
                               (bull_fvg, bear_fvg, bull_active, bear_active),
                               low=low, lookback=lookback, validity=validity)
    else:
        bull_fvg, bear_fvg, bull_active, bear_active = detect_fvg_vectorized(
            high, low, lookback, validity)
    
    # Add to dataframe
    df['FVG_Bull_Detected'] = bull_fvg
    df['FVG_Bear_Detected'] = bear_fvg
    df['FVG_Bull_Active'] = bull_active
    df['FVG_Bear_Active'] = bear_active
    
    # Add summary statistics
    df['FVG_Score'] = bull_active.astype(int) - bear_active.astype(int)
    
    logger.info(f"FVG calculation complete: Bull={bull_fvg.sum():,}, Bear={bear_fvg.sum():,}")
    
    return df


# === MLMI (Machine Learning Market Indicator) Implementation ===

# MLMI Data Container with expanded capacity
mlmi_spec = [
    ('features', types.float64[:, :]),
    ('labels', types.int64[:]),
    ('size', types.int64),
    ('capacity', types.int64),
    ('weights', types.float64[:])  # Add weights for time decay
]

@jitclass(mlmi_spec)
class MLMIDataContainer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.features = np.zeros((capacity, 2), dtype=np.float64)
        self.labels = np.zeros(capacity, dtype=np.int64)
        self.weights = np.ones(capacity, dtype=np.float64)
        self.size = 0
    
    def add(self, f1, f2, label, weight=1.0):
        if self.size >= self.capacity:
            # Shift data (FIFO)
            shift = self.capacity // 4
            self.features[:-shift] = self.features[shift:]
            self.labels[:-shift] = self.labels[shift:]
            self.weights[:-shift] = self.weights[shift:]
            self.size = self.capacity - shift
        
        self.features[self.size, 0] = f1
        self.features[self.size, 1] = f2
        self.labels[self.size] = label
        self.weights[self.size] = weight
        self.size += 1


@njit(fastmath=True, cache=True)
def wma_optimized(values: np.ndarray, period: int) -> np.ndarray:
    """Optimized weighted moving average"""
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)
    
    if period > n or period < 1:
        return result
    
    # Pre-calculate weights
    weights = np.arange(1, period + 1, dtype=np.float64)
    sum_weights = weights.sum()
    
    # Use cumulative sum for efficiency
    for i in range(period - 1, n):
        weighted_sum = 0.0
        for j in range(period):
            weighted_sum += values[i - period + j + 1] * weights[j]
        result[i] = weighted_sum / sum_weights
    
    return result


@njit(fastmath=True, cache=True)
def rsi_optimized(prices: np.ndarray, period: int) -> np.ndarray:
    """Optimized RSI calculation with Wilder's smoothing"""
    n = len(prices)
    rsi = np.full(n, 50.0, dtype=np.float64)
    
    if period >= n or period < 1:
        return rsi
    
    # Calculate price differences
    deltas = np.zeros(n)
    for i in range(1, n):
        deltas[i] = prices[i] - prices[i-1]
    
    # Separate gains and losses
    gains = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)
    
    # Initial averages
    avg_gain = np.mean(gains[1:period+1])
    avg_loss = np.mean(losses[1:period+1])
    
    # Calculate initial RSI
    if avg_loss != 0:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))
    else:
        rsi[period] = 100.0
    
    # Calculate remaining RSI values using Wilder's smoothing
    alpha = 1.0 / period
    for i in range(period + 1, n):
        avg_gain = avg_gain * (1 - alpha) + gains[i] * alpha
        avg_loss = avg_loss * (1 - alpha) + losses[i] * alpha
        
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi


@njit(fastmath=True)
def knn_predict_weighted(features: np.ndarray, labels: np.ndarray, 
                        weights: np.ndarray, query: np.ndarray, k: int) -> float:
    """Weighted k-nearest neighbors prediction"""
    n = len(labels)
    if n == 0 or k == 0:
        return 0.0
    
    # Calculate weighted distances
    distances = np.zeros(n)
    for i in range(n):
        dist_sq = 0.0
        for j in range(len(query)):
            diff = features[i, j] - query[j]
            dist_sq += diff * diff
        distances[i] = np.sqrt(dist_sq)
    
    # Find k nearest neighbors
    k = min(k, n)
    indices = np.argsort(distances)[:k]
    
    # Weighted voting
    vote = 0.0
    weight_sum = 0.0
    
    for i in range(k):
        idx = indices[i]
        # Distance-based weight (inverse distance weighting)
        if distances[idx] > 0:
            dist_weight = 1.0 / (1.0 + distances[idx])
        else:
            dist_weight = 1.0
        
        combined_weight = weights[idx] * dist_weight
        vote += labels[idx] * combined_weight
        weight_sum += combined_weight
    
    if weight_sum > 0:
        return vote / weight_sum
    else:
        return 0.0


def calculate_mlmi(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Calculate Machine Learning Market Indicator with advanced features
    
    Args:
        df: DataFrame with OHLC data
        config: Configuration dictionary with parameters
        
    Returns:
        DataFrame with MLMI indicators added
    """
    logger.info("Calculating MLMI indicators")
    
    # Default configuration
    default_config = {
        'ma_fast_period': 5,
        'ma_slow_period': 20,
        'rsi_fast_period': 5,
        'rsi_slow_period': 20,
        'smooth_period': 20,
        'k_neighbors': 200,
        'max_data_points': 10000,
        'use_time_decay': True,
        'decay_factor': 0.995
    }
    
    if config:
        default_config.update(config)
    config = default_config
    
    close = df['Close'].values
    n = len(close)
    
    # Calculate base indicators
    ma_fast = wma_optimized(close, config['ma_fast_period'])
    ma_slow = wma_optimized(close, config['ma_slow_period'])
    rsi_fast = rsi_optimized(close, config['rsi_fast_period'])
    rsi_slow = rsi_optimized(close, config['rsi_slow_period'])
    
    # Smooth RSI values
    rsi_fast_smooth = wma_optimized(rsi_fast, config['smooth_period'])
    rsi_slow_smooth = wma_optimized(rsi_slow, config['smooth_period'])
    
    # Initialize MLMI data container
    mlmi_data = MLMIDataContainer(min(config['max_data_points'], n))
    mlmi_values = np.zeros(n)
    mlmi_confidence = np.zeros(n)
    
    # Process crossovers and build model
    for i in range(1, n):
        # Detect MA crossovers
        if ma_fast[i] > ma_slow[i] and ma_fast[i-1] <= ma_slow[i-1]:
            # Bullish crossover
            if not np.isnan(rsi_fast_smooth[i]) and not np.isnan(rsi_slow_smooth[i]):
                # Determine label based on future price movement
                if i < n-1:
                    label = 1 if close[i+1] > close[i] else -1
                else:
                    label = 0
                
                # Calculate time decay weight
                if config['use_time_decay']:
                    time_weight = config['decay_factor'] ** (n - i)
                else:
                    time_weight = 1.0
                
                mlmi_data.add(rsi_slow_smooth[i], rsi_fast_smooth[i], label, time_weight)
        
        elif ma_fast[i] < ma_slow[i] and ma_fast[i-1] >= ma_slow[i-1]:
            # Bearish crossover
            if not np.isnan(rsi_fast_smooth[i]) and not np.isnan(rsi_slow_smooth[i]):
                # Determine label
                if i < n-1:
                    label = -1 if close[i+1] < close[i] else 1
                else:
                    label = 0
                
                # Calculate time decay weight
                if config['use_time_decay']:
                    time_weight = config['decay_factor'] ** (n - i)
                else:
                    time_weight = 1.0
                
                mlmi_data.add(rsi_slow_smooth[i], rsi_fast_smooth[i], label, time_weight)
        
        # Make prediction if we have enough data
        if mlmi_data.size >= 10 and not np.isnan(rsi_fast_smooth[i]) and not np.isnan(rsi_slow_smooth[i]):
            query = np.array([rsi_slow_smooth[i], rsi_fast_smooth[i]])
            
            # Get prediction
            prediction = knn_predict_weighted(
                mlmi_data.features[:mlmi_data.size],
                mlmi_data.labels[:mlmi_data.size],
                mlmi_data.weights[:mlmi_data.size],
                query,
                min(config['k_neighbors'], mlmi_data.size)
            )
            
            mlmi_values[i] = prediction
            
            # Calculate confidence based on neighbor agreement
            k_conf = min(10, mlmi_data.size)
            conf_prediction = knn_predict_weighted(
                mlmi_data.features[:mlmi_data.size],
                mlmi_data.labels[:mlmi_data.size],
                mlmi_data.weights[:mlmi_data.size],
                query,
                k_conf
            )
            mlmi_confidence[i] = abs(conf_prediction)
    
    # Add to dataframe
    df['mlmi'] = mlmi_values
    df['mlmi_ma'] = wma_optimized(mlmi_values, config['smooth_period'])
    df['mlmi_confidence'] = mlmi_confidence
    df['mlmi_bull'] = mlmi_values > 0
    df['mlmi_bear'] = mlmi_values < 0
    
    # Detect crossovers
    mlmi_bull_cross = np.zeros(n, dtype=bool)
    mlmi_bear_cross = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        if mlmi_values[i] > 0 and mlmi_values[i-1] <= 0:
            mlmi_bull_cross[i] = True
        elif mlmi_values[i] < 0 and mlmi_values[i-1] >= 0:
            mlmi_bear_cross[i] = True
    
    df['mlmi_bull_cross'] = mlmi_bull_cross
    df['mlmi_bear_cross'] = mlmi_bear_cross
    
    # Add feature importance (based on data points)
    df['mlmi_data_points'] = 0
    for i in range(n):
        df.loc[df.index[i], 'mlmi_data_points'] = min(i, mlmi_data.size)
    
    logger.info(f"MLMI calculation complete: Range=[{mlmi_values.min():.1f}, {mlmi_values.max():.1f}]")
    
    return df


# === NW-RQK (Nadaraya-Watson Rational Quadratic Kernel) Implementation ===

@njit(fastmath=True, cache=True, inline='always')
def rational_quadratic_kernel_fast(x: float, h: float, r: float) -> float:
    """Fast rational quadratic kernel function"""
    return (1.0 + (x * x) / (h * h * 2.0 * r)) ** (-r)


@njit(parallel=True, fastmath=True, cache=True)
def nadaraya_watson_optimized(prices: np.ndarray, h: float, r: float, 
                             min_periods: int = 25, max_window: int = 500) -> np.ndarray:
    """
    Optimized Nadaraya-Watson regression with rational quadratic kernel
    
    Args:
        prices: Price series
        h: Bandwidth parameter
        r: Kernel parameter
        min_periods: Minimum periods before calculating
        max_window: Maximum lookback window for performance
        
    Returns:
        Regression values
    """
    n = len(prices)
    result = np.full(n, np.nan, dtype=np.float64)
    
    # Validate inputs
    if n < min_periods:
        return result
    
    # Pre-calculate kernel weights for common distances
    max_dist = min(max_window, n)
    kernel_cache = np.zeros(max_dist)
    for i in range(max_dist):
        kernel_cache[i] = rational_quadratic_kernel_fast(float(i), h, r)
    
    # Calculate regression
    for i in prange(min_periods, n):
        weighted_sum = 0.0
        weight_sum = 0.0
        
        # Adaptive window size
        window_size = min(i + 1, max_window)
        
        for j in range(window_size):
            if i - j >= 0:
                if j < max_dist:
                    weight = kernel_cache[j]
                else:
                    weight = rational_quadratic_kernel_fast(float(j), h, r)
                
                weighted_sum += prices[i - j] * weight
                weight_sum += weight
        
        if weight_sum > 1e-10:  # Avoid division by zero
            result[i] = weighted_sum / weight_sum
    
    return result


def calculate_nwrqk(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Calculate NW-RQK indicators with trend detection
    
    Args:
        df: DataFrame with OHLC data
        config: Configuration dictionary
        
    Returns:
        DataFrame with NW-RQK indicators added
    """
    logger.info("Calculating NW-RQK indicators")
    
    # Default configuration
    default_config = {
        'bandwidth': 8.0,
        'kernel_parameter': 8.0,
        'lag': 2,
        'min_periods': 25,
        'max_window': 500,
        'use_hl_average': False  # Option to use (H+L)/2 instead of Close
    }
    
    if config:
        default_config.update(config)
    config = default_config
    
    # Select price series
    if config['use_hl_average'] and 'High' in df.columns and 'Low' in df.columns:
        prices = (df['High'].values + df['Low'].values) / 2
    else:
        prices = df['Close'].values
    
    # Calculate regression lines
    yhat1 = nadaraya_watson_optimized(
        prices, 
        config['bandwidth'], 
        config['kernel_parameter'],
        config['min_periods'],
        config['max_window']
    )
    
    yhat2 = nadaraya_watson_optimized(
        prices, 
        config['bandwidth'] - config['lag'], 
        config['kernel_parameter'],
        config['min_periods'],
        config['max_window']
    )
    
    # Store regression values
    df['nwrqk_yhat1'] = yhat1
    df['nwrqk_yhat2'] = yhat2
    
    # Calculate regression slope
    df['nwrqk_slope'] = np.gradient(yhat1)
    
    # Detect trend changes and crossovers
    n = len(df)
    bull_change = np.zeros(n, dtype=bool)
    bear_change = np.zeros(n, dtype=bool)
    bull_cross = np.zeros(n, dtype=bool)
    bear_cross = np.zeros(n, dtype=bool)
    
    # Advanced trend detection
    for i in range(2, n):
        if not np.isnan(yhat1[i]) and not np.isnan(yhat1[i-1]) and not np.isnan(yhat1[i-2]):
            # Calculate local trend
            prev_trend = yhat1[i-1] - yhat1[i-2]
            curr_trend = yhat1[i] - yhat1[i-1]
            
            # Detect trend reversals
            if prev_trend < 0 and curr_trend > 0:
                bull_change[i] = True
            elif prev_trend > 0 and curr_trend < 0:
                bear_change[i] = True
        
        # Detect line crossovers
        if not np.isnan(yhat1[i]) and not np.isnan(yhat2[i]):
            if i > 0 and not np.isnan(yhat1[i-1]) and not np.isnan(yhat2[i-1]):
                # Bullish cross: yhat2 crosses above yhat1
                if yhat2[i] > yhat1[i] and yhat2[i-1] <= yhat1[i-1]:
                    bull_cross[i] = True
                # Bearish cross: yhat2 crosses below yhat1
                elif yhat2[i] < yhat1[i] and yhat2[i-1] >= yhat1[i-1]:
                    bear_cross[i] = True
    
    # Add indicators to dataframe
    df['nwrqk_bull'] = bull_change
    df['nwrqk_bear'] = bear_change
    df['nwrqk_bull_cross'] = bull_cross
    df['nwrqk_bear_cross'] = bear_cross
    
    # Add trend strength indicator
    df['nwrqk_trend_strength'] = np.abs(df['nwrqk_slope'])
    
    # Add divergence indicator (price vs regression)
    price_change = prices[1:] - prices[:-1]
    regression_change = yhat1[1:] - yhat1[:-1]
    divergence = np.zeros(n)
    divergence[1:] = np.where(
        (price_change * regression_change < 0) & 
        (np.abs(price_change) > 0) & 
        (np.abs(regression_change) > 0),
        np.sign(price_change),
        0
    )
    df['nwrqk_divergence'] = divergence
    
    logger.info(f"NW-RQK calculation complete: Bull={bull_change.sum():,}, Bear={bear_change.sum():,}")
    
    return df


def clear_indicator_cache():
    """Clear the indicator cache"""
    _indicator_cache.clear()
    logger.info("Indicator cache cleared")