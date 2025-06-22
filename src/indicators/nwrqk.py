"""
Nadaraya-Watson Rational Quadratic Kernel (NW-RQK) Implementation
EXTRACTED EXACTLY from Strategy_Implementation.ipynb cell 381cc2a1
Default parameters: h=8.0, r=8.0, x_0=25, lag=2
"""

import numpy as np
import pandas as pd
from numba import jit, njit, prange, float64, int64, boolean
from typing import Dict, Any
from src.indicators.base import BaseIndicator
from src.core.events import EventBus, BarData


@njit(float64(float64[:], int64, float64, float64, int64))
def kernel_regression_numba(src, size, h_param, r_param, x_0_param):
    """
    Numba-optimized Nadaraya-Watson Regression using Rational Quadratic Kernel
    """
    current_weight = 0.0
    cumulative_weight = 0.0

    # Calculate only up to the available data points
    for i in range(min(size + x_0_param + 1, len(src))):
        if i < len(src):
            y = src[i]  # Value i bars back
            # Rational Quadratic Kernel
            w = (1 + (i**2 / ((h_param**2) * 2 * r_param)))**(-r_param)
            current_weight += y * w
            cumulative_weight += w

    if cumulative_weight == 0:
        return np.nan

    return current_weight / cumulative_weight


@njit(parallel=True)
def calculate_nw_regression(prices, h_param, h_lag_param, r_param, x_0_param):
    """
    Calculate Nadaraya-Watson regression for the entire price series
    """
    n = len(prices)
    yhat1 = np.full(n, np.nan)
    yhat2 = np.full(n, np.nan)

    # Reverse the array once to match PineScript indexing
    prices_reversed = np.zeros(n)
    for i in range(n):
        prices_reversed[i] = prices[n-i-1]

    # Calculate regression values for each bar in parallel
    for i in prange(n):
        if i >= x_0_param:  # Only start calculation after x_0 bars
            # Create window for current bar
            window_size = min(i + 1, n)
            src = np.zeros(window_size)
            for j in range(window_size):
                src[j] = prices[i-j]

            yhat1[i] = kernel_regression_numba(src, i, h_param, r_param, x_0_param)
            yhat2[i] = kernel_regression_numba(src, i, h_lag_param, r_param, x_0_param)

    return yhat1, yhat2


@njit
def detect_crosses(yhat1, yhat2):
    """
    Detect crossovers between two series
    """
    n = len(yhat1)
    bullish_cross = np.zeros(n, dtype=np.bool_)
    bearish_cross = np.zeros(n, dtype=np.bool_)

    for i in range(1, n):
        if not np.isnan(yhat1[i]) and not np.isnan(yhat2[i]) and \
           not np.isnan(yhat1[i-1]) and not np.isnan(yhat2[i-1]):
            # Bullish cross (yhat2 crosses above yhat1)
            if yhat2[i] > yhat1[i] and yhat2[i-1] <= yhat1[i-1]:
                bullish_cross[i] = True

            # Bearish cross (yhat2 crosses below yhat1)
            if yhat2[i] < yhat1[i] and yhat2[i-1] >= yhat1[i-1]:
                bearish_cross[i] = True

    return bullish_cross, bearish_cross


class NWRQKCalculator(BaseIndicator):
    """NW-RQK Calculator using EXACT parameters from notebook: h=8.0, r=8.0"""
    
    def __init__(self, config: Dict[str, Any], event_bus: EventBus):
        super().__init__(config, event_bus)
        # EXACT parameters from notebook
        self.h = config.get('h', 8.0)
        self.r = config.get('r', 8.0) 
        self.x_0 = config.get('x_0', 25)
        self.lag = config.get('lag', 2)
        self.smooth_colors = config.get('smooth_colors', False)
    
    
    def calculate_30m(self, bar: BarData) -> Dict[str, Any]:
        self.update_30m_history(bar)
        if len(self.history_30m) < 50:
            return {'nwrqk_value': 0.0, 'nwrqk_signal': 0}
        
        # Convert to DataFrame exactly as in notebook
        df = pd.DataFrame([{
            'Close': b.close, 'High': b.high, 'Low': b.low, 'Open': b.open, 'Volume': b.volume
        } for b in self.history_30m])
        
        # Apply EXACT calculation from notebook
        prices = df['Close'].values
        yhat1, yhat2 = calculate_nw_regression(prices, self.h, self.h-self.lag, self.r, self.x_0)
        
        # Add to dataframe exactly as in notebook
        df['yhat1'] = yhat1
        df['yhat2'] = yhat2
        
        # Calculate rates of change (vectorized) - EXACT from notebook
        df['wasBearish'] = df['yhat1'].shift(2) > df['yhat1'].shift(1)
        df['wasBullish'] = df['yhat1'].shift(2) < df['yhat1'].shift(1)
        df['isBearish'] = df['yhat1'].shift(1) > df['yhat1']
        df['isBullish'] = df['yhat1'].shift(1) < df['yhat1']
        df['isBearishChange'] = df['isBearish'] & df['wasBullish']
        df['isBullishChange'] = df['isBullish'] & df['wasBearish']
        
        # Calculate crossovers using Numba - EXACT from notebook
        bullish_cross, bearish_cross = detect_crosses(yhat1, yhat2)
        df['isBullishCross'] = bullish_cross
        df['isBearishCross'] = bearish_cross
        
        # Generate alert stream exactly as in notebook
        df['alertBullish'] = df['isBearishCross'] if self.smooth_colors else df['isBearishChange']
        df['alertBearish'] = df['isBullishCross'] if self.smooth_colors else df['isBullishChange']
        df['alertStream'] = np.where(df['alertBearish'], -1, np.where(df['alertBullish'], 1, 0))
        
        # Get current values
        current_idx = len(df) - 1
        nwrqk_value = yhat1[current_idx] if not np.isnan(yhat1[current_idx]) else 0.0
        nwrqk_signal = int(df['alertStream'].iloc[current_idx]) if current_idx < len(df) else 0
        
        return {'nwrqk_value': float(nwrqk_value), 'nwrqk_signal': nwrqk_signal}
    
    def get_current_values(self) -> Dict[str, Any]:
        return {'nwrqk_value': 0.0, 'nwrqk_signal': 0}
    
    def reset(self) -> None:
        self.history_30m = []