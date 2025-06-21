"""
MLMI (Machine Learning Market Indicator) Implementation
Extracts ONLY the core MLMI calculation functions from notebook
Default parameters: k=5, trend_length=14 (DO NOT CHANGE per PRD)
"""

import numpy as np
import pandas as pd
from numba import njit, prange, float64, int64
from numba.experimental import jitclass
from scipy.spatial import cKDTree
from typing import Dict, Any
from src.indicators.base import BaseIndicator
from src.core.events import EventBus, BarData


# EXTRACTED: JIT-compiled MLMI data storage
spec = [('parameter1', float64[:]), ('parameter2', float64[:]), 
        ('priceArray', float64[:]), ('resultArray', int64[:]), ('size', int64)]

@jitclass(spec)
class MLMIDataFast:
    def __init__(self, max_size=10000):
        self.parameter1 = np.zeros(max_size, dtype=np.float64)
        self.parameter2 = np.zeros(max_size, dtype=np.float64)
        self.priceArray = np.zeros(max_size, dtype=np.float64)
        self.resultArray = np.zeros(max_size, dtype=np.int64)
        self.size = 0

    def storePreviousTrade(self, p1, p2, close_price):
        if self.size > 0:
            result = 1 if close_price >= self.priceArray[self.size-1] else -1
            self.size += 1
            self.parameter1[self.size-1] = p1
            self.parameter2[self.size-1] = p2
            self.priceArray[self.size-1] = close_price
            self.resultArray[self.size-1] = result
        else:
            self.parameter1[0] = p1
            self.parameter2[0] = p2
            self.priceArray[0] = close_price
            self.resultArray[0] = 0
            self.size = 1


@njit(parallel=True)
def calculate_wma(series, length):
    """Weighted Moving Average calculation"""
    n = len(series)
    result = np.zeros(n)
    weights = np.arange(1, length + 1, dtype=np.float64)
    sum_weights = np.sum(weights)
    
    for i in prange(length-1, n):
        weighted_sum = 0.0
        for j in range(length):
            weighted_sum += series[i-j] * weights[length-j-1]
        result[i] = weighted_sum / sum_weights
    return result


@njit
def calculate_rsi_with_ma(prices, window):
    """RSI calculation with Wilder's smoothing"""
    n = len(prices)
    rsi = np.zeros(n)
    delta = np.zeros(n)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    for i in range(1, n):
        delta[i] = prices[i] - prices[i-1]
        if delta[i] > 0:
            gain[i] = delta[i]
        else:
            loss[i] = -delta[i]
    
    if window <= n:
        avg_gain = np.sum(gain[:window]) / window
        avg_loss = np.sum(loss[:window]) / window
        
        if avg_loss == 0:
            rsi[window-1] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[window-1] = 100 - (100 / (1 + rs))
    
    for i in range(window, n):
        avg_gain = (avg_gain * (window-1) + gain[i]) / window
        avg_loss = (avg_loss * (window-1) + loss[i]) / window
        
        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


def fast_knn_predict(param1_array, param2_array, result_array, p1, p2, k, size):
    """Fast k-NN prediction using cKDTree"""
    if size == 0:
        return 0
    points = np.column_stack((param1_array[:size], param2_array[:size]))
    tree = cKDTree(points)
    distances, indices = tree.query([p1, p2], k=min(k, size))
    return np.sum(result_array[indices])


class MLMICalculator(BaseIndicator):
    """MLMI Calculator - EXACT extraction from notebook cell 1331cd64. Default: neighbors=200, momentum=20"""
    
    def __init__(self, config: Dict[str, Any], event_bus: EventBus):
        super().__init__(config, event_bus)
        # EXACT parameters from notebook cell 1331cd64
        self.num_neighbors = config.get('num_neighbors', 200)
        self.momentum_window = config.get('momentum_window', 20)
        self.ma_quick_period = config.get('ma_quick_period', 5)
        self.ma_slow_period = config.get('ma_slow_period', 20)
        self.rsi_quick_period = config.get('rsi_quick_period', 5)
        self.rsi_slow_period = config.get('rsi_slow_period', 20)
        self.mlmi_ma_period = config.get('mlmi_ma_period', 20)
        self.band_lookback = config.get('band_lookback', 2000)
        self.std_window = config.get('std_window', 20)
        self.ema_std_span = config.get('ema_std_span', 20)
        
        self.mlmi_data = MLMIDataFast()
        self.last_mlmi_value = 0.0
        self.last_mlmi_signal = 0
    
    def calculate_5m(self, bar: BarData) -> Dict[str, Any]:
        return {}  # MLMI only uses 30m data
    
    def calculate_30m(self, bar: BarData) -> Dict[str, Any]:
        self.update_30m_history(bar)
        if len(self.history_30m) < 100:  # Need sufficient data
            return {'mlmi_value': 0.0, 'mlmi_signal': 0}
        
        # Convert to Heiken Ashi (per PRD DIR-DATA-01)
        ha_bars = self.convert_to_heiken_ashi(self.history_30m)
        close_prices = np.array([bar['close'] for bar in ha_bars])
        
        # Calculate indicators using EXACT notebook parameters
        ma_quick = calculate_wma(close_prices, self.ma_quick_period)
        ma_slow = calculate_wma(close_prices, self.ma_slow_period)
        rsi_quick = calculate_rsi_with_ma(close_prices, self.rsi_quick_period)
        rsi_slow = calculate_rsi_with_ma(close_prices, self.rsi_slow_period)
        rsi_quick_wma = calculate_wma(rsi_quick, self.momentum_window)
        rsi_slow_wma = calculate_wma(rsi_slow, self.momentum_window)
        
        # Detect crossovers and update MLMI data
        n = len(close_prices)
        for i in range(1, n):
            if ((ma_quick[i] > ma_slow[i] and ma_quick[i-1] <= ma_slow[i-1]) or
                (ma_quick[i] < ma_slow[i] and ma_quick[i-1] >= ma_slow[i-1])):
                if not np.isnan(rsi_slow_wma[i]) and not np.isnan(rsi_quick_wma[i]):
                    self.mlmi_data.storePreviousTrade(
                        rsi_slow_wma[i], rsi_quick_wma[i], close_prices[i])
        
        # Calculate current MLMI value
        current_idx = n - 1
        if (not np.isnan(rsi_slow_wma[current_idx]) and 
            not np.isnan(rsi_quick_wma[current_idx]) and self.mlmi_data.size > 0):
            mlmi_value = fast_knn_predict(
                self.mlmi_data.parameter1, self.mlmi_data.parameter2,
                self.mlmi_data.resultArray, rsi_slow_wma[current_idx],
                rsi_quick_wma[current_idx], self.num_neighbors, self.mlmi_data.size)
        else:
            mlmi_value = 0.0
        
        # Generate signal
        mlmi_signal = 0
        if current_idx > 0:
            ma_bullish = (ma_quick[current_idx] > ma_slow[current_idx] and 
                         ma_quick[current_idx-1] <= ma_slow[current_idx-1])
            ma_bearish = (ma_quick[current_idx] < ma_slow[current_idx] and 
                         ma_quick[current_idx-1] >= ma_slow[current_idx-1])
            
            if mlmi_value > 0 and ma_bullish:
                mlmi_signal = 1
            elif mlmi_value < 0 and ma_bearish:
                mlmi_signal = -1
            elif mlmi_value > 2:
                mlmi_signal = 1
            elif mlmi_value < -2:
                mlmi_signal = -1
        
        self.last_mlmi_value = float(mlmi_value)
        self.last_mlmi_signal = int(mlmi_signal)
        
        return {'mlmi_value': self.last_mlmi_value, 'mlmi_signal': self.last_mlmi_signal}
    
    def get_current_values(self) -> Dict[str, Any]:
        return {'mlmi_value': self.last_mlmi_value, 'mlmi_signal': self.last_mlmi_signal}
    
    def reset(self) -> None:
        self.mlmi_data = MLMIDataFast()
        self.last_mlmi_value = 0.0
        self.last_mlmi_signal = 0
        self.history_30m = []