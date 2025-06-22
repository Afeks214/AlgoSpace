"""
Fair Value Gap (FVG) Detection Implementation
EXTRACTED EXACTLY from Strategy_Implementation.ipynb cells 2cc4dd54 and 64b01841
Applied ONLY to 5-minute data as per notebook
"""

import numpy as np
import pandas as pd
from numba import njit
from typing import Dict, Any
from src.indicators.base import BaseIndicator
from src.core.events import EventBus, BarData


def detect_fvg(df, lookback_period=10, body_multiplier=1.5):
    """
    Detects Fair Value Gaps (FVGs) in historical price data.
    EXTRACTED EXACTLY from cell 2cc4dd54
    """
    fvg_list = [None] * len(df)

    if len(df) < 3:
        return fvg_list

    for i in range(2, len(df)):
        try:
            first_high = df['High'].iloc[i-2]
            first_low = df['Low'].iloc[i-2]
            middle_open = df['Open'].iloc[i-1]
            middle_close = df['Close'].iloc[i-1]
            third_low = df['Low'].iloc[i]
            third_high = df['High'].iloc[i]

            start_idx = max(0, i-1-lookback_period)
            prev_bodies = (df['Close'].iloc[start_idx:i-1] - df['Open'].iloc[start_idx:i-1]).abs()
            avg_body_size = prev_bodies.mean() if not prev_bodies.empty else 0.001
            avg_body_size = max(avg_body_size, 0.001)

            middle_body = abs(middle_close - middle_open)

            # Bullish FVG (gap up)
            if third_low > first_high and middle_body > avg_body_size * body_multiplier:
                fvg_list[i] = ('bullish', first_high, third_low, i)

            # Bearish FVG (gap down)
            elif third_high < first_low and middle_body > avg_body_size * body_multiplier:
                fvg_list[i] = ('bearish', first_low, third_high, i)

        except Exception as e:
            continue

    return fvg_list


@njit
def generate_fvg_data_fast(high, low, n):
    """
    Numba-optimized FVG generation
    EXTRACTED EXACTLY from cell 64b01841
    """
    bull_fvg_detected = np.zeros(n, dtype=np.bool_)
    bear_fvg_detected = np.zeros(n, dtype=np.bool_)
    is_bull_fvg_active = np.zeros(n, dtype=np.bool_)
    is_bear_fvg_active = np.zeros(n, dtype=np.bool_)

    for i in range(2, n):
        # Bullish FVG: Current low > Previous high
        if low[i] > high[i-2]:
            bull_fvg_detected[i] = True

            for j in range(i, min(i+20, n)):
                is_bull_fvg_active[j] = True
                if low[j] < high[i-2]:
                    break

        # Bearish FVG: Current high < Previous low
        if high[i] < low[i-2]:
            bear_fvg_detected[i] = True

            for j in range(i, min(i+20, n)):
                is_bear_fvg_active[j] = True
                if high[j] > low[i-2]:
                    break

    return bull_fvg_detected, bear_fvg_detected, is_bull_fvg_active, is_bear_fvg_active


class FVGDetector(BaseIndicator):
    """FVG Detector - ONLY processes 5-minute data as per notebook"""
    
    def __init__(self, config: Dict[str, Any], event_bus: EventBus):
        super().__init__(config, event_bus)
        self.threshold = config.get('threshold', 0.001)
        self.lookback_period = config.get('lookback_period', 10)
        self.body_multiplier = config.get('body_multiplier', 1.5)
    
    def calculate_5m(self, bar: BarData) -> Dict[str, Any]:
        """FVG detection on 5-minute bars - EXACT application from notebook"""
        self.update_5m_history(bar)
        if len(self.history_5m) < 3:
            return {'fvg_bullish_active': False, 'fvg_bearish_active': False}
        
        # Convert to DataFrame exactly as in notebook
        df = pd.DataFrame([{
            'High': b.high, 'Low': b.low, 'Open': b.open, 'Close': b.close
        } for b in self.history_5m])
        
        # Use fast Numba version (from cell 64b01841)
        high_array = df['High'].values
        low_array = df['Low'].values
        n = len(df)
        
        bull_fvg_detected, bear_fvg_detected, is_bull_fvg_active, is_bear_fvg_active = generate_fvg_data_fast(
            high_array, low_array, n
        )
        
        current_idx = n - 1
        return {
            'fvg_bullish_active': bool(is_bull_fvg_active[current_idx]),
            'fvg_bearish_active': bool(is_bear_fvg_active[current_idx]),
            'fvg_nearest_level': high_array[current_idx-2] if is_bull_fvg_active[current_idx] and current_idx >= 2 else (
                low_array[current_idx-2] if is_bear_fvg_active[current_idx] and current_idx >= 2 else 0.0),
            'fvg_age': 0,
            'fvg_mitigation_signal': False
        }
    
    
    def get_current_values(self) -> Dict[str, Any]:
        return {'fvg_bullish_active': False, 'fvg_bearish_active': False}
    
    def reset(self) -> None:
        self.history_5m = []