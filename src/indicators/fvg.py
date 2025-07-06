"""
Fair Value Gap (FVG) Detection Implementation
EXTRACTED EXACTLY from Strategy_Implementation.ipynb cells 2cc4dd54 and 64b01841
Applied ONLY to 5-minute data as per notebook
"""

import numpy as np
import pandas as pd
from numba import njit
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
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


class FVGGap:
    """Represents a single Fair Value Gap with full lifecycle tracking"""
    
    def __init__(self, gap_type: str, upper_level: float, lower_level: float, 
                 creation_bar: int, current_price: float):
        self.gap_type = gap_type  # 'bullish' or 'bearish'
        self.upper_level = upper_level
        self.lower_level = lower_level
        self.creation_bar = creation_bar
        self.creation_price = current_price
        self.is_active = True
        self.mitigation_bar = None
        self.mitigation_price = None
        self.max_penetration = 0.0
        self.mitigation_strength = 0.0
        
    @property
    def gap_size(self) -> float:
        """Gap size in absolute points"""
        return abs(self.upper_level - self.lower_level)
    
    def gap_size_pct(self, current_price: float) -> float:
        """Gap size as percentage of current price"""
        if current_price <= 0:
            return 0.0
        return (self.gap_size / current_price) * 100
    
    def age(self, current_bar: int) -> int:
        """Age of gap in bars"""
        return max(0, current_bar - self.creation_bar)
    
    def update_penetration(self, current_price: float, current_bar: int) -> bool:
        """Update penetration and check if gap is mitigated"""
        if not self.is_active:
            return False
            
        if self.gap_type == 'bullish':
            # Bullish gap: price below upper level indicates penetration
            if current_price < self.upper_level:
                penetration = (self.upper_level - current_price) / self.gap_size
                self.max_penetration = max(self.max_penetration, penetration)
                
                # Check if fully mitigated (price touches lower level)
                if current_price <= self.lower_level:
                    self.is_active = False
                    self.mitigation_bar = current_bar
                    self.mitigation_price = current_price
                    return True
        else:  # bearish
            # Bearish gap: price above lower level indicates penetration
            if current_price > self.lower_level:
                penetration = (current_price - self.lower_level) / self.gap_size
                self.max_penetration = max(self.max_penetration, penetration)
                
                # Check if fully mitigated (price touches upper level)
                if current_price >= self.upper_level:
                    self.is_active = False
                    self.mitigation_bar = current_bar
                    self.mitigation_price = current_price
                    return True
        
        return False
    
    def calculate_mitigation_strength(self, volume_ratio: float = 1.0) -> float:
        """Calculate mitigation strength based on multiple factors"""
        if self.is_active or self.mitigation_bar is None:
            return 0.0
            
        # Component weights
        penetration_weight = 0.4
        speed_weight = 0.3
        volume_weight = 0.2
        age_weight = 0.1
        
        # 1. Penetration depth score (higher penetration = higher strength)
        penetration_score = min(self.max_penetration, 1.0)
        
        # 2. Speed score (faster mitigation = higher strength)
        mitigation_speed = self.mitigation_bar - self.creation_bar
        speed_score = max(0.0, 1.0 - (mitigation_speed / 20.0))  # 20 bars = 0 score
        
        # 3. Volume score (higher volume = higher strength)
        volume_score = min(volume_ratio / 2.0, 1.0)  # 2x volume = max score
        
        # 4. Age factor (younger gaps have more significance)
        age_score = max(0.0, 1.0 - (self.age(self.mitigation_bar) / 50.0))  # 50 bars = 0 score
        
        # Weighted combination
        strength = (penetration_weight * penetration_score +
                   speed_weight * speed_score +
                   volume_weight * volume_score +
                   age_weight * age_score)
        
        return min(strength, 1.0)


class FVGDetector(BaseIndicator):
    """Enhanced FVG Detector with full lifecycle tracking and mitigation analysis"""
    
    def __init__(self, config: Dict[str, Any], event_bus: EventBus):
        super().__init__(config, event_bus)
        self.threshold = config.get('threshold', 0.001)
        self.lookback_period = config.get('lookback_period', 10)
        self.body_multiplier = config.get('body_multiplier', 1.5)
        
        # Enhanced FVG parameters
        fvg_config = config.get('fvg', {})
        self.max_age = fvg_config.get('max_age', 50)
        
        # Mitigation parameters
        mitigation_config = fvg_config.get('mitigation', {})
        self.min_penetration = mitigation_config.get('min_penetration', 0.5)
        self.volume_lookback = mitigation_config.get('volume_lookback', 20)
        
        # Gap tracking
        self.active_gaps: List[FVGGap] = []
        self.recent_gaps: deque = deque(maxlen=100)  # Keep recent gaps for analysis
        self.volume_history: deque = deque(maxlen=self.volume_lookback)
        self.current_bar = 0
    
    def calculate_5m(self, bar: BarData) -> Dict[str, Any]:
        """Enhanced FVG detection on 5-minute bars with full lifecycle tracking"""
        self.update_5m_history(bar)
        self.current_bar += 1
        
        # Update volume history
        self.volume_history.append(bar.volume)
        
        if len(self.history_5m) < 3:
            return self._get_default_features()
        
        # Convert to DataFrame exactly as in notebook
        df = pd.DataFrame([{
            'High': b.high, 'Low': b.low, 'Open': b.open, 'Close': b.close
        } for b in self.history_5m])
        
        current_price = bar.close
        current_volume = bar.volume
        
        # Detect new FVGs using enhanced logic
        self._detect_new_fvgs(df, current_price)
        
        # Update all active gaps
        volume_ratio = self._calculate_volume_ratio(current_volume)
        self._update_active_gaps(current_price, volume_ratio)
        
        # Clean up old gaps
        self._cleanup_old_gaps()
        
        # Calculate all 9 enhanced features
        return self._calculate_enhanced_features(current_price, volume_ratio)
    
    def _get_default_features(self) -> Dict[str, Any]:
        """Return default feature values when insufficient data"""
        return {
            'fvg_bullish_active': False,
            'fvg_bearish_active': False,
            'fvg_nearest_level': 0.0,
            'fvg_age': 0,
            'fvg_mitigation_signal': False,
            'fvg_gap_size': 0.0,
            'fvg_gap_size_pct': 0.0,
            'fvg_mitigation_strength': 0.0,
            'fvg_mitigation_depth': 0.0
        }
    
    def _detect_new_fvgs(self, df: pd.DataFrame, current_price: float):
        """Detect new FVGs and add them to active gaps"""
        if len(df) < 3:
            return
            
        # Get the last 3 bars
        i = len(df) - 1
        first_high = df['High'].iloc[i-2]
        first_low = df['Low'].iloc[i-2]
        middle_open = df['Open'].iloc[i-1]
        middle_close = df['Close'].iloc[i-1]
        third_low = df['Low'].iloc[i]
        third_high = df['High'].iloc[i]
        
        # Calculate average body size for validation
        start_idx = max(0, i-1-self.lookback_period)
        prev_bodies = (df['Close'].iloc[start_idx:i-1] - df['Open'].iloc[start_idx:i-1]).abs()
        avg_body_size = prev_bodies.mean() if not prev_bodies.empty else 0.001
        avg_body_size = max(avg_body_size, 0.001)
        
        middle_body = abs(middle_close - middle_open)
        
        # Bullish FVG (gap up)
        if third_low > first_high and middle_body > avg_body_size * self.body_multiplier:
            gap = FVGGap('bullish', third_low, first_high, self.current_bar, current_price)
            self.active_gaps.append(gap)
            self.recent_gaps.append(gap)
        
        # Bearish FVG (gap down)
        elif third_high < first_low and middle_body > avg_body_size * self.body_multiplier:
            gap = FVGGap('bearish', first_low, third_high, self.current_bar, current_price)
            self.active_gaps.append(gap)
            self.recent_gaps.append(gap)
    
    def _calculate_volume_ratio(self, current_volume: float) -> float:
        """Calculate volume ratio vs recent average"""
        if len(self.volume_history) < 2:
            return 1.0
            
        avg_volume = np.mean(self.volume_history)
        if avg_volume <= 0:
            return 1.0
            
        return current_volume / avg_volume
    
    def _update_active_gaps(self, current_price: float, volume_ratio: float):
        """Update all active gaps and calculate mitigation"""
        mitigated_gaps = []
        
        for gap in self.active_gaps:
            if gap.update_penetration(current_price, self.current_bar):
                # Gap was mitigated
                gap.mitigation_strength = gap.calculate_mitigation_strength(volume_ratio)
                mitigated_gaps.append(gap)
        
        # Remove mitigated gaps from active list
        for gap in mitigated_gaps:
            if gap in self.active_gaps:
                self.active_gaps.remove(gap)
    
    def _cleanup_old_gaps(self):
        """Remove gaps that are too old"""
        self.active_gaps = [gap for gap in self.active_gaps 
                           if gap.age(self.current_bar) <= self.max_age]
    
    def _calculate_enhanced_features(self, current_price: float, volume_ratio: float) -> Dict[str, Any]:
        """Calculate all 9 enhanced FVG features"""
        # Initialize features
        features = {
            'fvg_bullish_active': False,
            'fvg_bearish_active': False,
            'fvg_nearest_level': 0.0,
            'fvg_age': 0,
            'fvg_mitigation_signal': False,
            'fvg_gap_size': 0.0,
            'fvg_gap_size_pct': 0.0,
            'fvg_mitigation_strength': 0.0,
            'fvg_mitigation_depth': 0.0
        }
        
        if not self.active_gaps:
            return features
        
        # Find the nearest and most significant gaps
        nearest_gap = None
        nearest_distance = float('inf')
        max_mitigation_strength = 0.0
        max_mitigation_depth = 0.0
        
        for gap in self.active_gaps:
            # Calculate distance to current price
            if gap.gap_type == 'bullish':
                distance = abs(current_price - gap.lower_level)
            else:
                distance = abs(current_price - gap.upper_level)
            
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_gap = gap
            
            # Track maximum mitigation metrics
            if gap.mitigation_strength > max_mitigation_strength:
                max_mitigation_strength = gap.mitigation_strength
            
            if gap.max_penetration > max_mitigation_depth:
                max_mitigation_depth = gap.max_penetration
        
        # Check for recently mitigated gaps
        recently_mitigated = [gap for gap in self.recent_gaps 
                             if not gap.is_active and gap.mitigation_bar is not None 
                             and (self.current_bar - gap.mitigation_bar) <= 5]
        
        if recently_mitigated:
            max_mitigation_strength = max(max_mitigation_strength, 
                                        max(g.mitigation_strength for g in recently_mitigated))
        
        # Set basic features
        bullish_active = any(gap.gap_type == 'bullish' for gap in self.active_gaps)
        bearish_active = any(gap.gap_type == 'bearish' for gap in self.active_gaps)
        
        features.update({
            'fvg_bullish_active': bullish_active,
            'fvg_bearish_active': bearish_active,
            'fvg_mitigation_signal': len(recently_mitigated) > 0,
            'fvg_mitigation_strength': max_mitigation_strength,
            'fvg_mitigation_depth': max_mitigation_depth
        })
        
        # Set nearest gap features
        if nearest_gap:
            features.update({
                'fvg_nearest_level': nearest_gap.upper_level if nearest_gap.gap_type == 'bullish' else nearest_gap.lower_level,
                'fvg_age': nearest_gap.age(self.current_bar),
                'fvg_gap_size': nearest_gap.gap_size,
                'fvg_gap_size_pct': nearest_gap.gap_size_pct(current_price)
            })
        
        return features
    
    
    def get_current_values(self) -> Dict[str, Any]:
        """Get current FVG feature values"""
        if not self.active_gaps:
            return self._get_default_features()
        
        # Return features based on current active gaps
        current_price = self.history_5m[-1].close if self.history_5m else 0.0
        volume_ratio = 1.0
        if len(self.volume_history) > 1:
            volume_ratio = self._calculate_volume_ratio(self.volume_history[-1])
        
        return self._calculate_enhanced_features(current_price, volume_ratio)
    
    def reset(self) -> None:
        """Reset all FVG state"""
        self.history_5m = []
        self.active_gaps = []
        self.recent_gaps.clear()
        self.volume_history.clear()
        self.current_bar = 0