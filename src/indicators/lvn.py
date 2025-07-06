"""
Low Volume Node (LVN) Core Calculations with Advanced Strength Scoring

This module provides LVN identification and sophisticated strength score calculation
based on market profile analysis and historical price action.

Features:
- Real-time LVN detection using volume profile analysis
- Sophisticated strength scoring (0-100) based on:
  * Volume profile characteristics (40 points)
  * Historical interaction analysis (30 points)
  * Recency of price tests (20 points)
  * Distance from current price (10 points)
- Historical tracking of LVN levels and price interactions
- Comprehensive analysis of price rejection patterns

Enhanced from: LVN_Implementation.ipynb with production-ready streaming analysis
"""

import pandas as pd
import numpy as np
from numba import jit
import warnings
from typing import Dict, Any, List, Optional
from collections import deque
from ..core.events import BarData

warnings.filterwarnings("ignore", category=UserWarning)


# Mock MarketProfile class for now
class MarketProfile:
    """Simple mock implementation of MarketProfile for testing"""
    def __init__(self, df, tick_size=0.25):
        self.df = df
        self.tick_size = tick_size
        
    def __getitem__(self, time_slice):
        return self
        
    @property
    def profile(self):
        # Simple volume profile based on price levels
        if self.df.empty:
            return pd.Series()
        
        price_min = self.df['Low'].min()
        price_max = self.df['High'].max()
        
        # Create price levels
        levels = np.arange(price_min, price_max + self.tick_size, self.tick_size)
        volume_profile = pd.Series(index=levels, dtype=float)
        
        # Distribute volume across price levels
        for _, row in self.df.iterrows():
            level_range = np.arange(row['Low'], row['High'] + self.tick_size, self.tick_size)
            vol_per_level = row['Volume'] / len(level_range) if len(level_range) > 0 else 0
            for level in level_range:
                if level in volume_profile.index:
                    volume_profile[level] = volume_profile.get(level, 0) + vol_per_level
                    
        return volume_profile
        
    @property
    def poc_price(self):
        profile = self.profile
        if profile.empty:
            return 0
        return profile.idxmax()
        
    @property
    def value_area(self):
        profile = self.profile
        if profile.empty:
            return (0, 0)
        poc = self.poc_price
        return (poc + 1, poc - 1)  # Simplified value area


@jit(nopython=True)
def find_post_lvn_range(prices: np.ndarray) -> tuple[float, float]:
    """
    Numba-accelerated function to find the min and max price in a window.
    """
    if len(prices) == 0:
        return (np.nan, np.nan)
    return (np.min(prices), np.max(prices))


def identify_and_classify_lvns(price_df: pd.DataFrame,
                               observation_window: int = 5,
                               breach_threshold: float = 0.25) -> pd.DataFrame:
    """
    Identifies all daily LVNs and classifies their outcome.

    Args:
        price_df: DataFrame with OHLC data and datetime index
        observation_window: How many bars to look forward to classify the outcome
        breach_threshold: The price distance to confirm a breach through an LVN

    Returns:
        DataFrame containing all identified LVNs with their characteristics
    """
    all_lvns = []
    daily_groups = price_df.groupby(price_df.index.date)

    for date, day_df in daily_groups:
        if len(day_df) < 2:
            continue

        try:
            # A. Identify LVNs for the current day
            mp = MarketProfile(day_df, tick_size=0.25)
            mp_slice = mp[day_df.index.min():day_df.index.max()]

            volume_profile = mp_slice.profile
            if volume_profile is None or volume_profile.empty:
                continue

            # Define LVN as a price level with volume below 20% of the daily average
            mean_volume = volume_profile.mean()
            if pd.isna(mean_volume) or mean_volume == 0:
                continue

            lvn_prices = volume_profile[volume_profile < mean_volume * 0.20].index.tolist()

            if not lvn_prices:
                continue

            # Find the timestamp when the profile is complete (end of the day)
            day_end_time = day_df.index.max()

            for lvn_price in lvn_prices:
                all_lvns.append({
                    'lvn_timestamp': day_end_time,
                    'lvn_price': lvn_price
                )}

        except Exception:
            continue

    if not all_lvns:
        return pd.DataFrame()

    lvn_df = pd.DataFrame(all_lvns)
    lvn_df.set_index('lvn_timestamp', inplace=True)

    # B. Classify LVN Outcomes 
    outcomes = []
    for index, row in lvn_df.iterrows():
        lvn_price = row['lvn_price']

        # Define the observation window starting from the bar AFTER the LVN is confirmed
        window_start_idx = price_df.index.searchsorted(index) + 1
        window_end_idx = window_start_idx + observation_window

        observation_df = price_df.iloc[window_start_idx:window_end_idx]

        if observation_df.empty:
            outcomes.append(np.nan)
            continue

        # Determine the direction of approach to the LVN
        last_close_before_lvn = price_df.loc[index, 'Close']
        approached_from_below = last_close_before_lvn < lvn_price

        # Use the Numba function
        low_in_window, high_in_window = find_post_lvn_range(observation_df[['Low', 'High']].values)

        # Define Outcome Conditions
        is_weak = False
        if approached_from_below:
            # Weak if price breaks decisively above the LVN
            if high_in_window > lvn_price + breach_threshold:
                is_weak = True
        else: # Approached from above
            # Weak if price breaks decisively below the LVN
            if low_in_window < lvn_price - breach_threshold:
                is_weak = True

        # Outcome: 0 for Weak (Acceptance), 1 for Strong (Rejection)
        outcomes.append(0 if is_weak else 1)

    lvn_df['outcome'] = outcomes
    lvn_df.dropna(subset=['outcome'], inplace=True)
    lvn_df['outcome'] = lvn_df['outcome'].astype(int)

    return lvn_df


def calculate_lvn_characteristics(lvn_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate LVN characteristics: strength, width, range, etc.
    
    Args:
        lvn_df: DataFrame with identified LVNs
        price_df: Original price data
        
    Returns:
        DataFrame with LVN characteristics added
    """
    enhanced_lvns = []
    
    for index, row in lvn_df.iterrows():
        date = index.date()
        day_df = price_df[price_df.index.date == date]
        
        if day_df.empty:
            continue
            
        try:
            # Calculate market profile for the day
            mp = MarketProfile(day_df, tick_size=0.25)
            mp_slice = mp[day_df.index.min():day_df.index.max()]
            
            volume_profile = mp_slice.profile
            poc = mp_slice.poc_price
            value_area = mp_slice.value_area
            
            lvn_price = row['lvn_price']
            lvn_volume = volume_profile.loc[lvn_price] if lvn_price in volume_profile.index else 0
            
            # LVN Strength: relative to mean daily volume
            mean_volume = volume_profile.mean()
            strength = 1 - (lvn_volume / mean_volume) if mean_volume > 0 else 0
            
            # LVN Width: price range of low volume area around LVN
            low_vol_threshold = mean_volume * 0.20
            low_vol_levels = volume_profile[volume_profile <= low_vol_threshold]
            
            if len(low_vol_levels) > 0:
                width = low_vol_levels.index.max() - low_vol_levels.index.min()
            else:
                width = 0.25  # minimum tick size
            
            # Distance from POC and Value Area
            distance_from_poc = abs(lvn_price - poc)
            in_value_area = value_area[1] <= lvn_price <= value_area[0] if value_area else False
            
            # Day's trading range
            day_range = day_df['High'].max() - day_df['Low'].min()
            
            enhanced_lvns.append({
                'lvn_timestamp': index,
                'lvn_price': lvn_price,
                'outcome': row['outcome'],
                'strength': strength,
                'width': width,
                'distance_from_poc': distance_from_poc,
                'in_value_area': in_value_area,
                'day_range': day_range,
                'volume': lvn_volume
            )}
            
        except Exception:
            continue
    
    return pd.DataFrame(enhanced_lvns)


class LVNAnalyzer:
    """
    Real-time LVN analyzer for streaming bar data
    
    Provides the interface expected by IndicatorEngine to calculate
    Low Volume Nodes in a streaming context.
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: Any):
        """
        Initialize LVN Analyzer
        
        Args:
            config: Configuration parameters
            event_bus: System event bus (unused but required for interface)
        """
        self.lookback_periods = config.get('lookback_periods', 20)
        self.strength_threshold = config.get('strength_threshold', 0.7)
        self.max_history_length = config.get('max_history_length', 100)
        self.interaction_lookback = config.get('interaction_lookback', 50)  # Bars to analyze for interactions
        self.test_threshold = config.get('test_threshold', 0.5)  # % distance to consider a test
        
        # History buffer for streaming calculation
        self.history_buffer: deque = deque(maxlen=self.max_history_length)
        
        # Cache for LVN levels
        self.current_lvns: List[Dict[str, Any]] = []
        self.last_calculation_time = None
        
        # Historical LVN tracking for strength analysis
        self.lvn_history: deque = deque(maxlen=200)  # Track historical LVN levels
        self.interaction_history: deque = deque(maxlen=500)  # Track price interactions with LVNs
    
    def calculate(self, bar_data: BarData) -> Dict[str, Any]:
        """
        Main calculate method that returns both lvn_nearest_price and lvn_nearest_strength
        
        Args:
            bar_data: Current bar data
            
        Returns:
            Dictionary with lvn_nearest_price and lvn_nearest_strength
        """
        result = self.calculate_30m(bar_data)
        return {
            'lvn_nearest_price': result['lvn_nearest_price'],
            'lvn_nearest_strength': result['lvn_nearest_strength']
        }
        
    def calculate_30m(self, bar_data: BarData) -> Dict[str, Any]:
        """
        Calculate LVN features for the current bar
        
        Args:
            bar_data: Current bar data
            
        Returns:
            Dictionary with LVN features
        """
        try:
            # Add current bar to history
            self.history_buffer.append({
                'timestamp': bar_data.timestamp,
                'open': bar_data.open,
                'high': bar_data.high,
                'low': bar_data.low,
                'close': bar_data.close,
                'volume': bar_data.volume
            })
            
            # Need minimum data for calculation
            if len(self.history_buffer) < self.lookback_periods:
                return {
                    'lvn_nearest_price': 0.0,
                    'lvn_nearest_strength': 0.0,
                    'distance_to_nearest_lvn': 0.0
                }
            
            # Convert to DataFrame for market profile calculation
            df = pd.DataFrame(list(self.history_buffer))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Rename columns to match expected format
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Calculate market profile
            try:
                mp = MarketProfile(df, tick_size=0.25)
                mp_slice = mp[df.index.min():df.index.max()]
                
                volume_profile = mp_slice.profile
                if volume_profile is None or volume_profile.empty:
                    return self._default_result()
                
                # Find LVNs (volume < threshold of POC volume)
                poc_volume = volume_profile.max()
                lvn_threshold = poc_volume * (1 - self.strength_threshold)
                
                lvn_levels = volume_profile[volume_profile < lvn_threshold]
                
                if lvn_levels.empty:
                    return self._default_result()
                
                # Update historical tracking
                current_price = bar_data.close
                self._update_lvn_history(lvn_levels, current_price)
                
                # Find nearest LVN to current price
                distances = abs(lvn_levels.index - current_price)
                nearest_idx = distances.argmin()
                nearest_lvn_price = lvn_levels.index[nearest_idx]
                nearest_lvn_volume = lvn_levels.iloc[nearest_idx]
                
                # Calculate sophisticated strength score (0-100)
                strength_score = self._calculate_lvn_strength_score(
                    nearest_lvn_price, nearest_lvn_volume, poc_volume, current_price
                )
                
                # Distance in points
                distance = abs(current_price - nearest_lvn_price)
                
                return {
                    'lvn_nearest_price': float(nearest_lvn_price),
                    'lvn_nearest_strength': float(strength_score),
                    'distance_to_nearest_lvn': float(distance)
                }
                
            except Exception:
                return self._default_result()
                
        except Exception:
            return self._default_result()
    
    def _calculate_lvn_strength_score(self, lvn_price: float, lvn_volume: float, 
                                     poc_volume: float, current_price: float) -> float:
        """
        Calculate sophisticated strength score (0-100) for an LVN level
        
        Args:
            lvn_price: The LVN price level
            lvn_volume: Volume at the LVN level
            poc_volume: Volume at Point of Control
            current_price: Current market price
            
        Returns:
            Strength score from 0 to 100
        """
        # Base strength from volume profile (0-40 points)
        volume_strength = (1 - (lvn_volume / poc_volume)) * 40 if poc_volume > 0 else 0
        
        # Historical interaction analysis (0-30 points)
        interaction_score = self._analyze_historical_interactions(lvn_price)
        
        # Recency score (0-20 points)
        recency_score = self._calculate_recency_score(lvn_price)
        
        # Distance penalty (0-10 points)
        max_distance = self.history_buffer[-1]['high'] - self.history_buffer[-1]['low'] if self.history_buffer else 1
        distance_ratio = min(abs(current_price - lvn_price) / max_distance, 1.0)
        distance_score = (1 - distance_ratio) * 10
        
        # Combine all scores
        total_score = volume_strength + interaction_score + recency_score + distance_score
        
        return min(max(total_score, 0), 100)  # Ensure 0-100 range
    
    def _analyze_historical_interactions(self, lvn_price: float) -> float:
        """
        Analyze historical price interactions with the LVN level
        
        Returns:
            Score from 0 to 30 based on:
            - Number of tests (0-15 points)
            - Magnitude of rejections (0-15 points)
        """
        if len(self.history_buffer) < self.interaction_lookback:
            return 0
        
        tests = 0
        rejections = []
        
        # Analyze recent price action
        for i in range(max(1, len(self.history_buffer) - self.interaction_lookback), len(self.history_buffer)):
            prev_bar = self.history_buffer[i-1]
            curr_bar = self.history_buffer[i]
            
            # Check if price tested the LVN level
            test_distance = abs(curr_bar['high'] - curr_bar['low']) * self.test_threshold
            
            # Test from below
            if prev_bar['close'] < lvn_price and curr_bar['high'] >= lvn_price - test_distance:
                tests += 1
                if curr_bar['close'] < lvn_price:  # Rejection
                    rejection_magnitude = (curr_bar['high'] - curr_bar['close']) / (curr_bar['high'] - curr_bar['low'] + 0.0001)
                    rejections.append(rejection_magnitude)
            
            # Test from above
            elif prev_bar['close'] > lvn_price and curr_bar['low'] <= lvn_price + test_distance:
                tests += 1
                if curr_bar['close'] > lvn_price:  # Rejection
                    rejection_magnitude = (curr_bar['close'] - curr_bar['low']) / (curr_bar['high'] - curr_bar['low'] + 0.0001)
                    rejections.append(rejection_magnitude)
        
        # Score based on number of tests (max 15 points)
        test_score = min(tests * 3, 15)
        
        # Score based on rejection magnitude (max 15 points)
        rejection_score = 0
        if rejections:
            avg_rejection = sum(rejections) / len(rejections)
            rejection_score = avg_rejection * 15
        
        return test_score + rejection_score
    
    def _calculate_recency_score(self, lvn_price: float) -> float:
        """
        Calculate recency score based on when the LVN was last tested
        
        Returns:
            Score from 0 to 20 (more recent = higher score)
        """
        if not self.history_buffer:
            return 0
        
        last_test_idx = -1
        test_distance = abs(self.history_buffer[-1]['high'] - self.history_buffer[-1]['low']) * self.test_threshold
        
        # Find the most recent test
        for i in range(len(self.history_buffer) - 1, -1, -1):
            bar = self.history_buffer[i]
            if (bar['low'] <= lvn_price + test_distance and 
                bar['high'] >= lvn_price - test_distance):
                last_test_idx = i
                break
        
        if last_test_idx == -1:
            return 0
        
        # Calculate recency (0-20 points, linear decay)
        bars_since_test = len(self.history_buffer) - 1 - last_test_idx
        recency_ratio = 1 - (bars_since_test / len(self.history_buffer))
        
        return recency_ratio * 20
    
    def _update_lvn_history(self, lvn_levels: pd.Series, current_price: float):
        """
        Update historical tracking of LVN levels and interactions
        """
        # Store current LVN levels
        self.lvn_history.append({
            'timestamp': self.history_buffer[-1]['timestamp'],
            'levels': lvn_levels.index.tolist(),
            'volumes': lvn_levels.values.tolist()
        )}
        
        # Track interaction if price is near any LVN
        for lvn_price in lvn_levels.index:
            test_distance = abs(self.history_buffer[-1]['high'] - self.history_buffer[-1]['low']) * self.test_threshold
            if abs(current_price - lvn_price) <= test_distance:
                self.interaction_history.append({
                    'timestamp': self.history_buffer[-1]['timestamp'],
                    'lvn_price': lvn_price,
                    'price': current_price,
                    'type': 'test'
                })
    
    def get_all_lvn_levels(self, current_price: float) -> List[Dict[str, Any]]:
        """
        Get all current LVN levels with their strength scores
        
        Args:
            current_price: Current market price for distance calculations
            
        Returns:
            List of LVN levels with prices, strengths, and distances
        """
        if len(self.history_buffer) < self.lookback_periods:
            return []
        
        try:
            # Convert to DataFrame for market profile calculation
            df = pd.DataFrame(list(self.history_buffer))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Calculate market profile
            mp = MarketProfile(df, tick_size=0.25)
            mp_slice = mp[df.index.min():df.index.max()]
            
            volume_profile = mp_slice.profile
            if volume_profile is None or volume_profile.empty:
                return []
            
            # Find all LVNs
            poc_volume = volume_profile.max()
            lvn_threshold = poc_volume * (1 - self.strength_threshold)
            lvn_levels = volume_profile[volume_profile < lvn_threshold]
            
            if lvn_levels.empty:
                return []
            
            # Calculate strength for each LVN level
            results = []
            for lvn_price, lvn_volume in lvn_levels.items():
                strength_score = self._calculate_lvn_strength_score(
                    lvn_price, lvn_volume, poc_volume, current_price
                )
                
                results.append({
                    'price': float(lvn_price),
                    'strength': float(strength_score),
                    'distance': float(abs(current_price - lvn_price)),
                    'volume': float(lvn_volume)
                )}
            
            # Sort by distance from current price
            results.sort(key=lambda x: x['distance'])
            return results
            
        except Exception:
            return []
    
    def get_lvn_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about LVN analysis
        
        Returns:
            Dictionary with LVN analysis statistics
        """
        return {
            'history_buffer_size': len(self.history_buffer),
            'lvn_history_size': len(self.lvn_history),
            'interaction_history_size': len(self.interaction_history),
            'lookback_periods': self.lookback_periods,
            'strength_threshold': self.strength_threshold,
            'interaction_lookback': self.interaction_lookback,
            'test_threshold': self.test_threshold
        }
    
    def _default_result(self) -> Dict[str, Any]:
        """Return default result when calculation fails"""
        return {
            'lvn_nearest_price': 0.0,
            'lvn_nearest_strength': 0.0,
            'distance_to_nearest_lvn': 0.0
        }