"""
Low Volume Node (LVN) Core Calculations

This module provides the essential LVN identification and characteristics calculation
based on market profile analysis.

Extracted from: LVN_Implementation.ipynb (Phase 1 core functions only)
"""

import pandas as pd
import numpy as np
from market_profile import MarketProfile
from numba import jit
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='market_profile')


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
                })

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
            })
            
        except Exception:
            continue
    
    return pd.DataFrame(enhanced_lvns)