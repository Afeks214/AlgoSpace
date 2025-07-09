"""
Advanced Synergy Detection Module for AlgoSpace Strategy
Implements pattern detection with scoring and validation
"""

import numpy as np
import pandas as pd
from numba import njit, prange
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SynergyType(Enum):
    """Enumeration of synergy types"""
    TYPE_1 = "MLMI → FVG → NWRQK"
    TYPE_2 = "MLMI → NWRQK → FVG"
    TYPE_3 = "NWRQK → MLMI → FVG"
    TYPE_4 = "NWRQK → FVG → MLMI"


@dataclass
class SynergySignal:
    """Data class for synergy signal information"""
    timestamp: pd.Timestamp
    synergy_type: SynergyType
    direction: str  # 'long' or 'short'
    strength: float
    components: Dict[str, pd.Timestamp]
    confidence: float


@njit(fastmath=True, cache=True)
def calculate_synergy_strength(time_gaps: np.ndarray, indicator_strengths: np.ndarray,
                              max_window: int) -> float:
    """
    Calculate synergy strength based on timing and indicator strengths
    
    Args:
        time_gaps: Time gaps between indicators (in bars)
        indicator_strengths: Strength values for each indicator
        max_window: Maximum allowed window
        
    Returns:
        Synergy strength score (0-1)
    """
    # Time coherence score (closer timing = higher score)
    total_time = np.sum(time_gaps)
    time_score = 1.0 - (total_time / max_window)
    time_score = max(0.0, min(1.0, time_score))
    
    # Indicator strength score
    strength_score = np.mean(indicator_strengths)
    
    # Combined score with weights
    synergy_strength = 0.6 * time_score + 0.4 * strength_score
    
    return synergy_strength


@njit(parallel=True, fastmath=True, cache=True)
def detect_all_synergies_advanced(
    mlmi_bull: np.ndarray, mlmi_bear: np.ndarray,
    mlmi_strength: np.ndarray,
    fvg_bull: np.ndarray, fvg_bear: np.ndarray,
    fvg_strength: np.ndarray,
    nwrqk_bull: np.ndarray, nwrqk_bear: np.ndarray,
    nwrqk_strength: np.ndarray,
    window: int = 30,
    min_strength: float = 0.3
) -> Tuple:
    """
    Advanced synergy detection with strength scoring
    
    Returns:
        Tuple of arrays for each synergy type (long/short signals and strengths)
    """
    n = len(mlmi_bull)
    half_window = window // 2
    
    # Initialize arrays for all synergy types
    # Type 1: MLMI → FVG → NWRQK
    syn1_long = np.zeros(n, dtype=np.bool_)
    syn1_short = np.zeros(n, dtype=np.bool_)
    syn1_long_strength = np.zeros(n, dtype=np.float64)
    syn1_short_strength = np.zeros(n, dtype=np.float64)
    
    # Type 2: MLMI → NWRQK → FVG
    syn2_long = np.zeros(n, dtype=np.bool_)
    syn2_short = np.zeros(n, dtype=np.bool_)
    syn2_long_strength = np.zeros(n, dtype=np.float64)
    syn2_short_strength = np.zeros(n, dtype=np.float64)
    
    # Type 3: NWRQK → MLMI → FVG
    syn3_long = np.zeros(n, dtype=np.bool_)
    syn3_short = np.zeros(n, dtype=np.bool_)
    syn3_long_strength = np.zeros(n, dtype=np.float64)
    syn3_short_strength = np.zeros(n, dtype=np.float64)
    
    # Type 4: NWRQK → FVG → MLMI
    syn4_long = np.zeros(n, dtype=np.bool_)
    syn4_short = np.zeros(n, dtype=np.bool_)
    syn4_long_strength = np.zeros(n, dtype=np.float64)
    syn4_short_strength = np.zeros(n, dtype=np.float64)
    
    for i in prange(2, n - window):
        # Type 1: MLMI → FVG → NWRQK (Long)
        if mlmi_bull[i] and not mlmi_bull[i-1]:
            for j in range(i+1, min(i+half_window, n)):
                if fvg_bull[j]:
                    for k in range(j+1, min(i+window, n)):
                        if nwrqk_bull[k]:
                            # Calculate strength
                            time_gaps = np.array([j-i, k-j], dtype=np.float64)
                            strengths = np.array([mlmi_strength[i], fvg_strength[j], 
                                                nwrqk_strength[k]], dtype=np.float64)
                            strength = calculate_synergy_strength(time_gaps, strengths, window)
                            
                            if strength >= min_strength:
                                syn1_long[k] = True
                                syn1_long_strength[k] = strength
                            break
                    break
        
        # Type 1: MLMI → FVG → NWRQK (Short)
        if mlmi_bear[i] and not mlmi_bear[i-1]:
            for j in range(i+1, min(i+half_window, n)):
                if fvg_bear[j]:
                    for k in range(j+1, min(i+window, n)):
                        if nwrqk_bear[k]:
                            time_gaps = np.array([j-i, k-j], dtype=np.float64)
                            strengths = np.array([mlmi_strength[i], fvg_strength[j], 
                                                nwrqk_strength[k]], dtype=np.float64)
                            strength = calculate_synergy_strength(time_gaps, strengths, window)
                            
                            if strength >= min_strength:
                                syn1_short[k] = True
                                syn1_short_strength[k] = strength
                            break
                    break
        
        # Type 2: MLMI → NWRQK → FVG (Long)
        if mlmi_bull[i] and not mlmi_bull[i-1]:
            for j in range(i+1, min(i+half_window, n)):
                if nwrqk_bull[j]:
                    for k in range(j+1, min(i+window, n)):
                        if fvg_bull[k]:
                            time_gaps = np.array([j-i, k-j], dtype=np.float64)
                            strengths = np.array([mlmi_strength[i], nwrqk_strength[j], 
                                                fvg_strength[k]], dtype=np.float64)
                            strength = calculate_synergy_strength(time_gaps, strengths, window)
                            
                            if strength >= min_strength:
                                syn2_long[k] = True
                                syn2_long_strength[k] = strength
                            break
                    break
        
        # Type 2: MLMI → NWRQK → FVG (Short)
        if mlmi_bear[i] and not mlmi_bear[i-1]:
            for j in range(i+1, min(i+half_window, n)):
                if nwrqk_bear[j]:
                    for k in range(j+1, min(i+window, n)):
                        if fvg_bear[k]:
                            time_gaps = np.array([j-i, k-j], dtype=np.float64)
                            strengths = np.array([mlmi_strength[i], nwrqk_strength[j], 
                                                fvg_strength[k]], dtype=np.float64)
                            strength = calculate_synergy_strength(time_gaps, strengths, window)
                            
                            if strength >= min_strength:
                                syn2_short[k] = True
                                syn2_short_strength[k] = strength
                            break
                    break
        
        # Type 3: NWRQK → MLMI → FVG (Long)
        if nwrqk_bull[i]:
            for j in range(i+1, min(i+half_window, n)):
                if mlmi_bull[j]:
                    for k in range(j+1, min(i+window, n)):
                        if fvg_bull[k]:
                            time_gaps = np.array([j-i, k-j], dtype=np.float64)
                            strengths = np.array([nwrqk_strength[i], mlmi_strength[j], 
                                                fvg_strength[k]], dtype=np.float64)
                            strength = calculate_synergy_strength(time_gaps, strengths, window)
                            
                            if strength >= min_strength:
                                syn3_long[k] = True
                                syn3_long_strength[k] = strength
                            break
                    break
        
        # Type 3: NWRQK → MLMI → FVG (Short)
        if nwrqk_bear[i]:
            for j in range(i+1, min(i+half_window, n)):
                if mlmi_bear[j]:
                    for k in range(j+1, min(i+window, n)):
                        if fvg_bear[k]:
                            time_gaps = np.array([j-i, k-j], dtype=np.float64)
                            strengths = np.array([nwrqk_strength[i], mlmi_strength[j], 
                                                fvg_strength[k]], dtype=np.float64)
                            strength = calculate_synergy_strength(time_gaps, strengths, window)
                            
                            if strength >= min_strength:
                                syn3_short[k] = True
                                syn3_short_strength[k] = strength
                            break
                    break
        
        # Type 4: NWRQK → FVG → MLMI (Long)
        if nwrqk_bull[i]:
            for j in range(i+1, min(i+half_window, n)):
                if fvg_bull[j]:
                    for k in range(j+1, min(i+window, n)):
                        if mlmi_bull[k]:
                            time_gaps = np.array([j-i, k-j], dtype=np.float64)
                            strengths = np.array([nwrqk_strength[i], fvg_strength[j], 
                                                mlmi_strength[k]], dtype=np.float64)
                            strength = calculate_synergy_strength(time_gaps, strengths, window)
                            
                            if strength >= min_strength:
                                syn4_long[k] = True
                                syn4_long_strength[k] = strength
                            break
                    break
        
        # Type 4: NWRQK → FVG → MLMI (Short)
        if nwrqk_bear[i]:
            for j in range(i+1, min(i+half_window, n)):
                if fvg_bear[j]:
                    for k in range(j+1, min(i+window, n)):
                        if mlmi_bear[k]:
                            time_gaps = np.array([j-i, k-j], dtype=np.float64)
                            strengths = np.array([nwrqk_strength[i], fvg_strength[j], 
                                                mlmi_strength[k]], dtype=np.float64)
                            strength = calculate_synergy_strength(time_gaps, strengths, window)
                            
                            if strength >= min_strength:
                                syn4_short[k] = True
                                syn4_short_strength[k] = strength
                            break
                    break
    
    return (syn1_long, syn1_short, syn1_long_strength, syn1_short_strength,
            syn2_long, syn2_short, syn2_long_strength, syn2_short_strength,
            syn3_long, syn3_short, syn3_long_strength, syn3_short_strength,
            syn4_long, syn4_short, syn4_long_strength, syn4_short_strength)


class SynergyDetector:
    """Advanced synergy detection with pattern analysis"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize synergy detector with configuration"""
        self.config = config or {}
        self.detection_window = self.config.get('detection_window', 30)
        self.min_strength = self.config.get('min_strength', 0.3)
        self.synergy_history: List[SynergySignal] = []
        
    def calculate_indicator_strengths(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate strength scores for each indicator"""
        strengths = {}
        
        # MLMI strength (based on absolute value and confidence)
        if 'mlmi' in df.columns and 'mlmi_confidence' in df.columns:
            strengths['mlmi'] = np.abs(df['mlmi'].values) * df['mlmi_confidence'].values
        elif 'mlmi' in df.columns:
            strengths['mlmi'] = np.abs(df['mlmi'].values)
        else:
            strengths['mlmi'] = np.ones(len(df))
        
        # FVG strength (based on gap size)
        if 'High' in df.columns and 'Low' in df.columns:
            gap_size = (df['High'] - df['Low']).rolling(3).mean()
            strengths['fvg'] = gap_size / gap_size.rolling(20).mean()
            strengths['fvg'] = strengths['fvg'].fillna(1).values
        else:
            strengths['fvg'] = np.ones(len(df))
        
        # NWRQK strength (based on trend strength)
        if 'nwrqk_trend_strength' in df.columns:
            strengths['nwrqk'] = df['nwrqk_trend_strength'].values
        else:
            strengths['nwrqk'] = np.ones(len(df))
        
        # Normalize strengths to 0-1 range
        for key in strengths:
            arr = strengths[key]
            if np.std(arr) > 0:
                arr = (arr - np.mean(arr)) / np.std(arr)
                arr = 1 / (1 + np.exp(-arr))  # Sigmoid normalization
                strengths[key] = arr
        
        return strengths
    
    def detect_synergies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect all synergy patterns in the data
        
        Args:
            df: DataFrame with indicator columns
            
        Returns:
            DataFrame with synergy signals added
        """
        logger.info("Detecting synergy patterns")
        
        # Extract indicator arrays
        mlmi_bull = df.get('mlmi_bull_30m', df.get('mlmi_bull', pd.Series(False))).fillna(False).values
        mlmi_bear = df.get('mlmi_bear_30m', df.get('mlmi_bear', pd.Series(False))).fillna(False).values
        fvg_bull = df.get('FVG_Bull_Active', pd.Series(False)).values
        fvg_bear = df.get('FVG_Bear_Active', pd.Series(False)).values
        nwrqk_bull = df.get('nwrqk_bull_30m', df.get('nwrqk_bull', pd.Series(False))).fillna(False).values
        nwrqk_bear = df.get('nwrqk_bear_30m', df.get('nwrqk_bear', pd.Series(False))).fillna(False).values
        
        # Calculate indicator strengths
        strengths = self.calculate_indicator_strengths(df)
        
        # Detect synergies with strength scoring
        results = detect_all_synergies_advanced(
            mlmi_bull, mlmi_bear, strengths['mlmi'],
            fvg_bull, fvg_bear, strengths['fvg'],
            nwrqk_bull, nwrqk_bear, strengths['nwrqk'],
            self.detection_window, self.min_strength
        )
        
        # Unpack results
        synergy_data = {
            'syn1_long': results[0], 'syn1_short': results[1],
            'syn1_long_strength': results[2], 'syn1_short_strength': results[3],
            'syn2_long': results[4], 'syn2_short': results[5],
            'syn2_long_strength': results[6], 'syn2_short_strength': results[7],
            'syn3_long': results[8], 'syn3_short': results[9],
            'syn3_long_strength': results[10], 'syn3_short_strength': results[11],
            'syn4_long': results[12], 'syn4_short': results[13],
            'syn4_long_strength': results[14], 'syn4_short_strength': results[15]
        }
        
        # Add to dataframe
        for key, value in synergy_data.items():
            df[key] = value
        
        # Calculate aggregate synergy scores
        df['synergy_long_count'] = (
            df['syn1_long'].astype(int) + df['syn2_long'].astype(int) +
            df['syn3_long'].astype(int) + df['syn4_long'].astype(int)
        )
        
        df['synergy_short_count'] = (
            df['syn1_short'].astype(int) + df['syn2_short'].astype(int) +
            df['syn3_short'].astype(int) + df['syn4_short'].astype(int)
        )
        
        df['synergy_long_strength'] = (
            df['syn1_long_strength'] + df['syn2_long_strength'] +
            df['syn3_long_strength'] + df['syn4_long_strength']
        ) / 4
        
        df['synergy_short_strength'] = (
            df['syn1_short_strength'] + df['syn2_short_strength'] +
            df['syn3_short_strength'] + df['syn4_short_strength']
        ) / 4
        
        # Log summary
        total_long = sum([synergy_data[f'syn{i}_long'].sum() for i in range(1, 5)])
        total_short = sum([synergy_data[f'syn{i}_short'].sum() for i in range(1, 5)])
        logger.info(f"Synergies detected: Long={total_long:,}, Short={total_short:,}")
        
        return df
    
    def get_synergy_signals(self, df: pd.DataFrame, synergy_type: int) -> List[SynergySignal]:
        """
        Extract detailed synergy signals for a specific type
        
        Args:
            df: DataFrame with synergy detection results
            synergy_type: Synergy type number (1-4)
            
        Returns:
            List of SynergySignal objects
        """
        signals = []
        
        long_col = f'syn{synergy_type}_long'
        short_col = f'syn{synergy_type}_short'
        long_strength_col = f'syn{synergy_type}_long_strength'
        short_strength_col = f'syn{synergy_type}_short_strength'
        
        # Extract long signals
        long_mask = df[long_col]
        for idx in df[long_mask].index:
            signal = SynergySignal(
                timestamp=idx,
                synergy_type=SynergyType(f"TYPE_{synergy_type}"),
                direction='long',
                strength=df.loc[idx, long_strength_col],
                components={},  # Would need additional tracking for component timestamps
                confidence=df.loc[idx, long_strength_col]
            )
            signals.append(signal)
        
        # Extract short signals
        short_mask = df[short_col]
        for idx in df[short_mask].index:
            signal = SynergySignal(
                timestamp=idx,
                synergy_type=SynergyType(f"TYPE_{synergy_type}"),
                direction='short',
                strength=df.loc[idx, short_strength_col],
                components={},
                confidence=df.loc[idx, short_strength_col]
            )
            signals.append(signal)
        
        return signals
    
    def analyze_synergy_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze performance of different synergy types"""
        performance_data = []
        
        for synergy_type in range(1, 5):
            long_col = f'syn{synergy_type}_long'
            short_col = f'syn{synergy_type}_short'
            
            # Count signals
            long_signals = df[long_col].sum()
            short_signals = df[short_col].sum()
            
            # Calculate average strength
            long_strength_col = f'syn{synergy_type}_long_strength'
            short_strength_col = f'syn{synergy_type}_short_strength'
            
            avg_long_strength = df[df[long_col]][long_strength_col].mean() if long_signals > 0 else 0
            avg_short_strength = df[df[short_col]][short_strength_col].mean() if short_signals > 0 else 0
            
            performance_data.append({
                'Synergy Type': f"Type {synergy_type}",
                'Long Signals': long_signals,
                'Short Signals': short_signals,
                'Total Signals': long_signals + short_signals,
                'Avg Long Strength': avg_long_strength,
                'Avg Short Strength': avg_short_strength,
                'Avg Overall Strength': (avg_long_strength + avg_short_strength) / 2
            })
        
        return pd.DataFrame(performance_data)