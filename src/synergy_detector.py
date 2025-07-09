"""
Synergy Detection Module for Multi-Indicator Strategy
Detects and validates NW-RQK → MLMI → FVG synergies with configurable parameters
"""

import numpy as np
import pandas as pd
from numba import njit, prange, float64, int64, boolean
from typing import Tuple, Dict, Any, Optional
import logging

class SynergyDetector:
    """Detects synergies between multiple indicators with state management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('SynergyDetector')
        self._validate_config()
        self.state_history = []
        
    def _validate_config(self):
        """Validate synergy configuration"""
        required_params = ['window', 'nwrqk_strength_threshold', 
                          'mlmi_confidence_threshold', 'state_decay_window']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required parameter: {param}")
        
        # Validate strength calculation weights
        strength_config = self.config.get('strength_calculation', {})
        total_weight = (strength_config.get('nwrqk_weight', 0.33) + 
                       strength_config.get('mlmi_weight', 0.33) + 
                       strength_config.get('fvg_weight', 0.34))
        
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Strength weights must sum to 1.0, got {total_weight}")
    
    def detect_synergies(self, nwrqk_signals: pd.DataFrame, 
                        mlmi_signals: pd.DataFrame, 
                        fvg_signals: pd.DataFrame) -> pd.DataFrame:
        """
        Detect NW-RQK → MLMI → FVG synergies
        
        Args:
            nwrqk_signals: DataFrame with nwrqk_bull, nwrqk_bear, nwrqk_strength
            mlmi_signals: DataFrame with mlmi_bull, mlmi_bear, mlmi_confidence
            fvg_signals: DataFrame with fvg_bull, fvg_bear, fvg_size
            
        Returns:
            DataFrame with synergy signals and strength
        """
        self.logger.info("Starting synergy detection...")
        
        # Validate inputs
        self._validate_signal_inputs(nwrqk_signals, mlmi_signals, fvg_signals)
        
        # Align signals to same index
        aligned_signals = self._align_signals(nwrqk_signals, mlmi_signals, fvg_signals)
        
        # Detect synergies
        synergy_bull, synergy_bear, synergy_strength, state_info = self._detect_nwrqk_mlmi_fvg_synergy(
            aligned_signals['nwrqk_bull'].values,
            aligned_signals['nwrqk_bear'].values,
            aligned_signals['nwrqk_strength'].values,
            aligned_signals['mlmi_bull'].values,
            aligned_signals['mlmi_bear'].values,
            aligned_signals['mlmi_confidence'].values,
            aligned_signals['fvg_bull'].values,
            aligned_signals['fvg_bear'].values,
            aligned_signals['fvg_size'].values,
            window=self.config['window'],
            nwrqk_threshold=self.config['nwrqk_strength_threshold'],
            mlmi_threshold=self.config['mlmi_confidence_threshold'],
            decay_window=self.config['state_decay_window'],
            strength_weights=(
                self.config['strength_calculation']['nwrqk_weight'],
                self.config['strength_calculation']['mlmi_weight'],
                self.config['strength_calculation']['fvg_weight']
            )
        )
        
        # Create results DataFrame
        results = pd.DataFrame(index=aligned_signals.index)
        results['synergy_bull'] = synergy_bull
        results['synergy_bear'] = synergy_bear
        results['synergy_strength'] = synergy_strength
        
        # Add state information for debugging
        results['nwrqk_active_bull'] = state_info[0]
        results['nwrqk_active_bear'] = state_info[1]
        results['mlmi_confirmed_bull'] = state_info[2]
        results['mlmi_confirmed_bear'] = state_info[3]
        
        # Calculate synergy statistics
        total_synergies = synergy_bull.sum() + synergy_bear.sum()
        self.logger.info(f"Detected {total_synergies} total synergies "
                        f"({synergy_bull.sum()} bull, {synergy_bear.sum()} bear)")
        
        # Store state history for analysis
        self.state_history.append({
            'timestamp': aligned_signals.index[-1],
            'total_synergies': total_synergies,
            'bull_synergies': synergy_bull.sum(),
            'bear_synergies': synergy_bear.sum(),
            'avg_strength': synergy_strength[synergy_strength > 0].mean() if (synergy_strength > 0).any() else 0
        })
        
        return results
    
    def _validate_signal_inputs(self, nwrqk_signals: pd.DataFrame, 
                               mlmi_signals: pd.DataFrame, 
                               fvg_signals: pd.DataFrame):
        """Validate signal dataframes have required columns"""
        required_nwrqk = ['nwrqk_bull', 'nwrqk_bear', 'nwrqk_strength']
        required_mlmi = ['mlmi_bull', 'mlmi_bear', 'mlmi_confidence']
        required_fvg = ['fvg_bull', 'fvg_bear', 'fvg_size']
        
        for col in required_nwrqk:
            if col not in nwrqk_signals.columns:
                raise ValueError(f"Missing required NW-RQK column: {col}")
        
        for col in required_mlmi:
            if col not in mlmi_signals.columns:
                raise ValueError(f"Missing required MLMI column: {col}")
        
        for col in required_fvg:
            if col not in fvg_signals.columns:
                raise ValueError(f"Missing required FVG column: {col}")
    
    def _align_signals(self, nwrqk_signals: pd.DataFrame, 
                      mlmi_signals: pd.DataFrame, 
                      fvg_signals: pd.DataFrame) -> pd.DataFrame:
        """Align all signal dataframes to common index"""
        # Find common index range
        start_idx = max(nwrqk_signals.index[0], mlmi_signals.index[0], fvg_signals.index[0])
        end_idx = min(nwrqk_signals.index[-1], mlmi_signals.index[-1], fvg_signals.index[-1])
        
        # Create aligned dataframe
        aligned = pd.DataFrame(index=pd.date_range(start_idx, end_idx, freq=nwrqk_signals.index.freq))
        
        # Merge all signals
        for col in ['nwrqk_bull', 'nwrqk_bear', 'nwrqk_strength']:
            aligned[col] = nwrqk_signals[col].reindex(aligned.index, method='ffill').fillna(False)
        
        for col in ['mlmi_bull', 'mlmi_bear', 'mlmi_confidence']:
            aligned[col] = mlmi_signals[col].reindex(aligned.index, method='ffill').fillna(False)
        
        for col in ['fvg_bull', 'fvg_bear', 'fvg_size']:
            aligned[col] = fvg_signals[col].reindex(aligned.index, method='ffill').fillna(0)
        
        return aligned
    
    @staticmethod
    @njit(parallel=True, fastmath=True, cache=True)
    def _detect_nwrqk_mlmi_fvg_synergy(
        nwrqk_bull: np.ndarray, nwrqk_bear: np.ndarray, nwrqk_strength: np.ndarray,
        mlmi_bull: np.ndarray, mlmi_bear: np.ndarray, mlmi_confidence: np.ndarray,
        fvg_bull: np.ndarray, fvg_bear: np.ndarray, fvg_size: np.ndarray,
        window: int, nwrqk_threshold: float, mlmi_threshold: float,
        decay_window: int, strength_weights: Tuple[float, float, float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Detect NW-RQK → MLMI → FVG synergy pattern with improved state management
        
        Returns:
            Tuple containing:
            - synergy_bull: Boolean array of bull synergies
            - synergy_bear: Boolean array of bear synergies
            - synergy_strength: Float array of synergy strengths
            - state_info: Tuple of state arrays for debugging
        """
        n = len(nwrqk_bull)
        synergy_bull = np.zeros(n, dtype=np.bool_)
        synergy_bear = np.zeros(n, dtype=np.bool_)
        synergy_strength = np.zeros(n)
        
        # State tracking arrays
        nwrqk_active_bull = np.zeros(n, dtype=np.bool_)
        nwrqk_active_bear = np.zeros(n, dtype=np.bool_)
        mlmi_confirmed_bull = np.zeros(n, dtype=np.bool_)
        mlmi_confirmed_bear = np.zeros(n, dtype=np.bool_)
        
        # State activation timestamps
        nwrqk_bull_time = np.full(n, -1, dtype=np.int64)
        nwrqk_bear_time = np.full(n, -1, dtype=np.int64)
        mlmi_bull_time = np.full(n, -1, dtype=np.int64)
        mlmi_bear_time = np.full(n, -1, dtype=np.int64)
        
        for i in prange(1, n):
            # Carry forward states
            if i > 0:
                nwrqk_active_bull[i] = nwrqk_active_bull[i-1]
                nwrqk_active_bear[i] = nwrqk_active_bear[i-1]
                mlmi_confirmed_bull[i] = mlmi_confirmed_bull[i-1]
                mlmi_confirmed_bear[i] = mlmi_confirmed_bear[i-1]
                nwrqk_bull_time[i] = nwrqk_bull_time[i-1]
                nwrqk_bear_time[i] = nwrqk_bear_time[i-1]
                mlmi_bull_time[i] = mlmi_bull_time[i-1]
                mlmi_bear_time[i] = mlmi_bear_time[i-1]
            
            # Step 1: NW-RQK signal activation
            if nwrqk_bull[i] and nwrqk_strength[i] > nwrqk_threshold:
                nwrqk_active_bull[i] = True
                nwrqk_active_bear[i] = False
                mlmi_confirmed_bear[i] = False
                nwrqk_bull_time[i] = i
                nwrqk_bear_time[i] = -1
                mlmi_bear_time[i] = -1
            elif nwrqk_bear[i] and nwrqk_strength[i] > nwrqk_threshold:
                nwrqk_active_bear[i] = True
                nwrqk_active_bull[i] = False
                mlmi_confirmed_bull[i] = False
                nwrqk_bear_time[i] = i
                nwrqk_bull_time[i] = -1
                mlmi_bull_time[i] = -1
            
            # Step 2: MLMI confirmation
            if nwrqk_active_bull[i] and mlmi_bull[i] and mlmi_confidence[i] > mlmi_threshold:
                mlmi_confirmed_bull[i] = True
                mlmi_bull_time[i] = i
            elif nwrqk_active_bear[i] and mlmi_bear[i] and mlmi_confidence[i] > mlmi_threshold:
                mlmi_confirmed_bear[i] = True
                mlmi_bear_time[i] = i
            
            # Step 3: FVG validation for entry
            if mlmi_confirmed_bull[i] and fvg_bull[i]:
                synergy_bull[i] = True
                
                # Calculate synergy strength with component weights
                strength_components = np.zeros(3)
                
                # NW-RQK component
                if nwrqk_bull_time[i] >= 0:
                    time_since_nwrqk = i - nwrqk_bull_time[i]
                    decay_factor = max(0, 1 - time_since_nwrqk / float(decay_window))
                    for j in range(max(0, nwrqk_bull_time[i]), min(i + 1, nwrqk_bull_time[i] + window)):
                        if nwrqk_bull[j]:
                            strength_components[0] = max(strength_components[0], nwrqk_strength[j] * decay_factor)
                
                # MLMI component
                strength_components[1] = mlmi_confidence[i]
                
                # FVG component
                strength_components[2] = min(abs(fvg_size[i]) * 100, 1.0)
                
                # Weighted combination
                synergy_strength[i] = (
                    strength_components[0] * strength_weights[0] +
                    strength_components[1] * strength_weights[1] +
                    strength_components[2] * strength_weights[2]
                )
                
                # Reset states after signal
                nwrqk_active_bull[i] = False
                mlmi_confirmed_bull[i] = False
                nwrqk_bull_time[i] = -1
                mlmi_bull_time[i] = -1
                
            elif mlmi_confirmed_bear[i] and fvg_bear[i]:
                synergy_bear[i] = True
                
                # Calculate synergy strength
                strength_components = np.zeros(3)
                
                # NW-RQK component
                if nwrqk_bear_time[i] >= 0:
                    time_since_nwrqk = i - nwrqk_bear_time[i]
                    decay_factor = max(0, 1 - time_since_nwrqk / float(decay_window))
                    for j in range(max(0, nwrqk_bear_time[i]), min(i + 1, nwrqk_bear_time[i] + window)):
                        if nwrqk_bear[j]:
                            strength_components[0] = max(strength_components[0], nwrqk_strength[j] * decay_factor)
                
                # MLMI component
                strength_components[1] = mlmi_confidence[i]
                
                # FVG component
                strength_components[2] = min(abs(fvg_size[i]) * 100, 1.0)
                
                # Weighted combination
                synergy_strength[i] = (
                    strength_components[0] * strength_weights[0] +
                    strength_components[1] * strength_weights[1] +
                    strength_components[2] * strength_weights[2]
                )
                
                # Reset states after signal
                nwrqk_active_bear[i] = False
                mlmi_confirmed_bear[i] = False
                nwrqk_bear_time[i] = -1
                mlmi_bear_time[i] = -1
            
            # State decay - reset if signals are too old
            if nwrqk_bull_time[i] >= 0 and i - nwrqk_bull_time[i] > decay_window:
                nwrqk_active_bull[i] = False
                mlmi_confirmed_bull[i] = False
                nwrqk_bull_time[i] = -1
                mlmi_bull_time[i] = -1
            
            if nwrqk_bear_time[i] >= 0 and i - nwrqk_bear_time[i] > decay_window:
                nwrqk_active_bear[i] = False
                mlmi_confirmed_bear[i] = False
                nwrqk_bear_time[i] = -1
                mlmi_bear_time[i] = -1
        
        return synergy_bull, synergy_bear, synergy_strength, (
            nwrqk_active_bull, nwrqk_active_bear, 
            mlmi_confirmed_bull, mlmi_confirmed_bear
        )
    
    def get_synergy_stats(self) -> Dict[str, Any]:
        """Get statistics about detected synergies"""
        if not self.state_history:
            return {}
        
        df = pd.DataFrame(self.state_history)
        
        return {
            'total_detections': df['total_synergies'].sum(),
            'bull_detections': df['bull_synergies'].sum(),
            'bear_detections': df['bear_synergies'].sum(),
            'avg_strength': df['avg_strength'].mean(),
            'detections_per_period': df['total_synergies'].mean(),
            'bull_bear_ratio': df['bull_synergies'].sum() / max(1, df['bear_synergies'].sum())
        }