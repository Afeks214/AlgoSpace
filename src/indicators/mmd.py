"""
MMD Feature Extractor for Regime Detection Engine
Extracts ONLY the core MMD calculation functions from notebook
"""

import numpy as np
import pandas as pd
import numba as nb
from numba import jit, prange
from typing import Dict, Any
from src.indicators.base import BaseIndicator
from src.core.events import EventBus, BarData


@nb.jit(nopython=True, parallel=True)
def _compute_dists_sq_numba(data: np.ndarray) -> np.ndarray:
    """Optimized pairwise squared Euclidean distances for sigma estimation"""
    n_samples = data.shape[0]
    dists_sq = np.zeros(n_samples * (n_samples - 1) // 2, dtype=np.float64)
    k = 0
    for i in prange(n_samples):
        for j in range(i + 1, n_samples):
            diff = data[i] - data[j]
            dists_sq[k] = np.sum(diff * diff)
            k += 1
    return dists_sq


@nb.jit(nopython=True)
def gaussian_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    """Gaussian kernel function"""
    diff = x - y
    return np.exp(-np.sum(diff * diff) / (2.0 * sigma * sigma))


@nb.jit(nopython=True, parallel=True)
def compute_mmd(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    """Compute MMD between two samples"""
    n_x, n_y = X.shape[0], Y.shape[0]
    
    # K(X,X) term
    K_XX_sum = 0.0
    if n_x > 1:
        for i in prange(n_x):
            for j in range(n_x):
                if i != j:
                    K_XX_sum += gaussian_kernel(X[i], X[j], sigma)
        K_XX = K_XX_sum / (n_x * (n_x - 1))
    else:
        K_XX = 0.0
    
    # K(Y,Y) term  
    K_YY_sum = 0.0
    if n_y > 1:
        for i in prange(n_y):
            for j in range(n_y):
                if i != j:
                    K_YY_sum += gaussian_kernel(Y[i], Y[j], sigma)
        K_YY = K_YY_sum / (n_y * (n_y - 1))
    else:
        K_YY = 0.0
    
    # K(X,Y) term
    K_XY_sum = 0.0
    if n_x > 0 and n_y > 0:
        for i in prange(n_x):
            for j in range(n_y):
                K_XY_sum += gaussian_kernel(X[i], Y[j], sigma)
        K_XY = K_XY_sum / (n_x * n_y)
    else:
        K_XY = 0.0
    
    return max(0.0, np.sqrt(K_XX + K_YY - 2.0 * K_XY))


class MMDFeatureExtractor(BaseIndicator):
    """MMD feature extractor for RDE - extracts EXACT 13-dimensional feature vector from notebook"""
    
    def __init__(self, config: Dict[str, Any], event_bus: EventBus):
        super().__init__(config, event_bus)
        self.reference_window = config.get('reference_window', 500)
        self.test_window = config.get('test_window', 100)
        self.reference_data = None
        self.sigma = 1.0
        self.mmd_scores = []
    
    
    def calculate_30m(self, bar: BarData) -> Dict[str, Any]:
        self.update_30m_history(bar)
        if len(self.history_30m) < self.reference_window + self.test_window:
            return {'mmd_features': np.zeros(13)}
        
        # Convert bars to feature DataFrame
        df = pd.DataFrame([{
            'close': b.close, 'high': b.high, 'low': b.low, 'volume': b.volume
        } for b in self.history_30m])
        
        # Calculate features exactly as in notebook
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['range'] = (df['high'] - df['low']) / df['close'].shift(1)
        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252 * 48)
        df = df.dropna()
        
        # Extract MMD features
        data = df[['returns', 'log_returns', 'range', 'volatility']].values
        
        # Initialize reference if needed
        if self.reference_data is None:
            ref_data = data[:self.reference_window]
            ref_mean, ref_std = np.mean(ref_data, axis=0), np.std(ref_data, axis=0)
            ref_std[ref_std < 1e-8] = 1e-8
            self.reference_data = (ref_data - ref_mean) / ref_std
            self.ref_mean, self.ref_std = ref_mean, ref_std
            
            # Estimate sigma
            n_samples = min(200, self.reference_data.shape[0])
            if n_samples > 1:
                indices = np.random.choice(self.reference_data.shape[0], n_samples, replace=False)
                sampled = self.reference_data[indices]
                dists_sq = _compute_dists_sq_numba(sampled)
                if dists_sq.size > 0:
                    self.sigma = np.sqrt(np.median(dists_sq))
        
        # Calculate current MMD
        test_data = data[-self.test_window:]
        test_normalized = (test_data - self.ref_mean) / self.ref_std
        mmd_score = compute_mmd(self.reference_data, test_normalized, self.sigma)
        self.mmd_scores.append(mmd_score)
        if len(self.mmd_scores) > 100:
            self.mmd_scores.pop(0)
        
        # Create 13-dimensional feature vector exactly as in notebook
        recent = df.tail(1)  # Get current values
        
        # Calculate momentum indicators as in notebook
        df['momentum_20'] = df['close'].pct_change(periods=20)
        df['momentum_50'] = df['close'].pct_change(periods=50)
        
        # Volume ratio calculation
        df['volume_ma'] = df['volume'].rolling(window=20, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-9)
        
        # EXACT 13-feature vector as in notebook
        features = np.array([
            recent['close'].iloc[-1] if len(recent) > 0 else 0.0,    # open (using close as proxy)
            recent['high'].iloc[-1] if len(recent) > 0 else 0.0,     # high
            recent['low'].iloc[-1] if len(recent) > 0 else 0.0,      # low
            recent['close'].iloc[-1] if len(recent) > 0 else 0.0,    # close
            recent['volume'].iloc[-1] if len(recent) > 0 else 0.0,   # volume
            recent['returns'].iloc[-1] if len(recent) > 0 else 0.0,  # returns
            recent['log_returns'].iloc[-1] if len(recent) > 0 else 0.0,  # log_returns
            recent['range'].iloc[-1] if len(recent) > 0 else 0.0,    # range
            recent['volatility'].iloc[-1] if len(recent) > 0 else 0.0,  # volatility
            df['momentum_20'].iloc[-1] if len(df) > 20 else 0.0,     # momentum_20
            df['momentum_50'].iloc[-1] if len(df) > 50 else 0.0,     # momentum_50
            df['volume_ratio'].iloc[-1] if len(df) > 0 else 1.0,     # volume_ratio
            mmd_score                                                 # mmd_score
        ])
        
        return {'mmd_features': np.nan_to_num(features)}
    
    
    def get_current_values(self) -> Dict[str, Any]:
        return {'mmd_score': self.mmd_scores[-1] if self.mmd_scores else 0.0}
    
    def reset(self) -> None:
        self.reference_data = None
        self.mmd_scores = []
        self.history_30m = []