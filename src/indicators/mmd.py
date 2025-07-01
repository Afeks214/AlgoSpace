"""
Enhanced MMD Feature Extractor for Regime Detection Engine

This module implements a comprehensive Maximum Mean Discrepancy (MMD) feature extractor
that computes a full 23-dimensional feature vector for regime detection:

Features:
- 7 MMD scores against pre-defined reference distributions:
  * Normal Market (low volatility, moderate returns)
  * Bull Market (positive skew, higher returns) 
  * Bear Market (negative skew, higher volatility)
  * High Volatility Regime
  * Low Volatility Regime
  * Trending Market (momentum regime)
  * Mean Reverting Market (oscillating regime)

- 16 additional statistical and technical features:
  * Current returns, log returns, price range, volatility
  * Short/medium/long-term momentum indicators
  * Volume analysis (ratio, spikes)
  * Statistical moments (skewness, kurtosis)
  * Technical indicators (RSI, volatility ratio)
  * Price acceleration and regime similarity metrics

The extractor provides real-time regime classification and confidence scoring
to support advanced algorithmic trading strategies.
"""

import numpy as np
import pandas as pd
import numba as nb
from numba import jit, prange
from typing import Dict, Any, List, Tuple
from src.indicators.base import BaseIndicator
from src.core.events import EventBus, BarData
from sklearn.mixture import GaussianMixture
from scipy import stats


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
    """Enhanced MMD feature extractor for RDE - computes full MMD Feature Vector with 7 reference distributions"""
    
    def __init__(self, config: Dict[str, Any], event_bus: EventBus):
        super().__init__(config, event_bus)
        self.reference_window = config.get('reference_window', 500)
        self.test_window = config.get('test_window', 100)
        self.reference_distributions = {}
        self.sigma = 1.0
        self.mmd_scores = []
        self.feature_history = []
        self._initialize_reference_distributions()
    
    def _initialize_reference_distributions(self):
        """Initialize 7 pre-defined reference distributions for different market regimes"""
        np.random.seed(42)  # For reproducible reference distributions
        
        # Distribution 1: Normal Market (Low volatility, moderate returns)
        normal_returns = np.random.normal(0.0001, 0.01, (1000, 4))
        
        # Distribution 2: Bull Market (Positive skew, higher returns)
        bull_returns = np.column_stack([
            np.random.beta(2, 5, 1000) * 0.05 - 0.01,  # positive skewed returns
            np.random.lognormal(-5, 0.5, 1000),         # log returns
            np.random.gamma(2, 0.005, 1000),            # range
            np.random.gamma(1.5, 0.01, 1000)            # volatility
        ])
        
        # Distribution 3: Bear Market (Negative skew, higher volatility)
        bear_returns = np.column_stack([
            -np.random.beta(2, 5, 1000) * 0.08 + 0.01,  # negative skewed returns
            -np.random.lognormal(-4.5, 0.6, 1000),      # log returns  
            np.random.gamma(3, 0.008, 1000),             # range
            np.random.gamma(2.5, 0.015, 1000)           # volatility
        ])
        
        # Distribution 4: High Volatility Regime
        high_vol_returns = np.column_stack([
            np.random.normal(0, 0.03, 1000),             # high std returns
            np.random.normal(0, 0.025, 1000),            # log returns
            np.random.gamma(4, 0.01, 1000),              # range
            np.random.gamma(3, 0.02, 1000)               # volatility
        ])
        
        # Distribution 5: Low Volatility Regime
        low_vol_returns = np.column_stack([
            np.random.normal(0, 0.005, 1000),            # low std returns
            np.random.normal(0, 0.004, 1000),            # log returns
            np.random.gamma(1, 0.002, 1000),             # range
            np.random.gamma(0.5, 0.005, 1000)            # volatility
        ])
        
        # Distribution 6: Trending Market (Momentum regime)
        trend_returns = np.column_stack([
            np.cumsum(np.random.normal(0.0005, 0.015, 1000)) / 100,  # trending returns
            np.cumsum(np.random.normal(0.0003, 0.012, 1000)) / 100,  # log returns
            np.random.gamma(2, 0.006, 1000),                         # range
            np.random.gamma(1.8, 0.012, 1000)                       # volatility
        ])
        
        # Distribution 7: Mean Reverting Market (Oscillating regime)
        mean_rev_returns = np.column_stack([
            np.sin(np.linspace(0, 20*np.pi, 1000)) * 0.02 + np.random.normal(0, 0.008, 1000),
            np.cos(np.linspace(0, 15*np.pi, 1000)) * 0.015 + np.random.normal(0, 0.006, 1000),
            np.random.gamma(1.5, 0.004, 1000),
            np.random.gamma(1.2, 0.008, 1000)
        ])
        
        # Store normalized reference distributions
        self.reference_distributions = {
            'normal_market': self._normalize_data(normal_returns),
            'bull_market': self._normalize_data(bull_returns),
            'bear_market': self._normalize_data(bear_returns),
            'high_volatility': self._normalize_data(high_vol_returns),
            'low_volatility': self._normalize_data(low_vol_returns),
            'trending': self._normalize_data(trend_returns),
            'mean_reverting': self._normalize_data(mean_rev_returns)
        }
        
        # Estimate sigma for each reference distribution
        self.reference_sigmas = {}
        for name, ref_data in self.reference_distributions.items():
            n_samples = min(200, ref_data.shape[0])
            if n_samples > 1:
                indices = np.random.choice(ref_data.shape[0], n_samples, replace=False)
                sampled = ref_data[indices]
                dists_sq = _compute_dists_sq_numba(sampled)
                if dists_sq.size > 0:
                    self.reference_sigmas[name] = np.sqrt(np.median(dists_sq))
                else:
                    self.reference_sigmas[name] = 1.0
            else:
                self.reference_sigmas[name] = 1.0
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to zero mean and unit variance"""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std < 1e-8] = 1e-8
        return (data - mean) / std
    
    def _compute_additional_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute additional statistical features for the feature vector"""
        if len(df) < 20:
            return {
                'skewness': 0.0,
                'kurtosis': 0.0,
                'momentum_5': 0.0,
                'momentum_10': 0.0,
                'rsi': 50.0,
                'volatility_ratio': 1.0,
                'volume_spike': 0.0,
                'price_acceleration': 0.0
            }
        
        returns = df['returns'].dropna()
        close_prices = df['close']
        volumes = df['volume']
        
        # Statistical moments
        skewness = stats.skew(returns.tail(50)) if len(returns) >= 50 else 0.0
        kurtosis = stats.kurtosis(returns.tail(50)) if len(returns) >= 50 else 0.0
        
        # Short-term momentum
        momentum_5 = close_prices.pct_change(5).iloc[-1] if len(close_prices) > 5 else 0.0
        momentum_10 = close_prices.pct_change(10).iloc[-1] if len(close_prices) > 10 else 0.0
        
        # RSI calculation
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs)).iloc[-1] if len(rs) > 0 else 50.0
        
        # Volatility ratio (current vs historical)
        current_vol = returns.tail(20).std() if len(returns) >= 20 else 0.0
        historical_vol = returns.tail(100).std() if len(returns) >= 100 else current_vol
        volatility_ratio = current_vol / (historical_vol + 1e-10)
        
        # Volume spike detection
        volume_ma = volumes.rolling(20).mean()
        volume_spike = (volumes.iloc[-1] / volume_ma.iloc[-1]) if volume_ma.iloc[-1] > 0 else 1.0
        
        # Price acceleration (second derivative of price)
        price_change = close_prices.pct_change()
        price_acceleration = price_change.diff().iloc[-1] if len(price_change) > 1 else 0.0
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'momentum_5': momentum_5,
            'momentum_10': momentum_10,
            'rsi': rsi,
            'volatility_ratio': volatility_ratio,
            'volume_spike': volume_spike,
            'price_acceleration': price_acceleration
        }
    
    def calculate_30m(self, bar: BarData) -> Dict[str, Any]:
        self.update_30m_history(bar)
        
        # Require minimum data for meaningful MMD calculation
        if len(self.history_30m) < self.test_window:
            # Return zeros for all 23 features (7 MMD scores + 16 additional features)
            return {'mmd_features': np.zeros(23)}
        
        # Convert bars to feature DataFrame
        df = pd.DataFrame([{
            'close': b.close, 'high': b.high, 'low': b.low, 'volume': b.volume
        } for b in self.history_30m])
        
        # Calculate base market features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['range'] = (df['high'] - df['low']) / df['close'].shift(1)
        df['volatility'] = df['returns'].rolling(20, min_periods=1).std() * np.sqrt(252 * 48)
        df = df.dropna()
        
        if len(df) < self.test_window:
            return {'mmd_features': np.zeros(23)}
        
        # Extract test data (most recent window)
        test_data = df[['returns', 'log_returns', 'range', 'volatility']].tail(self.test_window).values
        
        # Normalize test data using robust statistics
        test_median = np.median(test_data, axis=0)
        test_mad = np.median(np.abs(test_data - test_median), axis=0)
        test_mad[test_mad < 1e-8] = 1e-8
        test_normalized = (test_data - test_median) / (1.4826 * test_mad)  # MAD-based normalization
        
        # Compute MMD scores against all 7 reference distributions
        mmd_scores = {}
        for regime_name, ref_distribution in self.reference_distributions.items():
            sigma = self.reference_sigmas.get(regime_name, 1.0)
            mmd_score = compute_mmd(ref_distribution, test_normalized, sigma)
            mmd_scores[regime_name] = mmd_score
        
        # Store MMD scores history
        self.mmd_scores.append(mmd_scores)
        if len(self.mmd_scores) > 100:
            self.mmd_scores.pop(0)
        
        # Compute additional statistical features
        additional_features = self._compute_additional_features(df)
        
        # Calculate momentum indicators
        df['momentum_20'] = df['close'].pct_change(periods=20)
        df['momentum_50'] = df['close'].pct_change(periods=50)
        
        # Volume ratio calculation
        df['volume_ma'] = df['volume'].rolling(window=20, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-9)
        
        # Create comprehensive MMD Feature Vector (23 dimensions)
        recent = df.tail(1)
        
        # Core MMD features (7 dimensions) - one for each reference distribution
        mmd_feature_vector = [
            mmd_scores['normal_market'],
            mmd_scores['bull_market'], 
            mmd_scores['bear_market'],
            mmd_scores['high_volatility'],
            mmd_scores['low_volatility'],
            mmd_scores['trending'],
            mmd_scores['mean_reverting']
        ]
        
        # Additional market features (16 dimensions) 
        extended_features = [
            recent['returns'].iloc[-1] if len(recent) > 0 else 0.0,           # current returns
            recent['log_returns'].iloc[-1] if len(recent) > 0 else 0.0,       # log returns
            recent['range'].iloc[-1] if len(recent) > 0 else 0.0,             # range
            recent['volatility'].iloc[-1] if len(recent) > 0 else 0.0,        # volatility
            df['momentum_20'].iloc[-1] if len(df) > 20 else 0.0,              # momentum_20
            df['momentum_50'].iloc[-1] if len(df) > 50 else 0.0,              # momentum_50
            df['volume_ratio'].iloc[-1] if len(df) > 0 else 1.0,              # volume_ratio
            additional_features['skewness'],                                  # returns skewness
            additional_features['kurtosis'],                                  # returns kurtosis
            additional_features['momentum_5'],                                # short momentum
            additional_features['momentum_10'],                               # medium momentum
            additional_features['rsi'],                                       # RSI
            additional_features['volatility_ratio'],                          # vol ratio
            additional_features['volume_spike'],                              # volume spike
            additional_features['price_acceleration'],                        # price acceleration
            np.mean(list(mmd_scores.values()))                                # average MMD score
        ]
        
        # Combine all features
        full_feature_vector = np.array(mmd_feature_vector + extended_features)
        
        # Store feature history for analysis
        self.feature_history.append(full_feature_vector.copy())
        if len(self.feature_history) > 100:
            self.feature_history.pop(0)
        
        return {
            'mmd_features': np.nan_to_num(full_feature_vector),
            'mmd_scores_by_regime': mmd_scores,
            'regime_similarity': self._get_regime_similarity(mmd_scores)
        }
    
    def _get_regime_similarity(self, mmd_scores: Dict[str, float]) -> Dict[str, Any]:
        """Analyze which regime the current data most closely resembles"""
        if not mmd_scores:
            return {'closest_regime': 'unknown', 'confidence': 0.0, 'distances': {}}
        
        # Lower MMD score means higher similarity
        sorted_regimes = sorted(mmd_scores.items(), key=lambda x: x[1])
        closest_regime = sorted_regimes[0][0]
        closest_distance = sorted_regimes[0][1]
        
        # Calculate confidence based on distance separation
        if len(sorted_regimes) > 1:
            second_distance = sorted_regimes[1][1]
            confidence = max(0.0, min(1.0, (second_distance - closest_distance) / (second_distance + 1e-10)))
        else:
            confidence = 1.0 if closest_distance < 0.5 else 0.5
        
        return {
            'closest_regime': closest_regime,
            'confidence': confidence,
            'distances': mmd_scores,
            'ranking': [name for name, _ in sorted_regimes]
        }
    
    def get_current_values(self) -> Dict[str, Any]:
        """Get current MMD analysis results"""
        if not self.mmd_scores:
            return {
                'mmd_scores': {},
                'feature_vector_size': 23,
                'closest_regime': 'unknown'
            }
        
        latest_scores = self.mmd_scores[-1]
        regime_analysis = self._get_regime_similarity(latest_scores)
        
        return {
            'mmd_scores': latest_scores,
            'feature_vector_size': 23,
            'closest_regime': regime_analysis['closest_regime'],
            'regime_confidence': regime_analysis['confidence'],
            'regime_ranking': regime_analysis['ranking']
        }
    
    def get_feature_names(self) -> List[str]:
        """Get names of all 23 features in the MMD feature vector"""
        mmd_features = [
            'mmd_normal_market',
            'mmd_bull_market', 
            'mmd_bear_market',
            'mmd_high_volatility',
            'mmd_low_volatility',
            'mmd_trending',
            'mmd_mean_reverting'
        ]
        
        additional_features = [
            'current_returns',
            'log_returns',
            'price_range',
            'volatility',
            'momentum_20',
            'momentum_50',
            'volume_ratio',
            'returns_skewness',
            'returns_kurtosis',
            'momentum_5',
            'momentum_10',
            'rsi',
            'volatility_ratio',
            'volume_spike',
            'price_acceleration',
            'average_mmd_score'
        ]
        
        return mmd_features + additional_features
    
    def get_regime_distributions(self) -> Dict[str, np.ndarray]:
        """Get the 7 reference distributions for analysis"""
        return self.reference_distributions.copy()
    
    def reset(self) -> None:
        """Reset the MMD feature extractor state"""
        self.mmd_scores = []
        self.feature_history = []
        self.history_30m = []
        # Keep reference distributions as they are pre-defined