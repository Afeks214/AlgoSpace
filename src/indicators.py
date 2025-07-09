"""
Production-ready Technical Indicators Module
Implements NW-RQK, MLMI, and FVG with validation and optimization
"""

import numpy as np
import pandas as pd
from numba import njit, prange, float64, int64, boolean
from numba.typed import List
from typing import Tuple, Optional, Dict, Any
import logging
from functools import lru_cache

class IndicatorBase:
    """Base class for all indicators with common functionality"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validate_config()
        self._cache = {}
        
    def _validate_config(self):
        """Validate indicator configuration"""
        raise NotImplementedError("Subclasses must implement _validate_config")
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicator values"""
        raise NotImplementedError("Subclasses must implement calculate")
    
    def _validate_input_data(self, data: pd.DataFrame, required_columns: list):
        """Validate input data has required columns"""
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if len(data) < self.config.get('min_periods', 0):
            raise ValueError(f"Insufficient data: need at least {self.config.get('min_periods', 0)} periods")

class NWRQK(IndicatorBase):
    """Nadaraya-Watson Rational Quadratic Kernel indicator with ensemble"""
    
    def _validate_config(self):
        """Validate NW-RQK configuration"""
        required_params = ['window', 'n_kernels', 'alphas', 'length_scales', 'threshold']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required parameter: {param}")
        
        # Validate array lengths
        if len(self.config['alphas']) != self.config['n_kernels']:
            raise ValueError("Number of alphas must match n_kernels")
        if len(self.config['length_scales']) != self.config['n_kernels']:
            raise ValueError("Number of length_scales must match n_kernels")
        
        # Validate parameter ranges
        if self.config['window'] < 10:
            self.logger.warning("Window size < 10 may produce unstable results")
        
        for alpha in self.config['alphas']:
            if not 0 < alpha < 1:
                raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate NW-RQK signals with caching"""
        self._validate_input_data(data, ['close'])
        
        # Check cache
        cache_key = f"nwrqk_{len(data)}_{data.index[-1]}"
        if self.config.get('cache_enabled', True) and cache_key in self._cache:
            self.logger.debug("Using cached NW-RQK values")
            return self._cache[cache_key]
        
        prices = data['close'].values
        
        # Calculate NW-RQK values
        nwrqk_values = self._nwrqk_ensemble(
            prices,
            window=self.config['window'],
            n_kernels=self.config['n_kernels'],
            alphas=np.array(self.config['alphas']),
            length_scales=np.array(self.config['length_scales'])
        )
        
        # Calculate signals
        bull_signals, bear_signals, signal_strength = self._calculate_signals(
            prices, nwrqk_values, 
            threshold=self.config['threshold'],
            volatility_adaptive=self.config.get('volatility_adaptive', True)
        )
        
        # Create results dataframe
        results = pd.DataFrame(index=data.index)
        results['nwrqk'] = nwrqk_values
        results['nwrqk_bull'] = bull_signals
        results['nwrqk_bear'] = bear_signals
        results['nwrqk_strength'] = signal_strength
        
        # Add additional metrics
        results['nwrqk_slope'] = results['nwrqk'].pct_change(5)
        results['price_deviation'] = (data['close'] - results['nwrqk']) / results['nwrqk']
        
        # Cache results
        if self.config.get('cache_enabled', True):
            self._cache[cache_key] = results
        
        return results
    
    @staticmethod
    @njit(fastmath=True, cache=True)
    def _rational_quadratic_kernel(x1: float, x2: float, alpha: float, length_scale: float) -> float:
        """Rational Quadratic Kernel function"""
        diff = x1 - x2
        return (1.0 + (diff * diff) / (2.0 * alpha * length_scale * length_scale)) ** (-alpha)
    
    @staticmethod
    @njit(parallel=True, fastmath=True, cache=True)
    def _nwrqk_ensemble(prices: np.ndarray, window: int, n_kernels: int, 
                       alphas: np.ndarray, length_scales: np.ndarray) -> np.ndarray:
        """Multi-kernel ensemble NW-RQK implementation"""
        n = len(prices)
        nwrqk_values = np.zeros(n)
        
        for i in prange(window, n):
            window_prices = prices[i-window:i]
            predictions = np.zeros(n_kernels)
            
            for k in range(n_kernels):
                weights = np.zeros(window)
                for j in range(window):
                    weights[j] = NWRQK._rational_quadratic_kernel(
                        float(i), float(i-window+j), 
                        alphas[k], length_scales[k]
                    )
                
                weight_sum = np.sum(weights)
                if weight_sum > 0:
                    weights /= weight_sum
                    predictions[k] = np.sum(weights * window_prices)
                else:
                    predictions[k] = window_prices[-1]
            
            nwrqk_values[i] = np.mean(predictions)
        
        # Fill initial values
        for i in range(window):
            nwrqk_values[i] = prices[i] if i < len(prices) else 0
        
        return nwrqk_values
    
    @staticmethod
    @njit(parallel=True, fastmath=True, cache=True)
    def _calculate_signals(prices: np.ndarray, nwrqk_values: np.ndarray, 
                          threshold: float, volatility_adaptive: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate NW-RQK trend signals with adaptive thresholds"""
        n = len(prices)
        bull_signals = np.zeros(n, dtype=np.bool_)
        bear_signals = np.zeros(n, dtype=np.bool_)
        signal_strength = np.zeros(n)
        
        for i in prange(1, n):
            if nwrqk_values[i] > 0 and prices[i] > 0:
                deviation = (prices[i] - nwrqk_values[i]) / nwrqk_values[i]
                
                if i > 5:
                    slope = (nwrqk_values[i] - nwrqk_values[i-5]) / nwrqk_values[i-5] if nwrqk_values[i-5] > 0 else 0
                    
                    adaptive_threshold = threshold
                    if volatility_adaptive and i > 20:
                        returns = np.zeros(20)
                        for j in range(20):
                            if prices[i-j-1] > 0:
                                returns[j] = (prices[i-j] - prices[i-j-1]) / prices[i-j-1]
                        volatility = np.std(returns)
                        adaptive_threshold = threshold * (1 + volatility * 10)
                    
                    if slope > adaptive_threshold and deviation > -0.01:
                        bull_signals[i] = True
                        signal_strength[i] = min(slope / adaptive_threshold, 2.0)
                    elif slope < -adaptive_threshold and deviation < 0.01:
                        bear_signals[i] = True
                        signal_strength[i] = min(abs(slope) / adaptive_threshold, 2.0)
        
        return bull_signals, bear_signals, signal_strength

class MLMI(IndicatorBase):
    """Machine Learning Market Intelligence indicator with KNN"""
    
    def _validate_config(self):
        """Validate MLMI configuration"""
        required_params = ['window', 'k_neighbors', 'feature_window', 'rsi_period', 
                          'bull_threshold', 'bear_threshold', 'confidence_threshold']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required parameter: {param}")
        
        # Validate parameter ranges
        if self.config['k_neighbors'] < 3:
            self.logger.warning("k_neighbors < 3 may produce unstable results")
        
        if not 0 < self.config['bull_threshold'] < 1:
            raise ValueError("bull_threshold must be between 0 and 1")
        if not 0 < self.config['bear_threshold'] < 1:
            raise ValueError("bear_threshold must be between 0 and 1")
        
        if self.config['bull_threshold'] <= self.config['bear_threshold']:
            raise ValueError("bull_threshold must be greater than bear_threshold")
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MLMI signals"""
        self._validate_input_data(data, ['close'])
        
        prices = data['close'].values
        
        # Calculate enhanced MLMI
        bull_signals, bear_signals, confidence = self._calculate_mlmi_enhanced(
            prices,
            window=self.config['window'],
            k=self.config['k_neighbors'],
            feature_window=self.config['feature_window'],
            rsi_period=self.config['rsi_period'],
            bull_threshold=self.config['bull_threshold'],
            bear_threshold=self.config['bear_threshold'],
            confidence_threshold=self.config['confidence_threshold'],
            volatility_window=self.config.get('volatility_window', 20),
            volatility_scale=self.config.get('volatility_scale', 2.0)
        )
        
        # Create results dataframe
        results = pd.DataFrame(index=data.index)
        results['mlmi_bull'] = bull_signals
        results['mlmi_bear'] = bear_signals
        results['mlmi_confidence'] = confidence
        
        # Add signal counts for monitoring
        results['mlmi_signal'] = 0
        results.loc[results['mlmi_bull'], 'mlmi_signal'] = 1
        results.loc[results['mlmi_bear'], 'mlmi_signal'] = -1
        
        return results
    
    @staticmethod
    @njit(fastmath=True, cache=True)
    def _calculate_rsi(prices: np.ndarray, period: int) -> np.ndarray:
        """Ultra-fast RSI calculation"""
        n = len(prices)
        rsi = np.zeros(n)
        
        if n < period + 1:
            return rsi
        
        deltas = np.zeros(n)
        for i in range(1, n):
            deltas[i] = prices[i] - prices[i-1]
        
        avg_gain = 0.0
        avg_loss = 0.0
        
        for i in range(1, period + 1):
            if deltas[i] > 0:
                avg_gain += deltas[i]
            else:
                avg_loss -= deltas[i]
        
        avg_gain /= period
        avg_loss /= period
        
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi[period] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[period] = 100.0
        
        for i in range(period + 1, n):
            if deltas[i] > 0:
                avg_gain = (avg_gain * (period - 1) + deltas[i]) / period
                avg_loss = avg_loss * (period - 1) / period
            else:
                avg_gain = avg_gain * (period - 1) / period
                avg_loss = (avg_loss * (period - 1) - deltas[i]) / period
            
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))
            else:
                rsi[i] = 100.0
        
        return rsi
    
    @staticmethod
    @njit(fastmath=True, cache=True)
    def _euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Euclidean distance between two vectors"""
        dist = 0.0
        for i in range(len(x1)):
            diff = x1[i] - x2[i]
            dist += diff * diff
        return np.sqrt(dist)
    
    @staticmethod
    @njit(fastmath=True, cache=True)
    def _volatility_adaptive_knn(features: np.ndarray, labels: np.ndarray, 
                                query: np.ndarray, k_base: int, 
                                volatility: float, vol_scale: float) -> float:
        """KNN with volatility-based K adjustment"""
        k = max(3, min(k_base, int(k_base * (1 - volatility * vol_scale))))
        
        n_samples = len(labels)
        if n_samples < k:
            return 0.5
        
        distances = np.zeros(n_samples)
        for i in range(n_samples):
            distances[i] = MLMI._euclidean_distance(features[i], query)
        
        indices = np.argsort(distances)[:k]
        
        bull_score = 0.0
        total_weight = 0.0
        
        for i in range(k):
            idx = indices[i]
            if distances[idx] > 0:
                weight = 1.0 / (1.0 + distances[idx])
            else:
                weight = 1.0
            
            bull_score += labels[idx] * weight
            total_weight += weight
        
        if total_weight > 0:
            return bull_score / total_weight
        else:
            return 0.5
    
    @staticmethod
    @njit(parallel=True, fastmath=True, cache=True)
    def _calculate_mlmi_enhanced(prices: np.ndarray, window: int, k: int, 
                                feature_window: int, rsi_period: int,
                                bull_threshold: float, bear_threshold: float,
                                confidence_threshold: float, volatility_window: int,
                                volatility_scale: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Enhanced MLMI with volatility adaptation"""
        n = len(prices)
        mlmi_bull = np.zeros(n, dtype=np.bool_)
        mlmi_bear = np.zeros(n, dtype=np.bool_)
        confidence = np.zeros(n)
        
        # Calculate RSI
        rsi = MLMI._calculate_rsi(prices, rsi_period)
        
        # Calculate volatility
        volatility = np.zeros(n)
        for i in range(volatility_window, n):
            returns = np.zeros(volatility_window)
            for j in range(volatility_window):
                if prices[i-j-1] > 0:
                    returns[j] = (prices[i-j] - prices[i-j-1]) / prices[i-j-1]
            volatility[i] = np.std(returns)
        
        # MLMI calculation
        lookback = max(window * 10, 100)
        
        for i in prange(lookback, n):
            start_idx = max(0, i - lookback)
            historical_size = i - start_idx - feature_window - 1
            
            if historical_size < k:
                continue
            
            features = np.zeros((historical_size, feature_window))
            labels = np.zeros(historical_size)
            
            for j in range(historical_size):
                idx = start_idx + j
                for f in range(feature_window):
                    features[j, f] = rsi[idx + f]
                
                if prices[idx + feature_window] > 0 and prices[idx + feature_window - 1] > 0:
                    ret = (prices[idx + feature_window] - prices[idx + feature_window - 1]) / prices[idx + feature_window - 1]
                    labels[j] = 1.0 if ret > 0 else 0.0
            
            query = np.zeros(feature_window)
            for f in range(feature_window):
                query[f] = rsi[i - feature_window + f]
            
            bull_prob = MLMI._volatility_adaptive_knn(features, labels, query, k, volatility[i], volatility_scale)
            confidence[i] = abs(bull_prob - 0.5) * 2
            
            if bull_prob > bull_threshold and confidence[i] > confidence_threshold:
                mlmi_bull[i] = True
            elif bull_prob < bear_threshold and confidence[i] > confidence_threshold:
                mlmi_bear[i] = True
        
        return mlmi_bull, mlmi_bear, confidence

class FVG(IndicatorBase):
    """Fair Value Gap indicator with volume confirmation"""
    
    def _validate_config(self):
        """Validate FVG configuration"""
        required_params = ['min_gap_pct', 'volume_factor']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required parameter: {param}")
        
        if self.config['min_gap_pct'] <= 0:
            raise ValueError("min_gap_pct must be positive")
        if self.config['volume_factor'] < 1:
            self.logger.warning("volume_factor < 1 may produce too many signals")
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate FVG signals"""
        self._validate_input_data(data, ['high', 'low', 'close', 'volume'])
        
        # Calculate FVG
        bull_signals, bear_signals, gap_size = self._detect_fvg_with_volume(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            data['volume'].values,
            min_gap_pct=self.config['min_gap_pct'],
            volume_factor=self.config['volume_factor'],
            volume_window=self.config.get('volume_window', 20)
        )
        
        # Create results dataframe
        results = pd.DataFrame(index=data.index)
        results['fvg_bull'] = bull_signals
        results['fvg_bear'] = bear_signals
        results['fvg_size'] = gap_size
        
        # Add gap statistics
        results['fvg_count'] = results['fvg_bull'].astype(int) + results['fvg_bear'].astype(int)
        
        return results
    
    @staticmethod
    @njit(parallel=True, fastmath=True, cache=True)
    def _detect_fvg_with_volume(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                               volume: np.ndarray, min_gap_pct: float, 
                               volume_factor: float, volume_window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect Fair Value Gaps with volume confirmation"""
        n = len(high)
        fvg_bull = np.zeros(n, dtype=np.bool_)
        fvg_bear = np.zeros(n, dtype=np.bool_)
        gap_size = np.zeros(n)
        
        # Calculate average volume
        avg_volume = np.zeros(n)
        for i in range(volume_window, n):
            avg_volume[i] = np.mean(volume[i-volume_window:i])
        
        for i in prange(2, n):
            if avg_volume[i] == 0:
                continue
            
            # Volume confirmation
            vol_confirmed = volume[i] > avg_volume[i] * volume_factor
            
            # Bullish FVG: gap up
            gap_up = low[i] - high[i-2]
            if gap_up > 0 and vol_confirmed:
                gap_pct = gap_up / close[i-1] if close[i-1] > 0 else 0
                if gap_pct > min_gap_pct:
                    fvg_bull[i] = True
                    gap_size[i] = gap_pct
            
            # Bearish FVG: gap down
            gap_down = low[i-2] - high[i]
            if gap_down > 0 and vol_confirmed:
                gap_pct = gap_down / close[i-1] if close[i-1] > 0 else 0
                if gap_pct > min_gap_pct:
                    fvg_bear[i] = True
                    gap_size[i] = -gap_pct
        
        return fvg_bull, fvg_bear, gap_size