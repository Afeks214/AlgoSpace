"""
MatrixAssembler5m - Short-term Tactical Matrix

This assembler creates a matrix capturing price action using 5-minute bars.
It focuses on immediate market dynamics, Fair Value Gaps, and short-term
momentum for tactical execution decisions.

Configuration is now driven externally from settings.yaml.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import deque

from .base import BaseMatrixAssembler
from .normalizers import (
    exponential_decay, percentage_from_price, log_transform,
    z_score_normalize, safe_divide
)


class MatrixAssembler5m(BaseMatrixAssembler):
    """
    Short-term tactician input matrix.
    
    All configuration including window_size and features list
    is now provided externally via the config parameter.
    
    This class contains only the specialized preprocessing and
    normalization logic for 5-minute timeframe features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MatrixAssembler5m with external configuration.
        
        Args:
            config: Configuration dictionary from settings.yaml
        """
        # Call parent constructor with config
        super().__init__(config)
        
        # Price history for momentum calculation
        self.price_history = deque(maxlen=6)  # Need 6 prices for 5-bar momentum
        
        # Volume tracking
        self.volume_ema = None
        self.volume_ema_alpha = 0.02  # 50-period EMA equivalent
        
        # Current price cache
        self.current_price = None
        
        # FVG tracking
        self.last_fvg_update = None
        
        self.logger.info(
            f"MatrixAssembler5m initialized for short-term tactical analysis "
            f"with {self.n_features} features and window_size={self.window_size}"
        )
    
    def extract_features(self, feature_store: Dict[str, Any]) -> Optional[List[float]]:
        """
        Extract 5-minute features from feature store with custom logic.
        
        Returns None to trigger safe extraction for most features.
        
        Args:
            feature_store: Complete feature dictionary from IndicatorEngine
            
        Returns:
            List of raw feature values or None to use default extraction
        """
        # Update current price and volume
        self.current_price = feature_store.get('current_price', self.current_price)
        if self.current_price is None:
            self.current_price = feature_store.get('close', None)
            
        current_volume = feature_store.get('current_volume', 0)
        if current_volume == 0:
            current_volume = feature_store.get('volume', 0)
        
        if self.current_price is None:
            self.logger.error("No current price available")
            return None
        
        # Update price history
        self.price_history.append(self.current_price)
        
        # Update volume EMA
        if self.volume_ema is None:
            self.volume_ema = max(current_volume, 1)  # Avoid zero
        else:
            self.volume_ema += self.volume_ema_alpha * (current_volume - self.volume_ema)
        
        # Check if we need to calculate custom features
        needs_custom = False
        if 'price_momentum_5' in self.feature_names:
            needs_custom = True
        if 'volume_ratio' in self.feature_names:
            needs_custom = True
            
        if not needs_custom:
            # Use default safe extraction for all features
            return None
            
        # Build features list with custom calculations
        features = []
        for feature_name in self.feature_names:
            if feature_name == 'price_momentum_5':
                # Calculate price momentum
                momentum = self._calculate_price_momentum()
                features.append(momentum)
            elif feature_name == 'volume_ratio':
                # Calculate volume ratio
                ratio = safe_divide(current_volume, self.volume_ema, default=1.0)
                features.append(ratio)
            else:
                # Use safe extraction for other features
                value = feature_store.get(feature_name, 0.0)
                features.append(value)
        
        return features
    
    def _calculate_price_momentum(self) -> float:
        """
        Calculate 5-bar price momentum as percentage change.
        
        Returns:
            Momentum percentage or 0.0 if insufficient data
        """
        if len(self.price_history) < 6:
            return 0.0
        
        # Get prices 5 bars ago and current
        old_price = self.price_history[0]
        current_price = self.price_history[-1]
        
        if old_price <= 0:
            return 0.0
        
        # Calculate percentage change
        momentum = ((current_price - old_price) / old_price) * 100
        
        # Clip extreme values
        momentum = np.clip(momentum, -10.0, 10.0)
        
        return momentum
    
    def preprocess_features(
        self, 
        raw_features: List[float], 
        feature_store: Dict[str, Any]
    ) -> np.ndarray:
        """
        Preprocess features for neural network input.
        
        Applies specific transformations optimized for short-term
        price action and FVG dynamics.
        """
        processed = np.zeros(len(raw_features), dtype=np.float32)
        
        try:
            # Process each feature based on its name
            for i, (feature_name, value) in enumerate(zip(self.feature_names, raw_features)):
                
                if feature_name == 'fvg_bullish_active':
                    # Binary, keep as is
                    processed[i] = float(value)
                    
                elif feature_name == 'fvg_bearish_active':
                    # Binary, keep as is
                    processed[i] = float(value)
                    
                elif feature_name == 'fvg_nearest_level':
                    # Normalize as % distance from current price
                    if self.current_price and self.current_price > 0 and value > 0:
                        # Calculate percentage distance
                        fvg_distance_pct = percentage_from_price(
                            value,
                            self.current_price,
                            clip_pct=2.0  # FVGs typically within 2%
                        )
                        # Scale to [-1, 1] where 0 means at current price
                        processed[i] = fvg_distance_pct / 2.0
                    else:
                        processed[i] = 0.0
                        
                elif feature_name == 'fvg_age':
                    # Apply exponential decay
                    # Newer FVGs (age=0) have value 1.0
                    # Older FVGs decay: age=10 → 0.37, age=20 → 0.14
                    processed[i] = exponential_decay(value, decay_rate=0.1)
                    
                elif feature_name == 'fvg_mitigation_signal':
                    # Binary, keep as is
                    processed[i] = float(value)
                    
                elif feature_name == 'price_momentum_5':
                    # Already in percentage, scale to [-1, 1]
                    # Assume ±5% is significant momentum for 5-bar period
                    processed[i] = np.clip(value / 5.0, -1.0, 1.0)
                    
                elif feature_name == 'volume_ratio':
                    # Log transform and normalize
                    if value > 0:
                        # Log transform to handle spikes
                        log_ratio = np.log1p(value - 1)  # log1p(x) = log(1+x)
                        # Normalize: ratio of 1 → 0, ratio of 3 → ~0.7, ratio of 10 → ~1.0
                        processed[i] = np.tanh(log_ratio)
                    else:
                        processed[i] = 0.0
                        
                else:
                    # Default processing for unknown features
                    normalizer = self.normalizers.get(feature_name)
                    if normalizer and normalizer.n_samples > 10:
                        processed[i] = normalizer.normalize_zscore(value)
                        processed[i] = np.clip(processed[i], -3, 3) / 3
                    else:
                        # Simple clipping for safety
                        processed[i] = np.clip(value, -10, 10) / 10
            
            # Final safety check
            if not np.all(np.isfinite(processed)):
                self.logger.warning("Non-finite values after preprocessing")
                processed = np.nan_to_num(processed, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Ensure reasonable range
            processed = np.clip(processed, -2.0, 2.0)
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Feature preprocessing failed: {e}")
            return np.zeros(len(raw_features), dtype=np.float32)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get estimated feature importance for interpretation.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Default importance scores for known features
        default_importance = {
            "fvg_bullish_active": 0.20,
            "fvg_bearish_active": 0.20,
            "fvg_nearest_level": 0.15,
            "fvg_age": 0.10,
            "fvg_mitigation_signal": 0.15,
            "price_momentum_5": 0.10,
            "volume_ratio": 0.10
        }
        
        # Build importance dict based on actual features
        importance = {}
        total = 0.0
        
        for feature in self.feature_names:
            if feature in default_importance:
                importance[feature] = default_importance[feature]
            else:
                # Assign equal weight to unknown features
                importance[feature] = 0.1
            total += importance[feature]
        
        # Normalize to sum to 1.0
        if total > 0:
            for feature in importance:
                importance[feature] /= total
        
        return importance
    
    def get_fvg_summary(self) -> Dict[str, Any]:
        """
        Get summary of current FVG state.
        
        Returns:
            Dictionary with FVG statistics
        """
        with self._lock:
            if not self.is_ready():
                return {"status": "not_ready"}
            
            matrix = self.get_matrix()
            if matrix is None:
                return {"status": "no_data"}
            
            # Find FVG feature indices
            bullish_idx = None
            bearish_idx = None
            mitigation_idx = None
            age_idx = None
            
            for i, feature_name in enumerate(self.feature_names):
                if feature_name == 'fvg_bullish_active':
                    bullish_idx = i
                elif feature_name == 'fvg_bearish_active':
                    bearish_idx = i
                elif feature_name == 'fvg_mitigation_signal':
                    mitigation_idx = i
                elif feature_name == 'fvg_age':
                    age_idx = i
            
            # Analyze FVG patterns in recent history
            last_20_bars = matrix[-20:] if len(matrix) >= 20 else matrix
            
            result = {
                "status": "ready",
                "last_20_bars": {}
            }
            
            if bullish_idx is not None:
                result["last_20_bars"]["bullish_fvg_count"] = int(np.sum(last_20_bars[:, bullish_idx]))
            
            if bearish_idx is not None:
                result["last_20_bars"]["bearish_fvg_count"] = int(np.sum(last_20_bars[:, bearish_idx]))
                
            if mitigation_idx is not None:
                result["last_20_bars"]["mitigation_count"] = int(np.sum(last_20_bars[:, mitigation_idx]))
            
            # Average FVG age when active
            if bullish_idx is not None and bearish_idx is not None and age_idx is not None:
                active_mask = (last_20_bars[:, bullish_idx] > 0) | (last_20_bars[:, bearish_idx] > 0)
                if np.any(active_mask):
                    avg_age = np.mean(last_20_bars[active_mask, age_idx])
                    result["last_20_bars"]["avg_fvg_age_when_active"] = float(avg_age)
                    result["last_20_bars"]["fvg_activity_rate"] = float(np.mean(active_mask))
            
            return result
    
    def validate_features(self, features: List[float]) -> bool:
        """
        Validate that features are within expected ranges.
        
        Args:
            features: Raw feature values
            
        Returns:
            True if all features are valid
        """
        if len(features) != self.n_features:
            return False
        
        # Check specific features
        for i, feature_name in enumerate(self.feature_names):
            value = features[i]
            
            # Check binary features
            if feature_name in ['fvg_bullish_active', 'fvg_bearish_active', 'fvg_mitigation_signal']:
                if value not in [0.0, 1.0]:
                    self.logger.warning(f"Binary feature {feature_name} not 0 or 1: {value}")
                    return False
                    
            elif feature_name == 'fvg_age':
                # Check FVG age is non-negative
                if value < 0:
                    self.logger.warning(f"Negative FVG age: {value}")
                    return False
                    
            elif feature_name == 'volume_ratio':
                # Check volume ratio is positive
                if value < 0:
                    self.logger.warning(f"Negative volume ratio: {value}")
                    return False
            
            # General validation
            if not np.isfinite(value):
                self.logger.warning(f"Non-finite value for {feature_name}: {value}")
                return False
        
        return True