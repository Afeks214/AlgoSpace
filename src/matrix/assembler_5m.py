"""
MatrixAssembler5m - Short-term Tactical Matrix

This assembler creates a 60x7 matrix capturing 5 hours of price action
using 5-minute bars. It focuses on immediate market dynamics, Fair Value
Gaps, and short-term momentum for tactical execution decisions.
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
    
    Features:
    1. fvg_bullish_active: Binary flag for active bullish FVG
    2. fvg_bearish_active: Binary flag for active bearish FVG
    3. fvg_nearest_level: Distance to nearest FVG level (normalized)
    4. fvg_age: Age of FVG with exponential decay
    5. fvg_mitigation_signal: Binary flag for recent mitigation
    6. price_momentum_5: 5-bar price momentum (percentage)
    7. volume_ratio: Current volume vs average (log-transformed)
    
    Matrix shape: (60, 7) representing 5 hours of 5-minute bars
    """
    
    def __init__(self, name: str, kernel: Any):
        """Initialize MatrixAssembler5m."""
        # Load configuration
        config = kernel.config.get('matrix_assemblers', {}).get('5m', {})
        
        # Set default configuration if not provided
        if not config:
            config = {
                'window_size': 60,  # 5 hours of 5-min bars
                'features': [
                    'fvg_bullish_active',
                    'fvg_bearish_active',
                    'fvg_nearest_level',
                    'fvg_age',
                    'fvg_mitigation_signal',
                    'price_momentum_5',
                    'volume_ratio'
                ],
                'warmup_period': 20,  # Need at least 20 bars for momentum
                'feature_configs': {
                    'price_momentum_5': {'ema_alpha': 0.1},
                    'volume_ratio': {'ema_alpha': 0.05}
                }
            }
        
        super().__init__(name, kernel, config)
        
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
            "MatrixAssembler5m initialized for short-term tactical analysis"
        )
    
    def extract_features(self, feature_store: Dict[str, Any]) -> Optional[List[float]]:
        """
        Extract 5-minute features from feature store.
        
        Args:
            feature_store: Complete feature dictionary from IndicatorEngine
            
        Returns:
            List of raw feature values or None if extraction fails
        """
        try:
            # Update current price and volume
            self.current_price = feature_store.get('current_price', self.current_price)
            current_volume = feature_store.get('current_volume', 0)
            
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
            
            # Extract FVG features
            fvg_bullish_active = int(feature_store.get('fvg_bullish_active', False))
            fvg_bearish_active = int(feature_store.get('fvg_bearish_active', False))
            fvg_nearest_level = feature_store.get('fvg_nearest_level', self.current_price)
            fvg_age = feature_store.get('fvg_age', 0)
            fvg_mitigation_signal = int(feature_store.get('fvg_mitigation_signal', False))
            
            # Calculate price momentum
            price_momentum = self._calculate_price_momentum()
            
            # Calculate volume ratio
            volume_ratio = safe_divide(current_volume, self.volume_ema, default=1.0)
            
            # Compile features
            features = [
                float(fvg_bullish_active),
                float(fvg_bearish_active),
                fvg_nearest_level,
                float(fvg_age),
                float(fvg_mitigation_signal),
                price_momentum,
                volume_ratio
            ]
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return None
    
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
            # 1. FVG Bullish Active: Binary, keep as is
            processed[0] = raw_features[0]
            
            # 2. FVG Bearish Active: Binary, keep as is
            processed[1] = raw_features[1]
            
            # 3. FVG Nearest Level: Normalize as % distance from current price
            fvg_level = raw_features[2]
            if self.current_price > 0 and fvg_level > 0:
                # Calculate percentage distance
                fvg_distance_pct = percentage_from_price(
                    fvg_level,
                    self.current_price,
                    clip_pct=2.0  # FVGs typically within 2%
                )
                # Scale to [-1, 1] where 0 means at current price
                processed[2] = fvg_distance_pct / 2.0
            else:
                processed[2] = 0.0
            
            # 4. FVG Age: Apply exponential decay
            # Newer FVGs (age=0) have value 1.0
            # Older FVGs decay: age=10 → 0.37, age=20 → 0.14
            fvg_age = raw_features[3]
            processed[3] = exponential_decay(fvg_age, decay_rate=0.1)
            
            # 5. FVG Mitigation Signal: Binary, keep as is
            processed[4] = raw_features[4]
            
            # 6. Price Momentum: Already in percentage, scale to [-1, 1]
            momentum = raw_features[5]
            # Assume ±5% is significant momentum for 5-bar period
            processed[5] = np.clip(momentum / 5.0, -1.0, 1.0)
            
            # 7. Volume Ratio: Log transform and normalize
            volume_ratio = raw_features[6]
            if volume_ratio > 0:
                # Log transform to handle spikes
                log_ratio = np.log1p(volume_ratio - 1)  # log1p(x) = log(1+x)
                # Normalize: ratio of 1 → 0, ratio of 3 → ~0.7, ratio of 10 → ~1.0
                processed[6] = np.tanh(log_ratio)
            else:
                processed[6] = 0.0
            
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
    
    def get_feature_names(self) -> List[str]:
        """Get human-readable feature names."""
        return [
            "FVG Bullish Active",
            "FVG Bearish Active",
            "FVG Level Distance (%)",
            "FVG Age (decay)",
            "FVG Mitigation Signal",
            "Price Momentum 5-bar (%)",
            "Volume Ratio (log)"
        ]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get estimated feature importance for interpretation.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Tactical features weighted by immediate impact
        return {
            "fvg_bullish_active": 0.20,
            "fvg_bearish_active": 0.20,
            "fvg_nearest_level": 0.15,
            "fvg_age": 0.10,
            "fvg_mitigation_signal": 0.15,
            "price_momentum_5": 0.10,
            "volume_ratio": 0.10
        }
    
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
            
            # Analyze FVG patterns in recent history
            last_20_bars = matrix[-20:] if len(matrix) >= 20 else matrix
            
            bullish_active_count = np.sum(last_20_bars[:, 0])
            bearish_active_count = np.sum(last_20_bars[:, 1])
            mitigation_count = np.sum(last_20_bars[:, 4])
            
            # Average FVG age when active
            active_mask = (last_20_bars[:, 0] > 0) | (last_20_bars[:, 1] > 0)
            avg_age = np.mean(last_20_bars[active_mask, 3]) if np.any(active_mask) else 0
            
            return {
                "status": "ready",
                "last_20_bars": {
                    "bullish_fvg_count": int(bullish_active_count),
                    "bearish_fvg_count": int(bearish_active_count),
                    "mitigation_count": int(mitigation_count),
                    "avg_fvg_age_when_active": float(avg_age),
                    "fvg_activity_rate": float(np.mean(active_mask))
                }
            }
    
    def validate_features(self, features: List[float]) -> bool:
        """
        Validate that features are within expected ranges.
        
        Args:
            features: Raw feature values
            
        Returns:
            True if all features are valid
        """
        if len(features) != 7:
            return False
        
        # Check binary features
        for i in [0, 1, 4]:  # Bullish, bearish, mitigation
            if features[i] not in [0.0, 1.0]:
                self.logger.warning(f"Binary feature {i} not 0 or 1: {features[i]}")
                return False
        
        # Check FVG age is non-negative
        if features[3] < 0:
            self.logger.warning(f"Negative FVG age: {features[3]}")
            return False
        
        # Check volume ratio is positive
        if features[6] < 0:
            self.logger.warning(f"Negative volume ratio: {features[6]}")
            return False
        
        return True