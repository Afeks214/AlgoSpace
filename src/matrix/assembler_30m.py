"""
MatrixAssembler30m - Long-term Market Structure Matrix

This assembler creates a 48x8 matrix capturing 24 hours of market structure
using 30-minute bars. It focuses on trend, momentum, and support/resistance
dynamics for strategic decision making.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base import BaseMatrixAssembler
from .normalizers import (
    min_max_scale, percentage_from_price, cyclical_encode,
    z_score_normalize, safe_divide
)


class MatrixAssembler30m(BaseMatrixAssembler):
    """
    Long-term structure analyzer input matrix.
    
    Features:
    1. mlmi_value: Machine Learning Market Index (0-100)
    2. mlmi_signal: Crossover signal (-1, 0, 1)
    3. nwrqk_value: Nadaraya-Watson regression value
    4. nwrqk_slope: Rate of change of NW-RQK
    5. lvn_distance_points: Distance to nearest LVN in points
    6. lvn_nearest_strength: Strength of nearest LVN (0-100)
    7. time_hour_sin: Cyclical encoding of hour (sin component)
    8. time_hour_cos: Cyclical encoding of hour (cos component)
    
    Matrix shape: (48, 8) representing 24 hours of 30-minute bars
    """
    
    def __init__(self, name: str, kernel: Any):
        """Initialize MatrixAssembler30m."""
        # Load configuration
        config = kernel.config.get('matrix_assemblers', {}).get('30m', {})
        
        # Set default configuration if not provided
        if not config:
            config = {
                'window_size': 48,  # 24 hours of 30-min bars
                'features': [
                    'mlmi_value',
                    'mlmi_signal', 
                    'nwrqk_value',
                    'nwrqk_slope',
                    'lvn_distance_points',
                    'lvn_nearest_strength',
                    'time_hour_sin',
                    'time_hour_cos'
                ],
                'warmup_period': 48,
                'feature_configs': {
                    'mlmi_value': {'ema_alpha': 0.02},
                    'nwrqk_slope': {'ema_alpha': 0.05},
                    'lvn_distance_points': {'ema_alpha': 0.01}
                }
            }
        
        super().__init__(name, kernel, config)
        
        # Cache for current price (needed for percentage calculations)
        self.current_price = None
        
        # Additional statistics for robust normalization
        self.price_ema = None
        self.price_ema_alpha = 0.001  # Slow adaptation for price level
        
        self.logger.info(
            "MatrixAssembler30m initialized for long-term structure analysis"
        )
    
    def extract_features(self, feature_store: Dict[str, Any]) -> Optional[List[float]]:
        """
        Extract 30-minute features from feature store.
        
        Args:
            feature_store: Complete feature dictionary from IndicatorEngine
            
        Returns:
            List of raw feature values or None if extraction fails
        """
        try:
            # Update current price
            self.current_price = feature_store.get('current_price', self.current_price)
            if self.current_price is None:
                self.logger.error("No current price available")
                return None
            
            # Update price EMA
            if self.price_ema is None:
                self.price_ema = self.current_price
            else:
                self.price_ema += self.price_ema_alpha * (self.current_price - self.price_ema)
            
            # Extract core features
            mlmi_value = feature_store.get('mlmi_value', 50.0)  # Default neutral
            mlmi_signal = feature_store.get('mlmi_signal', 0)
            nwrqk_value = feature_store.get('nwrqk_value', self.current_price)
            nwrqk_slope = feature_store.get('nwrqk_slope', 0.0)
            
            # LVN features
            lvn_distance = feature_store.get('lvn_distance_points', 0.0)
            lvn_strength = feature_store.get('lvn_nearest_strength', 0.0)
            
            # Time features (for cyclical encoding)
            timestamp = feature_store.get('timestamp', datetime.now())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            hour = timestamp.hour + timestamp.minute / 60.0  # Fractional hour
            
            # Validate features
            features = [
                mlmi_value,
                float(mlmi_signal),
                nwrqk_value,
                nwrqk_slope,
                lvn_distance,
                lvn_strength,
                hour,  # Will be split into sin/cos during preprocessing
                hour   # Placeholder for cos component
            ]
            
            # Check for validity
            if not all(isinstance(f, (int, float)) for f in features[:-2]):
                self.logger.error(f"Invalid feature types: {[type(f) for f in features]}")
                return None
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return None
    
    def preprocess_features(
        self, 
        raw_features: List[float], 
        feature_store: Dict[str, Any]
    ) -> np.ndarray:
        """
        Preprocess features for neural network input.
        
        Applies specific normalization for each feature type to ensure
        all values are in appropriate ranges for neural network processing.
        """
        processed = np.zeros(len(raw_features), dtype=np.float32)
        
        try:
            # 1. MLMI Value: Scale from [0,100] to [-1,1]
            mlmi_value = raw_features[0]
            processed[0] = min_max_scale(mlmi_value, 0, 100, (-1, 1))
            
            # 2. MLMI Signal: Already in [-1, 0, 1], just ensure float
            processed[1] = float(raw_features[1])
            
            # 3. NW-RQK Value: Normalize as percentage from current price
            nwrqk_value = raw_features[2]
            if self.current_price > 0:
                nwrqk_pct = percentage_from_price(
                    nwrqk_value, 
                    self.current_price,
                    clip_pct=5.0  # Clip at ±5%
                )
                # Scale percentage to [-1, 1]
                processed[2] = nwrqk_pct / 5.0
            else:
                processed[2] = 0.0
            
            # 4. NW-RQK Slope: Use rolling z-score normalization
            nwrqk_slope = raw_features[3]
            normalizer = self.normalizers.get('nwrqk_slope')
            if normalizer and normalizer.n_samples > 10:
                processed[3] = normalizer.normalize_zscore(nwrqk_slope)
                processed[3] = np.clip(processed[3], -2, 2) / 2  # Scale to [-1, 1]
            else:
                # During warmup, use simple scaling
                processed[3] = np.tanh(nwrqk_slope * 10)  # Assumes slope ~0.1 is significant
            
            # 5. LVN Distance: Convert points to percentage and scale
            lvn_distance = raw_features[4]
            if self.current_price > 0:
                lvn_distance_pct = (lvn_distance / self.current_price) * 100
                # Use exponential decay - closer LVNs are more important
                # 0% distance = 1.0, 1% distance ≈ 0.37, 2% distance ≈ 0.14
                processed[4] = np.exp(-lvn_distance_pct)
            else:
                processed[4] = 0.0
            
            # 6. LVN Strength: Scale from [0,100] to [0,1]
            lvn_strength = raw_features[5]
            processed[5] = min_max_scale(lvn_strength, 0, 100, (0, 1))
            
            # 7-8. Time features: Cyclical encoding
            hour = raw_features[6]
            hour_sin, hour_cos = cyclical_encode(hour, 24)
            processed[6] = hour_sin
            processed[7] = hour_cos
            
            # Final safety check - ensure all values are finite
            if not np.all(np.isfinite(processed)):
                self.logger.warning("Non-finite values after preprocessing, applying safety")
                processed = np.nan_to_num(processed, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Ensure all values are in reasonable range
            processed = np.clip(processed, -3.0, 3.0)
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Feature preprocessing failed: {e}")
            # Return safe defaults
            return np.zeros(len(raw_features), dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get human-readable feature names."""
        return [
            "MLMI Value (scaled)",
            "MLMI Signal",
            "NW-RQK Value (%)",
            "NW-RQK Slope (normalized)",
            "LVN Distance (decay)",
            "LVN Strength",
            "Hour (sin)",
            "Hour (cos)"
        ]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get estimated feature importance for interpretation.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # These are heuristic importances based on strategy design
        return {
            "mlmi_value": 0.20,
            "mlmi_signal": 0.15,
            "nwrqk_value": 0.15,
            "nwrqk_slope": 0.20,
            "lvn_distance_points": 0.10,
            "lvn_nearest_strength": 0.10,
            "time_hour_sin": 0.05,
            "time_hour_cos": 0.05
        }
    
    def validate_features(self, features: List[float]) -> bool:
        """
        Validate that features are within expected ranges.
        
        Args:
            features: Raw feature values
            
        Returns:
            True if all features are valid
        """
        if len(features) != 8:
            return False
        
        # Check MLMI value range
        if not 0 <= features[0] <= 100:
            self.logger.warning(f"MLMI value out of range: {features[0]}")
            return False
        
        # Check MLMI signal
        if features[1] not in [-1, 0, 1]:
            self.logger.warning(f"Invalid MLMI signal: {features[1]}")
            return False
        
        # Check LVN strength range
        if not 0 <= features[5] <= 100:
            self.logger.warning(f"LVN strength out of range: {features[5]}")
            return False
        
        # Check time hour range
        if not 0 <= features[6] < 24:
            self.logger.warning(f"Hour out of range: {features[6]}")
            return False
        
        return True