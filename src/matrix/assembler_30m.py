"""
MatrixAssembler30m - Long-term Market Structure Matrix

This assembler creates a matrix capturing market structure using 30-minute bars.
It focuses on trend, momentum, and support/resistance dynamics for strategic
decision making.

Configuration is now driven externally from settings.yaml.
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
    
    All configuration including window_size and features list
    is now provided externally via the config parameter.
    
    This class contains only the specialized preprocessing and
    normalization logic for 30-minute timeframe features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MatrixAssembler30m with external configuration.
        
        Args:
            config: Configuration dictionary from settings.yaml
        """
        # Call parent constructor with config
        super().__init__(config)
        
        # Cache for current price (needed for percentage calculations)
        self.current_price = None
        
        # Additional statistics for robust normalization
        self.price_ema = None
        self.price_ema_alpha = 0.001  # Slow adaptation for price level
        
        self.logger.info(
            f"MatrixAssembler30m initialized for long-term structure analysis "
            f"with {self.n_features} features and window_size={self.window_size}"
        )
    
    def extract_features(self, feature_store: Dict[str, Any]) -> Optional[List[float]]:
        """
        Extract 30-minute features from feature store with custom logic.
        
        Returns None to trigger safe extraction if custom logic is not needed.
        
        Args:
            feature_store: Complete feature dictionary from IndicatorEngine
            
        Returns:
            List of raw feature values or None to use default extraction
        """
        # Update current price if available
        self.current_price = feature_store.get('current_price', self.current_price)
        if self.current_price is None:
            # Try to get from close price
            self.current_price = feature_store.get('close', None)
            
        if self.current_price is None:
            self.logger.error("No current price available")
            return None
        
        # Update price EMA
        if self.price_ema is None:
            self.price_ema = self.current_price
        else:
            self.price_ema += self.price_ema_alpha * (self.current_price - self.price_ema)
        
        # Handle time features specially if they're in the feature list
        if 'time_hour_sin' in self.feature_names or 'time_hour_cos' in self.feature_names:
            # Extract timestamp for cyclical encoding
            timestamp = feature_store.get('timestamp', datetime.now())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            hour = timestamp.hour + timestamp.minute / 60.0  # Fractional hour
            
            # Build features list with special handling for time
            features = []
            for feature_name in self.feature_names:
                if feature_name == 'time_hour_sin' or feature_name == 'time_hour_cos':
                    # Both will use the same hour value, preprocessing will split them
                    features.append(hour)
                else:
                    # Use safe extraction for other features
                    value = feature_store.get(feature_name, 0.0)
                    features.append(value)
            
            return features
        
        # For non-time features, return None to use default safe extraction
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
            # Process each feature based on its name
            for i, (feature_name, value) in enumerate(zip(self.feature_names, raw_features)):
                
                if feature_name == 'mlmi_value':
                    # MLMI Value: Scale from [0,100] to [-1,1]
                    processed[i] = min_max_scale(value, 0, 100, (-1, 1))
                    
                elif feature_name == 'mlmi_signal':
                    # MLMI Signal: Already in [-1, 0, 1], just ensure float
                    processed[i] = float(value)
                    
                elif feature_name == 'nwrqk_value':
                    # NW-RQK Value: Normalize as percentage from current price
                    if self.current_price and self.current_price > 0:
                        nwrqk_pct = percentage_from_price(
                            value, 
                            self.current_price,
                            clip_pct=5.0  # Clip at ±5%
                        )
                        # Scale percentage to [-1, 1]
                        processed[i] = nwrqk_pct / 5.0
                    else:
                        processed[i] = 0.0
                        
                elif feature_name == 'nwrqk_slope':
                    # NW-RQK Slope: Use rolling z-score normalization
                    normalizer = self.normalizers.get('nwrqk_slope')
                    if normalizer and normalizer.n_samples > 10:
                        processed[i] = normalizer.normalize_zscore(value)
                        processed[i] = np.clip(processed[i], -2, 2) / 2  # Scale to [-1, 1]
                    else:
                        # During warmup, use simple scaling
                        processed[i] = np.tanh(value * 10)  # Assumes slope ~0.1 is significant
                        
                elif feature_name == 'lvn_distance_points':
                    # LVN Distance: Convert points to percentage and scale
                    if self.current_price and self.current_price > 0:
                        lvn_distance_pct = (value / self.current_price) * 100
                        # Use exponential decay - closer LVNs are more important
                        processed[i] = np.exp(-lvn_distance_pct)
                    else:
                        processed[i] = 0.0
                        
                elif feature_name == 'lvn_nearest_strength':
                    # LVN Strength: Scale from [0,100] to [0,1]
                    processed[i] = min_max_scale(value, 0, 100, (0, 1))
                    
                elif feature_name == 'time_hour_sin' or feature_name == 'time_hour_cos':
                    # Time features: Cyclical encoding
                    hour = value  # The raw value is the hour
                    hour_sin, hour_cos = cyclical_encode(hour, 24)
                    if feature_name == 'time_hour_sin':
                        processed[i] = hour_sin
                    else:
                        processed[i] = hour_cos
                        
                else:
                    # Default processing for unknown features
                    # Try to use normalizer if available
                    normalizer = self.normalizers.get(feature_name)
                    if normalizer and normalizer.n_samples > 10:
                        processed[i] = normalizer.normalize_zscore(value)
                        processed[i] = np.clip(processed[i], -3, 3) / 3
                    else:
                        # Simple clipping for safety
                        processed[i] = np.clip(value, -10, 10) / 10
            
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
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get estimated feature importance for interpretation.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Default importance scores - can be overridden by config
        default_importance = {
            "mlmi_value": 0.20,
            "mlmi_signal": 0.15,
            "nwrqk_value": 0.15,
            "nwrqk_slope": 0.20,
            "lvn_distance_points": 0.10,
            "lvn_nearest_strength": 0.10,
            "time_hour_sin": 0.05,
            "time_hour_cos": 0.05
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
        
        # Check specific features if they exist
        for i, feature_name in enumerate(self.feature_names):
            value = features[i]
            
            if feature_name == 'mlmi_value':
                # Check MLMI value range
                if not 0 <= value <= 100:
                    self.logger.warning(f"MLMI value out of range: {value}")
                    return False
                    
            elif feature_name == 'mlmi_signal':
                # Check MLMI signal
                if value not in [-1, 0, 1]:
                    self.logger.warning(f"Invalid MLMI signal: {value}")
                    return False
                    
            elif feature_name == 'lvn_nearest_strength':
                # Check LVN strength range
                if not 0 <= value <= 100:
                    self.logger.warning(f"LVN strength out of range: {value}")
                    return False
                    
            elif feature_name in ['time_hour_sin', 'time_hour_cos']:
                # Time features will be validated after encoding
                continue
                
            # General validation
            if not np.isfinite(value):
                self.logger.warning(f"Non-finite value for {feature_name}: {value}")
                return False
        
        return True