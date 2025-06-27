"""
MatrixAssemblerRegime - Market Regime Detection Matrix

This assembler creates a matrix capturing market behavior using 30-minute bars.
It focuses on Market Microstructure Dynamics (MMD) features and additional 
regime indicators for the Regime Detection Engine.

Configuration is now driven externally from settings.yaml.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from collections import deque

from .base import BaseMatrixAssembler
from .normalizers import (
    z_score_normalize, robust_percentile_scale, safe_divide,
    min_max_scale
)


class MatrixAssemblerRegime(BaseMatrixAssembler):
    """
    Regime detection input matrix.
    
    All configuration including window_size and features list
    is now provided externally via the config parameter.
    
    This class contains only the specialized preprocessing and
    normalization logic for regime detection features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MatrixAssemblerRegime with external configuration.
        
        Args:
            config: Configuration dictionary from settings.yaml
        """
        # Call parent constructor with config
        super().__init__(config)
        
        # Determine MMD dimension from features
        self.mmd_dimension = 0
        for feature in self.feature_names:
            if feature.startswith('mmd_feature_'):
                self.mmd_dimension += 1
        
        # If no individual MMD features, check for array feature
        if self.mmd_dimension == 0 and 'mmd_features' in self.feature_names:
            # MMD dimension will be determined dynamically
            self.mmd_dimension = -1  # Flag for dynamic dimension
        
        # Price and volume history for calculations
        self.price_history = deque(maxlen=31)  # For 30-period volatility
        self.volume_history = deque(maxlen=20)  # For volume profile
        self.price_velocity_history = deque(maxlen=3)  # For acceleration
        
        # Volatility tracking
        self.volatility_ema = None
        self.volatility_ema_alpha = 0.05
        
        # Volume profile statistics
        self.volume_mean = None
        self.volume_std = None
        
        # Percentile trackers for robust scaling
        self.percentile_trackers = {}
        
        self.logger.info(
            f"MatrixAssemblerRegime initialized with {self.n_features} features, "
            f"MMD dimension: {self.mmd_dimension if self.mmd_dimension > 0 else 'dynamic'}"
        )
    
    def extract_features(self, feature_store: Dict[str, Any]) -> Optional[List[float]]:
        """
        Extract regime detection features from feature store with custom logic.
        
        Returns None to trigger safe extraction for most features.
        
        Args:
            feature_store: Complete feature dictionary from IndicatorEngine
            
        Returns:
            List of raw feature values or None to use default extraction
        """
        # Update price and volume history
        current_price = feature_store.get('current_price', 0)
        if current_price == 0:
            current_price = feature_store.get('close', 0)
            
        if current_price > 0:
            self.price_history.append(current_price)
        
        current_volume = feature_store.get('current_volume', 0)
        if current_volume == 0:
            current_volume = feature_store.get('volume', 0)
            
        if current_volume >= 0:
            self.volume_history.append(current_volume)
        
        # Check if we need custom calculations
        needs_custom = False
        custom_features = ['volatility_30', 'volume_profile_skew', 'price_acceleration']
        
        for feature in custom_features:
            if feature in self.feature_names:
                needs_custom = True
                break
                
        # Handle MMD features array
        if 'mmd_features' in self.feature_names:
            needs_custom = True
            
        if not needs_custom:
            # Use default safe extraction
            return None
            
        # Build features list with custom calculations
        features = []
        
        for feature_name in self.feature_names:
            if feature_name == 'volatility_30':
                volatility = self._calculate_volatility()
                features.append(volatility)
                
            elif feature_name == 'volume_profile_skew':
                skew = self._calculate_volume_skew()
                features.append(skew)
                
            elif feature_name == 'price_acceleration':
                acceleration = self._calculate_price_acceleration()
                features.append(acceleration)
                
            elif feature_name == 'mmd_features':
                # Handle MMD features array
                mmd_array = feature_store.get('mmd_features', [])
                if isinstance(mmd_array, np.ndarray):
                    mmd_array = mmd_array.tolist()
                # Flatten the array into individual features
                features.extend(mmd_array)
                # Update dynamic dimension if needed
                if self.mmd_dimension == -1:
                    self.mmd_dimension = len(mmd_array)
                    
            elif feature_name.startswith('mmd_feature_'):
                # Individual MMD feature
                idx = int(feature_name.split('_')[-1])
                mmd_array = feature_store.get('mmd_features', [])
                if isinstance(mmd_array, (list, np.ndarray)) and idx < len(mmd_array):
                    features.append(float(mmd_array[idx]))
                else:
                    features.append(0.0)
                    
            else:
                # Use safe extraction
                value = feature_store.get(feature_name, 0.0)
                features.append(value)
        
        return features
    
    def _calculate_volatility(self) -> float:
        """
        Calculate 30-period volatility (standard deviation of returns).
        
        Returns:
            Volatility measure or 0.0 if insufficient data
        """
        if len(self.price_history) < 2:
            return 0.0
        
        # Calculate returns
        prices = np.array(list(self.price_history))
        returns = np.diff(prices) / prices[:-1]
        
        # Remove any non-finite values
        returns = returns[np.isfinite(returns)]
        
        if len(returns) == 0:
            return 0.0
        
        # Calculate volatility
        volatility = np.std(returns) * 100  # Convert to percentage
        
        # Update EMA
        if self.volatility_ema is None:
            self.volatility_ema = volatility
        else:
            self.volatility_ema += self.volatility_ema_alpha * (volatility - self.volatility_ema)
        
        return volatility
    
    def _calculate_volume_skew(self) -> float:
        """
        Calculate skewness of volume distribution.
        
        Positive skew indicates occasional volume spikes,
        negative skew indicates consistent high volume with occasional lulls.
        
        Returns:
            Skewness measure or 0.0 if insufficient data
        """
        if len(self.volume_history) < 3:
            return 0.0
        
        volumes = np.array(list(self.volume_history))
        
        # Calculate mean and std
        mean = np.mean(volumes)
        std = np.std(volumes)
        
        if std == 0 or mean == 0:
            return 0.0
        
        # Update rolling statistics
        if self.volume_mean is None:
            self.volume_mean = mean
            self.volume_std = std
        else:
            alpha = 0.05
            self.volume_mean += alpha * (mean - self.volume_mean)
            self.volume_std += alpha * (std - self.volume_std)
        
        # Calculate skewness
        skewness = np.mean(((volumes - mean) / std) ** 3)
        
        # Clip extreme values
        skewness = np.clip(skewness, -3.0, 3.0)
        
        return skewness
    
    def _calculate_price_acceleration(self) -> float:
        """
        Calculate price acceleration (second derivative).
        
        Positive acceleration indicates increasing momentum,
        negative indicates decreasing momentum.
        
        Returns:
            Acceleration measure or 0.0 if insufficient data
        """
        if len(self.price_history) < 3:
            return 0.0
        
        # Get last 3 prices
        prices = list(self.price_history)[-3:]
        
        # Calculate first derivatives (velocity)
        velocity1 = (prices[1] - prices[0]) / prices[0] * 100
        velocity2 = (prices[2] - prices[1]) / prices[1] * 100
        
        # Store velocities
        self.price_velocity_history.append(velocity2)
        
        # Calculate acceleration
        if len(self.price_velocity_history) >= 2:
            acceleration = velocity2 - velocity1
        else:
            acceleration = 0.0
        
        # Clip extreme values
        acceleration = np.clip(acceleration, -5.0, 5.0)
        
        return acceleration
    
    def preprocess_features(
        self, 
        raw_features: List[float], 
        feature_store: Dict[str, Any]
    ) -> np.ndarray:
        """
        Preprocess features for neural network input.
        
        MMD features are typically already normalized, others need specific handling.
        """
        processed = np.zeros(len(raw_features), dtype=np.float32)
        
        try:
            # Process each feature based on its name
            for i, (feature_name, value) in enumerate(zip(self.feature_names, raw_features)):
                
                # Handle MMD features
                if feature_name.startswith('mmd_feature_') or feature_name == 'mmd_features':
                    # MMD features should already be normalized
                    # Just ensure they're not extreme
                    processed[i] = np.clip(value, -3.0, 3.0)
                    
                elif feature_name == 'volatility_30':
                    # Use rolling normalization
                    normalizer = self.normalizers.get('volatility_30')
                    if normalizer and normalizer.n_samples > 10:
                        processed[i] = normalizer.normalize_zscore(value)
                        processed[i] = np.clip(processed[i], -2, 2)
                    else:
                        # During warmup, assume 1% daily vol is normal
                        processed[i] = np.tanh(value / 1.0)
                        
                elif feature_name == 'volume_profile_skew':
                    # Skew is already in [-3, 3] range, scale to [-1, 1]
                    processed[i] = value / 3.0
                    
                elif feature_name == 'price_acceleration':
                    # Acceleration is in [-5, 5] range, scale to [-1, 1]
                    processed[i] = value / 5.0
                    
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
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Feature preprocessing failed: {e}")
            return np.zeros(len(raw_features), dtype=np.float32)
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """
        Get summary of current regime characteristics.
        
        Returns:
            Dictionary with regime statistics
        """
        with self._lock:
            if not self.is_ready():
                return {"status": "not_ready"}
            
            matrix = self.get_matrix()
            if matrix is None:
                return {"status": "no_data"}
            
            # Analyze recent regime patterns
            recent_data = matrix[-20:] if len(matrix) >= 20 else matrix
            
            # Find feature indices
            vol_idx = None
            skew_idx = None
            accel_idx = None
            
            for i, feature_name in enumerate(self.feature_names):
                if feature_name == 'volatility_30':
                    vol_idx = i
                elif feature_name == 'volume_profile_skew':
                    skew_idx = i
                elif feature_name == 'price_acceleration':
                    accel_idx = i
            
            # Calculate statistics
            result = {
                "status": "ready",
                "regime_indicators": {}
            }
            
            if vol_idx is not None:
                result["regime_indicators"]["avg_volatility"] = float(np.mean(recent_data[:, vol_idx]))
                
            if skew_idx is not None:
                result["regime_indicators"]["avg_volume_skew"] = float(np.mean(recent_data[:, skew_idx]))
                
            if accel_idx is not None:
                result["regime_indicators"]["avg_acceleration"] = float(np.mean(recent_data[:, accel_idx]))
            
            # Regime stability (how much features are changing)
            feature_changes = np.diff(recent_data, axis=0)
            stability_score = 1.0 - np.mean(np.abs(feature_changes))
            result["regime_indicators"]["stability_score"] = float(stability_score)
            
            # Add interpretation
            result["interpretation"] = self._interpret_regime(
                result["regime_indicators"].get("avg_volatility", 0),
                result["regime_indicators"].get("avg_volume_skew", 0),
                result["regime_indicators"].get("avg_acceleration", 0),
                stability_score
            )
            
            return result
    
    def _interpret_regime(
        self, 
        volatility: float, 
        volume_skew: float, 
        acceleration: float,
        stability: float
    ) -> str:
        """
        Provide human-readable regime interpretation.
        
        Args:
            volatility: Average volatility (normalized)
            volume_skew: Average volume skew
            acceleration: Average price acceleration
            stability: Regime stability score
            
        Returns:
            String description of regime
        """
        interpretations = []
        
        # Volatility interpretation
        if volatility > 0.5:
            interpretations.append("High volatility")
        elif volatility < -0.5:
            interpretations.append("Low volatility")
        else:
            interpretations.append("Normal volatility")
        
        # Volume pattern
        if volume_skew > 0.3:
            interpretations.append("sporadic volume spikes")
        elif volume_skew < -0.3:
            interpretations.append("consistent high volume")
        else:
            interpretations.append("balanced volume")
        
        # Momentum
        if acceleration > 0.2:
            interpretations.append("accelerating trend")
        elif acceleration < -0.2:
            interpretations.append("decelerating trend")
        else:
            interpretations.append("steady momentum")
        
        # Stability
        if stability > 0.8:
            interpretations.append("stable regime")
        elif stability < 0.5:
            interpretations.append("transitioning regime")
        else:
            interpretations.append("moderately stable")
        
        return ", ".join(interpretations)
    
    def validate_features(self, features: List[float]) -> bool:
        """
        Validate that features are within expected ranges.
        
        Args:
            features: Raw feature values
            
        Returns:
            True if all features are valid
        """
        if len(features) != self.n_features:
            self.logger.warning(
                f"Feature count mismatch: expected {self.n_features}, "
                f"got {len(features)}"
            )
            return False
        
        # Check for non-finite values
        if not all(np.isfinite(f) for f in features):
            self.logger.warning("Non-finite values in features")
            return False
        
        return True