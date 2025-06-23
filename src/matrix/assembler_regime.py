"""
MatrixAssemblerRegime - Market Regime Detection Matrix

This assembler creates a 96xN matrix capturing 48 hours of market behavior
using 30-minute bars. It focuses on Market Microstructure Dynamics (MMD)
features and additional regime indicators for the Regime Detection Engine.
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
    
    Features:
    - MMD feature array (variable length, typically 8-12 dimensions)
    - volatility_30: 30-period volatility measure
    - volume_profile_skew: Skewness of volume distribution
    - price_acceleration: Second derivative of price movement
    
    Matrix shape: (96, N) where N depends on MMD dimensionality
    Represents 48 hours of 30-minute bars for regime context
    """
    
    def __init__(self, name: str, kernel: Any):
        """Initialize MatrixAssemblerRegime."""
        # Load configuration
        config = kernel.config.get('matrix_assemblers', {}).get('regime', {})
        
        # Determine MMD dimension from indicator config
        mmd_config = kernel.config.get('indicators', {}).get('mmd', {})
        mmd_dimension = mmd_config.get('signature_degree', 3) * 2 + 2  # Default: 8
        
        # Build feature list dynamically based on MMD dimension
        mmd_features = [f'mmd_feature_{i}' for i in range(mmd_dimension)]
        
        # Set default configuration if not provided
        if not config:
            config = {
                'window_size': 96,  # 48 hours of 30-min bars
                'features': mmd_features + [
                    'volatility_30',
                    'volume_profile_skew',
                    'price_acceleration'
                ],
                'warmup_period': 30,  # Need history for volatility calc
                'feature_configs': {
                    'volatility_30': {'ema_alpha': 0.05},
                    'volume_profile_skew': {'ema_alpha': 0.02},
                    'price_acceleration': {'ema_alpha': 0.1}
                }
            }
        
        super().__init__(name, kernel, config)
        
        # MMD configuration
        self.mmd_dimension = mmd_dimension
        
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
            f"MatrixAssemblerRegime initialized with MMD dimension {mmd_dimension}, "
            f"total features: {self.n_features}"
        )
    
    def extract_features(self, feature_store: Dict[str, Any]) -> Optional[List[float]]:
        """
        Extract regime detection features from feature store.
        
        Args:
            feature_store: Complete feature dictionary from IndicatorEngine
            
        Returns:
            List of raw feature values or None if extraction fails
        """
        try:
            # Extract MMD features
            mmd_features = feature_store.get('mmd_features', [])
            if len(mmd_features) != self.mmd_dimension:
                self.logger.warning(
                    f"MMD dimension mismatch: expected {self.mmd_dimension}, "
                    f"got {len(mmd_features)}"
                )
                # Pad or truncate as needed
                if len(mmd_features) < self.mmd_dimension:
                    mmd_features.extend([0.0] * (self.mmd_dimension - len(mmd_features)))
                else:
                    mmd_features = mmd_features[:self.mmd_dimension]
            
            # Update price history
            current_price = feature_store.get('current_price', 0)
            if current_price > 0:
                self.price_history.append(current_price)
            
            # Update volume history
            current_volume = feature_store.get('current_volume', 0)
            if current_volume >= 0:
                self.volume_history.append(current_volume)
            
            # Calculate volatility
            volatility = self._calculate_volatility()
            
            # Calculate volume profile skew
            volume_skew = self._calculate_volume_skew()
            
            # Calculate price acceleration
            price_acceleration = self._calculate_price_acceleration()
            
            # Compile all features
            features = list(mmd_features) + [
                volatility,
                volume_skew,
                price_acceleration
            ]
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return None
    
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
        
        MMD features are already normalized, others need specific handling.
        """
        processed = np.zeros(len(raw_features), dtype=np.float32)
        
        try:
            # Process MMD features (already normalized from MMD engine)
            for i in range(self.mmd_dimension):
                if i < len(raw_features):
                    # MMD features should already be in reasonable range
                    # Just ensure they're not extreme
                    processed[i] = np.clip(raw_features[i], -3.0, 3.0)
            
            # Process volatility
            volatility_idx = self.mmd_dimension
            if volatility_idx < len(raw_features):
                volatility = raw_features[volatility_idx]
                # Use rolling normalization
                normalizer = self.normalizers.get('volatility_30')
                if normalizer and normalizer.n_samples > 10:
                    processed[volatility_idx] = normalizer.normalize_zscore(volatility)
                    processed[volatility_idx] = np.clip(processed[volatility_idx], -2, 2)
                else:
                    # During warmup, assume 1% daily vol is normal
                    processed[volatility_idx] = np.tanh(volatility / 1.0)
            
            # Process volume skew
            skew_idx = self.mmd_dimension + 1
            if skew_idx < len(raw_features):
                skew = raw_features[skew_idx]
                # Skew is already in [-3, 3] range, scale to [-1, 1]
                processed[skew_idx] = skew / 3.0
            
            # Process price acceleration
            accel_idx = self.mmd_dimension + 2
            if accel_idx < len(raw_features):
                acceleration = raw_features[accel_idx]
                # Acceleration is in [-5, 5] range, scale to [-1, 1]
                processed[accel_idx] = acceleration / 5.0
            
            # Final safety check
            if not np.all(np.isfinite(processed)):
                self.logger.warning("Non-finite values after preprocessing")
                processed = np.nan_to_num(processed, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Feature preprocessing failed: {e}")
            return np.zeros(len(raw_features), dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get human-readable feature names."""
        names = [f"MMD Feature {i}" for i in range(self.mmd_dimension)]
        names.extend([
            "Volatility (30-period)",
            "Volume Profile Skew",
            "Price Acceleration"
        ])
        return names
    
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
            
            # MMD summary (first N features)
            mmd_features = recent_data[:, :self.mmd_dimension]
            mmd_mean = np.mean(mmd_features, axis=0)
            mmd_std = np.std(mmd_features, axis=0)
            
            # Other features
            volatility_mean = np.mean(recent_data[:, self.mmd_dimension])
            volume_skew_mean = np.mean(recent_data[:, self.mmd_dimension + 1])
            accel_mean = np.mean(recent_data[:, self.mmd_dimension + 2])
            
            # Regime stability (how much features are changing)
            feature_changes = np.diff(recent_data, axis=0)
            stability_score = 1.0 - np.mean(np.abs(feature_changes))
            
            return {
                "status": "ready",
                "mmd_summary": {
                    "mean_vector": mmd_mean.tolist(),
                    "std_vector": mmd_std.tolist(),
                    "dimensionality": self.mmd_dimension
                },
                "regime_indicators": {
                    "avg_volatility": float(volatility_mean),
                    "avg_volume_skew": float(volume_skew_mean),
                    "avg_acceleration": float(accel_mean),
                    "stability_score": float(stability_score)
                },
                "interpretation": self._interpret_regime(
                    volatility_mean, volume_skew_mean, accel_mean, stability_score
                )
            }
    
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