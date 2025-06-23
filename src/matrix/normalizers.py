"""
Feature Normalization Utilities for Matrix Assemblers

This module provides robust normalization functions for transforming
raw indicator values into neural network-friendly ranges. All functions
include comprehensive error handling and edge case management.
"""

import numpy as np
from typing import Union, Tuple, Optional
import warnings


def z_score_normalize(
    value: Union[float, np.ndarray], 
    mean: float, 
    std: float,
    clip_range: Optional[Tuple[float, float]] = (-3.0, 3.0)
) -> Union[float, np.ndarray]:
    """
    Z-score normalization with outlier clipping.
    
    Args:
        value: Value(s) to normalize
        mean: Population mean
        std: Population standard deviation
        clip_range: Optional range to clip normalized values
        
    Returns:
        Normalized value(s) in approximately [-1, 1] range
    """
    if std == 0 or np.isclose(std, 0, atol=1e-10):
        warnings.warn("Standard deviation is zero, returning 0")
        return np.zeros_like(value) if isinstance(value, np.ndarray) else 0.0
    
    normalized = (value - mean) / std
    
    if clip_range is not None:
        normalized = np.clip(normalized, clip_range[0], clip_range[1])
    
    return normalized


def min_max_scale(
    value: Union[float, np.ndarray],
    min_val: float,
    max_val: float,
    target_range: Tuple[float, float] = (-1.0, 1.0)
) -> Union[float, np.ndarray]:
    """
    Min-max scaling to target range.
    
    Args:
        value: Value(s) to scale
        min_val: Minimum value of the range
        max_val: Maximum value of the range
        target_range: Target range for scaling
        
    Returns:
        Scaled value(s) in target range
    """
    if np.isclose(max_val, min_val, atol=1e-10):
        warnings.warn("Min and max values are equal, returning midpoint of target range")
        midpoint = (target_range[0] + target_range[1]) / 2
        return np.full_like(value, midpoint) if isinstance(value, np.ndarray) else midpoint
    
    # Scale to [0, 1]
    scaled = (value - min_val) / (max_val - min_val)
    
    # Scale to target range
    target_min, target_max = target_range
    scaled = scaled * (target_max - target_min) + target_min
    
    # Ensure within bounds (handles numerical precision issues)
    scaled = np.clip(scaled, target_min, target_max)
    
    return scaled


def cyclical_encode(
    value: Union[float, np.ndarray],
    max_value: float
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Cyclical encoding for periodic features (e.g., hour of day).
    
    Args:
        value: Value(s) to encode
        max_value: Maximum value of the cycle (e.g., 24 for hours)
        
    Returns:
        Tuple of (sin_encoding, cos_encoding)
    """
    if max_value <= 0:
        raise ValueError(f"max_value must be positive, got {max_value}")
    
    # Normalize to [0, 2π]
    angle = 2 * np.pi * value / max_value
    
    return np.sin(angle), np.cos(angle)


def percentage_from_price(
    value: Union[float, np.ndarray],
    reference_price: float,
    clip_pct: float = 10.0
) -> Union[float, np.ndarray]:
    """
    Convert price to percentage distance from reference.
    
    Args:
        value: Price value(s)
        reference_price: Reference price (e.g., current price)
        clip_pct: Maximum percentage to clip at
        
    Returns:
        Percentage distance from reference
    """
    if reference_price <= 0:
        raise ValueError(f"reference_price must be positive, got {reference_price}")
    
    pct_diff = ((value - reference_price) / reference_price) * 100
    
    # Clip extreme values
    pct_diff = np.clip(pct_diff, -clip_pct, clip_pct)
    
    return pct_diff


def exponential_decay(
    value: Union[float, np.ndarray],
    decay_rate: float = 0.1
) -> Union[float, np.ndarray]:
    """
    Apply exponential decay transformation.
    
    Args:
        value: Value(s) to transform (e.g., age in bars)
        decay_rate: Decay rate parameter
        
    Returns:
        Decayed value(s) in range (0, 1]
    """
    if decay_rate <= 0:
        raise ValueError(f"decay_rate must be positive, got {decay_rate}")
    
    return np.exp(-decay_rate * np.abs(value))


def log_transform(
    value: Union[float, np.ndarray],
    epsilon: float = 1e-8
) -> Union[float, np.ndarray]:
    """
    Safe logarithmic transformation.
    
    Args:
        value: Value(s) to transform
        epsilon: Small constant to avoid log(0)
        
    Returns:
        Log-transformed value(s)
    """
    # Ensure non-negative
    safe_value = np.maximum(value, epsilon)
    
    # Use log1p for values close to 0
    return np.log1p(safe_value - 1)


def robust_percentile_scale(
    value: Union[float, np.ndarray],
    q25: float,
    q75: float,
    clip_range: Tuple[float, float] = (-2.0, 2.0)
) -> Union[float, np.ndarray]:
    """
    Robust scaling using interquartile range.
    
    Args:
        value: Value(s) to scale
        q25: 25th percentile
        q75: 75th percentile
        clip_range: Range to clip scaled values
        
    Returns:
        Robustly scaled value(s)
    """
    iqr = q75 - q25
    
    if iqr <= 0 or np.isclose(iqr, 0, atol=1e-10):
        warnings.warn("IQR is zero, returning 0")
        return np.zeros_like(value) if isinstance(value, np.ndarray) else 0.0
    
    median = (q25 + q75) / 2
    scaled = (value - median) / iqr
    
    if clip_range is not None:
        scaled = np.clip(scaled, clip_range[0], clip_range[1])
    
    return scaled


def safe_divide(
    numerator: Union[float, np.ndarray],
    denominator: Union[float, np.ndarray],
    default: float = 0.0
) -> Union[float, np.ndarray]:
    """
    Safe division with zero handling.
    
    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)
        default: Default value when denominator is zero
        
    Returns:
        Result of division or default value
    """
    if isinstance(denominator, np.ndarray):
        result = np.where(
            np.abs(denominator) > 1e-10,
            numerator / denominator,
            default
        )
    else:
        result = numerator / denominator if abs(denominator) > 1e-10 else default
    
    return result


class RollingNormalizer:
    """
    Maintains rolling statistics for online normalization.
    
    This class efficiently tracks mean, std, min, max, and percentiles
    using exponential moving averages for adaptive normalization.
    """
    
    def __init__(self, alpha: float = 0.01, warmup_samples: int = 100):
        """
        Initialize rolling normalizer.
        
        Args:
            alpha: EMA decay factor (smaller = slower adaptation)
            warmup_samples: Number of samples before statistics stabilize
        """
        self.alpha = alpha
        self.warmup_samples = warmup_samples
        self.n_samples = 0
        
        # Rolling statistics
        self.mean = 0.0
        self.variance = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        
        # Percentile tracking (approximate)
        self.q25 = 0.0
        self.q50 = 0.0
        self.q75 = 0.0
        
    def update(self, value: float) -> None:
        """Update rolling statistics with new value."""
        self.n_samples += 1
        
        # Update min/max
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        
        if self.n_samples == 1:
            # First sample
            self.mean = value
            self.variance = 0.0
            self.q25 = self.q50 = self.q75 = value
        else:
            # Exponential moving average
            alpha = self.alpha if self.n_samples > self.warmup_samples else 1.0 / self.n_samples
            
            # Update mean
            delta = value - self.mean
            self.mean += alpha * delta
            
            # Update variance (Welford's online algorithm adapted for EMA)
            self.variance = (1 - alpha) * (self.variance + alpha * delta * delta)
            
            # Update percentiles (P-Square algorithm approximation)
            if value < self.q25:
                self.q25 += alpha * (value - self.q25) * 0.25
            elif value < self.q50:
                self.q50 += alpha * (value - self.q50) * 0.50
                self.q25 += alpha * (value - self.q25) * 0.75
            elif value < self.q75:
                self.q75 += alpha * (value - self.q75) * 0.75
                self.q50 += alpha * (value - self.q50) * 0.50
            else:
                self.q75 += alpha * (value - self.q75) * 0.25
    
    @property
    def std(self) -> float:
        """Get current standard deviation."""
        return np.sqrt(max(self.variance, 0))
    
    def normalize_zscore(self, value: float) -> float:
        """Normalize using rolling z-score."""
        if self.n_samples < 2:
            return 0.0
        return z_score_normalize(value, self.mean, self.std)
    
    def normalize_minmax(self, value: float, target_range: Tuple[float, float] = (-1, 1)) -> float:
        """Normalize using rolling min-max."""
        if self.n_samples < 2:
            return (target_range[0] + target_range[1]) / 2
        return min_max_scale(value, self.min_val, self.max_val, target_range)
    
    def normalize_robust(self, value: float) -> float:
        """Normalize using rolling IQR."""
        if self.n_samples < self.warmup_samples:
            return 0.0
        return robust_percentile_scale(value, self.q25, self.q75)