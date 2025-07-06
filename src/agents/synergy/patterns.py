"""
Pattern detection implementations for specific indicators.

This module contains the concrete implementations for detecting
MLMI, NW-RQK, and FVG patterns according to the trading strategy.
"""

from typing import Dict, Any, Optional
from datetime import datetime

import structlog

from .base import BasePatternDetector, Signal

logger = structlog.get_logger()


class MLMIPatternDetector(BasePatternDetector):
    """
    Detects MLMI (Machine Learning Market Index) patterns.
    
    MLMI activates on crossover events with sufficient signal strength.
    The signal must deviate from the neutral line (50) by the threshold amount.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize MLMI detector with threshold configuration."""
        super().__init__(config)
        self.threshold = config.get('mlmi_threshold', 0.5)
        self.neutral_line = config.get('mlmi_neutral_line', 50)
        self.scaling_factor = config.get('mlmi_scaling_factor', 50)
        self.max_strength = config.get('mlmi_max_strength', 1.0)
        self._last_mlmi_value = None
        
    @BasePatternDetector.measure_performance
    def detect_pattern(self, features: Dict[str, Any]) -> Optional[Signal]:
        """
        Detect MLMI crossover pattern.
        
        Args:
            features: Feature store containing 'mlmi_signal' and 'mlmi_value'
            
        Returns:
            Signal if pattern detected, None otherwise
        """
        # Extract MLMI features with configurable defaults
        defaults = self.config.get('defaults', {})
        mlmi_signal = features.get('mlmi_signal', 0)
        mlmi_value = features.get('mlmi_value', defaults.get('mlmi_value', self.neutral_line))
        timestamp = features.get('timestamp', datetime.now())
        
        # No signal if no crossover
        if mlmi_signal == 0:
            return None
        
        # Check if signal strength meets threshold
        # MLMI ranges from 0-100, neutral at configured line
        deviation_from_neutral = abs(mlmi_value - self.neutral_line)
        required_deviation = self.threshold * self.scaling_factor  # threshold is 0-1, scale to configured factor
        
        if deviation_from_neutral < required_deviation:
            logger.debug(f"MLMI signal below threshold mlmi_value={mlmi_value} deviation={deviation_from_neutral} required={required_deviation}")
            return None
        
        # Calculate signal strength (0-1 scale)
        signal_strength = min(deviation_from_neutral / self.scaling_factor, self.max_strength)
        
        # Create signal
        signal = Signal(
            signal_type='mlmi',
            direction=mlmi_signal,  # 1 or -1
            timestamp=timestamp,
            value=mlmi_value,
            strength=signal_strength,
            metadata={
                'deviation_from_neutral': deviation_from_neutral,
                'threshold_used': self.threshold
            }
        )
        
        logger.info(f"MLMI pattern detected direction={mlmi_signal} value={mlmi_value} strength={signal_strength}")
        
        return signal
    
    def validate_signal(self, signal: Signal, features: Dict[str, Any]) -> bool:
        """Validate MLMI signal meets all requirements."""
        if signal.signal_type != 'mlmi':
            return False
            
        # Verify current MLMI state matches signal
        current_mlmi_signal = features.get('mlmi_signal', 0)
        if current_mlmi_signal != signal.direction:
            return False
            
        return True


class NWRQKPatternDetector(BasePatternDetector):
    """
    Detects NW-RQK (Nadaraya-Watson Rational Quadratic Kernel) patterns.
    
    NW-RQK activates on direction changes (slope changes) that exceed
    the configured threshold.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize NW-RQK detector with slope threshold."""
        super().__init__(config)
        self.threshold = config.get('nwrqk_threshold', 0.3)
        self.max_slope = config.get('nwrqk_max_slope', 2.0)
        self.max_strength = config.get('nwrqk_max_strength', 1.0)
        
    @BasePatternDetector.measure_performance
    def detect_pattern(self, features: Dict[str, Any]) -> Optional[Signal]:
        """
        Detect NW-RQK direction change pattern.
        
        Args:
            features: Feature store containing 'nwrqk_signal' and 'nwrqk_slope'
            
        Returns:
            Signal if pattern detected, None otherwise
        """
        # Extract NW-RQK features with configurable defaults
        defaults = self.config.get('defaults', {})
        nwrqk_signal = features.get('nwrqk_signal', 0)
        nwrqk_slope = features.get('nwrqk_slope', defaults.get('nwrqk_slope', 0.0))
        nwrqk_value = features.get('nwrqk_value', defaults.get('nwrqk_value', 0.0))
        timestamp = features.get('timestamp', datetime.now())
        
        # No signal if no direction change
        if nwrqk_signal == 0:
            return None
        
        # Check if slope magnitude meets threshold
        slope_magnitude = abs(nwrqk_slope)
        if slope_magnitude < self.threshold:
            logger.debug(f"NW-RQK slope below threshold slope={nwrqk_slope} magnitude={slope_magnitude} threshold={self.threshold}")
            return None
        
        # Calculate signal strength based on slope magnitude
        # Normalize to 0-1 scale using configured max slope
        signal_strength = min(slope_magnitude / self.max_slope, self.max_strength)
        
        # Create signal
        signal = Signal(
            signal_type='nwrqk',
            direction=nwrqk_signal,  # 1 or -1
            timestamp=timestamp,
            value=nwrqk_value,
            strength=signal_strength,
            metadata={
                'slope': nwrqk_slope,
                'slope_magnitude': slope_magnitude,
                'threshold_used': self.threshold
            }
        )
        
        logger.info(f"NW-RQK pattern detected direction={nwrqk_signal} slope={nwrqk_slope} strength={signal_strength}")
        
        return signal
    
    def validate_signal(self, signal: Signal, features: Dict[str, Any]) -> bool:
        """Validate NW-RQK signal meets all requirements."""
        if signal.signal_type != 'nwrqk':
            return False
            
        # Verify current slope still meets threshold
        current_slope = features.get('nwrqk_slope', 0.0)
        if abs(current_slope) < self.threshold:
            return False
            
        return True


class FVGPatternDetector(BasePatternDetector):
    """
    Detects FVG (Fair Value Gap) mitigation patterns.
    
    FVG activates when a significant gap is mitigated (filled).
    The gap size must exceed the minimum threshold.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize FVG detector with minimum gap size."""
        super().__init__(config)
        self.min_size = config.get('fvg_min_size', 0.001)  # 0.1% default
        self.max_gap_pct = config.get('fvg_max_gap_pct', 0.01)  # 1% max gap for normalization
        self.max_strength = config.get('fvg_max_strength', 1.0)
        
    @BasePatternDetector.measure_performance
    def detect_pattern(self, features: Dict[str, Any]) -> Optional[Signal]:
        """
        Detect FVG mitigation pattern.
        
        Args:
            features: Feature store containing FVG mitigation signals
            
        Returns:
            Signal if pattern detected, None otherwise
        """
        # Check for mitigation signal
        mitigation_signal = features.get('fvg_mitigation_signal', False)
        if not mitigation_signal:
            return None
        
        # Determine which type was mitigated and direction
        bullish_mitigated = features.get('fvg_bullish_mitigated', False)
        bearish_mitigated = features.get('fvg_bearish_mitigated', False)
        
        if not (bullish_mitigated or bearish_mitigated):
            return None
        
        # Get gap details with configurable defaults
        defaults = self.config.get('defaults', {})
        current_price = features.get('current_price', defaults.get('current_price', 0.0))
        timestamp = features.get('timestamp', datetime.now())
        
        # Determine direction based on mitigation type
        # Bullish gap mitigation suggests bullish continuation
        # Bearish gap mitigation suggests bearish continuation
        direction = 1 if bullish_mitigated else -1
        
        # Get mitigated gap size
        if bullish_mitigated:
            gap_size = features.get('fvg_bullish_size', 0.0)
            gap_level = features.get('fvg_bullish_level', current_price)
        else:
            gap_size = features.get('fvg_bearish_size', 0.0)
            gap_level = features.get('fvg_bearish_level', current_price)
        
        # Check minimum size requirement (as percentage)
        gap_size_pct = gap_size / current_price if current_price > 0 else 0
        if gap_size_pct < self.min_size:
            logger.debug(f"FVG gap size below threshold gap_size_pct={gap_size_pct} min_size={self.min_size}")
            return None
        
        # Signal strength based on gap size (larger gaps = stronger signal)
        # Normalize to 0-1 scale using configured max gap percentage
        signal_strength = min(gap_size_pct / self.max_gap_pct, self.max_strength)
        
        # Create signal
        signal = Signal(
            signal_type='fvg',
            direction=direction,
            timestamp=timestamp,
            value=gap_level,
            strength=signal_strength,
            metadata={
                'gap_type': 'bullish' if bullish_mitigated else 'bearish',
                'gap_size': gap_size,
                'gap_size_pct': gap_size_pct,
                'current_price': current_price,
                'min_size_threshold': self.min_size
            }
        )
        
        logger.info(f"FVG mitigation pattern detected gap_type={'bullish' if bullish_mitigated else 'bearish'} direction={direction} gap_size_pct={f"{gap_size_pct:.4f}"} strength={signal_strength}")
        
        return signal
    
    def validate_signal(self, signal: Signal, features: Dict[str, Any]) -> bool:
        """Validate FVG signal meets all requirements."""
        if signal.signal_type != 'fvg':
            return False
            
        # FVG signals are instantaneous events, no ongoing validation needed
        return True