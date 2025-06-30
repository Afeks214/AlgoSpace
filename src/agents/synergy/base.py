"""
Base classes for the SynergyDetector component.

This module provides the abstract base classes for pattern detection
and synergy identification in the AlgoSpace trading system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass
import time

import structlog

logger = structlog.get_logger()


@dataclass
class Signal:
    """Represents a detected trading signal."""
    signal_type: str  # 'mlmi', 'nwrqk', 'fvg'
    direction: int    # 1 (bullish) or -1 (bearish)
    timestamp: datetime
    value: float
    strength: float   # Signal strength (0-1)
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SynergyPattern:
    """Represents a completed synergy pattern."""
    synergy_type: str  # 'TYPE_1', 'TYPE_2', 'TYPE_3', 'TYPE_4'
    direction: int     # 1 (long) or -1 (short)
    signals: List[Signal]
    completion_time: datetime
    bars_to_complete: int
    confidence: float = 1.0  # Always 1.0 for hard-coded rules
    
    def get_signal_sequence(self) -> Tuple[str, ...]:
        """Return the sequence of signal types."""
        return tuple(s.signal_type for s in self.signals)


class BasePatternDetector(ABC):
    """
    Abstract base class for pattern detection.
    
    Provides the interface for detecting specific trading patterns
    from market indicators and features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pattern detector.
        
        Args:
            config: Configuration dictionary with detection parameters
        """
        self.config = config
        self._last_detection_time = None
        self._performance_metrics = {
            'total_detections': 0,
            'avg_detection_time_ms': 0.0,
            'max_detection_time_ms': 0.0
        }
    
    @abstractmethod
    def detect_pattern(self, features: Dict[str, Any]) -> Optional[Signal]:
        """
        Detect a specific pattern in the features.
        
        Args:
            features: Feature Store snapshot
            
        Returns:
            Detected signal if pattern found, None otherwise
        """
        pass
    
    @abstractmethod
    def validate_signal(self, signal: Signal, features: Dict[str, Any]) -> bool:
        """
        Validate that a signal meets all requirements.
        
        Args:
            signal: The signal to validate
            features: Current feature store state
            
        Returns:
            True if signal is valid, False otherwise
        """
        pass
    
    def measure_performance(func):
        """Decorator to measure detection performance."""
        def wrapper(self, *args, **kwargs):
            start_time = time.perf_counter()
            result = func(self, *args, **kwargs)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            # Update performance metrics
            self._performance_metrics['total_detections'] += 1
            n = self._performance_metrics['total_detections']
            avg = self._performance_metrics['avg_detection_time_ms']
            self._performance_metrics['avg_detection_time_ms'] = (
                (avg * (n - 1) + elapsed_ms) / n
            )
            self._performance_metrics['max_detection_time_ms'] = max(
                self._performance_metrics['max_detection_time_ms'],
                elapsed_ms
            )
            
            return result
        return wrapper
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get pattern detector performance metrics."""
        return self._performance_metrics.copy()


class BaseSynergyDetector(ABC):
    """
    Abstract base class for synergy detection.
    
    Provides the framework for detecting complex multi-signal
    synergy patterns that form valid trading setups.
    """
    
    # Define valid synergy patterns
    SYNERGY_PATTERNS = {
        ('mlmi', 'nwrqk', 'fvg'): 'TYPE_1',
        ('mlmi', 'fvg', 'nwrqk'): 'TYPE_2',
        ('nwrqk', 'fvg', 'mlmi'): 'TYPE_3',
        ('nwrqk', 'mlmi', 'fvg'): 'TYPE_4'
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the synergy detector.
        
        Args:
            config: Configuration with time windows, thresholds, etc.
        """
        self.config = config
        self.time_window = config.get('time_window_bars', 10)  # bars
        self.cooldown_bars = config.get('cooldown_bars', 5)
        
        # State tracking
        self.signal_sequence = []
        self.sequence_start_time = None
        self.last_synergy_time = None
        self.bars_processed_since_synergy = 0
        
        logger.info(
            "Initialized synergy detector",
            time_window=self.time_window,
            cooldown_bars=self.cooldown_bars
        )
    
    @abstractmethod
    def process_features(self, features: Dict[str, Any], timestamp: datetime) -> Optional[SynergyPattern]:
        """
        Process features to detect synergy patterns.
        
        Args:
            features: Complete Feature Store snapshot
            timestamp: Current bar timestamp
            
        Returns:
            Detected synergy pattern if found, None otherwise
        """
        pass
    
    def _is_sequence_valid(self) -> bool:
        """Check if current signal sequence is still valid."""
        if not self.signal_sequence:
            return True
            
        # Check time window constraint
        if self.sequence_start_time:
            # Assuming 5-minute bars
            time_diff = datetime.now() - self.sequence_start_time
            bars_elapsed = time_diff.total_seconds() / 300  # 300 seconds = 5 minutes
            if bars_elapsed > self.time_window:
                return False
        
        # Check direction consistency
        if len(self.signal_sequence) > 1:
            directions = [s.direction for s in self.signal_sequence]
            if not all(d == directions[0] for d in directions):
                return False
                
        return True
    
    def _check_synergy_completion(self) -> Optional[SynergyPattern]:
        """Check if current sequence forms a valid synergy."""
        if len(self.signal_sequence) < 3:
            return None
            
        # Get signal sequence
        sequence = tuple(s.signal_type for s in self.signal_sequence)
        
        # Check if it's a valid synergy pattern
        synergy_type = self.SYNERGY_PATTERNS.get(sequence)
        if not synergy_type:
            return None
            
        # All checks passed - we have a synergy!
        completion_time = self.signal_sequence[-1].timestamp
        bars_to_complete = len(self.signal_sequence)
        
        return SynergyPattern(
            synergy_type=synergy_type,
            direction=self.signal_sequence[0].direction,
            signals=self.signal_sequence.copy(),
            completion_time=completion_time,
            bars_to_complete=bars_to_complete
        )
    
    def _can_emit_synergy(self) -> bool:
        """Check if we're allowed to emit a synergy (cooldown check)."""
        if self.last_synergy_time is None:
            return True
            
        return self.bars_processed_since_synergy >= self.cooldown_bars
    
    def reset_sequence(self):
        """Reset the signal sequence."""
        self.signal_sequence = []
        self.sequence_start_time = None
        logger.debug("Signal sequence reset")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current detector state for monitoring."""
        return {
            'active_signals': len(self.signal_sequence),
            'sequence': [s.signal_type for s in self.signal_sequence],
            'in_cooldown': not self._can_emit_synergy(),
            'bars_since_synergy': self.bars_processed_since_synergy
        }