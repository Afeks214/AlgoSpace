"""
Signal sequence tracking for synergy detection.

This module handles the tracking and validation of signal sequences
that form synergy patterns.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

import structlog

from .base import Signal

logger = structlog.get_logger()


@dataclass
class SignalSequence:
    """
    Tracks a sequence of signals for synergy pattern detection.
    
    Manages signal ordering, time window constraints, and direction consistency.
    """
    
    time_window_bars: int = 10  # Maximum bars for sequence completion
    bar_duration_minutes: int = 5  # Duration of each bar in minutes
    required_signals: int = 3  # Number of signals required for complete synergy
    
    # Internal state
    signals: List[Signal] = field(default_factory=list)
    start_time: Optional[datetime] = None
    _bars_since_start: int = 0
    
    def add_signal(self, signal: Signal) -> bool:
        """
        Add a signal to the sequence.
        
        Args:
            signal: The signal to add
            
        Returns:
            True if signal was added, False if sequence was reset
        """
        # First signal starts the sequence
        if not self.signals:
            self.start_time = signal.timestamp
            self._bars_since_start = 0
            self.signals.append(signal)
            logger.info(f"Started new signal sequence signal_type={signal.signal_type} direction={signal.direction}")
            return True
        
        # Check if sequence has expired
        if self._is_expired(signal.timestamp):
            logger.info(f"Signal sequence expired bars_elapsed={self._bars_since_start} max_bars={self.time_window_bars}")
            self.reset()
            # Start new sequence with this signal
            self.start_time = signal.timestamp
            self.signals.append(signal)
            return False
        
        # Check direction consistency
        if not self._is_direction_consistent(signal):
            logger.info(f"Direction inconsistency detected, resetting sequence existing_direction={self.signals[0].direction} new_direction={signal.direction}")
            self.reset()
            # Start new sequence with this signal
            self.start_time = signal.timestamp
            self.signals.append(signal)
            return False
        
        # Check for duplicate signal types (can't have same signal twice)
        if self._has_signal_type(signal.signal_type):
            logger.debug(f"Duplicate signal type in sequence signal_type={signal.signal_type}")
            # Don't reset, just ignore this signal
            return True
        
        # All checks passed, add signal
        self.signals.append(signal)
        logger.info(f"Added signal to sequence signal_type={signal.signal_type} sequence_length={len(self.signals)} sequence={[s.signal_type for s in self.signals]}")
        return True
    
    def _is_expired(self, current_time: datetime) -> bool:
        """Check if sequence has exceeded time window."""
        if not self.start_time:
            return False
            
        time_diff = current_time - self.start_time
        self._bars_since_start = int(
            time_diff.total_seconds() / (self.bar_duration_minutes * 60)
        )
        
        return self._bars_since_start > self.time_window_bars
    
    def _is_direction_consistent(self, signal: Signal) -> bool:
        """Check if signal direction matches sequence direction."""
        if not self.signals:
            return True
            
        return signal.direction == self.signals[0].direction
    
    def _has_signal_type(self, signal_type: str) -> bool:
        """Check if signal type already exists in sequence."""
        return any(s.signal_type == signal_type for s in self.signals)
    
    def is_complete(self) -> bool:
        """Check if sequence has required number of signals (complete synergy)."""
        return len(self.signals) == self.required_signals
    
    def get_pattern(self) -> Optional[tuple]:
        """Get the signal pattern as a tuple."""
        if not self.signals:
            return None
        return tuple(s.signal_type for s in self.signals)
    
    def get_direction(self) -> Optional[int]:
        """Get the sequence direction."""
        if not self.signals:
            return None
        return self.signals[0].direction
    
    def get_completion_time(self) -> Optional[datetime]:
        """Get the completion timestamp."""
        if not self.is_complete():
            return None
        return self.signals[-1].timestamp
    
    def get_bars_to_complete(self) -> int:
        """Get number of bars taken to complete sequence."""
        if not self.signals or len(self.signals) < 2:
            return 0
            
        first_time = self.signals[0].timestamp
        last_time = self.signals[-1].timestamp
        time_diff = last_time - first_time
        
        return int(
            time_diff.total_seconds() / (self.bar_duration_minutes * 60)
        ) + 1  # +1 because we count inclusive
    
    def reset(self):
        """Reset the sequence."""
        self.signals = []
        self.start_time = None
        self._bars_since_start = 0
    
    def get_state(self) -> Dict[str, Any]:
        """Get sequence state for monitoring."""
        return {
            'active': len(self.signals) > 0,
            'signal_count': len(self.signals),
            'signals': [s.signal_type for s in self.signals],
            'direction': self.get_direction(),
            'bars_elapsed': self._bars_since_start,
            'start_time': self.start_time.isoformat() if self.start_time else None
        }


class CooldownTracker:
    """Tracks cooldown periods after synergy detection."""
    
    def __init__(self, cooldown_bars: int, bar_duration_minutes: int = 5):
        """
        Initialize cooldown tracker.
        
        Args:
            cooldown_bars: Number of bars to wait after detection
            bar_duration_minutes: Duration of each bar
        """
        self.cooldown_bars = cooldown_bars
        self.bar_duration_minutes = bar_duration_minutes
        self.last_synergy_time: Optional[datetime] = None
        self.bars_since_synergy = 0
    
    def start_cooldown(self, timestamp: datetime):
        """Start a new cooldown period."""
        self.last_synergy_time = timestamp
        self.bars_since_synergy = 0
        logger.info(f"Cooldown started cooldown_bars={self.cooldown_bars} timestamp={timestamp.isoformat()}")
    
    def update(self, current_time: datetime):
        """Update cooldown state with current time."""
        if self.last_synergy_time is None:
            return
            
        time_diff = current_time - self.last_synergy_time
        self.bars_since_synergy = int(
            time_diff.total_seconds() / (self.bar_duration_minutes * 60)
        )
    
    def is_active(self) -> bool:
        """Check if cooldown is currently active."""
        if self.last_synergy_time is None:
            return False
            
        return self.bars_since_synergy < self.cooldown_bars
    
    def can_emit(self) -> bool:
        """Check if we can emit a new synergy (cooldown expired)."""
        return not self.is_active()
    
    def get_remaining_bars(self) -> int:
        """Get number of bars remaining in cooldown."""
        if not self.is_active():
            return 0
            
        return self.cooldown_bars - self.bars_since_synergy
    
    def get_state(self) -> Dict[str, Any]:
        """Get cooldown state for monitoring."""
        return {
            'active': self.is_active(),
            'bars_since_synergy': self.bars_since_synergy,
            'remaining_bars': self.get_remaining_bars(),
            'last_synergy_time': (
                self.last_synergy_time.isoformat() 
                if self.last_synergy_time else None
            )
        }