"""
Unit tests for the SynergyDetector component.

Tests cover pattern detection, sequence tracking, cooldown management,
and integration with the event system.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import time

from src.agents.synergy import (
    SynergyDetector, Signal, SynergyPattern,
    MLMIPatternDetector, NWRQKPatternDetector, FVGPatternDetector,
    SignalSequence, CooldownTracker
)
from src.core.events import EventType, Event


class TestSignalSequence:
    """Test signal sequence tracking."""
    
    def test_add_first_signal(self):
        """Test adding the first signal starts a sequence."""
        sequence = SignalSequence(time_window_bars=10)
        signal = Signal(
            signal_type='mlmi',
            direction=1,
            timestamp=datetime.now(),
            value=65.0,
            strength=0.8
        )
        
        result = sequence.add_signal(signal)
        
        assert result is True
        assert len(sequence.signals) == 1
        assert sequence.start_time is not None
        assert sequence.get_pattern() == ('mlmi',)
        assert sequence.get_direction() == 1
    
    def test_direction_consistency(self):
        """Test that inconsistent directions reset sequence."""
        sequence = SignalSequence(time_window_bars=10)
        
        # Add first signal (bullish)
        signal1 = Signal('mlmi', 1, datetime.now(), 65.0, 0.8)
        sequence.add_signal(signal1)
        
        # Add opposing signal (bearish)
        signal2 = Signal('nwrqk', -1, datetime.now(), 100.0, 0.6)
        result = sequence.add_signal(signal2)
        
        assert result is False  # Sequence was reset
        assert len(sequence.signals) == 1  # Only new signal
        assert sequence.signals[0].signal_type == 'nwrqk'
        assert sequence.get_direction() == -1
    
    def test_time_window_expiration(self):
        """Test that sequences expire after time window."""
        sequence = SignalSequence(time_window_bars=10, bar_duration_minutes=5)
        
        # Add first signal
        time1 = datetime.now()
        signal1 = Signal('mlmi', 1, time1, 65.0, 0.8)
        sequence.add_signal(signal1)
        
        # Add signal after window expires (11 bars = 55 minutes)
        time2 = time1 + timedelta(minutes=56)
        signal2 = Signal('nwrqk', 1, time2, 100.0, 0.6)
        result = sequence.add_signal(signal2)
        
        assert result is False  # Sequence was reset
        assert len(sequence.signals) == 1  # Only new signal
        assert sequence.signals[0].signal_type == 'nwrqk'
    
    def test_duplicate_signal_types(self):
        """Test that duplicate signal types are ignored."""
        sequence = SignalSequence()
        
        # Add MLMI signal
        signal1 = Signal('mlmi', 1, datetime.now(), 65.0, 0.8)
        sequence.add_signal(signal1)
        
        # Try to add another MLMI signal
        signal2 = Signal('mlmi', 1, datetime.now(), 70.0, 0.9)
        result = sequence.add_signal(signal2)
        
        assert result is True  # Signal processed but ignored
        assert len(sequence.signals) == 1  # Still only one signal
        assert sequence.signals[0].value == 65.0  # Original remains
    
    def test_complete_sequence(self):
        """Test complete synergy sequence."""
        sequence = SignalSequence()
        base_time = datetime.now()
        
        # Add three signals to complete sequence
        signals = [
            Signal('mlmi', 1, base_time, 65.0, 0.8),
            Signal('nwrqk', 1, base_time + timedelta(minutes=5), 100.0, 0.6),
            Signal('fvg', 1, base_time + timedelta(minutes=10), 5150.0, 1.0)
        ]
        
        for signal in signals:
            sequence.add_signal(signal)
        
        assert sequence.is_complete() is True
        assert sequence.get_pattern() == ('mlmi', 'nwrqk', 'fvg')
        assert sequence.get_bars_to_complete() == 3  # 10 minutes = 2 bars + 1


class TestCooldownTracker:
    """Test cooldown period management."""
    
    def test_initial_state(self):
        """Test initial cooldown state."""
        cooldown = CooldownTracker(cooldown_bars=5)
        
        assert cooldown.is_active() is False
        assert cooldown.can_emit() is True
        assert cooldown.get_remaining_bars() == 0
    
    def test_start_cooldown(self):
        """Test starting cooldown period."""
        cooldown = CooldownTracker(cooldown_bars=5)
        start_time = datetime.now()
        
        cooldown.start_cooldown(start_time)
        
        assert cooldown.is_active() is True
        assert cooldown.can_emit() is False
        assert cooldown.last_synergy_time == start_time
    
    def test_cooldown_expiration(self):
        """Test cooldown expiration after time passes."""
        cooldown = CooldownTracker(cooldown_bars=5, bar_duration_minutes=5)
        start_time = datetime.now()
        
        cooldown.start_cooldown(start_time)
        
        # Update with time still in cooldown (4 bars = 20 minutes)
        cooldown.update(start_time + timedelta(minutes=20))
        assert cooldown.is_active() is True
        assert cooldown.get_remaining_bars() == 1
        
        # Update with time after cooldown (6 bars = 30 minutes)
        cooldown.update(start_time + timedelta(minutes=30))
        assert cooldown.is_active() is False
        assert cooldown.can_emit() is True
        assert cooldown.get_remaining_bars() == 0


class TestPatternDetectors:
    """Test individual pattern detectors."""
    
    def test_mlmi_pattern_detection(self):
        """Test MLMI crossover pattern detection."""
        config = {'mlmi_threshold': 0.5}
        detector = MLMIPatternDetector(config)
        
        # Test with signal but weak strength
        features = {
            'mlmi_signal': 1,
            'mlmi_value': 55,  # Only 5 points from neutral
            'timestamp': datetime.now()
        }
        signal = detector.detect_pattern(features)
        assert signal is None  # Below threshold
        
        # Test with strong signal
        features['mlmi_value'] = 75  # 25 points from neutral
        signal = detector.detect_pattern(features)
        
        assert signal is not None
        assert signal.signal_type == 'mlmi'
        assert signal.direction == 1
        assert signal.value == 75
        assert signal.strength == 0.5  # 25/50 = 0.5
    
    def test_nwrqk_pattern_detection(self):
        """Test NW-RQK direction change pattern detection."""
        config = {'nwrqk_threshold': 0.3}
        detector = NWRQKPatternDetector(config)
        
        # Test with weak slope
        features = {
            'nwrqk_signal': 1,
            'nwrqk_slope': 0.2,  # Below threshold
            'nwrqk_value': 100.0,
            'timestamp': datetime.now()
        }
        signal = detector.detect_pattern(features)
        assert signal is None
        
        # Test with strong slope
        features['nwrqk_slope'] = 0.5
        signal = detector.detect_pattern(features)
        
        assert signal is not None
        assert signal.signal_type == 'nwrqk'
        assert signal.direction == 1
        assert signal.strength == 0.25  # 0.5/2.0 = 0.25
    
    def test_fvg_pattern_detection(self):
        """Test FVG mitigation pattern detection."""
        config = {'fvg_min_size': 0.001}  # 0.1%
        detector = FVGPatternDetector(config)
        
        # Test with no mitigation
        features = {
            'fvg_mitigation_signal': False,
            'timestamp': datetime.now()
        }
        signal = detector.detect_pattern(features)
        assert signal is None
        
        # Test with bullish mitigation
        features = {
            'fvg_mitigation_signal': True,
            'fvg_bullish_mitigated': True,
            'fvg_bearish_mitigated': False,
            'fvg_bullish_size': 10.0,
            'fvg_bullish_level': 5150.0,
            'current_price': 5150.0,
            'timestamp': datetime.now()
        }
        signal = detector.detect_pattern(features)
        
        assert signal is not None
        assert signal.signal_type == 'fvg'
        assert signal.direction == 1  # Bullish
        gap_size_pct = 10.0 / 5150.0
        assert abs(signal.strength - min(gap_size_pct / 0.01, 1.0)) < 0.001


class TestSynergyDetector:
    """Test main SynergyDetector functionality."""
    
    @pytest.fixture
    def mock_kernel(self):
        """Create mock kernel with configuration."""
        kernel = Mock()
        kernel.config = {
            'synergy_detector': {
                'time_window': 10,
                'mlmi_threshold': 0.5,
                'nwrqk_threshold': 0.3,
                'fvg_min_size': 0.001,
                'cooldown_bars': 5
            }
        }
        kernel.event_bus = Mock()
        kernel.event_bus.subscribe = Mock()
        kernel.event_bus.publish = Mock()
        kernel.event_bus.create_event = Mock(return_value=Mock())
        return kernel
    
    def test_initialization(self, mock_kernel):
        """Test SynergyDetector initialization."""
        detector = SynergyDetector('SynergyDetector', mock_kernel)
        
        assert detector.name == 'SynergyDetector'
        assert detector.time_window == 10
        assert detector.cooldown_bars == 5
        assert detector.mlmi_detector is not None
        assert detector.nwrqk_detector is not None
        assert detector.fvg_detector is not None
    
    def test_single_signal_detection(self, mock_kernel):
        """Test detection of individual signals."""
        detector = SynergyDetector('SynergyDetector', mock_kernel)
        
        features = {
            'mlmi_signal': 1,
            'mlmi_value': 75,
            'nwrqk_signal': 0,
            'fvg_mitigation_signal': False,
            'timestamp': datetime.now()
        }
        
        signals = detector._detect_signals(features)
        
        assert len(signals) == 1
        assert signals[0].signal_type == 'mlmi'
        assert signals[0].direction == 1
    
    def test_synergy_pattern_detection(self, mock_kernel):
        """Test complete synergy pattern detection."""
        detector = SynergyDetector('SynergyDetector', mock_kernel)
        base_time = datetime.now()
        
        # Process features that form TYPE_1 synergy
        # Step 1: MLMI signal
        features1 = {
            'mlmi_signal': 1,
            'mlmi_value': 75,
            'nwrqk_signal': 0,
            'fvg_mitigation_signal': False,
            'timestamp': base_time,
            'current_price': 5150.0,
            'volatility_30': 12.5
        }
        synergy = detector.process_features(features1, base_time)
        assert synergy is None  # Not complete yet
        
        # Step 2: NW-RQK signal
        features2 = {
            'mlmi_signal': 0,
            'nwrqk_signal': 1,
            'nwrqk_slope': 0.5,
            'nwrqk_value': 100.0,
            'fvg_mitigation_signal': False,
            'timestamp': base_time + timedelta(minutes=5),
            'current_price': 5152.0,
            'volatility_30': 12.5
        }
        synergy = detector.process_features(features2, base_time + timedelta(minutes=5))
        assert synergy is None  # Not complete yet
        
        # Step 3: FVG mitigation completes TYPE_1
        features3 = {
            'mlmi_signal': 0,
            'nwrqk_signal': 0,
            'fvg_mitigation_signal': True,
            'fvg_bullish_mitigated': True,
            'fvg_bullish_size': 10.0,
            'fvg_bullish_level': 5150.0,
            'current_price': 5154.0,
            'volatility_30': 12.5,
            'timestamp': base_time + timedelta(minutes=10)
        }
        synergy = detector.process_features(features3, base_time + timedelta(minutes=10))
        
        assert synergy is not None
        assert synergy.synergy_type == 'TYPE_1'
        assert synergy.direction == 1
        assert len(synergy.signals) == 3
        assert synergy.confidence == 1.0
    
    def test_cooldown_enforcement(self, mock_kernel):
        """Test that cooldown prevents immediate re-detection."""
        detector = SynergyDetector('SynergyDetector', mock_kernel)
        base_time = datetime.now()
        
        # Create a complete sequence manually
        detector.sequence.signals = [
            Signal('mlmi', 1, base_time, 75.0, 0.8),
            Signal('nwrqk', 1, base_time + timedelta(minutes=5), 100.0, 0.6),
            Signal('fvg', 1, base_time + timedelta(minutes=10), 5150.0, 1.0)
        ]
        
        # First detection should succeed
        synergy1 = detector._check_and_create_synergy()
        assert synergy1 is not None
        
        # Start cooldown
        detector.cooldown.start_cooldown(base_time + timedelta(minutes=10))
        
        # Try to detect again immediately - should be blocked
        detector.sequence.signals = [
            Signal('mlmi', 1, base_time + timedelta(minutes=15), 75.0, 0.8),
            Signal('nwrqk', 1, base_time + timedelta(minutes=20), 100.0, 0.6),
            Signal('fvg', 1, base_time + timedelta(minutes=25), 5150.0, 1.0)
        ]
        
        # Update cooldown state (still within 5 bars)
        detector.cooldown.update(base_time + timedelta(minutes=25))
        
        # Should not emit due to cooldown
        assert detector.cooldown.can_emit() is False
    
    def test_performance_metrics(self, mock_kernel):
        """Test performance metric tracking."""
        detector = SynergyDetector('SynergyDetector', mock_kernel)
        
        # Process some features
        features = {
            'mlmi_signal': 0,
            'nwrqk_signal': 0,
            'fvg_mitigation_signal': False,
            'timestamp': datetime.now()
        }
        
        # Process multiple times
        for _ in range(5):
            detector.process_features(features, datetime.now())
        
        metrics = detector.performance_metrics
        assert metrics['events_processed'] == 5
        assert metrics['avg_processing_time_ms'] > 0
        assert metrics['max_processing_time_ms'] >= metrics['avg_processing_time_ms']
    
    def test_event_emission(self, mock_kernel):
        """Test SYNERGY_DETECTED event emission."""
        detector = SynergyDetector('SynergyDetector', mock_kernel)
        
        # Create a synergy pattern
        synergy = SynergyPattern(
            synergy_type='TYPE_1',
            direction=1,
            signals=[
                Signal('mlmi', 1, datetime.now(), 75.0, 0.8),
                Signal('nwrqk', 1, datetime.now(), 100.0, 0.6),
                Signal('fvg', 1, datetime.now(), 5150.0, 1.0)
            ],
            completion_time=datetime.now(),
            bars_to_complete=3
        )
        
        features = {
            'current_price': 5150.0,
            'volatility_30': 12.5,
            'volume_ratio': 1.2,
            'lvn_nearest_price': 5145.0,
            'lvn_nearest_strength': 85.0,
            'lvn_distance_points': 5.0
        }
        
        detector._emit_synergy_event(synergy, features)
        
        # Verify event was created and published
        mock_kernel.event_bus.create_event.assert_called_once()
        mock_kernel.event_bus.publish.assert_called_once()
        
        # Check event payload
        call_args = mock_kernel.event_bus.create_event.call_args
        assert call_args[0][0] == EventType.SYNERGY_DETECTED
        payload = call_args[0][1]
        assert payload['synergy_type'] == 'TYPE_1'
        assert payload['direction'] == 1
        assert len(payload['signal_sequence']) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])