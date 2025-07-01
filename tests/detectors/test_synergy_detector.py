"""
Comprehensive unit tests for the SynergyDetector component.

This test suite verifies all aspects of synergy detection including:
- Pattern recognition
- Time window constraints
- Direction consistency
- Cooldown mechanism
- Configuration handling
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List

from src.agents.synergy.detector import SynergyDetector
from src.agents.synergy.base import Signal, SynergyPattern
from src.core.events import EventType, Event


@pytest.fixture
def mock_kernel():
    """Create a mock kernel with required attributes."""
    kernel = Mock()
    kernel.config = {
        'synergy_detector': {
            # Time and sequence parameters
            'time_window_bars': 10,
            'cooldown_bars': 5,
            'bar_duration_minutes': 5,
            'required_signals': 3,
            
            # Signal activation thresholds
            'mlmi_threshold': 0.5,
            'mlmi_neutral_line': 50,
            'mlmi_scaling_factor': 50,
            'mlmi_max_strength': 1.0,
            
            'nwrqk_threshold': 0.3,
            'nwrqk_max_slope': 2.0,
            'nwrqk_max_strength': 1.0,
            
            'fvg_min_size': 0.001,
            'fvg_max_gap_pct': 0.01,
            'fvg_max_strength': 1.0,
            
            # Performance monitoring
            'processing_time_warning_ms': 1.0,
            
            # Default fallback values for missing features
            'defaults': {
                'current_price': 0.0,
                'volatility': 0.0,
                'volume_ratio': 1.0,
                'volume_momentum': 0.0,
                'mlmi_value': 50,
                'nwrqk_slope': 0.0,
                'nwrqk_value': 0.0
            }
        }
    }
    kernel.event_bus = Mock()
    kernel.event_bus.create_event = Mock(return_value=Mock())
    kernel.event_bus.publish = Mock()
    kernel.event_bus.subscribe = Mock()
    kernel.event_bus.unsubscribe = Mock()
    return kernel


@pytest.fixture
def synergy_detector(mock_kernel):
    """Create a SynergyDetector instance for testing."""
    return SynergyDetector('TestSynergyDetector', mock_kernel)


@pytest.fixture
def base_timestamp():
    """Base timestamp for consistent testing."""
    return datetime(2024, 1, 1, 10, 0, 0)


def create_indicator_event(features: Dict[str, Any], timestamp: datetime) -> Event:
    """Helper to create INDICATORS_READY events."""
    event = Mock(spec=Event)
    event.event_type = EventType.INDICATORS_READY
    event.payload = features
    event.timestamp = timestamp
    return event


def create_features_with_signal(
    signal_type: str,
    direction: int,
    timestamp: datetime,
    base_price: float = 5000.0
) -> Dict[str, Any]:
    """Create feature dictionary with specified signal active."""
    features = {
        'timestamp': timestamp,
        'current_price': base_price,
        'volatility_30': 0.15,
        'volume_ratio': 1.2,
        'volume_momentum_30': 0.1,
        'lvn_nearest_price': base_price - 10,
        'lvn_nearest_strength': 0.8,
        'lvn_distance_points': 10.0,
        # Default inactive signals
        'mlmi_signal': 0,
        'mlmi_value': 50.0,
        'nwrqk_signal': 0,
        'nwrqk_slope': 0.0,
        'nwrqk_value': base_price,
        'fvg_mitigation_signal': False,
        'fvg_bullish_mitigated': False,
        'fvg_bearish_mitigated': False,
        'fvg_bullish_size': 0.0,
        'fvg_bearish_size': 0.0,
        'fvg_bullish_level': base_price,
        'fvg_bearish_level': base_price,
    }
    
    # Activate the specified signal
    if signal_type == 'mlmi':
        features['mlmi_signal'] = direction
        features['mlmi_value'] = 75.0 if direction == 1 else 25.0  # Strong deviation
    elif signal_type == 'nwrqk':
        features['nwrqk_signal'] = direction
        features['nwrqk_slope'] = 0.5 if direction == 1 else -0.5  # Above threshold
    elif signal_type == 'fvg':
        features['fvg_mitigation_signal'] = True
        if direction == 1:
            features['fvg_bullish_mitigated'] = True
            features['fvg_bullish_size'] = 10.0  # 0.2% gap
        else:
            features['fvg_bearish_mitigated'] = True
            features['fvg_bearish_size'] = 10.0  # 0.2% gap
    
    return features


class TestSynergyDetector:
    """Test suite for SynergyDetector functionality."""
    
    def test_successful_detection_of_synergy_type_1(self, synergy_detector, base_timestamp):
        """Test successful detection of TYPE_1 synergy pattern (MLMI -> NW-RQK -> FVG)."""
        # Reset any previous state
        synergy_detector.sequence.reset()
        
        # Track emitted events
        emitted_events = []
        synergy_detector.kernel.event_bus.publish = lambda event: emitted_events.append(event)
        
        # Signal 1: MLMI bullish
        features1 = create_features_with_signal('mlmi', 1, base_timestamp)
        event1 = create_indicator_event(features1, base_timestamp)
        synergy_detector._handle_indicators_ready(event1)
        
        # Verify no synergy emitted yet
        assert len(emitted_events) == 0
        assert len(synergy_detector.sequence.signals) == 1
        
        # Signal 2: NW-RQK bullish (2 bars later)
        timestamp2 = base_timestamp + timedelta(minutes=10)  # 2 bars
        features2 = create_features_with_signal('nwrqk', 1, timestamp2)
        event2 = create_indicator_event(features2, timestamp2)
        synergy_detector._handle_indicators_ready(event2)
        
        # Still no synergy
        assert len(emitted_events) == 0
        assert len(synergy_detector.sequence.signals) == 2
        
        # Signal 3: FVG bullish (3 bars after start)
        timestamp3 = base_timestamp + timedelta(minutes=15)  # 3 bars
        features3 = create_features_with_signal('fvg', 1, timestamp3)
        event3 = create_indicator_event(features3, timestamp3)
        synergy_detector._handle_indicators_ready(event3)
        
        # Now we should have a TYPE_1 synergy
        assert len(emitted_events) == 1
        
        # Verify the emitted event
        create_event_call = synergy_detector.kernel.event_bus.create_event.call_args
        assert create_event_call[0][0] == EventType.SYNERGY_DETECTED
        
        payload = create_event_call[0][1]
        assert payload['synergy_type'] == 'TYPE_1'
        assert payload['direction'] == 1
        assert len(payload['signal_sequence']) == 3
        assert payload['signal_sequence'][0]['type'] == 'mlmi'
        assert payload['signal_sequence'][1]['type'] == 'nwrqk'
        assert payload['signal_sequence'][2]['type'] == 'fvg'
        
        # Verify metrics updated
        assert synergy_detector.performance_metrics['synergies_detected'] == 1
        assert synergy_detector.performance_metrics['patterns_by_type']['TYPE_1'] == 1
    
    def test_rejection_due_to_incorrect_signal_order(self, synergy_detector, base_timestamp):
        """Test rejection when signals appear in wrong order."""
        # Reset state
        synergy_detector.sequence.reset()
        
        # Track emitted events
        emitted_events = []
        synergy_detector.kernel.event_bus.publish = lambda event: emitted_events.append(event)
        
        # Wrong order: MLMI -> FVG -> NW-RQK (not a valid pattern)
        # Signal 1: MLMI bullish
        features1 = create_features_with_signal('mlmi', 1, base_timestamp)
        event1 = create_indicator_event(features1, base_timestamp)
        synergy_detector._handle_indicators_ready(event1)
        
        # Signal 2: FVG bullish
        timestamp2 = base_timestamp + timedelta(minutes=5)
        features2 = create_features_with_signal('fvg', 1, timestamp2)
        event2 = create_indicator_event(features2, timestamp2)
        synergy_detector._handle_indicators_ready(event2)
        
        # Signal 3: NW-RQK bullish
        timestamp3 = base_timestamp + timedelta(minutes=10)
        features3 = create_features_with_signal('nwrqk', 1, timestamp3)
        event3 = create_indicator_event(features3, timestamp3)
        synergy_detector._handle_indicators_ready(event3)
        
        # This pattern (mlmi, fvg, nwrqk) is TYPE_2, not invalid
        assert len(emitted_events) == 1
        create_event_call = synergy_detector.kernel.event_bus.create_event.call_args
        payload = create_event_call[0][1]
        assert payload['synergy_type'] == 'TYPE_2'
        
        # Test truly invalid pattern
        synergy_detector.sequence.reset()
        emitted_events.clear()
        synergy_detector.kernel.event_bus.create_event.reset_mock()
        
        # Create pattern that's not in SYNERGY_PATTERNS
        # We need to simulate duplicate signals which should be ignored
        features1 = create_features_with_signal('mlmi', 1, base_timestamp)
        event1 = create_indicator_event(features1, base_timestamp)
        synergy_detector._handle_indicators_ready(event1)
        
        # Same signal type again (should be ignored)
        timestamp2 = base_timestamp + timedelta(minutes=5)
        features2 = create_features_with_signal('mlmi', 1, timestamp2)
        event2 = create_indicator_event(features2, timestamp2)
        synergy_detector._handle_indicators_ready(event2)
        
        # Add two more different signals
        timestamp3 = base_timestamp + timedelta(minutes=10)
        features3 = create_features_with_signal('nwrqk', 1, timestamp3)
        event3 = create_indicator_event(features3, timestamp3)
        synergy_detector._handle_indicators_ready(event3)
        
        # Sequence still only has 2 signals (mlmi, nwrqk) - not complete
        assert len(synergy_detector.sequence.signals) == 2
        assert len(emitted_events) == 0
    
    def test_rejection_due_to_inconsistent_direction(self, synergy_detector, base_timestamp):
        """Test rejection when signals have inconsistent directions."""
        # Reset state
        synergy_detector.sequence.reset()
        
        # Track emitted events
        emitted_events = []
        synergy_detector.kernel.event_bus.publish = lambda event: emitted_events.append(event)
        
        # Signal 1: MLMI bullish (direction = 1)
        features1 = create_features_with_signal('mlmi', 1, base_timestamp)
        event1 = create_indicator_event(features1, base_timestamp)
        synergy_detector._handle_indicators_ready(event1)
        
        assert len(synergy_detector.sequence.signals) == 1
        
        # Signal 2: NW-RQK bearish (direction = -1) - should reset sequence
        timestamp2 = base_timestamp + timedelta(minutes=5)
        features2 = create_features_with_signal('nwrqk', -1, timestamp2)
        event2 = create_indicator_event(features2, timestamp2)
        synergy_detector._handle_indicators_ready(event2)
        
        # Sequence should be reset and start with new signal
        assert len(synergy_detector.sequence.signals) == 1
        assert synergy_detector.sequence.signals[0].signal_type == 'nwrqk'
        assert synergy_detector.sequence.signals[0].direction == -1
        
        # No synergy should be emitted
        assert len(emitted_events) == 0
    
    def test_sequence_expiration_due_to_time_window(self, synergy_detector, base_timestamp):
        """Test sequence expiration when time window is exceeded."""
        # Reset state
        synergy_detector.sequence.reset()
        
        # Track emitted events
        emitted_events = []
        synergy_detector.kernel.event_bus.publish = lambda event: emitted_events.append(event)
        
        # Signal 1: MLMI bullish
        features1 = create_features_with_signal('mlmi', 1, base_timestamp)
        event1 = create_indicator_event(features1, base_timestamp)
        synergy_detector._handle_indicators_ready(event1)
        
        # Signal 2: NW-RQK bullish (5 bars later - still within window)
        timestamp2 = base_timestamp + timedelta(minutes=25)  # 5 bars
        features2 = create_features_with_signal('nwrqk', 1, timestamp2)
        event2 = create_indicator_event(features2, timestamp2)
        synergy_detector._handle_indicators_ready(event2)
        
        assert len(synergy_detector.sequence.signals) == 2
        
        # Signal 3: FVG bullish (11 bars after start - exceeds 10 bar window)
        timestamp3 = base_timestamp + timedelta(minutes=55)  # 11 bars
        features3 = create_features_with_signal('fvg', 1, timestamp3)
        event3 = create_indicator_event(features3, timestamp3)
        synergy_detector._handle_indicators_ready(event3)
        
        # Sequence should be reset with new signal
        assert len(synergy_detector.sequence.signals) == 1
        assert synergy_detector.sequence.signals[0].signal_type == 'fvg'
        
        # No synergy should be emitted
        assert len(emitted_events) == 0
    
    def test_cooldown_mechanism(self, synergy_detector, base_timestamp):
        """Test cooldown period after synergy detection."""
        # Reset state
        synergy_detector.sequence.reset()
        synergy_detector.cooldown.last_synergy_time = None
        
        # Track emitted events
        emitted_events = []
        
        def track_event(event):
            emitted_events.append(event)
        
        synergy_detector.kernel.event_bus.publish = track_event
        
        # First synergy: TYPE_1
        features1 = create_features_with_signal('mlmi', 1, base_timestamp)
        event1 = create_indicator_event(features1, base_timestamp)
        synergy_detector._handle_indicators_ready(event1)
        
        timestamp2 = base_timestamp + timedelta(minutes=5)
        features2 = create_features_with_signal('nwrqk', 1, timestamp2)
        event2 = create_indicator_event(features2, timestamp2)
        synergy_detector._handle_indicators_ready(event2)
        
        timestamp3 = base_timestamp + timedelta(minutes=10)
        features3 = create_features_with_signal('fvg', 1, timestamp3)
        event3 = create_indicator_event(features3, timestamp3)
        synergy_detector._handle_indicators_ready(event3)
        
        # First synergy should be emitted
        assert len(emitted_events) == 1
        
        # Immediately try another synergy (TYPE_3: nwrqk -> fvg -> mlmi)
        # This should be blocked by cooldown
        timestamp4 = base_timestamp + timedelta(minutes=15)  # Only 3 bars after first synergy
        features4 = create_features_with_signal('nwrqk', 1, timestamp4)
        event4 = create_indicator_event(features4, timestamp4)
        synergy_detector._handle_indicators_ready(event4)
        
        timestamp5 = base_timestamp + timedelta(minutes=20)
        features5 = create_features_with_signal('fvg', 1, timestamp5)
        event5 = create_indicator_event(features5, timestamp5)
        synergy_detector._handle_indicators_ready(event5)
        
        timestamp6 = base_timestamp + timedelta(minutes=25)
        features6 = create_features_with_signal('mlmi', 1, timestamp6)
        event6 = create_indicator_event(features6, timestamp6)
        synergy_detector._handle_indicators_ready(event6)
        
        # Should still be only 1 event (cooldown active)
        assert len(emitted_events) == 1
        
        # Wait for cooldown to expire (5 bars = 25 minutes after first synergy)
        # Update cooldown state
        timestamp7 = base_timestamp + timedelta(minutes=35)  # 7 bars after first synergy
        synergy_detector.cooldown.update(timestamp7)
        
        # Now try another synergy
        features7 = create_features_with_signal('mlmi', 1, timestamp7)
        event7 = create_indicator_event(features7, timestamp7)
        synergy_detector._handle_indicators_ready(event7)
        
        timestamp8 = base_timestamp + timedelta(minutes=40)
        features8 = create_features_with_signal('nwrqk', 1, timestamp8)
        event8 = create_indicator_event(features8, timestamp8)
        synergy_detector._handle_indicators_ready(event8)
        
        timestamp9 = base_timestamp + timedelta(minutes=45)
        features9 = create_features_with_signal('fvg', 1, timestamp9)
        event9 = create_indicator_event(features9, timestamp9)
        synergy_detector._handle_indicators_ready(event9)
        
        # Now we should have 2 events (cooldown expired)
        assert len(emitted_events) == 2
    
    def test_configuration_is_used_correctly(self, mock_kernel):
        """Test that different configurations are correctly applied."""
        # Create first detector with default config
        detector1 = SynergyDetector('Detector1', mock_kernel)
        assert detector1.time_window_bars == 10
        assert detector1.cooldown_bars == 5
        
        # Create second detector with different config
        mock_kernel2 = Mock()
        mock_kernel2.config = {
            'synergy_detector': {
                # Time and sequence parameters
                'time_window_bars': 15,
                'cooldown_bars': 8,
                'bar_duration_minutes': 3,  # 3-minute bars instead of 5
                'required_signals': 3,
                
                # Signal activation thresholds
                'mlmi_threshold': 0.7,
                'mlmi_neutral_line': 45,  # Different neutral line
                'mlmi_scaling_factor': 45,
                'mlmi_max_strength': 0.9,  # Lower max strength
                
                'nwrqk_threshold': 0.4,
                'nwrqk_max_slope': 3.0,  # Higher max slope
                'nwrqk_max_strength': 0.8,
                
                'fvg_min_size': 0.002,
                'fvg_max_gap_pct': 0.02,  # 2% instead of 1%
                'fvg_max_strength': 0.95,
                
                # Performance monitoring
                'processing_time_warning_ms': 2.0,  # Higher threshold
                
                # Different defaults
                'defaults': {
                    'current_price': 5000.0,
                    'volatility': 0.1,
                    'volume_ratio': 1.5,
                    'volume_momentum': 0.05,
                    'mlmi_value': 45,  # Match neutral line
                    'nwrqk_slope': 0.1,
                    'nwrqk_value': 5000.0
                }
            }
        }
        mock_kernel2.event_bus = Mock()
        mock_kernel2.event_bus.create_event = Mock(return_value=Mock())
        mock_kernel2.event_bus.publish = Mock()
        mock_kernel2.event_bus.subscribe = Mock()
        mock_kernel2.event_bus.unsubscribe = Mock()
        
        detector2 = SynergyDetector('Detector2', mock_kernel2)
        assert detector2.time_window_bars == 15
        assert detector2.cooldown_bars == 8
        assert detector2.bar_duration_minutes == 3
        assert detector2.required_signals == 3
        assert detector2.processing_time_warning_ms == 2.0
        
        # Verify pattern detectors got correct configs
        assert detector1.mlmi_detector.threshold == 0.5
        assert detector2.mlmi_detector.threshold == 0.7
        assert detector1.mlmi_detector.neutral_line == 50
        assert detector2.mlmi_detector.neutral_line == 45
        assert detector1.mlmi_detector.scaling_factor == 50
        assert detector2.mlmi_detector.scaling_factor == 45
        assert detector1.mlmi_detector.max_strength == 1.0
        assert detector2.mlmi_detector.max_strength == 0.9
        
        assert detector1.nwrqk_detector.threshold == 0.3
        assert detector2.nwrqk_detector.threshold == 0.4
        assert detector1.nwrqk_detector.max_slope == 2.0
        assert detector2.nwrqk_detector.max_slope == 3.0
        assert detector1.nwrqk_detector.max_strength == 1.0
        assert detector2.nwrqk_detector.max_strength == 0.8
        
        assert detector1.fvg_detector.min_size == 0.001
        assert detector2.fvg_detector.min_size == 0.002
        assert detector1.fvg_detector.max_gap_pct == 0.01
        assert detector2.fvg_detector.max_gap_pct == 0.02
        assert detector1.fvg_detector.max_strength == 1.0
        assert detector2.fvg_detector.max_strength == 0.95
        
        # Test time window behavior difference
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        
        # For detector1 (10 bar window)
        detector1.sequence.reset()
        signal1 = Signal('mlmi', 1, base_time, 75.0, 0.8)
        detector1.sequence.add_signal(signal1)
        
        # Signal 11 bars later should expire for detector1
        time_11_bars = base_time + timedelta(minutes=55)
        signal2 = Signal('nwrqk', 1, time_11_bars, 5010.0, 0.7)
        result1 = detector1.sequence.add_signal(signal2)
        
        # Sequence should have been reset
        assert len(detector1.sequence.signals) == 1
        assert detector1.sequence.signals[0].signal_type == 'nwrqk'
        
        # For detector2 (15 bar window)
        detector2.sequence.reset()
        signal1_d2 = Signal('mlmi', 1, base_time, 75.0, 0.8)
        detector2.sequence.add_signal(signal1_d2)
        
        # Same signal 11 bars later should NOT expire for detector2
        signal2_d2 = Signal('nwrqk', 1, time_11_bars, 5010.0, 0.7)
        result2 = detector2.sequence.add_signal(signal2_d2)
        
        # Sequence should still have both signals
        assert len(detector2.sequence.signals) == 2
        assert detector2.sequence.signals[0].signal_type == 'mlmi'
        assert detector2.sequence.signals[1].signal_type == 'nwrqk'
    
    def test_configuration_defaults_are_used(self, mock_kernel, base_timestamp):
        """Test that configuration defaults are correctly applied when features are missing."""
        detector = SynergyDetector('TestDetector', mock_kernel)
        
        # Test MLMI with missing mlmi_value (should use default from config)
        features = {
            'timestamp': base_timestamp,
            'mlmi_signal': 1,
            # mlmi_value is missing - should use default of 50
            'current_price': 5000.0,
            'volatility_30': 0.15,
            'volume_ratio': 1.2,
            'volume_momentum_30': 0.1,
        }
        
        event = create_indicator_event(features, base_timestamp)
        detector._handle_indicators_ready(event)
        
        # Should still detect signal using default value
        assert len(detector.sequence.signals) == 1
        assert detector.sequence.signals[0].signal_type == 'mlmi'
        
        # Test NW-RQK with missing slope (should use default from config)
        detector.sequence.reset()
        features2 = {
            'timestamp': base_timestamp,
            'nwrqk_signal': 1,
            # nwrqk_slope is missing - should use default of 0.0
            'nwrqk_value': 5000.0,
            'current_price': 5000.0,
        }
        
        event2 = create_indicator_event(features2, base_timestamp)
        detector._handle_indicators_ready(event2)
        
        # Should not detect signal because default slope (0.0) is below threshold (0.3)
        assert len(detector.sequence.signals) == 0
        
        # Test FVG with missing current_price (should use default from config)
        features3 = {
            'timestamp': base_timestamp,
            'fvg_mitigation_signal': True,
            'fvg_bullish_mitigated': True,
            'fvg_bullish_size': 10.0,
            'fvg_bullish_level': 5000.0,
            # current_price is missing - should use default of 0.0
        }
        
        event3 = create_indicator_event(features3, base_timestamp)
        detector._handle_indicators_ready(event3)
        
        # Should not detect signal because gap_size_pct calculation will fail with price=0
        # or signal should be rejected due to insufficient gap size
        assert len(detector.sequence.signals) == 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_features(self, synergy_detector, base_timestamp):
        """Test handling of empty or minimal features."""
        # Empty features dict
        event = create_indicator_event({}, base_timestamp)
        
        # Should not crash
        synergy_detector._handle_indicators_ready(event)
        
        # No signals should be detected
        assert len(synergy_detector.sequence.signals) == 0
    
    def test_missing_timestamp(self, synergy_detector):
        """Test handling of missing timestamp in features."""
        features = create_features_with_signal('mlmi', 1, datetime.now())
        del features['timestamp']
        
        event = create_indicator_event(features, datetime.now())
        
        # Should not crash and should use current time
        synergy_detector._handle_indicators_ready(event)
        
        # Signal should still be processed
        assert len(synergy_detector.sequence.signals) == 1
    
    def test_exception_handling(self, synergy_detector, base_timestamp):
        """Test exception handling in event processing."""
        # Mock a detector to raise exception
        synergy_detector.mlmi_detector.detect_pattern = Mock(
            side_effect=Exception("Test exception")
        )
        
        features = create_features_with_signal('mlmi', 1, base_timestamp)
        event = create_indicator_event(features, base_timestamp)
        
        # Should not crash
        synergy_detector._handle_indicators_ready(event)
        
        # Metrics should still be updated
        assert synergy_detector.performance_metrics['events_processed'] == 1
    
    def test_performance_metrics(self, synergy_detector, base_timestamp):
        """Test performance metric tracking."""
        # Process multiple events
        for i in range(5):
            timestamp = base_timestamp + timedelta(minutes=i*5)
            features = create_features_with_signal('mlmi', 1, timestamp)
            event = create_indicator_event(features, timestamp)
            synergy_detector._handle_indicators_ready(event)
        
        # Check metrics
        assert synergy_detector.performance_metrics['events_processed'] == 5
        assert synergy_detector.performance_metrics['signals_detected'] == 5
        assert synergy_detector.performance_metrics['avg_processing_time_ms'] > 0
        assert synergy_detector.performance_metrics['max_processing_time_ms'] > 0
    
    @pytest.mark.asyncio
    async def test_lifecycle_methods(self, synergy_detector):
        """Test component lifecycle methods."""
        # Test initialization
        await synergy_detector.initialize()
        assert synergy_detector._initialized == True
        
        # Verify event subscription
        synergy_detector.kernel.event_bus.subscribe.assert_called_once_with(
            EventType.INDICATORS_READY,
            synergy_detector._handle_indicators_ready
        )
        
        # Test shutdown
        await synergy_detector.shutdown()
        
        # Verify event unsubscription
        synergy_detector.kernel.event_bus.unsubscribe.assert_called_once_with(
            EventType.INDICATORS_READY,
            synergy_detector._handle_indicators_ready
        )
    
    def test_get_status(self, synergy_detector, base_timestamp):
        """Test status reporting."""
        # Process some events to populate state
        features1 = create_features_with_signal('mlmi', 1, base_timestamp)
        event1 = create_indicator_event(features1, base_timestamp)
        synergy_detector._handle_indicators_ready(event1)
        
        status = synergy_detector.get_status()
        
        assert 'initialized' in status
        assert 'performance_metrics' in status
        assert 'sequence_state' in status
        assert 'cooldown_state' in status
        assert 'pattern_detectors' in status
        
        # Verify sequence state
        assert status['sequence_state']['active'] == True
        assert status['sequence_state']['signal_count'] == 1
        assert status['sequence_state']['signals'] == ['mlmi']