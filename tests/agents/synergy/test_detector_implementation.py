"""
Comprehensive tests for synergy detector implementation.
"""

import pytest
import torch
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
import time

from src.agents.synergy.detector import SynergyDetector
from src.agents.synergy.base import Signal, SynergyPattern
from src.core.events import EventType, Event


class TestSynergyDetectorImplementation:
    """Complete test suite for SynergyDetector implementation."""
    
    @pytest.fixture
    def mock_kernel(self):
        """Create mock kernel with required components."""
        kernel = Mock()
        kernel.config = Mock()
        kernel.config.get = Mock(side_effect=lambda key, default=None: {
            'synergy_detector': {
                'time_window_bars': 10,
                'cooldown_bars': 5,
                'bar_duration_minutes': 5,
                'required_signals': 3,
                'processing_time_warning_ms': 1.0,
                'mlmi_detector': {
                    'threshold': 0.5,
                    'neutral_line': 50,
                    'scaling_factor': 50,
                    'max_strength': 1.0
                },
                'nwrqk_detector': {
                    'threshold': 0.3,
                    'max_slope': 2.0,
                    'max_strength': 1.0
                },
                'fvg_detector': {
                    'min_size': 0.001,
                    'max_gap_pct': 0.01,
                    'require_mitigation': True
                }
            }
        }.get(key, default))
        
        # Mock event bus
        kernel.event_bus = Mock()
        kernel.event_bus.subscribe = Mock()
        kernel.event_bus.unsubscribe = Mock()
        kernel.event_bus.create_event = Mock(return_value=Mock(id='test-event'))
        kernel.event_bus.publish = Mock()
        
        return kernel
    
    def test_synergy_patterns_defined(self, mock_kernel):
        """Test that all 4 synergy patterns are properly defined."""
        detector = SynergyDetector('TestDetector', mock_kernel)
        
        # Verify SYNERGY_PATTERNS exists and has all 4 patterns
        assert hasattr(detector, 'SYNERGY_PATTERNS')
        assert len(detector.SYNERGY_PATTERNS) == 4
        
        # Verify all patterns
        expected_patterns = {
            ('mlmi', 'nwrqk', 'fvg'): 'TYPE_1',
            ('mlmi', 'fvg', 'nwrqk'): 'TYPE_2',
            ('nwrqk', 'fvg', 'mlmi'): 'TYPE_3',
            ('nwrqk', 'mlmi', 'fvg'): 'TYPE_4'
        }
        
        assert detector.SYNERGY_PATTERNS == expected_patterns
    
    def test_all_pattern_types_detection(self, mock_kernel):
        """Test detection of all 4 pattern types."""
        detector = SynergyDetector('TestDetector', mock_kernel)
        
        test_cases = [
            # TYPE_1: MLMI → NW-RQK → FVG
            {
                'signals': ['mlmi', 'nwrqk', 'fvg'],
                'expected_type': 'TYPE_1'
            },
            # TYPE_2: MLMI → FVG → NW-RQK
            {
                'signals': ['mlmi', 'fvg', 'nwrqk'],
                'expected_type': 'TYPE_2'
            },
            # TYPE_3: NW-RQK → FVG → MLMI
            {
                'signals': ['nwrqk', 'fvg', 'mlmi'],
                'expected_type': 'TYPE_3'
            },
            # TYPE_4: NW-RQK → MLMI → FVG
            {
                'signals': ['nwrqk', 'mlmi', 'fvg'],
                'expected_type': 'TYPE_4'
            }
        ]
        
        base_time = datetime.now()
        
        for test_case in test_cases:
            # Reset detector
            detector.sequence.reset()
            detector.cooldown.reset()
            
            # Build signal sequence
            for i, signal_type in enumerate(test_case['signals']):
                signal = Signal(
                    signal_type=signal_type,
                    direction=1,
                    timestamp=base_time + timedelta(minutes=i*5),
                    value=75.0 if signal_type == 'mlmi' else 1.0,
                    strength=0.8
                )
                detector.sequence.add_signal(signal)
            
            # Check pattern detection
            synergy = detector._check_and_create_synergy()
            assert synergy is not None
            assert synergy.synergy_type == test_case['expected_type']
            assert len(synergy.signals) == 3
    
    def test_performance_requirement(self, mock_kernel):
        """Test that processing time is under 1ms."""
        detector = SynergyDetector('TestDetector', mock_kernel)
        
        # Create test features
        features = {
            'timestamp': datetime.now(),
            'current_price': 5000.0,
            'mlmi_signal': 1,
            'mlmi_value': 75.0,
            'nwrqk_signal': 0,
            'nwrqk_slope': 0.1,
            'fvg_mitigation_signal': False,
            'volatility_30': 0.15,
            'volume_ratio': 1.2
        }
        
        # Warm up
        for _ in range(10):
            detector.process_features(features, features['timestamp'])
        
        # Measure processing time
        times = []
        for _ in range(100):
            start = time.perf_counter()
            detector.process_features(features, features['timestamp'])
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        avg_time = sum(times) / len(times)
        assert avg_time < 1.0, f"Average processing time {avg_time:.2f}ms exceeds 1ms requirement"
    
    def test_zero_false_negatives(self, mock_kernel):
        """Test that all valid patterns are detected (zero false negatives)."""
        detector = SynergyDetector('TestDetector', mock_kernel)
        
        # Test all valid patterns multiple times
        patterns_tested = 0
        patterns_detected = 0
        
        for pattern_type in ['TYPE_1', 'TYPE_2', 'TYPE_3', 'TYPE_4']:
            for _ in range(10):  # Test each pattern 10 times
                detector.sequence.reset()
                detector.cooldown.reset()
                
                # Get signal sequence for pattern
                pattern_map = {
                    'TYPE_1': ['mlmi', 'nwrqk', 'fvg'],
                    'TYPE_2': ['mlmi', 'fvg', 'nwrqk'],
                    'TYPE_3': ['nwrqk', 'fvg', 'mlmi'],
                    'TYPE_4': ['nwrqk', 'mlmi', 'fvg']
                }
                
                signals = pattern_map[pattern_type]
                base_time = datetime.now()
                
                # Add signals
                for i, signal_type in enumerate(signals):
                    signal = Signal(
                        signal_type=signal_type,
                        direction=1,
                        timestamp=base_time + timedelta(minutes=i*3),
                        value=70.0 if signal_type == 'mlmi' else 1.0,
                        strength=0.75
                    )
                    detector.sequence.add_signal(signal)
                
                # Check detection
                synergy = detector._check_and_create_synergy()
                patterns_tested += 1
                if synergy and synergy.synergy_type == pattern_type:
                    patterns_detected += 1
        
        # Assert zero false negatives
        assert patterns_detected == patterns_tested, \
            f"False negatives detected: {patterns_tested - patterns_detected} patterns missed"
    
    def test_event_emission(self, mock_kernel):
        """Test proper SYNERGY_DETECTED event emission."""
        detector = SynergyDetector('TestDetector', mock_kernel)
        
        # Create a complete pattern
        base_time = datetime.now()
        features = {
            'timestamp': base_time,
            'current_price': 5150.0,
            'volatility_30': 12.5,
            'volume_ratio': 1.2,
            'vix': 18,
            'trend_strength_5': 0.7,
            'trend_strength_30': 0.8,
            'lvn_nearest_price': 5145.0,
            'lvn_nearest_strength': 0.85,
            'lvn_distance_points': 5.0
        }
        
        # Build TYPE_1 pattern
        signals = [
            Signal('mlmi', 1, base_time, 75.0, 0.8),
            Signal('nwrqk', 1, base_time + timedelta(minutes=5), 1.0, 0.75),
            Signal('fvg', 1, base_time + timedelta(minutes=10), 1.0, 0.85)
        ]
        
        for signal in signals:
            detector.sequence.add_signal(signal)
        
        # Create synergy
        synergy = detector._check_and_create_synergy()
        assert synergy is not None
        
        # Emit event
        detector._emit_synergy_event(synergy, features)
        
        # Verify event was created and published
        mock_kernel.event_bus.create_event.assert_called_once()
        mock_kernel.event_bus.publish.assert_called_once()
        
        # Check event payload
        call_args = mock_kernel.event_bus.create_event.call_args
        event_type = call_args[0][0]
        payload = call_args[0][1]
        
        assert event_type == EventType.SYNERGY_DETECTED
        assert payload['synergy_type'] == 'TYPE_1'
        assert payload['direction'] == 1
        assert len(payload['signal_sequence']) == 3
        assert 'market_context' in payload
        assert 'signal_strengths' in payload
        assert 'metadata' in payload
    
    def test_time_window_enforcement(self, mock_kernel):
        """Test that signals outside the time window are rejected."""
        detector = SynergyDetector('TestDetector', mock_kernel)
        
        base_time = datetime.now()
        
        # Add first signal
        signal1 = Signal('mlmi', 1, base_time, 75.0, 0.8)
        added = detector.sequence.add_signal(signal1)
        assert added
        
        # Try to add signal after time window (>10 bars = >50 minutes)
        late_signal = Signal('nwrqk', 1, base_time + timedelta(minutes=55), 1.0, 0.8)
        added = detector.sequence.add_signal(late_signal)
        assert not added  # Should be rejected
    
    def test_cooldown_enforcement(self, mock_kernel):
        """Test that cooldown period is properly enforced."""
        detector = SynergyDetector('TestDetector', mock_kernel)
        
        base_time = datetime.now()
        
        # Start cooldown
        detector.cooldown.start_cooldown(base_time)
        assert detector.cooldown.is_in_cooldown()
        assert detector.cooldown.get_remaining_bars() == 5
        
        # Update time (3 bars = 15 minutes)
        detector.cooldown.update(base_time + timedelta(minutes=15))
        assert detector.cooldown.is_in_cooldown()
        assert detector.cooldown.get_remaining_bars() == 2
        
        # Update time (6 bars = 30 minutes)
        detector.cooldown.update(base_time + timedelta(minutes=30))
        assert not detector.cooldown.is_in_cooldown()
        assert detector.cooldown.can_emit()
    
    def test_direction_consistency(self, mock_kernel):
        """Test that mixed direction signals don't form synergy."""
        detector = SynergyDetector('TestDetector', mock_kernel)
        
        base_time = datetime.now()
        
        # Add signals with mixed directions
        signals = [
            Signal('mlmi', 1, base_time, 75.0, 0.8),  # Long
            Signal('nwrqk', -1, base_time + timedelta(minutes=5), 1.0, 0.75),  # Short
            Signal('fvg', 1, base_time + timedelta(minutes=10), 1.0, 0.85)  # Long
        ]
        
        for signal in signals:
            detector.sequence.add_signal(signal)
        
        # Check that no synergy is created due to direction mismatch
        synergy = detector._check_and_create_synergy()
        assert synergy is None
    
    def test_pattern_quality_calculation(self, mock_kernel):
        """Test pattern quality score calculation."""
        detector = SynergyDetector('TestDetector', mock_kernel)
        
        base_time = datetime.now()
        
        # Create high-quality pattern (fast completion, high strength)
        signals = [
            Signal('mlmi', 1, base_time, 75.0, 0.9),
            Signal('nwrqk', 1, base_time + timedelta(minutes=5), 1.0, 0.85),
            Signal('fvg', 1, base_time + timedelta(minutes=10), 1.0, 0.95)
        ]
        
        synergy = SynergyPattern(
            synergy_type='TYPE_1',
            direction=1,
            signals=signals,
            completion_time=base_time + timedelta(minutes=10),
            bars_to_complete=3
        )
        
        quality = detector._calculate_pattern_quality(synergy)
        
        # High quality expected (high strength, fast completion)
        assert quality > 0.8
        assert quality <= 1.0
    
    def test_feature_validation(self, mock_kernel):
        """Test feature validation in event handler."""
        detector = SynergyDetector('TestDetector', mock_kernel)
        
        # Valid features
        valid_features = {
            'timestamp': datetime.now(),
            'current_price': 5000.0,
            'mlmi_signal': 1,
            'mlmi_value': 75.0,
            'nwrqk_signal': 0,
            'nwrqk_slope': 0.1,
            'fvg_mitigation_signal': False
        }
        
        assert detector._validate_features(valid_features)
        
        # Invalid features (missing required field)
        invalid_features = {
            'timestamp': datetime.now(),
            'current_price': 5000.0,
            'mlmi_signal': 1,
            # Missing mlmi_value
            'nwrqk_signal': 0,
            'nwrqk_slope': 0.1,
            'fvg_mitigation_signal': False
        }
        
        assert not detector._validate_features(invalid_features)
    
    def test_reset_functionality(self, mock_kernel):
        """Test that reset properly clears detector state."""
        detector = SynergyDetector('TestDetector', mock_kernel)
        
        base_time = datetime.now()
        
        # Add some signals
        signal = Signal('mlmi', 1, base_time, 75.0, 0.8)
        detector.sequence.add_signal(signal)
        
        # Start cooldown
        detector.cooldown.start_cooldown(base_time)
        
        # Store features
        detector._last_features = {'test': 'data'}
        
        # Reset
        detector.reset()
        
        # Verify state is cleared
        assert len(detector.sequence.signals) == 0
        assert not detector.cooldown.is_in_cooldown()
        assert detector._last_features == {}
    
    def test_status_reporting(self, mock_kernel):
        """Test comprehensive status reporting."""
        detector = SynergyDetector('TestDetector', mock_kernel)
        
        # Process some events
        features = {
            'timestamp': datetime.now(),
            'current_price': 5000.0,
            'mlmi_signal': 1,
            'mlmi_value': 75.0,
            'nwrqk_signal': 0,
            'nwrqk_slope': 0.1,
            'fvg_mitigation_signal': False
        }
        
        detector.process_features(features, features['timestamp'])
        
        # Get status
        status = detector.get_status()
        
        # Verify status structure
        assert 'component' in status
        assert 'initialized' in status
        assert 'performance_metrics' in status
        assert 'sequence_status' in status
        assert 'cooldown_status' in status
        assert 'pattern_detectors' in status
        
        # Verify metrics
        assert status['performance_metrics']['events_processed'] == 1
        assert status['performance_metrics']['avg_processing_time_ms'] >= 0