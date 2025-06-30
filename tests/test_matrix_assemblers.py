"""
Unit tests for Matrix Assembler components.

Tests cover:
- Circular buffer mechanics
- Feature extraction and preprocessing
- Thread safety
- Edge cases and error handling
- Performance requirements
"""

import unittest
import numpy as np
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.kernel import AlgoSpaceKernel
from src.core.events import EventBus, EventType, Event
from src.matrix import MatrixAssembler30m, MatrixAssembler5m, MatrixAssemblerRegime
from src.matrix.normalizers import (
    z_score_normalize, min_max_scale, cyclical_encode,
    percentage_from_price, exponential_decay, log_transform,
    RollingNormalizer
)


class TestNormalizers(unittest.TestCase):
    """Test normalization utilities."""
    
    def test_z_score_normalize(self):
        """Test z-score normalization."""
        # Normal case
        result = z_score_normalize(5.0, mean=3.0, std=2.0)
        self.assertAlmostEqual(result, 1.0)
        
        # Zero std case
        result = z_score_normalize(5.0, mean=3.0, std=0.0)
        self.assertEqual(result, 0.0)
        
        # Array input
        values = np.array([1, 3, 5, 7, 9])
        result = z_score_normalize(values, mean=5.0, std=2.0)
        expected = np.array([-2, -1, 0, 1, 2])
        np.testing.assert_array_almost_equal(result, expected)
        
        # Clipping
        result = z_score_normalize(15.0, mean=5.0, std=2.0, clip_range=(-3, 3))
        self.assertEqual(result, 3.0)
    
    def test_min_max_scale(self):
        """Test min-max scaling."""
        # Normal case
        result = min_max_scale(5.0, min_val=0.0, max_val=10.0)
        self.assertAlmostEqual(result, 0.0)  # Maps to middle of [-1, 1]
        
        # Custom range
        result = min_max_scale(5.0, min_val=0.0, max_val=10.0, target_range=(0, 1))
        self.assertAlmostEqual(result, 0.5)
        
        # Equal min/max
        result = min_max_scale(5.0, min_val=5.0, max_val=5.0)
        self.assertEqual(result, 0.0)  # Midpoint of target range
    
    def test_cyclical_encode(self):
        """Test cyclical encoding."""
        # Hour encoding
        sin_val, cos_val = cyclical_encode(6.0, max_value=24.0)
        self.assertAlmostEqual(sin_val, 1.0, places=5)  # sin(π/2) = 1
        self.assertAlmostEqual(cos_val, 0.0, places=5)  # cos(π/2) = 0
        
        # Verify circular property
        sin_0, cos_0 = cyclical_encode(0.0, max_value=24.0)
        sin_24, cos_24 = cyclical_encode(24.0, max_value=24.0)
        self.assertAlmostEqual(sin_0, sin_24)
        self.assertAlmostEqual(cos_0, cos_24)
    
    def test_percentage_from_price(self):
        """Test percentage calculation."""
        result = percentage_from_price(105.0, reference_price=100.0)
        self.assertAlmostEqual(result, 5.0)
        
        result = percentage_from_price(95.0, reference_price=100.0)
        self.assertAlmostEqual(result, -5.0)
        
        # Clipping
        result = percentage_from_price(150.0, reference_price=100.0, clip_pct=10.0)
        self.assertEqual(result, 10.0)
    
    def test_exponential_decay(self):
        """Test exponential decay."""
        # Age 0 should give 1.0
        result = exponential_decay(0.0)
        self.assertAlmostEqual(result, 1.0)
        
        # Positive age decays
        result = exponential_decay(10.0, decay_rate=0.1)
        self.assertAlmostEqual(result, np.exp(-1.0))
        
        # Array input
        ages = np.array([0, 5, 10, 20])
        result = exponential_decay(ages, decay_rate=0.1)
        expected = np.exp(-0.1 * ages)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_rolling_normalizer(self):
        """Test rolling normalizer."""
        normalizer = RollingNormalizer(alpha=0.1)
        
        # Add samples
        values = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        for val in values:
            normalizer.update(val)
        
        # Check statistics
        self.assertGreater(normalizer.mean, 2.0)
        self.assertLess(normalizer.mean, 4.0)
        self.assertGreater(normalizer.std, 0.0)
        
        # Test normalization
        norm_val = normalizer.normalize_zscore(3.0)
        self.assertGreater(norm_val, -2.0)
        self.assertLess(norm_val, 2.0)
        
        # Test min-max
        norm_val = normalizer.normalize_minmax(3.0)
        self.assertGreaterEqual(norm_val, -1.0)
        self.assertLessEqual(norm_val, 1.0)


class TestBaseMatrixAssembler(unittest.TestCase):
    """Test base matrix assembler functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.kernel = AlgoSpaceKernel()
        self.event_bus = self.kernel.get_event_bus()
    
    def test_circular_buffer(self):
        """Test circular buffer mechanics."""
        # Create a small test assembler
        config = {
            'window_size': 5,
            'features': ['feat1', 'feat2'],
            'warmup_period': 3
        }
        
        # Create a concrete implementation for testing
        class TestAssembler(MatrixAssembler30m):
            def __init__(self, kernel, config):
                self.config = config
                self.kernel = kernel
                self.logger = kernel.logger
                self.name = "TestAssembler"
                super(MatrixAssembler30m, self).__init__(self.name, kernel, config)
        
        assembler = TestAssembler(self.kernel, config)
        
        # Verify initial state
        self.assertEqual(assembler.window_size, 5)
        self.assertEqual(assembler.n_features, 2)
        self.assertFalse(assembler.is_ready())
        
        # Simulate updates
        for i in range(7):  # More than window size
            assembler.matrix[assembler.current_index] = [i, i*2]
            assembler.current_index = (assembler.current_index + 1) % assembler.window_size
            assembler.n_updates += 1
            
            if assembler.n_updates >= assembler.window_size:
                assembler.is_full = True
            if assembler.n_updates >= assembler._warmup_period:
                assembler._is_ready = True
        
        # Check state after updates
        self.assertTrue(assembler.is_ready())
        self.assertTrue(assembler.is_full)
        self.assertEqual(assembler.n_updates, 7)
        
        # Get matrix and verify chronological order
        matrix = assembler.get_matrix()
        self.assertIsNotNone(matrix)
        self.assertEqual(matrix.shape, (5, 2))
        
        # Should contain values 2-6 (oldest to newest)
        expected_first_col = [2, 3, 4, 5, 6]
        np.testing.assert_array_equal(matrix[:, 0], expected_first_col)
    
    def test_thread_safety(self):
        """Test thread-safe access to matrix."""
        assembler = MatrixAssembler30m("Test30m", self.kernel)
        
        # Make it ready
        assembler._is_ready = True
        assembler.n_updates = 50
        
        results = []
        errors = []
        
        def reader_thread():
            """Read matrix multiple times."""
            try:
                for _ in range(100):
                    matrix = assembler.get_matrix()
                    if matrix is not None:
                        results.append(matrix.shape)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        def writer_thread():
            """Update matrix multiple times."""
            try:
                for i in range(100):
                    with assembler._lock:
                        assembler.matrix[i % assembler.window_size] = np.random.randn(8)
                        assembler.n_updates += 1
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        # Start threads
        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=reader_thread))
            threads.append(threading.Thread(target=writer_thread))
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # Check no errors occurred
        self.assertEqual(len(errors), 0)
        self.assertGreater(len(results), 0)


class TestMatrixAssembler30m(unittest.TestCase):
    """Test 30-minute matrix assembler."""
    
    def setUp(self):
        """Set up test environment."""
        self.kernel = AlgoSpaceKernel()
        self.assembler = MatrixAssembler30m("Test30m", self.kernel)
    
    def test_feature_extraction(self):
        """Test feature extraction from feature store."""
        feature_store = {
            'current_price': 100.0,
            'mlmi_value': 65.0,
            'mlmi_signal': 1,
            'nwrqk_value': 102.0,
            'nwrqk_slope': 0.5,
            'lvn_distance_points': 5.0,
            'lvn_nearest_strength': 85.0,
            'timestamp': datetime(2024, 1, 1, 14, 30)
        }
        
        features = self.assembler.extract_features(feature_store)
        
        self.assertIsNotNone(features)
        self.assertEqual(len(features), 8)
        self.assertEqual(features[0], 65.0)  # mlmi_value
        self.assertEqual(features[1], 1.0)   # mlmi_signal
        self.assertEqual(features[2], 102.0) # nwrqk_value
        self.assertAlmostEqual(features[6], 14.5)  # hour (14:30)
    
    def test_preprocessing(self):
        """Test feature preprocessing."""
        raw_features = [
            75.0,   # mlmi_value
            1.0,    # mlmi_signal
            105.0,  # nwrqk_value
            0.3,    # nwrqk_slope
            10.0,   # lvn_distance
            90.0,   # lvn_strength
            15.0,   # hour
            15.0    # hour (for cos)
        ]
        
        self.assembler.current_price = 100.0
        
        processed = self.assembler.preprocess_features(raw_features, {})
        
        self.assertEqual(len(processed), 8)
        
        # Check MLMI scaling
        self.assertAlmostEqual(processed[0], 0.5)  # (75-50)/50 = 0.5
        
        # Check signal unchanged
        self.assertEqual(processed[1], 1.0)
        
        # Check NW-RQK percentage
        self.assertAlmostEqual(processed[2], 1.0)  # 5% / 5% = 1.0
        
        # Check LVN strength scaling
        self.assertAlmostEqual(processed[5], 0.9)  # 90/100
        
        # Check time encoding
        sin_val, cos_val = cyclical_encode(15.0, 24.0)
        self.assertAlmostEqual(processed[6], sin_val)
        self.assertAlmostEqual(processed[7], cos_val)
    
    def test_missing_data_handling(self):
        """Test handling of missing data."""
        # Missing current price
        feature_store = {
            'mlmi_value': 50.0,
            'mlmi_signal': 0
        }
        
        features = self.assembler.extract_features(feature_store)
        self.assertIsNone(features)  # Should return None without price
        
        # Set price and try again
        self.assembler.current_price = 100.0
        features = self.assembler.extract_features(feature_store)
        self.assertIsNotNone(features)  # Should work with cached price


class TestMatrixAssembler5m(unittest.TestCase):
    """Test 5-minute matrix assembler."""
    
    def setUp(self):
        """Set up test environment."""
        self.kernel = AlgoSpaceKernel()
        self.assembler = MatrixAssembler5m("Test5m", self.kernel)
        self.assembler.current_price = 100.0
    
    def test_feature_extraction(self):
        """Test 5-minute feature extraction."""
        feature_store = {
            'current_price': 100.0,
            'current_volume': 1000,
            'fvg_bullish_active': True,
            'fvg_bearish_active': False,
            'fvg_nearest_level': 99.5,
            'fvg_age': 5,
            'fvg_mitigation_signal': False
        }
        
        # Prime price history for momentum
        for price in [98, 98.5, 99, 99.5, 100]:
            self.assembler.price_history.append(price)
        
        features = self.assembler.extract_features(feature_store)
        
        self.assertIsNotNone(features)
        self.assertEqual(len(features), 7)
        self.assertEqual(features[0], 1.0)  # bullish active
        self.assertEqual(features[1], 0.0)  # bearish not active
        self.assertEqual(features[3], 5.0)  # age
    
    def test_momentum_calculation(self):
        """Test price momentum calculation."""
        # Set up price history
        prices = [100, 101, 102, 101, 100, 99]
        self.assembler.price_history.clear()
        for p in prices:
            self.assembler.price_history.append(p)
        
        momentum = self.assembler._calculate_price_momentum()
        
        # Should be (99 - 100) / 100 * 100 = -1%
        self.assertAlmostEqual(momentum, -1.0, places=1)
    
    def test_fvg_preprocessing(self):
        """Test FVG-specific preprocessing."""
        raw_features = [
            1.0,    # bullish active
            0.0,    # bearish inactive
            98.0,   # fvg level
            10.0,   # age
            0.0,    # no mitigation
            2.0,    # 2% momentum
            2.5     # volume ratio
        ]
        
        self.assembler.current_price = 100.0
        processed = self.assembler.preprocess_features(raw_features, {})
        
        # Check FVG distance calculation
        self.assertAlmostEqual(processed[2], -1.0)  # -2% / 2 = -1.0
        
        # Check age decay
        expected_decay = np.exp(-1.0)  # e^(-0.1 * 10)
        self.assertAlmostEqual(processed[3], expected_decay)
        
        # Check momentum scaling
        self.assertAlmostEqual(processed[5], 0.4)  # 2% / 5% = 0.4
        
        # Check volume ratio transform
        log_ratio = np.log1p(1.5)  # log(1 + 1.5)
        expected_vol = np.tanh(log_ratio)
        self.assertAlmostEqual(processed[6], expected_vol)


class TestMatrixAssemblerRegime(unittest.TestCase):
    """Test regime detection matrix assembler."""
    
    def setUp(self):
        """Set up test environment."""
        self.kernel = AlgoSpaceKernel()
        self.assembler = MatrixAssemblerRegime("TestRegime", self.kernel)
    
    def test_dynamic_mmd_dimension(self):
        """Test dynamic MMD dimension handling."""
        # Should adapt to MMD configuration
        expected_mmd_dim = 8  # Default: signature_degree=3 -> 3*2+2=8
        total_features = expected_mmd_dim + 3  # +3 for other features
        self.assertEqual(self.assembler.n_features, total_features)
    
    def test_feature_extraction(self):
        """Test regime feature extraction."""
        # Create MMD features
        mmd_features = [0.1, -0.2, 0.3, -0.1, 0.0, 0.2, -0.3, 0.1]
        
        feature_store = {
            'mmd_features': mmd_features,
            'current_price': 100.0,
            'current_volume': 1000
        }
        
        # Prime price history
        for i in range(10):
            self.assembler.price_history.append(100 + i * 0.1)
        
        features = self.assembler.extract_features(feature_store)
        
        self.assertIsNotNone(features)
        self.assertEqual(len(features), 11)  # 8 MMD + 3 others
        
        # Check MMD features preserved
        for i in range(8):
            self.assertAlmostEqual(features[i], mmd_features[i])
    
    def test_volatility_calculation(self):
        """Test volatility calculation."""
        # Add prices with known pattern
        prices = [100]
        for i in range(30):
            # Oscillating pattern
            prices.append(100 + (i % 2) * 2)
        
        self.assembler.price_history.clear()
        for p in prices:
            self.assembler.price_history.append(p)
        
        volatility = self.assembler._calculate_volatility()
        
        # Should detect the oscillation
        self.assertGreater(volatility, 0.0)
        self.assertLess(volatility, 5.0)  # Reasonable range
    
    def test_regime_interpretation(self):
        """Test regime interpretation."""
        interpretation = self.assembler._interpret_regime(
            volatility=0.8,      # High
            volume_skew=0.5,     # Positive skew
            acceleration=0.3,    # Accelerating
            stability=0.4        # Transitioning
        )
        
        self.assertIn("High volatility", interpretation)
        self.assertIn("sporadic volume spikes", interpretation)
        self.assertIn("accelerating trend", interpretation)
        self.assertIn("transitioning regime", interpretation)


class TestIntegration(unittest.TestCase):
    """Integration tests for matrix assemblers."""
    
    def setUp(self):
        """Set up test environment."""
        self.kernel = AlgoSpaceKernel()
        self.event_bus = self.kernel.get_event_bus()
        
        # Create all assemblers
        self.assembler_30m = MatrixAssembler30m("Test30m", self.kernel)
        self.assembler_5m = MatrixAssembler5m("Test5m", self.kernel)
        self.assembler_regime = MatrixAssemblerRegime("TestRegime", self.kernel)
    
    def test_event_flow(self):
        """Test event subscription and handling."""
        # Track updates
        update_counts = {
            '30m': 0,
            '5m': 0,
            'regime': 0
        }
        
        # Override update methods to count
        original_30m = self.assembler_30m._update_matrix
        original_5m = self.assembler_5m._update_matrix
        original_regime = self.assembler_regime._update_matrix
        
        def count_30m(feature_store):
            update_counts['30m'] += 1
            return original_30m(feature_store)
        
        def count_5m(feature_store):
            update_counts['5m'] += 1
            return original_5m(feature_store)
        
        def count_regime(feature_store):
            update_counts['regime'] += 1
            return original_regime(feature_store)
        
        self.assembler_30m._update_matrix = count_30m
        self.assembler_5m._update_matrix = count_5m
        self.assembler_regime._update_matrix = count_regime
        
        # Emit test event
        test_features = {
            'current_price': 100.0,
            'current_volume': 1000,
            'mlmi_value': 50.0,
            'mlmi_signal': 0,
            'nwrqk_value': 100.0,
            'nwrqk_slope': 0.0,
            'lvn_distance_points': 5.0,
            'lvn_nearest_strength': 70.0,
            'fvg_bullish_active': False,
            'fvg_bearish_active': False,
            'fvg_nearest_level': 100.0,
            'fvg_age': 0,
            'fvg_mitigation_signal': False,
            'mmd_features': [0.0] * 8,
            'timestamp': datetime.now()
        }
        
        event = Event(EventType.INDICATORS_READY, test_features)
        self.event_bus.publish(event)
        
        # Allow time for async processing
        time.sleep(0.1)
        
        # Verify all assemblers received update
        self.assertEqual(update_counts['30m'], 1)
        self.assertEqual(update_counts['5m'], 1)
        self.assertEqual(update_counts['regime'], 1)
    
    def test_performance_requirements(self):
        """Test performance meets requirements."""
        # Prepare test data
        test_features = {
            'current_price': 100.0,
            'current_volume': 1000,
            'mlmi_value': 50.0,
            'mlmi_signal': 0,
            'nwrqk_value': 100.0,
            'nwrqk_slope': 0.0,
            'lvn_distance_points': 5.0,
            'lvn_nearest_strength': 70.0,
            'fvg_bullish_active': False,
            'fvg_bearish_active': False,
            'fvg_nearest_level': 100.0,
            'fvg_age': 0,
            'fvg_mitigation_signal': False,
            'mmd_features': [0.0] * 8,
            'timestamp': datetime.now()
        }
        
        # Warm up assemblers
        for _ in range(100):
            self.assembler_30m._update_matrix(test_features)
            self.assembler_5m._update_matrix(test_features)
            self.assembler_regime._update_matrix(test_features)
        
        # Test update latency
        start = time.perf_counter()
        for _ in range(100):
            self.assembler_30m._update_matrix(test_features)
        update_time = (time.perf_counter() - start) / 100 * 1000  # ms
        
        self.assertLess(update_time, 1.0)  # Should be < 1ms
        
        # Test matrix access latency
        start = time.perf_counter()
        for _ in range(1000):
            matrix = self.assembler_30m.get_matrix()
        access_time = (time.perf_counter() - start) / 1000 * 1000000  # μs
        
        self.assertLess(access_time, 100)  # Should be < 100μs


if __name__ == '__main__':
    unittest.main()