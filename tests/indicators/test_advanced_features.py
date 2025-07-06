"""
Comprehensive unit tests for enhanced indicator features

Tests the advanced enhancements made to the IndicatorEngine including:
- Enhanced LVN strength score calculation with historical interactions
- MMD feature extractor with 7 reference distributions
- Interaction features between indicators
- Error handling and edge cases
- Performance validation
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from collections import deque

from src.indicators.engine import IndicatorEngine
from src.core.events import EventType, Event, BarData
from src.core.kernel import AlgoSpaceKernel
from src.indicators.mmd import compute_mmd


@pytest.fixture
def mock_kernel():
    """Create a mock AlgoSpaceKernel for testing"""
    kernel = Mock(spec=AlgoSpaceKernel)
    kernel.event_bus = Mock()
    kernel.config = Mock()
    kernel.config.primary_symbol = "NQ"
    kernel.config.timeframes = ["5min", "30min"]
    kernel.config.get_section.return_value = {
        'mlmi': {'enabled': True},
        'nwrqk': {'enabled': True},
        'fvg': {'enabled': True},
        'lvn': {'enabled': True},
        'mmd': {'enabled': True}
    }
    return kernel


@pytest.fixture
def indicator_engine(mock_kernel):
    """Create an IndicatorEngine instance for testing"""
    with patch('src.indicators.engine.get_logger'):
        engine = IndicatorEngine("test_engine", mock_kernel)
        engine.event_bus = Mock()
        engine.logger = Mock()
        return engine


@pytest.fixture
def sample_price_data():
    """Create sample price data for various scenarios"""
    return {
        'trending_up': [15000, 15010, 15020, 15015, 15025, 15035, 15030, 15040],
        'trending_down': [15040, 15030, 15020, 15025, 15015, 15005, 15010, 15000],
        'ranging': [15020, 15015, 15025, 15020, 15018, 15022, 15019, 15021],
        'volatile': [15000, 15050, 15010, 15040, 15005, 15045, 15015, 15035]
    }


class TestLVNStrengthScore:
    """Test suite for enhanced LVN strength score calculation"""
    
    def test_lvn_strength_score_calculation(self, indicator_engine):
        """Test the enhanced LVN strength score with known price action scenarios"""
        lvn_price = 15000.0
        base_strength = 0.7
        
        # Test 1: First interaction with price near LVN
        current_price = 15010.0  # Within 0.25% threshold
        strength1 = indicator_engine._calculate_lvn_strength_score(
            lvn_price, base_strength, current_price
        )
        
        # Should be close to base strength on first interaction
        assert 0.6 <= strength1 <= 0.8
        assert strength1 >= base_strength * 0.8  # Should not decrease too much
        
        # Test 2: Price far from LVN (no interaction)
        far_price = 15200.0
        strength2 = indicator_engine._calculate_lvn_strength_score(
            lvn_price, base_strength, far_price
        )
        
        # Should be similar to first strength since no new interaction
        assert abs(strength2 - strength1) < 0.1
        
        # Test 3: Multiple interactions increase strength
        test_prices = [15005, 15008, 15012, 15007, 15015]
        for price in test_prices:
            indicator_engine._calculate_lvn_strength_score(
                lvn_price, base_strength, price
            )
        
        strength3 = indicator_engine._calculate_lvn_strength_score(
            lvn_price, base_strength, 15010.0
        )
        
        # Should increase with multiple tests
        assert strength3 > strength1
        assert strength3 <= 1.0
        
        # Test 4: Verify interaction history is recorded
        assert lvn_price in indicator_engine.lvn_interaction_history
        interactions = indicator_engine.lvn_interaction_history[lvn_price]
        assert len(interactions) >= 5  # At least 5 interactions recorded
        
        # Test 5: Verify interaction data structure
        latest_interaction = interactions[-1]
        assert 'timestamp' in latest_interaction
        assert 'test_price' in latest_interaction
        assert 'approached_from' in latest_interaction
        assert 'volume' in latest_interaction
        assert latest_interaction['approached_from'] in ['above', 'below']
        
    def test_lvn_strength_score_edge_cases(self, indicator_engine):
        """Test edge cases for LVN strength score calculation"""
        # Test 1: Zero LVN price
        strength = indicator_engine._calculate_lvn_strength_score(0.0, 0.7, 15000.0)
        assert strength == 0.0
        
        # Test 2: Very small base strength
        strength = indicator_engine._calculate_lvn_strength_score(15000.0, 0.001, 15000.0)
        assert strength >= 0.001
        assert strength <= 1.0
        
        # Test 3: Maximum base strength
        strength = indicator_engine._calculate_lvn_strength_score(15000.0, 1.0, 15000.0)
        assert strength <= 1.0
        
        # Test 4: Negative price (should handle gracefully)
        strength = indicator_engine._calculate_lvn_strength_score(-1000.0, 0.5, 15000.0)
        assert strength >= 0.0
        
    def test_lvn_interaction_history_cleanup(self, indicator_engine):
        """Test that old interactions are cleaned up properly"""
        lvn_price = 15000.0
        base_strength = 0.7
        
        # Create old interaction (8 days ago)
        old_timestamp = datetime.now() - timedelta(days=8)
        indicator_engine.lvn_interaction_history[lvn_price] = [{
            'timestamp': old_timestamp,
            'test_price': 15005.0,
            'approached_from': 'above',
            'volume': 10000
        }]
        
        # Add new interaction
        indicator_engine._calculate_lvn_strength_score(lvn_price, base_strength, 15010.0)
        
        # Old interaction should be cleaned up
        interactions = indicator_engine.lvn_interaction_history[lvn_price]
        assert len(interactions) == 1  # Only the new interaction
        assert interactions[0]['timestamp'] > old_timestamp
        
    def test_rejection_factor_calculation(self, indicator_engine):
        """Test that rejection factor is calculated correctly"""
        lvn_price = 15000.0
        
        # Simulate sequence of price rejections
        timestamps = [
            datetime.now() - timedelta(hours=4),
            datetime.now() - timedelta(hours=3),
            datetime.now() - timedelta(hours=2),
            datetime.now() - timedelta(hours=1)
        ]
        
        # Create interaction history with strong rejections
        indicator_engine.lvn_interaction_history[lvn_price] = [
            {
                'timestamp': timestamps[0],
                'test_price': 15005.0,  # Close to LVN
                'approached_from': 'above',
                'volume': 10000
            },
            {
                'timestamp': timestamps[1],
                'test_price': 15050.0,  # Rejected strongly
                'approached_from': 'above',
                'volume': 12000
            },
            {
                'timestamp': timestamps[2],
                'test_price': 14995.0,  # Close to LVN
                'approached_from': 'below',
                'volume': 11000
            },
            {
                'timestamp': timestamps[3],
                'test_price': 14950.0,  # Rejected strongly
                'approached_from': 'below',
                'volume': 13000
            }
        ]
        
        # Calculate strength - should be enhanced due to rejections
        strength = indicator_engine._calculate_lvn_strength_score(lvn_price, 0.5, 15000.0)
        
        # Should be higher than base strength due to rejection pattern
        assert strength > 0.5


class TestMMDFeatureVector:
    """Test suite for MMD feature extractor enhancements"""
    
    def test_mmd_feature_vector_shape(self, indicator_engine):
        """Test the MMD feature extractor output shape and contents"""
        # Create sample bar data
        bar_data = BarData(
            symbol="NQ",
            timestamp=datetime.now(),
            open=15000.0,
            high=15050.0,
            low=14950.0,
            close=15025.0,
            volume=10000,
            timeframe=30
        )
        
        # Populate history with enough data
        for i in range(100):
            hist_bar = BarData(
                symbol="NQ",
                timestamp=datetime.now() - timedelta(minutes=30 * i),
                open=15000.0 + np.random.normal(0, 10),
                high=15050.0 + np.random.normal(0, 10),
                low=14950.0 + np.random.normal(0, 10),
                close=15025.0 + np.random.normal(0, 10),
                volume=10000 + int(np.random.normal(0, 1000)),
                timeframe=30
            )
            indicator_engine.history_30m.append(hist_bar)
        
        # Mock the base MMD calculator
        indicator_engine.mmd.calculate_30m = Mock(return_value={
            'mmd_features': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
        })
        
        # Calculate enhanced MMD features
        result = indicator_engine._calculate_enhanced_mmd(bar_data)
        
        # Test shape and structure
        assert 'mmd_features' in result
        features = result['mmd_features']
        assert len(features) == 13
        assert isinstance(features, np.ndarray)
        
        # First 7 features should be MMD scores (non-negative)
        for i in range(7):
            assert features[i] >= 0.0
        
        # Last 6 features should be from base MMD calculator
        expected_statistical = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
        np.testing.assert_array_equal(features[7:], expected_statistical)
        
    def test_mmd_reference_distributions(self, indicator_engine):
        """Test MMD reference distributions are properly configured"""
        distributions = indicator_engine.mmd_reference_distributions
        
        # Should have exactly 7 reference distributions
        assert len(distributions) == 7
        
        # Each distribution should have correct shape
        for i, dist in enumerate(distributions):
            assert dist.shape == (100, 4)  # 100 samples, 4 features each
            assert not np.isnan(dist).any()
            assert not np.isinf(dist).any()
        
        # Test specific distribution characteristics
        # Distribution 0: Strong Trending Up
        trend_up = distributions[0]
        assert np.mean(trend_up[:, 0]) > 0  # Positive returns
        
        # Distribution 1: Strong Trending Down
        trend_down = distributions[1]
        assert np.mean(trend_down[:, 0]) < 0  # Negative returns
        
        # Distribution 2: Ranging (should have low mean return)
        ranging = distributions[2]
        assert abs(np.mean(ranging[:, 0])) < 0.002
        
        # Distribution 3: High Volatility
        high_vol = distributions[3]
        assert np.mean(high_vol[:, 3]) > np.mean(distributions[4][:, 3])  # Higher than low vol
        
        # Distribution 4: Low Volatility
        low_vol = distributions[4]
        assert np.mean(low_vol[:, 3]) < 0.01  # Should be low volatility
        
    def test_mmd_calculation_insufficient_data(self, indicator_engine):
        """Test MMD calculation when insufficient historical data"""
        # Clear history
        indicator_engine.history_30m.clear()
        
        # Add only a few bars (less than 100)
        for i in range(10):
            bar = BarData(
                symbol="NQ",
                timestamp=datetime.now() - timedelta(minutes=30 * i),
                open=15000.0,
                high=15050.0,
                low=14950.0,
                close=15025.0,
                volume=10000,
                timeframe=30
            )
            indicator_engine.history_30m.append(bar)
        
        # Mock base MMD calculator
        base_features = np.ones(13) * 0.5
        indicator_engine.mmd.calculate_30m = Mock(return_value={
            'mmd_features': base_features
        )}
        
        bar_data = BarData(
            symbol="NQ",
            timestamp=datetime.now(),
            open=15000.0,
            high=15050.0,
            low=14950.0,
            close=15025.0,
            volume=10000,
            timeframe=30
        )
        
        # Should fall back to base features
        result = indicator_engine._calculate_enhanced_mmd(bar_data)
        np.testing.assert_array_equal(result['mmd_features'], base_features)
        
    def test_mmd_score_calculation(self, indicator_engine, sample_price_data):
        """Test MMD score calculation with known data patterns"""
        # Create strongly trending data
        trending_data = np.array([
            [0.005, 0.0049, 0.003, 0.02],  # Strong positive trend
            [0.004, 0.0039, 0.002, 0.018],
            [0.006, 0.0058, 0.004, 0.022]
        ])
        
        # Create ranging data
        ranging_data = np.array([
            [0.0005, 0.0005, 0.001, 0.008],  # Low volatility ranging
            [-0.0003, -0.0003, 0.0008, 0.007],
            [0.0002, 0.0002, 0.0012, 0.009]
        ])
        
        # MMD between trending and ranging should be > 0
        mmd_score = compute_mmd(trending_data, ranging_data, sigma=1.0)
        assert mmd_score > 0.0
        
        # MMD between identical data should be close to 0
        mmd_identical = compute_mmd(trending_data, trending_data, sigma=1.0)
        assert mmd_identical < 0.1  # Should be very small
        
    def test_mmd_error_handling(self, indicator_engine):
        """Test MMD calculation error handling"""
        # Test with invalid bar data
        invalid_bar = BarData(
            symbol="NQ",
            timestamp=datetime.now(),
            open=float('nan'),
            high=15050.0,
            low=14950.0,
            close=15025.0,
            volume=10000,
            timeframe=30
        )
        
        # Should handle gracefully and return zeros
        result = indicator_engine._calculate_enhanced_mmd(invalid_bar)
        assert 'mmd_features' in result
        assert len(result['mmd_features']) == 13
        np.testing.assert_array_equal(result['mmd_features'], np.zeros(13))


class TestInteractionFeatures:
    """Test suite for interaction features between indicators"""
    
    def test_interaction_features_are_created(self, indicator_engine):
        """Test that IndicatorEngine creates interaction features properly"""
        # Set up base indicator values
        indicator_engine.feature_store['mlmi_value'] = 5.0
        indicator_engine.feature_store['nwrqk_value'] = 3.0
        
        # Calculate interaction features
        indicator_engine._calculate_interaction_features()
        
        # Test MLMI minus NWRQK
        expected_diff = 5.0 - 3.0
        assert indicator_engine.feature_store['mlmi_minus_nwrqk'] == expected_diff
        
        # Test MLMI divided by NWRQK
        expected_div = 5.0 / 3.0
        assert indicator_engine.feature_store['mlmi_div_nwrqk'] == expected_div
        
    def test_interaction_features_zero_division_protection(self, indicator_engine):
        """Test zero division protection in interaction features"""
        # Test case 1: Positive MLMI, zero NWRQK
        indicator_engine.feature_store['mlmi_value'] = 5.0
        indicator_engine.feature_store['nwrqk_value'] = 0.0
        
        indicator_engine._calculate_interaction_features()
        
        assert indicator_engine.feature_store['mlmi_minus_nwrqk'] == 5.0
        assert indicator_engine.feature_store['mlmi_div_nwrqk'] == 1.0
        
        # Test case 2: Negative MLMI, zero NWRQK
        indicator_engine.feature_store['mlmi_value'] = -5.0
        indicator_engine.feature_store['nwrqk_value'] = 0.0
        
        indicator_engine._calculate_interaction_features()
        
        assert indicator_engine.feature_store['mlmi_minus_nwrqk'] == -5.0
        assert indicator_engine.feature_store['mlmi_div_nwrqk'] == -1.0
        
        # Test case 3: Zero MLMI, zero NWRQK
        indicator_engine.feature_store['mlmi_value'] = 0.0
        indicator_engine.feature_store['nwrqk_value'] = 0.0
        
        indicator_engine._calculate_interaction_features()
        
        assert indicator_engine.feature_store['mlmi_minus_nwrqk'] == 0.0
        assert indicator_engine.feature_store['mlmi_div_nwrqk'] == 0.0
        
        # Test case 4: Very small NWRQK (below threshold)
        indicator_engine.feature_store['mlmi_value'] = 2.0
        indicator_engine.feature_store['nwrqk_value'] = 1e-10
        
        indicator_engine._calculate_interaction_features()
        
        assert indicator_engine.feature_store['mlmi_div_nwrqk'] == 1.0
        
    def test_interaction_features_various_scenarios(self, indicator_engine):
        """Test interaction features with various market scenarios"""
        test_cases = [
            # (mlmi, nwrqk, expected_diff, expected_div)
            (10.0, 5.0, 5.0, 2.0),
            (-10.0, 5.0, -15.0, -2.0),
            (10.0, -5.0, 15.0, -2.0),
            (-10.0, -5.0, -5.0, 2.0),
            (0.5, 0.1, 0.4, 5.0),
            (100.0, 1.0, 99.0, 100.0)
        ]
        
        for mlmi, nwrqk, expected_diff, expected_div in test_cases:
            indicator_engine.feature_store['mlmi_value'] = mlmi
            indicator_engine.feature_store['nwrqk_value'] = nwrqk
            
            indicator_engine._calculate_interaction_features()
            
            assert abs(indicator_engine.feature_store['mlmi_minus_nwrqk'] - expected_diff) < 1e-10
            assert abs(indicator_engine.feature_store['mlmi_div_nwrqk'] - expected_div) < 1e-10
            
    def test_interaction_features_error_handling(self, indicator_engine):
        """Test error handling in interaction features calculation"""
        # Remove required keys from feature store
        if 'mlmi_value' in indicator_engine.feature_store:
            del indicator_engine.feature_store['mlmi_value']
        if 'nwrqk_value' in indicator_engine.feature_store:
            del indicator_engine.feature_store['nwrqk_value']
        
        # Should handle missing keys gracefully
        indicator_engine._calculate_interaction_features()
        
        # Should create the interaction features with default values
        assert 'mlmi_minus_nwrqk' in indicator_engine.feature_store
        assert 'mlmi_div_nwrqk' in indicator_engine.feature_store


class TestPerformanceAndValidation:
    """Test suite for performance validation and edge cases"""
    
    def test_feature_store_integrity(self, indicator_engine):
        """Test Feature Store maintains integrity under various conditions"""
        # Initialize with known state
        original_keys = set(indicator_engine.feature_store.keys())
        
        # Simulate multiple updates
        indicator_engine.feature_store['mlmi_value'] = 1.0
        indicator_engine.feature_store['nwrqk_value'] = 2.0
        indicator_engine._calculate_interaction_features()
        
        # Check that no keys were lost
        assert set(indicator_engine.feature_store.keys()) >= original_keys
        
        # Check that feature_count exists
        assert 'feature_count' in indicator_engine.feature_store
        
    def test_memory_management(self, indicator_engine):
        """Test that memory usage stays bounded"""
        initial_history_size = len(indicator_engine.history_30m)
        
        # Add many bars to test maxlen enforcement
        for i in range(200):
            bar = BarData(
                symbol="NQ",
                timestamp=datetime.now() - timedelta(minutes=30 * i),
                open=15000.0,
                high=15050.0,
                low=14950.0,
                close=15025.0,
                volume=10000,
                timeframe=30
            )
            indicator_engine.history_30m.append(bar)
        
        # History should be bounded by maxlen
        assert len(indicator_engine.history_30m) == 100  # maxlen from initialization
        
        # Test LVN interaction history cleanup
        # Add many LVN levels
        for price in range(14900, 15100, 5):
            indicator_engine._calculate_lvn_strength_score(float(price), 0.5, 15000.0)
        
        # Should not grow unbounded
        total_interactions = sum(
            len(interactions) for interactions in indicator_engine.lvn_interaction_history.values()
        )
        assert total_interactions < 1000  # Reasonable bound
        
    def test_nan_and_inf_handling(self, indicator_engine):
        """Test handling of NaN and infinity values"""
        # Test with NaN values
        indicator_engine.feature_store['mlmi_value'] = float('nan')
        indicator_engine.feature_store['nwrqk_value'] = 5.0
        
        indicator_engine._calculate_interaction_features()
        
        # Should handle NaN gracefully
        assert not np.isnan(indicator_engine.feature_store['mlmi_minus_nwrqk'])
        assert not np.isnan(indicator_engine.feature_store['mlmi_div_nwrqk'])
        
        # Test with infinity values
        indicator_engine.feature_store['mlmi_value'] = float('inf')
        indicator_engine.feature_store['nwrqk_value'] = 5.0
        
        indicator_engine._calculate_interaction_features()
        
        # Should handle infinity gracefully
        assert not np.isinf(indicator_engine.feature_store['mlmi_minus_nwrqk'])
        assert not np.isinf(indicator_engine.feature_store['mlmi_div_nwrqk'])
        
    def test_concurrent_feature_updates(self, indicator_engine):
        """Test thread safety of feature updates"""
        import threading
        import time
        
        def update_features(thread_id):
            for i in range(10):
                indicator_engine.feature_store[f'test_feature_{thread_id}'] = i
                indicator_engine._calculate_interaction_features()
                time.sleep(0.001)  # Small delay to increase chance of race conditions
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=update_features, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Feature store should be in consistent state
        assert 'mlmi_minus_nwrqk' in indicator_engine.feature_store
        assert 'mlmi_div_nwrqk' in indicator_engine.feature_store
        
    @pytest.mark.asyncio
    async def test_async_feature_store_updates(self, indicator_engine):
        """Test asynchronous feature store updates"""
        # Mock publish_event to avoid actual event emission
        indicator_engine.publish_event = Mock()
        
        # Test 5-minute update
        features_5min = {
            'fvg_bullish_active': True,
            'fvg_bearish_active': False,
            'fvg_nearest_level': 15010.0
        }
        
        await indicator_engine._update_feature_store_5min(features_5min, datetime.now())
        
        assert indicator_engine.feature_store['fvg_bullish_active'] == True
        assert indicator_engine.feature_store['fvg_bearish_active'] == False
        assert indicator_engine.feature_store['fvg_nearest_level'] == 15010.0
        
        # Test 30-minute update
        features_30min = {
            'mlmi_value': 5.0,
            'nwrqk_value': 3.0,
            'lvn_nearest_strength': 0.8
        }
        
        indicator_engine.has_30min_data = True
        await indicator_engine._update_feature_store_30min(features_30min, datetime.now())
        
        assert indicator_engine.feature_store['mlmi_value'] == 5.0
        assert indicator_engine.feature_store['nwrqk_value'] == 3.0
        assert indicator_engine.feature_store['lvn_nearest_strength'] == 0.8
        
        # Interaction features should be calculated
        assert indicator_engine.feature_store['mlmi_minus_nwrqk'] == 2.0
        assert indicator_engine.feature_store['mlmi_div_nwrqk'] == 5.0 / 3.0


class TestRegressionAndValidation:
    """Test suite for regression testing and validation"""
    
    def test_feature_output_ranges(self, indicator_engine):
        """Test that feature outputs are within expected ranges"""
        # Test LVN strength score range
        for _ in range(10):
            lvn_price = 15000.0 + np.random.normal(0, 100)
            base_strength = np.random.uniform(0.1, 1.0)
            current_price = 15000.0 + np.random.normal(0, 50)
            
            strength = indicator_engine._calculate_lvn_strength_score(
                lvn_price, base_strength, current_price
            )
            
            assert 0.0 <= strength <= 1.0
        
        # Test MMD scores are non-negative
        distributions = indicator_engine.mmd_reference_distributions
        for dist in distributions:
            assert np.all(dist >= -1.0)  # Reasonable lower bound for normalized features
            assert np.all(np.isfinite(dist))
        
    def test_deterministic_behavior(self, indicator_engine):
        """Test that calculations are deterministic for same inputs"""
        # Test LVN strength score determinism
        lvn_price = 15000.0
        base_strength = 0.7
        current_price = 15010.0
        
        # Clear any existing history
        indicator_engine.lvn_interaction_history.clear()
        
        # Calculate twice with same inputs
        strength1 = indicator_engine._calculate_lvn_strength_score(
            lvn_price, base_strength, current_price
        )
        strength2 = indicator_engine._calculate_lvn_strength_score(
            lvn_price, base_strength, current_price
        )
        
        # Should be different due to recorded interaction, but predictably so
        assert strength2 >= strength1  # Strength should increase or stay same
        
        # Test interaction features determinism
        indicator_engine.feature_store['mlmi_value'] = 5.0
        indicator_engine.feature_store['nwrqk_value'] = 3.0
        
        indicator_engine._calculate_interaction_features()
        result1 = {
            'diff': indicator_engine.feature_store['mlmi_minus_nwrqk'],
            'div': indicator_engine.feature_store['mlmi_div_nwrqk']
        }
        
        indicator_engine._calculate_interaction_features()
        result2 = {
            'diff': indicator_engine.feature_store['mlmi_minus_nwrqk'],
            'div': indicator_engine.feature_store['mlmi_div_nwrqk']
        }
        
        assert result1 == result2  # Should be identical
        
    def test_backward_compatibility(self, indicator_engine):
        """Test that enhanced features don't break existing functionality"""
        # Test that all original feature store keys still exist
        expected_keys = [
            'mlmi_value', 'mlmi_signal',
            'nwrqk_value', 'nwrqk_slope', 'nwrqk_signal',
            'lvn_nearest_price', 'lvn_nearest_strength', 'lvn_distance_points',
            'fvg_bullish_active', 'fvg_bearish_active', 'fvg_nearest_level',
            'fvg_age', 'fvg_mitigation_signal',
            'mmd_features',
            'mlmi_minus_nwrqk', 'mlmi_div_nwrqk',
            'last_update_5min', 'last_update_30min',
            'calculation_status', 'feature_count'
        ]
        
        for key in expected_keys:
            assert key in indicator_engine.feature_store, f"Missing key: {key}"
        
        # Test that MMD features still has correct dimensionality
        assert len(indicator_engine.feature_store['mmd_features']) == 13
        
        # Test that reference distributions are properly loaded
        assert len(indicator_engine.mmd_reference_distributions) == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])