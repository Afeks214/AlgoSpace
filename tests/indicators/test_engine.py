"""
Unit tests for IndicatorEngine component

Tests the central feature calculation engine including:
- Dual timeframe processing (5min and 30min)
- Heiken Ashi conversion
- Advanced feature engineering (LVN strength, MMD vectors, interaction features)
- Event emission and Feature Store management
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import numpy as np
from collections import deque

from src.indicators.engine import IndicatorEngine
from src.core.events import EventType, Event, BarData
from src.core.kernel import AlgoSpaceKernel


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
def sample_bar_data():
    """Create sample bar data for testing"""
    return BarData(
        symbol="NQ",
        timestamp=datetime.now(),
        open=15000.0,
        high=15050.0,
        low=14950.0,
        close=15025.0,
        volume=10000,
        timeframe=30
    )


@pytest.fixture
def sample_5min_bar_data():
    """Create sample 5-minute bar data for testing"""
    return BarData(
        symbol="NQ",
        timestamp=datetime.now(),
        open=15000.0,
        high=15020.0,
        low=14990.0,
        close=15015.0,
        volume=5000,
        timeframe=5
    )


class TestIndicatorEngine:
    """Test suite for IndicatorEngine"""
    
    def test_initialization(self, indicator_engine):
        """Test proper initialization of IndicatorEngine"""
        assert indicator_engine.symbol == "NQ"
        assert indicator_engine.timeframes == ["5min", "30min"]
        assert isinstance(indicator_engine.feature_store, dict)
        assert isinstance(indicator_engine.history_5m, deque)
        assert isinstance(indicator_engine.history_30m, deque)
        assert isinstance(indicator_engine.ha_history_30m, deque)
        assert len(indicator_engine.mmd_reference_distributions) == 7
        assert indicator_engine.calculations_5min == 0
        assert indicator_engine.calculations_30min == 0
        
    def test_feature_store_initialization(self, indicator_engine):
        """Test Feature Store is initialized with correct keys"""
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
            assert key in indicator_engine.feature_store
            
        # Check MMD features is 13-dimensional
        assert len(indicator_engine.feature_store['mmd_features']) == 13
        
    def test_heiken_ashi_conversion(self, indicator_engine, sample_bar_data):
        """Test Heiken Ashi conversion logic"""
        # First bar
        ha_bar = indicator_engine._convert_to_heiken_ashi(sample_bar_data)
        
        expected_ha_close = (15000.0 + 15050.0 + 14950.0 + 15025.0) / 4
        expected_ha_open = (15000.0 + 15025.0) / 2  # First bar
        
        assert ha_bar['close'] == expected_ha_close
        assert ha_bar['open'] == expected_ha_open
        assert ha_bar['high'] == max(15050.0, expected_ha_open, expected_ha_close)
        assert ha_bar['low'] == min(14950.0, expected_ha_open, expected_ha_close)
        assert ha_bar['volume'] == 10000
        
        # Add to history and test second bar
        indicator_engine.ha_history_30m.append(ha_bar)
        
        second_bar = BarData(
            symbol="NQ",
            timestamp=datetime.now(),
            open=15025.0,
            high=15075.0,
            low=15000.0,
            close=15050.0,
            volume=12000,
            timeframe=30
        )
        
        ha_bar2 = indicator_engine._convert_to_heiken_ashi(second_bar)
        expected_ha_open2 = (ha_bar['open'] + ha_bar['close']) / 2
        
        assert ha_bar2['open'] == expected_ha_open2
        
    def test_bar_validation(self, indicator_engine, sample_bar_data):
        """Test bar data validation"""
        # Valid bar
        assert indicator_engine._validate_bar_data(sample_bar_data, '30min') == True
        
        # Wrong symbol
        wrong_symbol_bar = BarData(
            symbol="ES",
            timestamp=datetime.now(),
            open=4000.0,
            high=4050.0,
            low=3950.0,
            close=4025.0,
            volume=10000,
            timeframe=30
        )
        assert indicator_engine._validate_bar_data(wrong_symbol_bar, '30min') == False
        
    def test_lvn_strength_score_calculation(self, indicator_engine):
        """Test enhanced LVN strength score calculation"""
        lvn_price = 15000.0
        base_strength = 0.7
        current_price = 15010.0  # Within test threshold
        
        # First interaction
        strength1 = indicator_engine._calculate_lvn_strength_score(
            lvn_price, base_strength, current_price
        )
        
        # Should be mostly base strength on first interaction
        assert 0.6 <= strength1 <= 0.8
        
        # Simulate multiple interactions
        for i in range(5):
            indicator_engine._calculate_lvn_strength_score(
                lvn_price, base_strength, current_price + i
            )
        
        # After multiple tests, strength should increase
        strength2 = indicator_engine._calculate_lvn_strength_score(
            lvn_price, base_strength, current_price
        )
        
        assert strength2 > strength1
        assert strength2 <= 1.0
        
    def test_mmd_reference_distributions(self, indicator_engine):
        """Test MMD reference distributions are properly loaded"""
        distributions = indicator_engine.mmd_reference_distributions
        
        assert len(distributions) == 7
        
        for i, dist in enumerate(distributions):
            assert dist.shape == (100, 4)  # 100 samples, 4 features
            assert not np.isnan(dist).any()
            
        # Test specific characteristics
        # Distribution 0: Strong Trending Up (returns should be positive)
        assert np.mean(distributions[0][:, 0]) > 0
        
        # Distribution 1: Strong Trending Down (returns should be negative)
        assert np.mean(distributions[1][:, 0]) < 0
        
        # Distribution 3: High Volatility (higher volatility values)
        assert np.mean(distributions[3][:, 3]) > np.mean(distributions[4][:, 3])
        
    def test_interaction_features_calculation(self, indicator_engine):
        """Test calculation of interaction features"""
        # Set base values
        indicator_engine.feature_store['mlmi_value'] = 5.0
        indicator_engine.feature_store['nwrqk_value'] = 3.0
        
        indicator_engine._calculate_interaction_features()
        
        assert indicator_engine.feature_store['mlmi_minus_nwrqk'] == 2.0
        assert indicator_engine.feature_store['mlmi_div_nwrqk'] == 5.0 / 3.0
        
        # Test zero division protection
        indicator_engine.feature_store['nwrqk_value'] = 0.0
        indicator_engine._calculate_interaction_features()
        
        assert indicator_engine.feature_store['mlmi_div_nwrqk'] == 1.0  # Positive mlmi
        
        indicator_engine.feature_store['mlmi_value'] = -5.0
        indicator_engine._calculate_interaction_features()
        
        assert indicator_engine.feature_store['mlmi_div_nwrqk'] == -1.0  # Negative mlmi
        
    @pytest.mark.asyncio
    async def test_indicator_engine_flow(self, indicator_engine, sample_bar_data):
        """Test complete flow: bar processing and event emission"""
        # Mock the indicator calculators
        indicator_engine.mlmi.calculate_30m = Mock(return_value={
            'mlmi_value': 3.5,
            'mlmi_signal': 1
        )}
        indicator_engine.nwrqk.calculate_30m = Mock(return_value={
            'nwrqk_value': 15020.0,
            'nwrqk_signal': 0
        })
        indicator_engine.lvn.calculate_30m = Mock(return_value={
            'nearest_lvn_price': 15000.0,
            'nearest_lvn_strength': 0.8,
            'distance_to_nearest_lvn': 25.0
        })
        indicator_engine.mmd.calculate_30m = Mock(return_value={
            'mmd_features': np.ones(13) * 0.5
        })
        
        # Mock event publishing
        indicator_engine.publish_event = Mock()
        
        # Populate history to meet minimum requirements
        for i in range(50):
            bar = BarData(
                symbol="NQ",
                timestamp=datetime.now() - timedelta(minutes=30 * i),
                open=15000.0 + i,
                high=15050.0 + i,
                low=14950.0 + i,
                close=15025.0 + i,
                volume=10000,
                timeframe=30
            )
            indicator_engine.history_30m.append(bar)
            ha_bar = indicator_engine._convert_to_heiken_ashi(bar)
            indicator_engine.ha_history_30m.append(ha_bar)
        
        # Create 30-minute bar event
        event = Event(
            type=EventType.NEW_30MIN_BAR,
            payload=sample_bar_data,
            timestamp=datetime.now()
        )
        
        # Process the bar
        indicator_engine._on_30min_bar(event)
        
        # Allow async tasks to complete
        await asyncio.sleep(0.1)
        
        # Verify INDICATORS_READY event was published
        indicator_engine.publish_event.assert_called()
        call_args = indicator_engine.publish_event.call_args
        
        assert call_args[0][0] == EventType.INDICATORS_READY
        
        # Check the payload
        payload = call_args[0][1]
        
        # Verify expected keys in payload
        assert 'mlmi_value' in payload
        assert 'mlmi_signal' in payload
        assert 'nwrqk_value' in payload
        assert 'nwrqk_signal' in payload
        assert 'lvn_nearest_price' in payload
        assert 'lvn_nearest_strength' in payload
        assert 'mmd_features' in payload
        assert 'mlmi_minus_nwrqk' in payload
        assert 'mlmi_div_nwrqk' in payload
        assert 'feature_count' in payload
        assert 'calculation_status' in payload
        
        # Verify calculated values
        assert payload['mlmi_value'] == 3.5
        assert payload['mlmi_signal'] == 1
        assert payload['nwrqk_value'] == 15020.0
        assert payload['lvn_nearest_strength'] == 0.8
        assert len(payload['mmd_features']) == 13
        assert payload['mlmi_minus_nwrqk'] == 3.5 - 15020.0
        assert payload['calculation_status'] == 'complete'
        
    @pytest.mark.asyncio
    async def test_5min_bar_processing(self, indicator_engine, sample_5min_bar_data):
        """Test 5-minute bar processing for FVG detection"""
        # Mock FVG detector
        indicator_engine.fvg.calculate_5m = Mock(return_value={
            'fvg_bullish_active': True,
            'fvg_bearish_active': False,
            'fvg_nearest_level': 15010.0,
            'fvg_age': 2,
            'fvg_mitigation_signal': False
        )}
        
        # Mock event publishing
        indicator_engine.publish_event = Mock()
        
        # Add some history
        for i in range(5):
            bar = BarData(
                symbol="NQ",
                timestamp=datetime.now() - timedelta(minutes=5 * i),
                open=15000.0 + i,
                high=15020.0 + i,
                low=14990.0 + i,
                close=15015.0 + i,
                volume=5000,
                timeframe=5
            )
            indicator_engine.history_5m.append(bar)
        
        # Process 5-minute bar
        event = Event(
            type=EventType.NEW_5MIN_BAR,
            payload=sample_5min_bar_data,
            timestamp=datetime.now()
        )
        
        indicator_engine._on_5min_bar(event)
        
        # Allow async tasks to complete
        await asyncio.sleep(0.1)
        
        # Verify calculations
        assert indicator_engine.calculations_5min == 1
        
    def test_fvg_tracking(self, indicator_engine):
        """Test FVG tracking and aging"""
        # Add some FVGs
        indicator_engine.active_fvgs = [
            {'type': 'bullish', 'upper_bound': 15020, 'lower_bound': 15010, 'creation_bar': 0},
            {'type': 'bearish', 'upper_bound': 14990, 'lower_bound': 14980, 'creation_bar': 10},
            {'type': 'bullish', 'upper_bound': 15040, 'lower_bound': 15030, 'creation_bar': 45},
        ]
        
        # Create bar history
        bars = []
        for i in range(60):
            bars.append(BarData(
                symbol="NQ",
                timestamp=datetime.now() - timedelta(minutes=5 * (60-i)),
                open=15000.0,
                high=15020.0,
                low=14990.0,
                close=15015.0,
                volume=5000,
                timeframe=5
            ))
        
        # Update tracking
        indicator_engine._update_fvg_tracking(bars)
        
        # Old FVG (creation_bar=0) should be removed (>50 bars old)
        assert len(indicator_engine.active_fvgs) == 2
        assert all(fvg['creation_bar'] >= 10 for fvg in indicator_engine.active_fvgs)
        
    def test_enhanced_mmd_calculation(self, indicator_engine, sample_bar_data):
        """Test enhanced MMD calculation with 7 reference distributions"""
        # Mock base MMD result
        indicator_engine.mmd.calculate_30m = Mock(return_value={
            'mmd_features': np.array([0, 0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        )}
        
        # Populate enough history
        for i in range(100):
            bar = BarData(
                symbol="NQ",
                timestamp=datetime.now() - timedelta(minutes=30 * i),
                open=15000.0 + np.random.normal(0, 10),
                high=15050.0 + np.random.normal(0, 10),
                low=14950.0 + np.random.normal(0, 10),
                close=15025.0 + np.random.normal(0, 10),
                volume=10000,
                timeframe=30
            )
            indicator_engine.history_30m.append(bar)
        
        # Calculate enhanced MMD
        mmd_result = indicator_engine._calculate_enhanced_mmd(sample_bar_data)
        
        assert 'mmd_features' in mmd_result
        features = mmd_result['mmd_features']
        
        # Should have 13 features
        assert len(features) == 13
        
        # First 7 should be MMD scores against reference distributions
        for i in range(7):
            assert features[i] >= 0  # MMD scores are non-negative
            
        # Last 6 should be statistical features from base
        for i in range(7, 13):
            assert features[i] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6][i-7]
            
    @pytest.mark.asyncio
    async def test_concurrent_updates(self, indicator_engine):
        """Test thread-safe concurrent updates to Feature Store"""
        async def update_5min():
            await indicator_engine._update_feature_store_5min(
                {'fvg_bullish_active': True}, 
                datetime.now()
            )
            
        async def update_30min():
            await indicator_engine._update_feature_store_30min(
                {'mlmi_value': 5.0, 'nwrqk_value': 3.0}, 
                datetime.now()
            )
        
        # Mock event publishing
        indicator_engine.publish_event = Mock()
        indicator_engine.has_30min_data = True
        
        # Run concurrent updates
        await asyncio.gather(update_5min(), update_30min())
        
        # Both updates should be reflected
        assert indicator_engine.feature_store['fvg_bullish_active'] == True
        assert indicator_engine.feature_store['mlmi_value'] == 5.0
        assert indicator_engine.feature_store['mlmi_minus_nwrqk'] == 2.0
        
    def test_get_current_features(self, indicator_engine):
        """Test getting current features returns a copy"""
        # Modify feature store
        indicator_engine.feature_store['mlmi_value'] = 10.0
        
        # Get features
        features = indicator_engine.get_current_features()
        
        # Modify returned dict
        features['mlmi_value'] = 20.0
        
        # Original should be unchanged
        assert indicator_engine.feature_store['mlmi_value'] == 10.0
        
    def test_get_feature_summary(self, indicator_engine):
        """Test feature summary generation"""
        # Set some state
        indicator_engine.calculations_5min = 100
        indicator_engine.calculations_30min = 50
        indicator_engine.events_emitted = 45
        indicator_engine.has_30min_data = True
        
        summary = indicator_engine.get_feature_summary()
        
        assert summary['calculations_5min'] == 100
        assert summary['calculations_30min'] == 50
        assert summary['events_emitted'] == 45
        assert summary['has_30min_data'] == True
        assert 'history_sizes' in summary
        assert summary['history_sizes']['5min'] == len(indicator_engine.history_5m)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])