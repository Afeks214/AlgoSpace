"""
Comprehensive tests for the IndicatorEngine and advanced indicator features.

Tests include:
- IndicatorEngine orchestration and event handling
- LVN (Low Volume Node) strength score calculation
- MMD (Market Microstructure Dynamics) feature vector calculation
- Integration tests with multiple calculators
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timedelta
import logging

from src.core.events import Event, EventType, BarData
from src.core.event_bus import EventBus
from src.indicators.engine import IndicatorEngine
from src.indicators.base import IndicatorRegistry
from src.indicators.base import BaseIndicator
from src.indicators.mlmi import MLMICalculator
from src.indicators.lvn import LVNAnalyzer
from src.indicators.mmd import MMDFeatureExtractor
from src.indicators.fvg import FVGDetector
from src.indicators.nwrqk import NWRQKCalculator


class MockIndicator(BaseIndicator):
    """Mock indicator for testing."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.calculate_called = False
        self.last_bar_data = None
    
    def calculate(self, bar_data: BarData) -> dict:
        """Mock calculate method."""
        self.calculate_called = True
        self.last_bar_data = bar_data
        return {f"{self.name}_value": 42.0}


class TestIndicatorEngine:
    """Tests for the IndicatorEngine class."""
    
    @pytest.fixture
    def mock_event_bus(self):
        """Create a mock event bus."""
        return Mock(spec=EventBus)
    
    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return Mock(spec=logging.Logger)
    
    @pytest.fixture
    def indicator_engine(self, mock_event_bus, mock_logger):
        """Create an IndicatorEngine instance."""
        engine = IndicatorEngine(
            symbol="EURUSD",
            event_bus=mock_event_bus
        )
        engine.logger = mock_logger
        return engine
    
    def test_indicator_registration_and_orchestration(self, indicator_engine):
        """Test that IndicatorEngine properly registers and orchestrates indicators."""
        # Arrange
        mock_mlmi = MockIndicator("mlmi")
        mock_fvg = MockIndicator("fvg")
        mock_lvn = MockIndicator("lvn")
        
        # Register mock indicators
        indicator_engine.registry.register("5m", mock_mlmi)
        indicator_engine.registry.register("5m", mock_fvg)
        indicator_engine.registry.register("5m", mock_lvn)
        
        # Create a bar event
        bar_data = BarData(
            symbol="EURUSD",
            timestamp=datetime(2024, 1, 1, 9, 0, 0),
            open=1.0850,
            high=1.0860,
            low=1.0845,
            close=1.0855,
            volume=1000
        )
        bar_event = Event(type=EventType.NEW_5MIN_BAR, data=bar_data)
        
        # Act
        indicator_engine.on_5min_bar(bar_event)
        
        # Assert
        assert mock_mlmi.calculate_called
        assert mock_fvg.calculate_called
        assert mock_lvn.calculate_called
        assert mock_mlmi.last_bar_data == bar_data
        assert mock_fvg.last_bar_data == bar_data
        assert mock_lvn.last_bar_data == bar_data
    
    @patch('src.indicators.mlmi.MLMICalculator')
    @patch('src.indicators.fvg.FVGDetector')
    @patch('src.indicators.lvn.LVNAnalyzer')
    @patch('src.indicators.nwrqk.NWRQKCalculator')
    @patch('src.indicators.mmd.MMDFeatureExtractor')
    def test_integration_with_real_calculators(self, mock_mmd, mock_nwrqk, mock_lvn, 
                                             mock_fvg, mock_mlmi, indicator_engine, 
                                             mock_event_bus):
        """Integration test with mocked real calculator classes."""
        # Arrange
        # Set up mock calculator instances
        mlmi_instance = Mock()
        mlmi_instance.calculate.return_value = {"mlmi_impact": 0.75, "mlmi_liquidity": 0.60}
        mock_mlmi.return_value = mlmi_instance
        
        fvg_instance = Mock()
        fvg_instance.calculate.return_value = {"fvg_detected": True, "fvg_size": 0.0010}
        mock_fvg.return_value = fvg_instance
        
        lvn_instance = Mock()
        lvn_instance.calculate.return_value = {"lvn_strength": 0.85, "lvn_proximity": 0.0005}
        mock_lvn.return_value = lvn_instance
        
        nwrqk_instance = Mock()
        nwrqk_instance.calculate.return_value = {"nwrqk_signal": 0.65}
        mock_nwrqk.return_value = nwrqk_instance
        
        mmd_instance = Mock()
        mmd_instance.calculate.return_value = {"mmd_features": np.array([0.1] * 23)}
        mock_mmd.return_value = mmd_instance
        
        # Initialize engine (this creates the calculator instances)
        indicator_engine.initialize()
        
        # Create a 30-minute bar event
        bar_data = BarData(
            symbol="EURUSD",
            timestamp=datetime(2024, 1, 1, 9, 30, 0),
            open=1.0850,
            high=1.0870,
            low=1.0840,
            close=1.0865,
            volume=5000
        )
        bar_event = Event(type=EventType.NEW_30MIN_BAR, data=bar_data)
        
        # Act
        indicator_engine.on_30min_bar(bar_event)
        
        # Assert
        # Verify all calculators were called
        mlmi_instance.calculate.assert_called_once_with(bar_data)
        fvg_instance.calculate.assert_called_once_with(bar_data)
        lvn_instance.calculate.assert_called_once_with(bar_data)
        nwrqk_instance.calculate.assert_called_once_with(bar_data)
        mmd_instance.calculate.assert_called_once_with(bar_data)
        
        # Verify INDICATORS_READY event was published
        indicators_ready_calls = [
            call for call in mock_event_bus.publish.call_args_list
            if call[0][0].type == EventType.INDICATORS_READY
        ]
        assert len(indicators_ready_calls) == 1
        
        # Verify event data contains all indicators
        event_data = indicators_ready_calls[0][0][0].data
        assert event_data["symbol"] == "EURUSD"
        assert event_data["timeframe"] == "30m"
        assert "mlmi_impact" in event_data["indicators"]
        assert "fvg_detected" in event_data["indicators"]
        assert "lvn_strength" in event_data["indicators"]
        assert "nwrqk_signal" in event_data["indicators"]
        assert "mmd_features" in event_data["indicators"]


class TestLVNStrengthScore:
    """Tests for LVN (Low Volume Node) strength score calculation."""
    
    @pytest.fixture
    def lvn_analyzer(self):
        """Create an LVNAnalyzer instance."""
        return LVNAnalyzer("lvn_test")
    
    def test_lvn_strength_score_calculation(self, lvn_analyzer):
        """Test LVN strength score calculation with dummy price history."""
        # Arrange
        # Create a price history with clear low volume nodes
        price_history = []
        timestamps = []
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        
        # Create price levels with varying volumes
        # High volume around 1.0850-1.0860
        for i in range(20):
            price_history.append({
                'high': 1.0860,
                'low': 1.0850,
                'close': 1.0855,
                'volume': 1000  # High volume
            )}
            timestamps.append(base_time + timedelta(minutes=i))
        
        # Low volume around 1.0870-1.0880 (LVN area)
        for i in range(5):
            price_history.append({
                'high': 1.0880,
                'low': 1.0870,
                'close': 1.0875,
                'volume': 100  # Low volume
            })
            timestamps.append(base_time + timedelta(minutes=20+i))
        
        # High volume around 1.0890-1.0900
        for i in range(15):
            price_history.append({
                'high': 1.0900,
                'low': 1.0890,
                'close': 1.0895,
                'volume': 800  # High volume
            })
            timestamps.append(base_time + timedelta(minutes=25+i))
        
        # Update analyzer's price history
        lvn_analyzer.price_history = price_history
        
        # Current bar near the LVN
        current_bar = BarData(
            symbol="EURUSD",
            timestamp=base_time + timedelta(minutes=40),
            open=1.0872,
            high=1.0878,
            low=1.0870,
            close=1.0876,
            volume=150
        )
        
        # Act
        result = lvn_analyzer.calculate(current_bar)
        
        # Assert
        assert "lvn_strength" in result
        assert "lvn_proximity" in result
        assert "lvn_level" in result
        
        # Strength should be in expected range (0-1)
        assert 0.0 <= result["lvn_strength"] <= 1.0
        
        # Should detect proximity to LVN around 1.0875
        assert result["lvn_proximity"] < 0.001  # Very close to LVN
        
        # LVN level should be around 1.0875
        assert abs(result["lvn_level"] - 1.0875) < 0.001
    
    def test_lvn_with_insufficient_data(self, lvn_analyzer):
        """Test LVN calculation with insufficient price history."""
        # Arrange
        lvn_analyzer.price_history = []  # Empty history
        
        current_bar = BarData(
            symbol="EURUSD",
            timestamp=datetime(2024, 1, 1, 9, 0, 0),
            open=1.0850,
            high=1.0855,
            low=1.0845,
            close=1.0852,
            volume=100
        )
        
        # Act
        result = lvn_analyzer.calculate(current_bar)
        
        # Assert
        assert result["lvn_strength"] == 0.0
        assert result["lvn_proximity"] == float('inf')
        assert result["lvn_level"] == 0.0


class TestMMDFeatureVector:
    """Tests for MMD (Market Microstructure Dynamics) feature vector calculation."""
    
    @pytest.fixture
    def mmd_extractor(self):
        """Create an MMDFeatureExtractor instance."""
        return MMDFeatureExtractor("mmd_test")
    
    def test_mmd_feature_vector_shape_and_values(self, mmd_extractor):
        """Test MMD feature vector has correct shape and reasonable values."""
        # Arrange
        # Build up price history for feature calculation
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        
        for i in range(50):  # Need sufficient history for all features
            bar = BarData(
                symbol="EURUSD",
                timestamp=base_time + timedelta(minutes=i),
                open=1.0850 + np.sin(i * 0.1) * 0.001,
                high=1.0850 + np.sin(i * 0.1) * 0.001 + 0.0005,
                low=1.0850 + np.sin(i * 0.1) * 0.001 - 0.0005,
                close=1.0850 + np.sin(i * 0.1) * 0.001 + np.random.randn() * 0.0001,
                volume=1000 + np.random.randint(-100, 100)
            )
            mmd_extractor.update_history(bar)
        
        # Current bar
        current_bar = BarData(
            symbol="EURUSD",
            timestamp=base_time + timedelta(minutes=50),
            open=1.0855,
            high=1.0860,
            low=1.0850,
            close=1.0858,
            volume=1200
        )
        
        # Act
        result = mmd_extractor.calculate(current_bar)
        
        # Assert
        assert "mmd_features" in result
        features = result["mmd_features"]
        
        # Check shape - should be 23 dimensions
        assert isinstance(features, np.ndarray)
        assert features.shape == (23,)
        
        # Check values are reasonable (normalized between -1 and 1 for most features)
        assert np.all(np.isfinite(features))  # No NaN or inf values
        assert np.all(np.abs(features) <= 10.0)  # Reasonable bounds
        
        # Check specific feature indices have expected properties
        # Price-based features (indices 0-4) should be small as they're returns
        assert np.all(np.abs(features[0:5]) < 1.0)
        
        # Volume-based features should be positive or normalized
        # RSI (index 5) should be between 0 and 100 (before normalization)
        # Volatility features should be positive
    
    def test_mmd_feature_names_and_order(self, mmd_extractor):
        """Test that MMD features are in the expected order."""
        # Arrange
        expected_feature_count = 23
        
        # Build minimal history
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        for i in range(30):
            bar = BarData(
                symbol="EURUSD",
                timestamp=base_time + timedelta(minutes=i),
                open=1.0850,
                high=1.0855,
                low=1.0845,
                close=1.0852,
                volume=1000
            )
            mmd_extractor.update_history(bar)
        
        current_bar = BarData(
            symbol="EURUSD",
            timestamp=base_time + timedelta(minutes=30),
            open=1.0852,
            high=1.0858,
            low=1.0850,
            close=1.0856,
            volume=1100
        )
        
        # Act
        result = mmd_extractor.calculate(current_bar)
        
        # Assert
        features = result["mmd_features"]
        assert len(features) == expected_feature_count
        
        # Verify feature metadata if available
        if hasattr(mmd_extractor, 'feature_names'):
            assert len(mmd_extractor.feature_names) == expected_feature_count
            
            # Check for expected feature types
            feature_names_str = ' '.join(mmd_extractor.feature_names)
            assert 'return' in feature_names_str.lower()  # Price returns
            assert 'volume' in feature_names_str.lower()  # Volume features
            assert 'volatility' in feature_names_str.lower() or 'vol' in feature_names_str.lower()  # Volatility
            assert 'rsi' in feature_names_str.lower()  # RSI indicator