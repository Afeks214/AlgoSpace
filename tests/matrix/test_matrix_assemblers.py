"""
Comprehensive tests for MatrixAssembler components.

Tests include:
- Matrix shape verification after N events
- Rolling window logic
- Robustness with missing features
- Normalization logic verification
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import logging

from src.core.events import Event, EventType
from src.core.event_bus import EventBus
from src.matrix.assembler_5m import MatrixAssembler5m
from src.matrix.assembler_30m import MatrixAssembler30m
from src.matrix.assembler_regime import MatrixAssemblerRegime
from src.matrix.normalizers import RollingNormalizer


class TestMatrixAssembler5m:
    """Tests for the 5-minute MatrixAssembler."""
    
    @pytest.fixture
    def mock_event_bus(self):
        """Create a mock event bus."""
        return Mock(spec=EventBus)
    
    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return Mock(spec=logging.Logger)
    
    @pytest.fixture
    def assembler(self, mock_event_bus, mock_logger):
        """Create a MatrixAssembler5m instance."""
        assembler = MatrixAssembler5m(
            symbol="EURUSD",
            window_size=100,
            event_bus=mock_event_bus
        )
        assembler.logger = mock_logger
        return assembler
    
    def test_matrix_shape_after_n_events(self, assembler, mock_event_bus):
        """Test that assembler produces correct matrix shape after N events."""
        # Arrange
        n_events = 50
        expected_features = 10  # Based on 5m feature set
        
        # Create indicator events
        for i in range(n_events):
            event = Event(
                type=EventType.INDICATORS_READY,
                data={
                    "symbol": "EURUSD",
                    "timeframe": "5m",
                    "timestamp": datetime(2024, 1, 1, 9, 0, 0) + timedelta(minutes=5*i),
                    "indicators": {
                        "mlmi_impact": 0.5 + i * 0.01,
                        "mlmi_liquidity": 0.6 + i * 0.01,
                        "fvg_detected": i % 2 == 0,
                        "fvg_size": 0.001 * (i % 5),
                        "lvn_strength": 0.7 + i * 0.005,
                        "lvn_proximity": 0.0005 + i * 0.00001,
                        "price_return_1": 0.0001 * i,
                        "price_return_5": 0.0005 * i,
                        "volume_ratio": 1.0 + i * 0.01,
                        "spread": 0.0001 + i * 0.000001
                    }
                }
            )
            assembler.on_indicators_ready(event)
        
        # Assert
        # Check internal buffer shape
        assert assembler.feature_matrix.shape == (n_events, expected_features)
        
        # Verify MATRIX_READY event was published
        matrix_ready_calls = [
            call for call in mock_event_bus.publish.call_args_list
            if call[0][0].type == EventType.MATRIX_READY
        ]
        assert len(matrix_ready_calls) > 0
        
        # Check last published matrix
        last_matrix_event = matrix_ready_calls[-1][0][0]
        matrix_data = last_matrix_event.data
        assert matrix_data["symbol"] == "EURUSD"
        assert matrix_data["timeframe"] == "5m"
        assert matrix_data["matrix"].shape == (n_events, expected_features)
    
    def test_rolling_window_logic(self, assembler, mock_event_bus):
        """Test that rolling window maintains fixed size after exceeding window_size."""
        # Arrange
        window_size = 100
        n_events = 150  # More than window size
        expected_features = 10
        
        # Create more events than window size
        for i in range(n_events):
            event = Event(
                type=EventType.INDICATORS_READY,
                data={
                    "symbol": "EURUSD",
                    "timeframe": "5m",
                    "timestamp": datetime(2024, 1, 1, 9, 0, 0) + timedelta(minutes=5*i),
                    "indicators": {
                        "mlmi_impact": 0.5,
                        "mlmi_liquidity": 0.6,
                        "fvg_detected": True,
                        "fvg_size": 0.001,
                        "lvn_strength": 0.7,
                        "lvn_proximity": 0.0005,
                        "price_return_1": 0.0001,
                        "price_return_5": 0.0005,
                        "volume_ratio": 1.0,
                        "spread": 0.0001
                    }
                }
            )
            assembler.on_indicators_ready(event)
        
        # Assert
        # Matrix should maintain window size
        assert assembler.feature_matrix.shape == (window_size, expected_features)
        
        # Get last published matrix
        matrix_ready_calls = [
            call for call in mock_event_bus.publish.call_args_list
            if call[0][0].type == EventType.MATRIX_READY
        ]
        last_matrix_event = matrix_ready_calls[-1][0][0]
        matrix_data = last_matrix_event.data
        
        # Published matrix should also have window size
        assert matrix_data["matrix"].shape == (window_size, expected_features)
    
    def test_robustness_with_missing_features(self, assembler, mock_event_bus, mock_logger):
        """Test that assembler handles missing features gracefully."""
        # Arrange
        # Event with missing required features
        event = Event(
            type=EventType.INDICATORS_READY,
            data={
                "symbol": "EURUSD",
                "timeframe": "5m",
                "timestamp": datetime(2024, 1, 1, 9, 0, 0),
                "indicators": {
                    "mlmi_impact": 0.5,
                    "mlmi_liquidity": 0.6,
                    # Missing: fvg_detected, fvg_size, lvn_strength, etc.
                    "price_return_1": 0.0001,
                }
            }
        )
        
        # Act
        assembler.on_indicators_ready(event)
        
        # Assert
        # Should log warnings about missing features
        assert mock_logger.warning.call_count > 0
        warning_messages = [call[0][0] for call in mock_logger.warning.call_args_list]
        assert any("Missing feature" in msg for msg in warning_messages)
        
        # Should still add a row with default values
        assert assembler.feature_matrix.shape[0] == 1
        
        # Check that default values were used (0.0 for numeric features)
        row = assembler.feature_matrix[0]
        assert np.isfinite(row).all()  # No NaN or inf values
    
    def test_feature_extraction_accuracy(self, assembler):
        """Test that features are extracted in the correct order."""
        # Arrange
        indicators = {
            "mlmi_impact": 0.75,
            "mlmi_liquidity": 0.60,
            "fvg_detected": True,
            "fvg_size": 0.0015,
            "lvn_strength": 0.85,
            "lvn_proximity": 0.0003,
            "price_return_1": 0.0002,
            "price_return_5": 0.0008,
            "volume_ratio": 1.25,
            "spread": 0.0001
        }
        
        event = Event(
            type=EventType.INDICATORS_READY,
            data={
                "symbol": "EURUSD",
                "timeframe": "5m",
                "timestamp": datetime(2024, 1, 1, 9, 0, 0),
                "indicators": indicators
            }
        )
        
        # Act
        assembler.on_indicators_ready(event)
        
        # Assert
        row = assembler.feature_matrix[0]
        
        # Verify feature values match (considering boolean conversion)
        assert row[0] == pytest.approx(0.75)  # mlmi_impact
        assert row[1] == pytest.approx(0.60)  # mlmi_liquidity
        assert row[2] == 1.0  # fvg_detected (True -> 1.0)
        assert row[3] == pytest.approx(0.0015)  # fvg_size
        assert row[4] == pytest.approx(0.85)  # lvn_strength
        assert row[5] == pytest.approx(0.0003)  # lvn_proximity
        assert row[6] == pytest.approx(0.0002)  # price_return_1
        assert row[7] == pytest.approx(0.0008)  # price_return_5
        assert row[8] == pytest.approx(1.25)  # volume_ratio
        assert row[9] == pytest.approx(0.0001)  # spread


class TestRollingNormalizer:
    """Tests for the RollingNormalizer class."""
    
    @pytest.fixture
    def normalizer(self):
        """Create a RollingNormalizer instance."""
        return RollingNormalizer(window_size=100)
    
    def test_normalization_scales_correctly(self, normalizer):
        """Test that normalization scales values to [-1, 1] range."""
        # Arrange
        # Create data with known range
        data = np.array([
            [0, 50, 100],  # Values in 0-100 range
            [25, 75, 50],
            [50, 100, 0],
            [75, 25, 50],
            [100, 0, 75]
        ])
        
        # Act
        normalized = normalizer.fit_transform(data)
        
        # Assert
        # Check shape is preserved
        assert normalized.shape == data.shape
        
        # Check range is approximately [-1, 1]
        assert normalized.min() >= -1.0
        assert normalized.max() <= 1.0
        
        # Check specific values
        # Column 0: ranges from 0 to 100, so 50 should map to ~0
        col0_normalized = normalized[:, 0]
        idx_50 = np.where(data[:, 0] == 50)[0][0]
        assert abs(col0_normalized[idx_50]) < 0.1  # Close to 0
        
        # Column 1: check min and max mapping
        col1_normalized = normalized[:, 1]
        idx_min = np.argmin(data[:, 1])
        idx_max = np.argmax(data[:, 1])
        assert col1_normalized[idx_min] == pytest.approx(-1.0, abs=0.1)
        assert col1_normalized[idx_max] == pytest.approx(1.0, abs=0.1)
    
    def test_rolling_window_update(self, normalizer):
        """Test that rolling window updates statistics correctly."""
        # Arrange
        window_size = 5
        normalizer = RollingNormalizer(window_size=window_size)
        
        # Initial data
        initial_data = np.array([[1], [2], [3], [4], [5]])
        normalizer.fit_transform(initial_data)
        
        # New data point that should shift the window
        new_data = np.array([[10]])  # Outlier value
        
        # Act
        # Update with new data (simulating rolling window)
        combined_data = np.vstack([initial_data[1:], new_data])  # Drop first, add new
        normalized_new = normalizer.fit_transform(combined_data)
        
        # Assert
        # Statistics should be updated to include the outlier
        assert normalizer.running_mean.shape == (1,)
        assert normalizer.running_std.shape == (1,)
        
        # Mean should shift towards the higher value
        assert normalizer.running_mean[0] > 3.0  # Original mean was 3
        
        # Std should increase due to outlier
        assert normalizer.running_std[0] > 1.5  # Original std was ~1.41
    
    def test_normalization_with_constant_values(self, normalizer):
        """Test normalization when all values are the same."""
        # Arrange
        data = np.array([[5, 5, 5]] * 10)  # All values are 5
        
        # Act
        normalized = normalizer.fit_transform(data)
        
        # Assert
        # When std is 0, values should be normalized to 0
        assert np.allclose(normalized, 0.0)
    
    def test_normalization_preserves_relative_order(self, normalizer):
        """Test that normalization preserves the relative order of values."""
        # Arrange
        data = np.array([[1], [5], [3], [9], [2], [7], [4], [6], [8]])
        
        # Act
        normalized = normalizer.fit_transform(data)
        
        # Assert
        # Check that order is preserved
        original_order = np.argsort(data.flatten())
        normalized_order = np.argsort(normalized.flatten())
        assert np.array_equal(original_order, normalized_order)


class TestMatrixAssemblerIntegration:
    """Integration tests for all matrix assemblers."""
    
    @pytest.fixture
    def mock_event_bus(self):
        """Create a mock event bus."""
        return Mock(spec=EventBus)
    
    def test_multi_timeframe_matrix_assembly(self, mock_event_bus):
        """Test that different timeframe assemblers work together correctly."""
        # Arrange
        assembler_5m = MatrixAssembler5m("EURUSD", 50, mock_event_bus)
        assembler_30m = MatrixAssembler30m("EURUSD", 20, mock_event_bus)
        
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        
        # Generate events for both timeframes
        for i in range(30):  # 30 5-minute bars = 2.5 hours
            # 5-minute indicators
            event_5m = Event(
                type=EventType.INDICATORS_READY,
                data={
                    "symbol": "EURUSD",
                    "timeframe": "5m",
                    "timestamp": base_time + timedelta(minutes=5*i),
                    "indicators": {
                        "mlmi_impact": 0.5 + i * 0.01,
                        "mlmi_liquidity": 0.6,
                        "fvg_detected": True,
                        "fvg_size": 0.001,
                        "lvn_strength": 0.7,
                        "lvn_proximity": 0.0005,
                        "price_return_1": 0.0001,
                        "price_return_5": 0.0005,
                        "volume_ratio": 1.0,
                        "spread": 0.0001
                    }
                }
            )
            assembler_5m.on_indicators_ready(event_5m)
            
            # 30-minute indicators (every 6th 5-minute bar)
            if i % 6 == 5:
                event_30m = Event(
                    type=EventType.INDICATORS_READY,
                    data={
                        "symbol": "EURUSD",
                        "timeframe": "30m",
                        "timestamp": base_time + timedelta(minutes=5*i),
                        "indicators": {
                            "mlmi_impact": 0.6,
                            "mlmi_liquidity": 0.7,
                            "nwrqk_signal": 0.8,
                            "mmd_features": np.random.randn(23),
                            "fvg_detected": False,
                            "fvg_size": 0.002,
                            "lvn_strength": 0.75,
                            "market_regime": 1,
                            "volatility_regime": 0.5,
                            "trend_strength": 0.3
                        }
                    }
                )
                assembler_30m.on_indicators_ready(event_30m)
        
        # Assert
        # 5m assembler should have 30 rows
        assert assembler_5m.feature_matrix.shape[0] == 30
        assert assembler_5m.feature_matrix.shape[1] == 10  # 5m features
        
        # 30m assembler should have 5 rows (30 / 6)
        assert assembler_30m.feature_matrix.shape[0] == 5
        assert assembler_30m.feature_matrix.shape[1] >= 10  # 30m has more features
        
        # Both should have published MATRIX_READY events
        matrix_events = [
            call[0][0] for call in mock_event_bus.publish.call_args_list
            if call[0][0].type == EventType.MATRIX_READY
        ]
        
        # Check we have events from both timeframes
        timeframes = [event.data["timeframe"] for event in matrix_events]
        assert "5m" in timeframes
        assert "30m" in timeframes