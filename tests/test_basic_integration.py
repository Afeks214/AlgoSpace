"""
Basic integration tests for the AlgoSpace data pipeline.

These tests verify the core functionality of the data pipeline components
using their actual implementations with minimal mocking.
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime
import numpy as np

from src.core.events import Event, EventType, TickData, BarData
from src.core.event_bus import EventBus
from src.data.handlers import BacktestDataHandler
from src.components.bar_generator import BarGenerator
from src.indicators.engine import IndicatorEngine
from src.indicators.base import IndicatorRegistry
from src.matrix.assembler_5m import MatrixAssembler5m
from src.matrix.normalizers import RollingNormalizer


class TestDataPipelineIntegration:
    """Integration tests for the complete data pipeline."""
    
    def test_rolling_normalizer_basic_functionality(self):
        """Test that RollingNormalizer works with correct parameters."""
        # Create normalizer with correct parameters
        normalizer = RollingNormalizer(alpha=0.01, warmup_samples=10)
        
        # Create test data
        data = np.array([
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [1.2, 2.2, 3.2],
            [1.3, 2.3, 3.3],
            [1.4, 2.4, 3.4]
        ])
        
        # Test normalization
        normalized = normalizer.fit_transform(data)
        
        # Verify output shape
        assert normalized.shape == data.shape
        
        # Verify no NaN or inf values
        assert np.all(np.isfinite(normalized))
        
        # Verify normalization has effect
        assert not np.array_equal(data, normalized)
    
    def test_event_bus_basic_functionality(self):
        """Test that EventBus can publish and subscribe to events."""
        # Create event bus
        event_bus = EventBus()
        
        # Create a mock subscriber
        received_events = []
        def mock_handler(event):
            received_events.append(event)
        
        # Subscribe to tick events
        event_bus.subscribe(EventType.NEW_TICK, mock_handler)
        
        # Publish a tick event
        tick_data = TickData(
            symbol="EURUSD",
            timestamp=datetime(2024, 1, 1, 9, 0, 0),
            bid=1.0850,
            ask=1.0851
        )
        event = Event(type=EventType.NEW_TICK, data=tick_data)
        event_bus.publish(event)
        
        # Verify event was received
        assert len(received_events) == 1
        assert received_events[0].type == EventType.NEW_TICK
        assert received_events[0].data.symbol == "EURUSD"
    
    def test_bar_generator_initialization(self):
        """Test that BarGenerator can be initialized with correct config."""
        # Create config and event bus
        config = {
            "components": {
                "bar_generator": {
                    "timeframes": ["5m", "30m"]
                }
            }
        }
        event_bus = EventBus()
        
        # Create bar generator
        bar_generator = BarGenerator(config, event_bus)
        
        # Verify it was created successfully
        assert bar_generator is not None
        assert hasattr(bar_generator, 'on_tick')
    
    def test_data_handler_initialization(self):
        """Test that BacktestDataHandler can be initialized."""
        # Create config and event bus
        config = {
            "data": {
                "backtest_file": "test_data.csv"
            }
        }
        event_bus = EventBus()
        
        # Create data handler
        data_handler = BacktestDataHandler(config, event_bus)
        
        # Verify it was created successfully
        assert data_handler is not None
        assert data_handler.event_bus == event_bus
    
    def test_matrix_assembler_initialization(self):
        """Test that MatrixAssembler5m can be initialized."""
        # Create config
        config = {
            "matrix": {
                "assemblers": {
                    "5m": {
                        "window_size": 100,
                        "features": ["mlmi_impact", "fvg_detected", "lvn_strength"]
                    }
                }
            }
        }
        
        # Create matrix assembler
        assembler = MatrixAssembler5m(config)
        
        # Verify it was created successfully
        assert assembler is not None
        assert hasattr(assembler, 'on_indicators_ready')
    
    def test_indicator_registry_functionality(self):
        """Test that IndicatorRegistry can register and retrieve indicators."""
        # Create registry
        registry = IndicatorRegistry()
        
        # Create mock indicator
        mock_indicator = Mock()
        mock_indicator.name = "test_indicator"
        
        # Register indicator
        registry.register("5m", mock_indicator)
        
        # Retrieve indicators
        indicators = registry.get_indicators("5m")
        
        # Verify registration worked
        assert len(indicators) == 1
        assert indicators[0].name == "test_indicator"


class TestComponentInteractions:
    """Test interactions between different components."""
    
    def test_tick_to_bar_flow(self):
        """Test that ticks flow through to bar generation."""
        # Create event bus
        event_bus = EventBus()
        
        # Track published bars
        published_bars = []
        def track_bars(event):
            if event.type == EventType.NEW_5MIN_BAR:
                published_bars.append(event)
        
        event_bus.subscribe(EventType.NEW_5MIN_BAR, track_bars)
        
        # Create bar generator
        config = {
            "components": {
                "bar_generator": {
                    "timeframes": ["5m"]
                }
            }
        }
        bar_generator = BarGenerator(config, event_bus)
        
        # Subscribe bar generator to tick events
        event_bus.subscribe(EventType.NEW_TICK, bar_generator.on_tick)
        
        # Publish some tick events
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        for i in range(10):
            tick_data = TickData(
                symbol="EURUSD",
                timestamp=base_time.replace(second=i*30),
                bid=1.0850 + i * 0.0001,
                ask=1.0851 + i * 0.0001
            )
            event = Event(type=EventType.NEW_TICK, data=tick_data)
            event_bus.publish(event)
        
        # Move time forward to trigger bar completion
        future_tick = TickData(
            symbol="EURUSD",
            timestamp=base_time.replace(minute=5, second=1),
            bid=1.0900,
            ask=1.0901
        )
        event = Event(type=EventType.NEW_TICK, data=future_tick)
        event_bus.publish(event)
        
        # Verify bar was generated
        assert len(published_bars) >= 1
        bar_data = published_bars[0].data
        assert isinstance(bar_data, BarData)
        assert bar_data.symbol == "EURUSD"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])