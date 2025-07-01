"""
Final verification tests for AlgoSpace foundational components.

These tests verify the core functionality of the data pipeline, 
indicators, and matrix assembly components.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import csv
import os

from src.core.events import Event, EventType, TickData, BarData
from src.core.event_bus import EventBus
from src.data.handlers import BacktestDataHandler
from src.components.bar_generator import BarGenerator
from src.matrix.normalizers import RollingNormalizer
from src.indicators.base import IndicatorRegistry, BaseIndicator


class TestCoreComponents:
    """Test core system components."""
    
    def test_event_bus_publish_subscribe(self):
        """Test EventBus publish/subscribe functionality."""
        event_bus = EventBus()
        received_events = []
        
        # Subscribe to events
        def handler(event):
            received_events.append(event)
        
        event_bus.subscribe(EventType.NEW_TICK, handler)
        
        # Create and publish event
        tick = TickData(
            symbol="EURUSD",
            timestamp=datetime(2024, 1, 1, 9, 0, 0),
            price=1.0850,
            volume=100
        )
        event = Event(
            event_type=EventType.NEW_TICK,
            timestamp=datetime(2024, 1, 1, 9, 0, 0),
            payload=tick,
            source="test"
        )
        event_bus.publish(event)
        
        # Verify
        assert len(received_events) == 1
        assert received_events[0].payload.symbol == "EURUSD"
        assert received_events[0].payload.price == 1.0850
    
    def test_rolling_normalizer_zscore(self):
        """Test RollingNormalizer z-score normalization."""
        normalizer = RollingNormalizer(alpha=0.1, warmup_samples=5)
        
        # Update with values
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        normalized_values = []
        
        for val in values:
            normalizer.update(val)
            normalized = normalizer.normalize_zscore(val)
            normalized_values.append(normalized)
        
        # Verify normalization works
        assert len(normalized_values) == len(values)
        assert all(isinstance(v, float) for v in normalized_values)
        # Later values should be normalized
        assert normalized_values[-1] != values[-1]
    
    def test_rolling_normalizer_minmax(self):
        """Test RollingNormalizer min-max normalization."""
        normalizer = RollingNormalizer()
        
        # Update with values
        for i in range(20):
            normalizer.update(float(i))
        
        # Test normalization
        normalized = normalizer.normalize_minmax(10.0, target_range=(-1, 1))
        
        # Should be roughly in the middle of the range
        assert -1 <= normalized <= 1
        assert abs(normalized) < 0.5  # Should be near 0 for middle value
    
    def test_indicator_registry_operations(self):
        """Test IndicatorRegistry register and retrieve operations."""
        registry = IndicatorRegistry()
        
        # Create mock indicator
        mock_indicator = Mock(spec=BaseIndicator)
        mock_indicator.name = "test_indicator"
        
        # Register
        registry.register("test_indicator", mock_indicator)
        
        # Retrieve
        retrieved = registry.get("test_indicator")
        assert retrieved is not None
        assert retrieved.name == "test_indicator"
        
        # Get all
        all_indicators = registry.get_all()
        assert len(all_indicators) == 1
        assert "test_indicator" in all_indicators
        
        # List names
        names = registry.list_names()
        assert "test_indicator" in names


class TestDataPipeline:
    """Test data pipeline components."""
    
    def test_backtest_data_handler_with_temp_file(self):
        """Test BacktestDataHandler with a temporary CSV file."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'price', 'volume'])
            writer.writerow(['2024-01-01 09:00:00', '1.0850', '100'])
            writer.writerow(['2024-01-01 09:00:01', '1.0851', '150'])
            temp_path = f.name
        
        try:
            # Create config and event bus
            config = {
                "data": {
                    "backtest_file": temp_path
                }
            }
            event_bus = EventBus()
            
            # Create handler
            handler = BacktestDataHandler(config, event_bus)
            
            # Verify initialization
            assert handler.file_path == temp_path
            assert handler.event_bus == event_bus
            
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_bar_generator_tick_processing(self):
        """Test BarGenerator processes ticks correctly."""
        config = {
            "components": {
                "bar_generator": {
                    "timeframes": ["5m", "30m"]
                }
            }
        }
        event_bus = EventBus()
        
        # Create bar generator
        bar_gen = BarGenerator(config, event_bus)
        
        # Track published bars
        published_bars = []
        def track_bars(event):
            published_bars.append(event)
        
        event_bus.subscribe(EventType.NEW_5MIN_BAR, track_bars)
        event_bus.subscribe(EventType.NEW_30MIN_BAR, track_bars)
        
        # Process ticks
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        tick_data = {
            "symbol": "EURUSD",
            "timestamp": base_time.isoformat(),
            "price": 1.0850,
            "volume": 100
        }
        
        # Send first tick
        bar_gen.on_new_tick(tick_data)
        
        # Send tick in next period to trigger bar
        future_tick = {
            "symbol": "EURUSD", 
            "timestamp": (base_time + timedelta(minutes=5, seconds=1)).isoformat(),
            "price": 1.0860,
            "volume": 200
        }
        bar_gen.on_new_tick(future_tick)
        
        # Should have generated at least one bar
        stats = bar_gen.get_statistics()
        assert stats['bars_emitted_5min'] >= 1


class TestMatrixAssembly:
    """Test matrix assembly components."""
    
    def test_matrix_normalizer_array_operations(self):
        """Test RollingNormalizer with numpy arrays."""
        normalizer = RollingNormalizer()
        
        # Create test data matrix
        data = np.random.randn(100, 10)  # 100 samples, 10 features
        
        # Process each row
        normalized_data = []
        for row in data:
            normalized_row = []
            for i, val in enumerate(row):
                normalizer.update(val)
                norm_val = normalizer.normalize_zscore(val)
                normalized_row.append(norm_val)
            normalized_data.append(normalized_row)
        
        normalized_array = np.array(normalized_data)
        
        # Verify shape preserved
        assert normalized_array.shape == data.shape
        
        # Verify no NaN or inf
        assert np.all(np.isfinite(normalized_array))
    
    def test_feature_extraction_pattern(self):
        """Test typical feature extraction pattern."""
        # Simulate indicator results
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
        
        # Extract features in order
        feature_names = [
            "mlmi_impact", "mlmi_liquidity", "fvg_detected", "fvg_size",
            "lvn_strength", "lvn_proximity", "price_return_1", "price_return_5",
            "volume_ratio", "spread"
        ]
        
        features = []
        for name in feature_names:
            value = indicators.get(name, 0.0)
            # Convert boolean to float
            if isinstance(value, bool):
                value = 1.0 if value else 0.0
            features.append(value)
        
        # Verify extraction
        assert len(features) == len(feature_names)
        assert features[0] == 0.75  # mlmi_impact
        assert features[2] == 1.0   # fvg_detected (True -> 1.0)
        assert all(isinstance(f, float) for f in features)


class TestIntegrationScenarios:
    """Test integration scenarios."""
    
    def test_event_flow_simulation(self):
        """Simulate event flow through the system."""
        event_bus = EventBus()
        events_log = []
        
        # Log all events
        def log_event(event):
            events_log.append((event.event_type, event.payload))
        
        # Subscribe to all event types
        for event_type in [EventType.NEW_TICK, EventType.NEW_5MIN_BAR, 
                          EventType.NEW_30MIN_BAR, EventType.INDICATORS_READY]:
            event_bus.subscribe(event_type, log_event)
        
        # Publish sequence of events
        tick = TickData(
            symbol="EURUSD",
            timestamp=datetime(2024, 1, 1, 9, 0, 0),
            price=1.0850,
            volume=100
        )
        event_bus.publish(Event(
            event_type=EventType.NEW_TICK,
            timestamp=datetime(2024, 1, 1, 9, 0, 0),
            payload=tick,
            source="test"
        ))
        
        bar = BarData(
            symbol="EURUSD",
            timestamp=datetime(2024, 1, 1, 9, 0, 0),
            open=1.0850,
            high=1.0855,
            low=1.0845,
            close=1.0852,
            volume=1000,
            timeframe=5
        )
        event_bus.publish(Event(
            event_type=EventType.NEW_5MIN_BAR,
            timestamp=datetime(2024, 1, 1, 9, 0, 0),
            payload=bar,
            source="test"
        ))
        
        indicators = {
            "symbol": "EURUSD",
            "timeframe": "5m",
            "timestamp": datetime(2024, 1, 1, 9, 0, 0),
            "indicators": {"mlmi_impact": 0.5}
        }
        event_bus.publish(Event(
            event_type=EventType.INDICATORS_READY,
            timestamp=datetime(2024, 1, 1, 9, 0, 0),
            payload=indicators,
            source="test"
        ))
        
        # Verify event flow
        assert len(events_log) == 3
        assert events_log[0][0] == EventType.NEW_TICK
        assert events_log[1][0] == EventType.NEW_5MIN_BAR
        assert events_log[2][0] == EventType.INDICATORS_READY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])