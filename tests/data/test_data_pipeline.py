"""
Comprehensive tests for the data pipeline components.

Tests include:
- DataHandler: CSV reading, tick event publishing, error handling
- BarGenerator: Bar creation, gap handling, forward filling
"""

import pytest
from unittest.mock import Mock, patch, call, MagicMock
from datetime import datetime, timedelta
import csv
import logging

from src.core.events import Event, EventType, TickData, BarData
from src.core.event_bus import EventBus
from src.data.handlers import BacktestDataHandler
from src.components.bar_generator import BarGenerator


class TestBacktestDataHandler:
    """Tests for the BacktestDataHandler class."""
    
    @pytest.fixture
    def mock_event_bus(self):
        """Create a mock event bus."""
        return Mock(spec=EventBus)
    
    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return Mock(spec=logging.Logger)
    
    @pytest.fixture
    def data_handler(self, mock_event_bus, mock_logger):
        """Create a BacktestDataHandler instance."""
        handler = BacktestDataHandler(
            csv_path="test_data.csv",
            symbol="EURUSD",
            event_bus=mock_event_bus
        )
        handler.logger = mock_logger
        return handler
    
    @patch('builtins.open')
    @patch('csv.reader')
    def test_read_csv_and_publish_tick_events(self, mock_csv_reader, mock_open, data_handler, mock_event_bus):
        """Test that DataHandler correctly reads CSV rows and publishes NEW_TICK events."""
        # Arrange
        mock_csv_data = [
            ['timestamp', 'bid', 'ask'],
            ['2024-01-01 09:00:00', '1.0850', '1.0851'],
            ['2024-01-01 09:00:01', '1.0852', '1.0853'],
            ['2024-01-01 09:00:02', '1.0854', '1.0855']
        ]
        mock_csv_reader.return_value = iter(mock_csv_data)
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Act
        data_handler.start()
        
        # Assert
        assert mock_open.called
        assert mock_csv_reader.called
        
        # Should publish 3 tick events (skipping header)
        assert mock_event_bus.publish.call_count == 3
        
        # Verify first tick event
        first_call = mock_event_bus.publish.call_args_list[0]
        event = first_call[0][0]
        assert isinstance(event, Event)
        assert event.type == EventType.NEW_TICK
        assert isinstance(event.data, TickData)
        assert event.data.symbol == "EURUSD"
        assert event.data.bid == 1.0850
        assert event.data.ask == 1.0851
        assert event.data.timestamp == datetime(2024, 1, 1, 9, 0, 0)
    
    @patch('builtins.open')
    @patch('csv.reader')
    def test_malformed_csv_row_logs_warning(self, mock_csv_reader, mock_open, data_handler, mock_logger):
        """Test that malformed CSV rows log warnings and don't crash the system."""
        # Arrange
        mock_csv_data = [
            ['timestamp', 'bid', 'ask'],
            ['2024-01-01 09:00:00', '1.0850', '1.0851'],  # Valid row
            ['2024-01-01 09:00:01', 'invalid_price', '1.0853'],  # Invalid bid
            ['invalid_timestamp', '1.0854', '1.0855'],  # Invalid timestamp
            ['2024-01-01 09:00:02'],  # Missing columns
            ['2024-01-01 09:00:03', '1.0856', '1.0857']  # Valid row
        ]
        mock_csv_reader.return_value = iter(mock_csv_data)
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Act
        data_handler.start()
        
        # Assert
        # Should have 3 warnings for malformed rows
        assert mock_logger.warning.call_count >= 3
        
        # Should still publish 2 valid tick events
        valid_publishes = [
            call for call in data_handler.event_bus.publish.call_args_list
            if call[0][0].type == EventType.NEW_TICK
        ]
        assert len(valid_publishes) == 2
    
    @patch('builtins.open')
    def test_file_not_found_error(self, mock_open, data_handler, mock_logger):
        """Test that FileNotFoundError is handled gracefully."""
        # Arrange
        mock_open.side_effect = FileNotFoundError("File not found")
        
        # Act
        data_handler.start()
        
        # Assert
        mock_logger.error.assert_called_once()
        assert "Error reading CSV file" in mock_logger.error.call_args[0][0]


class TestBarGenerator:
    """Tests for the BarGenerator class."""
    
    @pytest.fixture
    def mock_event_bus(self):
        """Create a mock event bus."""
        return Mock(spec=EventBus)
    
    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return Mock(spec=logging.Logger)
    
    @pytest.fixture
    def bar_generator(self, mock_event_bus, mock_logger):
        """Create a BarGenerator instance."""
        generator = BarGenerator(
            symbol="EURUSD",
            event_bus=mock_event_bus
        )
        generator.logger = mock_logger
        return generator
    
    def test_bar_creation_from_ticks(self, bar_generator, mock_event_bus):
        """Test that BarGenerator correctly creates 5-minute bars from tick sequence."""
        # Arrange
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        ticks = []
        
        # Create ticks for 5 minutes with price movement
        for i in range(300):  # 5 minutes of second-by-second ticks
            tick_time = base_time + timedelta(seconds=i)
            bid = 1.0850 + i * 0.0001
            ask = bid + 0.0001
            
            tick = Event(
                type=EventType.NEW_TICK,
                data=TickData(
                    symbol="EURUSD",
                    timestamp=tick_time,
                    bid=bid,
                    ask=ask
                )
            )
            ticks.append(tick)
        
        # Act
        for tick in ticks:
            bar_generator.on_tick(tick)
        
        # Move to next period to trigger bar emission
        next_tick = Event(
            type=EventType.NEW_TICK,
            data=TickData(
                symbol="EURUSD",
                timestamp=base_time + timedelta(minutes=5, seconds=1),
                bid=1.1000,
                ask=1.1001
            )
        )
        bar_generator.on_tick(next_tick)
        
        # Assert
        # Should publish one NEW_5MIN_BAR event
        bar_events = [
            call for call in mock_event_bus.publish.call_args_list
            if call[0][0].type == EventType.NEW_5MIN_BAR
        ]
        assert len(bar_events) == 1
        
        # Verify bar data
        bar_event = bar_events[0][0][0]
        bar_data = bar_event.data
        assert isinstance(bar_data, BarData)
        assert bar_data.symbol == "EURUSD"
        assert bar_data.timestamp == base_time
        assert bar_data.open == pytest.approx(1.0850, rel=1e-6)
        assert bar_data.high == pytest.approx(1.0850 + 299 * 0.0001, rel=1e-6)
        assert bar_data.low == pytest.approx(1.0850, rel=1e-6)
        assert bar_data.close == pytest.approx(1.0850 + 299 * 0.0001, rel=1e-6)
        assert bar_data.volume == 300
    
    def test_gap_handling_with_forward_fill(self, bar_generator, mock_event_bus, mock_logger):
        """Test critical gap-handling logic with forward-filled bars."""
        # Arrange
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        
        # First tick
        tick1 = Event(
            type=EventType.NEW_TICK,
            data=TickData(
                symbol="EURUSD",
                timestamp=base_time,
                bid=1.0850,
                ask=1.0851
            )
        )
        
        # Second tick after a 15-minute gap (should generate 3 forward-filled bars)
        tick2 = Event(
            type=EventType.NEW_TICK,
            data=TickData(
                symbol="EURUSD",
                timestamp=base_time + timedelta(minutes=15, seconds=30),
                bid=1.0900,
                ask=1.0901
            )
        )
        
        # Act
        bar_generator.on_tick(tick1)
        bar_generator.on_tick(tick2)
        
        # Assert
        # Should emit 3 forward-filled bars for the gap
        bar_events = [
            call for call in mock_event_bus.publish.call_args_list
            if call[0][0].type == EventType.NEW_5MIN_BAR
        ]
        assert len(bar_events) == 3
        
        # Verify forward-filled bars
        expected_timestamps = [
            base_time,  # 09:00:00
            base_time + timedelta(minutes=5),  # 09:05:00
            base_time + timedelta(minutes=10)  # 09:10:00
        ]
        
        for i, (bar_call, expected_ts) in enumerate(zip(bar_events, expected_timestamps)):
            bar_data = bar_call[0][0].data
            assert bar_data.timestamp == expected_ts
            assert bar_data.open == pytest.approx(1.0850, rel=1e-6)
            assert bar_data.high == pytest.approx(1.0850, rel=1e-6)
            assert bar_data.low == pytest.approx(1.0850, rel=1e-6)
            assert bar_data.close == pytest.approx(1.0850, rel=1e-6)
            
            if i == 0:
                assert bar_data.volume == 1  # First bar has actual tick
            else:
                assert bar_data.volume == 0  # Forward-filled bars have 0 volume
        
        # Verify warning was logged
        mock_logger.warning.assert_called()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "Time gap detected" in warning_msg
        assert "15.50 minutes" in warning_msg
    
    def test_multiple_timeframe_bar_generation(self, bar_generator, mock_event_bus):
        """Test that BarGenerator emits bars for multiple timeframes."""
        # Arrange
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        
        # Generate ticks for 30 minutes
        for minute in range(31):
            tick_time = base_time + timedelta(minutes=minute)
            tick = Event(
                type=EventType.NEW_TICK,
                data=TickData(
                    symbol="EURUSD",
                    timestamp=tick_time,
                    bid=1.0850 + minute * 0.0001,
                    ask=1.0851 + minute * 0.0001
                )
            )
            bar_generator.on_tick(tick)
        
        # Assert
        # Should emit both 5-minute and 30-minute bars
        bar_5min_events = [
            call for call in mock_event_bus.publish.call_args_list
            if call[0][0].type == EventType.NEW_5MIN_BAR
        ]
        bar_30min_events = [
            call for call in mock_event_bus.publish.call_args_list
            if call[0][0].type == EventType.NEW_30MIN_BAR
        ]
        
        assert len(bar_5min_events) == 6  # 30 minutes = 6 x 5-minute bars
        assert len(bar_30min_events) == 1  # 30 minutes = 1 x 30-minute bar
        
        # Verify 30-minute bar aggregates correctly
        bar_30min_data = bar_30min_events[0][0][0].data
        assert bar_30min_data.timestamp == base_time
        assert bar_30min_data.open == pytest.approx(1.0850, rel=1e-6)
        assert bar_30min_data.high == pytest.approx(1.0850 + 29 * 0.0001, rel=1e-6)
        assert bar_30min_data.low == pytest.approx(1.0850, rel=1e-6)
        assert bar_30min_data.close == pytest.approx(1.0850 + 29 * 0.0001, rel=1e-6)
        assert bar_30min_data.volume == 30