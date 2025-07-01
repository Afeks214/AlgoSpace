"""Unit tests for the data pipeline components (handlers, bar_generator, validators).

This test suite validates the core functionality of the newly implemented
data pipeline components without external dependencies.
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, call, patch
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Direct imports avoiding __init__.py issues
import importlib.util

# Import handlers
handlers_path = os.path.join(os.path.dirname(__file__), 'src', 'data', 'handlers.py')
spec_handlers = importlib.util.spec_from_file_location("handlers", handlers_path)
handlers = importlib.util.module_from_spec(spec_handlers)
spec_handlers.loader.exec_module(handlers)
AbstractDataHandler = handlers.AbstractDataHandler
BacktestDataHandler = handlers.BacktestDataHandler
LiveDataHandler = handlers.LiveDataHandler
TickData = handlers.TickData

# Import bar_generator
bar_gen_path = os.path.join(os.path.dirname(__file__), 'src', 'components', 'bar_generator.py')
spec_bar = importlib.util.spec_from_file_location("bar_generator", bar_gen_path)
bar_generator = importlib.util.module_from_spec(spec_bar)
spec_bar.loader.exec_module(bar_generator)
BarGenerator = bar_generator.BarGenerator
BarData = bar_generator.BarData

# Import validators
validators_path = os.path.join(os.path.dirname(__file__), 'src', 'utils', 'validators.py')
spec_val = importlib.util.spec_from_file_location("validators", validators_path)
validators = importlib.util.module_from_spec(spec_val)
spec_val.loader.exec_module(validators)
ConfigValidator = validators.ConfigValidator
DataValidator = validators.DataValidator


class TestValidators:
    """Test suite for ConfigValidator and DataValidator."""
    
    def test_config_validator_valid_config(self):
        """Test that valid configurations pass validation."""
        valid_config = {
            'system': {'mode': 'backtest'},
            'data': {'backtest_file': __file__},  # Use this file as dummy
            'models': {},
            'indicators': {}
        }
        
        # Should not raise exception
        ConfigValidator.validate_main_config(valid_config)
    
    def test_config_validator_missing_keys(self):
        """Test that missing required keys raise appropriate errors."""
        invalid_config = {'system': {'mode': 'backtest'}}
        
        with pytest.raises(ValueError) as exc_info:
            ConfigValidator.validate_main_config(invalid_config)
        
        assert "Missing required configuration keys" in str(exc_info.value)
        assert "data" in str(exc_info.value)
    
    def test_config_validator_invalid_mode(self):
        """Test that invalid system mode raises error."""
        config = {
            'system': {'mode': 'invalid_mode'},
            'data': {},
            'models': {},
            'indicators': {}
        }
        
        with pytest.raises(ValueError) as exc_info:
            ConfigValidator.validate_main_config(config)
        
        assert "Invalid system mode" in str(exc_info.value)
    
    def test_config_validator_backtest_file_missing(self):
        """Test that backtest mode requires backtest_file."""
        config = {
            'system': {'mode': 'backtest'},
            'data': {},  # Missing backtest_file
            'models': {},
            'indicators': {}
        }
        
        with pytest.raises(ValueError) as exc_info:
            ConfigValidator.validate_main_config(config)
        
        assert "Missing 'backtest_file'" in str(exc_info.value)
    
    def test_data_validator_valid_tick(self):
        """Test that valid tick data passes validation."""
        valid_tick = {
            'timestamp': datetime.now(),
            'price': 100.5,
            'volume': 1000
        }
        
        # Should not raise exception
        DataValidator.validate_tick_data(valid_tick)
    
    def test_data_validator_tick_dataclass(self):
        """Test that TickData dataclass passes validation."""
        tick = TickData(
            timestamp=datetime.now(),
            price=100.5,
            volume=1000
        )
        
        # Should not raise exception
        DataValidator.validate_tick_data(tick)
    
    def test_data_validator_invalid_timestamp(self):
        """Test that invalid timestamp type raises error."""
        invalid_tick = {
            'timestamp': '2024-01-01',  # String instead of datetime
            'price': 100.5,
            'volume': 1000
        }
        
        with pytest.raises(TypeError) as exc_info:
            DataValidator.validate_tick_data(invalid_tick)
        
        assert "Timestamp must be datetime object" in str(exc_info.value)
    
    def test_data_validator_negative_price(self):
        """Test that negative price raises error."""
        invalid_tick = {
            'timestamp': datetime.now(),
            'price': -100.5,
            'volume': 1000
        }
        
        with pytest.raises(ValueError) as exc_info:
            DataValidator.validate_tick_data(invalid_tick)
        
        assert "Price must be positive" in str(exc_info.value)
    
    def test_data_validator_invalid_volume_type(self):
        """Test that non-integer volume raises error."""
        invalid_tick = {
            'timestamp': datetime.now(),
            'price': 100.5,
            'volume': 1000.5  # Float instead of int
        }
        
        with pytest.raises(TypeError) as exc_info:
            DataValidator.validate_tick_data(invalid_tick)
        
        assert "Volume must be integer" in str(exc_info.value)


class TestDataHandlers:
    """Test suite for data handler implementations."""
    
    def test_abstract_handler_requires_implementation(self):
        """Test that AbstractDataHandler cannot be instantiated."""
        mock_event_bus = Mock()
        config = {}
        
        with pytest.raises(TypeError):
            AbstractDataHandler(config, mock_event_bus)
    
    def test_backtest_handler_file_not_found(self):
        """Test that BacktestDataHandler raises FileNotFoundError for missing files."""
        config = {
            'data': {'backtest_file': '/non/existent/file.csv'}
        }
        mock_event_bus = Mock()
        
        with pytest.raises(FileNotFoundError) as exc_info:
            BacktestDataHandler(config, mock_event_bus)
        
        assert "Backtest file not found" in str(exc_info.value)
    
    def test_backtest_handler_successful_initialization(self):
        """Test successful initialization of BacktestDataHandler."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("timestamp,price,volume\n")
            f.write("2024-01-01T09:00:00,100.0,1000\n")
            temp_file = f.name
        
        try:
            config = {'data': {'backtest_file': temp_file}}
            mock_event_bus = Mock()
            
            handler = BacktestDataHandler(config, mock_event_bus)
            assert handler.file_path == temp_file
            assert handler.row_count == 0
            assert handler.error_count == 0
        finally:
            os.unlink(temp_file)
    
    def test_backtest_handler_stream_processing(self):
        """Test that BacktestDataHandler correctly processes CSV data."""
        # Create temporary CSV file with various data types
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("timestamp,price,volume\n")
            f.write("2024-01-01T09:00:00,100.0,1000\n")
            f.write("2024-01-01T09:00:01,100.5,500\n")
            f.write("2024-01-01T09:00:02,bad_price,200\n")  # Invalid row
            f.write("2024-01-01T09:00:03,101.0,300\n")
            f.write("incomplete_row\n")  # Malformed row
            temp_file = f.name
        
        try:
            config = {'data': {'backtest_file': temp_file}}
            mock_event_bus = Mock()
            
            handler = BacktestDataHandler(config, mock_event_bus)
            handler.start_stream()
            
            # Verify statistics
            assert handler.row_count == 5
            assert handler.error_count == 2  # Two invalid rows
            
            # Verify events published
            assert mock_event_bus.publish.call_count == 4  # 3 ticks + 1 completion
            
            # Verify tick events
            tick_calls = [call for call in mock_event_bus.publish.call_args_list 
                         if call[0][0] == 'NEW_TICK']
            assert len(tick_calls) == 3
            
            # Verify completion event
            completion_calls = [call for call in mock_event_bus.publish.call_args_list 
                               if call[0][0] == 'BACKTEST_COMPLETE']
            assert len(completion_calls) == 1
            
            # Check completion payload
            completion_payload = completion_calls[0][0][1]
            assert completion_payload['total_rows'] == 5
            assert completion_payload['error_count'] == 2
            assert completion_payload['success_count'] == 3
            
        finally:
            os.unlink(temp_file)
    
    def test_live_handler_initialization(self):
        """Test LiveDataHandler initialization."""
        config = {}
        mock_event_bus = Mock()
        
        # Should initialize without error
        handler = LiveDataHandler(config, mock_event_bus)
        assert handler.event_bus == mock_event_bus


class TestBarGenerator:
    """Test suite for BarGenerator component."""
    
    def test_bar_generator_initialization(self):
        """Test BarGenerator initialization."""
        config = {}
        mock_event_bus = Mock()
        
        generator = BarGenerator(config, mock_event_bus)
        
        assert generator.bars_5min is None
        assert generator.bars_30min is None
        assert generator.tick_count == 0
        assert generator.bars_emitted_5min == 0
        assert generator.bars_emitted_30min == 0
    
    def test_bar_time_calculation(self):
        """Test bar time bucket calculation."""
        config = {}
        mock_event_bus = Mock()
        generator = BarGenerator(config, mock_event_bus)
        
        # Test 5-minute buckets
        test_time = datetime(2024, 1, 1, 9, 32, 45)
        bar_time = generator._get_bar_time(test_time, 5)
        assert bar_time == datetime(2024, 1, 1, 9, 30, 0)
        
        # Test 30-minute buckets
        bar_time = generator._get_bar_time(test_time, 30)
        assert bar_time == datetime(2024, 1, 1, 9, 30, 0)
        
        # Test edge cases
        test_time = datetime(2024, 1, 1, 10, 0, 0)
        bar_time = generator._get_bar_time(test_time, 5)
        assert bar_time == datetime(2024, 1, 1, 10, 0, 0)
    
    def test_single_tick_processing(self):
        """Test processing of a single tick."""
        config = {}
        mock_event_bus = Mock()
        generator = BarGenerator(config, mock_event_bus)
        
        tick = TickData(
            timestamp=datetime(2024, 1, 1, 9, 0, 0),
            price=100.0,
            volume=1000
        )
        
        generator.on_new_tick(tick)
        
        # Verify bar creation
        assert generator.bars_5min is not None
        assert generator.bars_5min['open'] == 100.0
        assert generator.bars_5min['high'] == 100.0
        assert generator.bars_5min['low'] == 100.0
        assert generator.bars_5min['close'] == 100.0
        assert generator.bars_5min['volume'] == 1000
        
        # No bars should be emitted yet
        assert mock_event_bus.publish.call_count == 0
    
    def test_bar_completion_and_emission(self):
        """Test that bars are emitted when new bar period starts."""
        config = {}
        mock_event_bus = Mock()
        generator = BarGenerator(config, mock_event_bus)
        
        # First tick
        tick1 = TickData(
            timestamp=datetime(2024, 1, 1, 9, 0, 0),
            price=100.0,
            volume=1000
        )
        generator.on_new_tick(tick1)
        
        # Second tick in same bar
        tick2 = TickData(
            timestamp=datetime(2024, 1, 1, 9, 1, 0),
            price=101.0,
            volume=500
        )
        generator.on_new_tick(tick2)
        
        # Tick in new 5-minute bar
        tick3 = TickData(
            timestamp=datetime(2024, 1, 1, 9, 5, 0),
            price=102.0,
            volume=300
        )
        generator.on_new_tick(tick3)
        
        # Should have emitted one 5-minute bar
        assert mock_event_bus.publish.call_count == 1
        call_args = mock_event_bus.publish.call_args_list[0]
        assert call_args[0][0] == 'NEW_5MIN_BAR'
        
        bar_data = call_args[0][1]
        assert isinstance(bar_data, BarData)
        assert bar_data.open == 100.0
        assert bar_data.high == 101.0
        assert bar_data.low == 100.0
        assert bar_data.close == 101.0
        assert bar_data.volume == 1500
    
    def test_gap_handling(self):
        """Test that gaps in data are handled with synthetic bars."""
        config = {}
        mock_event_bus = Mock()
        generator = BarGenerator(config, mock_event_bus)
        
        # First tick
        tick1 = TickData(
            timestamp=datetime(2024, 1, 1, 9, 0, 0),
            price=100.0,
            volume=1000
        )
        generator.on_new_tick(tick1)
        
        # Tick with 15-minute gap (should generate 2 synthetic 5-min bars)
        tick2 = TickData(
            timestamp=datetime(2024, 1, 1, 9, 20, 0),
            price=105.0,
            volume=500
        )
        generator.on_new_tick(tick2)
        
        # Should have emitted 3 bars for 5-min timeframe:
        # 1. Original bar at 9:00
        # 2. Synthetic bar at 9:05
        # 3. Synthetic bar at 9:10
        # 4. Synthetic bar at 9:15
        
        bar_events = [call for call in mock_event_bus.publish.call_args_list 
                     if call[0][0] == 'NEW_5MIN_BAR']
        assert len(bar_events) == 4
        
        # Check synthetic bars
        synthetic_bar1 = bar_events[1][0][1]
        assert synthetic_bar1.open == 100.0
        assert synthetic_bar1.high == 100.0
        assert synthetic_bar1.low == 100.0
        assert synthetic_bar1.close == 100.0
        assert synthetic_bar1.volume == 0
        
        # Verify gap statistics
        assert generator.gaps_filled_5min == 3
    
    def test_concurrent_timeframes(self):
        """Test that both 5-min and 30-min bars are maintained correctly."""
        config = {}
        mock_event_bus = Mock()
        generator = BarGenerator(config, mock_event_bus)
        
        # Generate ticks across 35 minutes
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        
        for i in range(36):  # 36 minutes of data
            tick = TickData(
                timestamp=base_time + timedelta(minutes=i),
                price=100.0 + i * 0.1,
                volume=100
            )
            generator.on_new_tick(tick)
        
        # Count bar emissions
        bar_5min_events = [call for call in mock_event_bus.publish.call_args_list 
                          if call[0][0] == 'NEW_5MIN_BAR']
        bar_30min_events = [call for call in mock_event_bus.publish.call_args_list 
                           if call[0][0] == 'NEW_30MIN_BAR']
        
        # Should have 7 complete 5-minute bars (0-5, 5-10, ..., 30-35)
        assert len(bar_5min_events) == 7
        
        # Should have 1 complete 30-minute bar (0-30)
        assert len(bar_30min_events) == 1
        
        # Verify 30-minute bar aggregation
        bar_30min = bar_30min_events[0][0][1]
        assert bar_30min.open == 100.0  # First price
        assert bar_30min.close == 102.9  # Price at minute 29
        assert bar_30min.volume == 3000  # 30 ticks * 100 volume


if __name__ == "__main__":
    pytest.main([__file__, "-v"])