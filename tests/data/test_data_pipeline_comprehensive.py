"""
Comprehensive test suite for data pipeline components.

Tests all aspects of the data handler, bar generator, and utilities
with focus on performance, reliability, and edge cases.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.data.data_handler import (
    TickData, AbstractDataHandler, BacktestDataHandler, 
    LiveDataHandler, TickValidator
)
from src.data.bar_generator import (
    BarData, WorkingBar, BarGenerator, TimeframeBoundary
)
from src.data.data_utils import (
    DataQualityMonitor, DataRecorder, MockDataGenerator
)
from src.core.events import EventBus, Event, EventType


class TestTickData:
    """Test TickData structure."""
    
    def test_valid_tick_creation(self):
        """Test creating valid tick."""
        tick = TickData(
            timestamp=datetime.now(),
            symbol="TEST",
            price=100.5,
            volume=1000
        )
        
        assert tick.price == 100.5
        assert tick.volume == 1000
        
    def test_invalid_price_rejection(self):
        """Test that invalid prices are rejected."""
        with pytest.raises(ValueError):
            TickData(
                timestamp=datetime.now(),
                symbol="TEST",
                price=-10.0,  # Negative price
                volume=100
            )
            
    def test_tick_serialization(self):
        """Test tick serialization."""
        tick = TickData(
            timestamp=datetime(2024, 1, 1, 9, 30, 0),
            symbol="TEST",
            price=100.0,
            volume=500,
            bid=99.9,
            ask=100.1
        )
        
        data = tick.to_dict()
        
        assert data['price'] == 100.0
        assert data['bid'] == 99.9
        assert data['ask'] == 100.1
        assert 'timestamp' in data


class TestTickValidator:
    """Test tick validation logic."""
    
    @pytest.fixture
    def validator(self):
        config = {
            'symbol': 'TEST',
            'max_price_change': 0.1,
            'min_price': 0.01,
            'max_price': 10000.0,
            'max_volume': 1000000
        }
        return TickValidator(config)
        
    def test_valid_tick_passes(self, validator):
        """Test that valid ticks pass validation."""
        tick = TickData(
            timestamp=datetime.now(),
            symbol="TEST",
            price=100.0,
            volume=1000
        )
        
        is_valid, error = validator.validate(tick)
        
        assert is_valid
        assert error is None
        
    def test_price_spike_detection(self, validator):
        """Test price spike detection."""
        # Build price history
        base_time = datetime.now()
        
        for i in range(20):
            tick = TickData(
                timestamp=base_time + timedelta(seconds=i),
                symbol="TEST",
                price=100.0 + np.random.randn() * 0.5,
                volume=1000
            )
            validator.validate(tick)
            
        # Create spike
        spike_tick = TickData(
            timestamp=base_time + timedelta(seconds=21),
            symbol="TEST",
            price=120.0,  # 20% spike
            volume=1000
        )
        
        is_valid, error = validator.validate(spike_tick)
        
        assert not is_valid
        assert "spike" in error.lower()
        
    def test_timestamp_sequence_validation(self, validator):
        """Test timestamp sequence checking."""
        time1 = datetime.now()
        time2 = time1 - timedelta(seconds=1)  # Earlier timestamp
        
        # First tick
        tick1 = TickData(time1, "TEST", 100.0, 1000)
        validator.validate(tick1)
        
        # Second tick with earlier timestamp
        tick2 = TickData(time2, "TEST", 100.0, 1000)
        is_valid, error = validator.validate(tick2)
        
        assert not is_valid
        assert "timestamp" in error.lower()


class TestBarGenerator:
    """Test bar generation logic."""
    
    @pytest.fixture
    async def setup(self):
        """Setup test environment."""
        event_bus = EventBus()
        config = {
            'symbol': 'TEST',
            'timeframes': [5, 30]
        }
        
        generator = BarGenerator(config, event_bus)
        await generator.start()
        
        return generator, event_bus
        
    @pytest.mark.asyncio
    async def test_bar_creation(self, setup):
        """Test basic bar creation."""
        generator, event_bus = setup
        
        # Track emitted bars
        emitted_bars = []
        
        async def capture_bar(event):
            emitted_bars.append(event.data['bar'])
            
        await event_bus.subscribe(EventType.NEW_5MIN_BAR, capture_bar)
        
        # Generate ticks for one 5-minute bar
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        
        for i in range(5):
            tick = TickData(
                timestamp=base_time + timedelta(minutes=i, seconds=30),
                symbol="TEST",
                price=100.0 + i * 0.1,
                volume=100
            )
            
            event = Event(
                type=EventType.NEW_TICK,
                data={'tick': tick}
            )
            
            await event_bus.publish(event)
            
        # Emit tick to trigger bar completion
        final_tick = TickData(
            timestamp=base_time + timedelta(minutes=5, seconds=1),
            symbol="TEST",
            price=101.0,
            volume=100
        )
        
        await event_bus.publish(Event(
            type=EventType.NEW_TICK,
            data={'tick': final_tick}
        ))
        
        # Verify bar was created
        assert len(emitted_bars) == 1
        
        bar = emitted_bars[0]
        assert bar.timestamp == base_time
        assert bar.open == 100.0
        assert bar.high == 100.4
        assert bar.low == 100.0
        assert bar.close == 100.4
        assert bar.volume == 500
        assert bar.tick_count == 5
        
    @pytest.mark.asyncio
    async def test_gap_filling(self, setup):
        """Test forward-fill gap handling."""
        generator, event_bus = setup
        
        # Track all bars
        all_bars = {'5': [], '30': []}
        
        async def capture_5min(event):
            all_bars['5'].append(event.data['bar'])
            
        async def capture_30min(event):
            all_bars['30'].append(event.data['bar'])
            
        await event_bus.subscribe(EventType.NEW_5MIN_BAR, capture_5min)
        await event_bus.subscribe(EventType.NEW_30MIN_BAR, capture_30min)
        
        # Create gap scenario
        time1 = datetime(2024, 1, 1, 9, 0, 0)
        time2 = datetime(2024, 1, 1, 9, 20, 0)  # 20 minute gap
        
        # First tick
        tick1 = TickData(time1, "TEST", 100.0, 1000)
        await event_bus.publish(Event(
            type=EventType.NEW_TICK,
            data={'tick': tick1}
        ))
        
        # Second tick after gap
        tick2 = TickData(time2, "TEST", 101.0, 1000)
        await event_bus.publish(Event(
            type=EventType.NEW_TICK,
            data={'tick': tick2}
        ))
        
        # Force bar completion
        tick3 = TickData(time2 + timedelta(minutes=5), "TEST", 101.0, 100)
        await event_bus.publish(Event(
            type=EventType.NEW_TICK,
            data={'tick': tick3}
        ))
        
        # Check 5-minute bars (should have gaps filled)
        # Expected: 9:00, 9:05 (synthetic), 9:10 (synthetic), 9:15 (synthetic), 9:20
        await asyncio.sleep(0.1)  # Let events process
        
        assert len(all_bars['5']) >= 4
        
        # Verify synthetic bars
        for i in range(1, 4):
            bar = all_bars['5'][i]
            assert bar.is_synthetic
            assert bar.volume == 0
            assert bar.open == bar.high == bar.low == bar.close
            
    def test_timeframe_boundary_calculation(self):
        """Test precise timeframe boundary calculations."""
        # Test 5-minute boundaries
        assert TimeframeBoundary.floor_timestamp(
            datetime(2024, 1, 1, 9, 17, 45), 5
        ) == datetime(2024, 1, 1, 9, 15, 0)
        
        # Test 30-minute boundaries
        assert TimeframeBoundary.floor_timestamp(
            datetime(2024, 1, 1, 9, 17, 45), 30
        ) == datetime(2024, 1, 1, 9, 0, 0)
        
        # Test next boundary
        assert TimeframeBoundary.next_boundary(
            datetime(2024, 1, 1, 9, 17, 0), 5
        ) == datetime(2024, 1, 1, 9, 20, 0)
        
    def test_working_bar_updates(self):
        """Test WorkingBar state management."""
        bar = WorkingBar(datetime(2024, 1, 1, 9, 0, 0), 5)
        
        # First update
        bar.update(100.0, 500)
        assert bar.open == 100.0
        assert bar.high == 100.0
        assert bar.low == 100.0
        assert bar.close == 100.0
        assert bar.volume == 500
        
        # High update
        bar.update(101.0, 200)
        assert bar.high == 101.0
        assert bar.close == 101.0
        
        # Low update
        bar.update(99.5, 300)
        assert bar.low == 99.5
        assert bar.close == 99.5
        
        # Final state
        assert bar.volume == 1000
        assert bar.tick_count == 3
        
        # Convert to BarData
        bar_data = bar.to_bar()
        assert bar_data.vwap == pytest.approx(100.05, rel=0.01)


class TestBacktestDataHandler:
    """Test backtest data handler."""
    
    @pytest.fixture
    async def setup(self, tmp_path):
        """Setup test CSV file."""
        # Create test CSV
        csv_path = tmp_path / "test_data.csv"
        
        data = {
            'timestamp': pd.date_range('2024-01-01 09:00:00', periods=100, freq='1min'),
            'price': 100 + np.random.randn(100).cumsum() * 0.1,
            'volume': np.random.randint(100, 1000, 100)
        }
        
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        
        return csv_path
        
    @pytest.mark.asyncio
    async def test_csv_loading(self, setup):
        """Test loading data from CSV."""
        csv_path = setup
        event_bus = Mock()
        event_bus.publish = AsyncMock()
        
        config = {
            'symbol': 'TEST',
            'csv_path': str(csv_path),
            'replay_speed': 0  # As fast as possible
        }
        
        handler = BacktestDataHandler(config, event_bus)
        
        # Start handler
        await handler.start()
        
        # Check ticks were emitted
        assert event_bus.publish.call_count == 100
        
        # Verify tick structure
        first_call = event_bus.publish.call_args_list[0]
        event = first_call[0][0]
        
        assert event.type == EventType.NEW_TICK
        assert 'tick' in event.data
        
    @pytest.mark.asyncio
    async def test_date_filtering(self, setup):
        """Test date range filtering."""
        csv_path = setup
        event_bus = Mock()
        event_bus.publish = AsyncMock()
        
        config = {
            'symbol': 'TEST',
            'csv_path': str(csv_path),
            'start_date': datetime(2024, 1, 1, 9, 30, 0),
            'end_date': datetime(2024, 1, 1, 10, 0, 0)
        }
        
        handler = BacktestDataHandler(config, event_bus)
        await handler.start()
        
        # Should only emit 30 ticks (30 minutes of data)
        assert event_bus.publish.call_count == 30


class TestLiveDataHandler:
    """Test live data handler."""
    
    @pytest.mark.asyncio
    async def test_connection_retry(self):
        """Test connection retry logic."""
        event_bus = Mock()
        
        config = {
            'symbol': 'TEST',
            'rithmic': {
                'host': 'test.rithmic.com',
                'port': 443,
                'username': 'test',
                'password': 'test',
                'exchange': 'CME',
                'symbol_code': 'ES'
            },
            'max_reconnect_attempts': 3,
            'reconnect_delay': 0.1
        }
        
        handler = LiveDataHandler(config, event_bus)
        
        # Mock websocket to fail
        with patch('websockets.connect', side_effect=Exception("Connection failed")):
            with pytest.raises(RuntimeError, match="Max reconnection attempts"):
                await handler._connect()
                
        assert handler.reconnect_attempts == 3


class TestDataQualityMonitor:
    """Test data quality monitoring."""
    
    def test_anomaly_detection(self):
        """Test anomaly detection."""
        config = {'symbol': 'TEST'}
        monitor = DataQualityMonitor(config)
        
        # Feed normal ticks
        base_time = datetime.now()
        for i in range(50):
            tick = TickData(
                timestamp=base_time + timedelta(seconds=i),
                symbol="TEST",
                price=100.0 + np.random.randn() * 0.1,
                volume=1000
            )
            monitor.update_tick(tick)
            
        # Feed anomalous tick
        anomaly = TickData(
            timestamp=base_time + timedelta(seconds=51),
            symbol="TEST",
            price=110.0,  # 10% spike
            volume=10000
        )
        monitor.update_tick(anomaly)
        
        # Check detection
        assert monitor.metrics['anomalous_ticks'] > 0
        
        # Get report
        report = monitor.get_report()
        assert report['health_score'] < 1.0
        assert len(report['recommendations']) > 0


class TestMockDataGenerator:
    """Test mock data generation."""
    
    @pytest.mark.asyncio
    async def test_tick_generation(self):
        """Test generating mock ticks."""
        config = {
            'symbol': 'TEST',
            'base_price': 100.0,
            'volatility': 0.02,
            'trend': 0.0001,
            'base_volume': 1000
        }
        
        generator = MockDataGenerator(config)
        
        # Generate ticks
        ticks = await generator.generate_ticks(100)
        
        assert len(ticks) == 100
        
        # Check price evolution
        prices = [t.price for t in ticks]
        assert min(prices) > 90  # Reasonable bounds
        assert max(prices) < 110
        
        # Check timestamps are sequential
        for i in range(1, len(ticks)):
            assert ticks[i].timestamp > ticks[i-1].timestamp
            
    def test_historical_data_generation(self):
        """Test generating historical data."""
        config = {
            'symbol': 'TEST',
            'base_price': 100.0,
            'volatility': 0.01
        }
        
        generator = MockDataGenerator(config)
        
        # Generate one hour of data
        start = datetime(2024, 1, 1, 9, 0, 0)
        end = datetime(2024, 1, 1, 10, 0, 0)
        
        df = generator.generate_historical_data(start, end, tick_rate=1.0)
        
        assert len(df) == 3600  # One tick per second
        assert df.index[0] >= start
        assert df.index[-1] <= end


class TestIntegration:
    """Integration tests for complete data pipeline."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_flow(self, tmp_path):
        """Test complete flow from CSV to bars."""
        # Setup
        event_bus = EventBus()
        
        # Create test data
        csv_path = tmp_path / "test_data.csv"
        data = {
            'timestamp': pd.date_range('2024-01-01 09:00:00', periods=60, freq='1min'),
            'price': 100 + np.random.randn(60).cumsum() * 0.1,
            'volume': np.random.randint(100, 1000, 60)
        }
        pd.DataFrame(data).to_csv(csv_path, index=False)
        
        # Create components
        data_config = {
            'symbol': 'TEST',
            'csv_path': str(csv_path)
        }
        
        bar_config = {
            'symbol': 'TEST',
            'timeframes': [5, 30]
        }
        
        handler = BacktestDataHandler(data_config, event_bus)
        generator = BarGenerator(bar_config, event_bus)
        
        # Track outputs
        bars_5min = []
        bars_30min = []
        
        async def capture_5min(event):
            bars_5min.append(event.data['bar'])
            
        async def capture_30min(event):
            bars_30min.append(event.data['bar'])
            
        await event_bus.subscribe(EventType.NEW_5MIN_BAR, capture_5min)
        await event_bus.subscribe(EventType.NEW_30MIN_BAR, capture_30min)
        
        # Start components
        await generator.start()
        await handler.start()
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        # Verify results
        assert len(bars_5min) >= 11  # ~12 5-minute bars in 60 minutes
        assert len(bars_30min) >= 1   # At least 1 30-minute bar
        
        # Verify bar quality
        for bar in bars_5min:
            assert bar.high >= bar.low
            assert bar.high >= bar.open
            assert bar.high >= bar.close
            assert bar.low <= bar.open
            assert bar.low <= bar.close


class TestPerformance:
    """Performance tests for data pipeline."""
    
    @pytest.mark.asyncio
    async def test_tick_processing_speed(self):
        """Test tick processing performance."""
        event_bus = EventBus()
        config = {
            'symbol': 'TEST',
            'timeframes': [5, 30]
        }
        
        generator = BarGenerator(config, event_bus)
        await generator.start()
        
        # Generate many ticks
        base_time = datetime.now()
        processing_times = []
        
        for i in range(1000):
            tick = TickData(
                timestamp=base_time + timedelta(seconds=i * 0.1),
                symbol="TEST",
                price=100.0 + np.random.randn() * 0.1,
                volume=100
            )
            
            start = asyncio.get_event_loop().time()
            
            await event_bus.publish(Event(
                type=EventType.NEW_TICK,
                data={'tick': tick}
            ))
            
            end = asyncio.get_event_loop().time()
            processing_times.append((end - start) * 1_000_000)  # microseconds
            
        # Check performance
        avg_time = np.mean(processing_times)
        p95_time = np.percentile(processing_times, 95)
        
        assert avg_time < 100  # Average under 100μs
        assert p95_time < 200  # 95th percentile under 200μs


class TestDataRecorder:
    """Test data recording functionality."""
    
    @pytest.mark.asyncio
    async def test_recording_session(self, tmp_path):
        """Test recording market data."""
        config = {
            'output_dir': str(tmp_path),
            'buffer_size': 10
        }
        
        recorder = DataRecorder(config)
        
        # Start recording
        await recorder.start_recording("test_session")
        
        # Record some data
        for i in range(20):
            tick = TickData(
                timestamp=datetime.now() + timedelta(seconds=i),
                symbol="TEST",
                price=100.0 + i * 0.1,
                volume=100
            )
            await recorder.record_tick(tick)
            
        # Stop recording
        await recorder.stop_recording()
        
        # Verify files created
        tick_files = list(tmp_path.glob("ticks_*.parquet"))
        metadata_files = list(tmp_path.glob("metadata_*.json"))
        
        assert len(tick_files) == 1
        assert len(metadata_files) == 1
        
        # Verify data
        df = pd.read_parquet(tick_files[0])
        assert len(df) == 20
        assert 'price' in df.columns
        assert 'volume' in df.columns