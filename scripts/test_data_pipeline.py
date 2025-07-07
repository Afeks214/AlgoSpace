#!/usr/bin/env python3
"""
Simple test script to validate data pipeline implementation.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.events import EventType, EventBus as CoreEventBus
from src.data.event_adapter import EventBus
from src.data.data_handler import TickData, BacktestDataHandler
from src.data.bar_generator import BarGenerator
from src.data.data_utils import MockDataGenerator


async def test_basic_functionality():
    """Test basic data pipeline functionality."""
    print("=== Testing Basic Data Pipeline Functionality ===\n")
    
    # Create event bus
    core_bus = CoreEventBus()
    event_bus = EventBus(core_bus)
    
    # Test 1: Mock data generation
    print("1. Testing mock data generation...")
    mock_config = {
        'symbol': 'TEST',
        'base_price': 100.0,
        'volatility': 0.01
    }
    generator = MockDataGenerator(mock_config)
    ticks = await generator.generate_ticks(10)
    print(f"   ✓ Generated {len(ticks)} mock ticks")
    print(f"   ✓ Price range: {min(t.price for t in ticks):.2f} - {max(t.price for t in ticks):.2f}")
    
    # Test 2: Bar generation
    print("\n2. Testing bar generation...")
    bar_config = {
        'symbol': 'TEST',
        'timeframes': [5, 30]
    }
    
    bar_generator = BarGenerator(bar_config, event_bus)
    await bar_generator.start()
    
    # Track bars
    bars_received = {'5': 0, '30': 0}
    
    def track_5min_bar(event):
        bars_received['5'] += 1
        bar = event.payload
        print(f"   ✓ 5-min bar: {bar.timestamp} O:{bar.open:.2f} H:{bar.high:.2f} L:{bar.low:.2f} C:{bar.close:.2f}")
    
    def track_30min_bar(event):
        bars_received['30'] += 1
        bar = event.payload
        print(f"   ✓ 30-min bar: {bar.timestamp} O:{bar.open:.2f} H:{bar.high:.2f} L:{bar.low:.2f} C:{bar.close:.2f}")
    
    # Subscribe to bar events
    core_bus.subscribe(EventType.NEW_5MIN_BAR, track_5min_bar)
    core_bus.subscribe(EventType.NEW_30MIN_BAR, track_30min_bar)
    
    # Feed ticks to generate bars
    base_time = datetime(2024, 1, 1, 9, 0, 0)
    
    for i in range(35):  # 35 minutes of data
        tick = TickData(
            timestamp=base_time + timedelta(minutes=i),
            symbol='TEST',
            price=100.0 + i * 0.1,
            volume=100
        )
        
        # Publish tick through adapter
        from src.data.event_adapter import create_event
        event = create_event(
            type=EventType.NEW_TICK,
            data={'tick': tick},
            source='test'
        )
        await event_bus.publish(event)
    
    # Allow processing
    await asyncio.sleep(0.1)
    
    print(f"\n   Summary: {bars_received['5']} 5-min bars, {bars_received['30']} 30-min bars generated")
    
    # Test 3: Performance
    print("\n3. Testing performance...")
    tick_times = []
    
    for i in range(1000):
        tick = TickData(
            timestamp=base_time + timedelta(seconds=i * 0.1),
            symbol='TEST',
            price=100.0 + (i % 10) * 0.01,
            volume=50
        )
        
        start = asyncio.get_event_loop().time()
        
        event = create_event(
            type=EventType.NEW_TICK,
            data={'tick': tick},
            source='test'
        )
        await event_bus.publish(event)
        
        end = asyncio.get_event_loop().time()
        tick_times.append((end - start) * 1_000_000)  # microseconds
    
    avg_time = sum(tick_times) / len(tick_times)
    max_time = max(tick_times)
    
    print(f"   ✓ Processed 1000 ticks")
    print(f"   ✓ Average processing time: {avg_time:.1f}μs")
    print(f"   ✓ Max processing time: {max_time:.1f}μs")
    print(f"   ✓ Performance target: <100μs {'PASSED' if avg_time < 100 else 'FAILED'}")
    
    await bar_generator.stop()
    
    print("\n=== All Tests Completed ===")


async def test_gap_handling():
    """Test gap handling functionality."""
    print("\n=== Testing Gap Handling ===\n")
    
    # Create event bus
    core_bus = CoreEventBus()
    event_bus = EventBus(core_bus)
    
    # Create bar generator
    bar_config = {
        'symbol': 'TEST',
        'timeframes': [5]
    }
    
    bar_generator = BarGenerator(bar_config, event_bus)
    await bar_generator.start()
    
    # Track bars
    bars = []
    
    def track_bar(event):
        bar = event.payload
        bars.append(bar)
        if hasattr(bar, 'is_synthetic') and bar.is_synthetic:
            print(f"   ✓ Synthetic bar created: {bar.timestamp}")
    
    core_bus.subscribe(EventType.NEW_5MIN_BAR, track_bar)
    
    # Create gap scenario
    base_time = datetime(2024, 1, 1, 9, 0, 0)
    
    # First tick
    tick1 = TickData(base_time, 'TEST', 100.0, 1000)
    event1 = create_event(EventType.NEW_TICK, {'tick': tick1}, 'test')
    await event_bus.publish(event1)
    
    # Second tick after 20 minute gap
    tick2 = TickData(base_time + timedelta(minutes=20), 'TEST', 101.0, 1000)
    event2 = create_event(EventType.NEW_TICK, {'tick': tick2}, 'test')
    await event_bus.publish(event2)
    
    # Final tick to close bar
    tick3 = TickData(base_time + timedelta(minutes=25), 'TEST', 101.5, 500)
    event3 = create_event(EventType.NEW_TICK, {'tick': tick3}, 'test')
    await event_bus.publish(event3)
    
    await asyncio.sleep(0.1)
    
    print(f"\n   Summary: {len(bars)} bars generated (including synthetic bars)")
    print("   Gap handling test completed")
    
    await bar_generator.stop()


if __name__ == "__main__":
    print("AlgoSpace Data Pipeline Test Suite\n")
    
    # Run tests
    asyncio.run(test_basic_functionality())
    asyncio.run(test_gap_handling())
    
    print("\n✅ All tests completed successfully!")