# Product Requirements Document (PRD): BarGenerator Component

**Document Version**: 2.0  
**Date**: 2025-06-19  
**Status**: Refined  
**Component**: BarGenerator  

## 1.0 Overview

### 1.1 Purpose
The BarGenerator is a high-performance time-series aggregation engine that transforms a continuous stream of tick data into structured OHLCV (Open, High, Low, Close, Volume) bars across multiple concurrent timeframes. It serves as the critical bridge between raw market data and the structured data required by technical analysis and machine learning components.

### 1.2 Scope

**In Scope:**
- Multi-timeframe bar construction (5-minute, 30-minute)
- Precise timestamp-based bar finalization
- Real-time OHLCV aggregation logic
- Data gap handling and forward-fill mechanisms
- Event-driven bar publication
- Memory-efficient rolling window management
- Sub-millisecond processing performance

**Out of Scope:**
- Technical indicator calculations (handled by IndicatorEngine)
- Data storage or historical bar persistence
- Bar visualization or charting
- Non-time-based bar types (volume bars, tick bars)

### 1.3 Architectural Position
The BarGenerator occupies the second position in the data processing pipeline:
DataHandler → **BarGenerator** → IndicatorEngine → MatrixAssembler → AI Agents

## 2.0 Functional Requirements

### FR-BG-01: Multi-Timeframe Concurrent Processing
**Requirement**: The BarGenerator MUST simultaneously maintain and update bars for multiple timeframes.

**Specification**:
```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict
import numpy as np

@dataclass
class BarData:
    timestamp: datetime    # Bar start timestamp (floored)
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    tick_count: int = 0   # Number of ticks in this bar
    vwap: float = 0.0     # Volume Weighted Average Price

class MultiTimeframeBarGenerator:
    def __init__(self, timeframes: Dict[str, int]):
        # timeframes = {'5min': 5, '30min': 30}
        self.timeframes = timeframes
        self.active_bars = {}  # Current bars being built
        self.last_tick_time = None
        
    def process_tick(self, tick_data: TickData) -> List[BarData]:
        """Process incoming tick and return any completed bars"""
        completed_bars = []
        
        for timeframe_name, minutes in self.timeframes.items():
            bar = self._update_or_create_bar(tick_data, timeframe_name, minutes)
            if self._should_finalize_bar(tick_data.timestamp, timeframe_name, minutes):
                completed_bars.append(self._finalize_bar(timeframe_name))
                self._start_new_bar(tick_data, timeframe_name)
                
        return completed_bars
```

### FR-BG-02: Precise Bar Construction Logic
**Requirement**: The bar construction MUST follow standard OHLCV aggregation rules with perfect accuracy.

**OHLCV Calculation Rules**:
```python
def _update_bar_with_tick(self, bar: BarData, tick: TickData) -> BarData:
    """Update existing bar with new tick data"""
    
    # Open: Set only once (first tick of the bar)
    if bar.tick_count == 0:
        bar.open = tick.price
        bar.high = tick.price
        bar.low = tick.price
    
    # High: Continuous maximum
    bar.high = max(bar.high, tick.price)
    
    # Low: Continuous minimum  
    bar.low = min(bar.low, tick.price)
    
    # Close: Always the latest tick price
    bar.close = tick.price
    
    # Volume: Cumulative sum
    bar.volume += tick.volume
    
    # Tick count for quality metrics
    bar.tick_count += 1
    
    # VWAP calculation
    total_price_volume = (bar.vwap * (bar.volume - tick.volume)) + (tick.price * tick.volume)
    bar.vwap = total_price_volume / bar.volume if bar.volume > 0 else tick.price
    
    return bar
```

### FR-BG-03: Timestamp-Based Bar Finalization
**Requirement**: Bar finalization MUST be based on precise timestamp boundary calculations.

**Finalization Logic**:
```python
def _calculate_bar_boundaries(self, timestamp: datetime, timeframe_minutes: int) -> tuple:
    """Calculate bar start and end boundaries"""
    
    # Floor timestamp to the nearest timeframe boundary
    minutes_since_midnight = timestamp.hour * 60 + timestamp.minute
    bar_start_minutes = (minutes_since_midnight // timeframe_minutes) * timeframe_minutes
    
    bar_start = timestamp.replace(
        hour=bar_start_minutes // 60,
        minute=bar_start_minutes % 60,
        second=0,
        microsecond=0
    )
    
    bar_end = bar_start + timedelta(minutes=timeframe_minutes)
    
    return bar_start, bar_end

def _should_finalize_bar(self, tick_timestamp: datetime, timeframe: str, minutes: int) -> bool:
    """Determine if current bar should be finalized"""
    
    if self.active_bars.get(timeframe) is None:
        return False
        
    current_bar = self.active_bars[timeframe]
    _, bar_end = self._calculate_bar_boundaries(current_bar.timestamp, minutes)
    
    # Finalize if tick timestamp crosses the bar boundary
    return tick_timestamp >= bar_end
```

### FR-BG-04: Data Gap Handling
**Requirement**: The system MUST handle periods with no market activity gracefully.

**Gap Handling Strategy**:
```python
def _handle_data_gap(self, timeframe: str, gap_start: datetime, gap_end: datetime) -> List[BarData]:
    """Create forward-filled bars for data gaps"""
    
    gap_bars = []
    current_time = gap_start
    timeframe_minutes = self.timeframes[timeframe]
    
    # Get last known price for forward-filling
    last_close = self.active_bars[timeframe].close if self.active_bars.get(timeframe) else None
    
    if last_close is None:
        return gap_bars
    
    while current_time < gap_end:
        # Create a forward-filled bar
        gap_bar = BarData(
            timestamp=current_time,
            symbol=self.symbol,
            open=last_close,
            high=last_close,
            low=last_close,
            close=last_close,
            volume=0,
            tick_count=0,
            vwap=last_close
        )
        gap_bars.append(gap_bar)
        current_time += timedelta(minutes=timeframe_minutes)
    
    return gap_bars
```

### FR-BG-05: Event-Driven Bar Publication
**Requirement**: Completed bars MUST be published immediately via the event bus.

**Event Publishing**:
```python
async def _publish_completed_bar(self, bar: BarData, timeframe: str):
    """Publish completed bar to event bus"""
    
    event_type = f"NEW_{timeframe.upper()}_BAR"  # e.g., "NEW_5MIN_BAR"
    
    await self.event_bus.publish(
        event_type=event_type,
        payload=bar,
        priority="HIGH",
        timestamp=bar.timestamp,
        metadata={
            'timeframe': timeframe,
            'tick_count': bar.tick_count,
            'processing_time_us': self._get_processing_time()
        }
    )
    
    # Update metrics
    self.metrics.bars_generated += 1
    self.metrics.last_bar_timestamp = bar.timestamp
```

### FR-BG-06: Performance Monitoring
**Requirement**: The component MUST provide comprehensive performance metrics.

**Performance Metrics**:
```python
@dataclass
class BarGeneratorMetrics:
    ticks_processed: int = 0
    bars_generated: int = 0
    bars_per_timeframe: Dict[str, int] = field(default_factory=dict)
    average_processing_time_us: float = 0.0
    peak_processing_time_us: float = 0.0
    memory_usage_mb: float = 0.0
    data_gaps_handled: int = 0
    last_tick_timestamp: Optional[datetime] = None
    last_bar_timestamp: Optional[datetime] = None
    
    def update_processing_time(self, time_us: float):
        """Update processing time statistics"""
        self.peak_processing_time_us = max(self.peak_processing_time_us, time_us)
        # Exponential moving average
        alpha = 0.1
        self.average_processing_time_us = (
            alpha * time_us + (1 - alpha) * self.average_processing_time_us
        )
```

## 3.0 Interface Specifications

### 3.1 Configuration Interface
```yaml
bar_generator:
  timeframes:
    - name: "5min"
      minutes: 5
      priority: "high"
    - name: "30min" 
      minutes: 30
      priority: "medium"
      
  settings:
    symbol: "ES"
    gap_threshold_seconds: 30
    forward_fill_gaps: true
    max_gap_fill_bars: 10
    
  performance:
    max_processing_time_us: 100
    memory_limit_mb: 50
    metrics_update_interval: 60
    
  validation:
    min_tick_count_per_bar: 1
    max_price_change_percent: 10.0
    alert_on_unusual_volume: true
```

### 3.2 Event Interface

**Subscribed Events**:
- **NEW_TICK**: Primary input from DataHandler

**Published Events**:
- **NEW_5MIN_BAR**: 5-minute bar completion
- **NEW_30MIN_BAR**: 30-minute bar completion  
- **DATA_GAP_FILLED**: When gaps are forward-filled
- **BAR_GENERATION_ERROR**: Processing errors

**Event Payload Structure**:
```python
{
    "bar_data": BarData,
    "metadata": {
        "timeframe": "5min",
        "tick_count": 145,
        "processing_time_us": 75.2,
        "gap_filled": false,
        "quality_score": 0.98
    }
}
```

### 3.3 Management Interface
```python
class BarGeneratorManager:
    def get_active_bars(self) -> Dict[str, BarData]
    def get_metrics(self) -> BarGeneratorMetrics
    def reset_metrics(self) -> None
    def pause_timeframe(self, timeframe: str) -> None
    def resume_timeframe(self, timeframe: str) -> None
    def force_finalize_all_bars(self) -> List[BarData]
```

## 4.0 Dependencies & Interactions

### 4.1 Upstream Dependencies
- **DataHandler**: Source of NEW_TICK events
- **Event Bus**: For receiving tick data
- **Configuration System**: For timeframe and behavior settings

### 4.2 Downstream Dependencies
- **IndicatorEngine**: Primary consumer of bar events
- **Performance Monitor**: Consumer of metrics
- **Risk Management**: Consumer of bar quality indicators

### 4.3 Processing Flow
```
NEW_TICK → Timestamp Analysis → Bar Update → Finalization Check → Event Publication
    ↓              ↓                ↓              ↓                    ↓
Gap Detection → Forward Fill → OHLCV Calc → Quality Check → Metrics Update
```

## 5.0 Non-Functional Requirements

### 5.1 Performance
- **NFR-BG-01**: Tick processing MUST complete in under 100 microseconds (95th percentile)
- **NFR-BG-02**: Bar finalization MUST add under 1ms latency to the pipeline
- **NFR-BG-03**: Memory footprint MUST remain constant (no memory leaks)
- **NFR-BG-04**: Support 10,000+ ticks/second sustained throughput

### 5.2 Accuracy
- **NFR-BG-05**: OHLCV calculations MUST be 100% accurate (no rounding errors)
- **NFR-BG-06**: Timestamp boundaries MUST be precise to the microsecond
- **NFR-BG-07**: All data gaps MUST be detected and handled appropriately

### 5.3 Reliability
- **NFR-BG-08**: Component MUST handle malformed tick data gracefully
- **NFR-BG-09**: Processing MUST continue if one timeframe fails
- **NFR-BG-10**: System MUST recover from memory pressure situations

## 6.0 Implementation Specifications

### 6.1 Technology Stack
```python
import asyncio
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import deque
import logging
import psutil
```

### 6.2 Optimization Techniques
```python
# Use slots for memory efficiency
@dataclass
class BarData:
    __slots__ = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'tick_count', 'vwap']

# Pre-allocated arrays for high-frequency operations
class OptimizedBarGenerator:
    def __init__(self):
        # Pre-allocate commonly used datetime objects
        self._time_cache = {}
        
        # Use deque for efficient append/pop operations
        self._recent_ticks = deque(maxlen=1000)
        
        # Numpy arrays for vectorized calculations
        self._price_buffer = np.zeros(1000, dtype=np.float64)
        self._volume_buffer = np.zeros(1000, dtype=np.int32)
```

### 6.3 Error Handling
```python
class BarGeneratorError(Exception):
    """Base exception for BarGenerator errors"""
    pass

class TimestampError(BarGeneratorError):
    """Timestamp-related errors"""
    pass

class DataQualityError(BarGeneratorError):
    """Data quality issues"""
    pass

# Graceful error handling
async def safe_process_tick(self, tick: TickData) -> List[BarData]:
    try:
        return await self.process_tick(tick)
    except TimestampError as e:
        logger.warning(f"Timestamp issue: {e}, skipping tick")
        return []
    except DataQualityError as e:
        logger.error(f"Data quality issue: {e}")
        self.metrics.quality_errors += 1
        return []
    except Exception as e:
        logger.critical(f"Unexpected error: {e}")
        # Emergency bar finalization to prevent data loss
        return self.emergency_finalize_all_bars()
```

## 7.0 Testing Requirements

### 7.1 Unit Tests
- OHLCV calculation accuracy across different price movements
- Timestamp boundary calculations for various timeframes
- Data gap detection and forward-fill logic
- Memory management and leak detection
- Error handling for malformed data

### 7.2 Integration Tests
- End-to-end tick-to-bar processing
- Event bus integration and event publishing
- Multi-timeframe concurrent processing
- Performance under high tick volumes
- Configuration-driven behavior changes

### 7.3 Performance Tests
```python
# Throughput test
async def test_high_volume_processing():
    """Test processing 10,000 ticks/second for 1 hour"""
    generator = BarGenerator({'5min': 5, '30min': 30})
    
    start_time = time.time()
    tick_count = 0
    
    for _ in range(36_000_000):  # 10k ticks/sec * 3600 sec
        tick = generate_synthetic_tick()
        await generator.process_tick(tick)
        tick_count += 1
        
        if tick_count % 100_000 == 0:
            processing_rate = tick_count / (time.time() - start_time)
            assert processing_rate >= 10_000, f"Processing rate too slow: {processing_rate}"

# Latency test
def test_processing_latency():
    """Ensure sub-100 microsecond processing"""
    generator = BarGenerator({'5min': 5})
    
    latencies = []
    for _ in range(10_000):
        tick = generate_synthetic_tick()
        start = time.perf_counter()
        generator.process_tick(tick)
        end = time.perf_counter()
        latencies.append((end - start) * 1_000_000)  # Convert to microseconds
    
    p95_latency = np.percentile(latencies, 95)
    assert p95_latency < 100, f"95th percentile latency too high: {p95_latency}μs"
```

## 8.0 Future Enhancements

### 8.1 V2.0 Features
- **Non-Time-Based Bars**: Volume bars, tick bars, Renko bars
- **Configurable Timeframes**: Dynamic timeframe configuration at runtime
- **Advanced Gap Handling**: Intelligent gap filling based on market conditions
- **Bar Compression**: Real-time bar data compression for storage efficiency
- **Quality Scoring**: Advanced bar quality metrics and scoring algorithms

### 8.2 Performance Optimizations
- **SIMD Operations**: Vectorized calculations using NumPy and SIMD instructions
- **Memory Pool**: Pre-allocated memory pools for zero-allocation processing
- **Parallel Processing**: Multi-threaded processing for independent timeframes
- **GPU Acceleration**: CUDA-based calculations for ultra-high frequency data

This PRD establishes the foundation for a high-performance, accurate, and reliable bar generation system that can handle the demanding requirements of real-time algorithmic trading while maintaining the precision necessary for sophisticated technical analysis and machine learning applications.