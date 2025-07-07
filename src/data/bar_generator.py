"""
High-performance bar generator with dual timeframe support.

This module provides production-grade bar generation with sub-100μs tick processing,
precise temporal boundaries, and intelligent gap handling.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
import logging
import numpy as np

from src.core.events import EventType
from src.data.event_adapter import EventBus, Event
from src.data.data_handler import TickData
from src.utils.time_utils import TimeUtils
from src.utils.monitoring import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class BarData:
    """
    OHLCV bar data structure.
    
    Attributes:
        timestamp: Bar open timestamp (floored to timeframe)
        open: Opening price
        high: Highest price
        low: Lowest price
        close: Closing price
        volume: Total volume
        tick_count: Number of ticks in bar
        vwap: Volume-weighted average price
        is_synthetic: True if forward-filled
        timeframe: Bar timeframe in minutes
    """
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    tick_count: int
    vwap: float
    is_synthetic: bool = False
    timeframe: int = 5
    
    def __post_init__(self):
        """Validate bar data."""
        if self.high < self.low:
            raise ValueError(f"High {self.high} < Low {self.low}")
        if not (self.low <= self.open <= self.high):
            raise ValueError(f"Open {self.open} outside range [{self.low}, {self.high}]")
        if not (self.low <= self.close <= self.high):
            raise ValueError(f"Close {self.close} outside range [{self.low}, {self.high}]")
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'tick_count': self.tick_count,
            'vwap': self.vwap,
            'is_synthetic': self.is_synthetic,
            'timeframe': self.timeframe
        }


class WorkingBar:
    """
    Efficiently maintains state of bar under construction.
    
    Optimized for minimal memory footprint and fast updates.
    """
    
    def __init__(self, timestamp: datetime, timeframe: int):
        self.timestamp = timestamp
        self.timeframe = timeframe
        self.open: Optional[float] = None
        self.high: Optional[float] = None
        self.low: Optional[float] = None
        self.close: Optional[float] = None
        self.volume: int = 0
        self.tick_count: int = 0
        self.volume_sum: float = 0.0  # For VWAP calculation
        
    def update(self, price: float, volume: int):
        """Update bar with new tick."""
        if self.open is None:
            # First tick
            self.open = price
            self.high = price
            self.low = price
        else:
            # Update high/low
            self.high = max(self.high, price)
            self.low = min(self.low, price)
            
        # Always update close
        self.close = price
        
        # Update volume and tick count
        self.volume += volume
        self.tick_count += 1
        self.volume_sum += price * volume
        
    def to_bar(self, is_synthetic: bool = False) -> BarData:
        """Convert to completed BarData."""
        if self.open is None:
            raise ValueError("Cannot create bar with no data")
            
        # Calculate VWAP
        vwap = self.volume_sum / self.volume if self.volume > 0 else self.close
        
        return BarData(
            timestamp=self.timestamp,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            tick_count=self.tick_count,
            vwap=vwap,
            is_synthetic=is_synthetic,
            timeframe=self.timeframe
        )


class TimeframeBoundary:
    """
    Handles precise timeframe boundary calculations.
    
    Ensures bars align exactly with clock boundaries.
    """
    
    @staticmethod
    def floor_timestamp(timestamp: datetime, timeframe_minutes: int) -> datetime:
        """
        Floor timestamp to timeframe boundary.
        
        Examples:
            - 09:17:45 with 5-min → 09:15:00
            - 09:17:45 with 30-min → 09:00:00
        """
        # Remove seconds and microseconds
        timestamp = timestamp.replace(second=0, microsecond=0)
        
        # Floor to timeframe boundary
        minutes = timestamp.minute
        floored_minutes = (minutes // timeframe_minutes) * timeframe_minutes
        
        return timestamp.replace(minute=floored_minutes)
        
    @staticmethod
    def next_boundary(timestamp: datetime, timeframe_minutes: int) -> datetime:
        """Get next timeframe boundary."""
        current_boundary = TimeframeBoundary.floor_timestamp(timestamp, timeframe_minutes)
        return current_boundary + timedelta(minutes=timeframe_minutes)
        
    @staticmethod
    def get_missing_boundaries(
        last_timestamp: datetime,
        current_timestamp: datetime,
        timeframe_minutes: int
    ) -> List[datetime]:
        """
        Get list of missing boundaries for gap filling.
        
        Returns boundaries that should have bars but don't.
        """
        missing = []
        
        # Start from next boundary after last timestamp
        boundary = TimeframeBoundary.next_boundary(last_timestamp, timeframe_minutes)
        current_boundary = TimeframeBoundary.floor_timestamp(current_timestamp, timeframe_minutes)
        
        # Collect all boundaries up to (but not including) current
        while boundary < current_boundary:
            missing.append(boundary)
            boundary += timedelta(minutes=timeframe_minutes)
            
        return missing


class BarGenerator:
    """
    High-performance bar generator with dual timeframe support.
    
    Features:
    - Simultaneous 5-minute and 30-minute bar generation
    - Forward-fill gap handling
    - Sub-100μs tick processing
    - Deterministic output
    - Memory-efficient design
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.symbol = config['symbol']
        
        # Timeframes to generate
        self.timeframes = config.get('timeframes', [5, 30])
        
        # Active bars
        self.working_bars: Dict[int, Optional[WorkingBar]] = {
            tf: None for tf in self.timeframes
        }
        
        # Last completed bars (for gap filling)
        self.last_bars: Dict[int, Optional[BarData]] = {
            tf: None for tf in self.timeframes
        }
        
        # Last tick timestamp (for gap detection)
        self.last_tick_timestamp: Optional[datetime] = None
        
        # Metrics
        self.metrics = MetricsCollector(f"bar_generator_{self.symbol}")
        self.tick_count = 0
        self.bars_emitted = {tf: 0 for tf in self.timeframes}
        self.gaps_filled = {tf: 0 for tf in self.timeframes}
        
        # Performance optimization
        self._boundary_cache = {}
        
        logger.info(
            f"Initialized BarGenerator for {self.symbol} "
            f"with timeframes: {self.timeframes}"
        )
        
    async def start(self):
        """Start the bar generator."""
        logger.info(f"Starting BarGenerator for {self.symbol}")
        
        # Subscribe to tick events
        await self.event_bus.subscribe(
            EventType.NEW_TICK,
            self._on_tick,
            filter_func=lambda e: e.data['tick'].symbol == self.symbol
        )
        
    async def stop(self):
        """Stop the bar generator."""
        logger.info(f"Stopping BarGenerator for {self.symbol}")
        
        # Finalize any incomplete bars
        for timeframe in self.timeframes:
            if self.working_bars[timeframe] is not None:
                await self._complete_bar(timeframe, finalize=True)
                
        # Log statistics
        total_bars = sum(self.bars_emitted.values())
        total_gaps = sum(self.gaps_filled.values())
        
        logger.info(
            f"BarGenerator statistics: "
            f"{self.tick_count} ticks processed, "
            f"{total_bars} bars emitted, "
            f"{total_gaps} gaps filled"
        )
        
    async def _on_tick(self, event: Event):
        """Process incoming tick event."""
        tick: TickData = event.data['tick']
        
        # Update metrics
        self.tick_count += 1
        self.metrics.increment('ticks_processed')
        
        # Process tick with timing
        start_time = asyncio.get_event_loop().time()
        
        try:
            await self._process_tick(tick)
        except Exception as e:
            logger.error(f"Error processing tick: {e}")
            self.metrics.increment('tick_errors')
            
        # Measure processing time
        processing_time = (asyncio.get_event_loop().time() - start_time) * 1_000_000  # microseconds
        self.metrics.observe('tick_processing_time_us', processing_time)
        
        if processing_time > 100:
            logger.warning(f"Slow tick processing: {processing_time:.0f}μs")
            
    async def _process_tick(self, tick: TickData):
        """
        Process tick and update bars.
        
        Handles boundary detection and gap filling.
        """
        # Check for gaps if we have a last tick
        if self.last_tick_timestamp:
            await self._check_and_fill_gaps(tick.timestamp)
            
        # Process tick for each timeframe
        for timeframe in self.timeframes:
            await self._update_timeframe_bar(timeframe, tick)
            
        # Update last tick timestamp
        self.last_tick_timestamp = tick.timestamp
        
    async def _update_timeframe_bar(self, timeframe: int, tick: TickData):
        """Update bar for specific timeframe."""
        # Get current boundary
        boundary = self._get_boundary(tick.timestamp, timeframe)
        
        # Check if we need to complete current bar
        if self.working_bars[timeframe] is not None:
            if boundary > self.working_bars[timeframe].timestamp:
                # Complete current bar
                await self._complete_bar(timeframe)
                
        # Create new working bar if needed
        if self.working_bars[timeframe] is None:
            self.working_bars[timeframe] = WorkingBar(boundary, timeframe)
            
        # Update working bar
        self.working_bars[timeframe].update(tick.price, tick.volume)
        
    def _get_boundary(self, timestamp: datetime, timeframe: int) -> datetime:
        """Get timeframe boundary with caching."""
        cache_key = (timestamp.replace(second=0, microsecond=0), timeframe)
        
        if cache_key not in self._boundary_cache:
            self._boundary_cache[cache_key] = TimeframeBoundary.floor_timestamp(
                timestamp, timeframe
            )
            
            # Limit cache size
            if len(self._boundary_cache) > 1000:
                self._boundary_cache.clear()
                
        return self._boundary_cache[cache_key]
        
    async def _complete_bar(self, timeframe: int, finalize: bool = False):
        """Complete and emit bar."""
        working_bar = self.working_bars[timeframe]
        if working_bar is None:
            return
            
        try:
            # Convert to BarData
            bar = working_bar.to_bar()
            
            # Emit bar event
            await self._emit_bar(bar, timeframe)
            
            # Update last bar
            self.last_bars[timeframe] = bar
            
            # Clear working bar (unless finalizing)
            if not finalize:
                self.working_bars[timeframe] = None
                
        except Exception as e:
            logger.error(f"Error completing bar: {e}")
            self.metrics.increment('bar_errors')
            
    async def _emit_bar(self, bar: BarData, timeframe: int):
        """Emit bar event."""
        # Determine event type
        event_type = EventType.NEW_5MIN_BAR if timeframe == 5 else EventType.NEW_30MIN_BAR
        
        # Create event
        event = Event(
            type=event_type,
            data={'bar': bar},
            source=f"bar_generator_{self.symbol}"
        )
        
        # Publish event
        await self.event_bus.publish(event)
        
        # Update metrics
        self.bars_emitted[timeframe] += 1
        self.metrics.increment(f'bars_emitted_{timeframe}min')
        
        logger.debug(
            f"Emitted {timeframe}-min bar: "
            f"{bar.timestamp} O:{bar.open} H:{bar.high} "
            f"L:{bar.low} C:{bar.close} V:{bar.volume}"
        )
        
    async def _check_and_fill_gaps(self, current_timestamp: datetime):
        """Check for gaps and forward-fill if needed."""
        for timeframe in self.timeframes:
            last_bar = self.last_bars[timeframe]
            if last_bar is None:
                continue
                
            # Get missing boundaries
            missing = TimeframeBoundary.get_missing_boundaries(
                last_bar.timestamp,
                current_timestamp,
                timeframe
            )
            
            # Forward-fill missing bars
            for boundary in missing:
                await self._create_synthetic_bar(boundary, timeframe, last_bar)
                
    async def _create_synthetic_bar(
        self,
        timestamp: datetime,
        timeframe: int,
        last_bar: BarData
    ):
        """Create synthetic bar for gap filling."""
        # Forward-fill with last close price
        synthetic_bar = BarData(
            timestamp=timestamp,
            open=last_bar.close,
            high=last_bar.close,
            low=last_bar.close,
            close=last_bar.close,
            volume=0,
            tick_count=0,
            vwap=last_bar.close,
            is_synthetic=True,
            timeframe=timeframe
        )
        
        # Emit synthetic bar
        await self._emit_bar(synthetic_bar, timeframe)
        
        # Update metrics
        self.gaps_filled[timeframe] += 1
        self.metrics.increment(f'gaps_filled_{timeframe}min')
        
        logger.info(
            f"Created synthetic {timeframe}-min bar at {timestamp} "
            f"(gap fill from {last_bar.timestamp})"
        )
        
        # Update last bar
        self.last_bars[timeframe] = synthetic_bar


class BarValidator:
    """
    Validates bar data quality and consistency.
    """
    
    @staticmethod
    def validate_bar_sequence(bars: List[BarData]) -> Tuple[bool, List[str]]:
        """
        Validate a sequence of bars.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if not bars:
            return True, errors
            
        # Check temporal ordering
        for i in range(1, len(bars)):
            if bars[i].timestamp <= bars[i-1].timestamp:
                errors.append(
                    f"Bar {i} timestamp {bars[i].timestamp} "
                    f"not after previous {bars[i-1].timestamp}"
                )
                
        # Check price continuity
        for i in range(1, len(bars)):
            prev_close = bars[i-1].close
            curr_open = bars[i].open
            
            # Allow small gap for market gaps
            max_gap = prev_close * 0.05  # 5%
            
            if abs(curr_open - prev_close) > max_gap and not bars[i].is_synthetic:
                errors.append(
                    f"Large gap detected: prev close {prev_close} "
                    f"to open {curr_open} ({abs(curr_open - prev_close)/prev_close:.1%})"
                )
                
        return len(errors) == 0, errors