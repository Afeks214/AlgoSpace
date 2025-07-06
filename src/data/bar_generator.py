"""
BarGenerator Component - Time-Series Aggregation Engine

This module transforms continuous tick data into discrete, time-based OHLCV bars.
It simultaneously maintains two timeframes (5-minute and 30-minute) with temporal
accuracy and gap handling as specified in the trading strategy.

Based on Master PRD - BarGenerator Component v1.0
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import structlog

from ..core.kernel import ComponentBase, AlgoSpaceKernel
from ..core.events import EventType, Event, TickData, BarData
from ..utils.logger import get_logger
from .validators import BarValidator


class BarGenerator(ComponentBase):
    """
    Time-Series Aggregation Engine
    
    Transforms tick data into OHLCV bars for multiple timeframes.
    Maintains temporal accuracy and handles data gaps gracefully.
    
    Key Features:
    - Simultaneous 5-minute and 30-minute bar generation
    - Forward-fill gap handling
    - Sub-100 microsecond processing latency
    - Deterministic output for backtesting consistency
    """
    
    def __init__(self, name: str, kernel: AlgoSpaceKernel):
        """
        Initialize BarGenerator
        
        Args:
            name: Component name
            kernel: System kernel instance
        """
        super().__init__(name, kernel)
        
        # Configuration
        self.timeframes = self.config.timeframes  # [5, 30] from config
        self.symbol = self.config.primary_symbol
        self.gap_fill = True  # Always enabled per PRD
        
        # Active bars being constructed
        self.active_bars: Dict[int, Optional['WorkingBar']] = {
            timeframe: None for timeframe in self.timeframes
        }
        
        # Statistics
        self.tick_count = 0
        self.bars_emitted = {timeframe: 0 for timeframe in self.timeframes}
        self.gaps_filled = {timeframe: 0 for timeframe in self.timeframes}
        
        # Last bar timestamps for gap detection
        self.last_bar_timestamps: Dict[int, Optional[datetime]] = {
            timeframe: None for timeframe in self.timeframes
        }
        
        # Initialize bar validator
        self.bar_validator = BarValidator()
        
        self.logger.info(f"BarGenerator initialized timeframes={self.timeframes} symbol={self.symbol} gap_fill={self.gap_fill}")
    
    async def start(self) -> None:
        """Start the BarGenerator component"""
        await super().start()
        
        # Subscribe to NEW_TICK events from DataHandler
        self.subscribe_to_event(EventType.NEW_TICK, self._on_new_tick)
        
        self.logger.info("BarGenerator started - subscribed to NEW_TICK events")
    
    async def stop(self) -> None:
        """Stop the BarGenerator component"""
        # Finalize any incomplete bars
        for timeframe in self.timeframes:
            if self.active_bars[timeframe] is not None:
                await self._finalize_bar(timeframe)
        
        # Log final statistics
        total_bars = sum(self.bars_emitted.values())
        total_gaps = sum(self.gaps_filled.values())
        
        self.logger.info(f"BarGenerator stopped total_ticks_processed={self.tick_count} total_bars_emitted={total_bars} total_gaps_filled={total_gaps} bars_by_timeframe={self.bars_emitted}")
        
        await super().stop()
    
    def _on_new_tick(self, event: Event) -> None:
        """
        Process incoming tick data
        
        Args:
            event: NEW_TICK event containing TickData
        """
        tick_data: TickData = event.payload
        
        try:
            # Validate tick data
            if not self._validate_tick(tick_data):
                return
            
            self.tick_count += 1
            
            # Process tick for each timeframe
            for timeframe in self.timeframes:
                self._process_tick_for_timeframe(tick_data, timeframe)
            
            # Debug logging (if enabled)
            if self.tick_count % 10000 == 0:  # Every 10k ticks
                self.logger.debug(f"Tick processing progress tick_count={self.tick_count} bars_emitted={self.bars_emitted}")
        
        except Exception as e:
            self.logger.error(f"Error processing tick tick_data={tick_data} error={str(e)}")
            # Continue processing to maintain system stability
    
    def _validate_tick(self, tick_data: TickData) -> bool:
        """
        Validate incoming tick data
        
        Args:
            tick_data: Tick data to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check for required fields
            if not tick_data.symbol or not tick_data.timestamp:
                self.logger.warning(f"Invalid tick - missing symbol or timestamp tick={tick_data}")
                return False
            
            # Check price validity
            if tick_data.price <= 0:
                self.logger.warning(f"Invalid tick - negative or zero price tick={tick_data}")
                return False
            
            # Check volume validity (allow zero volume)
            if tick_data.volume < 0:
                self.logger.warning(f"Invalid tick - negative volume tick={tick_data}")
                return False
            
            # Check symbol match
            if tick_data.symbol != self.symbol:
                self.logger.debug(f"Tick for different symbol ignored expected={self.symbol} received={tick_data.symbol}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating tick tick={tick_data} error={str(e)}")
            return False
    
    def _process_tick_for_timeframe(self, tick_data: TickData, timeframe: int) -> None:
        """
        Process tick for specific timeframe
        
        Args:
            tick_data: Incoming tick data
            timeframe: Timeframe in minutes (5 or 30)
        """
        # Calculate bar timestamp (floor to timeframe boundary)
        bar_timestamp = self._floor_timestamp(tick_data.timestamp, timeframe)
        
        # Check if this tick belongs to a new bar
        current_bar = self.active_bars[timeframe]
        
        if current_bar is None or bar_timestamp > current_bar.timestamp:
            # New bar period detected
            
            # Finalize previous bar if exists
            if current_bar is not None:
                self._emit_bar(timeframe, current_bar)
            
            # Handle gaps (missing bars)
            self._handle_gaps(timeframe, bar_timestamp)
            
            # Start new bar
            self.active_bars[timeframe] = WorkingBar(
                symbol=tick_data.symbol,
                timestamp=bar_timestamp,
                timeframe=timeframe
            )
            
            current_bar = self.active_bars[timeframe]
        
        # Update the current bar with tick data
        current_bar.update_with_tick(tick_data)
    
    def _floor_timestamp(self, timestamp: datetime, timeframe: int) -> datetime:
        """
        Floor timestamp to timeframe boundary
        
        Args:
            timestamp: Original timestamp
            timeframe: Timeframe in minutes
            
        Returns:
            Floored timestamp
            
        Examples:
            10:32:45 with 5min -> 10:30:00
            10:34:59 with 5min -> 10:30:00
            10:35:00 with 5min -> 10:35:00
            10:32:45 with 30min -> 10:30:00
            10:59:59 with 30min -> 10:30:00
            11:00:00 with 30min -> 11:00:00
        """
        # Floor to minute boundary first
        floored = timestamp.replace(second=0, microsecond=0)
        
        # Calculate minutes since epoch
        minutes_since_midnight = floored.hour * 60 + floored.minute
        
        # Floor to timeframe boundary
        floored_minutes = (minutes_since_midnight // timeframe) * timeframe
        
        # Convert back to timestamp
        hours = floored_minutes // 60
        minutes = floored_minutes % 60
        
        return floored.replace(hour=hours, minute=minutes)
    
    def _handle_gaps(self, timeframe: int, new_bar_timestamp: datetime) -> None:
        """
        Handle gaps in data by forward-filling missing bars
        
        Args:
            timeframe: Timeframe in minutes
            new_bar_timestamp: Timestamp of new bar
        """
        last_timestamp = self.last_bar_timestamps[timeframe]
        
        if last_timestamp is None:
            # First bar for this timeframe
            self.last_bar_timestamps[timeframe] = new_bar_timestamp
            return
        
        # Calculate expected next bar timestamp
        expected_next = last_timestamp + timedelta(minutes=timeframe)
        
        # Check for gaps
        gap_count = 0
        current_gap_timestamp = expected_next
        
        while current_gap_timestamp < new_bar_timestamp:
            # Create synthetic bar (forward-fill)
            gap_bar = self._create_gap_fill_bar(timeframe, current_gap_timestamp)
            
            if gap_bar:
                self._emit_bar(timeframe, gap_bar, is_gap_fill=True)
                gap_count += 1
            
            current_gap_timestamp += timedelta(minutes=timeframe)
        
        if gap_count > 0:
            self.gaps_filled[timeframe] += gap_count
            self.logger.info(f"Gap detected and filled",
                           timeframe=timeframe,
                           gap_count=gap_count,
                           gap_start=expected_next.isoformat(),
                           gap_end=new_bar_timestamp.isoformat())
        
        self.last_bar_timestamps[timeframe] = new_bar_timestamp
    
    def _create_gap_fill_bar(self, timeframe: int, timestamp: datetime) -> Optional['WorkingBar']:
        """
        Create a gap-fill bar using the last known price
        
        Args:
            timeframe: Timeframe in minutes
            timestamp: Gap bar timestamp
            
        Returns:
            Gap-fill bar or None if no previous data
        """
        # Get last known price from the most recent active bar
        last_price = None
        
        for tf in self.timeframes:
            active_bar = self.active_bars[tf]
            if active_bar and active_bar.close is not None:
                last_price = active_bar.close
                break
        
        if last_price is None:
            # No previous price data available
            self.logger.warning(f"Cannot create gap-fill bar - no previous price data timeframe={timeframe} timestamp={timestamp.isoformat()}")
            return None
        
        # Create synthetic bar with OHLC = last_price, volume = 0
        gap_bar = WorkingBar(
            symbol=self.symbol,
            timestamp=timestamp,
            timeframe=timeframe
        )
        
        # Set all OHLC values to last known price
        gap_bar.open = last_price
        gap_bar.high = last_price
        gap_bar.low = last_price
        gap_bar.close = last_price
        gap_bar.volume = 0
        gap_bar.is_gap_fill = True
        
        return gap_bar
    
    def _emit_bar(self, timeframe: int, working_bar: 'WorkingBar', is_gap_fill: bool = False) -> None:
        """
        Emit a completed bar as an event
        
        Args:
            timeframe: Timeframe in minutes
            working_bar: Completed working bar
            is_gap_fill: Whether this is a gap-fill bar
        """
        try:
            # Create BarData object
            bar_data = BarData(
                symbol=working_bar.symbol,
                timestamp=working_bar.timestamp,
                open=working_bar.open,
                high=working_bar.high,
                low=working_bar.low,
                close=working_bar.close,
                volume=working_bar.volume,
                timeframe=timeframe
            )
            
            # Validate bar before emitting
            is_valid, validation_errors = self.bar_validator.validate_bar(bar_data)
            
            if not is_valid:
                self.logger.error(f"Invalid bar data detected timeframe={timeframe} bar={bar_data} errors={validation_errors} is_gap_fill={is_gap_fill}")
                # Skip emitting invalid bars
                return
            
            # Determine event type based on timeframe
            if timeframe == 5:
                event_type = EventType.NEW_5MIN_BAR
            elif timeframe == 30:
                event_type = EventType.NEW_30MIN_BAR
            else:
                event_type = EventType.NEW_BAR  # Generic fallback
            
            # Publish valid bar
            self.publish_event(event_type, bar_data)
            
            # Update statistics
            self.bars_emitted[timeframe] += 1
            
            # Log bar completion
            log_level = "debug" if is_gap_fill else "info"
            getattr(self.logger, log_level)(
                f"{timeframe}-min bar completed",
                timestamp=working_bar.timestamp.isoformat(),
                ohlcv=f"O:{working_bar.open:.2f} H:{working_bar.high:.2f} "
                      f"L:{working_bar.low:.2f} C:{working_bar.close:.2f} V:{working_bar.volume}",
                is_gap_fill=is_gap_fill
            )
            
        except Exception as e:
            self.logger.error(f"Error emitting bar timeframe={timeframe} bar={working_bar} error={str(e)}")
    
    async def _finalize_bar(self, timeframe: int) -> None:
        """
        Finalize incomplete bar during shutdown
        
        Args:
            timeframe: Timeframe to finalize
        """
        working_bar = self.active_bars[timeframe]
        
        if working_bar and working_bar.open is not None:
            # Ensure bar has valid OHLC values
            if working_bar.close is None:
                working_bar.close = working_bar.open
            if working_bar.high is None:
                working_bar.high = working_bar.open
            if working_bar.low is None:
                working_bar.low = working_bar.open
            
            self._emit_bar(timeframe, working_bar)
            self.logger.info(f"Finalized incomplete {timeframe}-min bar during shutdown")
        
        self.active_bars[timeframe] = None


class WorkingBar:
    """
    Internal representation of a bar being constructed
    
    Efficiently tracks OHLCV values as ticks arrive.
    """
    
    def __init__(self, symbol: str, timestamp: datetime, timeframe: int):
        """
        Initialize working bar
        
        Args:
            symbol: Trading symbol
            timestamp: Bar start timestamp
            timeframe: Timeframe in minutes
        """
        self.symbol = symbol
        self.timestamp = timestamp
        self.timeframe = timeframe
        
        # OHLCV values
        self.open: Optional[float] = None
        self.high: Optional[float] = None
        self.low: Optional[float] = None
        self.close: Optional[float] = None
        self.volume: int = 0
        
        # Metadata
        self.tick_count = 0
        self.is_gap_fill = False
    
    def update_with_tick(self, tick_data: TickData) -> None:
        """
        Update bar with new tick data
        
        Args:
            tick_data: Incoming tick data
        """
        price = tick_data.price
        volume = tick_data.volume
        
        # First tick of bar
        if self.open is None:
            self.open = price
            self.high = price
            self.low = price
        else:
            # Update high and low
            if price > self.high:
                self.high = price
            if price < self.low:
                self.low = price
        
        # Always update close and volume
        self.close = price
        self.volume += volume
        self.tick_count += 1
    
    def __str__(self) -> str:
        """String representation of working bar"""
        return (f"WorkingBar({self.symbol} {self.timestamp.isoformat()} "
                f"OHLCV: {self.open}/{self.high}/{self.low}/{self.close}/{self.volume})")
    
    def __repr__(self) -> str:
        """Detailed representation of working bar"""
        return (f"WorkingBar(symbol='{self.symbol}', timestamp={self.timestamp}, "
                f"timeframe={self.timeframe}, open={self.open}, high={self.high}, "
                f"low={self.low}, close={self.close}, volume={self.volume}, "
                f"tick_count={self.tick_count})")