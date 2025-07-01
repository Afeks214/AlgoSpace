"""Bar generator component for AlgoSpace trading system.

This module aggregates tick data into OHLCV bars for multiple timeframes.
It handles time gaps in data by generating synthetic bars to ensure continuous
time series for technical indicators.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class BarData:
    """Standardized bar data structure.
    
    Attributes:
        timestamp: The bar start timestamp
        open: Opening price
        high: Highest price during the bar
        low: Lowest price during the bar
        close: Closing price
        volume: Total volume during the bar
        timeframe: Timeframe in minutes (5 or 30)
    """
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: int


class BarGenerator:
    """Aggregates tick data into OHLCV bars for multiple timeframes.
    
    Maintains concurrent 5-minute and 30-minute bars, handles gaps in data
    by generating synthetic bars, and emits bar events through the event bus.
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: Any) -> None:
        """Initialize the bar generator.
        
        Args:
            config: Configuration dictionary
            event_bus: Event bus instance for publishing bar events
        """
        self.config = config
        self.event_bus = event_bus
        
        # Initialize bars for both timeframes
        self.bars_5min: Optional[Dict[str, Any]] = None
        self.bars_30min: Optional[Dict[str, Any]] = None
        
        # Track last completed bar timestamps for gap detection
        self.last_bar_time_5min: Optional[datetime] = None
        self.last_bar_time_30min: Optional[datetime] = None
        
        # Track last known price for gap filling
        self.last_close_price: Optional[float] = None
        
        # Statistics
        self.tick_count = 0
        self.bars_emitted_5min = 0
        self.bars_emitted_30min = 0
        self.gaps_filled_5min = 0
        self.gaps_filled_30min = 0
        
        logger.info("BarGenerator initialized")
    
    def on_new_tick(self, tick_data: Dict[str, Any]) -> None:
        """Process incoming tick data and update bars.
        
        This is the main entry point called by the event system when
        a NEW_TICK event is received.
        
        Args:
            tick_data: Dictionary containing tick information with keys:
                      timestamp (datetime), price (float), volume (int)
        """
        try:
            self.tick_count += 1
            
            # Extract tick data - handle both dict and dataclass formats
            if hasattr(tick_data, 'timestamp'):
                # It's a dataclass
                timestamp = tick_data.timestamp
                price = tick_data.price
                volume = tick_data.volume
            else:
                # It's a dict
                timestamp = tick_data['timestamp']
                price = tick_data['price']
                volume = tick_data['volume']
            
            # Update both timeframes
            self._update_bar_5min(timestamp, price, volume)
            self._update_bar_30min(timestamp, price, volume)
            
            # Update last known price for gap filling
            self.last_close_price = price
            
        except Exception as e:
            logger.error(f"Error processing tick: {e}")
    
    def _update_bar_5min(self, timestamp: datetime, price: float, volume: int) -> None:
        """Update or create 5-minute bar.
        
        Args:
            timestamp: Tick timestamp
            price: Tick price
            volume: Tick volume
        """
        bar_time = self._get_bar_time(timestamp, 5)
        
        # Check if we need to start a new bar
        if self.bars_5min is None or self.bars_5min['timestamp'] != bar_time:
            # Emit previous bar if it exists
            if self.bars_5min is not None:
                self._emit_bar(self.bars_5min, 5)
                
            # Check for gaps
            self._handle_gaps_5min(bar_time)
            
            # Start new bar
            self.bars_5min = {
                'timestamp': bar_time,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': volume
            }
        else:
            # Update existing bar
            self.bars_5min['high'] = max(self.bars_5min['high'], price)
            self.bars_5min['low'] = min(self.bars_5min['low'], price)
            self.bars_5min['close'] = price
            self.bars_5min['volume'] += volume
    
    def _update_bar_30min(self, timestamp: datetime, price: float, volume: int) -> None:
        """Update or create 30-minute bar.
        
        Args:
            timestamp: Tick timestamp
            price: Tick price
            volume: Tick volume
        """
        bar_time = self._get_bar_time(timestamp, 30)
        
        # Check if we need to start a new bar
        if self.bars_30min is None or self.bars_30min['timestamp'] != bar_time:
            # Emit previous bar if it exists
            if self.bars_30min is not None:
                self._emit_bar(self.bars_30min, 30)
                
            # Check for gaps
            self._handle_gaps_30min(bar_time)
            
            # Start new bar
            self.bars_30min = {
                'timestamp': bar_time,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': volume
            }
        else:
            # Update existing bar
            self.bars_30min['high'] = max(self.bars_30min['high'], price)
            self.bars_30min['low'] = min(self.bars_30min['low'], price)
            self.bars_30min['close'] = price
            self.bars_30min['volume'] += volume
    
    def _get_bar_time(self, timestamp: datetime, timeframe_minutes: int) -> datetime:
        """Calculate the bar bucket timestamp for a given tick timestamp.
        
        Rounds down to the nearest timeframe boundary.
        
        Args:
            timestamp: Original tick timestamp
            timeframe_minutes: Timeframe in minutes (5 or 30)
            
        Returns:
            Bar bucket timestamp
        """
        # Remove seconds and microseconds
        timestamp = timestamp.replace(second=0, microsecond=0)
        
        # Calculate minutes since midnight
        minutes_since_midnight = timestamp.hour * 60 + timestamp.minute
        
        # Round down to timeframe boundary
        bar_minutes = (minutes_since_midnight // timeframe_minutes) * timeframe_minutes
        
        # Create new timestamp
        bar_hour = bar_minutes // 60
        bar_minute = bar_minutes % 60
        
        return timestamp.replace(hour=bar_hour, minute=bar_minute)
    
    def _handle_gaps_5min(self, new_bar_time: datetime) -> None:
        """Handle gaps in 5-minute data by generating synthetic bars.
        
        Args:
            new_bar_time: Timestamp of the new bar being created
        """
        if self.last_bar_time_5min is None or self.last_close_price is None:
            self.last_bar_time_5min = new_bar_time
            return
        
        # Calculate expected next bar time
        expected_time = self.last_bar_time_5min + timedelta(minutes=5)
        
        # Generate synthetic bars for any gaps
        while expected_time < new_bar_time:
            synthetic_bar = {
                'timestamp': expected_time,
                'open': self.last_close_price,
                'high': self.last_close_price,
                'low': self.last_close_price,
                'close': self.last_close_price,
                'volume': 0
            }
            self._emit_bar(synthetic_bar, 5)
            self.gaps_filled_5min += 1
            logger.info(f"Generated synthetic 5-min bar for gap at {expected_time}")
            
            expected_time += timedelta(minutes=5)
        
        self.last_bar_time_5min = new_bar_time
    
    def _handle_gaps_30min(self, new_bar_time: datetime) -> None:
        """Handle gaps in 30-minute data by generating synthetic bars.
        
        Args:
            new_bar_time: Timestamp of the new bar being created
        """
        if self.last_bar_time_30min is None or self.last_close_price is None:
            self.last_bar_time_30min = new_bar_time
            return
        
        # Calculate expected next bar time
        expected_time = self.last_bar_time_30min + timedelta(minutes=30)
        
        # Generate synthetic bars for any gaps
        while expected_time < new_bar_time:
            synthetic_bar = {
                'timestamp': expected_time,
                'open': self.last_close_price,
                'high': self.last_close_price,
                'low': self.last_close_price,
                'close': self.last_close_price,
                'volume': 0
            }
            self._emit_bar(synthetic_bar, 30)
            self.gaps_filled_30min += 1
            logger.info(f"Generated synthetic 30-min bar for gap at {expected_time}")
            
            expected_time += timedelta(minutes=30)
        
        self.last_bar_time_30min = new_bar_time
    
    def _emit_bar(self, bar_dict: Dict[str, Any], timeframe: int) -> None:
        """Emit a completed bar through the event bus.
        
        Args:
            bar_dict: Dictionary containing bar OHLCV data
            timeframe: Timeframe in minutes (5 or 30)
        """
        # Create BarData object
        bar_data = BarData(
            timestamp=bar_dict['timestamp'],
            open=bar_dict['open'],
            high=bar_dict['high'],
            low=bar_dict['low'],
            close=bar_dict['close'],
            volume=bar_dict['volume'],
            timeframe=timeframe
        )
        
        # Determine event type based on timeframe
        if timeframe == 5:
            event_type = 'NEW_5MIN_BAR'
            self.bars_emitted_5min += 1
        elif timeframe == 30:
            event_type = 'NEW_30MIN_BAR'
            self.bars_emitted_30min += 1
        else:
            logger.error(f"Unsupported timeframe: {timeframe}")
            return
        
        # Publish bar event
        self.event_bus.publish(event_type, bar_data)
        
        # Log bar emission
        logger.debug(f"Emitted {timeframe}-min bar: {bar_data.timestamp} "
                    f"OHLC={bar_data.open:.2f}/{bar_data.high:.2f}/{bar_data.low:.2f}/{bar_data.close:.2f} "
                    f"V={bar_data.volume}")
    
    def get_statistics(self) -> Dict[str, int]:
        """Get bar generation statistics.
        
        Returns:
            Dictionary containing statistics about bar generation
        """
        return {
            'tick_count': self.tick_count,
            'bars_emitted_5min': self.bars_emitted_5min,
            'bars_emitted_30min': self.bars_emitted_30min,
            'gaps_filled_5min': self.gaps_filled_5min,
            'gaps_filled_30min': self.gaps_filled_30min
        }