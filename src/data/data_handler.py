"""
Production-grade data handler implementation.

This module provides unified data handling for both backtest and live trading modes,
with comprehensive validation, monitoring, and error handling.
"""

import asyncio
import csv
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any, List, AsyncIterator, Union, Tuple
import numpy as np
import pandas as pd
from collections import deque
import aiofiles

from src.core.events import EventType
from src.data.event_adapter import EventBus, Event
from src.utils.monitoring import MetricsCollector
from src.utils.time_utils import TimeUtils

logger = logging.getLogger(__name__)


@dataclass
class TickData:
    """
    Standardized tick data structure.
    
    Attributes:
        timestamp: Tick timestamp with microsecond precision
        symbol: Trading symbol
        price: Tick price
        volume: Tick volume
        bid: Bid price (optional)
        ask: Ask price (optional)
        source: Data source identifier
        sequence_number: For ordering and gap detection
    """
    timestamp: datetime
    symbol: str
    price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    source: str = "unknown"
    sequence_number: Optional[int] = None
    
    def __post_init__(self):
        """Validate tick data on creation."""
        if self.price <= 0:
            raise ValueError(f"Invalid price: {self.price}")
        if self.volume < 0:
            raise ValueError(f"Invalid volume: {self.volume}")
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'price': self.price,
            'volume': self.volume,
            'bid': self.bid,
            'ask': self.ask,
            'source': self.source,
            'sequence_number': self.sequence_number
        }


class TickValidator:
    """
    Validates and normalizes tick data.
    
    Implements sanity checks and outlier detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.symbol = config['symbol']
        
        # Validation parameters
        self.max_price_change = config.get('max_price_change', 0.1)  # 10%
        self.min_price = config.get('min_price', 0.0001)
        self.max_price = config.get('max_price', 1000000.0)
        self.max_volume = config.get('max_volume', 1000000)
        
        # State for outlier detection
        self.price_history = deque(maxlen=100)
        self.last_timestamp = None
        
        # Metrics
        self.validation_stats = {
            'total_ticks': 0,
            'valid_ticks': 0,
            'rejected_ticks': 0,
            'price_outliers': 0,
            'timestamp_errors': 0,
            'volume_errors': 0
        }
        
    def validate(self, tick: TickData) -> Tuple[bool, Optional[str]]:
        """
        Validate tick data.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        self.validation_stats['total_ticks'] += 1
        
        # Price range check
        if not self.min_price <= tick.price <= self.max_price:
            self.validation_stats['rejected_ticks'] += 1
            return False, f"Price {tick.price} out of range [{self.min_price}, {self.max_price}]"
            
        # Volume check
        if tick.volume > self.max_volume:
            self.validation_stats['volume_errors'] += 1
            self.validation_stats['rejected_ticks'] += 1
            return False, f"Volume {tick.volume} exceeds maximum {self.max_volume}"
            
        # Timestamp sequence check
        if self.last_timestamp and tick.timestamp <= self.last_timestamp:
            self.validation_stats['timestamp_errors'] += 1
            self.validation_stats['rejected_ticks'] += 1
            return False, f"Timestamp {tick.timestamp} not after {self.last_timestamp}"
            
        # Price spike detection
        if self.price_history:
            median_price = np.median(list(self.price_history))
            price_change = abs(tick.price - median_price) / median_price
            
            if price_change > self.max_price_change:
                self.validation_stats['price_outliers'] += 1
                self.validation_stats['rejected_ticks'] += 1
                return False, f"Price spike detected: {price_change:.2%} change"
                
        # Update state
        self.price_history.append(tick.price)
        self.last_timestamp = tick.timestamp
        self.validation_stats['valid_ticks'] += 1
        
        return True, None
        
    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total = self.validation_stats['total_ticks']
        if total == 0:
            return self.validation_stats
            
        return {
            **self.validation_stats,
            'valid_rate': self.validation_stats['valid_ticks'] / total,
            'rejection_rate': self.validation_stats['rejected_ticks'] / total
        }


class AbstractDataHandler(ABC):
    """
    Abstract base class for all data handlers.
    
    Provides common functionality and enforces interface.
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.symbol = config['symbol']
        self.is_running = False
        
        # Tick validation
        self.validator = TickValidator(config)
        
        # Performance metrics
        self.metrics = MetricsCollector(f"data_handler_{self.symbol}")
        
        # Rate limiting
        self.rate_limiter = self._create_rate_limiter()
        
        logger.info(f"Initialized {self.__class__.__name__} for {self.symbol}")
        
    def _create_rate_limiter(self) -> Optional['RateLimiter']:
        """Create rate limiter if configured."""
        if 'rate_limit' in self.config:
            return RateLimiter(
                max_rate=self.config['rate_limit']['ticks_per_second'],
                window=self.config['rate_limit'].get('window', 1.0)
            )
        return None
        
    async def start(self):
        """Start the data handler."""
        if self.is_running:
            logger.warning(f"{self.__class__.__name__} already running")
            return
            
        self.is_running = True
        logger.info(f"Starting {self.__class__.__name__}")
        
        try:
            await self._start_impl()
        except Exception as e:
            logger.error(f"Error in data handler: {e}")
            self.is_running = False
            raise
            
    async def stop(self):
        """Stop the data handler."""
        logger.info(f"Stopping {self.__class__.__name__}")
        self.is_running = False
        await self._stop_impl()
        
        # Log final statistics
        stats = self.validator.get_stats()
        logger.info(f"Data handler statistics: {stats}")
        
    @abstractmethod
    async def _start_impl(self):
        """Implementation-specific start logic."""
        pass
        
    @abstractmethod
    async def _stop_impl(self):
        """Implementation-specific stop logic."""
        pass
        
    async def _emit_tick(self, tick: TickData):
        """
        Validate and emit tick event.
        
        Handles validation, rate limiting, and metrics.
        """
        # Validate tick
        is_valid, error_msg = self.validator.validate(tick)
        if not is_valid:
            logger.warning(f"Invalid tick rejected: {error_msg}")
            self.metrics.increment('ticks_rejected')
            return
            
        # Rate limiting
        if self.rate_limiter and not await self.rate_limiter.allow():
            logger.debug("Tick dropped due to rate limit")
            self.metrics.increment('ticks_rate_limited')
            return
            
        # Emit event
        event = Event(
            type=EventType.NEW_TICK,
            data={'tick': tick},
            source=f"data_handler_{self.symbol}"
        )
        
        await self.event_bus.publish(event)
        
        # Update metrics
        self.metrics.increment('ticks_emitted')
        self.metrics.observe('tick_price', tick.price)
        self.metrics.observe('tick_volume', tick.volume)


class BacktestDataHandler(AbstractDataHandler):
    """
    Data handler for backtesting from CSV files.
    
    Features:
    - Temporal simulation with configurable speed
    - Support for compressed files
    - Progress tracking
    - Deterministic replay
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: EventBus):
        super().__init__(config, event_bus)
        
        # Backtest-specific configuration
        self.csv_path = Path(config['csv_path'])
        self.replay_speed = config.get('replay_speed', 0)  # 0 = as fast as possible
        self.start_date = config.get('start_date')
        self.end_date = config.get('end_date')
        
        # Progress tracking
        self.total_rows = 0
        self.processed_rows = 0
        self.start_time = None
        
        # CSV parsing configuration
        self.csv_config = {
            'timestamp_column': config.get('timestamp_column', 'timestamp'),
            'price_column': config.get('price_column', 'price'),
            'volume_column': config.get('volume_column', 'volume'),
            'delimiter': config.get('delimiter', ','),
            'compression': config.get('compression', 'infer')
        }
        
    async def _start_impl(self):
        """Start reading from CSV file."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
            
        self.start_time = datetime.now()
        logger.info(f"Starting backtest from {self.csv_path}")
        
        # Count total rows for progress tracking
        self.total_rows = await self._count_rows()
        logger.info(f"Total rows to process: {self.total_rows}")
        
        # Process file
        await self._process_csv()
        
    async def _stop_impl(self):
        """Clean up backtest resources."""
        if self.start_time:
            duration = datetime.now() - self.start_time
            rate = self.processed_rows / duration.total_seconds()
            logger.info(
                f"Backtest completed: {self.processed_rows} ticks in {duration}, "
                f"rate: {rate:.0f} ticks/sec"
            )
            
    async def _count_rows(self) -> int:
        """Count total rows in CSV for progress tracking."""
        if self.csv_config['compression'] == 'gzip':
            import gzip
            open_func = gzip.open
        else:
            open_func = open
            
        count = 0
        with open_func(self.csv_path, 'rt') as f:
            for _ in f:
                count += 1
                
        return count - 1  # Subtract header
        
    async def _process_csv(self):
        """Process CSV file and emit ticks."""
        # Use pandas for efficient CSV reading
        chunk_size = 10000
        
        # Date filtering
        date_parser = lambda x: pd.to_datetime(x)
        
        try:
            # Read CSV in chunks for memory efficiency
            for chunk in pd.read_csv(
                self.csv_path,
                chunksize=chunk_size,
                parse_dates=[self.csv_config['timestamp_column']],
                date_parser=date_parser,
                delimiter=self.csv_config['delimiter'],
                compression=self.csv_config['compression']
            ):
                # Apply date filtering
                if self.start_date:
                    chunk = chunk[chunk[self.csv_config['timestamp_column']] >= self.start_date]
                if self.end_date:
                    chunk = chunk[chunk[self.csv_config['timestamp_column']] <= self.end_date]
                    
                # Process each row
                for _, row in chunk.iterrows():
                    if not self.is_running:
                        break
                        
                    tick = self._parse_row(row)
                    if tick:
                        await self._emit_tick_with_timing(tick)
                        
                    self.processed_rows += 1
                    
                    # Progress update every 10000 ticks
                    if self.processed_rows % 10000 == 0:
                        progress = self.processed_rows / self.total_rows * 100
                        logger.info(f"Backtest progress: {progress:.1f}%")
                        
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            raise
            
    def _parse_row(self, row: pd.Series) -> Optional[TickData]:
        """Parse a CSV row into TickData."""
        try:
            tick = TickData(
                timestamp=row[self.csv_config['timestamp_column']].to_pydatetime(),
                symbol=self.symbol,
                price=float(row[self.csv_config['price_column']]),
                volume=int(row[self.csv_config['volume_column']]),
                source='backtest',
                sequence_number=self.processed_rows
            )
            
            # Optional bid/ask
            if 'bid' in row:
                tick.bid = float(row['bid'])
            if 'ask' in row:
                tick.ask = float(row['ask'])
                
            return tick
            
        except Exception as e:
            logger.warning(f"Error parsing row {self.processed_rows}: {e}")
            return None
            
    async def _emit_tick_with_timing(self, tick: TickData):
        """Emit tick with optional replay timing."""
        if self.replay_speed > 0:
            # Calculate sleep time for temporal simulation
            if hasattr(self, '_last_tick_time'):
                time_diff = (tick.timestamp - self._last_tick_time).total_seconds()
                sleep_time = time_diff / self.replay_speed
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
            self._last_tick_time = tick.timestamp
            
        await self._emit_tick(tick)


class LiveDataHandler(AbstractDataHandler):
    """
    Data handler for live trading via Rithmic API.
    
    Features:
    - WebSocket connection management
    - Automatic reconnection
    - Heartbeat monitoring
    - Market hours awareness
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: EventBus):
        super().__init__(config, event_bus)
        
        # Rithmic configuration
        self.api_config = {
            'host': config['rithmic']['host'],
            'port': config['rithmic']['port'],
            'username': config['rithmic']['username'],
            'password': config['rithmic']['password'],
            'exchange': config['rithmic']['exchange'],
            'symbol_code': config['rithmic']['symbol_code']
        }
        
        # Connection state
        self.websocket = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = config.get('max_reconnect_attempts', 10)
        self.reconnect_delay = config.get('reconnect_delay', 5.0)
        
        # Heartbeat monitoring
        self.last_heartbeat = None
        self.heartbeat_interval = 30.0
        self.heartbeat_task = None
        
        # Market hours
        self.market_hours = MarketHours(config.get('market_hours', {}))
        
    async def _start_impl(self):
        """Start live data connection."""
        await self._connect()
        
        # Start heartbeat monitoring
        self.heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
        
    async def _stop_impl(self):
        """Stop live data connection."""
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            
        await self._disconnect()
        
    async def _connect(self):
        """Establish connection to Rithmic."""
        while self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                logger.info(f"Connecting to Rithmic (attempt {self.reconnect_attempts + 1})")
                
                # Create WebSocket connection
                import websockets
                
                uri = f"wss://{self.api_config['host']}:{self.api_config['port']}/ws"
                self.websocket = await websockets.connect(uri)
                
                # Authenticate
                await self._authenticate()
                
                # Subscribe to market data
                await self._subscribe_market_data()
                
                self.is_connected = True
                self.reconnect_attempts = 0
                self.last_heartbeat = datetime.now()
                
                logger.info("Successfully connected to Rithmic")
                
                # Start receiving data
                await self._receive_loop()
                
                break
                
            except Exception as e:
                logger.error(f"Connection failed: {e}")
                self.reconnect_attempts += 1
                
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    await asyncio.sleep(self.reconnect_delay)
                else:
                    raise RuntimeError("Max reconnection attempts reached")
                    
    async def _disconnect(self):
        """Disconnect from Rithmic."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            self.is_connected = False
            
        logger.info("Disconnected from Rithmic")
        
    async def _authenticate(self):
        """Authenticate with Rithmic."""
        auth_msg = {
            'type': 'auth',
            'username': self.api_config['username'],
            'password': self.api_config['password']
        }
        
        await self.websocket.send(json.dumps(auth_msg))
        
        # Wait for auth response
        response = await self.websocket.recv()
        auth_response = json.loads(response)
        
        if auth_response.get('status') != 'authenticated':
            raise RuntimeError(f"Authentication failed: {auth_response}")
            
    async def _subscribe_market_data(self):
        """Subscribe to market data feed."""
        subscribe_msg = {
            'type': 'subscribe',
            'exchange': self.api_config['exchange'],
            'symbol': self.api_config['symbol_code'],
            'data_types': ['tick', 'quote']
        }
        
        await self.websocket.send(json.dumps(subscribe_msg))
        
        # Wait for subscription confirmation
        response = await self.websocket.recv()
        sub_response = json.loads(response)
        
        if sub_response.get('status') != 'subscribed':
            raise RuntimeError(f"Subscription failed: {sub_response}")
            
    async def _receive_loop(self):
        """Main loop for receiving market data."""
        while self.is_connected and self.is_running:
            try:
                # Check market hours
                if not self.market_hours.is_open():
                    await asyncio.sleep(60)  # Check every minute
                    continue
                    
                # Receive message with timeout
                message = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=self.heartbeat_interval
                )
                
                await self._process_message(message)
                
            except asyncio.TimeoutError:
                logger.warning("No data received within heartbeat interval")
                await self._check_connection()
                
            except websockets.ConnectionClosed:
                logger.error("WebSocket connection closed")
                self.is_connected = False
                
                if self.is_running:
                    await self._connect()  # Attempt reconnection
                    
            except Exception as e:
                logger.error(f"Error in receive loop: {e}")
                self.metrics.increment('receive_errors')
                
    async def _process_message(self, message: str):
        """Process incoming market data message."""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'tick':
                await self._process_tick(data)
            elif msg_type == 'quote':
                await self._process_quote(data)
            elif msg_type == 'heartbeat':
                self.last_heartbeat = datetime.now()
            else:
                logger.debug(f"Unhandled message type: {msg_type}")
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.metrics.increment('message_errors')
            
    async def _process_tick(self, data: Dict[str, Any]):
        """Process tick data from Rithmic."""
        tick = TickData(
            timestamp=datetime.fromisoformat(data['timestamp']),
            symbol=self.symbol,
            price=float(data['price']),
            volume=int(data['volume']),
            source='rithmic',
            sequence_number=data.get('sequence')
        )
        
        await self._emit_tick(tick)
        
    async def _process_quote(self, data: Dict[str, Any]):
        """Process quote data from Rithmic."""
        # Create synthetic tick from quote midpoint
        bid = float(data['bid'])
        ask = float(data['ask'])
        mid_price = (bid + ask) / 2
        
        tick = TickData(
            timestamp=datetime.fromisoformat(data['timestamp']),
            symbol=self.symbol,
            price=mid_price,
            volume=0,  # No volume in quotes
            bid=bid,
            ask=ask,
            source='rithmic_quote',
            sequence_number=data.get('sequence')
        )
        
        await self._emit_tick(tick)
        
    async def _heartbeat_monitor(self):
        """Monitor connection heartbeat."""
        while self.is_running:
            await asyncio.sleep(self.heartbeat_interval)
            
            if self.is_connected:
                time_since_heartbeat = (datetime.now() - self.last_heartbeat).total_seconds()
                
                if time_since_heartbeat > self.heartbeat_interval * 2:
                    logger.warning(f"Heartbeat timeout: {time_since_heartbeat}s")
                    await self._check_connection()
                    
    async def _check_connection(self):
        """Check and restore connection if needed."""
        try:
            # Send ping
            pong = await self.websocket.ping()
            await asyncio.wait_for(pong, timeout=5.0)
            
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            self.is_connected = False
            
            if self.is_running:
                await self._connect()


class RateLimiter:
    """
    Token bucket rate limiter for tick flow control.
    """
    
    def __init__(self, max_rate: float, window: float = 1.0):
        self.max_rate = max_rate
        self.window = window
        self.tokens = max_rate
        self.last_update = asyncio.get_event_loop().time()
        
    async def allow(self) -> bool:
        """Check if request is allowed under rate limit."""
        current_time = asyncio.get_event_loop().time()
        time_passed = current_time - self.last_update
        
        # Replenish tokens
        self.tokens = min(
            self.max_rate,
            self.tokens + time_passed * self.max_rate / self.window
        )
        
        self.last_update = current_time
        
        if self.tokens >= 1:
            self.tokens -= 1
            return True
            
        return False


class MarketHours:
    """
    Market hours checker for live trading.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.timezone = config.get('timezone', 'America/Chicago')
        
        # Default futures hours (Sunday 6PM - Friday 5PM CT)
        self.schedule = config.get('schedule', {
            'sunday': {'open': '18:00', 'close': '23:59'},
            'monday': {'open': '00:00', 'close': '23:59'},
            'tuesday': {'open': '00:00', 'close': '23:59'},
            'wednesday': {'open': '00:00', 'close': '23:59'},
            'thursday': {'open': '00:00', 'close': '23:59'},
            'friday': {'open': '00:00', 'close': '17:00'}
        })
        
    def is_open(self) -> bool:
        """Check if market is currently open."""
        import pytz
        
        tz = pytz.timezone(self.timezone)
        now = datetime.now(tz)
        
        day_name = now.strftime('%A').lower()
        
        if day_name not in self.schedule:
            return False
            
        day_schedule = self.schedule[day_name]
        current_time = now.strftime('%H:%M')
        
        return day_schedule['open'] <= current_time <= day_schedule['close']