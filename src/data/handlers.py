"""
DataHandler Component - Market Data Abstraction Layer

This module implements the exclusive entry point for all market data into the system.
It provides complete abstraction between data sources (live feed or historical file)
and ensures identical data structures regardless of backtest or live mode.

Based on Master PRD - DataHandler Component v1.0
"""

import asyncio
import csv
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, AsyncGenerator
import pandas as pd
import structlog

from ..core.kernel import ComponentBase, SystemKernel
from ..core.events import EventType, TickData
from ..utils.logger import get_logger
from .validators import TickValidator, DataQualityMonitor


class AbstractDataHandler(ComponentBase, ABC):
    """
    Abstract base class for all data handlers
    
    Ensures consistent interface regardless of data source (live/backtest).
    All subclasses must emit identical NEW_TICK events.
    """
    
    def __init__(self, name: str, kernel: SystemKernel):
        """
        Initialize data handler
        
        Args:
            name: Component name
            kernel: System kernel instance
        """
        super().__init__(name, kernel)
        self.symbol = self.config.primary_symbol
        self.is_connected = False
        self.tick_count = 0
        
        # Initialize validators
        self.tick_validator = TickValidator()
        self.data_quality_monitor = DataQualityMonitor()
        
        self.logger.info("DataHandler initialized", 
                        symbol=self.symbol,
                        mode=self.config.system_mode)
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to data source"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from data source"""
        pass
    
    @abstractmethod
    async def start_data_stream(self) -> None:
        """Start streaming data"""
        pass
    
    @abstractmethod
    async def stop_data_stream(self) -> None:
        """Stop streaming data"""
        pass
    
    def _emit_tick(self, tick_data: TickData) -> None:
        """
        Emit NEW_TICK event - standardized across all handlers
        
        Args:
            tick_data: Standardized tick data structure
        """
        # Validate tick before emitting
        is_valid, validation_errors = self.tick_validator.validate_tick(tick_data)
        
        if not is_valid:
            self.logger.warning("Invalid tick data detected", 
                              tick=tick_data,
                              errors=validation_errors)
            # Update quality monitor
            self.data_quality_monitor.add_tick(tick_data, is_valid=False)
            # Skip emitting invalid ticks
            return
        
        # Update quality monitor with valid tick
        self.data_quality_monitor.add_tick(tick_data, is_valid=True)
        
        # Emit valid tick
        self.publish_event(EventType.NEW_TICK, tick_data)
        self.tick_count += 1
        
        # Log every 1000 ticks as per PRD
        if self.tick_count % 1000 == 0:
            quality_report = self.data_quality_monitor.get_report()
            self.logger.info(f"{self.tick_count} ticks processed",
                           quality_score=quality_report.get('overall_quality_score', 0),
                           total_ticks=quality_report.get('total_ticks', 0),
                           invalid_ticks=quality_report.get('invalid_ticks', 0))
    
    async def start(self) -> None:
        """Start the data handler component"""
        await super().start()
        
        try:
            await self.connect()
            await self.start_data_stream()
            self.logger.info("DataHandler started successfully")
            
        except Exception as e:
            self.logger.error("Failed to start DataHandler", error=str(e))
            raise
    
    async def stop(self) -> None:
        """Stop the data handler component"""
        try:
            await self.stop_data_stream()
            await self.disconnect()
            
            # Log final data quality report
            quality_report = self.data_quality_monitor.get_report()
            self.logger.info("DataHandler shutdown complete", 
                           total_ticks=self.tick_count,
                           quality_score=quality_report.get('overall_quality_score', 0),
                           invalid_ticks=quality_report.get('invalid_ticks', 0),
                           gap_count=quality_report.get('gap_count', 0),
                           spike_count=quality_report.get('spike_count', 0))
            
        except Exception as e:
            self.logger.error("Error during DataHandler shutdown", error=str(e))
        
        await super().stop()


class BacktestDataHandler(AbstractDataHandler):
    """
    Backtest data handler for CSV files
    
    Processes historical OHLCV bar data and converts to simulated tick events.
    Maintains temporal accuracy and supports replay speed configuration.
    """
    
    def __init__(self, name: str, kernel: SystemKernel):
        """Initialize backtest data handler"""
        super().__init__(name, kernel)
        
        # Get backtest configuration
        self.data_config = self.config.get_section('data_handler')
        self.backtest_config = self.config.get_section('backtesting')
        
        # File path configuration
        self.file_path = self._resolve_file_path()
        self.replay_speed = self.data_config.get('replay_speed', 1.0)
        
        # Data processing state
        self.data_frame: Optional[pd.DataFrame] = None
        self.current_row_index = 0
        self.is_streaming = False
        self.stream_task: Optional[asyncio.Task] = None
        
        self.logger.info("BacktestDataHandler configured",
                        file_path=self.file_path,
                        replay_speed=self.replay_speed)
    
    def _resolve_file_path(self) -> str:
        """Resolve the data file path"""
        # Try multiple possible file paths
        possible_paths = [
            self.data_config.get('backtest_file', ''),
            'data/historical/ES - 5 min.csv',  # Default for 5min data
            'data/historical/ES - 30 min - New.csv'  # Default for 30min data
        ]
        
        for path_str in possible_paths:
            if path_str:
                path = Path(path_str)
                if not path.is_absolute():
                    path = Path.cwd() / path
                
                if path.exists():
                    return str(path)
        
        # Default fallback
        default_path = Path.cwd() / 'data' / 'historical' / 'ES - 5 min.csv'
        return str(default_path)
    
    async def connect(self) -> None:
        """Load and validate CSV file"""
        try:
            file_path = Path(self.file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.file_path}")
            
            self.logger.info("Loading backtest data", file_path=self.file_path)
            
            # Load CSV with proper parsing
            self.data_frame = pd.read_csv(
                self.file_path,
                parse_dates=['Timestamp'],
                date_format='mixed'
            )
            
            # Clean column names (remove extra spaces)
            self.data_frame.columns = self.data_frame.columns.str.strip()
            
            # Validate required columns
            required_cols = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in self.data_frame.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Sort by timestamp to ensure chronological order
            self.data_frame = self.data_frame.sort_values('Timestamp').reset_index(drop=True)
            
            # Validate data integrity
            if len(self.data_frame) == 0:
                raise ValueError("Empty data file")
            
            row_count = len(self.data_frame)
            start_date = self.data_frame['Timestamp'].iloc[0]
            end_date = self.data_frame['Timestamp'].iloc[-1]
            
            self.is_connected = True
            
            self.logger.info("Backtest data loaded successfully",
                           rows=row_count,
                           start_date=start_date.isoformat(),
                           end_date=end_date.isoformat(),
                           columns=list(self.data_frame.columns))
            
        except Exception as e:
            self.logger.error("Failed to load backtest data", 
                            file_path=self.file_path,
                            error=str(e))
            raise
    
    async def disconnect(self) -> None:
        """Clean up resources"""
        self.data_frame = None
        self.is_connected = False
        self.logger.debug("Disconnected from backtest data")
    
    async def start_data_stream(self) -> None:
        """Start streaming simulated tick data"""
        if not self.is_connected or self.data_frame is None:
            raise RuntimeError("Must connect to data source before streaming")
        
        self.is_streaming = True
        self.current_row_index = 0
        
        # Start the streaming task
        self.stream_task = asyncio.create_task(self._stream_data())
        
        self.logger.info("Data stream started", 
                        total_bars=len(self.data_frame),
                        replay_speed=self.replay_speed)
    
    async def stop_data_stream(self) -> None:
        """Stop data streaming"""
        self.is_streaming = False
        
        if self.stream_task and not self.stream_task.done():
            self.stream_task.cancel()
            try:
                await self.stream_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Data stream stopped")
    
    async def _stream_data(self) -> None:
        """
        Stream data with temporal accuracy
        
        Converts OHLCV bars to simulated tick events:
        - Open price at bar start
        - High price at 25% of bar duration  
        - Low price at 50% of bar duration
        - Close price at bar end
        """
        try:
            prev_timestamp = None
            
            while self.is_streaming and self.current_row_index < len(self.data_frame):
                row = self.data_frame.iloc[self.current_row_index]
                
                # Extract bar data
                timestamp = row['Timestamp']
                open_price = float(row['Open'])
                high_price = float(row['High'])
                low_price = float(row['Low'])
                close_price = float(row['Close'])
                volume = int(row['Volume'])
                
                # Calculate inter-tick delay for replay speed
                if prev_timestamp is not None and self.replay_speed > 0:
                    time_diff = (timestamp - prev_timestamp).total_seconds()
                    delay = time_diff / self.replay_speed
                    
                    if delay > 0:
                        await asyncio.sleep(min(delay, 10.0))  # Cap at 10 seconds
                
                # Generate simulated ticks from OHLCV bar
                tick_volume = max(1, volume // 4)  # Distribute volume across ticks
                
                # Tick 1: Open price (bar start)
                tick_data = TickData(
                    symbol=self.symbol,
                    timestamp=timestamp,
                    price=open_price,
                    volume=tick_volume
                )
                self._emit_tick(tick_data)
                await asyncio.sleep(0.001)  # Small delay between ticks
                
                # Tick 2: High price (25% through bar)
                if high_price != open_price:
                    tick_data = TickData(
                        symbol=self.symbol,
                        timestamp=timestamp + timedelta(milliseconds=250),
                        price=high_price,
                        volume=tick_volume
                    )
                    self._emit_tick(tick_data)
                    await asyncio.sleep(0.001)
                
                # Tick 3: Low price (50% through bar)
                if low_price != high_price and low_price != open_price:
                    tick_data = TickData(
                        symbol=self.symbol,
                        timestamp=timestamp + timedelta(milliseconds=500),
                        price=low_price,
                        volume=tick_volume
                    )
                    self._emit_tick(tick_data)
                    await asyncio.sleep(0.001)
                
                # Tick 4: Close price (bar end)
                if close_price != low_price and close_price != high_price and close_price != open_price:
                    tick_data = TickData(
                        symbol=self.symbol,
                        timestamp=timestamp + timedelta(milliseconds=750),
                        price=close_price,
                        volume=volume - (3 * tick_volume)  # Remaining volume
                    )
                    self._emit_tick(tick_data)
                
                prev_timestamp = timestamp
                self.current_row_index += 1
                
                # Yield control to other tasks
                await asyncio.sleep(0)
            
            # End of data reached
            if self.current_row_index >= len(self.data_frame):
                self.logger.info("Backtest data stream complete", 
                               total_bars_processed=self.current_row_index,
                               total_ticks_emitted=self.tick_count)
                
                # Emit backtest complete event
                self.publish_event(EventType.BACKTEST_COMPLETE, {
                    'total_bars': self.current_row_index,
                    'total_ticks': self.tick_count,
                    'symbol': self.symbol
                })
                
                self.is_streaming = False
        
        except asyncio.CancelledError:
            self.logger.info("Data streaming cancelled")
            raise
        except Exception as e:
            self.logger.error("Error in data streaming", error=str(e))
            self.publish_event(EventType.SYSTEM_ERROR, {
                'component': self.name,
                'error': str(e),
                'context': 'data_streaming'
            })
            raise


class LiveDataHandler(AbstractDataHandler):
    """
    Live data handler for Rithmic API
    
    Note: This is a placeholder implementation for future live trading.
    Full Rithmic integration will be implemented in later phases.
    """
    
    def __init__(self, name: str, kernel: SystemKernel):
        """Initialize live data handler"""
        super().__init__(name, kernel)
        self.logger.warning("LiveDataHandler is placeholder - not yet implemented")
    
    async def connect(self) -> None:
        """Connect to Rithmic API"""
        raise NotImplementedError("Live data handler not yet implemented")
    
    async def disconnect(self) -> None:
        """Disconnect from Rithmic API"""
        raise NotImplementedError("Live data handler not yet implemented")
    
    async def start_data_stream(self) -> None:
        """Start live data stream"""
        raise NotImplementedError("Live data handler not yet implemented")
    
    async def stop_data_stream(self) -> None:
        """Stop live data stream"""
        raise NotImplementedError("Live data handler not yet implemented")


def create_data_handler(kernel: SystemKernel) -> AbstractDataHandler:
    """
    Factory function to create appropriate data handler based on configuration
    
    Args:
        kernel: System kernel instance
        
    Returns:
        Configured data handler instance
    """
    config = kernel.get_config()
    mode = config.system_mode
    
    if mode == 'backtest':
        return BacktestDataHandler('DataHandler', kernel)
    elif mode == 'live':
        return LiveDataHandler('DataHandler', kernel)
    else:
        raise ValueError(f"Unsupported data handler mode: {mode}")