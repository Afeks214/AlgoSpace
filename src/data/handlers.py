"""Data handler implementations for AlgoSpace trading system.

This module provides abstract and concrete implementations for data ingestion
from various sources including CSV files (backtest) and live data feeds.
"""

import csv
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class TickData:
    """Standardized tick data structure.
    
    Attributes:
        timestamp: The datetime when the tick occurred
        price: The tick price as a float
        volume: The tick volume as an integer
    """
    timestamp: datetime
    price: float
    volume: int


class AbstractDataHandler(ABC):
    """Abstract base class for data handlers.
    
    All data handlers must implement the start_stream method to begin
    data ingestion and event emission.
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: Any) -> None:
        """Initialize the data handler.
        
        Args:
            config: Configuration dictionary containing data source settings
            event_bus: Event bus instance for publishing data events
        """
        self.config = config
        self.event_bus = event_bus
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def start_stream(self) -> None:
        """Start the data streaming process.
        
        This method should implement the main data ingestion loop and
        emit events through the event bus.
        """
        pass


class BacktestDataHandler(AbstractDataHandler):
    """Data handler for backtesting using CSV files.
    
    Reads historical data from CSV files and emits tick events to simulate
    real-time data flow during backtesting.
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: Any) -> None:
        """Initialize the backtest data handler.
        
        Args:
            config: Configuration dictionary containing backtest file path
            event_bus: Event bus instance for publishing data events
            
        Raises:
            FileNotFoundError: If the specified backtest file doesn't exist
        """
        super().__init__(config, event_bus)
        
        self.file_path = config['data']['backtest_file']
        self.file_handle: Optional[Any] = None
        self.csv_reader: Optional[Any] = None
        self.row_count = 0
        self.error_count = 0
        
        # Open CSV file with error handling
        try:
            self.file_handle = open(self.file_path, 'r')
            self.csv_reader = csv.reader(self.file_handle)
            # Skip header if present
            next(self.csv_reader, None)
            logger.info(f"Opened backtest file: {self.file_path}")
        except FileNotFoundError:
            logger.error(f"Backtest file not found: {self.file_path}")
            raise FileNotFoundError(f"Backtest file not found: {self.file_path}")
        except Exception as e:
            logger.error(f"Error opening backtest file: {e}")
            raise
    
    def start_stream(self) -> None:
        """Read CSV data line by line and emit tick events.
        
        Processes each row of the CSV file, creates TickData objects,
        and publishes them as NEW_TICK events. Handles malformed rows
        gracefully by logging warnings and continuing.
        """
        logger.info("Starting backtest data stream")
        
        try:
            for row in self.csv_reader:
                self.row_count += 1
                
                # Handle malformed rows
                try:
                    if len(row) < 3:
                        logger.warning(f"Row {self.row_count}: Insufficient columns, skipping")
                        self.error_count += 1
                        continue
                    
                    # Parse CSV columns: timestamp, price, volume
                    timestamp_str = row[0].strip()
                    price_str = row[1].strip()
                    volume_str = row[2].strip()
                    
                    # Convert to appropriate types
                    timestamp = datetime.fromisoformat(timestamp_str)
                    price = float(price_str)
                    volume = int(volume_str)
                    
                    # Create tick data object
                    tick_data = TickData(
                        timestamp=timestamp,
                        price=price,
                        volume=volume
                    )
                    
                    # Publish tick event
                    self.event_bus.publish('NEW_TICK', tick_data)
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"Row {self.row_count}: Error parsing data - {e}, skipping")
                    self.error_count += 1
                    continue
                except Exception as e:
                    logger.warning(f"Row {self.row_count}: Unexpected error - {e}, skipping")
                    self.error_count += 1
                    continue
            
            # Log summary statistics
            logger.info(f"Backtest data stream completed. Total rows: {self.row_count}, "
                       f"Errors: {self.error_count}, "
                       f"Successfully processed: {self.row_count - self.error_count}")
            
            # Publish completion event
            self.event_bus.publish('BACKTEST_COMPLETE', {
                'total_rows': self.row_count,
                'error_count': self.error_count,
                'success_count': self.row_count - self.error_count
            })
            
        finally:
            # Clean up file handle
            if self.file_handle:
                self.file_handle.close()
                logger.info("Closed backtest file")


class LiveDataHandler(AbstractDataHandler):
    """Data handler for live trading using Rithmic API.
    
    Placeholder implementation for live data streaming. In production,
    this would connect to the Rithmic API and stream real-time market data.
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: Any) -> None:
        """Initialize the live data handler.
        
        Loads Rithmic API credentials from environment variables.
        
        Args:
            config: Configuration dictionary containing live data settings
            event_bus: Event bus instance for publishing data events
        """
        super().__init__(config, event_bus)
        
        # Load Rithmic credentials from environment variables
        self.rithmic_username = os.environ.get('RITHMIC_USERNAME', '')
        self.rithmic_password = os.environ.get('RITHMIC_PASSWORD', '')
        self.rithmic_api_key = os.environ.get('RITHMIC_API_KEY', '')
        
        if not all([self.rithmic_username, self.rithmic_password, self.rithmic_api_key]):
            logger.warning("Rithmic credentials not fully configured in environment variables")
        else:
            logger.info("Rithmic credentials loaded from environment variables")
    
    def start_stream(self) -> None:
        """Start live data streaming from Rithmic API.
        
        This is a placeholder implementation that simulates a connection
        to the Rithmic API. In production, this would establish a real
        connection and stream live market data.
        """
        logger.info("Connecting to Rithmic API...")
        logger.info("Live data handler running in simulation mode")
        
        # Placeholder infinite loop to simulate running process
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Live data stream interrupted by user")
        except Exception as e:
            logger.error(f"Error in live data stream: {e}")
            raise


def create_data_handler(kernel):
    """Factory function to create appropriate data handler based on configuration.
    
    Args:
        kernel: AlgoSpaceKernel instance containing configuration
        
    Returns:
        DataHandler instance (BacktestDataHandler or LiveDataHandler)
    """
    # Get data configuration from kernel config dict
    data_config = kernel.config.get('data_handler', {})
    handler_type = data_config.get('type', 'backtest')
    
    # Create event bus reference
    event_bus = kernel.event_bus
    
    if handler_type == 'backtest':
        return BacktestDataHandler(data_config, event_bus)
    elif handler_type in ['rithmic', 'ib']:
        return LiveDataHandler(data_config, event_bus)
    else:
        raise ValueError(f"Unknown data handler type: {handler_type}")