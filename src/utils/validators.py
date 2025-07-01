"""Validation utilities for AlgoSpace trading system.

This module provides static validation methods for configuration
and data integrity checks throughout the system.
"""

import os
from datetime import datetime
from typing import Any, Dict


class ConfigValidator:
    """Validates system configuration dictionaries.
    
    Provides static methods to validate various configuration
    aspects and ensure required keys and values are present.
    """
    
    @staticmethod
    def validate_main_config(config: Dict[str, Any]) -> None:
        """Validate the main system configuration.
        
        Checks for required top-level keys and validates specific
        configuration values based on the system mode.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValueError: If required keys are missing or values are invalid
        """
        # Check for required top-level keys
        required_keys = ['system', 'data', 'models', 'indicators']
        missing_keys = []
        
        for key in required_keys:
            if key not in config:
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
        
        # Validate system section
        if 'system' not in config:
            raise ValueError("Missing 'system' section in configuration")
        
        system_config = config['system']
        
        # Check for mode
        if 'mode' not in system_config:
            raise ValueError("Missing 'mode' in system configuration")
        
        # Validate mode value
        valid_modes = ['live', 'backtest']
        mode = system_config['mode']
        
        if mode not in valid_modes:
            raise ValueError(f"Invalid system mode '{mode}'. Must be one of: {valid_modes}")
        
        # Additional validation for backtest mode
        if mode == 'backtest':
            # Check data section exists
            if 'data' not in config:
                raise ValueError("Missing 'data' section in configuration for backtest mode")
            
            data_config = config['data']
            
            # Check for backtest_file
            if 'backtest_file' not in data_config:
                raise ValueError("Missing 'backtest_file' in data configuration for backtest mode")
            
            # Verify file exists
            backtest_file = data_config['backtest_file']
            if not os.path.exists(backtest_file):
                raise ValueError(f"Backtest file not found: {backtest_file}")
    
    @staticmethod
    def validate_data_config(config: Dict[str, Any]) -> None:
        """Validate data-specific configuration.
        
        Args:
            config: Data configuration dictionary
            
        Raises:
            ValueError: If data configuration is invalid
        """
        if not isinstance(config, dict):
            raise ValueError("Data configuration must be a dictionary")
        
        # For backtest mode, ensure file path is provided
        if 'backtest_file' in config:
            file_path = config['backtest_file']
            if not isinstance(file_path, str) or not file_path:
                raise ValueError("Backtest file path must be a non-empty string")
    
    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> None:
        """Validate model configuration.
        
        Args:
            config: Model configuration dictionary
            
        Raises:
            ValueError: If model configuration is invalid
        """
        if not isinstance(config, dict):
            raise ValueError("Model configuration must be a dictionary")
        
        # Add specific model validation as needed
        pass


class DataValidator:
    """Validates data structures used throughout the system.
    
    Provides static methods to validate tick data, bar data,
    and other data structures for integrity and type correctness.
    """
    
    @staticmethod
    def validate_tick_data(tick: Dict[str, Any]) -> None:
        """Validate tick data structure.
        
        Ensures tick data contains required fields with correct types.
        
        Args:
            tick: Tick data dictionary to validate
            
        Raises:
            TypeError: If field types are incorrect
            ValueError: If required fields are missing
        """
        # Check if tick is a dictionary (or has dict-like interface)
        if not hasattr(tick, '__getitem__'):
            # Try to convert dataclass to dict if needed
            if hasattr(tick, '__dict__'):
                tick = tick.__dict__
            else:
                raise TypeError("Tick data must be a dictionary or have dict-like interface")
        
        # Check for required keys
        required_keys = ['timestamp', 'price', 'volume']
        missing_keys = []
        
        for key in required_keys:
            if key not in tick and not hasattr(tick, key):
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing required tick data fields: {missing_keys}")
        
        # Validate timestamp
        timestamp = tick.get('timestamp') if hasattr(tick, 'get') else getattr(tick, 'timestamp')
        if not isinstance(timestamp, datetime):
            raise TypeError(f"Timestamp must be datetime object, got {type(timestamp).__name__}")
        
        # Validate price
        price = tick.get('price') if hasattr(tick, 'get') else getattr(tick, 'price')
        if not isinstance(price, (int, float)):
            raise TypeError(f"Price must be numeric (int or float), got {type(price).__name__}")
        
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")
        
        # Validate volume
        volume = tick.get('volume') if hasattr(tick, 'get') else getattr(tick, 'volume')
        if not isinstance(volume, int):
            raise TypeError(f"Volume must be integer, got {type(volume).__name__}")
        
        if volume < 0:
            raise ValueError(f"Volume must be non-negative, got {volume}")
    
    @staticmethod
    def validate_bar_data(bar: Dict[str, Any]) -> None:
        """Validate bar data structure.
        
        Ensures bar data contains required OHLCV fields with correct types.
        
        Args:
            bar: Bar data dictionary to validate
            
        Raises:
            TypeError: If field types are incorrect
            ValueError: If required fields are missing or values invalid
        """
        # Check for required keys
        required_keys = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_keys = []
        
        for key in required_keys:
            if key not in bar and not hasattr(bar, key):
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing required bar data fields: {missing_keys}")
        
        # Extract values (handle both dict and dataclass)
        if hasattr(bar, '__getitem__'):
            timestamp = bar['timestamp']
            open_price = bar['open']
            high_price = bar['high']
            low_price = bar['low']
            close_price = bar['close']
            volume = bar['volume']
        else:
            timestamp = bar.timestamp
            open_price = bar.open
            high_price = bar.high
            low_price = bar.low
            close_price = bar.close
            volume = bar.volume
        
        # Validate timestamp
        if not isinstance(timestamp, datetime):
            raise TypeError(f"Timestamp must be datetime object, got {type(timestamp).__name__}")
        
        # Validate OHLC prices
        for price, name in [(open_price, 'open'), (high_price, 'high'), 
                           (low_price, 'low'), (close_price, 'close')]:
            if not isinstance(price, (int, float)):
                raise TypeError(f"{name} price must be numeric, got {type(price).__name__}")
            if price <= 0:
                raise ValueError(f"{name} price must be positive, got {price}")
        
        # Validate OHLC relationships
        if high_price < max(open_price, close_price):
            raise ValueError("High price must be >= max(open, close)")
        
        if low_price > min(open_price, close_price):
            raise ValueError("Low price must be <= min(open, close)")
        
        if high_price < low_price:
            raise ValueError("High price must be >= low price")
        
        # Validate volume
        if not isinstance(volume, int):
            raise TypeError(f"Volume must be integer, got {type(volume).__name__}")
        
        if volume < 0:
            raise ValueError(f"Volume must be non-negative, got {volume}")
    
    @staticmethod
    def validate_event_data(event_type: str, payload: Any) -> None:
        """Validate event data based on event type.
        
        Args:
            event_type: Type of event
            payload: Event payload to validate
            
        Raises:
            ValueError: If event data is invalid
        """
        if not event_type:
            raise ValueError("Event type cannot be empty")
        
        if payload is None:
            raise ValueError("Event payload cannot be None")
        
        # Add specific validation based on event type
        if event_type == 'NEW_TICK':
            DataValidator.validate_tick_data(payload)
        elif event_type in ['NEW_5MIN_BAR', 'NEW_30MIN_BAR']:
            DataValidator.validate_bar_data(payload)