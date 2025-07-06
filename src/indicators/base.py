from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np
import pandas as pd
import structlog

from src.core.events import EventBus, Event, EventType, BarData

logger = structlog.get_logger()


class BaseIndicator(ABC):
    """
    Abstract base class for all technical indicators in the AlgoSpace system.
    
    All indicators must implement this interface to ensure consistent
    behavior and integration with the IndicatorEngine.
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: EventBus):
        """
        Initialize base indicator
        
        Args:
            config: Indicator configuration parameters
            event_bus: System event bus for communication
        """
        self.config = config
        self.event_bus = event_bus
        self.logger = structlog.get_logger(self.__class__.__name__)
        
        # State tracking
        self.is_initialized = False
        self.last_update = None
        self.last_values = {}
        
        # History buffers - to be managed by subclasses
        self.history_5m: List[BarData] = []
        self.history_30m: List[BarData] = []
        self.max_history_length = config.get('max_history_length', 1000)
        
        self.logger.info(f"Indicator initialized config={config}")
    
    def calculate_5m(self, bar: BarData) -> Dict[str, Any]:
        """
        Calculate indicator values for 5-minute bar - override if supported
        
        Args:
            bar: New 5-minute bar data
            
        Returns:
            Dictionary of calculated features (empty by default)
        """
        return {}
    
    def calculate_30m(self, bar: BarData) -> Dict[str, Any]:
        """
        Calculate indicator values for 30-minute bar - override if supported
        
        Args:
            bar: New 30-minute bar data
            
        Returns:
            Dictionary of calculated features (empty by default)
        """
        return {}
    
    @abstractmethod
    def get_current_values(self) -> Dict[str, Any]:
        """
        Get current indicator values
        
        Returns:
            Dictionary of current indicator features
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset indicator state"""
        pass
    
    def update_5m_history(self, bar: BarData) -> None:
        """Update 5-minute history buffer"""
        self.history_5m.append(bar)
        if len(self.history_5m) > self.max_history_length:
            self.history_5m.pop(0)
    
    def update_30m_history(self, bar: BarData) -> None:
        """Update 30-minute history buffer"""
        self.history_30m.append(bar)
        if len(self.history_30m) > self.max_history_length:
            self.history_30m.pop(0)
    
    def has_sufficient_data_5m(self, min_bars: int = 1) -> bool:
        """Check if we have sufficient 5-minute data"""
        return len(self.history_5m) >= min_bars
    
    def has_sufficient_data_30m(self, min_bars: int = 1) -> bool:
        """Check if we have sufficient 30-minute data"""
        return len(self.history_30m) >= min_bars
    
    def get_5m_prices(self, field: str = 'close', count: Optional[int] = None) -> np.ndarray:
        """
        Get price array from 5-minute history
        
        Args:
            field: Price field to extract ('open', 'high', 'low', 'close', 'volume')
            count: Number of bars to return (None for all)
            
        Returns:
            Numpy array of prices
        """
        if not self.history_5m:
            return np.array([])
        
        bars = self.history_5m if count is None else self.history_5m[-count:]
        return np.array([getattr(bar, field) for bar in bars])
    
    def get_30m_prices(self, field: str = 'close', count: Optional[int] = None) -> np.ndarray:
        """
        Get price array from 30-minute history
        
        Args:
            field: Price field to extract ('open', 'high', 'low', 'close', 'volume')
            count: Number of bars to return (None for all)
            
        Returns:
            Numpy array of prices
        """
        if not self.history_30m:
            return np.array([])
        
        bars = self.history_30m if count is None else self.history_30m[-count:]
        return np.array([getattr(bar, field) for bar in bars])
    
    def convert_to_heiken_ashi(self, bars: List[BarData]) -> List[Dict[str, float]]:
        """
        Convert standard OHLCV bars to Heiken Ashi
        
        Args:
            bars: List of standard OHLCV bars
            
        Returns:
            List of Heiken Ashi OHLC dictionaries
        """
        if not bars:
            return []
        
        ha_bars = []
        ha_open = (bars[0].open + bars[0].close) / 2  # First bar initialization
        
        for i, bar in enumerate(bars):
            # HA Close = (O + H + L + C) / 4
            ha_close = (bar.open + bar.high + bar.low + bar.close) / 4
            
            # HA High = max(High, HA_Open, HA_Close)
            ha_high = max(bar.high, ha_open, ha_close)
            
            # HA Low = min(Low, HA_Open, HA_Close)
            ha_low = min(bar.low, ha_open, ha_close)
            
            ha_bars.append({
                'open': ha_open,
                'high': ha_high,
                'low': ha_low,
                'close': ha_close,
                'volume': bar.volume,
                'timestamp': bar.timestamp
            })
            
            # HA Open for next bar = (HA_Open + HA_Close) / 2
            ha_open = (ha_open + ha_close) / 2
        
        return ha_bars
    
    def log_calculation(self, timeframe: str, values: Dict[str, Any]) -> None:
        """Log calculation completion"""
        self.logger.debug(f"Calculation completed timeframe={timeframe} values_count={len(values)}")
    
    def validate_config(self, required_keys: List[str]) -> None:
        """
        Validate that required configuration keys are present
        
        Args:
            required_keys: List of required configuration keys
            
        Raises:
            KeyError: If required key is missing
        """
        for key in required_keys:
            if key not in self.config:
                raise KeyError(f"Required configuration key '{key}' not found for {self.__class__.__name__}")
    
    def safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """
        Safe division with default value for zero denominator
        
        Args:
            numerator: Numerator value
            denominator: Denominator value  
            default: Default value if denominator is zero
            
        Returns:
            Division result or default value
        """
        if abs(denominator) < 1e-10:  # Avoid division by zero
            return default
        return numerator / denominator
    
    def __str__(self) -> str:
        """String representation of indicator"""
        return f"{self.__class__.__name__}(initialized={self.is_initialized})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"{self.__class__.__name__}("
                f"initialized={self.is_initialized}, "
                f"5m_bars={len(self.history_5m)}, "
                f"30m_bars={len(self.history_30m)}")


class IndicatorRegistry:
    """Registry for managing indicator instances"""
    
    def __init__(self):
        self._indicators: Dict[str, BaseIndicator] = {}
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def register(self, name: str, indicator: BaseIndicator) -> None:
        """Register an indicator instance"""
        self._indicators[name] = indicator
        self.logger.info(f"Indicator registered name={name} type={type(indicator).__name__}")
    
    def get(self, name: str) -> Optional[BaseIndicator]:
        """Get indicator by name"""
        return self._indicators.get(name)
    
    def get_all(self) -> Dict[str, BaseIndicator]:
        """Get all registered indicators"""
        return self._indicators.copy()
    
    def remove(self, name: str) -> bool:
        """Remove indicator from registry"""
        if name in self._indicators:
            del self._indicators[name]
            self.logger.info(f"Indicator removed name={name}")
            return True
        return False
    
    def reset_all(self) -> None:
        """Reset all registered indicators"""
        for indicator in self._indicators.values():
            indicator.reset()
        self.logger.info("All indicators reset")
    
    def list_names(self) -> List[str]:
        """List all registered indicator names"""
        return list(self._indicators.keys())
    
    def __len__(self) -> int:
        """Number of registered indicators"""
        return len(self._indicators)