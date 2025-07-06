"""
Data Validators - Data Quality Checks

This module provides comprehensive data validation for the AlgoSpace trading system.
It ensures data integrity, detects anomalies, and validates trading data quality
throughout the pipeline.

Key Features:
- Tick data validation
- Bar data validation
- Price consistency checks
- Volume validation
- Gap detection
- Anomaly detection
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import math
from collections import deque
import structlog

from ..core.events import TickData, BarData
from ..utils.logger import get_logger


class ValidationResult:
    """Container for validation results"""
    
    def __init__(self):
        self.is_valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.metrics: Dict[str, Any] = {}
    
    def add_error(self, error: str) -> None:
        """Add validation error"""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add validation warning"""
        self.warnings.append(warning)
    
    def add_metric(self, key: str, value: Any) -> None:
        """Add validation metric"""
        self.metrics[key] = value
    
    def __str__(self) -> str:
        """String representation"""
        status = "VALID" if self.is_valid else "INVALID"
        return f"ValidationResult({status}, errors={len(self.errors)}, warnings={len(self.warnings))}"


class BaseValidator(ABC):
    """Abstract base class for all validators"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize validator
        
        Args:
            config: Validator configuration
        """
        self.config = config or {}
        self.logger = get_logger(self.__class__.__name__)
        
        # Statistics
        self.total_validated = 0
        self.total_errors = 0
        self.total_warnings = 0
    
    @abstractmethod
    def validate(self, data: Any) -> ValidationResult:
        """
        Validate data
        
        Args:
            data: Data to validate
            
        Returns:
            ValidationResult with errors, warnings, and metrics
        """
        pass
    
    def reset_statistics(self) -> None:
        """Reset validation statistics"""
        self.total_validated = 0
        self.total_errors = 0
        self.total_warnings = 0


class TickValidator(BaseValidator):
    """
    Validates tick data for quality and consistency
    
    Checks:
    - Price validity (positive, reasonable range)
    - Volume validity (non-negative)
    - Timestamp consistency
    - Price spike detection
    - Stale data detection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize tick validator with default thresholds"""
        default_config = {
            'min_price': 1.0,
            'max_price': 100000.0,
            'max_price_change_percent': 5.0,  # 5% max change between ticks
            'max_volume': 1000000,
            'stale_threshold_seconds': 300,  # 5 minutes
            'spike_threshold_percent': 2.0,  # 2% spike warning
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        
        # State tracking
        self.last_tick: Optional[TickData] = None
        self.price_history: deque = deque(maxlen=100)
    
    def validate(self, tick: TickData) -> ValidationResult:
        """Validate tick data"""
        result = ValidationResult()
        self.total_validated += 1
        
        # Basic field validation
        self._validate_basic_fields(tick, result)
        
        # Price validation
        self._validate_price(tick, result)
        
        # Volume validation
        self._validate_volume(tick, result)
        
        # Timestamp validation
        self._validate_timestamp(tick, result)
        
        # Consistency checks (if we have previous tick)
        if self.last_tick:
            self._validate_consistency(tick, result)
        
        # Update state
        self.last_tick = tick
        self.price_history.append(tick.price)
        
        # Update statistics
        self.total_errors += len(result.errors)
        self.total_warnings += len(result.warnings)
        
        return result
    
    def _validate_basic_fields(self, tick: TickData, result: ValidationResult) -> None:
        """Validate basic tick fields"""
        if not tick.symbol:
            result.add_error("Missing symbol")
        
        if not tick.timestamp:
            result.add_error("Missing timestamp")
    
    def _validate_price(self, tick: TickData, result: ValidationResult) -> None:
        """Validate tick price"""
        # Check price range
        if tick.price <= 0:
            result.add_error(f"Invalid price: {tick.price} (must be positive)")
        elif tick.price < self.config['min_price']:
            result.add_warning(f"Price below minimum: {tick.price} < {self.config['min_price']}")
        elif tick.price > self.config['max_price']:
            result.add_warning(f"Price above maximum: {tick.price} > {self.config['max_price']}")
        
        # Check for price spikes
        if self.price_history and len(self.price_history) >= 10:
            avg_price = sum(self.price_history) / len(self.price_history)
            price_change_percent = abs(tick.price - avg_price) / avg_price * 100
            
            if price_change_percent > self.config['spike_threshold_percent']:
                result.add_warning(
                    f"Price spike detected: {price_change_percent:.2f}% "
                    f"from average {avg_price:.2f}"
                )
            
            result.add_metric('price_change_percent', price_change_percent)
    
    def _validate_volume(self, tick: TickData, result: ValidationResult) -> None:
        """Validate tick volume"""
        if tick.volume < 0:
            result.add_error(f"Invalid volume: {tick.volume} (must be non-negative)")
        elif tick.volume > self.config['max_volume']:
            result.add_warning(f"Unusually high volume: {tick.volume}")
        
        result.add_metric('volume', tick.volume)
    
    def _validate_timestamp(self, tick: TickData, result: ValidationResult) -> None:
        """Validate tick timestamp"""
        now = datetime.now()
        
        # Check if timestamp is in the future
        if tick.timestamp > now:
            result.add_error(f"Timestamp in future: {tick.timestamp}")
        
        # Check for stale data
        age_seconds = (now - tick.timestamp).total_seconds()
        if age_seconds > self.config['stale_threshold_seconds']:
            result.add_warning(f"Stale data: {age_seconds:.0f} seconds old")
        
        result.add_metric('data_age_seconds', age_seconds)
    
    def _validate_consistency(self, tick: TickData, result: ValidationResult) -> None:
        """Validate consistency with previous tick"""
        # Check timestamp ordering
        if tick.timestamp <= self.last_tick.timestamp:
            result.add_error(
                f"Timestamp not increasing: {tick.timestamp} <= {self.last_tick.timestamp}"
            )
        
        # Check price change
        price_change = abs(tick.price - self.last_tick.price)
        price_change_percent = (price_change / self.last_tick.price) * 100
        
        if price_change_percent > self.config['max_price_change_percent']:
            result.add_error(
                f"Excessive price change: {price_change_percent:.2f}% "
                f"({self.last_tick.price:.2f} -> {tick.price:.2f})"
            )
        
        result.add_metric('tick_price_change_percent', price_change_percent)


class BarValidator(BaseValidator):
    """
    Validates OHLCV bar data for quality and consistency
    
    Checks:
    - OHLC relationship (High >= all, Low <= all)
    - Volume validity
    - Timestamp alignment
    - Gap detection
    - Bar size consistency
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize bar validator"""
        default_config = {
            'min_price': 1.0,
            'max_price': 100000.0,
            'max_volume': 10000000,
            'max_gap_percent': 10.0,  # 10% max gap between bars
            'expected_bar_duration': None,  # Set based on timeframe
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        
        # State tracking
        self.last_bar: Optional[BarData] = None
        self.bar_durations: deque = deque(maxlen=100)
    
    def validate(self, bar: BarData) -> ValidationResult:
        """Validate bar data"""
        result = ValidationResult()
        self.total_validated += 1
        
        # Basic validation
        self._validate_basic_fields(bar, result)
        
        # OHLC relationship validation
        self._validate_ohlc_relationship(bar, result)
        
        # Price range validation
        self._validate_price_range(bar, result)
        
        # Volume validation
        self._validate_volume(bar, result)
        
        # Timestamp validation
        self._validate_timestamp(bar, result)
        
        # Consistency checks (if we have previous bar)
        if self.last_bar:
            self._validate_consistency(bar, result)
        
        # Update state
        self.last_bar = bar
        
        # Update statistics
        self.total_errors += len(result.errors)
        self.total_warnings += len(result.warnings)
        
        return result
    
    def _validate_basic_fields(self, bar: BarData, result: ValidationResult) -> None:
        """Validate basic bar fields"""
        if not bar.symbol:
            result.add_error("Missing symbol")
        
        if not bar.timestamp:
            result.add_error("Missing timestamp")
        
        if bar.timeframe <= 0:
            result.add_error(f"Invalid timeframe: {bar.timeframe}")
    
    def _validate_ohlc_relationship(self, bar: BarData, result: ValidationResult) -> None:
        """Validate OHLC relationships"""
        # High must be highest
        if bar.high < max(bar.open, bar.close):
            result.add_error(
                f"Invalid high: {bar.high} < max(open={bar.open}, close={bar.close})"
            )
        
        # Low must be lowest
        if bar.low > min(bar.open, bar.close):
            result.add_error(
                f"Invalid low: {bar.low} > min(open={bar.open}, close={bar.close})"
            )
        
        # High >= Low
        if bar.high < bar.low:
            result.add_error(f"High < Low: {bar.high} < {bar.low}")
        
        # Calculate bar metrics
        bar_range = bar.high - bar.low
        bar_body = abs(bar.close - bar.open)
        
        result.add_metric('bar_range', bar_range)
        result.add_metric('bar_body', bar_body)
        result.add_metric('body_to_range_ratio', bar_body / bar_range if bar_range > 0 else 0)
    
    def _validate_price_range(self, bar: BarData, result: ValidationResult) -> None:
        """Validate price ranges"""
        prices = [bar.open, bar.high, bar.low, bar.close]
        
        for price in prices:
            if price <= 0:
                result.add_error(f"Invalid price: {price} (must be positive)")
            elif price < self.config['min_price']:
                result.add_warning(f"Price below minimum: {price}")
            elif price > self.config['max_price']:
                result.add_warning(f"Price above maximum: {price}")
    
    def _validate_volume(self, bar: BarData, result: ValidationResult) -> None:
        """Validate bar volume"""
        if bar.volume < 0:
            result.add_error(f"Invalid volume: {bar.volume} (must be non-negative)")
        elif bar.volume > self.config['max_volume']:
            result.add_warning(f"Unusually high volume: {bar.volume}")
        
        result.add_metric('volume', bar.volume)
    
    def _validate_timestamp(self, bar: BarData, result: ValidationResult) -> None:
        """Validate bar timestamp"""
        # Check timeframe alignment
        minutes = bar.timestamp.hour * 60 + bar.timestamp.minute
        
        if minutes % bar.timeframe != 0:
            result.add_warning(
                f"Timestamp not aligned to {bar.timeframe}-minute boundary: {bar.timestamp}"
            )
    
    def _validate_consistency(self, bar: BarData, result: ValidationResult) -> None:
        """Validate consistency with previous bar"""
        # Check timestamp ordering
        if bar.timestamp <= self.last_bar.timestamp:
            result.add_error(
                f"Timestamp not increasing: {bar.timestamp} <= {self.last_bar.timestamp}"
            )
        
        # Check for gaps
        expected_timestamp = self.last_bar.timestamp + timedelta(minutes=bar.timeframe)
        if bar.timestamp > expected_timestamp:
            gap_minutes = (bar.timestamp - expected_timestamp).total_seconds() / 60
            result.add_warning(f"Gap detected: {gap_minutes:.0f} minutes")
        
        # Check price gaps
        price_gap = abs(bar.open - self.last_bar.close)
        gap_percent = (price_gap / self.last_bar.close) * 100
        
        if gap_percent > self.config['max_gap_percent']:
            result.add_warning(
                f"Large price gap: {gap_percent:.2f}% "
                f"({self.last_bar.close:.2f} -> {bar.open:.2f})"
            )
        
        result.add_metric('gap_percent', gap_percent)
        
        # Track bar duration
        duration = (bar.timestamp - self.last_bar.timestamp).total_seconds() / 60
        self.bar_durations.append(duration)
        result.add_metric('bar_duration_minutes', duration)


class DataQualityMonitor:
    """
    Monitors overall data quality across the system
    
    Features:
    - Aggregate validation statistics
    - Quality metrics tracking
    - Anomaly detection
    - Quality reports
    """
    
    def __init__(self):
        """Initialize data quality monitor"""
        self.logger = get_logger(self.__class__.__name__)
        
        # Validators
        self.tick_validator = TickValidator()
        self.bar_validators: Dict[int, BarValidator] = {}  # Keyed by timeframe
        
        # Quality metrics
        self.quality_metrics = {
            'total_ticks_validated': 0,
            'total_bars_validated': 0,
            'tick_error_rate': 0.0,
            'bar_error_rate': 0.0,
            'last_update': None
        }
        
        # Anomaly tracking
        self.anomalies: List[Dict[str, Any]] = []
        self.max_anomalies = 100
    
    def validate_tick(self, tick: TickData) -> ValidationResult:
        """Validate tick and update metrics"""
        result = self.tick_validator.validate(tick)
        
        # Update metrics
        self.quality_metrics['total_ticks_validated'] += 1
        self.quality_metrics['tick_error_rate'] = (
            self.tick_validator.total_errors / self.tick_validator.total_validated
        )
        
        # Track anomalies
        if not result.is_valid:
            self._record_anomaly('tick', tick, result)
        
        return result
    
    def validate_bar(self, bar: BarData) -> ValidationResult:
        """Validate bar and update metrics"""
        # Get or create validator for timeframe
        if bar.timeframe not in self.bar_validators:
            self.bar_validators[bar.timeframe] = BarValidator({
                'expected_bar_duration': bar.timeframe
            })
        
        validator = self.bar_validators[bar.timeframe]
        result = validator.validate(bar)
        
        # Update metrics
        self.quality_metrics['total_bars_validated'] += 1
        total_bar_errors = sum(v.total_errors for v in self.bar_validators.values())
        total_bar_validated = sum(v.total_validated for v in self.bar_validators.values())
        
        if total_bar_validated > 0:
            self.quality_metrics['bar_error_rate'] = total_bar_errors / total_bar_validated
        
        # Track anomalies
        if not result.is_valid:
            self._record_anomaly('bar', bar, result)
        
        return result
    
    def _record_anomaly(self, data_type: str, data: Any, result: ValidationResult) -> None:
        """Record data anomaly"""
        anomaly = {
            'timestamp': datetime.now(),
            'data_type': data_type,
            'errors': result.errors,
            'warnings': result.warnings,
            'data_summary': str(data)
        }
        
        self.anomalies.append(anomaly)
        
        # Keep only recent anomalies
        if len(self.anomalies) > self.max_anomalies:
            self.anomalies = self.anomalies[-self.max_anomalies:]
        
        self.logger.warning(f"Data anomaly detected data_type={data_type} errors={len(result.errors}"),
            warnings=len(result.warnings)
        )
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Generate data quality report"""
        self.quality_metrics['last_update'] = datetime.now()
        
        report = {
            'summary': self.quality_metrics.copy(),
            'tick_validation': {
                'total_validated': self.tick_validator.total_validated,
                'total_errors': self.tick_validator.total_errors,
                'total_warnings': self.tick_validator.total_warnings,
                'error_rate': self.quality_metrics['tick_error_rate']
            },
            'bar_validation': {},
            'recent_anomalies': self.anomalies[-10:],  # Last 10 anomalies
            'anomaly_count': len(self.anomalies)
        }
        
        # Add bar validation stats per timeframe
        for timeframe, validator in self.bar_validators.items():
            report['bar_validation'][f'{timeframe}min'] = {
                'total_validated': validator.total_validated,
                'total_errors': validator.total_errors,
                'total_warnings': validator.total_warnings,
                'error_rate': validator.total_errors / validator.total_validated if validator.total_validated > 0 else 0
            }
        
        return report
    
    def reset_statistics(self) -> None:
        """Reset all validation statistics"""
        self.tick_validator.reset_statistics()
        for validator in self.bar_validators.values():
            validator.reset_statistics()
        
        self.quality_metrics = {
            'total_ticks_validated': 0,
            'total_bars_validated': 0,
            'tick_error_rate': 0.0,
            'bar_error_rate': 0.0,
            'last_update': None
        }
        
        self.anomalies.clear()
        self.logger.info("Data quality statistics reset")


# Convenience functions
def validate_tick(tick: TickData, config: Optional[Dict[str, Any]] = None) -> ValidationResult:
    """Validate a single tick"""
    validator = TickValidator(config)
    return validator.validate(tick)


def validate_bar(bar: BarData, config: Optional[Dict[str, Any]] = None) -> ValidationResult:
    """Validate a single bar"""
    validator = BarValidator(config)
    return validator.validate(bar)


def create_quality_monitor() -> DataQualityMonitor:
    """Create a new data quality monitor instance"""
    return DataQualityMonitor()