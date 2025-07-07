"""
Time utilities for high-precision timing and temporal operations.

This module provides utilities for precise time measurements and
temporal boundary calculations required by the data pipeline.
"""

from datetime import datetime, timedelta
from typing import Optional, List
import time


class TimeUtils:
    """Collection of time-related utility functions."""
    
    @staticmethod
    def now_utc() -> datetime:
        """Get current UTC time with microsecond precision."""
        return datetime.utcnow()
    
    @staticmethod
    def high_resolution_time() -> float:
        """
        Get high-resolution time in seconds.
        
        Uses the highest resolution clock available on the platform.
        """
        return time.perf_counter()
    
    @staticmethod
    def measure_latency_ns(func, *args, **kwargs) -> tuple:
        """
        Measure function execution time in nanoseconds.
        
        Args:
            func: Function to measure
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Tuple of (result, latency_ns)
        """
        start = time.perf_counter_ns()
        result = func(*args, **kwargs)
        end = time.perf_counter_ns()
        
        return result, end - start
    
    @staticmethod
    def floor_to_interval(timestamp: datetime, interval_minutes: int) -> datetime:
        """
        Floor timestamp to interval boundary.
        
        Args:
            timestamp: Timestamp to floor
            interval_minutes: Interval in minutes
            
        Returns:
            Floored timestamp
        """
        # Remove seconds and microseconds
        timestamp = timestamp.replace(second=0, microsecond=0)
        
        # Floor to interval boundary
        minutes = timestamp.minute
        floored_minutes = (minutes // interval_minutes) * interval_minutes
        
        return timestamp.replace(minute=floored_minutes)
    
    @staticmethod
    def next_interval(timestamp: datetime, interval_minutes: int) -> datetime:
        """
        Get next interval boundary.
        
        Args:
            timestamp: Current timestamp
            interval_minutes: Interval in minutes
            
        Returns:
            Next interval boundary
        """
        current_boundary = TimeUtils.floor_to_interval(timestamp, interval_minutes)
        return current_boundary + timedelta(minutes=interval_minutes)
    
    @staticmethod
    def get_missing_intervals(
        start: datetime,
        end: datetime,
        interval_minutes: int
    ) -> List[datetime]:
        """
        Get list of interval boundaries between start and end.
        
        Args:
            start: Start timestamp
            end: End timestamp
            interval_minutes: Interval in minutes
            
        Returns:
            List of interval boundaries
        """
        intervals = []
        
        # Start from next boundary after start
        current = TimeUtils.next_interval(start, interval_minutes)
        end_boundary = TimeUtils.floor_to_interval(end, interval_minutes)
        
        # Collect all boundaries up to (but not including) end boundary
        while current < end_boundary:
            intervals.append(current)
            current += timedelta(minutes=interval_minutes)
            
        return intervals
    
    @staticmethod
    def is_market_hours(timestamp: datetime, timezone: str = "America/Chicago") -> bool:
        """
        Check if timestamp is within market hours.
        
        Args:
            timestamp: Timestamp to check
            timezone: Market timezone
            
        Returns:
            True if within market hours
        """
        # This is a simplified implementation
        # In production, would integrate with market calendar
        import pytz
        
        tz = pytz.timezone(timezone)
        local_time = timestamp.astimezone(tz)
        
        # Futures market hours (simplified)
        # Sunday 6PM - Friday 5PM CT
        weekday = local_time.weekday()
        hour = local_time.hour
        
        # Saturday is closed
        if weekday == 5:
            return False
            
        # Sunday opens at 6PM
        if weekday == 6:
            return hour >= 18
            
        # Friday closes at 5PM
        if weekday == 4:
            return hour < 17
            
        # Monday-Thursday: 24 hours
        return True
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """
        Format duration in human-readable format.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted string
        """
        if seconds < 1e-6:
            return f"{seconds * 1e9:.0f}ns"
        elif seconds < 1e-3:
            return f"{seconds * 1e6:.0f}μs"
        elif seconds < 1:
            return f"{seconds * 1e3:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}min"
        else:
            return f"{seconds / 3600:.1f}h"


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        
        if self.name:
            formatted = TimeUtils.format_duration(self.duration)
            print(f"{self.name}: {formatted}")
            
    def get_duration(self) -> float:
        """Get duration in seconds."""
        if self.duration is None:
            if self.start_time is not None and self.end_time is None:
                # Still running
                return time.perf_counter() - self.start_time
            return 0.0
        return self.duration