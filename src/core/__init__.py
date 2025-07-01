"""Core module for the AlgoSpace trading system.

This module contains the fundamental components of the system including
the kernel, event bus, component base classes, and core data structures.
"""

from .kernel import AlgoSpaceKernel
from .component_base import ComponentBase
from .event_bus import EventBus
from .events import Event, EventType, BarData, TickData
from .config import ConfigurationError

__all__ = [
    "AlgoSpaceKernel",
    "ComponentBase",
    "EventBus",
    "Event",
    "EventType",
    "BarData",
    "TickData",
    "ConfigurationError",
]