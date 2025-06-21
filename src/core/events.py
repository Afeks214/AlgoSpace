from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Callable, Optional
import structlog

logger = structlog.get_logger()


class EventType(Enum):
    """Event types used throughout the system"""
    # Data Events
    NEW_TICK = "NEW_TICK"
    NEW_5MIN_BAR = "NEW_5MIN_BAR"
    NEW_30MIN_BAR = "NEW_30MIN_BAR"
    NEW_BAR = "NEW_BAR"  # Generic bar event (for backward compatibility)
    
    # System Events
    SYSTEM_START = "SYSTEM_START"
    SYSTEM_SHUTDOWN = "SYSTEM_SHUTDOWN"
    BACKTEST_COMPLETE = "BACKTEST_COMPLETE"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    COMPONENT_STARTED = "COMPONENT_STARTED"
    COMPONENT_STOPPED = "COMPONENT_STOPPED"
    
    # Connection Events
    CONNECTION_LOST = "CONNECTION_LOST"
    CONNECTION_RESTORED = "CONNECTION_RESTORED"
    
    # Indicator Events
    INDICATOR_UPDATE = "INDICATOR_UPDATE"
    INDICATORS_READY = "INDICATORS_READY"
    SYNERGY_DETECTED = "SYNERGY_DETECTED"
    
    # MARL Events
    TRADE_QUALIFIED = "TRADE_QUALIFIED"
    TRADE_REJECTED = "TRADE_REJECTED"
    EXECUTE_TRADE = "EXECUTE_TRADE"
    
    # Execution Events
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    
    # Risk Events
    RISK_BREACH = "RISK_BREACH"
    POSITION_UPDATE = "POSITION_UPDATE"


@dataclass
class TickData:
    """Standard tick data structure"""
    symbol: str
    timestamp: datetime
    price: float
    volume: int


@dataclass
class BarData:
    """Standard bar data structure"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: int  # 5 or 30 (minutes)


@dataclass
class Event:
    """Base event structure"""
    event_type: EventType
    timestamp: datetime
    payload: Any
    source: str


class EventBus:
    """Central event bus for system-wide communication"""
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._logger = structlog.get_logger(self.__class__.__name__)
    
    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Subscribe to an event type"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
        self._logger.debug("Subscriber registered", event_type=event_type.value)
    
    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Unsubscribe from an event type"""
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(callback)
                self._logger.debug("Subscriber removed", event_type=event_type.value)
            except ValueError:
                self._logger.warning("Callback not found for unsubscribe", event_type=event_type.value)
    
    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers"""
        if event.event_type in self._subscribers:
            for callback in self._subscribers[event.event_type]:
                try:
                    callback(event)
                except Exception as e:
                    self._logger.error(
                        "Error in event callback",
                        event_type=event.event_type.value,
                        error=str(e)
                    )
        
        # Log high-frequency events at debug level only
        if event.event_type in [EventType.NEW_TICK, EventType.NEW_BAR]:
            self._logger.debug("Event published", event_type=event.event_type.value)
        else:
            self._logger.info("Event published", event_type=event.event_type.value)
    
    def create_event(self, event_type: EventType, payload: Any, source: str) -> Event:
        """Factory method to create events with timestamp"""
        return Event(
            event_type=event_type,
            timestamp=datetime.now(),
            payload=payload,
            source=source
        )