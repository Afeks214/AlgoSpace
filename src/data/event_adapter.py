"""
Event adapter for integration with existing event system.

This module provides adapters to ensure the new data pipeline components
work seamlessly with the existing event infrastructure.
"""

from datetime import datetime
from typing import Dict, Any, Callable, Optional
import asyncio
import logging

from src.core.events import Event, EventType, EventBus as CoreEventBus

logger = logging.getLogger(__name__)


class AsyncEventBus:
    """
    Async wrapper for the existing synchronous EventBus.
    
    Provides async methods while maintaining compatibility with
    the existing event system.
    """
    
    def __init__(self, core_bus: Optional[CoreEventBus] = None):
        self.core_bus = core_bus or CoreEventBus()
        self._async_subscribers = {}
        
    async def subscribe(self, event_type: EventType, callback: Callable, filter_func: Optional[Callable] = None):
        """
        Async subscribe method.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Async callback function
            filter_func: Optional filter function
        """
        # Create wrapper for async callback
        def sync_wrapper(event: Event):
            # Apply filter if provided
            if filter_func and not filter_func(event):
                return
                
            # Convert event format
            adapted_event = self._adapt_event(event)
            
            # Schedule async callback
            asyncio.create_task(self._run_async_callback(callback, adapted_event))
            
        # Subscribe to core bus
        self.core_bus.subscribe(event_type, sync_wrapper)
        
        # Track for cleanup
        if event_type not in self._async_subscribers:
            self._async_subscribers[event_type] = []
        self._async_subscribers[event_type].append((callback, sync_wrapper))
        
    async def publish(self, event: 'AsyncEvent'):
        """
        Async publish method.
        
        Args:
            event: Event to publish
        """
        # Convert to core event format
        core_event = Event(
            event_type=event.type,
            timestamp=datetime.now(),
            payload=event.data.get('tick') or event.data.get('bar') or event.data,
            source=event.source
        )
        
        # Publish synchronously
        self.core_bus.publish(core_event)
        
    def _adapt_event(self, core_event: Event) -> 'AsyncEvent':
        """Adapt core event to async event format."""
        # Determine data key based on event type
        if core_event.event_type == EventType.NEW_TICK:
            data = {'tick': core_event.payload}
        elif core_event.event_type in [EventType.NEW_5MIN_BAR, EventType.NEW_30MIN_BAR]:
            data = {'bar': core_event.payload}
        else:
            data = core_event.payload
            
        return AsyncEvent(
            type=core_event.event_type,
            data=data,
            source=core_event.source
        )
        
    async def _run_async_callback(self, callback: Callable, event: 'AsyncEvent'):
        """Run async callback with error handling."""
        try:
            await callback(event)
        except Exception as e:
            logger.error(f"Error in async callback: {e}")


class AsyncEvent:
    """
    Async event structure matching the new implementation.
    """
    
    def __init__(self, type: EventType, data: Dict[str, Any], source: str):
        self.type = type
        self.data = data
        self.source = source


class EventBus:
    """
    Drop-in replacement EventBus for the new data pipeline.
    
    Provides the interface expected by new components while
    integrating with the existing event system.
    """
    
    def __init__(self, core_bus: Optional[CoreEventBus] = None):
        self._async_bus = AsyncEventBus(core_bus)
        
    async def subscribe(self, event_type: EventType, callback: Callable, filter_func: Optional[Callable] = None):
        """Subscribe to events."""
        await self._async_bus.subscribe(event_type, callback, filter_func)
        
    async def publish(self, event: AsyncEvent):
        """Publish events."""
        await self._async_bus.publish(event)


def create_event(type: EventType, data: Dict[str, Any], source: str) -> AsyncEvent:
    """
    Factory function to create events in the new format.
    
    Args:
        type: Event type
        data: Event data
        source: Event source
        
    Returns:
        AsyncEvent instance
    """
    return AsyncEvent(type=type, data=data, source=source)


# Re-export Event as AsyncEvent for compatibility
Event = AsyncEvent