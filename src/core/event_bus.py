# src/core/event_bus.py
"""
A simple, in-memory, thread-safe event bus for inter-component communication.
"""
import queue
from collections import defaultdict
from typing import Callable, Any, Dict, List
import threading
import logging

logger = logging.getLogger(__name__)


class EventBus:
    """Thread-safe event bus for decoupled component communication."""
    
    def __init__(self):
        """Initializes the event bus with a dictionary for subscribers."""
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_queue: queue.Queue = queue.Queue()
        self._running = False
        self._lock = threading.Lock()
        logger.info("EventBus initialized")

    def subscribe(self, event_type: str, handler: Callable) -> None:
        """
        Subscribes a handler function to a specific event type.
        
        Args:
            event_type: The name of the event to subscribe to.
            handler: The function/method to call when the event is published.
        """
        with self._lock:
            self.subscribers[event_type].append(handler)
            logger.debug(f"Handler {handler.__name__} subscribed to {event_type}")

    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """
        Unsubscribes a handler from a specific event type.
        
        Args:
            event_type: The name of the event to unsubscribe from.
            handler: The handler function to remove.
        """
        with self._lock:
            if event_type in self.subscribers and handler in self.subscribers[event_type]:
                self.subscribers[event_type].remove(handler)
                logger.debug(f"Handler {handler.__name__} unsubscribed from {event_type}")

    def publish(self, event_type: str, payload: Any = None) -> None:
        """
        Publishes an event, placing it onto the event queue for dispatching.
        
        Args:
            event_type: The name of the event.
            payload: The data associated with the event. Defaults to None.
        """
        if event_type is None:  # Shutdown signal
            self.event_queue.put(None)
            return
            
        event = {'type': event_type, 'payload': payload}
        self.event_queue.put(event)
        logger.debug(f"Event published: {event_type}")

    def dispatch_forever(self) -> None:
        """
        Starts an infinite loop to dispatch events from the queue to subscribers.
        This is the main event loop of the system.
        """
        self._running = True
        logger.info("Event bus dispatcher started. Waiting for events...")
        
        while self._running:
            try:
                event = self.event_queue.get(timeout=1.0)
                
                if event is None:  # Sentinel for shutdown
                    logger.info("Shutdown signal received")
                    break
                
                event_type = event.get('type')
                payload = event.get('payload')
                
                # Create a copy of subscribers to avoid modification during iteration
                with self._lock:
                    handlers = self.subscribers.get(event_type, []).copy()
                
                if handlers:
                    logger.debug(f"Dispatching {event_type} to {len(handlers)} handlers")
                    for handler in handlers:
                        try:
                            handler(payload)
                        except Exception as e:
                            logger.error(f"Error in handler {handler.__name__} for event {event_type}: {e}", 
                                       exc_info=True)
                else:
                    logger.warning(f"No handlers registered for event: {event_type}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Critical error in event dispatcher: {e}", exc_info=True)
        
        logger.info("Event bus dispatcher stopped")

    def stop(self) -> None:
        """Stops the event dispatcher loop."""
        self._running = False
        self.publish(None)  # Send shutdown signal