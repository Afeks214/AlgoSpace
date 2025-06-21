"""
System Kernel - Main Orchestration Engine

This module implements the central orchestration engine that manages all system components,
coordinates their lifecycle, and handles the event-driven architecture.

Based on Master PRD - System Kernel & Orchestration v1.0
"""

import asyncio
import signal
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime
import structlog

from .config import get_config, Config
from .events import EventBus, EventType, Event
from ..utils.logger import get_logger


class SystemKernel:
    """
    Main system orchestration engine
    
    Responsibilities:
    - Component lifecycle management
    - Event bus coordination
    - Configuration management
    - Graceful startup and shutdown
    - Error handling and recovery
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the System Kernel
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.logger = get_logger(self.__class__.__name__)
        self.config = get_config(config_path)
        self.event_bus = EventBus()
        
        # Component registry
        self.components: Dict[str, Any] = {}
        self.component_order: List[str] = []
        
        # System state
        self.is_running = False
        self.is_shutting_down = False
        self.startup_timestamp: Optional[datetime] = None
        
        # Register signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        self.logger.info("System Kernel initialized", 
                        mode=self.config.system_mode,
                        symbols=self.config.symbols)
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown"""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            self.logger.debug("Signal handlers registered")
        except Exception as e:
            self.logger.warning("Failed to register signal handlers", error=str(e))
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals"""
        self.logger.info("Shutdown signal received", signal=signum)
        if not self.is_shutting_down:
            asyncio.create_task(self.shutdown())
    
    def register_component(self, name: str, component: Any, 
                          dependencies: Optional[List[str]] = None) -> None:
        """
        Register a system component
        
        Args:
            name: Component name
            component: Component instance
            dependencies: List of component names this depends on
        """
        if name in self.components:
            raise ValueError(f"Component '{name}' already registered")
        
        self.components[name] = component
        
        # Simple dependency ordering (topological sort would be better for complex deps)
        if dependencies:
            # Insert after dependencies
            insert_pos = 0
            for dep in dependencies:
                if dep in self.component_order:
                    insert_pos = max(insert_pos, self.component_order.index(dep) + 1)
            self.component_order.insert(insert_pos, name)
        else:
            self.component_order.append(name)
        
        self.logger.debug("Component registered", 
                         name=name, 
                         dependencies=dependencies or [],
                         position=self.component_order.index(name))
    
    async def start(self) -> None:
        """
        Start the system and all components
        
        Startup sequence:
        1. System initialization
        2. Component startup (in dependency order)
        3. Event bus activation
        4. System ready notification
        """
        if self.is_running:
            self.logger.warning("System already running")
            return
        
        try:
            self.logger.info("--- AlgoSpace System Kernel Starting ---")
            self.startup_timestamp = datetime.now()
            
            # Emit system start event
            start_event = self.event_bus.create_event(
                EventType.SYSTEM_START,
                {
                    'timestamp': self.startup_timestamp,
                    'mode': self.config.system_mode,
                    'symbols': self.config.symbols,
                    'components': list(self.components.keys())
                },
                'SystemKernel'
            )
            self.event_bus.publish(start_event)
            
            # Start components in dependency order
            for component_name in self.component_order:
                await self._start_component(component_name)
            
            self.is_running = True
            
            self.logger.info("--- System Kernel Started Successfully ---",
                           components=len(self.components),
                           mode=self.config.system_mode)
            
            # Keep running until shutdown
            await self._run_main_loop()
            
        except Exception as e:
            self.logger.error("System startup failed", error=str(e))
            await self._emergency_shutdown()
            raise
    
    async def _start_component(self, name: str) -> None:
        """Start a single component"""
        component = self.components[name]
        
        try:
            self.logger.info(f"Starting component: {name}")
            
            # Check if component has async start method
            if hasattr(component, 'start') and callable(component.start):
                if asyncio.iscoroutinefunction(component.start):
                    await component.start()
                else:
                    component.start()
            
            # Emit component started event
            started_event = self.event_bus.create_event(
                EventType.COMPONENT_STARTED,
                {
                    'component_name': name,
                    'timestamp': datetime.now()
                },
                'SystemKernel'
            )
            self.event_bus.publish(started_event)
            
            self.logger.info(f"Component started successfully: {name}")
            
        except Exception as e:
            self.logger.error(f"Failed to start component: {name}", error=str(e))
            raise
    
    async def _run_main_loop(self) -> None:
        """Main system loop - keeps the system running"""
        try:
            while self.is_running and not self.is_shutting_down:
                # System heartbeat and monitoring
                await asyncio.sleep(1.0)
                
                # Basic health checks could go here
                # For now, just maintain the event loop
                
        except asyncio.CancelledError:
            self.logger.info("Main loop cancelled - shutting down")
        except Exception as e:
            self.logger.error("Error in main loop", error=str(e))
            await self._emergency_shutdown()
    
    async def shutdown(self) -> None:
        """
        Graceful system shutdown
        
        Shutdown sequence:
        1. Set shutdown flag
        2. Stop components (reverse order)
        3. Cleanup resources
        4. Final notification
        """
        if self.is_shutting_down:
            self.logger.warning("Shutdown already in progress")
            return
        
        self.is_shutting_down = True
        self.logger.info("--- System Kernel Shutdown Initiated ---")
        
        try:
            # Emit shutdown event
            shutdown_event = self.event_bus.create_event(
                EventType.SYSTEM_SHUTDOWN,
                {
                    'timestamp': datetime.now(),
                    'uptime_seconds': (datetime.now() - self.startup_timestamp).total_seconds() if self.startup_timestamp else 0
                },
                'SystemKernel'
            )
            self.event_bus.publish(shutdown_event)
            
            # Stop components in reverse order
            for component_name in reversed(self.component_order):
                await self._stop_component(component_name)
            
            self.is_running = False
            
            self.logger.info("--- System Kernel Shutdown Complete ---")
            
        except Exception as e:
            self.logger.error("Error during shutdown", error=str(e))
        finally:
            # Ensure we exit
            sys.exit(0)
    
    async def _stop_component(self, name: str) -> None:
        """Stop a single component"""
        component = self.components[name]
        
        try:
            self.logger.info(f"Stopping component: {name}")
            
            # Check if component has stop method
            if hasattr(component, 'stop') and callable(component.stop):
                if asyncio.iscoroutinefunction(component.stop):
                    await component.stop()
                else:
                    component.stop()
            
            # Emit component stopped event
            stopped_event = self.event_bus.create_event(
                EventType.COMPONENT_STOPPED,
                {
                    'component_name': name,
                    'timestamp': datetime.now()
                },
                'SystemKernel'
            )
            self.event_bus.publish(stopped_event)
            
            self.logger.info(f"Component stopped successfully: {name}")
            
        except Exception as e:
            self.logger.error(f"Error stopping component: {name}", error=str(e))
    
    async def _emergency_shutdown(self) -> None:
        """Emergency shutdown when normal shutdown fails"""
        self.logger.critical("Emergency shutdown initiated")
        
        # Emit error event
        error_event = self.event_bus.create_event(
            EventType.SYSTEM_ERROR,
            {
                'error_type': 'emergency_shutdown',
                'timestamp': datetime.now()
            },
            'SystemKernel'
        )
        self.event_bus.publish(error_event)
        
        # Force exit
        sys.exit(1)
    
    def get_component(self, name: str) -> Optional[Any]:
        """Get a registered component by name"""
        return self.components.get(name)
    
    def get_event_bus(self) -> EventBus:
        """Get the system event bus"""
        return self.event_bus
    
    def get_config(self) -> Config:
        """Get the system configuration"""
        return self.config
    
    @property
    def system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            'mode': self.config.system_mode,
            'symbols': self.config.symbols,
            'timeframes': self.config.timeframes,
            'is_running': self.is_running,
            'is_shutting_down': self.is_shutting_down,
            'component_count': len(self.components),
            'components': list(self.components.keys()),
            'startup_timestamp': self.startup_timestamp.isoformat() if self.startup_timestamp else None
        }


class ComponentBase:
    """
    Base class for all system components
    
    Provides standard lifecycle methods and event bus access
    """
    
    def __init__(self, name: str, kernel: SystemKernel):
        """
        Initialize component
        
        Args:
            name: Component name
            kernel: System kernel instance
        """
        self.name = name
        self.kernel = kernel
        self.config = kernel.get_config()
        self.event_bus = kernel.get_event_bus()
        self.logger = get_logger(f"{self.__class__.__name__}({name})")
        
        self.is_running = False
    
    async def start(self) -> None:
        """Start the component (override in subclasses)"""
        self.is_running = True
        self.logger.info(f"Component {self.name} started")
    
    async def stop(self) -> None:
        """Stop the component (override in subclasses)"""
        self.is_running = False
        self.logger.info(f"Component {self.name} stopped")
    
    def subscribe_to_event(self, event_type: EventType, callback) -> None:
        """Subscribe to an event type"""
        self.event_bus.subscribe(event_type, callback)
        self.logger.debug(f"Subscribed to {event_type.value}")
    
    def publish_event(self, event_type: EventType, payload: Any) -> None:
        """Publish an event"""
        event = self.event_bus.create_event(event_type, payload, self.name)
        self.event_bus.publish(event)