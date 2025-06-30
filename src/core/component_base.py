"""
Base component class for AlgoSpace system components.

This module provides the abstract base class that all system components
should inherit from to ensure consistent interface and behavior.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .kernel import AlgoSpaceKernel


class ComponentBase(ABC):
    """
    Abstract base class for all AlgoSpace system components.
    
    Provides common functionality for:
    - Component lifecycle management
    - Event bus integration
    - Configuration management
    - Status reporting
    """
    
    def __init__(self, name: str, kernel: 'AlgoSpaceKernel'):
        """
        Initialize the component.
        
        Args:
            name: Component name for identification
            kernel: Reference to the system kernel
        """
        self.name = name
        self.kernel = kernel
        self.config = kernel.config
        self.event_bus = kernel.event_bus
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the component.
        
        This method should:
        - Subscribe to relevant events
        - Initialize internal state
        - Perform any setup operations
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """
        Shutdown the component gracefully.
        
        This method should:
        - Unsubscribe from events
        - Clean up resources
        - Save state if necessary
        """
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current component status.
        
        Returns:
            Dictionary containing component status information
        """
        pass
    
    def is_initialized(self) -> bool:
        """Check if component is initialized."""
        return self._initialized