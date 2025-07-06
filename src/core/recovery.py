"""System recovery and failover mechanisms"""
import time
import traceback
from typing import Optional, Callable, Dict, Any
import logging

logger = logging.getLogger(__name__)

class RecoverySystem:
    """Handles system recovery from failures"""
    
    def __init__(self, config: Dict[str, Any]):
        self.max_attempts = config.get('max_recovery_attempts', 3)
        self.timeout = config.get('recovery_timeout', 30)
        self.recovery_strategies = {}
        self._component_states = {}
        
    def register_recovery(self, component: str, strategy: Callable):
        """Register recovery strategy for component"""
        self.recovery_strategies[component] = strategy
    
    def mark_failure(self, component: str, error: Exception):
        """Mark component as failed"""
        self._component_states[component] = {
            'status': 'failed',
            'error': str(error),
            'traceback': traceback.format_exc(),
            'timestamp': time.time(),
            'attempts': 0
        }
        
        # Attempt recovery
        self.recover_component(component)
    
    def recover_component(self, component: str) -> bool:
        """Attempt to recover failed component"""
        if component not in self.recovery_strategies:
            logger.error(f"No recovery strategy for {component}")
            return False
        
        state = self._component_states.get(component, {})
        attempts = state.get('attempts', 0)
        
        if attempts >= self.max_attempts:
            logger.error(f"Max recovery attempts reached for {component}")
            return False
        
        logger.info(f"Attempting recovery for {component} (attempt {attempts + 1})")
        
        try:
            # Execute recovery strategy
            strategy = self.recovery_strategies[component]
            strategy()
            
            # Mark as recovered
            self._component_states[component] = {
                'status': 'recovered',
                'timestamp': time.time(),
                'attempts': attempts + 1
            }
            
            logger.info(f"Successfully recovered {component}")
            return True
            
        except Exception as e:
            logger.error(f"Recovery failed for {component}: {e}")
            state['attempts'] = attempts + 1
            return False

# Global recovery system
recovery_system = RecoverySystem({})