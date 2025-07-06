"""Thread safety utilities for AlgoSpace"""
import threading
from contextlib import contextmanager
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ThreadSafeManager:
    """Central thread safety manager"""
    
    def __init__(self):
        # Create locks for each component
        self._locks = {
            'event_bus': threading.RLock(),
            'feature_store': threading.RLock(),
            'matrix_assemblers': threading.RLock(),
            'model_inference': threading.RLock(),
            'decision_making': threading.RLock(),
            'order_execution': threading.RLock()
        }
        
        # Track lock acquisition for deadlock detection
        self._lock_holders = {}
        self._acquisition_order = []
        
    @contextmanager
    def acquire(self, component: str):
        """Safely acquire component lock"""
        thread_id = threading.get_ident()
        
        # Create lock for new components
        if component not in self._locks:
            self._locks[component] = threading.RLock()
        
        # Check for potential deadlock
        if thread_id in self._lock_holders:
            if component in self._lock_holders[thread_id]:
                logger.warning(f"Thread {thread_id} attempting to re-acquire {component}")
        
        try:
            self._locks[component].acquire()
            
            # Track acquisition
            if thread_id not in self._lock_holders:
                self._lock_holders[thread_id] = set()
            self._lock_holders[thread_id].add(component)
            
            yield
            
        finally:
            self._locks[component].release()
            if thread_id in self._lock_holders:
                self._lock_holders[thread_id].discard(component)
    
    def get_lock_stats(self) -> Dict[str, Any]:
        """Get current lock statistics"""
        return {
            'active_locks': len([h for h in self._lock_holders.values() if h]),
            'waiting_threads': threading.active_count() - 1,
            'potential_deadlocks': self._detect_deadlocks()
        }
    
    def _detect_deadlocks(self) -> int:
        """Simple deadlock detection"""
        # Check for circular dependencies
        return 0  # Implement actual detection

# Global thread safety manager
thread_safety = ThreadSafeManager()