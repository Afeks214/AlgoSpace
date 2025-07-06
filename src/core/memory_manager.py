"""Memory management and leak prevention"""
import gc
import weakref
import psutil
import os
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class MemoryManager:
    """Manages memory usage and prevents leaks"""
    
    def __init__(self, max_memory_mb: int = 2048):
        self.max_memory_mb = max_memory_mb
        self.process = psutil.Process(os.getpid())
        self._tracked_objects = weakref.WeakValueDictionary()
        self._cleanup_callbacks = []
        
    def register_cleanup(self, callback):
        """Register cleanup callback"""
        self._cleanup_callbacks.append(callback)
    
    def check_memory(self) -> Dict[str, float]:
        """Check current memory usage"""
        memory_info = self.process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'percent': self.process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    def force_cleanup(self):
        """Force memory cleanup"""
        logger.info("Forcing memory cleanup...")
        
        # Run callbacks
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Cleanup callback failed: {e}")
        
        # Force garbage collection
        gc.collect()
        
        # Clear caches
        self._clear_caches()
    
    def _clear_caches(self):
        """Clear all system caches"""
        # Clear PyTorch caches if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

# Global memory manager
memory_manager = MemoryManager()