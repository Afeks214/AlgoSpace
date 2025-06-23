"""
Base Matrix Assembler Implementation

Provides the foundation for all matrix assemblers with thread-safe
circular buffer management, robust error handling, and efficient
memory usage.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import threading
from collections import deque
import logging

from src.core.events import EventType, Event
from src.utils.logger import get_logger
from .normalizers import RollingNormalizer


class BaseMatrixAssembler(ABC):
    """
    Abstract base class for all matrix assemblers.
    
    Provides:
    - Thread-safe circular buffer implementation
    - Event subscription and handling
    - Common normalization utilities
    - Robust error handling and logging
    - Memory-efficient matrix management
    """
    
    def __init__(self, name: str, kernel: Any, config: Dict[str, Any]):
        """
        Initialize base matrix assembler.
        
        Args:
            name: Component name
            kernel: System kernel reference
            config: Configuration dictionary
        """
        self.name = name
        self.kernel = kernel
        self.config = config
        self.logger = get_logger(name)
        
        # Extract configuration
        self.window_size = config.get('window_size', 48)
        self.features = config.get('features', [])
        self.n_features = len(self.features)
        
        # Initialize matrix as float32 for neural network efficiency
        self.matrix = np.zeros((self.window_size, self.n_features), dtype=np.float32)
        
        # Circular buffer management
        self.current_index = 0
        self.n_updates = 0
        self.is_full = False
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Normalization tracking
        self.normalizers: Dict[str, RollingNormalizer] = {}
        self._init_normalizers()
        
        # Performance tracking
        self.last_update_time = None
        self.update_latencies = deque(maxlen=1000)
        
        # Error tracking
        self.error_count = 0
        self.last_error_time = None
        
        # State tracking
        self._is_ready = False
        self._warmup_period = config.get('warmup_period', self.window_size)
        
        # Subscribe to events
        self._subscribe_to_events()
        
        self.logger.info(
            f"Initialized {name} with window_size={self.window_size}, "
            f"n_features={self.n_features}"
        )
    
    def _init_normalizers(self) -> None:
        """Initialize rolling normalizers for each feature."""
        for feature in self.features:
            # Feature-specific configuration
            feature_config = self.config.get('feature_configs', {}).get(feature, {})
            alpha = feature_config.get('ema_alpha', 0.01)
            warmup = feature_config.get('warmup_samples', 100)
            
            self.normalizers[feature] = RollingNormalizer(
                alpha=alpha,
                warmup_samples=warmup
            )
    
    def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events."""
        event_bus = self.kernel.get_event_bus()
        event_bus.subscribe(EventType.INDICATORS_READY, self._on_indicators_ready)
        self.logger.debug(f"Subscribed to {EventType.INDICATORS_READY}")
    
    def _on_indicators_ready(self, event: Event) -> None:
        """Handle INDICATORS_READY event."""
        try:
            start_time = datetime.now()
            
            # Thread-safe update
            with self._lock:
                feature_store = event.payload
                self._update_matrix(feature_store)
            
            # Track performance
            latency = (datetime.now() - start_time).total_seconds() * 1000
            self.update_latencies.append(latency)
            
            if len(self.update_latencies) % 100 == 0:
                avg_latency = np.mean(list(self.update_latencies))
                self.logger.debug(f"Average update latency: {avg_latency:.2f}ms")
                
        except Exception as e:
            self.error_count += 1
            self.last_error_time = datetime.now()
            self.logger.error(f"Error in matrix update: {e}", exc_info=True)
    
    def _update_matrix(self, feature_store: Dict[str, Any]) -> None:
        """
        Update matrix with new features.
        
        Args:
            feature_store: Dictionary of current feature values
        """
        # Extract features
        raw_features = self.extract_features(feature_store)
        
        if raw_features is None:
            self.logger.warning("No features extracted, skipping update")
            return
        
        # Validate features
        if len(raw_features) != self.n_features:
            self.logger.error(
                f"Feature count mismatch: expected {self.n_features}, "
                f"got {len(raw_features)}"
            )
            return
        
        # Update normalizers
        for i, (feature_name, value) in enumerate(zip(self.features, raw_features)):
            if not np.isfinite(value):
                self.logger.warning(f"Non-finite value for {feature_name}: {value}")
                raw_features[i] = 0.0  # Safe default
            else:
                self.normalizers[feature_name].update(value)
        
        # Preprocess features
        processed_features = self.preprocess_features(raw_features, feature_store)
        
        # Update circular buffer
        self.matrix[self.current_index] = processed_features
        
        # Update indices
        self.current_index = (self.current_index + 1) % self.window_size
        self.n_updates += 1
        
        # Check if buffer is full
        if not self.is_full and self.n_updates >= self.window_size:
            self.is_full = True
            self.logger.info(f"Matrix buffer is now full after {self.n_updates} updates")
        
        # Check if ready
        if not self._is_ready and self.n_updates >= self._warmup_period:
            self._is_ready = True
            self.logger.info(f"Matrix is ready after {self.n_updates} updates")
        
        # Log periodically
        if self.n_updates % 100 == 0:
            self.logger.debug(f"Processed {self.n_updates} updates")
    
    @abstractmethod
    def extract_features(self, feature_store: Dict[str, Any]) -> Optional[List[float]]:
        """
        Extract relevant features from feature store.
        
        Args:
            feature_store: Complete feature store from IndicatorEngine
            
        Returns:
            List of raw feature values or None if extraction fails
        """
        pass
    
    @abstractmethod
    def preprocess_features(
        self, 
        raw_features: List[float], 
        feature_store: Dict[str, Any]
    ) -> np.ndarray:
        """
        Preprocess raw features for neural network input.
        
        Args:
            raw_features: List of raw feature values
            feature_store: Complete feature store for additional context
            
        Returns:
            Preprocessed feature array
        """
        pass
    
    def get_matrix(self) -> Optional[np.ndarray]:
        """
        Get current matrix for neural network input.
        
        Returns:
            Copy of matrix in correct chronological order or None if not ready
        """
        with self._lock:
            if not self.is_ready():
                self.logger.warning("Matrix not ready yet")
                return None
            
            # Return matrix in chronological order (oldest to newest)
            if self.is_full:
                # Reorder circular buffer
                matrix_copy = np.vstack([
                    self.matrix[self.current_index:],
                    self.matrix[:self.current_index]
                ])
            else:
                # Partial matrix
                matrix_copy = self.matrix[:self.n_updates].copy()
            
            return matrix_copy
    
    def get_latest_features(self) -> Optional[np.ndarray]:
        """
        Get the most recent feature vector.
        
        Returns:
            Latest feature vector or None if no data
        """
        with self._lock:
            if self.n_updates == 0:
                return None
            
            # Get the last updated row
            last_index = (self.current_index - 1) % self.window_size
            return self.matrix[last_index].copy()
    
    def is_ready(self) -> bool:
        """Check if matrix has enough data for use."""
        with self._lock:
            return self._is_ready
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current statistics about the matrix.
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            stats = {
                'name': self.name,
                'window_size': self.window_size,
                'n_features': self.n_features,
                'n_updates': self.n_updates,
                'is_full': self.is_full,
                'is_ready': self._is_ready,
                'current_index': self.current_index,
                'error_count': self.error_count,
                'last_error_time': self.last_error_time,
                'features': self.features
            }
            
            # Add performance stats
            if self.update_latencies:
                latencies = list(self.update_latencies)
                stats['performance'] = {
                    'avg_latency_ms': np.mean(latencies),
                    'max_latency_ms': np.max(latencies),
                    'min_latency_ms': np.min(latencies),
                    'p95_latency_ms': np.percentile(latencies, 95)
                }
            
            # Add matrix stats if ready
            if self.is_ready():
                matrix = self.get_matrix()
                if matrix is not None:
                    stats['matrix_stats'] = {
                        'shape': matrix.shape,
                        'mean': float(np.mean(matrix)),
                        'std': float(np.std(matrix)),
                        'min': float(np.min(matrix)),
                        'max': float(np.max(matrix)),
                        'non_finite_count': int(np.sum(~np.isfinite(matrix)))
                    }
            
            return stats
    
    def reset(self) -> None:
        """Reset the matrix assembler to initial state."""
        with self._lock:
            self.matrix.fill(0)
            self.current_index = 0
            self.n_updates = 0
            self.is_full = False
            self._is_ready = False
            self.error_count = 0
            self.last_error_time = None
            self.update_latencies.clear()
            
            # Reset normalizers
            self._init_normalizers()
            
            self.logger.info("Matrix assembler reset")
    
    def validate_matrix(self) -> Tuple[bool, List[str]]:
        """
        Validate matrix integrity.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        with self._lock:
            # Check for NaN or Inf values
            if np.any(~np.isfinite(self.matrix[:self.n_updates])):
                issues.append("Matrix contains non-finite values")
            
            # Check value ranges
            matrix_data = self.matrix[:self.n_updates]
            if matrix_data.size > 0:
                if np.max(np.abs(matrix_data)) > 10.0:
                    issues.append("Matrix contains values outside expected range [-10, 10]")
            
            # Check update consistency
            if self.n_updates > 0 and self.current_index >= self.window_size:
                issues.append("Current index exceeds window size")
            
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"window_size={self.window_size}, "
            f"n_features={self.n_features}, "
            f"n_updates={self.n_updates}, "
            f"is_ready={self.is_ready()})"
        )