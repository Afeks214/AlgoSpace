"""
Metrics collection and monitoring utilities.

This module provides lightweight metrics collection for system monitoring
and performance tracking. It's designed for minimal overhead in production.
"""

from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, Any, Optional, List
import time
import threading
import json
from pathlib import Path


class MetricsCollector:
    """
    Lightweight metrics collector for performance monitoring.
    
    Features:
    - Counter metrics (increment)
    - Gauge metrics (set value)
    - Histogram metrics (observe values)
    - Thread-safe operations
    - Export to various formats
    """
    
    def __init__(self, namespace: str):
        self.namespace = namespace
        self._lock = threading.Lock()
        
        # Metric storage
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Timing
        self._start_time = time.time()
        
    def increment(self, metric: str, value: float = 1.0):
        """Increment a counter metric."""
        with self._lock:
            self._counters[metric] += value
            
    def gauge(self, metric: str, value: float):
        """Set a gauge metric."""
        with self._lock:
            self._gauges[metric] = value
            
    def observe(self, metric: str, value: float):
        """Record an observation for histogram."""
        with self._lock:
            self._histograms[metric].append(value)
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics as dictionary."""
        with self._lock:
            # Calculate histogram statistics
            histogram_stats = {}
            for name, values in self._histograms.items():
                if values:
                    values_list = list(values)
                    histogram_stats[name] = {
                        'count': len(values_list),
                        'min': min(values_list),
                        'max': max(values_list),
                        'mean': sum(values_list) / len(values_list),
                        'p50': self._percentile(values_list, 50),
                        'p95': self._percentile(values_list, 95),
                        'p99': self._percentile(values_list, 99)
                    }
                    
            return {
                'namespace': self.namespace,
                'uptime_seconds': time.time() - self._start_time,
                'counters': dict(self._counters),
                'gauges': dict(self._gauges),
                'histograms': histogram_stats
            }
            
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._start_time = time.time()
            
    @staticmethod
    def _percentile(values: List[float], p: float) -> float:
        """Calculate percentile of values."""
        sorted_values = sorted(values)
        index = int(len(sorted_values) * p / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
        
    def export_json(self, filepath: Path):
        """Export metrics to JSON file."""
        metrics = self.get_metrics()
        metrics['timestamp'] = datetime.utcnow().isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
            
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        metrics = self.get_metrics()
        
        # Add namespace to all metrics
        prefix = f"{self.namespace}_"
        
        # Export counters
        for name, value in metrics['counters'].items():
            lines.append(f"{prefix}{name}_total {value}")
            
        # Export gauges
        for name, value in metrics['gauges'].items():
            lines.append(f"{prefix}{name} {value}")
            
        # Export histograms
        for name, stats in metrics['histograms'].items():
            for stat_name, stat_value in stats.items():
                if stat_name != 'count':
                    lines.append(f"{prefix}{name}_{stat_name} {stat_value}")
                    
        return '\n'.join(lines)


class MetricsRegistry:
    """
    Global registry for metrics collectors.
    
    Provides centralized access to all metrics in the system.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._collectors = {}
        return cls._instance
        
    def get_collector(self, namespace: str) -> MetricsCollector:
        """Get or create metrics collector for namespace."""
        if namespace not in self._collectors:
            self._collectors[namespace] = MetricsCollector(namespace)
        return self._collectors[namespace]
        
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics from all collectors."""
        return {
            namespace: collector.get_metrics()
            for namespace, collector in self._collectors.items()
        }
        
    def export_all_json(self, filepath: Path):
        """Export all metrics to JSON."""
        metrics = self.get_all_metrics()
        metrics['timestamp'] = datetime.utcnow().isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)


def setup_prometheus_metrics(port: int = 9090):
    """
    Setup Prometheus metrics endpoint.
    
    Args:
        port: Port to serve metrics on
    """
    try:
        from prometheus_client import start_http_server, Counter, Gauge, Histogram
        
        # Start metrics server
        start_http_server(port)
        
        return True
        
    except ImportError:
        # Prometheus client not available
        return False


class PerformanceTracker:
    """
    Track performance metrics for specific operations.
    """
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self._active_operations = {}
        
    def start_operation(self, operation: str) -> str:
        """Start tracking an operation."""
        op_id = f"{operation}_{time.time_ns()}"
        self._active_operations[op_id] = time.perf_counter()
        return op_id
        
    def end_operation(self, op_id: str):
        """End tracking an operation."""
        if op_id in self._active_operations:
            start_time = self._active_operations.pop(op_id)
            duration = time.perf_counter() - start_time
            
            # Extract operation name
            operation = op_id.split('_')[0]
            
            # Record metrics
            self.collector.observe(f"{operation}_duration_seconds", duration)
            self.collector.increment(f"{operation}_total")
            
            return duration
        return None
        
    def track_operation(self, operation: str):
        """Context manager for tracking operations."""
        class OperationTracker:
            def __init__(self, tracker, operation):
                self.tracker = tracker
                self.operation = operation
                self.op_id = None
                
            def __enter__(self):
                self.op_id = self.tracker.start_operation(self.operation)
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.tracker.end_operation(self.op_id)
                
        return OperationTracker(self, operation)