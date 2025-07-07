"""
File: src/agents/main_core/regime_monitoring.py (NEW FILE)
Monitoring utilities for regime embedder
"""

import torch
import numpy as np
from typing import Dict, List, Any
from collections import deque
import logging
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# Prometheus metrics
regime_transitions = Counter('regime_transitions_total', 
                           'Total number of regime transitions detected')
regime_stability = Gauge('regime_stability', 
                        'Current regime stability score')
embedding_latency = Histogram('regime_embedding_latency_seconds',
                            'Latency of regime embedding computation')
component_importance = Gauge('regime_component_importance',
                           'Importance of regime components',
                           ['dimension'])

class RegimeEmbedderMonitor:
    """
    Monitoring system for regime embedder in production.
    Tracks performance, stability, and anomalies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_window = config.get('metrics_window', 100)
        
        # Metric buffers
        self.latencies = deque(maxlen=self.metrics_window)
        self.transition_scores = deque(maxlen=self.metrics_window)
        self.uncertainties = deque(maxlen=self.metrics_window)
        self.regime_magnitudes = deque(maxlen=self.metrics_window)
        
        # Anomaly detection
        self.anomaly_threshold = config.get('anomaly_threshold', 3.0)
        self.anomaly_count = 0
        
    def record_inference(self, regime_vector: torch.Tensor,
                        embedding_result: 'RegimeEmbedding',
                        latency: float):
        """Record metrics from an inference."""
        # Update latency
        self.latencies.append(latency)
        embedding_latency.observe(latency)
        
        # Update transition score
        if embedding_result.transition_score is not None:
            self.transition_scores.append(embedding_result.transition_score)
            if embedding_result.transition_score > 0.5:
                regime_transitions.inc()
                
        # Update stability
        if len(self.transition_scores) > 10:
            stability = 1.0 - np.mean(list(self.transition_scores)[-10:])
            regime_stability.set(stability)
            
        # Update uncertainties
        avg_uncertainty = embedding_result.std.mean().item()
        self.uncertainties.append(avg_uncertainty)
        
        # Update regime magnitude
        magnitude = torch.norm(regime_vector).item()
        self.regime_magnitudes.append(magnitude)
        
        # Check for anomalies
        self._check_anomalies(regime_vector, magnitude)
        
        # Update component importance
        if embedding_result.component_importance:
            for dim, importance in embedding_result.component_importance.items():
                component_importance.labels(dimension=dim).set(importance)
                
    def _check_anomalies(self, regime_vector: torch.Tensor, magnitude: float):
        """Check for anomalous regimes."""
        # Check magnitude
        if len(self.regime_magnitudes) > 20:
            mean_mag = np.mean(list(self.regime_magnitudes)[:-1])
            std_mag = np.std(list(self.regime_magnitudes)[:-1])
            
            z_score = abs(magnitude - mean_mag) / (std_mag + 1e-6)
            if z_score > self.anomaly_threshold:
                self.anomaly_count += 1
                logger.warning(
                    f"Anomalous regime detected: magnitude={magnitude:.3f}, "
                    f"z_score={z_score:.2f}"
                )
                
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get current health metrics."""
        return {
            'avg_latency_ms': np.mean(self.latencies) * 1000 if self.latencies else 0,
            'p99_latency_ms': np.percentile(self.latencies, 99) * 1000 if self.latencies else 0,
            'regime_stability': 1.0 - np.mean(self.transition_scores) if self.transition_scores else 1.0,
            'avg_uncertainty': np.mean(self.uncertainties) if self.uncertainties else 0,
            'anomaly_rate': self.anomaly_count / max(len(self.regime_magnitudes), 1),
            'is_healthy': self._is_healthy()
        }
        
    def _is_healthy(self) -> bool:
        """Determine if embedder is healthy."""
        if not self.latencies:
            return True
            
        # Check latency
        if np.mean(self.latencies) * 1000 > 5.0:  # >5ms average
            return False
            
        # Check anomaly rate
        if self.anomaly_count / max(len(self.regime_magnitudes), 1) > 0.05:  # >5% anomalies
            return False
            
        return True