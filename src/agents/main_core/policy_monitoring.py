"""
File: src/agents/main_core/policy_monitoring.py (NEW FILE)
Production monitoring for shared policy
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque
import logging
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# Prometheus metrics
policy_decisions = Counter('policy_decisions_total',
                          'Total policy decisions',
                          ['action'])
decision_confidence = Histogram('decision_confidence',
                               'Decision confidence distribution')
mc_dropout_uncertainty = Histogram('mc_dropout_uncertainty',
                                 'MC Dropout uncertainty')
reasoning_scores = Gauge('reasoning_scores',
                        'Multi-head reasoning scores',
                        ['head'])
policy_latency = Histogram('policy_latency_seconds',
                          'Policy inference latency')

class SharedPolicyMonitor:
    """
    Production monitoring for shared policy network.
    Tracks decision quality, uncertainty, and performance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_window = config.get('metrics_window', 100)
        
        # Metric buffers
        self.decision_history = deque(maxlen=self.metrics_window)
        self.confidence_history = deque(maxlen=self.metrics_window)
        self.uncertainty_history = deque(maxlen=self.metrics_window)
        self.reasoning_history = deque(maxlen=self.metrics_window)
        self.latency_history = deque(maxlen=self.metrics_window)
        
        # Action distribution tracking
        self.action_counts = {'initiate_trade': 0, 'do_nothing': 0}
        
    def record_decision(self, output: 'PolicyOutput', 
                       mc_result: 'MCDropoutResult',
                       latency: float):
        """Record decision metrics."""
        # Action taken
        action = output.action_probs.argmax(dim=-1).item()
        action_name = 'initiate_trade' if action == 0 else 'do_nothing'
        
        self.action_counts[action_name] += 1
        policy_decisions.labels(action=action_name).inc()
        
        # Confidence
        confidence = output.action_probs.max(dim=-1)[0].item()
        self.confidence_history.append(confidence)
        decision_confidence.observe(confidence)
        
        # MC Dropout uncertainty
        self.uncertainty_history.append(mc_result.epistemic_uncertainty)
        mc_dropout_uncertainty.observe(mc_result.epistemic_uncertainty)
        
        # Reasoning scores
        if output.reasoning_scores:
            self.reasoning_history.append(output.reasoning_scores)
            for head, score in output.reasoning_scores.items():
                if head.endswith('_weight'):
                    continue
                reasoning_scores.labels(head=head).set(score)
                
        # Latency
        self.latency_history.append(latency)
        policy_latency.observe(latency)
        
        # Store full decision
        self.decision_history.append({
            'action': action_name,
            'confidence': confidence,
            'uncertainty': mc_result.epistemic_uncertainty,
            'should_qualify': mc_result.should_qualify,
            'reasoning': output.reasoning_scores
        })
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics."""
        if not self.decision_history:
            return {}
            
        # Action distribution
        total_decisions = sum(self.action_counts.values())
        action_distribution = {
            k: v / total_decisions if total_decisions > 0 else 0
            for k, v in self.action_counts.items()
        }
        
        # Recent metrics
        recent_decisions = list(self.decision_history)[-20:]
        qualify_rate = sum(1 for d in recent_decisions 
                          if d['should_qualify']) / len(recent_decisions)
        
        return {
            'total_decisions': total_decisions,
            'action_distribution': action_distribution,
            'qualify_rate': qualify_rate,
            'avg_confidence': np.mean(self.confidence_history) if self.confidence_history else 0,
            'avg_uncertainty': np.mean(self.uncertainty_history) if self.uncertainty_history else 0,
            'avg_latency_ms': np.mean(self.latency_history) * 1000 if self.latency_history else 0,
            'p99_latency_ms': np.percentile(self.latency_history, 99) * 1000 if self.latency_history else 0,
            'reasoning_balance': self._calculate_reasoning_balance()
        }
        
    def _calculate_reasoning_balance(self) -> Dict[str, float]:
        """Calculate balance of reasoning heads."""
        if not self.reasoning_history:
            return {}
            
        recent = list(self.reasoning_history)[-10:]
        head_scores = {
            'structure': [],
            'timing': [],
            'risk': [],
            'regime': []
        }
        
        for reasoning in recent:
            for head in head_scores.keys():
                if head in reasoning:
                    head_scores[head].append(reasoning[head])
                    
        return {
            head: np.mean(scores) if scores else 0
            for head, scores in head_scores.items()
        }