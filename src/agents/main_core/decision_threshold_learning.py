"""
File: src/agents/main_core/decision_threshold_learning.py (NEW FILE)
Adaptive threshold learning for DecisionGate
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import logging
import time

logger = logging.getLogger(__name__)


class AdaptiveThresholdLearner:
    """
    Learns optimal decision thresholds from historical performance.
    
    Uses reinforcement learning principles to adjust thresholds
    based on trading outcomes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Base thresholds
        self.base_thresholds = {
            'execution': 0.65,
            'validation': 0.60,
            'risk': 0.70
        }
        
        # Learned adjustments
        self.threshold_adjustments = {
            'execution': 0.0,
            'validation': 0.0,
            'risk': 0.0
        }
        
        # Learning parameters
        self.learning_rate = config.get('learning_rate', 0.01)
        self.momentum = config.get('momentum', 0.9)
        self.decay = config.get('decay', 0.995)
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.threshold_history = deque(maxlen=1000)
        
        # Regime-specific adjustments
        self.regime_adjustments = {
            'trending': {'execution': -0.05, 'validation': -0.03, 'risk': 0.0},
            'volatile': {'execution': 0.08, 'validation': 0.05, 'risk': 0.10},
            'ranging': {'execution': 0.03, 'validation': 0.02, 'risk': 0.05},
            'transitioning': {'execution': 0.10, 'validation': 0.08, 'risk': 0.15}
        }
        
        # Moving averages for smooth updates
        self.adjustment_momentum = {
            'execution': 0.0,
            'validation': 0.0,
            'risk': 0.0
        }
        
    def get_threshold(
        self,
        threshold_type: str,
        market_regime: str = 'unknown',
        recent_performance: Optional[float] = None
    ) -> float:
        """
        Get current threshold with all adjustments.
        
        Args:
            threshold_type: Type of threshold (execution, validation, risk)
            market_regime: Current market regime
            recent_performance: Recent win rate or Sharpe ratio
            
        Returns:
            Adjusted threshold value
        """
        # Base threshold
        threshold = self.base_thresholds[threshold_type]
        
        # Learned adjustment
        threshold += self.threshold_adjustments[threshold_type]
        
        # Regime adjustment
        if market_regime in self.regime_adjustments:
            threshold += self.regime_adjustments[market_regime][threshold_type]
            
        # Performance-based adjustment
        if recent_performance is not None:
            perf_adjustment = self._calculate_performance_adjustment(
                threshold_type,
                recent_performance
            )
            threshold += perf_adjustment
            
        # Ensure valid range
        threshold = np.clip(threshold, 0.5, 0.95)
        
        return threshold
        
    def update_from_outcome(
        self,
        decision: str,
        outcome: Dict[str, Any],
        thresholds_used: Dict[str, float],
        market_context: Dict[str, Any]
    ):
        """
        Update thresholds based on trading outcome.
        
        Args:
            decision: EXECUTE or REJECT
            outcome: Trading outcome (if executed)
            thresholds_used: Thresholds used for decision
            market_context: Market conditions during decision
        """
        # Record performance
        performance_record = {
            'decision': decision,
            'outcome': outcome,
            'thresholds': thresholds_used,
            'market_context': market_context,
            'timestamp': time.time()
        }
        
        self.performance_history.append(performance_record)
        
        # Calculate reward
        reward = self._calculate_reward(decision, outcome)
        
        # Update thresholds
        for threshold_type in self.threshold_adjustments:
            if threshold_type in thresholds_used:
                self._update_threshold(
                    threshold_type,
                    thresholds_used[threshold_type],
                    reward,
                    decision,
                    outcome
                )
                
        # Decay adjustments
        self._apply_decay()
        
        # Log updates
        if len(self.performance_history) % 10 == 0:
            self._log_threshold_stats()
            
    def _calculate_reward(
        self,
        decision: str,
        outcome: Dict[str, Any]
    ) -> float:
        """Calculate reward for threshold update."""
        if decision == 'REJECT':
            # Small negative reward for rejections (opportunity cost)
            return -0.01
            
        elif decision == 'EXECUTE':
            if outcome.get('profitable', False):
                # Positive reward scaled by profit
                pnl_ratio = outcome.get('pnl_ratio', 1.0)
                return min(1.0, pnl_ratio / 2.0)  # Cap at 1.0
            else:
                # Negative reward for losses
                pnl_ratio = outcome.get('pnl_ratio', -1.0)
                return max(-1.0, pnl_ratio)  # Floor at -1.0
                
        return 0.0
        
    def _update_threshold(
        self,
        threshold_type: str,
        threshold_used: float,
        reward: float,
        decision: str,
        outcome: Dict[str, Any]
    ):
        """Update specific threshold based on reward."""
        # Calculate gradient
        if decision == 'EXECUTE' and reward > 0:
            # Good execution - maybe we can be less strict
            gradient = -self.learning_rate * reward
        elif decision == 'EXECUTE' and reward < 0:
            # Bad execution - need to be more strict
            gradient = self.learning_rate * abs(reward)
        elif decision == 'REJECT':
            # Rejection - slight pressure to be less strict
            gradient = -self.learning_rate * 0.01
        else:
            gradient = 0.0
            
        # Apply momentum
        self.adjustment_momentum[threshold_type] = (
            self.momentum * self.adjustment_momentum[threshold_type] +
            (1 - self.momentum) * gradient
        )
        
        # Update adjustment
        self.threshold_adjustments[threshold_type] += self.adjustment_momentum[threshold_type]
        
        # Clip adjustment range
        self.threshold_adjustments[threshold_type] = np.clip(
            self.threshold_adjustments[threshold_type],
            -0.15,
            0.15
        )
        
    def _calculate_performance_adjustment(
        self,
        threshold_type: str,
        recent_performance: float
    ) -> float:
        """Calculate performance-based adjustment."""
        # If performance is good, we can be slightly less strict
        # If performance is bad, we need to be more strict
        
        if threshold_type == 'execution':
            if recent_performance > 0.7:  # High win rate
                return -0.02
            elif recent_performance < 0.5:  # Low win rate
                return 0.05
        elif threshold_type == 'risk':
            if recent_performance < 0.4:  # Poor performance
                return 0.10  # Much stricter on risk
                
        return 0.0
        
    def _apply_decay(self):
        """Apply decay to adjustments."""
        for threshold_type in self.threshold_adjustments:
            self.threshold_adjustments[threshold_type] *= self.decay
            self.adjustment_momentum[threshold_type] *= self.decay
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get threshold learning statistics."""
        if not self.performance_history:
            return {}
            
        recent_performance = list(self.performance_history)[-100:]
        
        # Calculate metrics
        executions = [p for p in recent_performance if p['decision'] == 'EXECUTE']
        rejections = [p for p in recent_performance if p['decision'] == 'REJECT']
        
        if executions:
            win_rate = sum(1 for e in executions if e['outcome'].get('profitable', False)) / len(executions)
            avg_pnl = np.mean([e['outcome'].get('pnl', 0) for e in executions])
        else:
            win_rate = 0.5
            avg_pnl = 0.0
            
        return {
            'current_adjustments': self.threshold_adjustments.copy(),
            'effective_thresholds': {
                k: self.base_thresholds[k] + self.threshold_adjustments[k]
                for k in self.base_thresholds
            },
            'recent_win_rate': win_rate,
            'recent_avg_pnl': avg_pnl,
            'execution_rate': len(executions) / len(recent_performance) if recent_performance else 0,
            'total_decisions': len(self.performance_history)
        }
        
    def _log_threshold_stats(self):
        """Log current threshold statistics."""
        stats = self.get_statistics()
        
        logger.info("Threshold Learning Update:")
        logger.info(f"  Adjustments: {stats['current_adjustments']}")
        logger.info(f"  Effective thresholds: {stats['effective_thresholds']}")
        logger.info(f"  Recent win rate: {stats['recent_win_rate']:.2%}")
        logger.info(f"  Execution rate: {stats['execution_rate']:.2%}")


class ThresholdOptimizer(nn.Module):
    """
    Neural network-based threshold optimizer.
    
    Learns optimal thresholds as a function of market conditions.
    """
    
    def __init__(self, input_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3),  # 3 threshold types
            nn.Sigmoid()
        )
        
        # Scale to appropriate range
        self.scale = nn.Parameter(torch.tensor([0.3, 0.3, 0.3]))
        self.bias = nn.Parameter(torch.tensor([0.6, 0.6, 0.7]))
        
    def forward(self, market_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict optimal thresholds given market features.
        
        Returns:
            Dictionary of threshold values
        """
        raw_output = self.network(market_features)
        
        # Scale and shift to appropriate ranges
        thresholds = raw_output * self.scale + self.bias
        
        return {
            'execution': thresholds[:, 0],
            'validation': thresholds[:, 1],
            'risk': thresholds[:, 2]
        }
        
    def compute_loss(
        self,
        predicted_thresholds: Dict[str, torch.Tensor],
        outcomes: torch.Tensor,
        decisions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss for threshold optimization.
        
        Balances between profit and risk.
        """
        # Extract thresholds
        exec_threshold = predicted_thresholds['execution']
        
        # Calculate components
        # 1. Profit loss - maximize profitable trades
        profit_mask = (outcomes > 0) & (decisions == 1)
        profit_loss = -torch.mean(outcomes[profit_mask]) if profit_mask.any() else 0
        
        # 2. Risk loss - minimize losing trades
        loss_mask = (outcomes < 0) & (decisions == 1)
        risk_loss = torch.mean(torch.abs(outcomes[loss_mask])) if loss_mask.any() else 0
        
        # 3. Opportunity loss - don't be too conservative
        rejection_rate = 1.0 - decisions.float().mean()
        opportunity_loss = rejection_rate * 0.1
        
        # 4. Threshold stability - avoid jumpy thresholds
        if hasattr(self, 'last_thresholds'):
            stability_loss = torch.mean(
                (exec_threshold - self.last_thresholds) ** 2
            ) * 0.1
        else:
            stability_loss = 0
            
        self.last_thresholds = exec_threshold.detach()
        
        # Combined loss
        total_loss = profit_loss + 2.0 * risk_loss + opportunity_loss + stability_loss
        
        return total_loss