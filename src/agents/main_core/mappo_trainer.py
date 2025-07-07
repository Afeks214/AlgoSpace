"""
File: src/agents/main_core/mappo_trainer.py (NEW FILE)
MAPPO training utilities for shared policy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import logging

logger = logging.getLogger(__name__)

class MAPPOTrainer:
    """
    Multi-Agent PPO trainer for shared policy network.
    Implements the complete training pipeline with multi-objective optimization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Training hyperparameters
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_param = config.get('clip_param', 0.2)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        
        # Multi-objective weights
        self.objective_weights = config.get('objective_weights', {
            'return': 0.4,
            'risk_adjusted': 0.3,
            'timing': 0.2,
            'regime': 0.1
        })
        
        # Training settings
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.mini_batch_size = config.get('mini_batch_size', 64)
        self.buffer_size = config.get('buffer_size', 2048)
        
        # Initialize networks
        self.policy = None  # Set externally
        self.value_function = None  # Set externally
        self.optimizer = None  # Set externally
        
        # Training statistics
        self.training_stats = {
            'policy_loss': deque(maxlen=100),
            'value_loss': deque(maxlen=100),
            'entropy': deque(maxlen=100),
            'kl_divergence': deque(maxlen=100),
            'explained_variance': deque(maxlen=100)
        }
        
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                    next_values: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Reward sequence
            values: Value estimates
            next_values: Next value estimates
            dones: Done flags
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
        
    def ppo_update(self, rollout_buffer: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform PPO update on collected rollout data.
        
        Args:
            rollout_buffer: Dictionary containing rollout data
            
        Returns:
            Dictionary of training metrics
        """
        # Extract data
        states = rollout_buffer['states']
        actions = rollout_buffer['actions']
        old_log_probs = rollout_buffer['log_probs']
        advantages = rollout_buffer['advantages']
        returns = rollout_buffer['returns']
        
        # Training metrics
        policy_losses = []
        value_losses = []
        entropies = []
        kl_divs = []
        
        # Multiple epochs of updates
        for epoch in range(self.ppo_epochs):
            # Create random minibatches
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_indices = indices[start:end]
                
                # Get minibatch data
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                # Evaluate actions with current policy
                eval_result = self.policy.evaluate_actions(mb_states, mb_actions)
                log_probs = eval_result['log_probs']
                entropy = eval_result['entropy']
                values = eval_result['values']
                
                # Calculate ratio for PPO
                ratio = torch.exp(log_probs - mb_old_log_probs)
                
                # Clipped surrogate loss
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, mb_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss + 
                    self.value_loss_coef * value_loss + 
                    self.entropy_coef * entropy_loss
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                with torch.no_grad():
                    kl_div = (mb_old_log_probs - log_probs).mean()
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropies.append(entropy.mean().item())
                    kl_divs.append(kl_div.item())
                    
                # Early stopping if KL divergence too high
                if kl_div.abs() > 0.02:
                    logger.info(f"Early stopping at epoch {epoch} due to high KL divergence")
                    break
                    
        # Calculate explained variance
        with torch.no_grad():
            values_pred = self.value_function(states)['value'].squeeze()
            explained_var = 1 - (returns - values_pred).var() / returns.var()
            
        # Update statistics
        metrics = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
            'kl_divergence': np.mean(kl_divs),
            'explained_variance': explained_var.item()
        }
        
        for key, value in metrics.items():
            self.training_stats[key].append(value)
            
        return metrics
        
    def calculate_multi_objective_reward(self, 
                                       trade_result: Dict[str, Any],
                                       market_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate multi-objective rewards for training.
        
        Args:
            trade_result: Results from trade execution
            market_state: Current market conditions
            
        Returns:
            Dictionary of rewards for each objective
        """
        rewards = {}
        
        # Return objective
        if trade_result['executed']:
            rewards['return'] = trade_result['pnl'] / trade_result['risk']
        else:
            rewards['return'] = -0.01  # Small penalty for no action
            
        # Risk-adjusted return
        if trade_result['executed']:
            sharpe_contrib = trade_result['pnl'] / (trade_result['risk'] * 
                                                   market_state['volatility'])
            rewards['risk_adjusted'] = np.tanh(sharpe_contrib)  # Bounded [-1, 1]
        else:
            rewards['risk_adjusted'] = 0.0
            
        # Timing quality
        if trade_result['executed']:
            # Reward based on how well we timed the entry
            entry_quality = trade_result.get('entry_quality', 0.5)
            rewards['timing'] = 2 * (entry_quality - 0.5)  # [-1, 1]
        else:
            rewards['timing'] = 0.0
            
        # Regime alignment
        if trade_result['executed']:
            # Reward trades aligned with regime
            regime_alignment = trade_result.get('regime_alignment', 0.5)
            rewards['regime'] = 2 * (regime_alignment - 0.5)  # [-1, 1]
        else:
            rewards['regime'] = 0.0
            
        return rewards