"""
MAPPO Training Implementation for Main MARL Core.

This module implements Multi-Agent PPO training for the Main MARL Core
shared policy network with centralized training and decentralized execution.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class MAPPOTrainer:
    """
    Multi-Agent PPO training for Main MARL Core.
    
    Implements centralized training with decentralized execution (CTDE).
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda'))
        
        # Learning parameters
        self.lr = config['learning_rate']
        self.gamma = config['gamma']
        self.gae_lambda = config['gae_lambda']
        self.clip_ratio = config['clip_ratio']
        self.value_clip = config['value_clip']
        self.entropy_coef = config['entropy_coef']
        self.max_grad_norm = config['max_grad_norm']
        
        # Batch parameters
        self.batch_size = config['batch_size']
        self.n_epochs = config['n_epochs']
        self.n_minibatches = config['n_minibatches']
        
        # Initialize networks (will be set by training script)
        self.policy_network = None
        self.value_network = None
        self.decision_gate = None
        
        # Optimizers
        self.policy_optimizer = None
        self.value_optimizer = None
        self.gate_optimizer = None
        
        # Training metrics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'approx_kl': [],
            'clip_fraction': []
        }
        
    def setup_optimizers(self, policy_net, value_net, decision_gate):
        """Setup optimizers for all components."""
        self.policy_network = policy_net
        self.value_network = value_net
        self.decision_gate = decision_gate
        
        # Separate optimizers for stability
        self.policy_optimizer = optim.Adam(
            list(policy_net.parameters()),
            lr=self.lr,
            eps=1e-5
        )
        
        self.value_optimizer = optim.Adam(
            value_net.parameters(),
            lr=self.lr * 2,  # Higher LR for value function
            eps=1e-5
        )
        
        self.gate_optimizer = optim.Adam(
            decision_gate.parameters(),
            lr=self.lr,
            eps=1e-5
        )
        
        # Learning rate schedulers
        self.policy_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.policy_optimizer,
            T_max=self.config['training_steps']
        )
        
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Reward sequence [batch, seq_len]
            values: Value estimates [batch, seq_len + 1]
            dones: Done flags [batch, seq_len]
            
        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Compute GAE
        gae = 0
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = 0
            else:
                next_value = values[:, t + 1]
                
            delta = rewards[:, t] + self.gamma * next_value * (1 - dones[:, t]) - values[:, t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[:, t]) * gae
            advantages[:, t] = gae
            returns[:, t] = advantages[:, t] + values[:, t]
            
        return advantages, returns
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step on a batch.
        
        Args:
            batch: Training batch with states, actions, rewards, etc.
            
        Returns:
            Dictionary of training metrics
        """
        # Unpack batch
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        old_log_probs = batch['log_probs'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        dones = batch['dones'].to(self.device)
        old_values = batch['values'].to(self.device)
        
        # Compute advantages and returns
        with torch.no_grad():
            advantages, returns = self.compute_gae(rewards, old_values, dones)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        # Training epochs
        for epoch in range(self.n_epochs):
            # Generate minibatches
            indices = torch.randperm(states.size(0))
            
            for start_idx in range(0, states.size(0), self.batch_size // self.n_minibatches):
                end_idx = min(start_idx + self.batch_size // self.n_minibatches, states.size(0))
                mb_indices = indices[start_idx:end_idx]
                
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_old_values = old_values[mb_indices]
                
                # Forward pass through policy
                policy_output = self.policy_network(mb_states)
                
                # Compute log probabilities
                action_dist = torch.distributions.Categorical(
                    logits=policy_output.action_logits
                )
                mb_log_probs = action_dist.log_prob(mb_actions)
                entropy = action_dist.entropy().mean()
                
                # Compute value estimates
                mb_values = self.value_network(mb_states)
                
                # PPO loss
                ratio = torch.exp(mb_log_probs - mb_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                
                policy_loss = -torch.min(
                    ratio * mb_advantages,
                    clipped_ratio * mb_advantages
                ).mean()
                
                # Value loss with clipping
                value_pred_clipped = mb_old_values + torch.clamp(
                    mb_values - mb_old_values,
                    -self.value_clip,
                    self.value_clip
                )
                
                value_loss = torch.max(
                    F.mse_loss(mb_values, mb_returns),
                    F.mse_loss(value_pred_clipped, mb_returns)
                )
                
                # Total loss
                total_loss = (
                    policy_loss +
                    0.5 * value_loss -
                    self.entropy_coef * entropy
                )
                
                # Optimize
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                
                total_loss.backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    list(self.policy_network.parameters()) +
                    list(self.value_network.parameters()),
                    self.max_grad_norm
                )
                
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                # Track metrics
                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - mb_log_probs).mean()
                    clip_fraction = ((ratio - 1).abs() > self.clip_ratio).float().mean()
                    
                self.training_stats['policy_loss'].append(policy_loss.item())
                self.training_stats['value_loss'].append(value_loss.item())
                self.training_stats['entropy'].append(entropy.item())
                self.training_stats['approx_kl'].append(approx_kl.item())
                self.training_stats['clip_fraction'].append(clip_fraction.item())
                
        # Update learning rate
        self.policy_scheduler.step()
        
        # Return average metrics
        return {
            'policy_loss': np.mean(self.training_stats['policy_loss'][-self.n_epochs:]),
            'value_loss': np.mean(self.training_stats['value_loss'][-self.n_epochs:]),
            'entropy': np.mean(self.training_stats['entropy'][-self.n_epochs:]),
            'approx_kl': np.mean(self.training_stats['approx_kl'][-self.n_epochs:]),
            'clip_fraction': np.mean(self.training_stats['clip_fraction'][-self.n_epochs:])
        }
        
    def train_decision_gate(self, gate_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train the decision gate component."""
        states = gate_batch['extended_states'].to(self.device)
        risk_proposals = gate_batch['risk_proposals']
        targets = gate_batch['execution_targets'].to(self.device)
        
        # Forward pass
        gate_outputs = []
        for i in range(states.size(0)):
            output = self.decision_gate(
                states[i:i+1],
                risk_proposals[i],
                gate_batch['mc_consensus'][i]
            )
            gate_outputs.append(output['decision_logits'])
            
        gate_logits = torch.cat(gate_outputs, dim=0)
        
        # Compute loss
        gate_loss = F.cross_entropy(gate_logits, targets)
        
        # Add regularization for risk awareness
        risk_scores = torch.tensor([
            rp['risk_metrics']['portfolio_risk_score'] 
            for rp in risk_proposals
        ]).to(self.device)
        
        # Penalize high-risk approvals
        risk_penalty = (
            F.softmax(gate_logits, dim=-1)[:, 0] *  # Execute probability
            risk_scores
        ).mean()
        
        total_gate_loss = gate_loss + 0.1 * risk_penalty
        
        # Optimize
        self.gate_optimizer.zero_grad()
        total_gate_loss.backward()
        nn.utils.clip_grad_norm_(self.decision_gate.parameters(), self.max_grad_norm)
        self.gate_optimizer.step()
        
        # Metrics
        with torch.no_grad():
            gate_accuracy = (gate_logits.argmax(dim=-1) == targets).float().mean()
            
        return {
            'gate_loss': gate_loss.item(),
            'risk_penalty': risk_penalty.item(),
            'gate_accuracy': gate_accuracy.item()
        }


class ExperienceBuffer:
    """Buffer for storing training experiences."""
    
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        
        # Storage
        self.states = torch.zeros((capacity, state_dim))
        self.actions = torch.zeros((capacity,), dtype=torch.long)
        self.rewards = torch.zeros((capacity,))
        self.dones = torch.zeros((capacity,))
        self.log_probs = torch.zeros((capacity,))
        self.values = torch.zeros((capacity,))
        
    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float
    ):
        """Add experience to buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch from buffer."""
        indices = torch.randint(0, self.size, (batch_size,))
        
        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'dones': self.dones[indices],
            'log_probs': self.log_probs[indices],
            'values': self.values[indices]
        }
        
    def get_all(self) -> Dict[str, torch.Tensor]:
        """Get all experiences."""
        return {
            'states': self.states[:self.size],
            'actions': self.actions[:self.size],
            'rewards': self.rewards[:self.size],
            'dones': self.dones[:self.size],
            'log_probs': self.log_probs[:self.size],
            'values': self.values[:self.size]
        }
        
    def clear(self):
        """Clear buffer."""
        self.size = 0
        self.ptr = 0


class TrainingOrchestrator:
    """Orchestrates the entire training process."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda'))
        
        # Initialize trainer
        self.trainer = MAPPOTrainer(config['training'])
        
        # Initialize experience buffer
        self.experience_buffer = ExperienceBuffer(
            capacity=config['buffer_size'],
            state_dim=config['state_dim']
        )
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_metrics = []
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints/marl'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def train_episode(
        self, 
        environment, 
        policy_network, 
        value_network,
        decision_gate
    ) -> Dict[str, float]:
        """Train for one episode."""
        # Setup optimizers
        self.trainer.setup_optimizers(policy_network, value_network, decision_gate)
        
        # Reset environment
        state = environment.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Get action from policy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                policy_output = policy_network(state_tensor)
                
                # Sample action
                action_dist = torch.distributions.Categorical(
                    logits=policy_output.action_logits
                )
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                
                # Get value estimate
                value = value_network(state_tensor)
                
            # Take action in environment
            next_state, reward, done, info = environment.step(action.item())
            
            # Store experience
            self.experience_buffer.add(
                state=state_tensor.squeeze(0).cpu(),
                action=action.item(),
                reward=reward,
                done=done,
                log_prob=log_prob.item(),
                value=value.item()
            )
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
                
            state = next_state
            
        # Train on collected experiences
        if self.experience_buffer.size >= self.config['min_buffer_size']:
            experiences = self.experience_buffer.get_all()
            training_metrics = self.trainer.train_step(experiences)
            self.training_metrics.append(training_metrics)
            
            # Clear buffer after training
            self.experience_buffer.clear()
        else:
            training_metrics = {}
            
        # Record episode metrics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            **training_metrics
        }
        
    def save_checkpoint(
        self, 
        policy_network, 
        value_network, 
        decision_gate,
        episode: int
    ):
        """Save training checkpoint."""
        checkpoint = {
            'episode': episode,
            'policy_network': policy_network.state_dict(),
            'value_network': value_network.state_dict(),
            'decision_gate': decision_gate.state_dict(),
            'policy_optimizer': self.trainer.policy_optimizer.state_dict(),
            'value_optimizer': self.trainer.value_optimizer.state_dict(),
            'gate_optimizer': self.trainer.gate_optimizer.state_dict(),
            'training_metrics': self.training_metrics,
            'episode_rewards': self.episode_rewards,
            'config': self.config
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_episode_{episode}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if self.episode_rewards and self.episode_rewards[-1] == max(self.episode_rewards):
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.training_metrics = checkpoint.get('training_metrics', [])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint
        
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        if not self.episode_rewards:
            return {}
            
        recent_rewards = self.episode_rewards[-100:]  # Last 100 episodes
        
        summary = {
            'total_episodes': len(self.episode_rewards),
            'mean_reward': np.mean(self.episode_rewards),
            'recent_mean_reward': np.mean(recent_rewards),
            'best_reward': max(self.episode_rewards),
            'mean_episode_length': np.mean(self.episode_lengths),
            'total_training_steps': len(self.training_metrics)
        }
        
        if self.training_metrics:
            latest_metrics = self.training_metrics[-1]
            summary.update({
                'latest_policy_loss': latest_metrics.get('policy_loss', 0),
                'latest_value_loss': latest_metrics.get('value_loss', 0),
                'latest_entropy': latest_metrics.get('entropy', 0)
            })
            
        return summary


def calculate_trading_reward(
    action: str,
    outcome: Dict[str, Any],
    config: Dict[str, Any]
) -> float:
    """
    Calculate reward for MARL training.
    
    Balances profit, risk, and decision quality.
    """
    if action == 'pass':
        # Small negative for missing opportunities
        # But only if it was actually a good opportunity
        if outcome.get('was_profitable', False):
            return config['miss_penalty']  # e.g., -0.02
        else:
            return config['correct_pass_reward']  # e.g., 0.01
            
    elif action == 'qualify':
        if outcome['trade_executed']:
            # Risk-adjusted return
            pnl = outcome['pnl']
            risk = outcome['risk_taken']
            
            if risk > 0:
                risk_adjusted_return = pnl / risk
            else:
                risk_adjusted_return = 0
                
            # Base reward on risk-adjusted performance
            if risk_adjusted_return > 0:
                reward = 1.0 + min(risk_adjusted_return, 3.0)  # Cap at 4.0
            else:
                reward = -1.0 + max(risk_adjusted_return, -3.0)  # Floor at -4.0
                
            # Additional shaping
            # Sharpe contribution
            if 'sharpe_contribution' in outcome:
                reward += 0.1 * outcome['sharpe_contribution']
                
            # Drawdown penalty
            if 'drawdown_increase' in outcome:
                reward -= 0.5 * outcome['drawdown_increase']
                
            # Execution quality
            if 'slippage' in outcome:
                reward -= 0.1 * abs(outcome['slippage'])
                
        else:
            # Qualified but rejected by risk gate
            # Small penalty to discourage over-qualification
            reward = config['false_qualify_penalty']  # e.g., -0.05
            
    return reward


class RewardShaper:
    """Shapes rewards for better learning dynamics."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reward_history = []
        self.baseline = 0.0
        
    def shape_reward(
        self,
        base_reward: float,
        state: torch.Tensor,
        action: int,
        next_state: torch.Tensor,
        additional_info: Dict[str, Any]
    ) -> float:
        """
        Apply reward shaping techniques.
        
        Args:
            base_reward: Original reward from environment
            state: Current state
            action: Action taken
            next_state: Next state
            additional_info: Additional context
            
        Returns:
            Shaped reward
        """
        shaped_reward = base_reward
        
        # Potential-based reward shaping
        if self.config.get('use_potential_shaping', True):
            potential_current = self._calculate_potential(state)
            potential_next = self._calculate_potential(next_state)
            
            # Gamma * Φ(s') - Φ(s)
            gamma = self.config.get('gamma', 0.99)
            potential_bonus = gamma * potential_next - potential_current
            shaped_reward += 0.1 * potential_bonus
            
        # Uncertainty bonus for exploration
        if self.config.get('use_uncertainty_bonus', True):
            uncertainty = additional_info.get('uncertainty', 0.0)
            uncertainty_bonus = 0.01 * uncertainty
            shaped_reward += uncertainty_bonus
            
        # Consistency bonus
        if self.config.get('use_consistency_bonus', True):
            consistency = additional_info.get('consistency_score', 0.0)
            shaped_reward += 0.05 * consistency
            
        # Track reward history for adaptive baselines
        self.reward_history.append(shaped_reward)
        if len(self.reward_history) > 1000:
            self.reward_history.pop(0)
            
        # Update baseline
        if len(self.reward_history) >= 100:
            self.baseline = np.mean(self.reward_history[-100:])
            
        return shaped_reward
        
    def _calculate_potential(self, state: torch.Tensor) -> float:
        """
        Calculate potential function for reward shaping.
        
        This should be based on domain knowledge about what
        makes a state more valuable.
        """
        # Simple potential based on state norm
        # In practice, this would be more sophisticated
        return 0.1 * torch.norm(state).item()
        
    def get_shaped_baseline(self) -> float:
        """Get current reward baseline."""
        return self.baseline


# Training utilities
def create_training_config() -> Dict[str, Any]:
    """Create default training configuration."""
    return {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'training': {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_ratio': 0.2,
            'value_clip': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
            'batch_size': 256,
            'n_epochs': 10,
            'n_minibatches': 4,
            'training_steps': 1000000
        },
        'buffer_size': 10000,
        'min_buffer_size': 1000,
        'state_dim': 512,
        'checkpoint_dir': 'checkpoints/marl',
        'rewards': {
            'miss_penalty': -0.02,
            'correct_pass_reward': 0.01,
            'false_qualify_penalty': -0.05
        }
    }


def train_main_marl_core(
    policy_network,
    value_network, 
    decision_gate,
    environment,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main training function for MARL Core.
    
    Args:
        policy_network: Shared policy network
        value_network: Value function network
        decision_gate: Decision gate network
        environment: Trading environment
        config: Training configuration
        
    Returns:
        Training results and metrics
    """
    if config is None:
        config = create_training_config()
        
    # Initialize orchestrator
    orchestrator = TrainingOrchestrator(config)
    
    # Training loop
    num_episodes = config.get('num_episodes', 1000)
    
    for episode in range(num_episodes):
        # Train episode
        episode_metrics = orchestrator.train_episode(
            environment=environment,
            policy_network=policy_network,
            value_network=value_network,
            decision_gate=decision_gate
        )
        
        # Log progress
        if episode % 10 == 0:
            summary = orchestrator.get_training_summary()
            logger.info(
                f"Episode {episode}: "
                f"Reward={episode_metrics['episode_reward']:.3f}, "
                f"Length={episode_metrics['episode_length']}, "
                f"Mean Recent Reward={summary.get('recent_mean_reward', 0):.3f}"
            )
            
        # Save checkpoint
        if episode % 100 == 0:
            orchestrator.save_checkpoint(
                policy_network=policy_network,
                value_network=value_network,
                decision_gate=decision_gate,
                episode=episode
            )
            
    # Final summary
    final_summary = orchestrator.get_training_summary()
    logger.info(f"Training completed. Final summary: {final_summary}")
    
    return final_summary