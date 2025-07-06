"""
Multi-Agent Proximal Policy Optimization (MAPPO) trainer.

Implements centralized training with decentralized execution for
multi-agent reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
from collections import defaultdict

from .base_trainer import BaseTrainer
from ..utils.helpers import get_optimizer, get_scheduler, MetricsTracker

import structlog

logger = structlog.get_logger()


class MAPPOTrainer(BaseTrainer):
    """
    MAPPO trainer for multi-agent reinforcement learning.
    
    Features:
    - Centralized critic for value estimation
    - Individual actor policies for each agent
    - PPO clipping for stable updates
    - GAE for advantage estimation
    """
    
    def __init__(
        self,
        agents: Dict[str, nn.Module],
        config: Dict[str, Any]
    ):
        """
        Initialize MAPPO trainer.
        
        Args:
            agents: Dictionary of agent models
            config: Training configuration
        """
        super().__init__(config)
        
        self.agents = agents
        self.agent_names = list(agents.keys())
        
        # Move agents to device
        for agent in self.agents.values():
            agent.to(self.device)
        
        # PPO-specific parameters
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.num_epochs = config.get('num_epochs', 4)
        self.mini_batch_size = config.get('mini_batch_size', 64)
        
        # Centralized critic
        self.centralized_critic = self._build_centralized_critic()
        self.centralized_critic.to(self.device)
        
        # Optimizers
        self.actor_optimizers = {}
        for name, agent in self.agents.items():
            self.actor_optimizers[name] = get_optimizer(
                agent,
                config.get('actor_optimizer', {'type': 'adam', 'learning_rate': 3e-4})
            )
        
        self.critic_optimizer = get_optimizer(
            self.centralized_critic,
            config.get('critic_optimizer', {'type': 'adam', 'learning_rate': 3e-4})
        )
        
        # Learning rate schedulers
        self.actor_schedulers = {}
        self.critic_scheduler = None
        
        if 'scheduler' in config:
            for name, optimizer in self.actor_optimizers.items():
                self.actor_schedulers[name] = get_scheduler(optimizer, config['scheduler'])
            self.critic_scheduler = get_scheduler(self.critic_optimizer, config['scheduler'])
        
        # Metrics tracking
        self.metrics_tracker = MetricsTracker([
            'total_loss', 'policy_loss', 'value_loss', 'entropy',
            'kl_divergence', 'clip_fraction', 'explained_variance'
        ])
        
        logger.info(f"Initialized MAPPO trainer num_agents={len(self.agents}")
            clip_epsilon=self.clip_epsilon,
            entropy_coef=self.entropy_coef
        )
    
    def _build_centralized_critic(self) -> nn.Module:
        """Build centralized critic network."""
        class CentralizedCritic(nn.Module):
            def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256]):
                super().__init__()
                
                layers = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.LayerNorm(hidden_dim)
                    ])
                    prev_dim = hidden_dim
                
                layers.append(nn.Linear(prev_dim, 1))
                
                self.network = nn.Sequential(*layers)
                
                # Initialize weights
                for layer in self.network:
                    if isinstance(layer, nn.Linear):
                        nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                        nn.init.constant_(layer.bias, 0.0)
            
            def forward(self, global_state: torch.Tensor) -> torch.Tensor:
                return self.network(global_state)
        
        # Calculate input dimension (concatenated observations from all agents)
        # This is a placeholder - actual dimension depends on observation space
        input_dim = 256 * len(self.agents)  # Assuming 256-dim embedding per agent
        
        return CentralizedCritic(input_dim)
    
    def collect_experience(self, env, num_steps: int) -> Dict[str, Any]:
        """
        Collect experience from environment using current policies.
        
        Args:
            env: Multi-agent environment
            num_steps: Number of steps to collect
            
        Returns:
            Dictionary of collected experience
        """
        # Initialize storage
        experience = {
            'observations': defaultdict(list),
            'actions': defaultdict(list),
            'rewards': defaultdict(list),
            'dones': defaultdict(list),
            'values': defaultdict(list),
            'log_probs': defaultdict(list),
            'global_states': []
        }
        
        # Reset environment
        observations = env.reset()
        
        # Set agents to eval mode
        for agent in self.agents.values():
            agent.eval()
        
        with torch.no_grad():
            for step in range(num_steps):
                # Store observations
                for agent_name, obs in observations.items():
                    experience['observations'][agent_name].append(obs)
                
                # Get actions from all agents
                actions = {}
                log_probs = {}
                values = {}
                
                # Construct global state for centralized critic
                global_state = self._construct_global_state(observations)
                experience['global_states'].append(global_state)
                
                # Get centralized value estimate
                global_state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
                global_value = self.centralized_critic(global_state_tensor).squeeze().cpu().numpy()
                
                for agent_name, agent in self.agents.items():
                    # Convert observation to tensor
                    obs_dict = observations[agent_name]
                    
                    # Forward pass through agent
                    agent_output = self._agent_forward(agent, obs_dict)
                    
                    # Sample action
                    action, log_prob = self._sample_action(agent_output)
                    
                    actions[agent_name] = action.cpu().numpy()
                    log_probs[agent_name] = log_prob.cpu().numpy()
                    values[agent_name] = global_value  # Use centralized value
                
                # Store actions and values
                for agent_name in self.agent_names:
                    experience['actions'][agent_name].append(actions[agent_name])
                    experience['log_probs'][agent_name].append(log_probs[agent_name])
                    experience['values'][agent_name].append(values[agent_name])
                
                # Step environment
                next_observations, rewards, dones, infos = env.step(actions)
                
                # Store rewards and dones
                for agent_name in self.agent_names:
                    experience['rewards'][agent_name].append(rewards[agent_name])
                    experience['dones'][agent_name].append(dones[agent_name])
                
                # Update observations
                observations = next_observations
                
                # Check if episode ended
                if any(dones.values()):
                    observations = env.reset()
        
        # Convert lists to arrays
        for key in ['observations', 'actions', 'rewards', 'dones', 'values', 'log_probs']:
            for agent_name in self.agent_names:
                if key == 'observations':
                    # Handle dictionary observations
                    pass  # Keep as list of dicts for now
                else:
                    experience[key][agent_name] = np.array(experience[key][agent_name])
        
        experience['global_states'] = np.array(experience['global_states'])
        
        # Set agents back to train mode
        for agent in self.agents.values():
            agent.train()
        
        return experience
    
    def update_policies(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """
        Update agent policies using collected experience.
        
        Args:
            experience: Collected experience data
            
        Returns:
            Dictionary of training metrics
        """
        self.metrics_tracker.reset()
        
        # Prepare data for updates
        prepared_data = self._prepare_training_data(experience)
        
        # Multiple epochs of updates
        for epoch in range(self.num_epochs):
            # Create mini-batches
            mini_batches = self._create_mini_batches(prepared_data)
            
            for batch in mini_batches:
                # Update each agent
                for agent_name in self.agent_names:
                    agent_metrics = self._update_agent(agent_name, batch)
                    
                    # Track metrics
                    for metric_name, value in agent_metrics.items():
                        self.metrics_tracker.update(f"{agent_name}_{metric_name}", value)
                
                # Update centralized critic
                critic_metrics = self._update_critic(batch)
                for metric_name, value in critic_metrics.items():
                    self.metrics_tracker.update(f"critic_{metric_name}", value)
        
        # Get average metrics
        metrics = self.metrics_tracker.get_all_averages()
        
        # Update learning rate schedulers
        if self.actor_schedulers:
            for scheduler in self.actor_schedulers.values():
                if scheduler is not None:
                    scheduler.step()
        
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()
        
        return metrics
    
    def _agent_forward(self, agent: nn.Module, observation: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through agent model.
        
        Args:
            agent: Agent model
            observation: Agent observation
            
        Returns:
            Agent output dictionary
        """
        # Convert observation components to tensors
        obs_tensors = {}
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                obs_tensors[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
            elif isinstance(value, (int, float)):
                obs_tensors[key] = torch.tensor(value).unsqueeze(0).to(self.device)
            else:
                obs_tensors[key] = value
        
        # Dummy inputs for now - should match agent architecture
        market_matrix = obs_tensors.get('market_matrix', torch.zeros(1, 100, 8).to(self.device))
        regime_vector = obs_tensors.get('regime_vector', torch.zeros(1, 8).to(self.device))
        synergy_context = observation.get('synergy_context', {})
        
        # Forward pass
        output = agent(market_matrix, regime_vector, synergy_context)
        
        return output
    
    def _sample_action(self, agent_output: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from agent output.
        
        Args:
            agent_output: Output from agent forward pass
            
        Returns:
            action: Sampled action
            log_prob: Log probability of action
        """
        # Get action logits
        action_logits = agent_output.get('action', torch.zeros(1, 3))
        
        # Create categorical distribution
        dist = torch.distributions.Categorical(logits=action_logits)
        
        # Sample action
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Get continuous components (size, timing)
        # For now, use fixed values - should come from agent output
        size = torch.tensor([0.5])
        timing = torch.tensor([0])
        
        # Combine into full action
        full_action = torch.stack([action.float(), size, timing], dim=-1)
        
        return full_action, log_prob
    
    def _construct_global_state(self, observations: Dict[str, Dict[str, Any]]) -> np.ndarray:
        """
        Construct global state from individual observations.
        
        Args:
            observations: Dictionary of agent observations
            
        Returns:
            Global state vector
        """
        # Placeholder implementation
        # In practice, this would concatenate relevant features from all agents
        global_features = []
        
        for agent_name in self.agent_names:
            obs = observations[agent_name]
            # Extract key features (placeholder)
            features = np.zeros(256)  # Placeholder features
            global_features.append(features)
        
        return np.concatenate(global_features)
    
    def _prepare_training_data(self, experience: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Prepare experience data for training.
        
        Args:
            experience: Raw experience data
            
        Returns:
            Prepared tensors for training
        """
        prepared = {}
        
        # Convert to tensors
        for agent_name in self.agent_names:
            agent_data = {}
            
            # Actions, rewards, dones, values, log_probs
            for key in ['actions', 'rewards', 'dones', 'values', 'log_probs']:
                data = experience[key][agent_name]
                agent_data[key] = torch.FloatTensor(data).to(self.device)
            
            # Compute returns and advantages
            with torch.no_grad():
                # Get final value estimate
                final_obs = experience['observations'][agent_name][-1]
                final_output = self._agent_forward(self.agents[agent_name], final_obs)
                
                # Use centralized critic for final value
                final_global_state = self._construct_global_state({agent_name: final_obs})
                final_global_state_tensor = torch.FloatTensor(final_global_state).unsqueeze(0).to(self.device)
                final_value = self.centralized_critic(final_global_state_tensor).squeeze()
                
                # Compute GAE
                advantages, returns = self.compute_gae(
                    agent_data['rewards'].unsqueeze(0),
                    agent_data['values'].unsqueeze(0),
                    agent_data['dones'].unsqueeze(0),
                    final_value.unsqueeze(0),
                    self.gae_lambda
                )
                
                agent_data['advantages'] = advantages.squeeze(0)
                agent_data['returns'] = returns.squeeze(0)
                
                # Normalize advantages
                agent_data['advantages'] = self.normalize_advantages(agent_data['advantages'])
            
            prepared[agent_name] = agent_data
        
        # Global states
        prepared['global_states'] = torch.FloatTensor(experience['global_states']).to(self.device)
        
        return prepared
    
    def _create_mini_batches(self, data: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Create mini-batches for training.
        
        Args:
            data: Prepared training data
            
        Returns:
            List of mini-batch dictionaries
        """
        # Get batch size from first agent
        first_agent = self.agent_names[0]
        batch_size = len(data[first_agent]['actions'])
        
        # Create random indices
        indices = np.random.permutation(batch_size)
        
        # Split into mini-batches
        mini_batches = []
        for start_idx in range(0, batch_size, self.mini_batch_size):
            end_idx = min(start_idx + self.mini_batch_size, batch_size)
            batch_indices = indices[start_idx:end_idx]
            
            batch = {}
            
            # Add agent-specific data
            for agent_name in self.agent_names:
                agent_batch = {}
                for key, tensor in data[agent_name].items():
                    agent_batch[key] = tensor[batch_indices]
                batch[agent_name] = agent_batch
            
            # Add global states
            batch['global_states'] = data['global_states'][batch_indices]
            
            mini_batches.append(batch)
        
        return mini_batches
    
    def _update_agent(self, agent_name: str, batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """
        Update a single agent's policy.
        
        Args:
            agent_name: Name of agent to update
            batch: Mini-batch of data
            
        Returns:
            Update metrics
        """
        agent = self.agents[agent_name]
        optimizer = self.actor_optimizers[agent_name]
        agent_batch = batch[agent_name]
        
        # Get current policy outputs
        # Note: This is simplified - need to properly reconstruct observations
        current_output = agent(
            torch.zeros(len(agent_batch['actions']), 100, 8).to(self.device),  # Placeholder
            torch.zeros(len(agent_batch['actions']), 8).to(self.device),      # Placeholder
            {}  # Placeholder synergy context
        )
        
        # Get action distribution
        action_logits = current_output['action']
        dist = torch.distributions.Categorical(logits=action_logits)
        
        # Get old actions (just the discrete part for now)
        old_actions = agent_batch['actions'][:, 0].long()
        
        # Calculate log probabilities
        new_log_probs = dist.log_prob(old_actions)
        old_log_probs = agent_batch['log_probs']
        
        # Calculate ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Calculate clipped surrogate loss
        advantages = agent_batch['advantages']
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Calculate entropy bonus
        entropy = dist.entropy().mean()
        
        # Total loss
        total_loss = policy_loss - self.entropy_coef * entropy
        
        # Update
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(agent.parameters(), self.max_grad_norm)
        
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            kl_div = (old_log_probs - new_log_probs).mean().item()
            clip_fraction = ((ratio - 1).abs() > self.clip_epsilon).float().mean().item()
        
        return {
            'policy_loss': policy_loss.item(),
            'entropy': entropy.item(),
            'kl_divergence': kl_div,
            'clip_fraction': clip_fraction,
            'total_loss': total_loss.item()
        }
    
    def _update_critic(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Update centralized critic.
        
        Args:
            batch: Mini-batch of data
            
        Returns:
            Update metrics
        """
        optimizer = self.critic_optimizer
        
        # Get global states and returns
        global_states = batch['global_states']
        
        # Aggregate returns from all agents (using average for now)
        all_returns = []
        for agent_name in self.agent_names:
            all_returns.append(batch[agent_name]['returns'])
        returns = torch.stack(all_returns).mean(dim=0)
        
        # Get value predictions
        values = self.centralized_critic(global_states).squeeze()
        
        # Calculate value loss
        value_loss = F.mse_loss(values, returns)
        
        # Update
        optimizer.zero_grad()
        value_loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(self.centralized_critic.parameters(), self.max_grad_norm)
        
        optimizer.step()
        
        # Calculate explained variance
        with torch.no_grad():
            var_returns = returns.var()
            explained_var = 1 - (returns - values).var() / (var_returns + 1e-8)
        
        return {
            'value_loss': value_loss.item(),
            'explained_variance': explained_var.item()
        }
    
    def save_checkpoint(self, path: Path, episode: int):
        """
        Save training checkpoint.
        
        Args:
            path: Checkpoint file path
            episode: Current episode number
        """
        checkpoint = {
            'episode': episode,
            'config': self.config,
            'centralized_critic': self.centralized_critic.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'metrics_history': self.metrics_history
        }
        
        # Save agent states
        for agent_name, agent in self.agents.items():
            checkpoint[f'agent_{agent_name}'] = agent.state_dict()
            checkpoint[f'optimizer_{agent_name}'] = self.actor_optimizers[agent_name].state_dict()
        
        # Save schedulers if present
        if self.actor_schedulers:
            for agent_name, scheduler in self.actor_schedulers.items():
                if scheduler is not None:
                    checkpoint[f'scheduler_{agent_name}'] = scheduler.state_dict()
        
        if self.critic_scheduler is not None:
            checkpoint['critic_scheduler'] = self.critic_scheduler.state_dict()
        
        torch.save(checkpoint, path)
        
        logger.info(f"Saved checkpoint path={str(path}"), episode=episode)
    
    def load_checkpoint(self, path: Path):
        """
        Load training checkpoint.
        
        Args:
            path: Checkpoint file path
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load critic
        self.centralized_critic.load_state_dict(checkpoint['centralized_critic'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        # Load agents
        for agent_name in self.agent_names:
            self.agents[agent_name].load_state_dict(checkpoint[f'agent_{agent_name}'])
            self.actor_optimizers[agent_name].load_state_dict(checkpoint[f'optimizer_{agent_name}'])
        
        # Load schedulers
        if self.actor_schedulers:
            for agent_name in self.agent_names:
                scheduler_key = f'scheduler_{agent_name}'
                if scheduler_key in checkpoint and self.actor_schedulers[agent_name] is not None:
                    self.actor_schedulers[agent_name].load_state_dict(checkpoint[scheduler_key])
        
        if 'critic_scheduler' in checkpoint and self.critic_scheduler is not None:
            self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler'])
        
        # Load metrics
        if 'metrics_history' in checkpoint:
            self.metrics_history = checkpoint['metrics_history']
        
        episode = checkpoint.get('episode', 0)
        
        logger.info(f"Loaded checkpoint path={str(path}"), episode=episode)