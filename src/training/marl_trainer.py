"""Multi-Agent PPO (MAPPO) Trainer for MARL System.

This module implements the centralized training with decentralized execution
paradigm using MAPPO algorithm for training our trading agents.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from pathlib import Path
import logging
from datetime import datetime
import json

from src.training.environment import MultiAgentTradingEnv
from src.training.experience import ExperienceBuffer, Trajectory
from src.agents.base.marl_agent import BaseMARLAgent


logger = logging.getLogger(__name__)


class MAPPOTrainer:
    """Multi-Agent PPO Trainer for centralized training."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the MAPPO trainer.
        
        Args:
            config: Training configuration including:
                - env_config: Environment configuration
                - agent_configs: Configuration for each agent type
                - learning_rate: Learning rate for optimization
                - gamma: Discount factor
                - gae_lambda: GAE lambda for advantage estimation
                - ppo_epochs: Number of PPO epochs per update
                - clip_param: PPO clip parameter
                - value_loss_coef: Value loss coefficient
                - entropy_coef: Entropy coefficient
                - max_grad_norm: Maximum gradient norm for clipping
                - batch_size: Batch size for updates
                - device: Training device (cuda/cpu)
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Training hyperparameters
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.ppo_epochs = config.get('ppo_epochs', 10)
        self.clip_param = config.get('clip_param', 0.2)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.batch_size = config.get('batch_size', 64)
        
        # Initialize environment
        self.env = MultiAgentTradingEnv(config['env_config'])
        self.n_agents = self.env.n_agents
        self.agent_names = self.env.agents
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Initialize optimizers
        self.optimizers = self._initialize_optimizers()
        
        # Experience buffer
        buffer_size = config.get('buffer_size', 10000)
        self.experience_buffer = ExperienceBuffer(buffer_size, self.n_agents)
        
        # Logging
        self.log_dir = Path(config.get('log_dir', 'logs/mappo'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir / datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        # Training state
        self.total_steps = 0
        self.episode_count = 0
        self.best_reward = -float('inf')
        
        logger.info(f"Initialized MAPPO trainer with {self.n_agents} agents on {self.device}")
    
    def _initialize_agents(self) -> Dict[str, BaseMARLAgent]:
        """Initialize all agents.
        
        Returns:
            Dictionary of initialized agents
        """
        agents = {}
        
        for agent_name in self.agent_names:
            agent_config = self.config['agent_configs'].get(agent_name, {})
            
            # Create agent based on type
            if agent_name == 'regime':
                from src.agents.regime.agent import RegimeAgent
                agent = RegimeAgent(agent_config)
            elif agent_name == 'structure':
                from src.agents.structure.agent import StructureAgent
                agent = StructureAgent(agent_config)
            elif agent_name == 'tactical':
                from src.agents.tactical.agent import TacticalAgent
                agent = TacticalAgent(agent_config)
            elif agent_name == 'risk':
                from src.agents.risk.agent import RiskAgent
                agent = RiskAgent(agent_config)
            else:
                raise ValueError(f"Unknown agent type: {agent_name}")
            
            agent.to(self.device)
            agents[agent_name] = agent
            
        return agents
    
    def _initialize_optimizers(self) -> Dict[str, optim.Optimizer]:
        """Initialize optimizers for each agent.
        
        Returns:
            Dictionary of optimizers
        """
        optimizers = {}
        
        for agent_name, agent in self.agents.items():
            optimizer = optim.Adam(
                agent.parameters(),
                lr=self.learning_rate,
                eps=1e-5
            )
            optimizers[agent_name] = optimizer
            
        return optimizers
    
    def train(self, n_episodes: int, save_freq: int = 100):
        """Train the multi-agent system.
        
        Args:
            n_episodes: Number of episodes to train
            save_freq: Frequency of model saving
        """
        logger.info(f"Starting MAPPO training for {n_episodes} episodes")
        
        for episode in range(n_episodes):
            # Collect trajectories
            trajectories = self.collect_trajectories()
            
            # Update all agents
            update_info = self.update_agents(trajectories)
            
            # Log progress
            self._log_training_progress(episode, trajectories, update_info)
            
            # Save models periodically
            if (episode + 1) % save_freq == 0:
                self.save_models(episode)
            
            self.episode_count += 1
        
        # Final save
        self.save_models(n_episodes, is_final=True)
        logger.info("Training completed")
    
    def collect_trajectories(self) -> List[Trajectory]:
        """Collect trajectories by running episodes.
        
        Returns:
            List of trajectories for all agents
        """
        trajectories = []
        observations = self.env.reset()
        
        episode_data = defaultdict(list)
        episode_rewards = defaultdict(float)
        
        done = False
        while not done:
            # Get actions from all agents
            actions = {}
            action_log_probs = {}
            values = {}
            
            with torch.no_grad():
                for agent_name, agent in self.agents.items():
                    obs_tensor = self._prepare_observation(observations[agent_name], agent_name)
                    
                    # Get action, log_prob, and value
                    action, log_prob, value = agent.get_action_and_value(obs_tensor)
                    
                    actions[agent_name] = action.cpu().numpy()
                    action_log_probs[agent_name] = log_prob.cpu()
                    values[agent_name] = value.cpu()
            
            # Execute actions in environment
            next_observations, rewards, done, info = self.env.step(actions)
            
            # Store transition data
            for agent_name in self.agent_names:
                episode_data[agent_name].append({
                    'observation': observations[agent_name],
                    'action': actions[agent_name],
                    'log_prob': action_log_probs[agent_name],
                    'value': values[agent_name],
                    'reward': rewards[agent_name],
                    'done': done
                })
                episode_rewards[agent_name] += rewards[agent_name]
            
            observations = next_observations
            self.total_steps += 1
        
        # Process episode data into trajectories
        for agent_name in self.agent_names:
            trajectory = self._process_episode_data(episode_data[agent_name], agent_name)
            trajectories.append(trajectory)
            
            # Add to experience buffer
            self.experience_buffer.add_trajectory(trajectory)
        
        return trajectories
    
    def _process_episode_data(self, episode_data: List[Dict], agent_name: str) -> Trajectory:
        """Process episode data into a trajectory with advantages.
        
        Args:
            episode_data: Raw episode data
            agent_name: Name of the agent
            
        Returns:
            Processed trajectory with advantages
        """
        # Extract data
        observations = [step['observation'] for step in episode_data]
        actions = [step['action'] for step in episode_data]
        log_probs = torch.stack([step['log_prob'] for step in episode_data])
        values = torch.stack([step['value'] for step in episode_data])
        rewards = torch.tensor([step['reward'] for step in episode_data], dtype=torch.float32)
        dones = torch.tensor([step['done'] for step in episode_data], dtype=torch.float32)
        
        # Compute advantages using GAE
        advantages = self._compute_gae(rewards, values, dones)
        
        # Compute returns
        returns = advantages + values
        
        return Trajectory(
            agent_name=agent_name,
            observations=observations,
            actions=actions,
            log_probs=log_probs,
            values=values,
            rewards=rewards,
            advantages=advantages,
            returns=returns
        )
    
    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, 
                     dones: torch.Tensor) -> torch.Tensor:
        """Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Episode rewards
            values: Value estimates
            dones: Done flags
            
        Returns:
            Computed advantages
        """
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
        
        return advantages
    
    def update_agents(self, trajectories: List[Trajectory]) -> Dict[str, Dict[str, float]]:
        """Update all agents using collected trajectories.
        
        Args:
            trajectories: List of trajectories for all agents
            
        Returns:
            Update information for logging
        """
        update_info = defaultdict(dict)
        
        # Update each agent
        for trajectory in trajectories:
            agent_name = trajectory.agent_name
            agent = self.agents[agent_name]
            optimizer = self.optimizers[agent_name]
            
            # Convert trajectory to tensors
            obs_batch = self._prepare_observation_batch(trajectory.observations, agent_name)
            action_batch = torch.tensor(np.array(trajectory.actions), dtype=torch.float32).to(self.device)
            old_log_probs = trajectory.log_probs.to(self.device)
            returns = trajectory.returns.to(self.device)
            advantages = trajectory.advantages.to(self.device)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO update
            for epoch in range(self.ppo_epochs):
                # Get current policy outputs
                _, log_probs, values, entropy = agent.evaluate_actions(obs_batch, action_batch)
                
                # Calculate ratio
                ratio = torch.exp(log_probs - old_log_probs)
                
                # Clipped surrogate loss
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), returns)
                
                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), self.max_grad_norm)
                optimizer.step()
                
                # Store update info
                if epoch == self.ppo_epochs - 1:
                    update_info[agent_name]['policy_loss'] = policy_loss.item()
                    update_info[agent_name]['value_loss'] = value_loss.item()
                    update_info[agent_name]['entropy'] = entropy.mean().item()
                    update_info[agent_name]['total_loss'] = loss.item()
        
        return update_info
    
    def _prepare_observation(self, observation: Any, agent_name: str) -> torch.Tensor:
        """Prepare observation tensor for agent.
        
        Args:
            observation: Raw observation from environment
            agent_name: Name of the agent
            
        Returns:
            Prepared observation tensor
        """
        if agent_name == 'risk':
            # Risk agent has composite observation
            matrices = torch.tensor(observation['matrices'], dtype=torch.float32)
            portfolio = torch.tensor(observation['portfolio'], dtype=torch.float32)
            obs_tensor = torch.cat([matrices.flatten(), portfolio], dim=0)
        else:
            # Other agents have matrix observations
            obs_tensor = torch.tensor(observation, dtype=torch.float32)
        
        return obs_tensor.unsqueeze(0).to(self.device)
    
    def _prepare_observation_batch(self, observations: List[Any], agent_name: str) -> torch.Tensor:
        """Prepare batch of observations for agent.
        
        Args:
            observations: List of observations
            agent_name: Name of the agent
            
        Returns:
            Batch of observation tensors
        """
        processed_obs = []
        
        for obs in observations:
            if agent_name == 'risk':
                matrices = torch.tensor(obs['matrices'], dtype=torch.float32)
                portfolio = torch.tensor(obs['portfolio'], dtype=torch.float32)
                obs_tensor = torch.cat([matrices.flatten(), portfolio], dim=0)
            else:
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
            
            processed_obs.append(obs_tensor)
        
        return torch.stack(processed_obs).to(self.device)
    
    def _log_training_progress(self, episode: int, trajectories: List[Trajectory], 
                              update_info: Dict[str, Dict[str, float]]):
        """Log training progress to tensorboard.
        
        Args:
            episode: Current episode number
            trajectories: Episode trajectories
            update_info: Update information from training
        """
        # Calculate episode statistics
        episode_rewards = {}
        episode_lengths = {}
        
        for trajectory in trajectories:
            agent_name = trajectory.agent_name
            episode_rewards[agent_name] = trajectory.rewards.sum().item()
            episode_lengths[agent_name] = len(trajectory.rewards)
        
        # Log to tensorboard
        for agent_name in self.agent_names:
            # Rewards
            self.writer.add_scalar(f'{agent_name}/episode_reward', 
                                 episode_rewards.get(agent_name, 0), episode)
            
            # Losses
            if agent_name in update_info:
                for loss_name, loss_value in update_info[agent_name].items():
                    self.writer.add_scalar(f'{agent_name}/{loss_name}', loss_value, episode)
        
        # Log overall metrics
        total_reward = sum(episode_rewards.values())
        self.writer.add_scalar('overall/total_reward', total_reward, episode)
        self.writer.add_scalar('overall/episode_length', 
                             max(episode_lengths.values()), episode)
        
        # Update best reward
        if total_reward > self.best_reward:
            self.best_reward = total_reward
            self.save_models(episode, is_best=True)
        
        # Console logging
        if episode % 10 == 0:
            logger.info(
                f"Episode {episode}: "
                f"Total Reward: {total_reward:.2f}, "
                f"Best: {self.best_reward:.2f}, "
                f"Steps: {self.total_steps}"
            )
    
    def save_models(self, episode: int, is_best: bool = False, is_final: bool = False):
        """Save model checkpoints.
        
        Args:
            episode: Current episode number
            is_best: Whether this is the best model so far
            is_final: Whether this is the final save
        """
        save_dir = self.log_dir / 'checkpoints'
        save_dir.mkdir(exist_ok=True)
        
        # Determine save path
        if is_best:
            save_path = save_dir / 'best_models.pt'
        elif is_final:
            save_path = save_dir / 'final_models.pt'
        else:
            save_path = save_dir / f'models_ep{episode}.pt'
        
        # Save all agent models
        checkpoint = {
            'episode': episode,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
            'config': self.config,
            'models': {},
            'optimizers': {}
        }
        
        for agent_name, agent in self.agents.items():
            checkpoint['models'][agent_name] = agent.state_dict()
            checkpoint['optimizers'][agent_name] = self.optimizers[agent_name].state_dict()
        
        torch.save(checkpoint, save_path)
        logger.info(f"Saved models to {save_path}")
    
    def load_models(self, checkpoint_path: str):
        """Load model checkpoints.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load agent models
        for agent_name, agent in self.agents.items():
            if agent_name in checkpoint['models']:
                agent.load_state_dict(checkpoint['models'][agent_name])
                
        # Load optimizers
        for agent_name, optimizer in self.optimizers.items():
            if agent_name in checkpoint['optimizers']:
                optimizer.load_state_dict(checkpoint['optimizers'][agent_name])
        
        # Restore training state
        self.episode_count = checkpoint.get('episode', 0)
        self.total_steps = checkpoint.get('total_steps', 0)
        self.best_reward = checkpoint.get('best_reward', -float('inf'))
        
        logger.info(f"Loaded models from {checkpoint_path}")
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the trained agents.
        
        Args:
            n_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating agents for {n_episodes} episodes")
        
        # Set agents to eval mode
        for agent in self.agents.values():
            agent.eval()
        
        eval_rewards = defaultdict(list)
        eval_lengths = []
        
        for episode in range(n_episodes):
            observations = self.env.reset()
            episode_rewards = defaultdict(float)
            episode_length = 0
            done = False
            
            while not done:
                actions = {}
                
                with torch.no_grad():
                    for agent_name, agent in self.agents.items():
                        obs_tensor = self._prepare_observation(observations[agent_name], agent_name)
                        action, _, _ = agent.get_action_and_value(obs_tensor)
                        actions[agent_name] = action.cpu().numpy()
                
                observations, rewards, done, _ = self.env.step(actions)
                
                for agent_name in self.agent_names:
                    episode_rewards[agent_name] += rewards[agent_name]
                
                episode_length += 1
            
            # Store episode results
            for agent_name in self.agent_names:
                eval_rewards[agent_name].append(episode_rewards[agent_name])
            eval_lengths.append(episode_length)
        
        # Set agents back to train mode
        for agent in self.agents.values():
            agent.train()
        
        # Calculate statistics
        eval_metrics = {
            'avg_total_reward': np.mean([sum(r.values()) for r in 
                                        [dict(zip(self.agent_names, rewards)) 
                                         for rewards in zip(*eval_rewards.values())]]),
            'avg_episode_length': np.mean(eval_lengths)
        }
        
        for agent_name in self.agent_names:
            eval_metrics[f'{agent_name}_avg_reward'] = np.mean(eval_rewards[agent_name])
            eval_metrics[f'{agent_name}_std_reward'] = np.std(eval_rewards[agent_name])
        
        logger.info(f"Evaluation complete: {eval_metrics}")
        return eval_metrics