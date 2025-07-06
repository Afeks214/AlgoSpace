"""
Base trainer class for MARL algorithms.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np

import structlog

logger = structlog.get_logger()


class BaseTrainer(ABC):
    """
    Abstract base class for MARL trainers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base trainer.
        
        Args:
            config: Trainer configuration
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Training parameters
        self.batch_size = config.get('batch_size', 512)
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.epochs = config.get('epochs', 1000)
        
        # Logging and checkpointing
        self.log_interval = config.get('log_interval', 10)
        self.save_interval = config.get('save_interval', 50)
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics tracking
        self.episode_rewards = []
        self.training_losses = []
        self.metrics_history = {}
        
        logger.info(f"Initialized base trainer device={str(self.device}")
            batch_size=self.batch_size,
            learning_rate=self.learning_rate
        )
    
    @abstractmethod
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Execute one training step.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Dictionary of losses and metrics
        """
        pass
    
    @abstractmethod
    def collect_experience(self, env, num_steps: int) -> Dict[str, Any]:
        """
        Collect experience from environment.
        
        Args:
            env: Training environment
            num_steps: Number of steps to collect
            
        Returns:
            Dictionary of collected experience
        """
        pass
    
    @abstractmethod
    def update_policies(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """
        Update agent policies based on experience.
        
        Args:
            experience: Collected experience
            
        Returns:
            Dictionary of update metrics
        """
        pass
    
    def train(self, env, num_episodes: int = 1000):
        """
        Main training loop.
        
        Args:
            env: Training environment
            num_episodes: Number of episodes to train
        """
        logger.info(f"Starting training num_episodes={num_episodes}")
        
        for episode in range(num_episodes):
            # Collect experience
            experience = self.collect_experience(env, self.config.get('steps_per_episode', 1000))
            
            # Update policies
            metrics = self.update_policies(experience)
            
            # Track metrics
            self._track_metrics(episode, metrics)
            
            # Logging
            if episode % self.log_interval == 0:
                self._log_progress(episode, metrics)
            
            # Checkpointing
            if episode % self.save_interval == 0:
                self._save_checkpoint(episode)
        
        logger.info("Training completed")
    
    def _track_metrics(self, episode: int, metrics: Dict[str, float]):
        """Track training metrics."""
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append((episode, value))
    
    def _log_progress(self, episode: int, metrics: Dict[str, float]):
        """Log training progress."""
        logger.info(f"Training progress episode={episode} {**metrics}")
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_ep{episode}.pt"
        self.save_checkpoint(checkpoint_path, episode)
        logger.info(f"Saved checkpoint path={str(checkpoint_path}"))
    
    @abstractmethod
    def save_checkpoint(self, path: Path, episode: int):
        """Save model checkpoint."""
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        pass
    
    def compute_returns(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute discounted returns.
        
        Args:
            rewards: Reward tensor [batch, time]
            dones: Done flags [batch, time]
            next_value: Bootstrap value [batch]
            
        Returns:
            Returns tensor [batch, time]
        """
        returns = torch.zeros_like(rewards)
        running_return = next_value
        
        for t in reversed(range(rewards.size(1))):
            running_return = rewards[:, t] + self.gamma * running_return * (1 - dones[:, t])
            returns[:, t] = running_return
        
        return returns
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
        gae_lambda: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Reward tensor [batch, time]
            values: Value estimates [batch, time]
            dones: Done flags [batch, time]
            next_value: Bootstrap value [batch]
            gae_lambda: GAE lambda parameter
            
        Returns:
            advantages: Advantage estimates [batch, time]
            returns: Return targets [batch, time]
        """
        advantages = torch.zeros_like(rewards)
        running_gae = 0
        
        for t in reversed(range(rewards.size(1))):
            if t == rewards.size(1) - 1:
                next_val = next_value
            else:
                next_val = values[:, t + 1]
            
            delta = rewards[:, t] + self.gamma * next_val * (1 - dones[:, t]) - values[:, t]
            running_gae = delta + self.gamma * gae_lambda * running_gae * (1 - dones[:, t])
            advantages[:, t] = running_gae
        
        returns = advantages + values
        
        return advantages, returns
    
    def normalize_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        """
        Normalize advantages for stable training.
        
        Args:
            advantages: Advantage tensor
            
        Returns:
            Normalized advantages
        """
        mean = advantages.mean()
        std = advantages.std() + 1e-8
        return (advantages - mean) / std