"""
Experience buffer for MARL training.

Manages collection, storage, and sampling of agent experiences
for efficient training.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from collections import deque, defaultdict
import random

import structlog

logger = structlog.get_logger()


class ExperienceBuffer:
    """
    Experience buffer for storing and sampling agent trajectories.
    """
    
    def __init__(self, capacity: int, agent_names: List[str]):
        """
        Initialize experience buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            agent_names: List of agent names
        """
        self.capacity = capacity
        self.agent_names = agent_names
        
        # Storage for each agent
        self.buffers = {
            agent: {
                'observations': deque(maxlen=capacity),
                'actions': deque(maxlen=capacity),
                'rewards': deque(maxlen=capacity),
                'next_observations': deque(maxlen=capacity),
                'dones': deque(maxlen=capacity),
                'log_probs': deque(maxlen=capacity),
                'values': deque(maxlen=capacity)
            }
            for agent in agent_names
        }
        
        # Global state buffer
        self.global_states = deque(maxlen=capacity)
        self.next_global_states = deque(maxlen=capacity)
        
        # Episode boundaries
        self.episode_starts = []
        self.current_size = 0
        
        logger.info(f"Initialized experience buffer capacity={capacity} agents={agent_names}")
    
    def add_transition(
        self,
        observations: Dict[str, Any],
        actions: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        next_observations: Dict[str, Any],
        dones: Dict[str, bool],
        log_probs: Optional[Dict[str, float]] = None,
        values: Optional[Dict[str, float]] = None,
        global_state: Optional[np.ndarray] = None,
        next_global_state: Optional[np.ndarray] = None
    ):
        """
        Add a transition to the buffer.
        
        Args:
            observations: Current observations for each agent
            actions: Actions taken by each agent
            rewards: Rewards received by each agent
            next_observations: Next observations for each agent
            dones: Done flags for each agent
            log_probs: Action log probabilities
            values: Value estimates
            global_state: Global state representation
            next_global_state: Next global state
        """
        for agent in self.agent_names:
            self.buffers[agent]['observations'].append(observations[agent])
            self.buffers[agent]['actions'].append(actions[agent])
            self.buffers[agent]['rewards'].append(rewards[agent])
            self.buffers[agent]['next_observations'].append(next_observations[agent])
            self.buffers[agent]['dones'].append(dones[agent])
            
            if log_probs:
                self.buffers[agent]['log_probs'].append(log_probs.get(agent, 0.0))
            if values:
                self.buffers[agent]['values'].append(values.get(agent, 0.0))
        
        if global_state is not None:
            self.global_states.append(global_state)
        if next_global_state is not None:
            self.next_global_states.append(next_global_state)
        
        self.current_size = min(self.current_size + 1, self.capacity)
        
        # Track episode boundaries
        if any(dones.values()):
            self.episode_starts.append(self.current_size)
    
    def add_trajectory(
        self,
        trajectory: Dict[str, List[Any]]
    ):
        """
        Add a complete trajectory to the buffer.
        
        Args:
            trajectory: Dictionary containing lists of transitions
        """
        num_steps = len(trajectory['rewards'][self.agent_names[0]])
        
        for step in range(num_steps):
            transition_data = {
                'observations': {},
                'actions': {},
                'rewards': {},
                'next_observations': {},
                'dones': {},
                'log_probs': {},
                'values': {}
            }
            
            # Extract step data for each agent
            for agent in self.agent_names:
                for key in ['observations', 'actions', 'rewards', 'dones']:
                    if key in trajectory and agent in trajectory[key]:
                        transition_data[key][agent] = trajectory[key][agent][step]
                
                # Next observations
                if step < num_steps - 1:
                    transition_data['next_observations'][agent] = trajectory['observations'][agent][step + 1]
                else:
                    transition_data['next_observations'][agent] = trajectory['observations'][agent][step]
                
                # Optional data
                if 'log_probs' in trajectory and agent in trajectory['log_probs']:
                    transition_data['log_probs'][agent] = trajectory['log_probs'][agent][step]
                if 'values' in trajectory and agent in trajectory['values']:
                    transition_data['values'][agent] = trajectory['values'][agent][step]
            
            # Global states
            global_state = None
            next_global_state = None
            if 'global_states' in trajectory:
                global_state = trajectory['global_states'][step]
                if step < num_steps - 1:
                    next_global_state = trajectory['global_states'][step + 1]
                else:
                    next_global_state = global_state
            
            self.add_transition(
                **transition_data,
                global_state=global_state,
                next_global_state=next_global_state
            )
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Batch of transitions as tensors
        """
        if self.current_size < batch_size:
            raise ValueError(f"Not enough samples. Have {self.current_size}, need {batch_size}")
        
        # Sample indices
        indices = random.sample(range(self.current_size), batch_size)
        
        batch = defaultdict(dict)
        
        # Sample for each agent
        for agent in self.agent_names:
            for key in ['actions', 'rewards', 'dones', 'log_probs', 'values']:
                if len(self.buffers[agent][key]) > 0:
                    data = [self.buffers[agent][key][i] for i in indices]
                    batch[agent][key] = torch.FloatTensor(data)
            
            # Handle observations (might be dictionaries)
            obs_list = [self.buffers[agent]['observations'][i] for i in indices]
            next_obs_list = [self.buffers[agent]['next_observations'][i] for i in indices]
            
            # Convert observations to tensors (simplified - assumes numpy arrays)
            if isinstance(obs_list[0], np.ndarray):
                batch[agent]['observations'] = torch.FloatTensor(np.stack(obs_list))
                batch[agent]['next_observations'] = torch.FloatTensor(np.stack(next_obs_list))
            else:
                # Handle dictionary observations
                batch[agent]['observations'] = obs_list
                batch[agent]['next_observations'] = next_obs_list
        
        # Sample global states if available
        if len(self.global_states) > 0:
            batch['global_states'] = torch.FloatTensor(
                np.stack([self.global_states[i] for i in indices])
            )
            batch['next_global_states'] = torch.FloatTensor(
                np.stack([self.next_global_states[i] for i in indices])
            )
        
        return dict(batch)
    
    def sample_trajectories(self, num_trajectories: int) -> List[Dict[str, Any]]:
        """
        Sample complete trajectories from the buffer.
        
        Args:
            num_trajectories: Number of trajectories to sample
            
        Returns:
            List of trajectory dictionaries
        """
        if len(self.episode_starts) < num_trajectories:
            raise ValueError(f"Not enough episodes. Have {len(self.episode_starts)}, need {num_trajectories}")
        
        trajectories = []
        
        # Sample episode start indices
        sampled_episodes = random.sample(range(len(self.episode_starts)), num_trajectories)
        
        for ep_idx in sampled_episodes:
            start_idx = 0 if ep_idx == 0 else self.episode_starts[ep_idx - 1]
            end_idx = self.episode_starts[ep_idx]
            
            trajectory = defaultdict(dict)
            
            # Extract trajectory for each agent
            for agent in self.agent_names:
                for key in self.buffers[agent]:
                    trajectory[agent][key] = list(self.buffers[agent][key])[start_idx:end_idx]
            
            # Extract global states
            if len(self.global_states) > 0:
                trajectory['global_states'] = list(self.global_states)[start_idx:end_idx]
            
            trajectories.append(dict(trajectory))
        
        return trajectories
    
    def clear(self):
        """Clear the buffer."""
        for agent in self.agent_names:
            for key in self.buffers[agent]:
                self.buffers[agent][key].clear()
        
        self.global_states.clear()
        self.next_global_states.clear()
        self.episode_starts.clear()
        self.current_size = 0
        
        logger.info("Cleared experience buffer")
    
    def __len__(self):
        """Get current buffer size."""
        return self.current_size


class PrioritizedExperienceBuffer(ExperienceBuffer):
    """
    Prioritized experience replay buffer for MARL.
    
    Samples transitions based on their TD error magnitude.
    """
    
    def __init__(
        self,
        capacity: int,
        agent_names: List[str],
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001
    ):
        """
        Initialize prioritized buffer.
        
        Args:
            capacity: Maximum buffer size
            agent_names: List of agent names
            alpha: Priority exponent
            beta: Importance sampling exponent
            beta_increment: Beta increment per sampling
        """
        super().__init__(capacity, agent_names)
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        
        # Priority storage
        self.priorities = deque(maxlen=capacity)
        
    def add_transition(self, **kwargs):
        """Add transition with maximum priority."""
        super().add_transition(**kwargs)
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """
        Sample batch with prioritized replay.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Batch with importance weights
        """
        if self.current_size < batch_size:
            raise ValueError(f"Not enough samples. Have {self.current_size}, need {batch_size}")
        
        # Calculate sampling probabilities
        priorities = np.array(list(self.priorities))[:self.current_size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(self.current_size, batch_size, p=probs)
        
        # Calculate importance weights
        weights = (self.current_size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Get batch using parent method
        batch = self._sample_at_indices(indices)
        batch['importance_weights'] = torch.FloatTensor(weights)
        batch['indices'] = indices
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return batch
    
    def _sample_at_indices(self, indices: np.ndarray) -> Dict[str, torch.Tensor]:
        """Sample specific indices from buffer."""
        batch = defaultdict(dict)
        
        # Sample for each agent
        for agent in self.agent_names:
            for key in ['actions', 'rewards', 'dones', 'log_probs', 'values']:
                if len(self.buffers[agent][key]) > 0:
                    data = [self.buffers[agent][key][i] for i in indices]
                    batch[agent][key] = torch.FloatTensor(data)
            
            # Handle observations
            obs_list = [self.buffers[agent]['observations'][i] for i in indices]
            next_obs_list = [self.buffers[agent]['next_observations'][i] for i in indices]
            
            if isinstance(obs_list[0], np.ndarray):
                batch[agent]['observations'] = torch.FloatTensor(np.stack(obs_list))
                batch[agent]['next_observations'] = torch.FloatTensor(np.stack(next_obs_list))
            else:
                batch[agent]['observations'] = obs_list
                batch[agent]['next_observations'] = next_obs_list
        
        # Sample global states if available
        if len(self.global_states) > 0:
            batch['global_states'] = torch.FloatTensor(
                np.stack([self.global_states[i] for i in indices])
            )
            batch['next_global_states'] = torch.FloatTensor(
                np.stack([self.next_global_states[i] for i in indices])
            )
        
        return dict(batch)
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on TD errors.
        
        Args:
            indices: Indices to update
            td_errors: TD errors for priority calculation
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)