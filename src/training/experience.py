"""Experience Buffer and Trajectory Management for MARL Training.

This module implements experience replay buffers and trajectory collection
for efficient multi-agent reinforcement learning.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from collections import deque, defaultdict
import random
from dataclasses import dataclass
import logging
import pickle
from pathlib import Path


logger = logging.getLogger(__name__)


class Trajectory(NamedTuple):
    """Container for agent trajectory data."""
    agent_name: str
    observations: List[np.ndarray]
    actions: List[np.ndarray]
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    
    def __len__(self) -> int:
        """Get trajectory length."""
        return len(self.observations)


@dataclass
class Transition:
    """Single transition in the replay buffer."""
    observation: Dict[str, np.ndarray]
    action: Dict[str, np.ndarray]
    reward: Dict[str, float]
    next_observation: Dict[str, np.ndarray]
    done: bool
    info: Dict[str, Any]


class ExperienceBuffer:
    """Experience replay buffer for multi-agent training."""
    
    def __init__(self, capacity: int, n_agents: int, prioritized: bool = False):
        """Initialize experience buffer.
        
        Args:
            capacity: Maximum buffer size
            n_agents: Number of agents
            prioritized: Whether to use prioritized replay
        """
        self.capacity = capacity
        self.n_agents = n_agents
        self.prioritized = prioritized
        
        # Storage
        self.buffer = deque(maxlen=capacity)
        self.trajectories = defaultdict(list)
        
        # Priority replay components
        if self.prioritized:
            self.priorities = deque(maxlen=capacity)
            self.priority_alpha = 0.6
            self.priority_beta = 0.4
            self.priority_beta_increment = 0.001
            self.max_priority = 1.0
        
        # Statistics
        self.total_transitions = 0
        self.total_trajectories = 0
        
        logger.info(f"Initialized ExperienceBuffer with capacity {capacity}")
    
    def add_transition(self, transition: Transition):
        """Add a single transition to the buffer.
        
        Args:
            transition: Transition to add
        """
        self.buffer.append(transition)
        
        if self.prioritized:
            self.priorities.append(self.max_priority)
        
        self.total_transitions += 1
    
    def add_trajectory(self, trajectory: Trajectory):
        """Add a complete trajectory to the buffer.
        
        Args:
            trajectory: Agent trajectory to add
        """
        self.trajectories[trajectory.agent_name].append(trajectory)
        
        # Convert trajectory to transitions
        for i in range(len(trajectory) - 1):
            transition = Transition(
                observation={trajectory.agent_name: trajectory.observations[i]},
                action={trajectory.agent_name: trajectory.actions[i]},
                reward={trajectory.agent_name: trajectory.rewards[i].item()},
                next_observation={trajectory.agent_name: trajectory.observations[i + 1]},
                done=i == len(trajectory) - 2,
                info={'trajectory_id': self.total_trajectories}
            )
            self.add_transition(transition)
        
        self.total_trajectories += 1
        
        # Maintain trajectory buffer size
        if len(self.trajectories[trajectory.agent_name]) > self.capacity // self.n_agents:
            self.trajectories[trajectory.agent_name].pop(0)
    
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            List of sampled transitions
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        if self.prioritized:
            return self._prioritized_sample(batch_size)
        else:
            return random.sample(self.buffer, batch_size)
    
    def _prioritized_sample(self, batch_size: int) -> List[Transition]:
        """Sample using prioritized experience replay.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            List of sampled transitions with importance weights
        """
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.priority_alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.priority_beta)
        weights /= weights.max()
        
        # Update beta
        self.priority_beta = min(1.0, self.priority_beta + self.priority_beta_increment)
        
        # Get transitions
        transitions = [self.buffer[idx] for idx in indices]
        
        # Attach weights to transitions
        for trans, weight in zip(transitions, weights):
            trans.info['importance_weight'] = weight
        
        return transitions
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for prioritized replay.
        
        Args:
            indices: Indices of transitions to update
            priorities: New priority values
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def sample_trajectories(self, n_trajectories: int, 
                          agent_name: Optional[str] = None) -> List[Trajectory]:
        """Sample complete trajectories.
        
        Args:
            n_trajectories: Number of trajectories to sample
            agent_name: Specific agent to sample from (None for all)
            
        Returns:
            List of sampled trajectories
        """
        if agent_name:
            agent_trajectories = self.trajectories.get(agent_name, [])
            if len(agent_trajectories) >= n_trajectories:
                return random.sample(agent_trajectories, n_trajectories)
            else:
                return list(agent_trajectories)
        else:
            # Sample from all agents
            all_trajectories = []
            for trajectories in self.trajectories.values():
                all_trajectories.extend(trajectories)
            
            if len(all_trajectories) >= n_trajectories:
                return random.sample(all_trajectories, n_trajectories)
            else:
                return all_trajectories
    
    def get_recent_trajectories(self, n_trajectories: int, 
                               agent_name: Optional[str] = None) -> List[Trajectory]:
        """Get most recent trajectories.
        
        Args:
            n_trajectories: Number of recent trajectories
            agent_name: Specific agent (None for all)
            
        Returns:
            List of recent trajectories
        """
        if agent_name:
            agent_trajectories = self.trajectories.get(agent_name, [])
            return list(agent_trajectories[-n_trajectories:])
        else:
            recent = []
            for trajectories in self.trajectories.values():
                recent.extend(trajectories[-n_trajectories:])
            return sorted(recent, key=lambda t: t.rewards.sum(), reverse=True)[:n_trajectories]
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.trajectories.clear()
        if self.prioritized:
            self.priorities.clear()
        self.total_transitions = 0
        self.total_trajectories = 0
    
    def save(self, path: Path):
        """Save buffer to disk.
        
        Args:
            path: Path to save buffer
        """
        data = {
            'buffer': list(self.buffer),
            'trajectories': dict(self.trajectories),
            'priorities': list(self.priorities) if self.prioritized else None,
            'total_transitions': self.total_transitions,
            'total_trajectories': self.total_trajectories,
            'config': {
                'capacity': self.capacity,
                'n_agents': self.n_agents,
                'prioritized': self.prioritized
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved experience buffer to {path}")
    
    def load(self, path: Path):
        """Load buffer from disk.
        
        Args:
            path: Path to load buffer from
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # Restore buffer state
        self.buffer = deque(data['buffer'], maxlen=self.capacity)
        self.trajectories = defaultdict(list, data['trajectories'])
        if self.prioritized and data['priorities']:
            self.priorities = deque(data['priorities'], maxlen=self.capacity)
        self.total_transitions = data['total_transitions']
        self.total_trajectories = data['total_trajectories']
        
        logger.info(f"Loaded experience buffer from {path}")
    
    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"ExperienceBuffer(size={len(self)}/{self.capacity}, "
                f"trajectories={self.total_trajectories})")


class TrajectoryBatch:
    """Batch of trajectories for efficient processing."""
    
    def __init__(self, trajectories: List[Trajectory]):
        """Initialize trajectory batch.
        
        Args:
            trajectories: List of trajectories to batch
        """
        self.trajectories = trajectories
        self.agent_names = list(set(t.agent_name for t in trajectories))
        
        # Group by agent
        self.agent_trajectories = defaultdict(list)
        for traj in trajectories:
            self.agent_trajectories[traj.agent_name].append(traj)
    
    def get_agent_batch(self, agent_name: str) -> Dict[str, torch.Tensor]:
        """Get batched data for specific agent.
        
        Args:
            agent_name: Name of agent
            
        Returns:
            Dictionary of batched tensors
        """
        agent_trajs = self.agent_trajectories.get(agent_name, [])
        if not agent_trajs:
            return {}
        
        # Concatenate all trajectories
        all_observations = []
        all_actions = []
        all_log_probs = []
        all_values = []
        all_rewards = []
        all_advantages = []
        all_returns = []
        
        for traj in agent_trajs:
            all_observations.extend(traj.observations)
            all_actions.extend(traj.actions)
            all_log_probs.append(traj.log_probs)
            all_values.append(traj.values)
            all_rewards.append(traj.rewards)
            all_advantages.append(traj.advantages)
            all_returns.append(traj.returns)
        
        # Convert to tensors
        batch_data = {
            'observations': torch.tensor(np.array(all_observations), dtype=torch.float32),
            'actions': torch.tensor(np.array(all_actions), dtype=torch.float32),
            'log_probs': torch.cat(all_log_probs),
            'values': torch.cat(all_values),
            'rewards': torch.cat(all_rewards),
            'advantages': torch.cat(all_advantages),
            'returns': torch.cat(all_returns)
        }
        
        return batch_data
    
    def shuffle_and_split(self, batch_size: int) -> List['TrajectoryBatch']:
        """Shuffle and split into mini-batches.
        
        Args:
            batch_size: Size of mini-batches
            
        Returns:
            List of mini-batches
        """
        # Shuffle trajectories
        shuffled = self.trajectories.copy()
        random.shuffle(shuffled)
        
        # Split into batches
        mini_batches = []
        for i in range(0, len(shuffled), batch_size):
            batch_trajectories = shuffled[i:i + batch_size]
            mini_batches.append(TrajectoryBatch(batch_trajectories))
        
        return mini_batches


class RolloutBuffer:
    """Buffer for collecting rollouts during environment interaction."""
    
    def __init__(self, n_agents: int, device: torch.device = torch.device('cpu')):
        """Initialize rollout buffer.
        
        Args:
            n_agents: Number of agents
            device: Device for tensor operations
        """
        self.n_agents = n_agents
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset the buffer."""
        self.observations = defaultdict(list)
        self.actions = defaultdict(list)
        self.rewards = defaultdict(list)
        self.values = defaultdict(list)
        self.log_probs = defaultdict(list)
        self.dones = []
        self.infos = []
    
    def add(self, observations: Dict[str, np.ndarray], 
            actions: Dict[str, np.ndarray],
            rewards: Dict[str, float],
            values: Dict[str, torch.Tensor],
            log_probs: Dict[str, torch.Tensor],
            done: bool,
            info: Dict[str, Any]):
        """Add step data to buffer.
        
        Args:
            observations: Agent observations
            actions: Agent actions
            rewards: Agent rewards
            values: Value estimates
            log_probs: Action log probabilities
            done: Episode done flag
            info: Additional information
        """
        for agent_name in observations.keys():
            self.observations[agent_name].append(observations[agent_name])
            self.actions[agent_name].append(actions[agent_name])
            self.rewards[agent_name].append(rewards[agent_name])
            self.values[agent_name].append(values[agent_name])
            self.log_probs[agent_name].append(log_probs[agent_name])
        
        self.dones.append(done)
        self.infos.append(info)
    
    def compute_returns_and_advantages(self, last_values: Dict[str, torch.Tensor], 
                                     gamma: float, gae_lambda: float) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute returns and advantages using GAE.
        
        Args:
            last_values: Last value estimates for bootstrap
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            
        Returns:
            Dictionary of (returns, advantages) for each agent
        """
        results = {}
        
        for agent_name in self.observations.keys():
            rewards = torch.tensor(self.rewards[agent_name], dtype=torch.float32, device=self.device)
            values = torch.stack(self.values[agent_name])
            dones = torch.tensor(self.dones, dtype=torch.float32, device=self.device)
            
            # Add last value for bootstrapping
            values = torch.cat([values, last_values[agent_name].unsqueeze(0)])
            
            # Compute advantages
            advantages = torch.zeros_like(rewards)
            last_gae_lambda = 0
            
            for t in reversed(range(len(rewards))):
                next_value = values[t + 1]
                delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
                advantages[t] = last_gae_lambda = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae_lambda
            
            # Compute returns
            returns = advantages + values[:-1]
            
            results[agent_name] = (returns, advantages)
        
        return results
    
    def get_trajectories(self) -> List[Trajectory]:
        """Convert buffer to list of trajectories.
        
        Returns:
            List of agent trajectories
        """
        trajectories = []
        
        # Get returns and advantages
        last_values = {agent: torch.zeros(1, device=self.device) 
                      for agent in self.observations.keys()}
        returns_advantages = self.compute_returns_and_advantages(
            last_values, gamma=0.99, gae_lambda=0.95
        )
        
        for agent_name in self.observations.keys():
            returns, advantages = returns_advantages[agent_name]
            
            trajectory = Trajectory(
                agent_name=agent_name,
                observations=self.observations[agent_name],
                actions=self.actions[agent_name],
                log_probs=torch.stack(self.log_probs[agent_name]),
                values=torch.stack(self.values[agent_name]),
                rewards=torch.tensor(self.rewards[agent_name], dtype=torch.float32),
                advantages=advantages,
                returns=returns
            )
            trajectories.append(trajectory)
        
        return trajectories