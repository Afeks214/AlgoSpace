"""
Environment wrappers for observation and action processing.
"""

import gym
import numpy as np
from typing import Dict, Any, Tuple
import torch

import structlog

logger = structlog.get_logger()


class ObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper for processing observations before passing to agents.
    
    Handles normalization, feature engineering, and tensor conversion.
    """
    
    def __init__(self, env: gym.Env, agent_type: str):
        """
        Initialize observation wrapper.
        
        Args:
            env: Base environment
            agent_type: Type of agent ('structure', 'tactical', 'arbitrageur')
        """
        super().__init__(env)
        self.agent_type = agent_type
        
        # Agent-specific normalizations
        self.normalization_params = self._load_normalization_params()
        
    def observation(self, observation: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Process observation for the agent.
        
        Args:
            observation: Raw observation from environment
            
        Returns:
            Processed observation as torch tensors
        """
        processed = {}
        
        # Normalize market matrix
        market_matrix = observation['market_matrix']
        normalized_matrix = self._normalize_matrix(market_matrix)
        processed['market_matrix'] = torch.from_numpy(normalized_matrix).float()
        
        # Process regime vector
        regime_vector = observation['regime_vector']
        processed['regime_vector'] = torch.from_numpy(regime_vector).float()
        
        # Process position information
        position = observation['position']
        processed['position'] = torch.from_numpy(position).float()
        
        # Add agent-specific features
        if self.agent_type == 'structure':
            processed['trend_features'] = self._extract_trend_features(market_matrix)
        elif self.agent_type == 'tactical':
            processed['microstructure_features'] = self._extract_microstructure_features(market_matrix)
        elif self.agent_type == 'arbitrageur':
            processed['cross_timeframe_features'] = self._extract_cross_timeframe_features(market_matrix)
        
        # Synergy context
        processed['synergy_active'] = torch.tensor(
            observation.get('synergy_active', 0), 
            dtype=torch.long
        )
        
        return processed
    
    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Normalize market matrix using agent-specific parameters.
        
        Args:
            matrix: Raw market matrix
            
        Returns:
            Normalized matrix
        """
        # Z-score normalization with clipping
        mean = self.normalization_params.get('mean', 0.0)
        std = self.normalization_params.get('std', 1.0)
        
        normalized = (matrix - mean) / (std + 1e-8)
        normalized = np.clip(normalized, -5, 5)  # Clip extreme values
        
        return normalized
    
    def _extract_trend_features(self, matrix: np.ndarray) -> torch.Tensor:
        """Extract trend features for structure analyzer."""
        # Simple moving averages
        if len(matrix) >= 20:
            sma_20 = np.mean(matrix[-20:, 3], axis=0)  # Close prices
            sma_50 = np.mean(matrix[:, 3], axis=0) if len(matrix) >= 50 else sma_20
            
            trend_strength = (sma_20 - sma_50) / (sma_50 + 1e-8)
            trend_consistency = np.std(matrix[-20:, 3]) / (np.mean(matrix[-20:, 3]) + 1e-8)
            
            features = np.array([trend_strength, trend_consistency])
        else:
            features = np.zeros(2)
        
        return torch.from_numpy(features).float()
    
    def _extract_microstructure_features(self, matrix: np.ndarray) -> torch.Tensor:
        """Extract microstructure features for tactical agent."""
        # Recent price action
        if len(matrix) >= 5:
            recent_returns = np.diff(matrix[-6:, 3]) / (matrix[-6:-1, 3] + 1e-8)
            momentum = np.sum(recent_returns)
            volatility = np.std(recent_returns)
            
            # Volume profile
            recent_volume = matrix[-5:, 4]
            volume_ratio = recent_volume[-1] / (np.mean(recent_volume[:-1]) + 1e-8)
            
            features = np.array([momentum, volatility, volume_ratio])
        else:
            features = np.zeros(3)
        
        return torch.from_numpy(features).float()
    
    def _extract_cross_timeframe_features(self, matrix: np.ndarray) -> torch.Tensor:
        """Extract cross-timeframe features for arbitrageur."""
        # Split macro and micro components
        if len(matrix) >= 10:
            # Macro trend (first half)
            macro_data = matrix[:len(matrix)//2]
            macro_return = (macro_data[-1, 3] - macro_data[0, 3]) / (macro_data[0, 3] + 1e-8)
            
            # Micro trend (second half)
            micro_data = matrix[len(matrix)//2:]
            micro_return = (micro_data[-1, 3] - micro_data[0, 3]) / (micro_data[0, 3] + 1e-8)
            
            # Divergence
            divergence = macro_return - micro_return
            
            features = np.array([macro_return, micro_return, divergence])
        else:
            features = np.zeros(3)
        
        return torch.from_numpy(features).float()
    
    def _load_normalization_params(self) -> Dict[str, float]:
        """Load normalization parameters for the agent type."""
        # Default parameters - would be loaded from training statistics
        default_params = {
            'structure': {'mean': 0.0, 'std': 1.0},
            'tactical': {'mean': 0.0, 'std': 0.5},
            'arbitrageur': {'mean': 0.0, 'std': 0.8}
        }
        
        return default_params.get(self.agent_type, {'mean': 0.0, 'std': 1.0})


class ActionWrapper(gym.ActionWrapper):
    """
    Wrapper for processing agent actions before environment execution.
    
    Handles action validation, scaling, and safety checks.
    """
    
    def __init__(self, env: gym.Env, agent_type: str):
        """
        Initialize action wrapper.
        
        Args:
            env: Base environment
            agent_type: Type of agent
        """
        super().__init__(env)
        self.agent_type = agent_type
        
        # Agent-specific action constraints
        self.action_constraints = self._get_action_constraints()
    
    def action(self, action: torch.Tensor) -> np.ndarray:
        """
        Process agent action before execution.
        
        Args:
            action: Raw action from agent
            
        Returns:
            Processed action for environment
        """
        # Convert to numpy if needed
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        
        # Ensure correct shape
        if action.ndim == 1:
            processed_action = action
        else:
            processed_action = action.squeeze()
        
        # Apply constraints
        processed_action = self._apply_constraints(processed_action)
        
        # Validate action
        processed_action = self._validate_action(processed_action)
        
        return processed_action
    
    def reverse_action(self, action: np.ndarray) -> torch.Tensor:
        """
        Reverse action processing for logging/analysis.
        
        Args:
            action: Environment action
            
        Returns:
            Original agent action
        """
        return torch.from_numpy(action).float()
    
    def _apply_constraints(self, action: np.ndarray) -> np.ndarray:
        """
        Apply agent-specific action constraints.
        
        Args:
            action: Raw action
            
        Returns:
            Constrained action
        """
        # Action format: [action_type, size, timing]
        constrained = action.copy()
        
        # Ensure action type is valid (0, 1, or 2)
        constrained[0] = np.clip(constrained[0], 0, 2)
        
        # Apply size constraints
        max_size = self.action_constraints['max_position_size']
        constrained[1] = np.clip(constrained[1], 0, max_size)
        
        # Apply timing constraints
        if self.agent_type == 'tactical':
            # Tactical agent can use full timing range
            constrained[2] = np.clip(constrained[2], 0, 5)
        else:
            # Other agents typically execute immediately
            constrained[2] = np.clip(constrained[2], 0, 2)
        
        return constrained
    
    def _validate_action(self, action: np.ndarray) -> np.ndarray:
        """
        Validate and sanitize action.
        
        Args:
            action: Constrained action
            
        Returns:
            Valid action
        """
        # Round discrete components
        action[0] = np.round(action[0])  # Action type
        action[2] = np.round(action[2])  # Timing
        
        # Ensure size is reasonable
        if action[1] < 0.01:  # Minimum position size
            action[0] = 0  # Convert to pass
            action[1] = 0
        
        return action
    
    def _get_action_constraints(self) -> Dict[str, Any]:
        """Get action constraints for the agent type."""
        constraints = {
            'structure': {
                'max_position_size': 1.0,
                'min_position_size': 0.1,
                'max_timing_delay': 2
            },
            'tactical': {
                'max_position_size': 0.5,
                'min_position_size': 0.05,
                'max_timing_delay': 5
            },
            'arbitrageur': {
                'max_position_size': 0.7,
                'min_position_size': 0.05,
                'max_timing_delay': 3
            }
        }
        
        return constraints.get(self.agent_type, constraints['structure'])


class RewardWrapper(gym.RewardWrapper):
    """
    Wrapper for modifying rewards based on agent type and objectives.
    """
    
    def __init__(self, env: gym.Env, agent_type: str, reward_config: Dict[str, Any]):
        """
        Initialize reward wrapper.
        
        Args:
            env: Base environment
            agent_type: Type of agent
            reward_config: Reward shaping configuration
        """
        super().__init__(env)
        self.agent_type = agent_type
        self.reward_config = reward_config
        
        # Reward shaping parameters
        self.risk_penalty = reward_config.get('risk_penalty', 0.01)
        self.consistency_bonus = reward_config.get('consistency_bonus', 0.005)
        
    def reward(self, reward: float) -> float:
        """
        Modify reward based on agent objectives.
        
        Args:
            reward: Original reward from environment
            
        Returns:
            Shaped reward
        """
        shaped_reward = reward
        
        # Agent-specific reward shaping
        if self.agent_type == 'structure':
            # Structure analyzer focuses on long-term performance
            shaped_reward *= 1.2  # Emphasize returns
            
        elif self.agent_type == 'tactical':
            # Tactical agent focuses on execution quality
            # Add bonus for quick, profitable trades
            if reward > 0:
                shaped_reward += self.consistency_bonus
                
        elif self.agent_type == 'arbitrageur':
            # Arbitrageur focuses on efficiency
            # Penalty for holding positions too long
            shaped_reward -= self.risk_penalty * 0.5
        
        return shaped_reward