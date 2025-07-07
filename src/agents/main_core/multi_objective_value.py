"""
File: src/agents/main_core/multi_objective_value.py (NEW FILE)
Multi-objective value function for MAPPO training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class MultiObjectiveValueFunction(nn.Module):
    """
    Value function that considers multiple objectives:
    - Expected return
    - Risk-adjusted return
    - Timing quality
    - Regime alignment
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Shared encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Objective-specific heads
        self.value_heads = nn.ModuleDict({
            'return': self._create_value_head(),
            'risk_adjusted': self._create_value_head(),
            'timing': self._create_value_head(),
            'regime': self._create_value_head()
        })
        
        # Objective weights (learnable)
        self.objective_weights = nn.Parameter(torch.ones(4) / 4)
        
        # Final value projection
        self.final_value = nn.Linear(4, 1)
        
    def _create_value_head(self) -> nn.Module:
        """Create a value head for specific objective."""
        return nn.Sequential(
            nn.Linear(self.hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, features: torch.Tensor, 
                return_components: bool = False) -> Dict[str, torch.Tensor]:
        """
        Compute multi-objective value.
        
        Args:
            features: Input features
            return_components: Whether to return individual objectives
            
        Returns:
            Dictionary with value and optional components
        """
        # Shared encoding
        shared = self.shared_encoder(features)
        
        # Compute individual objective values
        objective_values = {}
        for name, head in self.value_heads.items():
            value = head(shared)
            objective_values[name] = value
            
        # Stack objective values
        stacked_values = torch.cat(list(objective_values.values()), dim=-1)
        
        # Apply learned weights
        weights = F.softmax(self.objective_weights, dim=0)
        weighted_values = stacked_values * weights.unsqueeze(0)
        
        # Final value
        value = self.final_value(weighted_values)
        
        result = {'value': value}
        
        if return_components:
            result['components'] = objective_values
            result['weights'] = {
                name: weight.item() 
                for name, weight in zip(objective_values.keys(), weights)
            }
            
        return result
        
    def compute_advantages(self, rewards: Dict[str, torch.Tensor],
                          values: Dict[str, torch.Tensor],
                          next_values: Dict[str, torch.Tensor],
                          dones: torch.Tensor,
                          gamma: float = 0.99) -> torch.Tensor:
        """
        Compute multi-objective advantages for MAPPO.
        
        Args:
            rewards: Dictionary of rewards for each objective
            values: Current state values
            next_values: Next state values
            dones: Episode termination flags
            gamma: Discount factor
            
        Returns:
            Combined advantages
        """
        advantages = {}
        
        # Compute advantages for each objective
        for obj_name in self.value_heads.keys():
            if obj_name in rewards:
                # TD error
                td_error = (
                    rewards[obj_name] + 
                    gamma * next_values['components'][obj_name] * (1 - dones) - 
                    values['components'][obj_name]
                )
                advantages[obj_name] = td_error
                
        # Combine advantages using current weights
        weights = F.softmax(self.objective_weights, dim=0)
        combined_advantage = torch.zeros_like(dones)
        
        for i, obj_name in enumerate(self.value_heads.keys()):
            if obj_name in advantages:
                combined_advantage += weights[i] * advantages[obj_name].squeeze()
                
        return combined_advantage