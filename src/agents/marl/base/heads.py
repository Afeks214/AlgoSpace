"""
Output head implementations for MARL agents.

Provides specialized neural network heads for different
decision outputs (actions, confidence, reasoning, etc.).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class PolicyHead(nn.Module):
    """
    Base policy head that combines multiple output heads.
    
    Manages the final decision-making layers of an agent,
    producing actions, confidence scores, and reasoning features.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_heads: Dict[str, Dict[str, int]],
        dropout: float = 0.2
    ):
        """
        Initialize policy head.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_heads: Dictionary of output heads with their dimensions
                         e.g., {'action': {'dim': 3}, 'confidence': {'dim': 1}}
            dropout: Dropout rate
        """
        super().__init__()
        
        # Build shared layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            
            # Add dropout to all but last hidden layer
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Build output heads
        self.output_heads = nn.ModuleDict()
        final_hidden = hidden_dims[-1] if hidden_dims else input_dim
        
        for head_name, head_config in output_heads.items():
            self.output_heads[head_name] = self._build_output_head(
                final_hidden,
                head_config['dim'],
                head_config.get('activation', None)
            )
    
    def _build_output_head(
        self,
        input_dim: int,
        output_dim: int,
        activation: Optional[str] = None
    ) -> nn.Module:
        """
        Build a single output head.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            activation: Optional activation function
            
        Returns:
            Output head module
        """
        layers = [nn.Linear(input_dim, output_dim)]
        
        if activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'softmax':
            # Softmax applied in forward pass with dim parameter
            pass
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through policy head.
        
        Args:
            x: Input features
            
        Returns:
            Dictionary of outputs from each head
        """
        # Pass through shared layers
        shared_features = self.shared_layers(x)
        
        # Generate outputs from each head
        outputs = {}
        for head_name, head_module in self.output_heads.items():
            output = head_module(shared_features)
            
            # Apply softmax for action head if needed
            if head_name == 'action' and output.size(-1) > 1:
                # Keep as logits for training, can apply softmax later
                pass
            
            outputs[head_name] = output
        
        return outputs


class ActionHead(nn.Module):
    """
    Specialized head for action selection.
    
    Produces action logits for discrete action spaces.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        temperature: float = 1.0
    ):
        """
        Initialize action head.
        
        Args:
            input_dim: Input dimension
            num_actions: Number of possible actions
            temperature: Temperature for action sampling
        """
        super().__init__()
        
        self.num_actions = num_actions
        self.temperature = temperature
        
        # Action logits layer
        self.action_layer = nn.Linear(input_dim, num_actions)
        
        # Initialize with small values for stable training
        nn.init.xavier_uniform_(self.action_layer.weight, gain=0.01)
        nn.init.constant_(self.action_layer.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        return_probs: bool = False
    ) -> torch.Tensor:
        """
        Generate action logits or probabilities.
        
        Args:
            x: Input features
            return_probs: Whether to return probabilities instead of logits
            
        Returns:
            Action logits or probabilities
        """
        logits = self.action_layer(x)
        
        if return_probs:
            # Apply temperature scaling
            scaled_logits = logits / self.temperature
            return F.softmax(scaled_logits, dim=-1)
        
        return logits
    
    def sample_action(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sample action from distribution.
        
        Args:
            x: Input features
            
        Returns:
            Sampled action indices
        """
        probs = self.forward(x, return_probs=True)
        return torch.multinomial(probs, 1).squeeze(-1)


class ConfidenceHead(nn.Module):
    """
    Specialized head for confidence estimation.
    
    Produces calibrated confidence scores for decisions.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        use_beta: bool = True
    ):
        """
        Initialize confidence head.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            use_beta: Whether to use Beta distribution parameters
        """
        super().__init__()
        
        self.use_beta = use_beta
        
        if use_beta:
            # Output alpha and beta parameters for Beta distribution
            self.confidence_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),  # alpha, beta
                nn.Softplus()  # Ensure positive parameters
            )
        else:
            # Direct confidence estimation
            self.confidence_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Estimate confidence.
        
        Args:
            x: Input features
            
        Returns:
            Dictionary with confidence and optionally distribution params
        """
        if self.use_beta:
            params = self.confidence_net(x)
            alpha = params[:, 0:1] + 1  # Add 1 to ensure alpha > 1
            beta = params[:, 1:2] + 1   # Add 1 to ensure beta > 1
            
            # Mean of Beta distribution as confidence
            confidence = alpha / (alpha + beta)
            
            # Variance as uncertainty measure
            variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
            
            return {
                'confidence': confidence,
                'alpha': alpha,
                'beta': beta,
                'uncertainty': torch.sqrt(variance)
            }
        else:
            confidence = self.confidence_net(x)
            return {'confidence': confidence}


class ReasoningHead(nn.Module):
    """
    Specialized head for interpretable reasoning features.
    
    Produces human-interpretable features that explain decisions.
    """
    
    def __init__(
        self,
        input_dim: int,
        reasoning_dim: int,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize reasoning head.
        
        Args:
            input_dim: Input dimension
            reasoning_dim: Number of reasoning features
            feature_names: Optional names for reasoning features
        """
        super().__init__()
        
        self.reasoning_dim = reasoning_dim
        self.feature_names = feature_names or [f'reason_{i}' for i in range(reasoning_dim)]
        
        # Reasoning extraction layers
        self.reasoning_net = nn.Sequential(
            nn.Linear(input_dim, reasoning_dim * 2),
            nn.ReLU(),
            nn.Linear(reasoning_dim * 2, reasoning_dim),
            nn.Tanh()  # Bounded output for interpretability
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract reasoning features.
        
        Args:
            x: Input features
            
        Returns:
            Reasoning features tensor
        """
        return self.reasoning_net(x)
    
    def get_reasoning_dict(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Get reasoning as named dictionary.
        
        Args:
            x: Input features
            
        Returns:
            Dictionary mapping feature names to values
        """
        reasoning = self.forward(x)
        
        # Convert to dictionary
        reasoning_dict = {}
        for i, name in enumerate(self.feature_names):
            reasoning_dict[name] = reasoning[:, i].item()
        
        return reasoning_dict


class TimingHead(nn.Module):
    """
    Specialized head for timing decisions.
    
    Used by Short-term Tactician to decide execution timing.
    """
    
    def __init__(self, input_dim: int, max_delay: int = 5):
        """
        Initialize timing head.
        
        Args:
            input_dim: Input dimension
            max_delay: Maximum bars to delay (0 = immediate)
        """
        super().__init__()
        
        self.max_delay = max_delay
        
        # Timing prediction
        self.timing_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, max_delay + 1)  # 0 to max_delay
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict optimal timing.
        
        Args:
            x: Input features
            
        Returns:
            Timing logits
        """
        return self.timing_net(x)