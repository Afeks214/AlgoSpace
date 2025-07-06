"""
Main MARL Core Neural Network Models.

This module contains all neural network architectures for the unified
intelligence system, including specialized embedders, the shared policy
network, and the final decision gate.
"""

from typing import Dict, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class StructureEmbedder(nn.Module):
    """
    Embedder for processing 30-minute market structure data.
    
    Processes the 48×8 matrix from MatrixAssembler30m to extract
    long-term structural patterns and trends.
    
    Args:
        input_channels: Number of input features (default: 8)
        output_dim: Output embedding dimension (default: 64)
        dropout_rate: Dropout probability (default: 0.2)
    """
    
    def __init__(
        self,
        input_channels: int = 8,
        output_dim: int = 64,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_dim = output_dim
        
        # 1D CNN for temporal pattern extraction
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Second conv block
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Third conv block
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # Global pooling
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through structure embedder.
        
        Args:
            x: Input tensor [batch_size, 48, 8]
            
        Returns:
            Embedded features [batch_size, output_dim]
        """
        # Transpose for Conv1d: [batch, features, time]
        x = x.transpose(1, 2)
        
        # Extract features through CNN
        features = self.conv_layers(x)
        
        # Flatten: [batch, 128, 1] -> [batch, 128]
        features = features.squeeze(-1)
        
        # Project to output dimension
        embedded = self.projection(features)
        
        return embedded


class TacticalEmbedder(nn.Module):
    """
    Embedder for processing 5-minute tactical data.
    
    Processes the 60×9 matrix from MatrixAssembler5m using LSTM
    and attention to capture short-term dynamics including enhanced FVG features.
    
    Args:
        input_dim: Number of input features (default: 9)
        hidden_dim: LSTM hidden dimension (default: 64)
        output_dim: Output embedding dimension (default: 32)
        dropout_rate: Dropout probability (default: 0.2)
    """
    
    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 64,
        output_dim: int = 32,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # LSTM for sequential processing
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # Bidirectional
            num_heads=4,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through tactical embedder.
        
        Args:
            x: Input tensor [batch_size, 60, 9]
            
        Returns:
            Embedded features [batch_size, output_dim]
        """
        # Process through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled = torch.mean(attended, dim=1)
        
        # Project to output dimension
        embedded = self.projection(pooled)
        
        return embedded


class RegimeEmbedder(nn.Module):
    """
    Embedder for processing regime context vector.
    
    Projects the 8-dimensional regime vector from RDE into
    a higher-dimensional embedding space.
    
    Args:
        input_dim: Input vector dimension (default: 8)
        output_dim: Output embedding dimension (default: 16)
        hidden_dim: Hidden layer dimension (default: 32)
    """
    
    def __init__(
        self,
        input_dim: int = 8,
        output_dim: int = 16,
        hidden_dim: int = 32
    ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through regime embedder.
        
        Args:
            x: Input regime vector [batch_size, 8]
            
        Returns:
            Embedded features [batch_size, output_dim]
        """
        return self.mlp(x)


class LVNEmbedder(nn.Module):
    """
    Embedder for processing LVN (Low Volume Node) context.
    
    Processes tactical features related to nearby support/resistance
    levels identified through volume profile analysis.
    
    Args:
        input_dim: Number of LVN features (default: 5)
        output_dim: Output embedding dimension (default: 8)
        hidden_dim: Hidden layer dimension (default: 16)
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        output_dim: int = 8,
        hidden_dim: int = 16
    ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LVN embedder.
        
        Args:
            x: LVN features [batch_size, input_dim]
            
        Returns:
            Embedded features [batch_size, output_dim]
        """
        return self.mlp(x)


class SharedPolicy(nn.Module):
    """
    Unified shared policy network (MAPPO Actor) for Gate 1.
    
    This is the main decision-making network that processes the
    concatenated embeddings from all sources and outputs action
    probabilities for initiating the trade process.
    
    Args:
        input_dim: Dimension of unified state vector (default: 144)
        hidden_dims: List of hidden layer dimensions
        dropout_rate: Dropout probability for MC Dropout (default: 0.2)
        action_dim: Number of actions (default: 2)
    """
    
    def __init__(
        self,
        input_dim: int = 144,  # 64 + 32 + 16 + 32 (updated dimensions)
        hidden_dims: List[int] = None,
        dropout_rate: float = 0.2,
        action_dim: int = 2  # ['Initiate_Trade_Process', 'Do_Nothing']
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
            
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.dropout_rate = dropout_rate
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)  # Critical for MC Dropout
            ])
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.policy_network = nn.Sequential(*layers)
        
        # Value head for MAPPO training (not used in inference)
        self.value_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(
        self, 
        unified_state: torch.Tensor,
        return_value: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through shared policy.
        
        Args:
            unified_state: Concatenated state vector [batch_size, input_dim]
            return_value: Whether to compute value function
            
        Returns:
            Dictionary containing:
                - action_logits: Raw logits for actions [batch_size, 2]
                - action_probs: Softmax probabilities [batch_size, 2]
                - value: Value estimate (if requested)
        """
        # Get action logits
        action_logits = self.policy_network(unified_state)
        action_probs = F.softmax(action_logits, dim=-1)
        
        output = {
            'action_logits': action_logits,
            'action_probs': action_probs
        }
        
        # Optionally compute value
        if return_value:
            value = self.value_head(unified_state)
            output['value'] = value.squeeze(-1)
            
        return output
    
    def enable_mc_dropout(self):
        """Enable dropout for MC Dropout evaluation."""
        self.train()  # This enables dropout
        
    def disable_mc_dropout(self):
        """Disable dropout for deterministic evaluation."""
        self.eval()  # This disables dropout


class DecisionGate(nn.Module):
    """
    Final decision gate network for Gate 2.
    
    This network makes the final EXECUTE/REJECT decision after
    incorporating the risk proposal from M-RMS.
    
    Args:
        input_dim: Dimension of final state (unified + risk) (default: 152)
        hidden_dim: Hidden layer dimension (default: 64)
        dropout_rate: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        input_dim: int = 152,  # 144 + 8 (risk vector)
        hidden_dim: int = 64,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.decision_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, 2)  # ['EXECUTE', 'REJECT']
        )
        
    def forward(self, final_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through decision gate.
        
        Args:
            final_state: Final state with risk vector [batch_size, input_dim]
            
        Returns:
            Dictionary containing:
                - decision_logits: Raw logits [batch_size, 2]
                - decision_probs: Softmax probabilities [batch_size, 2]
                - execute_probability: Probability of EXECUTE [batch_size]
        """
        decision_logits = self.decision_network(final_state)
        decision_probs = F.softmax(decision_logits, dim=-1)
        
        return {
            'decision_logits': decision_logits,
            'decision_probs': decision_probs,
            'execute_probability': decision_probs[:, 0]  # EXECUTE is index 0
        }


class MCDropoutEvaluator:
    """
    Helper class for MC Dropout consensus evaluation.
    
    Implements the superposition decision-making principle by running
    multiple forward passes with dropout enabled.
    """
    
    def __init__(self, n_passes: int = 50):
        """
        Initialize MC Dropout evaluator.
        
        Args:
            n_passes: Number of forward passes for consensus
        """
        self.n_passes = n_passes
        
    def evaluate(
        self,
        model: SharedPolicy,
        unified_state: torch.Tensor,
        confidence_threshold: float = 0.65
    ) -> Dict[str, any]:
        """
        Run MC Dropout evaluation.
        
        Args:
            model: SharedPolicy network
            unified_state: Input state vector
            confidence_threshold: Minimum confidence for consensus
            
        Returns:
            Consensus results dictionary
        """
        # Enable MC Dropout
        model.enable_mc_dropout()
        
        # Collect predictions
        all_probs = []
        
        with torch.no_grad():
            for _ in range(self.n_passes):
                output = model(unified_state)
                all_probs.append(output['action_probs'])
        
        # Stack predictions: [n_passes, batch_size, 2]
        all_probs = torch.stack(all_probs)
        
        # Calculate statistics
        mean_probs = all_probs.mean(dim=0)
        std_probs = all_probs.std(dim=0)
        
        # Get predicted action (0: Initiate, 1: Do_Nothing)
        predicted_action = mean_probs.argmax(dim=-1)
        
        # Calculate confidence (probability of predicted action)
        confidence = mean_probs.gather(1, predicted_action.unsqueeze(-1)).squeeze(-1)
        
        # Calculate uncertainty metrics
        entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1)
        
        # Determine if we should proceed
        should_proceed = (
            (predicted_action == 0) &  # Initiate action
            (confidence >= confidence_threshold)
        )
        
        # Restore model to eval mode
        model.disable_mc_dropout()
        
        return {
            'predicted_action': predicted_action,
            'mean_probs': mean_probs,
            'std_probs': std_probs,
            'confidence': confidence,
            'entropy': entropy,
            'should_proceed': should_proceed,
            'uncertainty_metrics': {
                'mean_std': std_probs.mean().item(),
                'max_std': std_probs.max().item(),
                'entropy': entropy.mean().item()
            }
        }