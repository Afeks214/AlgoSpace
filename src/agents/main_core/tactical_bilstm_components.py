"""
BiLSTM-specific components for enhanced temporal processing.

This module provides advanced components specifically designed for 
BiLSTM architectures to improve temporal pattern recognition in 
the Tactical Embedder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BiLSTMGateController(nn.Module):
    """
    Adaptive gating mechanism for BiLSTM outputs.
    Controls information flow from forward and backward directions.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # Gates for forward and backward features
        self.forward_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.backward_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Sigmoid()
        )
        
    def forward(self, forward_features: torch.Tensor, 
                backward_features: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive gating to BiLSTM outputs.
        
        Args:
            forward_features: Forward LSTM output [batch, seq, hidden]
            backward_features: Backward LSTM output [batch, seq, hidden]
            
        Returns:
            Gated features [batch, seq, hidden * 2]
        """
        # Concatenate for gate computation
        combined = torch.cat([forward_features, backward_features], dim=-1)
        
        # Compute gates
        f_gate = self.forward_gate(combined)
        b_gate = self.backward_gate(combined)
        
        # Apply gates
        gated_forward = forward_features * f_gate
        gated_backward = backward_features * b_gate
        
        # Final fusion
        gated_combined = torch.cat([gated_forward, gated_backward], dim=-1)
        fusion = self.fusion_gate(gated_combined)
        
        return gated_combined * fusion


class TemporalPyramidPooling(nn.Module):
    """
    Hierarchical pooling of BiLSTM outputs at multiple temporal scales.
    Captures patterns at different time resolutions.
    """
    
    def __init__(self, input_dim: int, pyramid_levels: list = [1, 2, 4, 8]):
        super().__init__()
        self.pyramid_levels = pyramid_levels
        self.input_dim = input_dim
        
        # Projection for each pyramid level
        self.level_projections = nn.ModuleList([
            nn.Linear(input_dim, input_dim // len(pyramid_levels))
            for _ in pyramid_levels
        ])
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal pyramid pooling.
        
        Args:
            x: BiLSTM output [batch, seq, dim]
            
        Returns:
            Pooled features [batch, dim]
        """
        batch_size, seq_len, _ = x.shape
        pyramid_features = []
        
        for i, level in enumerate(self.pyramid_levels):
            # Adaptive pooling to level size
            if seq_len % level == 0:
                # Reshape and pool
                reshaped = x.view(batch_size, level, seq_len // level, -1)
                pooled = reshaped.mean(dim=2)  # [batch, level, dim]
            else:
                # Use adaptive pooling
                pooled = F.adaptive_avg_pool1d(
                    x.transpose(1, 2), level
                ).transpose(1, 2)
            
            # Project and pool across temporal dimension
            projected = self.level_projections[i](pooled)
            level_feature = projected.mean(dim=1)  # [batch, dim//n_levels]
            pyramid_features.append(level_feature)
        
        # Concatenate all levels
        combined = torch.cat(pyramid_features, dim=-1)
        return self.fusion(combined)


class BiLSTMPositionalEncoding(nn.Module):
    """
    Positional encoding specifically designed for BiLSTM outputs.
    Helps maintain temporal awareness after pooling operations.
    """
    
    def __init__(self, hidden_dim: int, max_len: int = 100):
        super().__init__()
        
        # Learnable positional embeddings for forward and backward
        self.forward_pos_embed = nn.Parameter(torch.randn(1, max_len, hidden_dim) * 0.01)
        self.backward_pos_embed = nn.Parameter(torch.randn(1, max_len, hidden_dim) * 0.01)
        
        # Temporal importance weights
        self.temporal_weights = nn.Parameter(torch.ones(1, max_len, 1))
        
    def forward(self, forward_features: torch.Tensor, 
                backward_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add positional encoding to BiLSTM features.
        
        Args:
            forward_features: Forward features [batch, seq, hidden]
            backward_features: Backward features [batch, seq, hidden]
            
        Returns:
            Tuple of encoded features
        """
        batch_size, seq_len, _ = forward_features.shape
        
        # Add positional encoding
        forward_encoded = forward_features + self.forward_pos_embed[:, :seq_len, :]
        backward_encoded = backward_features + self.backward_pos_embed[:, :seq_len, :]
        
        # Apply temporal weights
        weights = torch.sigmoid(self.temporal_weights[:, :seq_len, :])
        forward_weighted = forward_encoded * weights
        backward_weighted = backward_encoded * weights.flip(dims=[1])  # Reverse for backward
        
        return forward_weighted, backward_weighted


class DirectionalFeatureFusion(nn.Module):
    """
    Advanced fusion mechanism for combining forward and backward
    directional features from BiLSTM.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # Attention mechanism for fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Feature transformation
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, forward_features: torch.Tensor,
                backward_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse directional features using cross-attention.
        
        Args:
            forward_features: [batch, seq, hidden]
            backward_features: [batch, seq, hidden]
            
        Returns:
            Fused features [batch, seq, hidden]
        """
        # Cross-attention: forward attends to backward
        attended_forward, _ = self.cross_attention(
            forward_features, backward_features, backward_features
        )
        
        # Concatenate and transform
        combined = torch.cat([attended_forward, backward_features], dim=-1)
        fused = self.transform(combined)
        
        return fused


class BiLSTMTemporalMasking(nn.Module):
    """
    Adaptive temporal masking for BiLSTM outputs to handle
    variable sequence lengths and focus on relevant time steps.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # Learnable masking mechanism
        self.mask_generator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, bilstm_output: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply adaptive temporal masking.
        
        Args:
            bilstm_output: [batch, seq, hidden*2]
            lengths: Optional sequence lengths [batch]
            
        Returns:
            Masked output [batch, seq, hidden*2]
        """
        # Generate masks
        masks = self.mask_generator(bilstm_output)  # [batch, seq, 1]
        
        # Apply length-based masking if provided
        if lengths is not None:
            batch_size, seq_len = bilstm_output.shape[:2]
            length_mask = torch.arange(seq_len, device=bilstm_output.device)[None, :] < lengths[:, None]
            masks = masks * length_mask.unsqueeze(-1).float()
        
        # Apply masking
        masked_output = bilstm_output * masks
        
        return masked_output