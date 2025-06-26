"""
Shared embedding architectures for MARL agents.

Provides reusable neural network components for feature extraction
and representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SharedEmbedder(nn.Module):
    """
    Shared convolutional embedding architecture.
    
    Processes time series data through multiple Conv1D layers with
    batch normalization and dropout for robust feature extraction.
    """
    
    def __init__(
        self,
        input_features: int,
        hidden_dim: int = 256,
        dropout: float = 0.2
    ):
        """
        Initialize shared embedder.
        
        Args:
            input_features: Number of input features
            hidden_dim: Hidden dimension size
            dropout: Dropout rate
        """
        super().__init__()
        
        # Progressive feature extraction
        self.conv1 = nn.Conv1d(
            input_features, 64,
            kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            64, 128,
            kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout)
        
        self.conv3 = nn.Conv1d(
            128, hidden_dim,
            kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through embedder.
        
        Args:
            x: Input tensor [batch, features, time]
            
        Returns:
            Embedded features [batch, hidden_dim, time]
        """
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        return x


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for focusing on relevant time steps.
    
    Uses multi-head self-attention to capture temporal dependencies
    and identify important moments in the time series.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.2
    ):
        """
        Initialize temporal attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization and residual connection
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temporal attention.
        
        Args:
            x: Input tensor [batch, time, features]
            mask: Optional attention mask
            
        Returns:
            Tuple of:
                - Attended features [batch, time, features]
                - Attention weights [batch, num_heads, time, time]
        """
        # Self-attention with residual connection
        attn_out, attn_weights = self.attention(
            x, x, x,
            attn_mask=mask,
            need_weights=True
        )
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x, attn_weights


class SynergyEncoder(nn.Module):
    """
    Base class for encoding synergy context into feature vectors.
    
    Provides common functionality for extracting relevant information
    from synergy detection events.
    """
    
    def __init__(self, output_dim: int):
        """
        Initialize synergy encoder.
        
        Args:
            output_dim: Output feature dimension
        """
        super().__init__()
        self.output_dim = output_dim
    
    def encode_synergy_type(self, synergy_type: str) -> torch.Tensor:
        """
        One-hot encode synergy type.
        
        Args:
            synergy_type: 'TYPE_1', 'TYPE_2', 'TYPE_3', or 'TYPE_4'
            
        Returns:
            One-hot encoded tensor [4]
        """
        synergy_map = {
            'TYPE_1': 0,
            'TYPE_2': 1,
            'TYPE_3': 2,
            'TYPE_4': 3
        }
        
        encoding = torch.zeros(4)
        if synergy_type in synergy_map:
            encoding[synergy_map[synergy_type]] = 1.0
        
        return encoding
    
    def encode_direction(self, direction: int) -> torch.Tensor:
        """
        Encode trade direction.
        
        Args:
            direction: 1 (long) or -1 (short)
            
        Returns:
            Encoded direction [2]
        """
        if direction == 1:
            return torch.tensor([1.0, 0.0])
        else:
            return torch.tensor([0.0, 1.0])
    
    def normalize_value(self, value: float, min_val: float, max_val: float) -> float:
        """
        Normalize value to [0, 1] range.
        
        Args:
            value: Value to normalize
            min_val: Minimum expected value
            max_val: Maximum expected value
            
        Returns:
            Normalized value
        """
        if max_val == min_val:
            return 0.5
        return (value - min_val) / (max_val - min_val)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for time-aware processing.
    
    Adds sinusoidal positional embeddings to help the model
    understand temporal ordering.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        # Create constant buffer
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding.
        
        Args:
            x: Input tensor [seq_len, batch, features]
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(0), :]