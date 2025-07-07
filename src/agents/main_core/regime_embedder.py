"""
File: src/agents/main_core/regime_embedder.py (NEW FILE)
Complete production-ready regime embedder implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import logging
import time

logger = logging.getLogger(__name__)

@dataclass
class RegimeEmbedding:
    """Structured output from regime embedder."""
    mean: torch.Tensor  # μ_regime [batch, 16]
    std: torch.Tensor   # σ_regime [batch, 16]
    attention_weights: Optional[torch.Tensor] = None
    transition_score: Optional[float] = None
    component_importance: Optional[Dict[str, float]] = None

class TemporalRegimeBuffer:
    """
    Maintains temporal history of regime vectors for context.
    Implements efficient circular buffer with GPU support.
    """
    
    def __init__(self, buffer_size: int = 20, regime_dim: int = 8):
        self.buffer_size = buffer_size
        self.regime_dim = regime_dim
        self.buffer = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)
        
    def add(self, regime_vector: torch.Tensor, timestamp: Optional[float] = None):
        """Add new regime vector to buffer."""
        self.buffer.append(regime_vector.detach().cpu())
        self.timestamps.append(timestamp or time.time())
        
    def get_sequence(self, device: torch.device) -> torch.Tensor:
        """Get temporal sequence of regimes."""
        if len(self.buffer) == 0:
            return torch.zeros(1, 1, self.regime_dim).to(device)
            
        # Stack all regime vectors
        sequence = torch.stack(list(self.buffer)).to(device)
        return sequence.unsqueeze(0)  # [1, seq_len, regime_dim]
        
    def get_transition_mask(self, threshold: float = 0.3) -> torch.Tensor:
        """Identify regime transitions based on vector distance."""
        if len(self.buffer) < 2:
            return torch.zeros(1)
            
        transitions = []
        for i in range(1, len(self.buffer)):
            dist = torch.norm(self.buffer[i] - self.buffer[i-1])
            transitions.append(float(dist > threshold))
            
        return torch.tensor(transitions)

class RegimeAttentionAnalyzer(nn.Module):
    """
    Multi-head attention mechanism for analyzing regime components.
    Provides interpretable analysis of regime vector dimensions.
    """
    
    def __init__(self, regime_dim: int = 8, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.regime_dim = regime_dim
        self.n_heads = n_heads
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=regime_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Component importance projection
        self.importance_proj = nn.Sequential(
            nn.Linear(regime_dim, regime_dim * 2),
            nn.ReLU(),
            nn.Linear(regime_dim * 2, regime_dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Linear(regime_dim, regime_dim * 2)
        
    def forward(self, regime_vector: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Analyze regime vector components.
        
        Args:
            regime_vector: [batch, regime_dim]
            
        Returns:
            Tuple of (analyzed_features, attention_info)
        """
        # Expand for attention
        regime_expanded = regime_vector.unsqueeze(1)  # [batch, 1, regime_dim]
        
        # Self-attention to understand component relationships
        attended, attention_weights = self.self_attention(
            regime_expanded, regime_expanded, regime_expanded
        )
        
        # Component importance
        importance = self.importance_proj(regime_vector)
        
        # Apply importance weighting
        weighted_regime = regime_vector * importance
        
        # Project to feature space
        features = self.output_proj(weighted_regime)
        
        attention_info = {
            'attention_weights': attention_weights,
            'component_importance': importance,
            'weighted_regime': weighted_regime
        }
        
        return features, attention_info

class RegimeTransitionDetector(nn.Module):
    """
    Detects and characterizes regime transitions.
    Provides features about regime stability and change dynamics.
    """
    
    def __init__(self, regime_dim: int = 8, hidden_dim: int = 32):
        super().__init__()
        
        # Transition analyzer
        self.transition_net = nn.Sequential(
            nn.Linear(regime_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)  # [stability, volatility, direction, magnitude]
        )
        
        # Transition feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(regime_dim * 2 + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 16)
        )
        
    def forward(self, current_regime: torch.Tensor, regime_history: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Detect and analyze regime transitions.
        
        Args:
            current_regime: [batch, regime_dim]
            regime_history: [batch, seq_len, regime_dim]
            
        Returns:
            Tuple of (transition_features, transition_metrics)
        """
        batch_size = current_regime.size(0)
        
        # Get most recent historical regime
        if regime_history.size(1) > 0:
            prev_regime = regime_history[:, -1, :]
        else:
            prev_regime = current_regime  # No history, no transition
            
        # Concatenate for transition analysis
        regime_pair = torch.cat([prev_regime, current_regime], dim=-1)
        
        # Analyze transition
        transition_scores = self.transition_net(regime_pair)
        
        # Extract features
        combined = torch.cat([regime_pair, transition_scores], dim=-1)
        transition_features = self.feature_extractor(combined)
        
        # Calculate metrics
        transition_metrics = {
            'stability': torch.sigmoid(transition_scores[:, 0]).mean().item(),
            'volatility': torch.sigmoid(transition_scores[:, 1]).mean().item(),
            'direction': torch.tanh(transition_scores[:, 2]).mean().item(),
            'magnitude': torch.sigmoid(transition_scores[:, 3]).mean().item()
        }
        
        return transition_features, transition_metrics

class TemporalRegimeEncoder(nn.Module):
    """
    LSTM-based encoder for temporal regime patterns.
    Captures regime evolution and persistence patterns.
    """
    
    def __init__(self, regime_dim: int = 8, hidden_dim: int = 32, n_layers: int = 2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=regime_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1 if n_layers > 1 else 0
        )
        
        # Temporal feature projection
        self.temporal_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 16)
        )
        
        # Hidden state initialization
        self.h0 = nn.Parameter(torch.zeros(n_layers, 1, hidden_dim))
        self.c0 = nn.Parameter(torch.zeros(n_layers, 1, hidden_dim))
        
    def forward(self, regime_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode temporal regime sequence.
        
        Args:
            regime_sequence: [batch, seq_len, regime_dim]
            
        Returns:
            Tuple of (temporal_features, hidden_state)
        """
        batch_size = regime_sequence.size(0)
        
        # Initialize hidden state
        h0 = self.h0.expand(-1, batch_size, -1).contiguous()
        c0 = self.c0.expand(-1, batch_size, -1).contiguous()
        
        # LSTM encoding
        lstm_out, (hn, cn) = self.lstm(regime_sequence, (h0, c0))
        
        # Use last hidden state for features
        temporal_features = self.temporal_proj(lstm_out[:, -1, :])
        
        return temporal_features, hn

class RegimeEmbedder(nn.Module):
    """
    State-of-the-art Regime Embedder with temporal memory,
    attention analysis, and transition detection.
    
    Transforms 8D regime vectors into rich 16D embeddings with
    uncertainty quantification and interpretable components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Configuration
        self.regime_dim = config.get('regime_dim', 8)
        self.output_dim = config.get('output_dim', 16)
        self.hidden_dim = config.get('hidden_dim', 32)
        self.buffer_size = config.get('buffer_size', 20)
        self.n_heads = config.get('n_heads', 4)
        self.dropout = config.get('dropout', 0.1)
        
        # Temporal buffer
        self.regime_buffer = TemporalRegimeBuffer(self.buffer_size, self.regime_dim)
        
        # Components
        self.attention_analyzer = RegimeAttentionAnalyzer(
            self.regime_dim, self.n_heads, self.dropout
        )
        self.transition_detector = RegimeTransitionDetector(
            self.regime_dim, self.hidden_dim
        )
        self.temporal_encoder = TemporalRegimeEncoder(
            self.regime_dim, self.hidden_dim
        )
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(16 + 16 + 16, self.hidden_dim * 2),  # attention + transition + temporal
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        
        # Output heads
        self.mean_head = nn.Linear(self.hidden_dim, self.output_dim)
        self.std_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Softplus()  # Ensure positive std
        )
        
        # Component importance tracker
        self.component_importance_ema = torch.ones(self.regime_dim) / self.regime_dim
        self.ema_alpha = 0.01
        
        logger.info(f"Initialized RegimeEmbedder with output_dim={self.output_dim}")
        
    def forward(self, regime_vector: torch.Tensor, 
                return_attention: bool = False) -> RegimeEmbedding:
        """
        Transform regime vector into rich embedding.
        
        Args:
            regime_vector: [batch, 8] regime vector from RDE
            return_attention: Whether to return attention weights
            
        Returns:
            RegimeEmbedding with mean, std, and optional attention info
        """
        device = regime_vector.device
        batch_size = regime_vector.size(0)
        
        # Add to temporal buffer (for first item in batch)
        self.regime_buffer.add(regime_vector[0])
        
        # Get temporal sequence
        regime_sequence = self.regime_buffer.get_sequence(device)
        # Expand to match batch size
        if regime_sequence.size(0) == 1 and batch_size > 1:
            regime_sequence = regime_sequence.expand(batch_size, -1, -1)
        
        # 1. Attention-based analysis
        attention_features, attention_info = self.attention_analyzer(regime_vector)
        
        # Update component importance EMA
        with torch.no_grad():
            new_importance = attention_info['component_importance'][0].cpu()
            self.component_importance_ema = (
                (1 - self.ema_alpha) * self.component_importance_ema +
                self.ema_alpha * new_importance
            )
        
        # 2. Transition detection
        transition_features, transition_metrics = self.transition_detector(
            regime_vector, regime_sequence
        )
        
        # 3. Temporal encoding
        temporal_features, _ = self.temporal_encoder(regime_sequence)
        temporal_features = temporal_features.expand(batch_size, -1)
        
        # 4. Feature fusion
        combined_features = torch.cat([
            attention_features[:, :16],  # Take first 16 dims
            transition_features,
            temporal_features
        ], dim=-1)
        
        fused_features = self.feature_fusion(combined_features)
        
        # 5. Generate mean and std
        regime_mean = self.mean_head(fused_features)
        regime_std = self.std_head(fused_features) + 1e-6  # Numerical stability
        
        # Create output
        embedding = RegimeEmbedding(
            mean=regime_mean,
            std=regime_std,
            attention_weights=attention_info['attention_weights'] if return_attention else None,
            transition_score=transition_metrics['magnitude'],
            component_importance={
                f'dim_{i}': float(self.component_importance_ema[i]) 
                for i in range(self.regime_dim)
            }
        )
        
        return embedding
    
    def get_regime_interpretation(self) -> Dict[str, Any]:
        """Get human-interpretable regime analysis."""
        # Identify most important dimensions
        importance = self.component_importance_ema.numpy()
        top_dims = np.argsort(importance)[-3:][::-1]
        
        interpretation = {
            'dominant_dimensions': top_dims.tolist(),
            'dimension_importance': {
                f'dim_{i}': float(importance[i]) 
                for i in range(len(importance))
            },
            'regime_stability': float(self.regime_buffer.get_transition_mask().mean()),
            'buffer_fullness': len(self.regime_buffer.buffer) / self.buffer_size
        }
        
        return interpretation