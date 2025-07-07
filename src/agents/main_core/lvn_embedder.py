"""
Advanced LVN Embedder with Spatial Awareness and Temporal Tracking.

This module implements the sophisticated LVN embedding system that transforms
support/resistance level data into actionable intelligence for trading decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass

from .models import MCDropoutMixin


@dataclass
class LVNContext:
    """Rich context for LVN analysis."""
    prices: torch.Tensor  # LVN price levels
    strengths: torch.Tensor  # Strength scores (0-100)
    distances: torch.Tensor  # Distances from current price
    volumes: torch.Tensor  # Volume at each level
    current_price: float
    price_history: torch.Tensor  # Recent price trajectory
    interaction_history: Optional[Dict[str, Any]] = None


class SpatialRelationshipModule(nn.Module):
    """Model spatial relationships between LVN levels."""
    
    def __init__(self, hidden_dim: int = 64, n_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Multi-head attention for LVN relationships
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Spatial encoding
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Strength modulation
        self.strength_gate = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        lvn_embeddings: torch.Tensor,
        distances: torch.Tensor,
        strengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Model spatial relationships between LVN levels.
        
        Args:
            lvn_embeddings: [batch, n_levels, hidden_dim]
            distances: [batch, n_levels, 1]
            strengths: [batch, n_levels, 1]
            
        Returns:
            Spatially-aware embeddings [batch, n_levels, hidden_dim]
        """
        # Encode distances
        distance_encoding = self.distance_encoder(distances)
        
        # Add distance information to embeddings
        lvn_embeddings = lvn_embeddings + distance_encoding
        
        # Apply strength gating
        strength_gate = self.strength_gate(strengths)
        lvn_embeddings = lvn_embeddings * strength_gate
        
        # Cross-attention between levels
        attended, _ = self.cross_attention(
            lvn_embeddings,
            lvn_embeddings,
            lvn_embeddings
        )
        
        return attended + lvn_embeddings  # Residual connection


class TemporalTrackingModule(nn.Module):
    """Track temporal evolution of price-LVN interactions."""
    
    def __init__(self, hidden_dim: int = 64, sequence_length: int = 20):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        
        # LSTM for tracking price trajectory relative to LVNs
        self.trajectory_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Interaction pattern recognition
        self.pattern_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.LayerNorm(hidden_dim)
        )
        
        # Test/rejection classifier
        self.interaction_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3)  # [bounce, break, no_interaction]
        )
        
    def forward(
        self,
        price_trajectory: torch.Tensor,
        lvn_embeddings: torch.Tensor,
        historical_interactions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Track temporal evolution of price-LVN interactions.
        
        Args:
            price_trajectory: [batch, seq_len, features]
            lvn_embeddings: [batch, n_levels, hidden_dim]
            historical_interactions: Optional past interactions
            
        Returns:
            temporal_features: [batch, hidden_dim]
            interaction_predictions: [batch, n_levels, 3]
        """
        batch_size = price_trajectory.shape[0]
        n_levels = lvn_embeddings.shape[1]
        
        # Process price trajectory
        trajectory_out, (h_n, _) = self.trajectory_lstm(price_trajectory)
        
        # Extract temporal patterns
        trajectory_conv = self.pattern_conv(
            trajectory_out.transpose(1, 2)
        ).transpose(1, 2)
        
        # Global temporal feature
        temporal_features = torch.cat([
            h_n[-1],  # Last hidden state
            trajectory_conv.mean(dim=1)  # Average patterns
        ], dim=-1)
        
        # Predict interactions for each LVN level
        interaction_predictions = []
        for i in range(n_levels):
            lvn_feature = lvn_embeddings[:, i, :]
            combined = torch.cat([temporal_features, lvn_feature], dim=-1)
            interaction_pred = self.interaction_classifier(combined)
            interaction_predictions.append(interaction_pred)
            
        interaction_predictions = torch.stack(interaction_predictions, dim=1)
        
        return temporal_features[:, :self.hidden_dim], interaction_predictions


class AttentionRelevanceModule(nn.Module):
    """Dynamic attention mechanism for LVN relevance scoring."""
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        
        # Query from current market state
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Key/Value from LVN features
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Relevance scoring
        self.relevance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self,
        market_state: torch.Tensor,
        lvn_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate attention-based relevance scores for LVN levels.
        
        Args:
            market_state: [batch, hidden_dim]
            lvn_features: [batch, n_levels, hidden_dim]
            mask: Optional mask for invalid levels
            
        Returns:
            weighted_features: [batch, hidden_dim]
            relevance_scores: [batch, n_levels]
        """
        batch_size = market_state.shape[0]
        n_levels = lvn_features.shape[1]
        
        # Project query, key, value
        query = self.query_proj(market_state).unsqueeze(1)  # [batch, 1, hidden_dim]
        keys = self.key_proj(lvn_features)  # [batch, n_levels, hidden_dim]
        values = self.value_proj(lvn_features)  # [batch, n_levels, hidden_dim]
        
        # Calculate attention scores
        scores = torch.bmm(query, keys.transpose(1, 2))  # [batch, 1, n_levels]
        scores = scores / np.sqrt(self.query_proj.out_features)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        
        # Calculate relevance scores (different from attention)
        relevance_scores = []
        for i in range(n_levels):
            lvn_feat = lvn_features[:, i, :]
            relevance = self.relevance_head(lvn_feat)
            relevance_scores.append(relevance)
            
        relevance_scores = torch.cat(relevance_scores, dim=-1)  # [batch, n_levels]
        
        # Combine attention and relevance
        combined_weights = attention_weights.squeeze(1) * relevance_scores
        combined_weights = combined_weights / (combined_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Weighted sum of values
        weighted_features = torch.bmm(
            combined_weights.unsqueeze(1),
            values
        ).squeeze(1)
        
        weighted_features = self.output_proj(weighted_features)
        
        return weighted_features, relevance_scores


class LVNEmbedder(nn.Module, MCDropoutMixin):
    """
    Advanced LVN Embedder with spatial awareness and temporal tracking.
    
    Transforms LVN data into rich embeddings for trading decisions.
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        output_dim: int = 32,
        hidden_dim: int = 64,
        max_levels: int = 10,
        price_history_length: int = 20,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.max_levels = max_levels
        
        # Base embedding for LVN features
        self.base_embedder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Price trajectory encoder
        self.price_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Core modules
        self.spatial_module = SpatialRelationshipModule(hidden_dim)
        self.temporal_module = TemporalTrackingModule(hidden_dim, price_history_length)
        self.attention_module = AttentionRelevanceModule(hidden_dim)
        
        # Market state encoder
        self.market_state_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Final output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Uncertainty quantification
        self.uncertainty_head = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, 1),
            nn.Softplus()
        )
        
    def forward(
        self,
        lvn_context: LVNContext,
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of LVN embedder.
        
        Args:
            lvn_context: LVN context data
            return_intermediates: Whether to return intermediate features
            
        Returns:
            Dictionary containing embeddings and auxiliary outputs
        """
        batch_size = 1 if len(lvn_context.prices.shape) == 1 else lvn_context.prices.shape[0]
        
        # Prepare inputs
        if len(lvn_context.prices.shape) == 1:
            prices = lvn_context.prices.unsqueeze(0)
            strengths = lvn_context.strengths.unsqueeze(0)
            distances = lvn_context.distances.unsqueeze(0)
            volumes = lvn_context.volumes.unsqueeze(0)
            price_history = lvn_context.price_history.unsqueeze(0)
        else:
            prices = lvn_context.prices
            strengths = lvn_context.strengths
            distances = lvn_context.distances
            volumes = lvn_context.volumes
            price_history = lvn_context.price_history
            
        n_levels = prices.shape[1]
        
        # Pad to max_levels if necessary
        if n_levels < self.max_levels:
            pad_size = self.max_levels - n_levels
            prices = F.pad(prices, (0, pad_size), value=0)
            strengths = F.pad(strengths, (0, pad_size), value=0)
            distances = F.pad(distances, (0, pad_size), value=1e6)  # Large distance
            volumes = F.pad(volumes, (0, pad_size), value=0)
            
        # Create mask for padded values
        mask = torch.zeros(batch_size, self.max_levels, dtype=torch.bool)
        mask[:, n_levels:] = True
        
        # Normalize inputs
        prices_norm = (prices - lvn_context.current_price) / (lvn_context.current_price + 1e-8)
        strengths_norm = strengths / 100.0
        distances_norm = distances / (distances.max() + 1e-8)
        volumes_norm = volumes / (volumes.max() + 1e-8)
        
        # Combine features
        lvn_features = torch.stack([
            prices_norm,
            strengths_norm,
            distances_norm,
            volumes_norm,
            torch.sign(prices_norm)  # Above/below current price
        ], dim=-1)  # [batch, max_levels, 5]
        
        # Base embedding
        lvn_embeddings = self.base_embedder(lvn_features)  # [batch, max_levels, hidden_dim]
        
        # Spatial relationships
        spatial_features = self.spatial_module(
            lvn_embeddings,
            distances_norm.unsqueeze(-1),
            strengths_norm.unsqueeze(-1)
        )
        
        # Encode price history
        price_history_encoded = self.price_encoder(price_history.unsqueeze(-1))
        
        # Temporal tracking
        temporal_features, interaction_preds = self.temporal_module(
            price_history_encoded,
            spatial_features
        )
        
        # Current market state
        market_state = self.market_state_encoder(
            torch.cat([
                temporal_features,
                price_history_encoded[:, -1, :]  # Most recent price
            ], dim=-1)
        )
        
        # Attention-based relevance
        attended_features, relevance_scores = self.attention_module(
            market_state,
            spatial_features,
            mask
        )
        
        # Combine all features
        combined_features = torch.cat([
            market_state,
            attended_features,
            temporal_features
        ], dim=-1)
        
        # Final projection
        output_embedding = self.output_projection(combined_features)
        
        # Uncertainty quantification
        uncertainty = self.uncertainty_head(output_embedding)
        
        results = {
            'embedding': output_embedding,
            'uncertainty': uncertainty,
            'relevance_scores': relevance_scores[:, :n_levels],  # Remove padding
            'interaction_predictions': interaction_preds[:, :n_levels, :],
            'spatial_features': spatial_features[:, :n_levels, :] if return_intermediates else None,
            'temporal_features': temporal_features if return_intermediates else None
        }
        
        # Remove None values
        results = {k: v for k, v in results.items() if v is not None}
        
        return results
        
    def get_top_levels(
        self,
        lvn_context: LVNContext,
        k: int = 3
    ) -> Dict[str, torch.Tensor]:
        """
        Get top-k most relevant LVN levels.
        
        Args:
            lvn_context: LVN context data
            k: Number of top levels to return
            
        Returns:
            Dictionary with top levels and their features
        """
        results = self.forward(lvn_context, return_intermediates=True)
        
        # Get relevance scores
        relevance_scores = results['relevance_scores']
        
        # Get top-k indices
        _, top_indices = torch.topk(relevance_scores, k=min(k, relevance_scores.shape[1]), dim=1)
        
        # Extract top levels
        batch_size = relevance_scores.shape[0]
        top_levels = {
            'indices': top_indices,
            'prices': torch.gather(lvn_context.prices, 1, top_indices),
            'strengths': torch.gather(lvn_context.strengths, 1, top_indices),
            'distances': torch.gather(lvn_context.distances, 1, top_indices),
            'relevance_scores': torch.gather(relevance_scores, 1, top_indices)
        }
        
        return top_levels