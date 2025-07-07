"""
File: src/agents/main_core/decision_gate_attention.py (NEW FILE)
Advanced attention mechanisms for DecisionGate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math


class HierarchicalAttention(nn.Module):
    """
    Hierarchical attention mechanism for multi-level decision factors.
    
    Attends to different levels of information:
    1. Component level (structure, tactical, regime, etc.)
    2. Risk factor level  
    3. Temporal level (if applicable)
    """
    
    def __init__(self, hidden_dim: int = 384, n_heads: int = 8):
        super().__init__()
        
        # Component-level attention
        self.component_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Risk-factor attention
        self.risk_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads // 2,
            dropout=0.1,
            batch_first=True
        )
        
        # Cross-level attention
        self.cross_level_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Gating mechanism
        self.component_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.risk_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        component_features: torch.Tensor,
        risk_features: torch.Tensor,
        global_context: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply hierarchical attention.
        
        Args:
            component_features: [batch, n_components, hidden_dim]
            risk_features: [batch, n_risks, hidden_dim]
            global_context: [batch, hidden_dim]
            
        Returns:
            Tuple of (attended_features, attention_weights)
        """
        batch_size = component_features.size(0)
        
        # Expand global context
        global_expanded = global_context.unsqueeze(1)  # [batch, 1, hidden]
        
        # Component-level attention
        comp_attended, comp_weights = self.component_attention(
            query=global_expanded,
            key=component_features,
            value=component_features
        )
        
        # Risk-factor attention
        risk_attended, risk_weights = self.risk_attention(
            query=global_expanded,
            key=risk_features,
            value=risk_features
        )
        
        # Cross-level attention
        combined_features = torch.cat([
            comp_attended,
            risk_attended
        ], dim=1)  # [batch, 2, hidden]
        
        cross_attended, cross_weights = self.cross_level_attention(
            query=global_expanded,
            key=combined_features,
            value=combined_features
        )
        
        # Gated combination
        comp_gate = self.component_gate(
            torch.cat([comp_attended.squeeze(1), global_context], dim=-1)
        )
        risk_gate = self.risk_gate(
            torch.cat([risk_attended.squeeze(1), global_context], dim=-1)
        )
        
        # Final combination
        final_features = (
            comp_gate * comp_attended.squeeze(1) +
            risk_gate * risk_attended.squeeze(1) +
            cross_attended.squeeze(1)
        ) / 3.0
        
        attention_dict = {
            'component_weights': comp_weights,
            'risk_weights': risk_weights,
            'cross_weights': cross_weights,
            'component_gate': comp_gate,
            'risk_gate': risk_gate
        }
        
        return final_features, attention_dict


class FactorizedAttention(nn.Module):
    """
    Factorized attention for efficient computation of risk factors.
    
    Decomposes attention into:
    1. Factor embeddings
    2. Factor interactions
    3. Factor aggregation
    """
    
    def __init__(self, hidden_dim: int = 384, n_factors: int = 16):
        super().__init__()
        
        self.n_factors = n_factors
        self.factor_dim = hidden_dim // n_factors
        
        # Factor embedding
        self.factor_embed = nn.Linear(hidden_dim, n_factors * self.factor_dim)
        
        # Factor interaction
        self.factor_interaction = nn.Parameter(
            torch.randn(n_factors, n_factors) * 0.01
        )
        
        # Factor aggregation
        self.factor_aggregate = nn.Sequential(
            nn.Linear(n_factors * self.factor_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply factorized attention.
        
        Returns:
            Tuple of (output, factor_scores)
        """
        batch_size = x.size(0)
        
        # Embed into factors
        factors = self.factor_embed(x)  # [batch, n_factors * factor_dim]
        factors = factors.view(batch_size, self.n_factors, self.factor_dim)
        
        # Compute factor scores
        factor_norms = torch.norm(factors, dim=-1)  # [batch, n_factors]
        factor_scores = F.softmax(factor_norms, dim=-1)
        
        # Factor interactions
        interaction_weights = F.softmax(self.factor_interaction, dim=-1)
        
        # Apply interactions
        interacted_factors = []
        for i in range(self.n_factors):
            weighted_factors = []
            for j in range(self.n_factors):
                weight = interaction_weights[i, j]
                weighted_factors.append(weight * factors[:, j, :])
            
            interacted = torch.stack(weighted_factors).sum(dim=0)
            interacted_factors.append(interacted)
            
        interacted_factors = torch.stack(interacted_factors, dim=1)
        
        # Flatten and aggregate
        flat_factors = interacted_factors.view(batch_size, -1)
        output = self.factor_aggregate(flat_factors)
        
        return output, factor_scores


class CausalAttention(nn.Module):
    """
    Causal attention for sequential decision dependencies.
    
    Ensures decisions consider causal relationships between factors.
    """
    
    def __init__(self, hidden_dim: int = 384, max_seq_len: int = 20):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # Causal mask
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        )
        
        # Attention components
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = 1.0 / math.sqrt(hidden_dim)
        
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply causal attention.
        
        Args:
            x: Input sequence [batch, seq_len, hidden_dim]
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))
        
        # Apply additional mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))
            
        # Compute attention weights
        weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(weights, v)
        
        # Output projection
        output = self.out_proj(attended)
        
        return output, weights