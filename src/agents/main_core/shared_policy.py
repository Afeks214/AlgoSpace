"""
File: src/agents/main_core/shared_policy.py (NEW FILE)
Complete production-ready shared policy implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import math

logger = logging.getLogger(__name__)

@dataclass
class PolicyOutput:
    """Structured output from shared policy."""
    action_logits: torch.Tensor
    action_probs: torch.Tensor
    state_value: Optional[torch.Tensor] = None
    policy_features: Optional[torch.Tensor] = None
    attention_weights: Optional[Dict[str, torch.Tensor]] = None
    reasoning_scores: Optional[Dict[str, float]] = None

class CrossFeatureAttention(nn.Module):
    """
    Cross-attention between different embedder outputs.
    Models interactions between structure, tactical, regime, and LVN features.
    """
    
    def __init__(self, embed_dims: List[int], n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dims = embed_dims
        self.n_heads = n_heads
        
        # Project each embedding to common dimension
        self.common_dim = 64
        self.projections = nn.ModuleList([
            nn.Linear(dim, self.common_dim) for dim in embed_dims
        ])
        
        # Cross-attention layers
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.common_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.common_dim * len(embed_dims), 128)
        
    def forward(self, embeddings: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-attention between embeddings.
        
        Args:
            embeddings: List of [batch_size, embed_dim] tensors
            
        Returns:
            Tuple of (fused_features, attention_weights)
        """
        batch_size = embeddings[0].size(0)
        
        # Project to common dimension
        projected = []
        for i, (embed, proj) in enumerate(zip(embeddings, self.projections)):
            projected.append(proj(embed))
            
        # Stack for attention [batch, n_embeds, common_dim]
        stacked = torch.stack(projected, dim=1)
        
        # Self-attention across embeddings
        attended, attention_weights = self.cross_attention(
            stacked, stacked, stacked
        )
        
        # Flatten and project
        flattened = attended.reshape(batch_size, -1)
        output = self.output_proj(flattened)
        
        return output, attention_weights

class TemporalConsistencyModule(nn.Module):
    """
    Maintains temporal consistency in decisions.
    Prevents erratic switching between actions.
    """
    
    def __init__(self, hidden_dim: int = 64, memory_size: int = 20):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        
        # Decision history encoder
        self.history_encoder = nn.LSTM(
            input_size=2 + hidden_dim,  # action_probs + features
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Consistency scorer
        self.consistency_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Decision memory
        self.decision_memory = []
        
        # Learned initial states
        self.h0 = nn.Parameter(torch.zeros(2, 1, hidden_dim))
        self.c0 = nn.Parameter(torch.zeros(2, 1, hidden_dim))
        
    def forward(self, current_features: torch.Tensor, 
                current_probs: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Score temporal consistency of current decision.
        
        Args:
            current_features: Current policy features
            current_probs: Current action probabilities
            
        Returns:
            Tuple of (consistency_features, consistency_score)
        """
        device = current_features.device
        batch_size = current_features.size(0)
        
        # Initialize hidden states
        h0 = self.h0.expand(-1, batch_size, -1).contiguous()
        c0 = self.c0.expand(-1, batch_size, -1).contiguous()
        
        if len(self.decision_memory) > 0:
            # Encode decision history
            history = torch.stack(self.decision_memory[-self.memory_size:]).to(device)
            history = history.unsqueeze(0).expand(batch_size, -1, -1)
            
            history_encoded, (hn, cn) = self.history_encoder(history, (h0, c0))
            history_features = hn[-1]
        else:
            # No history yet
            history_features = h0[-1]
            
        # Combine with current
        combined = torch.cat([history_features, current_features], dim=-1)
        
        # Score consistency
        consistency_score = self.consistency_net(combined)
        
        # Update memory (detach to prevent gradient flow)
        memory_entry = torch.cat([
            current_probs.detach(),
            current_features.detach()
        ], dim=-1)
        self.decision_memory.append(memory_entry[0])  # Store first in batch
        
        # Trim memory
        if len(self.decision_memory) > self.memory_size:
            self.decision_memory.pop(0)
            
        return history_features, consistency_score.mean()

class MultiHeadReasoner(nn.Module):
    """
    Multi-head reasoning module for different decision aspects.
    Each head specializes in a different aspect of the trading decision.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Shared encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Specialized reasoning heads
        self.heads = nn.ModuleDict({
            'structure': self._create_head('Market Structure Assessment'),
            'timing': self._create_head('Tactical Timing Quality'),
            'risk': self._create_head('Risk Conditions'),
            'regime': self._create_head('Regime Suitability')
        })
        
        # Head importance weights (learnable)
        self.head_weights = nn.Parameter(torch.ones(4) / 4)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 4, hidden_dim)
        
    def _create_head(self, name: str) -> nn.Module:
        """Create a specialized reasoning head."""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU()
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Apply multi-head reasoning.
        
        Args:
            features: Input features [batch_size, input_dim]
            
        Returns:
            Tuple of (reasoned_features, head_scores)
        """
        # Shared encoding
        shared = self.shared_encoder(features)
        
        # Apply each head
        head_outputs = {}
        head_scores = {}
        
        for name, head in self.heads.items():
            output = head(shared)
            head_outputs[name] = output
            
            # Score each head's confidence (average activation)
            score = torch.sigmoid(output.mean(dim=-1)).mean().item()
            head_scores[name] = score
            
        # Combine with learned weights
        weights = F.softmax(self.head_weights, dim=0)
        
        combined = []
        for i, name in enumerate(self.heads.keys()):
            # Pad to hidden_dim
            padded = F.pad(head_outputs[name], (0, self.hidden_dim // 2))
            weighted = padded * weights[i]
            combined.append(weighted)
            
        # Stack and project
        stacked = torch.cat(combined, dim=-1)
        output = self.output_proj(stacked)
        
        # Add weights to scores for monitoring
        head_names = list(head_scores.keys())
        for i, name in enumerate(head_names):
            head_scores[f'{name}_weight'] = weights[i].item()
            
        return output, head_scores

class ActionDistributionModule(nn.Module):
    """
    Sophisticated action distribution modeling with temperature control.
    Ensures well-calibrated probabilities for MC Dropout.
    """
    
    def __init__(self, input_dim: int, action_dim: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        
        # Pre-logit projection with bottleneck
        self.pre_logit = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),  # Critical for MC Dropout
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Action logits
        self.action_head = nn.Linear(32, action_dim)
        
        # Temperature network (learned temperature scheduling)
        self.temperature_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 0-1 range
        )
        
        # Base temperature range
        self.min_temp = 0.5
        self.max_temp = 2.0
        
    def forward(self, features: torch.Tensor, 
                use_temperature: bool = True) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Generate action distribution.
        
        Args:
            features: Input features
            use_temperature: Whether to apply learned temperature
            
        Returns:
            Tuple of (logits, probabilities, temperature)
        """
        # Generate pre-logits
        pre_logits = self.pre_logit(features)
        
        # Generate raw logits
        logits = self.action_head(pre_logits)
        
        # Calculate temperature
        if use_temperature:
            temp_score = self.temperature_net(features)
            temperature = self.min_temp + (self.max_temp - self.min_temp) * temp_score
            temperature = temperature.squeeze(-1)
            
            # Apply temperature
            scaled_logits = logits / temperature.unsqueeze(-1)
        else:
            scaled_logits = logits
            temperature = torch.ones(features.size(0)).to(features.device)
            
        # Convert to probabilities
        probs = F.softmax(scaled_logits, dim=-1)
        
        return logits, probs, temperature.mean().item()

class SharedPolicy(nn.Module):
    """
    State-of-the-art Shared Policy Network for MAPPO.
    
    Implements sophisticated decision making with multi-head reasoning,
    temporal consistency, and calibrated action distributions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Configuration
        self.input_dim = config.get('input_dim', 136)  # 64+48+16+8
        self.hidden_dim = config.get('hidden_dim', 256)
        self.action_dim = config.get('action_dim', 2)
        self.dropout_rate = config.get('dropout_rate', 0.2)
        self.use_temporal = config.get('use_temporal_consistency', True)
        
        # Embedding dimensions for cross-attention
        self.embed_dims = [64, 48, 16, 8]  # structure, tactical, regime, lvn
        
        # Components
        self.cross_attention = CrossFeatureAttention(
            self.embed_dims,
            n_heads=4,
            dropout=0.1
        )
        
        if self.use_temporal:
            self.temporal_consistency = TemporalConsistencyModule(
                hidden_dim=64
            )
        
        self.multi_head_reasoner = MultiHeadReasoner(
            input_dim=128,  # From cross-attention
            hidden_dim=self.hidden_dim
        )
        
        # Feature aggregation
        self.feature_aggregator = nn.Sequential(
            nn.Linear(self.hidden_dim + (64 if self.use_temporal else 0), self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # Action distribution
        self.action_distribution = ActionDistributionModule(
            input_dim=self.hidden_dim,
            action_dim=self.action_dim
        )
        
        # Value head for MAPPO
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized SharedPolicy with input_dim={self.input_dim}")
        
    def _initialize_weights(self):
        """Careful weight initialization for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                
    def forward(self, unified_state: torch.Tensor,
                return_value: bool = True,
                return_features: bool = False) -> PolicyOutput:
        """
        Forward pass through shared policy.
        
        Args:
            unified_state: Concatenated embeddings [batch_size, 136]
            return_value: Whether to compute state value
            return_features: Whether to return intermediate features
            
        Returns:
            PolicyOutput with action distribution and optional value
        """
        batch_size = unified_state.size(0)
        device = unified_state.device
        
        # Split unified state back into embeddings
        embeddings = []
        start_idx = 0
        for dim in self.embed_dims:
            embeddings.append(unified_state[:, start_idx:start_idx + dim])
            start_idx += dim
            
        # 1. Cross-feature attention
        cross_features, attention_weights = self.cross_attention(embeddings)
        
        # 2. Multi-head reasoning
        reasoned_features, head_scores = self.multi_head_reasoner(cross_features)
        
        # 3. Get action distribution (for temporal consistency)
        temp_logits, temp_probs, _ = self.action_distribution(
            reasoned_features, use_temperature=False
        )
        
        # 4. Temporal consistency (if enabled)
        if self.use_temporal:
            # Project reasoned features to match temporal module input
            temporal_input = self.feature_aggregator[0](reasoned_features)[:, :64]
            temporal_features, consistency_score = self.temporal_consistency(
                temporal_input, temp_probs
            )
            
            # Combine features
            combined_features = torch.cat([reasoned_features, temporal_features], dim=-1)
        else:
            combined_features = reasoned_features
            consistency_score = torch.ones(1).to(device)
            
        # 5. Final feature aggregation
        final_features = self.feature_aggregator(combined_features)
        
        # 6. Generate final action distribution
        action_logits, action_probs, temperature = self.action_distribution(
            final_features, use_temperature=True
        )
        
        # 7. Compute value (if needed)
        state_value = None
        if return_value:
            state_value = self.value_head(final_features)
            
        # Create output
        output = PolicyOutput(
            action_logits=action_logits,
            action_probs=action_probs,
            state_value=state_value,
            policy_features=final_features if return_features else None,
            attention_weights={'cross_attention': attention_weights} if return_features else None,
            reasoning_scores={
                **head_scores,
                'consistency_score': consistency_score.mean().item(),
                'temperature': temperature
            }
        )
        
        return output
        
    def get_action(self, unified_state: torch.Tensor,
                   deterministic: bool = False) -> Tuple[int, float]:
        """
        Get action for inference.
        
        Args:
            unified_state: Input state
            deterministic: Whether to use argmax (True) or sample (False)
            
        Returns:
            Tuple of (action, confidence)
        """
        with torch.no_grad():
            output = self.forward(unified_state, return_value=False)
            
            if deterministic:
                action = output.action_probs.argmax(dim=-1)
            else:
                dist = torch.distributions.Categorical(output.action_probs)
                action = dist.sample()
                
            confidence = output.action_probs[0, action].item()
            
        return action.item(), confidence
        
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Evaluate actions for MAPPO training.
        
        Args:
            states: Batch of states
            actions: Batch of actions taken
            
        Returns:
            Dictionary with log_probs, entropy, and values
        """
        output = self.forward(states, return_value=True)
        
        # Create action distribution
        dist = torch.distributions.Categorical(output.action_probs)
        
        # Calculate log probabilities
        log_probs = dist.log_prob(actions)
        
        # Calculate entropy
        entropy = dist.entropy()
        
        return {
            'log_probs': log_probs,
            'entropy': entropy,
            'values': output.state_value.squeeze(-1),
            'action_probs': output.action_probs
        }