"""
Base trading agent class for MARL system.

Provides the foundation architecture that all specialized agents
(Structure Analyzer, Short-term Tactician, Mid-frequency Arbitrageur)
build upon.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

import structlog

from .embedders import SharedEmbedder, TemporalAttention

logger = structlog.get_logger()


class BaseTradeAgent(nn.Module, ABC):
    """
    Abstract base class for all trading agents in the MARL system.
    
    Provides:
    - Shared embedding architecture
    - Temporal attention mechanism
    - Communication interfaces
    - State management
    - Abstract methods for specialization
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base trading agent.
        
        Args:
            config: Agent configuration containing:
                - window: Time window size
                - hidden_dim: Hidden layer dimension
                - n_layers: Number of layers
                - dropout: Dropout rate
                - input_features: Number of input features
        """
        super().__init__()
        self.config = config
        self.window = config['window']
        self.hidden_dim = config['hidden_dim']
        self.dropout_rate = config['dropout']
        
        # Shared embedding architecture
        self.embedder = SharedEmbedder(
            input_features=config['input_features'],
            hidden_dim=self.hidden_dim,
            dropout=self.dropout_rate
        )
        
        # Temporal attention mechanism
        self.temporal_attention = TemporalAttention(
            embed_dim=256,  # Output from embedder
            num_heads=8,
            dropout=self.dropout_rate
        )
        
        # Agent-specific policy head (implemented by subclasses)
        self.policy_head = self._build_policy_head()
        
        # Communication state
        self.communication_state = None
        self.last_message = None
        
        # Performance tracking
        self.decision_count = 0
        self.confidence_history = []
        
        logger.info(
            f"Initialized {self.__class__.__name__}",
            window=self.window,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout_rate
        )
    
    @abstractmethod
    def _build_policy_head(self) -> nn.Module:
        """
        Build agent-specific policy head.
        
        Must be implemented by each specialized agent.
        
        Returns:
            Policy head module
        """
        pass
    
    @abstractmethod
    def _encode_synergy(self, synergy_context: Dict[str, Any]) -> torch.Tensor:
        """
        Extract agent-specific features from synergy context.
        
        Must be implemented by each specialized agent.
        
        Args:
            synergy_context: Synergy detection context
            
        Returns:
            Encoded features tensor
        """
        pass
    
    def forward(
        self,
        market_matrix: torch.Tensor,
        regime_vector: torch.Tensor,
        synergy_context: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the agent.
        
        Args:
            market_matrix: Market data matrix [batch, time, features]
            regime_vector: Regime context vector [batch, 8]
            synergy_context: Synergy detection context
            
        Returns:
            Dictionary containing:
                - action: Action logits/probabilities
                - confidence: Confidence score
                - reasoning: Interpretable features
                - attention_weights: Temporal attention weights
        """
        # Ensure inputs are tensors
        if not isinstance(market_matrix, torch.Tensor):
            market_matrix = torch.tensor(market_matrix, dtype=torch.float32)
        if not isinstance(regime_vector, torch.Tensor):
            regime_vector = torch.tensor(regime_vector, dtype=torch.float32)
            
        # Process market data through embedder
        # Transpose for Conv1D: [batch, features, time]
        x = market_matrix.transpose(1, 2)
        embedded = self.embedder(x)
        
        # Transpose back for attention: [batch, time, features]
        embedded = embedded.transpose(1, 2)
        
        # Apply temporal attention
        attended, attention_weights = self.temporal_attention(embedded)
        
        # Global pooling over time dimension
        pooled = torch.mean(attended, dim=1)  # [batch, features]
        
        # Encode synergy context
        synergy_features = self._encode_synergy(synergy_context)
        if synergy_features.dim() == 1:
            synergy_features = synergy_features.unsqueeze(0)  # Add batch dim
        
        # Ensure batch dimensions match
        if pooled.size(0) != synergy_features.size(0):
            synergy_features = synergy_features.expand(pooled.size(0), -1)
        if pooled.size(0) != regime_vector.size(0):
            regime_vector = regime_vector.expand(pooled.size(0), -1)
        
        # Concatenate all context
        context = torch.cat([
            pooled,
            regime_vector,
            synergy_features
        ], dim=-1)
        
        # Generate decision through policy head
        decision = self.policy_head(context)
        
        # Add attention weights to output
        decision['attention_weights'] = attention_weights
        
        # Update tracking
        self.decision_count += 1
        if 'confidence' in decision:
            conf_value = decision['confidence'].detach().cpu().item()
            self.confidence_history.append(conf_value)
        
        return decision
    
    def get_hidden_state(
        self,
        inputs: Dict[str, torch.Tensor],
        regime_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Get hidden state representation for communication.
        
        Args:
            inputs: Agent inputs
            regime_vector: Regime context
            
        Returns:
            Hidden state tensor for communication
        """
        # Process through embedder and attention
        market_matrix = inputs['market_matrix']
        x = market_matrix.transpose(1, 2)
        embedded = self.embedder(x)
        embedded = embedded.transpose(1, 2)
        
        attended, _ = self.temporal_attention(embedded)
        pooled = torch.mean(attended, dim=1)
        
        # Store as communication state
        self.communication_state = pooled
        
        return pooled
    
    def update_state(self, communicated_state: torch.Tensor):
        """
        Update internal state after communication.
        
        Args:
            communicated_state: State after communication rounds
        """
        self.communication_state = communicated_state
    
    def reset_communication(self):
        """Reset communication state between decisions."""
        self.communication_state = None
        self.last_message = None
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get agent performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = {
            'decision_count': self.decision_count,
            'avg_confidence': (
                sum(self.confidence_history) / len(self.confidence_history)
                if self.confidence_history else 0.0
            ),
            'confidence_std': (
                torch.std(torch.tensor(self.confidence_history)).item()
                if len(self.confidence_history) > 1 else 0.0
            )
        }
        
        return metrics
    
    def train_mode(self):
        """Set agent to training mode (enables dropout)."""
        self.train()
    
    def eval_mode(self):
        """Set agent to evaluation mode (disables dropout)."""
        self.eval()
    
    def save_checkpoint(self, path: str):
        """
        Save agent checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'state_dict': self.state_dict(),
            'config': self.config,
            'metrics': self.get_metrics(),
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """
        Load agent checkpoint.
        
        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['state_dict'])
        logger.info(
            f"Loaded checkpoint from {path}",
            timestamp=checkpoint.get('timestamp')
        )