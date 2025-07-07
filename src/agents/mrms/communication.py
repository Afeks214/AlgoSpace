"""
MRMS Communication LSTM Layer

Provides temporal coherence and uncertainty estimation for risk management decisions.
"""

from typing import Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskMemory:
    """Stores recent risk decisions and outcomes"""
    recent_stops: torch.Tensor      # Circular buffer of stop outcomes
    recent_targets: torch.Tensor    # Circular buffer of target outcomes
    recent_sizes: torch.Tensor      # Recent position sizes
    performance_stats: torch.Tensor # Rolling performance metrics
    

class MRMSCommunicationLSTM(nn.Module):
    """
    LSTM layer that adds temporal context and uncertainty to MRMS decisions.
    
    This layer maintains memory of recent trading outcomes and adapts
    risk parameters based on performance patterns.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Dimensions
        self.input_dim = config.get('risk_vector_dim', 4)
        self.outcome_dim = config.get('outcome_dim', 3)
        self.hidden_dim = config.get('hidden_dim', 16)
        self.output_dim = config.get('output_dim', 8)
        self.memory_size = config.get('memory_size', 20)
        
        # LSTM for risk proposal processing
        self.risk_lstm = nn.LSTM(
            input_size=self.input_dim + self.outcome_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Performance context encoder
        self.performance_encoder = nn.Sequential(
            nn.Linear(self.memory_size * 3, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Output projections with uncertainty
        self.mu_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 16, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.Tanh()  # Bounded output
        )
        
        self.sigma_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 16, self.output_dim),
            nn.Softplus(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.Softplus()
        )
        
        # Risk memory
        self.risk_memory = self._initialize_memory()
        
        # Hidden state
        self.hidden = None
        
    def _initialize_memory(self) -> RiskMemory:
        """Initialize risk memory buffers"""
        return RiskMemory(
            recent_stops=torch.zeros(self.memory_size),
            recent_targets=torch.zeros(self.memory_size),
            recent_sizes=torch.zeros(self.memory_size),
            performance_stats=torch.zeros(4)  # [win_rate, avg_rr, sharpe, drawdown]
        )
    
    def forward(
        self, 
        risk_vector: torch.Tensor,
        recent_outcome: torch.Tensor,
        update_memory: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process risk proposal with temporal context.
        
        Args:
            risk_vector: Current risk proposal [batch, 4]
            recent_outcome: Recent trade outcome [batch, 3]
            update_memory: Whether to update memory buffers
            
        Returns:
            mu_risk: Risk embedding with context [batch, 8]
            sigma_risk: Uncertainty estimates [batch, 8]
        """
        # Combine risk and outcome
        combined_input = torch.cat([risk_vector, recent_outcome], dim=-1)
        
        # Add sequence dimension
        if combined_input.dim() == 2:
            combined_input = combined_input.unsqueeze(1)
        
        # Process through LSTM
        lstm_out, self.hidden = self.risk_lstm(combined_input, self.hidden)
        lstm_features = lstm_out[:, -1, :]  # Take last timestep
        
        # Get performance context
        perf_context = self._get_performance_context()
        
        # Combine LSTM features with performance context
        combined_features = torch.cat([lstm_features, perf_context], dim=-1)
        
        # Generate mean and uncertainty
        mu_risk = self.mu_head(combined_features)
        sigma_risk = self.sigma_head(combined_features) + 1e-6
        
        # Update memory if requested
        if update_memory:
            self._update_memory(risk_vector, recent_outcome)
        
        return mu_risk, sigma_risk
    
    def _get_performance_context(self) -> torch.Tensor:
        """Extract performance patterns from memory"""
        # Flatten memory buffers
        perf_features = torch.cat([
            self.risk_memory.recent_stops,
            self.risk_memory.recent_targets,
            self.risk_memory.recent_sizes
        ], dim=-1)
        
        # Add batch dimension if needed
        if perf_features.dim() == 1:
            perf_features = perf_features.unsqueeze(0)
        
        # Encode performance context
        return self.performance_encoder(perf_features)
    
    def _update_memory(
        self, 
        risk_vector: torch.Tensor, 
        outcome: torch.Tensor
    ) -> None:
        """Update risk memory with latest outcome"""
        # Shift buffers
        self.risk_memory.recent_stops[:-1] = self.risk_memory.recent_stops[1:].clone()
        self.risk_memory.recent_targets[:-1] = self.risk_memory.recent_targets[1:].clone()
        self.risk_memory.recent_sizes[:-1] = self.risk_memory.recent_sizes[1:].clone()
        
        # Add new data
        self.risk_memory.recent_stops[-1] = outcome[0, 0]  # Hit stop?
        self.risk_memory.recent_targets[-1] = outcome[0, 1]  # Hit target?
        self.risk_memory.recent_sizes[-1] = risk_vector[0, 0]  # Position size
        
        # Update performance stats
        self._update_performance_stats()
    
    def _update_performance_stats(self) -> None:
        """Update rolling performance statistics"""
        # Calculate win rate
        wins = self.risk_memory.recent_targets.sum()
        total = (self.risk_memory.recent_stops + self.risk_memory.recent_targets).sum()
        if total > 0:
            self.risk_memory.performance_stats[0] = wins / total
        
        # Calculate average risk-reward
        if self.risk_memory.recent_targets.sum() > 0:
            avg_rr = 2.0  # Default RR, can be made dynamic
            self.risk_memory.performance_stats[1] = avg_rr
        
        # Simplified Sharpe approximation
        returns = self.risk_memory.recent_targets * 2.0 - self.risk_memory.recent_stops
        if returns.std() > 0:
            self.risk_memory.performance_stats[2] = returns.mean() / returns.std()
        
        # Track consecutive losses for drawdown
        consecutive_losses = 0
        for i in range(len(self.risk_memory.recent_stops) - 1, -1, -1):
            if self.risk_memory.recent_stops[i] == 1:
                consecutive_losses += 1
            else:
                break
        self.risk_memory.performance_stats[3] = consecutive_losses / self.memory_size
    
    def _adapt_risk_parameters(
        self, 
        base_position_size: float,
        uncertainty: float
    ) -> float:
        """Adapt position size based on performance and uncertainty"""
        # Base adaptation from performance
        win_rate = self.risk_memory.performance_stats[0].item()
        drawdown = self.risk_memory.performance_stats[3].item()
        
        # Reduce size if losing
        if win_rate < 0.4:
            size_multiplier = 0.5 + 0.5 * (win_rate / 0.4)
        else:
            size_multiplier = 1.0
        
        # Further reduction based on drawdown
        if drawdown > 0.2:
            size_multiplier *= (1.0 - min(drawdown, 0.5))
        
        # Uncertainty adjustment
        if uncertainty > 0.3:
            size_multiplier *= (1.0 - min(uncertainty - 0.3, 0.3))
        
        return base_position_size * size_multiplier
    
    def reset_hidden_state(self) -> None:
        """Reset LSTM hidden state for new sequences"""
        self.hidden = None
    
    def get_recent_outcome_vector(self) -> torch.Tensor:
        """Get recent outcome vector for external use"""
        # Calculate recent performance metrics
        recent_stops = self.risk_memory.recent_stops[-5:].mean()
        recent_targets = self.risk_memory.recent_targets[-5:].mean()
        recent_pnl = recent_targets * 2.0 - recent_stops  # Simplified PnL
        
        return torch.tensor([
            recent_stops.item(),
            recent_targets.item(),
            recent_pnl.item()
        ], dtype=torch.float32).unsqueeze(0)