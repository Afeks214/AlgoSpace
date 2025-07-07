"""
RDE Communication LSTM Layer

Provides temporal coherence and uncertainty quantification for regime vectors
from the Regime Detection Engine (RDE).
"""

from typing import Dict, Tuple, Optional, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import logging
import time

logger = logging.getLogger(__name__)


class StateManager:
    """
    Manages hidden state persistence and recovery for RDE Communication LSTM.
    """
    
    def __init__(self, hidden_dim: int, num_layers: int, device: torch.device):
        """
        Initialize state manager.
        
        Args:
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            device: Torch device (CPU/GPU)
        """
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        
        # State buffer for multi-step predictions
        self.state_buffer = deque(maxlen=10)
        
        # Checkpointing
        self.checkpoint_state = None
        self.checkpoint_time = None
        
    def save_state(self, hidden: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Save current hidden state."""
        h, c = hidden
        self.checkpoint_state = (h.clone().detach(), c.clone().detach())
        self.checkpoint_time = time.time()
        
    def load_state(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Load saved hidden state."""
        if self.checkpoint_state is None:
            return None
        h, c = self.checkpoint_state
        return (h.clone(), c.clone())
    
    def validate_state(self, hidden: Tuple[torch.Tensor, torch.Tensor]) -> bool:
        """
        Validate hidden state for corruption.
        
        Args:
            hidden: LSTM hidden state tuple (h, c)
            
        Returns:
            True if state is valid, False if corrupted
        """
        h, c = hidden
        
        # Check for NaN or Inf
        if torch.isnan(h).any() or torch.isinf(h).any():
            logger.warning("Hidden state h contains NaN or Inf")
            return False
            
        if torch.isnan(c).any() or torch.isinf(c).any():
            logger.warning("Cell state c contains NaN or Inf")
            return False
            
        # Check for extreme values (potential gradient explosion)
        h_norm = torch.norm(h)
        c_norm = torch.norm(c)
        
        if h_norm > 100.0:
            logger.warning(f"Hidden state norm too high: {h_norm.item():.2f}")
            return False
            
        if c_norm > 100.0:
            logger.warning(f"Cell state norm too high: {c_norm.item():.2f}")
            return False
            
        return True
    
    def interpolate_states(
        self,
        state1: Tuple[torch.Tensor, torch.Tensor],
        state2: Tuple[torch.Tensor, torch.Tensor],
        alpha: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Interpolate between two hidden states.
        
        Args:
            state1: First hidden state
            state2: Second hidden state  
            alpha: Interpolation factor (0 = state1, 1 = state2)
            
        Returns:
            Interpolated hidden state
        """
        h1, c1 = state1
        h2, c2 = state2
        
        h_interp = (1 - alpha) * h1 + alpha * h2
        c_interp = (1 - alpha) * c1 + alpha * c2
        
        return (h_interp, c_interp)


class RDECommunicationLSTM(nn.Module):
    """
    LSTM layer for temporal coherence and uncertainty estimation
    over RDE regime vectors.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Core dimensions
        self.input_dim = config.get('input_dim', 8)      # RDE output
        self.hidden_dim = config.get('hidden_dim', 32)   # LSTM hidden
        self.output_dim = config.get('output_dim', 16)   # Final embedding
        self.num_layers = config.get('num_layers', 2)    # LSTM layers
        
        # Device handling
        self.device = torch.device(config.get('device', 'cpu'))
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.1 if self.num_layers > 1 else 0.0
        )
        
        # Output projections
        self.mu_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.LayerNorm(self.output_dim)
        )
        
        self.sigma_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Softplus()  # Ensures σ > 0
        )
        
        # Learnable initial hidden state
        self.h0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_dim))
        self.c0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_dim))
        
        # State manager
        self.state_manager = StateManager(self.hidden_dim, self.num_layers, self.device)
        
        # Current hidden state
        self.hidden = None
        
        # Performance monitoring
        self.inference_times = deque(maxlen=100)
        
        logger.info(f"RDE Communication LSTM initialized: "
                   f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
                   f"output_dim={self.output_dim}, device={self.device}")
        
    def forward(
        self,
        regime_vector: torch.Tensor,
        return_hidden: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process regime vector through LSTM.
        
        Args:
            regime_vector: RDE output vector [batch_size, 8]
            return_hidden: Whether to return hidden state
            
        Returns:
            mu_regime: Mean embedding [batch_size, 16]
            sigma_regime: Uncertainty (log variance) [batch_size, 16]
            hidden (optional): LSTM hidden state if return_hidden=True
        """
        start_time = time.time()
        
        # Ensure correct shape
        if regime_vector.dim() == 1:
            regime_vector = regime_vector.unsqueeze(0)
        
        batch_size = regime_vector.size(0)
        
        # Add sequence dimension if needed
        if regime_vector.dim() == 2:
            regime_vector = regime_vector.unsqueeze(1)  # [batch, 1, features]
        
        # Initialize or reuse hidden state
        if self.hidden is None:
            h0 = self.h0.expand(self.num_layers, batch_size, self.hidden_dim).contiguous()
            c0 = self.c0.expand(self.num_layers, batch_size, self.hidden_dim).contiguous()
            self.hidden = (h0, c0)
        else:
            # Adjust hidden state for batch size if needed
            h, c = self.hidden
            if h.size(1) != batch_size:
                # Reset hidden state if batch size changed
                h0 = self.h0.expand(self.num_layers, batch_size, self.hidden_dim).contiguous()
                c0 = self.c0.expand(self.num_layers, batch_size, self.hidden_dim).contiguous()
                self.hidden = (h0, c0)
        
        # Validate hidden state
        if not self.state_manager.validate_state(self.hidden):
            logger.warning("Hidden state corrupted, resetting")
            self.reset_hidden_state(batch_size)
        
        # LSTM forward pass
        lstm_out, self.hidden = self.lstm(regime_vector, self.hidden)
        
        # Extract features from last timestep
        features = lstm_out[:, -1, :]  # [batch, hidden_dim]
        
        # Generate mean and uncertainty
        mu_regime = self.mu_head(features)
        sigma_regime = self.sigma_head(features) + 1e-6  # Numerical stability
        
        # Track inference time
        inference_time = (time.time() - start_time) * 1000  # ms
        self.inference_times.append(inference_time)
        
        if return_hidden:
            return mu_regime, sigma_regime, self.hidden
        else:
            return mu_regime, sigma_regime
    
    def reset_hidden_state(self, batch_size: Optional[int] = None) -> None:
        """
        Reset hidden state to learned initial values.
        
        Args:
            batch_size: Batch size for hidden state
        """
        if batch_size is None:
            self.hidden = None
        else:
            h0 = self.h0.expand(self.num_layers, batch_size, self.hidden_dim).contiguous()
            c0 = self.c0.expand(self.num_layers, batch_size, self.hidden_dim).contiguous()
            self.hidden = (h0, c0)
            
        logger.debug("Hidden state reset")
    
    def save_checkpoint(self) -> None:
        """Save current hidden state as checkpoint."""
        if self.hidden is not None:
            self.state_manager.save_state(self.hidden)
            
    def load_checkpoint(self) -> bool:
        """
        Load hidden state from checkpoint.
        
        Returns:
            True if checkpoint loaded successfully
        """
        checkpoint = self.state_manager.load_state()
        if checkpoint is not None:
            self.hidden = checkpoint
            return True
        return False
    
    def get_uncertainty_metrics(self) -> Dict[str, float]:
        """
        Get uncertainty-related metrics for monitoring.
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Inference latency statistics
        if self.inference_times:
            times = list(self.inference_times)
            metrics['inference_latency_mean_ms'] = np.mean(times)
            metrics['inference_latency_p95_ms'] = np.percentile(times, 95)
            metrics['inference_latency_p99_ms'] = np.percentile(times, 99)
        
        # Hidden state statistics
        if self.hidden is not None:
            h, c = self.hidden
            metrics['hidden_norm'] = torch.norm(h).item()
            metrics['cell_norm'] = torch.norm(c).item()
            
        return metrics
    
    def multi_step_prediction(
        self,
        initial_regime: torch.Tensor,
        n_steps: int = 3
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate multi-step regime predictions.
        
        Args:
            initial_regime: Starting regime vector
            n_steps: Number of steps to predict
            
        Returns:
            List of (mu, sigma) tuples for each step
        """
        predictions = []
        
        # Save current state
        original_hidden = self.hidden
        
        # Generate predictions
        regime = initial_regime
        for _ in range(n_steps):
            mu, sigma = self.forward(regime)
            predictions.append((mu.detach(), sigma.detach()))
            
            # Use mean as next input (autoregressive)
            # Project back to input dimension
            regime = F.linear(mu, torch.randn(self.input_dim, self.output_dim).to(self.device))
        
        # Restore original state
        self.hidden = original_hidden
        
        return predictions
    
    def estimate_regime_transition_probability(
        self,
        regime1: torch.Tensor,
        regime2: torch.Tensor
    ) -> float:
        """
        Estimate probability of transitioning between regimes.
        
        Args:
            regime1: Current regime vector
            regime2: Target regime vector
            
        Returns:
            Transition probability estimate
        """
        # Process current regime
        mu1, sigma1 = self.forward(regime1)
        
        # Process target regime
        mu2, sigma2 = self.forward(regime2)
        
        # Calculate distance in embedding space
        distance = torch.norm(mu2 - mu1)
        
        # Account for uncertainty
        avg_uncertainty = (sigma1.mean() + sigma2.mean()) / 2
        
        # Convert to probability (inverse relationship with distance)
        transition_prob = torch.exp(-distance / (1 + avg_uncertainty))
        
        return transition_prob.item()
    
    def to(self, device: torch.device):
        """Override to() to handle state manager."""
        super().to(device)
        self.device = device
        self.state_manager.device = device
        
        # Move hidden state if exists
        if self.hidden is not None:
            h, c = self.hidden
            self.hidden = (h.to(device), c.to(device))
            
        return self