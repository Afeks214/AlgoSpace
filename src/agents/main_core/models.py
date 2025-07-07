"""
Main MARL Core Neural Network Models.

This module contains all neural network architectures for the unified
intelligence system, including specialized embedders, the shared policy
network, and the final decision gate.
"""

from typing import Dict, Tuple, Optional, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Single transformer encoder block with pre-norm."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=False  # We'll use seq_first format
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalizations
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # MC Dropout layers (always active)
        self.mc_dropout1 = nn.Dropout(dropout)
        self.mc_dropout2 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with pre-norm architecture."""
        # Multi-head attention with residual
        attn_out = self.norm1(x)
        attn_out, _ = self.attention(attn_out, attn_out, attn_out, attn_mask=mask)
        x = x + self.mc_dropout1(attn_out)
        
        # Feed-forward with residual
        ffn_out = self.norm2(x)
        ffn_out = self.ffn(ffn_out)
        x = x + self.mc_dropout2(ffn_out)
        
        return x


class StructureEmbedder(nn.Module):
    """
    Transformer-based embedder for processing 30-minute market structure data.
    
    Replaces CNN with transformer architecture to better capture long-range
    dependencies and provides uncertainty estimates through dual output heads.
    
    Architecture:
        1. Input projection: 8 → d_model
        2. Positional encoding
        3. 3 transformer layers
        4. Global attention pooling
        5. Dual heads for μ and σ
    
    Args:
        input_channels: Number of input features (default: 8)
        output_dim: Output embedding dimension (default: 64)
        d_model: Transformer model dimension (default: 128)
        n_heads: Number of attention heads (default: 4)
        n_layers: Number of transformer layers (default: 3)
        d_ff: Feed-forward dimension (default: 512)
        dropout_rate: Dropout probability (default: 0.2)
        max_seq_len: Maximum sequence length (default: 48)
    """
    
    def __init__(
        self,
        input_channels: int = 8,
        output_dim: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout_rate: float = 0.2,
        max_seq_len: int = 48
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_channels, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout_rate)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout_rate)
            for _ in range(n_layers)
        ])
        
        # Global attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=0)
        )
        
        # Dual output heads for uncertainty quantification
        self.mu_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.sigma_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, output_dim),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(
        self, 
        x: torch.Tensor,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through structure embedder.
        
        Args:
            x: Input tensor [batch_size, seq_len=48, features=8]
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Tuple of (mu, sigma) where:
                mu: Mean embedding [batch_size, output_dim]
                sigma: Uncertainty estimates [batch_size, output_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to model dimension
        x = self.input_projection(x)  # [batch, seq, d_model]
        
        # Reshape for transformer: [seq, batch, d_model]
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x)
            
        # Global attention pooling
        # Compute attention weights over sequence
        attention_weights = self.attention_pool(x)  # [seq, batch, 1]
        
        # Weighted sum over sequence dimension
        pooled = torch.sum(x * attention_weights, dim=0)  # [batch, d_model]
        
        # Generate mean and uncertainty
        mu = self.mu_head(pooled)
        sigma = self.sigma_head(pooled) + 1e-6  # Prevent zero uncertainty
        
        if return_attention_weights:
            return mu, sigma, attention_weights.squeeze(-1).transpose(0, 1)
        
        return mu, sigma

    def enable_inference_mode(self):
        """Optimize model for inference by fusing operations."""
        # Fuse linear layers where possible
        for module in self.modules():
            if isinstance(module, nn.Sequential):
                # Check for fusable patterns
                for i in range(len(module) - 1):
                    if isinstance(module[i], nn.Linear) and isinstance(module[i+1], nn.LayerNorm):
                        # Could implement fusion here if needed
                        pass
        
        # Disable gradient computation
        for param in self.parameters():
            param.requires_grad = False
            
        # Set to eval mode
        self.eval()

    @torch.jit.export
    def forward_jit(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """JIT-compatible forward pass for production deployment."""
        # Simplified forward without optional returns
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Project and transpose
        x = self.input_projection(x)
        x = x.transpose(0, 1)
        
        # Add positional encoding (simplified for JIT)
        x = x + self.pos_encoder.pe[:seq_len, :]
        x = F.dropout(x, p=0.1, training=self.training)
        
        # Transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        # Global pooling
        attention_weights = F.softmax(self.attention_pool[0](x), dim=0)
        pooled = torch.sum(x * attention_weights, dim=0)
        
        # Dual heads
        mu = self.mu_head(pooled)
        sigma = self.sigma_head(pooled) + 1e-6
        
        return mu, sigma


class TemporalPositionEncoding(nn.Module):
    """Learnable temporal position encoding for momentum patterns."""
    
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        
        # Learnable position embeddings
        self.position_embeddings = nn.Parameter(
            torch.randn(max_len, d_model) * 0.1
        )
        
        # Momentum-aware scaling
        self.momentum_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor, momentum_signal: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add position encoding with optional momentum scaling.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            momentum_signal: Optional momentum strength [batch, seq_len]
        """
        seq_len = x.size(1)
        position_enc = self.position_embeddings[:seq_len, :].unsqueeze(0)
        
        if momentum_signal is not None:
            # Scale position encoding by momentum strength
            momentum_scale = momentum_signal.unsqueeze(-1) * self.momentum_scale
            position_enc = position_enc * momentum_scale
            
        return x + position_enc


class MultiScaleAttention(nn.Module):
    """Multi-scale temporal attention for different momentum horizons."""
    
    def __init__(self, d_model: int, n_heads: int = 4, scales: List[int] = [5, 15, 30]):
        super().__init__()
        
        self.scales = scales
        self.d_model = d_model
        
        # Attention modules for each scale
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=0.1,
                batch_first=True
            ) for _ in scales
        ])
        
        # Gated fusion mechanism
        self.scale_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            ) for _ in scales
        ])
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model * len(scales), d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply multi-scale attention.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask
        """
        batch_size, seq_len, _ = x.shape
        scale_outputs = []
        
        for scale, attention, gate in zip(self.scales, self.scale_attentions, self.scale_gates):
            # Create scale-specific view
            if seq_len >= scale:
                # Use sliding window for this scale
                scale_view = F.unfold(
                    x.transpose(1, 2).unsqueeze(2),  # [batch, d_model, 1, seq_len]
                    kernel_size=(1, scale),
                    stride=1,
                    padding=(0, scale//2)
                ).transpose(1, 2)[:, :seq_len, :]  # [batch, seq_len, d_model*scale]
                
                scale_view = scale_view.view(batch_size, seq_len, scale, self.d_model)
                scale_view = scale_view.mean(dim=2)  # Average over scale window
            else:
                scale_view = x
                
            # Apply attention at this scale
            attended, _ = attention(scale_view, scale_view, scale_view, attn_mask=mask)
            
            # Gated fusion with original
            gate_input = torch.cat([attended, x], dim=-1)
            gate_value = gate(gate_input)
            gated_output = gate_value * attended + (1 - gate_value) * x
            
            scale_outputs.append(gated_output)
            
        # Fuse all scales
        fused = torch.cat(scale_outputs, dim=-1)
        output = self.fusion(fused)
        
        return output


class TacticalEmbedder(nn.Module):
    """
    Advanced Bidirectional LSTM embedder for 5-minute tactical momentum.
    
    Processes 60×7 matrix to extract short-term momentum patterns with
    uncertainty quantification and multi-scale attention mechanisms.
    
    Architecture:
        1. Input projection with momentum feature extraction
        2. Bidirectional LSTM layers (3 layers)
        3. Multi-scale temporal attention
        4. Dual heads for μ and σ with MC Dropout
    
    Args:
        input_dim: Number of input features (default: 7)
        hidden_dim: LSTM hidden dimension (default: 128)
        output_dim: Output embedding dimension (default: 48)
        n_layers: Number of LSTM layers (default: 3)
        dropout_rate: Dropout probability (default: 0.2)
        attention_scales: Time scales for attention (default: [5, 15, 30])
    """
    
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 128,
        output_dim: int = 48,
        n_layers: int = 3,
        dropout_rate: float = 0.2,
        attention_scales: List[int] = [5, 15, 30]
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Input projection with momentum features
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Momentum feature extractor
        self.momentum_extractor = nn.Conv1d(
            input_dim, 
            hidden_dim // 4, 
            kernel_size=3, 
            padding=1
        )
        
        # Temporal position encoding
        self.position_encoder = TemporalPositionEncoding(hidden_dim)
        
        # Bidirectional LSTM layers
        self.lstm_layers = nn.ModuleList()
        for i in range(n_layers):
            input_size = hidden_dim if i == 0 else hidden_dim * 2
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    dropout=0 if i == n_layers - 1 else dropout_rate,
                    bidirectional=True
                )
            )
            
        # Layer normalization for LSTM outputs
        self.lstm_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim * 2) for _ in range(n_layers)
        ])
        
        # Multi-scale attention mechanism
        self.multi_scale_attention = MultiScaleAttention(
            d_model=hidden_dim * 2,
            n_heads=4,
            scales=attention_scales
        )
        
        # Temporal pooling with learned weights
        self.temporal_pool = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Softmax(dim=1)
        )
        
        # MC Dropout layers (always active)
        self.mc_dropout = nn.Dropout(dropout_rate)
        
        # Dual output heads for uncertainty quantification
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
                        
    def extract_momentum_signal(self, x: torch.Tensor) -> torch.Tensor:
        """Extract momentum strength signal from input."""
        # Transpose for Conv1d: [batch, features, time]
        x_t = x.transpose(1, 2)
        
        # Extract momentum features
        momentum_features = self.momentum_extractor(x_t)
        
        # Global momentum strength per timestep
        momentum_signal = torch.norm(momentum_features, dim=1)  # [batch, time]
        
        # Normalize
        momentum_signal = F.softmax(momentum_signal, dim=-1)
        
        return momentum_signal
        
    def forward(
        self, 
        x: torch.Tensor,
        return_attention_weights: bool = False,
        return_all_states: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through tactical embedder.
        
        Args:
            x: Input tensor [batch_size, seq_len=60, features=7]
            return_attention_weights: Whether to return attention weights
            return_all_states: Whether to return all hidden states
            
        Returns:
            Tuple of (mu, sigma) where:
                mu: Mean embedding [batch_size, output_dim]
                sigma: Uncertainty estimates [batch_size, output_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Extract momentum signal for position encoding
        momentum_signal = self.extract_momentum_signal(x)
        
        # Project input
        h = self.input_projection(x)  # [batch, seq, hidden]
        
        # Add position encoding with momentum awareness
        h = self.position_encoder(h, momentum_signal)
        
        # Process through LSTM layers with residual connections
        lstm_states = []
        for i, (lstm, norm) in enumerate(zip(self.lstm_layers, self.lstm_norms)):
            h_prev = h
            h, _ = lstm(h)
            h = norm(h)
            
            # Residual connection for deeper layers
            if i > 0 and h_prev.shape == h.shape:
                h = h + h_prev
                
            lstm_states.append(h)
            
        # Apply multi-scale attention
        h = self.multi_scale_attention(h)
        
        # MC Dropout for uncertainty
        h = self.mc_dropout(h)
        
        # Temporal pooling with attention
        attention_weights = self.temporal_pool(h)  # [batch, seq, 1]
        pooled = torch.sum(h * attention_weights, dim=1)  # [batch, hidden*2]
        
        # Generate mean and uncertainty through dual heads
        mu = self.mu_head(pooled)
        sigma = self.sigma_head(pooled) + 1e-6  # Prevent zero uncertainty
        
        if return_attention_weights and return_all_states:
            return mu, sigma, attention_weights.squeeze(-1), lstm_states
        elif return_attention_weights:
            return mu, sigma, attention_weights.squeeze(-1)
        elif return_all_states:
            return mu, sigma, lstm_states
        else:
            return mu, sigma
            
    def get_mc_predictions(self, x: torch.Tensor, n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get MC Dropout predictions for uncertainty estimation.
        
        Args:
            x: Input tensor
            n_samples: Number of MC samples
            
        Returns:
            Mean and std of predictions
        """
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(n_samples):
            mu, _ = self.forward(x)
            predictions.append(mu)
            
        predictions = torch.stack(predictions)
        
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        return mean, std
        
    def enable_inference_mode(self):
        """Optimize model for inference by fusing operations."""
        # Disable gradient computation
        for param in self.parameters():
            param.requires_grad = False
            
        # Fuse LSTM operations if possible
        for lstm in self.lstm_layers:
            if hasattr(torch.jit, '_script_lstm'):
                lstm = torch.jit.script(lstm)
                
        # Set to eval mode
        self.eval()
        
    @torch.jit.export  
    def forward_optimized(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimized forward pass for production."""
        batch_size = x.size(0)
        
        # Skip momentum signal extraction in optimized mode
        h = self.input_projection(x)
        
        # Simplified position encoding
        h = h + self.position_encoder.position_embeddings[:x.size(1), :].unsqueeze(0)
        
        # Process through LSTM layers
        for i, (lstm, norm) in enumerate(zip(self.lstm_layers, self.lstm_norms)):
            h, _ = lstm(h)
            h = norm(h)
            
        # Skip multi-scale attention in fast mode - use simple attention
        attention_weights = F.softmax(self.temporal_pool[0](h), dim=1)
        pooled = torch.sum(h * attention_weights.unsqueeze(-1), dim=1)
        
        # Dual heads
        mu = self.mu_head(pooled)
        sigma = self.sigma_head(pooled) + 1e-6
        
        return mu, sigma


class MomentumAnalyzer:
    """
    Analyzes momentum patterns from tactical embedder outputs.
    
    Provides interpretable momentum metrics and pattern detection
    for the 5-minute timeframe.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.momentum_history = []
        self.pattern_detectors = {
            'acceleration': self._detect_acceleration,
            'deceleration': self._detect_deceleration,
            'reversal': self._detect_reversal,
            'continuation': self._detect_continuation
        }
        
    def analyze(
        self, 
        embeddings: torch.Tensor,
        attention_weights: torch.Tensor,
        lstm_states: List[torch.Tensor]
    ) -> Dict[str, any]:
        """
        Analyze momentum patterns from embedder outputs.
        
        Args:
            embeddings: Tactical embeddings [batch, dim]
            attention_weights: Attention over time [batch, seq_len]
            lstm_states: Hidden states from each LSTM layer
            
        Returns:
            Dictionary of momentum metrics and patterns
        """
        metrics = {}
        
        # Attention-based momentum strength
        # High entropy = distributed attention = unclear momentum
        # Low entropy = focused attention = clear momentum
        attention_entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1)
        momentum_clarity = 1.0 / (1.0 + attention_entropy)
        
        metrics['momentum_clarity'] = momentum_clarity.mean().item()
        
        # Attention focus analysis
        attention_peak_idx = attention_weights.argmax(dim=-1)
        attention_peak_value = attention_weights.max(dim=-1)[0]
        
        # Recent focus (last 20 bars) vs historical
        recent_attention = attention_weights[:, -20:].sum(dim=1)
        historical_attention = attention_weights[:, :-20].sum(dim=1)
        
        metrics['recent_focus_ratio'] = (recent_attention / (historical_attention + 1e-8)).mean().item()
        metrics['attention_peak_strength'] = attention_peak_value.mean().item()
        
        # LSTM state dynamics
        if lstm_states:
            # Analyze final LSTM layer
            final_states = lstm_states[-1]  # [batch, seq, hidden*2]
            
            # State volatility (momentum stability)
            state_diff = torch.diff(final_states, dim=1)
            state_volatility = state_diff.std(dim=1).mean()
            metrics['momentum_stability'] = 1.0 / (1.0 + state_volatility.item())
            
            # Directional consistency
            state_direction = torch.sign(state_diff).float()
            direction_changes = torch.abs(torch.diff(state_direction, dim=1)).sum(dim=1)
            metrics['directional_consistency'] = 1.0 - (direction_changes / state_diff.size(1)).mean().item()
        
        # Pattern detection
        patterns_detected = []
        for pattern_name, detector in self.pattern_detectors.items():
            if detector(attention_weights, lstm_states):
                patterns_detected.append(pattern_name)
                
        metrics['patterns'] = patterns_detected
        
        # Update history
        self.momentum_history.append({
            'timestamp': torch.tensor(0.0),  # Placeholder - would use time.time() in production
            'clarity': momentum_clarity,
            'metrics': metrics
        })
        
        if len(self.momentum_history) > self.window_size:
            self.momentum_history.pop(0)
            
        return metrics
        
    def _detect_acceleration(self, attention_weights: torch.Tensor, lstm_states: List[torch.Tensor]) -> bool:
        """Detect momentum acceleration pattern."""
        # Acceleration: attention shifting to more recent bars
        recent_weight = attention_weights[:, -10:].sum(dim=1).mean()
        older_weight = attention_weights[:, 10:20].sum(dim=1).mean()
        return recent_weight > older_weight * 1.5
        
    def _detect_deceleration(self, attention_weights: torch.Tensor, lstm_states: List[torch.Tensor]) -> bool:
        """Detect momentum deceleration pattern."""
        # Deceleration: attention becoming more distributed
        entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1).mean()
        return entropy > 3.5  # High entropy threshold
        
    def _detect_reversal(self, attention_weights: torch.Tensor, lstm_states: List[torch.Tensor]) -> bool:
        """Detect potential momentum reversal."""
        if lstm_states:
            final_states = lstm_states[-1]
            # Check for state direction changes in recent timesteps
            recent_states = final_states[:, -10:, :]
            state_diff = torch.diff(recent_states, dim=1)
            direction_changes = (torch.sign(state_diff[:, :-1]) != torch.sign(state_diff[:, 1:])).float().mean()
            return direction_changes > 0.4
        return False
        
    def _detect_continuation(self, attention_weights: torch.Tensor, lstm_states: List[torch.Tensor]) -> bool:
        """Detect momentum continuation pattern."""
        # Continuation: consistent attention pattern
        attention_std = attention_weights.std(dim=0).mean()
        return attention_std < 0.15  # Low variability threshold


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


class UncertaintyAggregator:
    """
    Aggregates uncertainties from multiple embedders for decision making.
    
    Combines uncertainty estimates to create confidence scores and
    dynamic thresholds for the decision gates.
    """
    
    def __init__(self, aggregation_method: str = 'weighted_mean'):
        """
        Initialize uncertainty aggregator.
        
        Args:
            aggregation_method: How to combine uncertainties
                - 'weighted_mean': Weighted average based on importance
                - 'max': Maximum uncertainty (most conservative)
                - 'learned': Learned aggregation (requires training)
        """
        self.method = aggregation_method
        
        # Importance weights for each embedder
        self.weights = {
            'structure': 0.35,   # Market structure most important
            'tactical': 0.30,    # Tactical momentum second
            'regime': 0.25,      # Regime context third
            'lvn': 0.10          # LVN least important
        }
        
    def aggregate(self, uncertainties: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Aggregate uncertainties from all embedders.
        
        Args:
            uncertainties: Dict mapping embedder names to uncertainty tensors
            
        Returns:
            Aggregated uncertainty score [batch_size]
        """
        if self.method == 'weighted_mean':
            total_uncertainty = 0
            total_weight = 0
            
            for name, sigma in uncertainties.items():
                if name in self.weights:
                    # Average uncertainty across features for each embedder
                    embedder_uncertainty = sigma.mean(dim=-1)
                    total_uncertainty += self.weights[name] * embedder_uncertainty
                    total_weight += self.weights[name]
                    
            return total_uncertainty / total_weight
            
        elif self.method == 'max':
            # Conservative approach: use maximum uncertainty
            max_uncertainties = []
            for sigma in uncertainties.values():
                max_uncertainties.append(sigma.mean(dim=-1))
            return torch.stack(max_uncertainties).max(dim=0)[0]
            
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")
            
    def compute_dynamic_threshold(
        self, 
        base_threshold: float,
        aggregated_uncertainty: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute dynamic confidence threshold based on uncertainty.
        
        Higher uncertainty → Higher threshold (more conservative)
        
        Args:
            base_threshold: Base confidence threshold (e.g., 0.65)
            aggregated_uncertainty: Aggregated uncertainty scores
            
        Returns:
            Dynamic thresholds [batch_size]
        """
        # Sigmoid scaling to map uncertainty to threshold adjustment
        uncertainty_factor = torch.sigmoid(aggregated_uncertainty * 2 - 1)
        
        # Adjust threshold: higher uncertainty → higher threshold
        # Range: [base_threshold, base_threshold + 0.2]
        dynamic_threshold = base_threshold + 0.2 * uncertainty_factor
        
        return dynamic_threshold