"""
Neural network architectures for the Regime Detection Engine (RDE).

This module contains the Transformer + VAE architecture components used
for market regime detection from MMD (Matrix Market Distance) features.
"""

from typing import Dict, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer architecture.
    
    Adds positional information to the input embeddings to help the model
    understand the temporal order of the sequence.
    
    Args:
        d_model: The dimension of the model embeddings
        max_len: Maximum sequence length to support
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape (seq_len, batch, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for processing MMD feature sequences.
    
    Uses self-attention to capture temporal dependencies in the market
    microstructure data and outputs a context vector summarizing the sequence.
    
    Args:
        input_dim: Dimension of input features (MMD features)
        d_model: Internal dimension of the transformer
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        dropout: Dropout probability for regularization
    """
    
    def __init__(
        self, 
        input_dim: int, 
        d_model: int = 256, 
        n_heads: int = 8, 
        n_layers: int = 3, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process sequence through transformer encoder.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Context vector of shape (batch, d_model)
        """
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass through transformer
        x = self.transformer(x)
        
        # Return mean pooling as context vector
        context = x.mean(dim=1)  # (batch, d_model)
        
        return context


class VAEHead(nn.Module):
    """
    Variational Autoencoder head that maps context vector to latent space.
    
    Learns a probabilistic latent representation of market regimes by
    encoding the context vector into mean and variance parameters.
    
    Args:
        context_dim: Dimension of the input context vector
        latent_dim: Dimension of the latent space (regime vector)
    """
    
    def __init__(self, context_dim: int, latent_dim: int = 8):
        super().__init__()
        
        # Encoder networks for mean and log variance
        self.fc_mu = nn.Sequential(
            nn.Linear(context_dim, context_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(context_dim // 2, latent_dim)
        )
        
        self.fc_log_var = nn.Sequential(
            nn.Linear(context_dim, context_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(context_dim // 2, latent_dim)
        )
        
        self.latent_dim = latent_dim
        
    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode context vector to latent distribution parameters.
        
        Args:
            context: Context vector from transformer
            
        Returns:
            Tuple of (mean, log_variance) tensors
        """
        mu = self.fc_mu(context)
        log_var = self.fc_log_var(context)
        
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE training.
        
        Samples from the latent distribution while maintaining differentiability.
        
        Args:
            mu: Mean of the latent distribution
            log_var: Log variance of the latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


class Decoder(nn.Module):
    """
    Decoder that reconstructs context from latent space.
    
    Used during training to ensure the latent space captures meaningful
    information about the market regime.
    
    Args:
        latent_dim: Dimension of the latent space
        context_dim: Dimension of the output context vector
    """
    
    def __init__(self, latent_dim: int, context_dim: int):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, context_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(context_dim // 2, context_dim)
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct context vector from latent representation.
        
        Args:
            z: Latent vector
            
        Returns:
            Reconstructed context vector
        """
        return self.decoder(z)


class RegimeDetectionEngine(nn.Module):
    """
    Complete Regime Detection Engine with Transformer + VAE architecture.
    
    This model processes sequences of MMD features to identify market regimes,
    outputting an 8-dimensional regime vector that captures the essential
    characteristics of the current market state.
    
    Architecture:
        1. TransformerEncoder: Processes MMD sequences with self-attention
        2. VAEHead: Maps transformer output to probabilistic latent space
        3. Decoder: Reconstructs context (used only during training)
    
    Args:
        input_dim: Number of MMD features
        d_model: Internal transformer dimension
        latent_dim: Dimension of regime vector (output)
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        dropout: Dropout probability
    """
    
    def __init__(
        self, 
        input_dim: int, 
        d_model: int = 256, 
        latent_dim: int = 8, 
        n_heads: int = 8, 
        n_layers: int = 3, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Sub-modules
        self.transformer_encoder = TransformerEncoder(
            input_dim, d_model, n_heads, n_layers, dropout
        )
        self.vae_head = VAEHead(d_model, latent_dim)
        self.decoder = Decoder(latent_dim, d_model)
        
        # Store dimensions
        self.input_dim = input_dim
        self.d_model = d_model
        self.latent_dim = latent_dim
        
    def forward(
        self, 
        x: torch.Tensor, 
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input sequences of shape (batch, seq_len, input_dim)
            training: Whether in training mode (affects output)
            
        Returns:
            Dictionary containing:
                - mu: Mean of latent distribution
                - log_var: Log variance of latent distribution
                - z: Sampled latent vector (if training)
                - reconstructed: Reconstructed context (if training)
                - context: Transformer context vector
        """
        # Get context vector from transformer
        context = self.transformer_encoder(x)
        
        # Get latent distribution
        mu, log_var = self.vae_head(context)
        
        if training:
            # Sample from latent space
            z = self.vae_head.reparameterize(mu, log_var)
            
            # Reconstruct context
            reconstructed = self.decoder(z)
            
            return {
                'mu': mu,
                'log_var': log_var,
                'z': z,
                'reconstructed': reconstructed,
                'context': context
            }
        else:
            # For inference, just return the latent representation
            return {
                'mu': mu,
                'log_var': log_var,
                'context': context
            }
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode sequences to latent space (for inference).
        
        This is the main method used in production to get regime vectors.
        
        Args:
            x: Input sequences of shape (batch, seq_len, input_dim)
            
        Returns:
            Regime vectors of shape (batch, latent_dim)
        """
        with torch.no_grad():
            context = self.transformer_encoder(x)
            mu, log_var = self.vae_head(context)
            return mu  # Return mean as the regime vector