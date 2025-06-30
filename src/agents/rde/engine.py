"""
Production-ready Regime Detection Engine component.

This module provides the high-level interface for the RDE, handling model
loading, inference, and integration with the AlgoSpace system.
"""

from typing import Dict, Any, Optional
import os
import json
import logging

import numpy as np
import torch

from .model import RegimeDetectionEngine


logger = logging.getLogger(__name__)


class RDEComponent:
    """
    High-level wrapper for the Regime Detection Engine.
    
    This class provides the main interface for the rest of the AlgoSpace system
    to interact with the RDE. It handles model initialization, loading pre-trained
    weights, and performing inference on MMD feature sequences.
    
    Attributes:
        config: RDE-specific configuration dictionary
        model: The underlying RegimeDetectionEngine neural network
        device: PyTorch device for computation (CPU by default)
        model_loaded: Flag indicating if model weights are loaded
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RDE component.
        
        Args:
            config: RDE-specific configuration dictionary containing:
                - input_dim: Number of MMD features (required)
                - d_model: Transformer dimension (default: 256)
                - latent_dim: Regime vector dimension (default: 8)
                - n_heads: Number of attention heads (default: 8)
                - n_layers: Number of transformer layers (default: 3)
                - dropout: Dropout probability (default: 0.1)
                - device: Computation device (default: 'cpu')
                - sequence_length: Expected sequence length (default: 24)
        """
        self.config = config
        self.model_loaded = False
        
        # Extract model configuration
        self.input_dim = config.get('input_dim', 155)  # Default from training
        self.d_model = config.get('d_model', 256)
        self.latent_dim = config.get('latent_dim', 8)
        self.n_heads = config.get('n_heads', 8)
        self.n_layers = config.get('n_layers', 3)
        self.dropout = config.get('dropout', 0.1)
        self.sequence_length = config.get('sequence_length', 24)
        
        # Set device (CPU for production stability)
        self.device = torch.device(config.get('device', 'cpu'))
        
        # Initialize model
        self.model = RegimeDetectionEngine(
            input_dim=self.input_dim,
            d_model=self.d_model,
            latent_dim=self.latent_dim,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Set to evaluation mode by default
        self.model.eval()
        
        logger.info(
            f"RDE initialized with architecture: "
            f"input_dim={self.input_dim}, d_model={self.d_model}, "
            f"latent_dim={self.latent_dim}, device={self.device}"
        )
    
    def load_model(self, model_path: str) -> None:
        """
        Load pre-trained weights from file.
        
        Args:
            model_path: Path to the .pth file containing model weights
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If weights fail to load
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract state dict (handle different checkpoint formats)
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    # Assume the checkpoint is the state dict itself
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Load weights
            self.model.load_state_dict(state_dict)
            
            # Ensure model is in eval mode
            self.model.eval()
            
            self.model_loaded = True
            
            # Log success
            logger.info(f"Successfully loaded RDE model from: {model_path}")
            
            # If checkpoint contains metadata, log it
            if isinstance(checkpoint, dict):
                if 'epoch' in checkpoint:
                    logger.info(f"Model trained for {checkpoint['epoch']} epochs")
                if 'val_loss' in checkpoint:
                    logger.info(f"Model validation loss: {checkpoint['val_loss']:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def get_regime_vector(self, mmd_matrix: np.ndarray) -> np.ndarray:
        """
        Perform inference to get regime vector from MMD features.
        
        This is the primary method for getting regime predictions. It handles
        all necessary data conversions and returns a clean NumPy array.
        
        Args:
            mmd_matrix: NumPy array of shape (N, F) where:
                - N is the sequence length (typically 24 for 12 hours)
                - F is the number of MMD features
                
        Returns:
            Regime vector as NumPy array of shape (8,)
            
        Raises:
            ValueError: If input shape is invalid
            RuntimeError: If model weights not loaded
        """
        if not self.model_loaded:
            raise RuntimeError("Model weights not loaded. Call load_model() first.")
        
        # Validate input shape
        if not isinstance(mmd_matrix, np.ndarray):
            raise ValueError("Input must be a NumPy array")
        
        if mmd_matrix.ndim != 2:
            raise ValueError(
                f"Expected 2D array (sequence_length, features), "
                f"got shape {mmd_matrix.shape}"
            )
        
        seq_len, n_features = mmd_matrix.shape
        
        if n_features != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features, got {n_features}"
            )
        
        # Log warning if sequence length doesn't match expected
        if seq_len != self.sequence_length:
            logger.warning(
                f"Sequence length {seq_len} differs from expected "
                f"{self.sequence_length}. This may affect performance."
            )
        
        # Convert to PyTorch tensor
        mmd_tensor = torch.FloatTensor(mmd_matrix)
        
        # Add batch dimension
        mmd_tensor = mmd_tensor.unsqueeze(0)  # Shape: (1, seq_len, features)
        
        # Move to device
        mmd_tensor = mmd_tensor.to(self.device)
        
        # Perform inference
        with torch.no_grad():
            regime_vector = self.model.encode(mmd_tensor)
            
            # Convert back to NumPy
            regime_vector = regime_vector.cpu().numpy()
            
            # Remove batch dimension
            regime_vector = regime_vector.squeeze(0)  # Shape: (8,)
        
        return regime_vector
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model architecture and status.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        
        return {
            'architecture': 'Transformer + VAE',
            'input_dim': self.input_dim,
            'd_model': self.d_model,
            'latent_dim': self.latent_dim,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_loaded': self.model_loaded,
            'device': str(self.device),
            'expected_sequence_length': self.sequence_length
        }
    
    def validate_config(self, config_path: Optional[str] = None) -> bool:
        """
        Validate configuration against saved model config.
        
        Args:
            config_path: Path to saved model configuration JSON
            
        Returns:
            True if configurations match, False otherwise
        """
        if config_path is None:
            logger.warning("No config path provided for validation")
            return True
        
        try:
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
            
            # Check critical parameters
            mismatches = []
            
            if saved_config.get('input_dim') != self.input_dim:
                mismatches.append(
                    f"input_dim: saved={saved_config.get('input_dim')}, "
                    f"current={self.input_dim}"
                )
            
            if saved_config.get('latent_dim') != self.latent_dim:
                mismatches.append(
                    f"latent_dim: saved={saved_config.get('latent_dim')}, "
                    f"current={self.latent_dim}"
                )
            
            if mismatches:
                logger.warning(
                    f"Configuration mismatches detected: {'; '.join(mismatches)}"
                )
                return False
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate config: {str(e)}")
            return False
    
    def __repr__(self) -> str:
        """String representation of the RDE component."""
        return (
            f"RDEComponent(input_dim={self.input_dim}, "
            f"latent_dim={self.latent_dim}, "
            f"model_loaded={self.model_loaded}, "
            f"device={self.device})"
        )