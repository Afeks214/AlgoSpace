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
                - enable_caching: Enable inference caching (default: True)
                - cache_size: Maximum cache entries (default: 1000)
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
        
        # Performance optimization settings
        self.enable_caching = config.get('enable_caching', True)
        self.cache_size = config.get('cache_size', 1000)
        
        # Set device (CPU for production stability)
        self.device = torch.device(config.get('device', 'cpu'))
        
        # Initialize inference cache
        self._inference_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
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
        
        # Pre-allocated tensors for performance
        self._tensor_cache = {
            'input_tensor': None,
            'last_shape': None
        }
        
        logger.info(
            f"RDE initialized with architecture: "
            f"input_dim={self.input_dim}, d_model={self.d_model}, "
            f"latent_dim={self.latent_dim}, device={self.device}, "
            f"caching={self.enable_caching}"
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
            # Load checkpoint with weights_only=False for compatibility
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
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
            
            # Load weights (allow missing keys for development)
            self.model.load_state_dict(state_dict, strict=False)
            
            # Ensure model is in eval mode
            self.model.eval()
            
            # Apply performance optimizations after loading
            self._apply_optimizations()
            
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
    
    def _apply_optimizations(self) -> None:
        """Apply post-loading performance optimizations."""
        try:
            # Try to compile model with torch.jit for performance
            if hasattr(torch, 'jit') and torch.jit.is_available():
                logger.info("Applying torch.jit optimization...")
                # Create sample input for tracing
                sample_input = torch.randn(1, self.sequence_length, self.input_dim, device=self.device)
                self.model = torch.jit.trace(self.model, sample_input)
                logger.info("✅ Model optimized with torch.jit")
            
            # Set model to optimized inference mode
            if hasattr(self.model, 'set_swish_optimization'):
                self.model.set_swish_optimization(True)
                
        except Exception as e:
            logger.warning(f"Some optimizations failed: {e}")
            # Continue anyway - optimizations are not critical
    
    def _compute_cache_key(self, mmd_matrix: np.ndarray) -> str:
        """Compute a hash key for caching inference results."""
        # Use a fast hash of the matrix content
        return str(hash(mmd_matrix.tobytes()))
    
    def clear_cache(self) -> None:
        """Clear the inference cache."""
        self._inference_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Inference cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._inference_cache),
            'max_cache_size': self.cache_size
        }
    
    def get_regime_vector(self, mmd_matrix: np.ndarray) -> np.ndarray:
        """
        Perform inference to get regime vector from MMD features.
        
        OPTIMIZED VERSION: Uses caching, tensor reuse, and minimal validation
        for maximum performance in production.
        
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
        
        # Fast path: check cache first
        if self.enable_caching:
            cache_key = self._compute_cache_key(mmd_matrix)
            if cache_key in self._inference_cache:
                self._cache_hits += 1
                return self._inference_cache[cache_key].copy()
            self._cache_misses += 1
        
        # Minimal validation for performance
        seq_len, n_features = mmd_matrix.shape
        if n_features != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {n_features}")
        
        # Reuse tensor allocation if shape matches
        current_shape = (1, seq_len, n_features)
        if (self._tensor_cache['input_tensor'] is None or 
            self._tensor_cache['last_shape'] != current_shape):
            self._tensor_cache['input_tensor'] = torch.empty(
                current_shape, dtype=torch.float32, device=self.device
            )
            self._tensor_cache['last_shape'] = current_shape
        
        # Fast tensor conversion - copy directly into pre-allocated tensor
        mmd_tensor = self._tensor_cache['input_tensor']
        mmd_tensor[0] = torch.from_numpy(mmd_matrix).to(device=self.device, dtype=torch.float32)
        
        # Optimized inference
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
            regime_vector = self.model.encode(mmd_tensor)
            
            # Fast conversion back to NumPy
            regime_vector_np = regime_vector[0].cpu().numpy()
        
        # Cache the result
        if self.enable_caching and len(self._inference_cache) < self.cache_size:
            self._inference_cache[cache_key] = regime_vector_np.copy()
        
        return regime_vector_np
    
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
        
        info = {
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
            'expected_sequence_length': self.sequence_length,
            'optimizations': {
                'caching_enabled': self.enable_caching,
                'cache_size_limit': self.cache_size,
                'jit_compiled': 'torch.jit' in str(type(self.model))
            }
        }
        
        # Add cache statistics if available
        if self.enable_caching:
            info['cache_stats'] = self.get_cache_stats()
            
        return info
    
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


# Alias for backward compatibility
RDEEngine = RDEComponent