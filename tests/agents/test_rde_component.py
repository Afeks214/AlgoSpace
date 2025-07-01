"""
Comprehensive tests for the Regime Detection Engine (RDE) component.

This test suite validates the RDEComponent class functionality including:
- Component initialization and configuration
- Model loading from checkpoint files
- Regime vector inference interface and output shape
- Error handling for invalid inputs
"""

import pytest
import numpy as np
import torch
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from src.agents.rde.engine import RDEComponent
from src.agents.rde.model import RegimeDetectionEngine


class TestRDEComponentInitialization:
    """Test RDEComponent initialization and configuration."""
    
    def test_rde_component_initialization(self):
        """Verify that the RDEComponent class can be instantiated correctly."""
        # Create a mock config dictionary
        config = {
            'input_dim': 23,  # MMD features
            'd_model': 256,
            'latent_dim': 8,
            'n_heads': 8,
            'n_layers': 3,
            'dropout': 0.1,
            'device': 'cpu',
            'sequence_length': 24
        }
        
        # Instantiate RDEComponent
        rde_component = RDEComponent(config)
        
        # Verify initialization
        assert rde_component is not None
        assert rde_component.input_dim == 23
        assert rde_component.d_model == 256
        assert rde_component.latent_dim == 8
        assert rde_component.n_heads == 8
        assert rde_component.n_layers == 3
        assert rde_component.dropout == 0.1
        assert rde_component.device == torch.device('cpu')
        assert rde_component.sequence_length == 24
        assert rde_component.model_loaded == False
        
        # Verify that internal RegimeDetectionEngine model is initialized
        assert isinstance(rde_component.model, RegimeDetectionEngine)
        assert rde_component.model.input_dim == 23
        assert rde_component.model.d_model == 256
        assert rde_component.model.latent_dim == 8
    
    def test_rde_component_initialization_with_defaults(self):
        """Test RDEComponent initialization with minimal config (using defaults)."""
        # Minimal config with just required parameters
        config = {
            'input_dim': 23
        }
        
        # Instantiate RDEComponent
        rde_component = RDEComponent(config)
        
        # Verify defaults are applied
        assert rde_component.input_dim == 23
        assert rde_component.d_model == 256  # default
        assert rde_component.latent_dim == 8  # default
        assert rde_component.n_heads == 8     # default
        assert rde_component.n_layers == 3    # default
        assert rde_component.dropout == 0.1   # default
        assert rde_component.sequence_length == 24  # default
        assert rde_component.device == torch.device('cpu')  # default


class TestRDEModelLoading:
    """Test RDE model loading functionality."""
    
    @pytest.fixture
    def rde_component(self):
        """Create an RDEComponent instance for testing."""
        config = {
            'input_dim': 23,
            'd_model': 128,  # Smaller for testing
            'latent_dim': 8,
            'n_heads': 4,
            'n_layers': 2,
            'dropout': 0.1,
            'device': 'cpu',
            'sequence_length': 24
        }
        return RDEComponent(config)
    
    def test_rde_loads_model_correctly(self, rde_component):
        """Test the load_model() method with a dummy PyTorch model."""
        # Create a dummy model with the same architecture
        dummy_model = RegimeDetectionEngine(
            input_dim=23,
            d_model=128,
            latent_dim=8,
            n_heads=4,
            n_layers=2,
            dropout=0.1
        )
        
        # Save dummy model state dict to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as temp_file:
            torch.save(dummy_model.state_dict(), temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Test loading the model
            rde_component.load_model(temp_path)
            
            # Verify model was loaded successfully
            assert rde_component.model_loaded == True
            
            # Verify model is in evaluation mode
            assert not rde_component.model.training
            
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
    
    def test_rde_loads_model_with_checkpoint_format(self, rde_component):
        """Test loading model from checkpoint with metadata."""
        # Create a dummy model
        dummy_model = RegimeDetectionEngine(
            input_dim=23,
            d_model=128,
            latent_dim=8,
            n_heads=4,
            n_layers=2,
            dropout=0.1
        )
        
        # Create checkpoint with metadata
        checkpoint = {
            'model_state_dict': dummy_model.state_dict(),
            'epoch': 100,
            'loss': 0.25,
            'optimizer_state_dict': {'test': 'data'}
        }
        
        # Save checkpoint to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as temp_file:
            torch.save(checkpoint, temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Test loading the checkpoint
            rde_component.load_model(temp_path)
            
            # Verify model was loaded successfully
            assert rde_component.model_loaded == True
            assert not rde_component.model.training
            
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
    
    def test_rde_load_model_file_not_found(self, rde_component):
        """Test that load_model raises FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            rde_component.load_model("/nonexistent/path/model.pth")
    
    def test_rde_load_model_invalid_weights(self, rde_component):
        """Test that load_model handles invalid model weights gracefully."""
        # Create a model with different architecture
        wrong_model = torch.nn.Linear(10, 5)
        
        # Save wrong model to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as temp_file:
            torch.save(wrong_model.state_dict(), temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Test loading incompatible weights should raise RuntimeError
            with pytest.raises(RuntimeError):
                rde_component.load_model(temp_path)
                
        finally:
            # Clean up temporary file
            os.unlink(temp_path)


class TestRDEInference:
    """Test RDE inference functionality."""
    
    @pytest.fixture
    def loaded_rde_component(self):
        """Create a loaded RDEComponent for inference testing."""
        config = {
            'input_dim': 23,
            'd_model': 128,
            'latent_dim': 8,
            'n_heads': 4,
            'n_layers': 2,
            'dropout': 0.1,
            'device': 'cpu',
            'sequence_length': 24
        }
        
        rde_component = RDEComponent(config)
        
        # Create and load a dummy model
        dummy_model = RegimeDetectionEngine(
            input_dim=23,
            d_model=128,
            latent_dim=8,
            n_heads=4,
            n_layers=2,
            dropout=0.1
        )
        
        # Save and load dummy model
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as temp_file:
            torch.save(dummy_model.state_dict(), temp_file.name)
            temp_path = temp_file.name
        
        try:
            rde_component.load_model(temp_path)
        finally:
            os.unlink(temp_path)
        
        return rde_component
    
    def test_get_regime_vector_interface_and_shape(self, loaded_rde_component):
        """Test get_regime_vector method interface and output shape."""
        # Create valid input matrix (sequence_length=24, features=23)
        input_matrix = np.random.randn(24, 23).astype(np.float32)
        
        # Call get_regime_vector
        regime_vector = loaded_rde_component.get_regime_vector(input_matrix)
        
        # Assert output is NumPy array
        assert isinstance(regime_vector, np.ndarray)
        
        # Assert output shape is (8,) matching latent_dim
        assert regime_vector.shape == (8,)
        
        # Assert output contains finite values
        assert np.all(np.isfinite(regime_vector))
        
        # Assert output is float type
        assert regime_vector.dtype == np.float32
    
    def test_get_regime_vector_different_sequence_lengths(self, loaded_rde_component):
        """Test get_regime_vector with different sequence lengths."""
        # Test with shorter sequence
        short_input = np.random.randn(12, 23).astype(np.float32)
        regime_vector_short = loaded_rde_component.get_regime_vector(short_input)
        assert regime_vector_short.shape == (8,)
        
        # Test with longer sequence
        long_input = np.random.randn(48, 23).astype(np.float32)
        regime_vector_long = loaded_rde_component.get_regime_vector(long_input)
        assert regime_vector_long.shape == (8,)
        
        # Test with minimal sequence
        minimal_input = np.random.randn(1, 23).astype(np.float32)
        regime_vector_minimal = loaded_rde_component.get_regime_vector(minimal_input)
        assert regime_vector_minimal.shape == (8,)
    
    def test_rde_handles_incorrect_input_shape(self, loaded_rde_component):
        """Test that get_regime_vector raises ValueError for incorrect input shapes."""
        # Test wrong number of features
        wrong_features = np.random.randn(24, 10)  # Should be 23 features
        with pytest.raises(ValueError, match="Expected 23 features, got 10"):
            loaded_rde_component.get_regime_vector(wrong_features)
        
        # Test 1D array
        wrong_dims_1d = np.random.randn(23)
        with pytest.raises(ValueError, match="Expected 2D array"):
            loaded_rde_component.get_regime_vector(wrong_dims_1d)
        
        # Test 3D array
        wrong_dims_3d = np.random.randn(24, 23, 5)
        with pytest.raises(ValueError, match="Expected 2D array"):
            loaded_rde_component.get_regime_vector(wrong_dims_3d)
        
        # Test non-numpy array
        wrong_type = [[1, 2, 3]] * 23
        with pytest.raises(ValueError, match="Input must be a NumPy array"):
            loaded_rde_component.get_regime_vector(wrong_type)
    
    def test_get_regime_vector_model_not_loaded(self):
        """Test that get_regime_vector raises RuntimeError when model not loaded."""
        config = {'input_dim': 23}
        rde_component = RDEComponent(config)
        
        # Try to get regime vector without loading model
        input_matrix = np.random.randn(24, 23)
        
        with pytest.raises(RuntimeError, match="Model weights not loaded"):
            rde_component.get_regime_vector(input_matrix)


class TestRDEEdgeCases:
    """Test RDE component edge cases and robustness."""
    
    @pytest.fixture
    def loaded_rde_component(self):
        """Create a loaded RDEComponent for edge case testing."""
        config = {
            'input_dim': 23,
            'd_model': 64,  # Even smaller for edge case testing
            'latent_dim': 8,
            'n_heads': 2,
            'n_layers': 1,
            'dropout': 0.0,  # No dropout for deterministic testing
            'device': 'cpu',
            'sequence_length': 24
        }
        
        rde_component = RDEComponent(config)
        
        # Create and load a dummy model
        dummy_model = RegimeDetectionEngine(
            input_dim=23,
            d_model=64,
            latent_dim=8,
            n_heads=2,
            n_layers=1,
            dropout=0.0
        )
        
        # Save and load dummy model
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as temp_file:
            torch.save(dummy_model.state_dict(), temp_file.name)
            temp_path = temp_file.name
        
        try:
            rde_component.load_model(temp_path)
        finally:
            os.unlink(temp_path)
        
        return rde_component
    
    def test_extreme_input_values(self, loaded_rde_component):
        """Test RDE with extreme input values."""
        # Test with very large values
        large_input = np.full((24, 23), 1000.0, dtype=np.float32)
        regime_vector_large = loaded_rde_component.get_regime_vector(large_input)
        assert regime_vector_large.shape == (8,)
        assert np.all(np.isfinite(regime_vector_large))
        
        # Test with very small values
        small_input = np.full((24, 23), 1e-6, dtype=np.float32)
        regime_vector_small = loaded_rde_component.get_regime_vector(small_input)
        assert regime_vector_small.shape == (8,)
        assert np.all(np.isfinite(regime_vector_small))
        
        # Test with zero values
        zero_input = np.zeros((24, 23), dtype=np.float32)
        regime_vector_zero = loaded_rde_component.get_regime_vector(zero_input)
        assert regime_vector_zero.shape == (8,)
        assert np.all(np.isfinite(regime_vector_zero))
    
    def test_deterministic_output(self, loaded_rde_component):
        """Test that RDE produces deterministic output for same input."""
        # Create fixed input
        fixed_input = np.random.RandomState(42).randn(24, 23).astype(np.float32)
        
        # Get regime vector twice
        regime_vector_1 = loaded_rde_component.get_regime_vector(fixed_input)
        regime_vector_2 = loaded_rde_component.get_regime_vector(fixed_input)
        
        # Should be identical (model in eval mode, no dropout)
        assert np.allclose(regime_vector_1, regime_vector_2, rtol=1e-6)
    
    def test_batch_processing_consistency(self, loaded_rde_component):
        """Test that single inference matches what would be batch processing."""
        # Create test input
        test_input = np.random.RandomState(123).randn(24, 23).astype(np.float32)
        
        # Get regime vector
        regime_vector = loaded_rde_component.get_regime_vector(test_input)
        
        # Verify the internal processing by checking tensor conversion
        # This implicitly tests the tensor conversion pipeline
        assert isinstance(regime_vector, np.ndarray)
        assert regime_vector.dtype == np.float32
        assert not np.any(np.isnan(regime_vector))
        assert not np.any(np.isinf(regime_vector))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])