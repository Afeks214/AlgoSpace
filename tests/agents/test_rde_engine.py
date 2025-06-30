"""
Unit tests for the Regime Detection Engine (RDE) component.

This test suite validates the functionality and robustness of the RDEComponent,
ensuring it correctly handles model loading, inference, and edge cases.
"""

import os
import tempfile
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest
import torch

from src.agents.rde.engine import RDEComponent
from src.agents.rde.model import RegimeDetectionEngine


class TestRDEComponent:
    """Test suite for the RDE Component."""
    
    @pytest.fixture
    def base_config(self):
        """Provide a base configuration for RDE tests."""
        return {
            'input_dim': 155,
            'd_model': 256,
            'latent_dim': 8,
            'n_heads': 8,
            'n_layers': 3,
            'dropout': 0.1,
            'device': 'cpu',
            'sequence_length': 24
        }
    
    @pytest.fixture
    def rde_component(self, base_config):
        """Create an RDE component instance for testing."""
        return RDEComponent(base_config)
    
    def test_rde_component_initialization(self, base_config):
        """Verify that the RDEComponent can be created successfully."""
        # Initialize component
        rde = RDEComponent(base_config)
        
        # Assert component is created
        assert rde is not None
        assert isinstance(rde, RDEComponent)
        
        # Assert internal model is correct type
        assert isinstance(rde.model, RegimeDetectionEngine)
        
        # Assert configuration is stored correctly
        assert rde.input_dim == 155
        assert rde.d_model == 256
        assert rde.latent_dim == 8
        assert rde.n_heads == 8
        assert rde.n_layers == 3
        assert rde.dropout == 0.1
        assert rde.sequence_length == 24
        
        # Assert model is in eval mode by default
        assert not rde.model.training
        assert not rde.model_loaded
    
    def test_load_model_successfully(self, rde_component):
        """Verify that the component can load a pre-trained model file correctly."""
        # Create a dummy model and save it
        dummy_model = RegimeDetectionEngine(
            input_dim=155,
            d_model=256,
            latent_dim=8,
            n_heads=8,
            n_layers=3,
            dropout=0.1
        )
        
        # Create temporary file for model
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            model_path = tmp_file.name
            
            # Save model state dict
            torch.save({
                'model_state_dict': dummy_model.state_dict(),
                'epoch': 50,
                'val_loss': 0.1234
            }, model_path)
        
        try:
            # Load model
            rde_component.load_model(model_path)
            
            # Assert model is loaded
            assert rde_component.model_loaded == True
            
            # Assert model is still in eval mode
            assert not rde_component.model.training
            
            # Verify state dict was loaded (weights should match)
            loaded_state = rde_component.model.state_dict()
            dummy_state = dummy_model.state_dict()
            
            for key in loaded_state:
                assert torch.allclose(loaded_state[key], dummy_state[key])
            
        finally:
            # Clean up temporary file
            os.unlink(model_path)
    
    def test_load_model_with_different_checkpoint_formats(self, rde_component):
        """Test loading models with different checkpoint formats."""
        dummy_model = RegimeDetectionEngine(
            input_dim=155,
            d_model=256,
            latent_dim=8
        )
        
        # Test format 1: Direct state dict
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            model_path = tmp_file.name
            torch.save(dummy_model.state_dict(), model_path)
        
        try:
            rde_component.load_model(model_path)
            assert rde_component.model_loaded == True
        finally:
            os.unlink(model_path)
        
        # Reset
        rde_component.model_loaded = False
        
        # Test format 2: Dict with 'state_dict' key
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            model_path = tmp_file.name
            torch.save({'state_dict': dummy_model.state_dict()}, model_path)
        
        try:
            rde_component.load_model(model_path)
            assert rde_component.model_loaded == True
        finally:
            os.unlink(model_path)
    
    def test_load_model_file_not_found(self, rde_component):
        """Test that loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            rde_component.load_model("/non/existent/path.pth")
    
    def test_get_regime_vector_interface(self, rde_component):
        """Verify that the main inference method works with correct input/output."""
        # First load a model (using dummy weights)
        dummy_model = RegimeDetectionEngine(
            input_dim=155,
            d_model=256,
            latent_dim=8
        )
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            model_path = tmp_file.name
            torch.save(dummy_model.state_dict(), model_path)
        
        try:
            rde_component.load_model(model_path)
            
            # Create valid input matrix
            dummy_matrix = np.random.randn(24, 155).astype(np.float32)
            
            # Mock the encode method to verify no_grad context
            original_encode = rde_component.model.encode
            encode_called_in_no_grad = False
            
            def mock_encode(x):
                nonlocal encode_called_in_no_grad
                encode_called_in_no_grad = not torch.is_grad_enabled()
                return original_encode(x)
            
            rde_component.model.encode = mock_encode
            
            # Call inference
            result = rde_component.get_regime_vector(dummy_matrix)
            
            # Assert output type and shape
            assert isinstance(result, np.ndarray)
            assert result.shape == (8,)
            assert result.dtype == np.float32
            
            # Assert model was called in no_grad context
            assert encode_called_in_no_grad
            
        finally:
            os.unlink(model_path)
    
    def test_get_regime_vector_different_sequence_lengths(self, rde_component):
        """Test that the component handles different sequence lengths with warning."""
        # Load model first
        dummy_model = RegimeDetectionEngine(input_dim=155, latent_dim=8)
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            model_path = tmp_file.name
            torch.save(dummy_model.state_dict(), model_path)
        
        try:
            rde_component.load_model(model_path)
            
            # Test with different sequence length (should log warning but work)
            with patch('src.agents.rde.engine.logger') as mock_logger:
                dummy_matrix = np.random.randn(48, 155)  # Different from expected 24
                result = rde_component.get_regime_vector(dummy_matrix)
                
                # Should still work
                assert result.shape == (8,)
                
                # Should log warning
                mock_logger.warning.assert_called_once()
                warning_msg = mock_logger.warning.call_args[0][0]
                assert "Sequence length 48 differs from expected 24" in warning_msg
                
        finally:
            os.unlink(model_path)
    
    def test_rde_handles_incorrect_input_shape(self, rde_component):
        """Verify that the component fails gracefully with bad input."""
        # Load model first
        dummy_model = RegimeDetectionEngine(input_dim=155, latent_dim=8)
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            model_path = tmp_file.name
            torch.save(dummy_model.state_dict(), model_path)
        
        try:
            rde_component.load_model(model_path)
            
            # Test 1: Wrong number of dimensions
            with pytest.raises(ValueError, match="Expected 2D array"):
                bad_matrix = np.random.randn(24)  # 1D instead of 2D
                rde_component.get_regime_vector(bad_matrix)
            
            # Test 2: Wrong number of features
            with pytest.raises(ValueError, match="Expected 155 features, got 100"):
                bad_matrix = np.random.randn(24, 100)  # Wrong feature dimension
                rde_component.get_regime_vector(bad_matrix)
            
            # Test 3: Not a numpy array
            with pytest.raises(ValueError, match="Input must be a NumPy array"):
                bad_input = [[1, 2, 3]] * 24  # List instead of numpy array
                rde_component.get_regime_vector(bad_input)
                
        finally:
            os.unlink(model_path)
    
    def test_get_regime_vector_without_loaded_model(self, rde_component):
        """Test that inference fails if model is not loaded."""
        dummy_matrix = np.random.randn(24, 155)
        
        with pytest.raises(RuntimeError, match="Model weights not loaded"):
            rde_component.get_regime_vector(dummy_matrix)
    
    def test_get_model_info(self, rde_component):
        """Test the model info method returns correct information."""
        info = rde_component.get_model_info()
        
        assert info['architecture'] == 'Transformer + VAE'
        assert info['input_dim'] == 155
        assert info['d_model'] == 256
        assert info['latent_dim'] == 8
        assert info['n_heads'] == 8
        assert info['n_layers'] == 3
        assert info['model_loaded'] == False
        assert info['device'] == 'cpu'
        assert info['expected_sequence_length'] == 24
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
    
    def test_validate_config(self, rde_component):
        """Test configuration validation against saved config."""
        # Create a matching config file
        config_data = {
            'input_dim': 155,
            'latent_dim': 8,
            'd_model': 256
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            import json
            json.dump(config_data, tmp_file)
            config_path = tmp_file.name
        
        try:
            # Should pass validation
            assert rde_component.validate_config(config_path) == True
            
            # Now test with mismatched config
            rde_component.input_dim = 100  # Change to create mismatch
            assert rde_component.validate_config(config_path) == False
            
        finally:
            os.unlink(config_path)
    
    def test_repr(self, rde_component):
        """Test string representation of RDE component."""
        repr_str = repr(rde_component)
        assert "RDEComponent" in repr_str
        assert "input_dim=155" in repr_str
        assert "latent_dim=8" in repr_str
        assert "model_loaded=False" in repr_str
        assert "device=cpu" in repr_str


class TestRDEIntegration:
    """Integration tests for RDE with the rest of the system."""
    
    def test_full_inference_pipeline(self):
        """Test the complete inference pipeline from raw data to regime vector."""
        # Create component with minimal config
        config = {
            'input_dim': 10,  # Smaller for testing
            'latent_dim': 4,
            'd_model': 32,
            'n_heads': 4,
            'n_layers': 2
        }
        
        rde = RDEComponent(config)
        
        # Create and load a simple model
        model = RegimeDetectionEngine(
            input_dim=10,
            d_model=32,
            latent_dim=4,
            n_heads=4,
            n_layers=2
        )
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            model_path = tmp_file.name
            torch.save(model.state_dict(), model_path)
        
        try:
            # Load model
            rde.load_model(model_path)
            
            # Create sequence of data
            sequence = np.random.randn(24, 10).astype(np.float32)
            
            # Run inference
            regime_vector = rde.get_regime_vector(sequence)
            
            # Verify output
            assert regime_vector.shape == (4,)
            assert not np.any(np.isnan(regime_vector))
            assert not np.any(np.isinf(regime_vector))
            
        finally:
            os.unlink(model_path)
    
    def test_device_handling(self):
        """Test that the component correctly handles device configuration."""
        # Test CPU device (default)
        config = {'device': 'cpu'}
        rde = RDEComponent(config)
        assert rde.device.type == 'cpu'
        
        # Test CUDA device (if available)
        if torch.cuda.is_available():
            config = {'device': 'cuda'}
            rde = RDEComponent(config)
            assert rde.device.type == 'cuda'
            
            # Ensure model is on correct device
            assert next(rde.model.parameters()).device.type == 'cuda'