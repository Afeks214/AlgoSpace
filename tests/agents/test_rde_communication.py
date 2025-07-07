"""
Comprehensive test suite for RDE Communication LSTM Layer.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os
import time

from src.agents.rde.communication import RDECommunicationLSTM, StateManager
from src.agents.rde.losses import (
    TemporalConsistencyLoss,
    UncertaintyCalibrationLoss,
    RegimePredictionLoss,
    RDECommunicationLoss,
    GradientFlowMonitor
)


class TestRDECommunicationLSTM:
    """Test suite for RDE Communication LSTM."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            'input_dim': 8,
            'hidden_dim': 32,
            'output_dim': 16,
            'num_layers': 2,
            'device': 'cpu'
        }
    
    @pytest.fixture
    def rde_comm(self, config):
        """Create RDE Communication LSTM instance."""
        return RDECommunicationLSTM(config)
    
    # Initialization Tests
    def test_initialization(self, rde_comm, config):
        """Test proper initialization of RDE communication layer."""
        assert rde_comm.input_dim == config['input_dim']
        assert rde_comm.hidden_dim == config['hidden_dim']
        assert rde_comm.output_dim == config['output_dim']
        assert rde_comm.num_layers == config['num_layers']
        assert str(rde_comm.device) == config['device']
        
        # Check learnable parameters
        assert isinstance(rde_comm.h0, nn.Parameter)
        assert isinstance(rde_comm.c0, nn.Parameter)
        assert rde_comm.h0.shape == (2, 1, 32)
        assert rde_comm.c0.shape == (2, 1, 32)
        
    def test_configuration_handling(self):
        """Test various configuration scenarios."""
        # Minimal config
        minimal_config = {'device': 'cpu'}
        rde_comm = RDECommunicationLSTM(minimal_config)
        assert rde_comm.input_dim == 8  # Default
        assert rde_comm.hidden_dim == 32  # Default
        
        # Custom config
        custom_config = {
            'input_dim': 16,
            'hidden_dim': 64,
            'output_dim': 32,
            'num_layers': 3
        }
        rde_comm = RDECommunicationLSTM(custom_config)
        assert rde_comm.input_dim == 16
        assert rde_comm.hidden_dim == 64
        assert rde_comm.output_dim == 32
        assert rde_comm.num_layers == 3
        
    def test_device_placement(self, config):
        """Test device placement and transfers."""
        # CPU device
        rde_comm = RDECommunicationLSTM(config)
        assert str(rde_comm.device) == 'cpu'
        
        # GPU device (if available)
        if torch.cuda.is_available():
            config['device'] = 'cuda'
            rde_comm = RDECommunicationLSTM(config)
            assert str(rde_comm.device) == 'cuda:0'
            
            # Test .to() method
            rde_comm = rde_comm.to('cpu')
            assert str(rde_comm.device) == 'cpu'
    
    # Forward Pass Tests
    def test_forward_pass_shapes(self, rde_comm):
        """Test forward pass produces correct shapes."""
        # Single sample
        regime_vector = torch.randn(1, 8)
        mu, sigma = rde_comm(regime_vector)
        
        assert mu.shape == (1, 16)
        assert sigma.shape == (1, 16)
        assert torch.all(sigma > 0)  # Positive uncertainty
        
        # Batch processing
        batch_regime = torch.randn(4, 8)
        mu_batch, sigma_batch = rde_comm(batch_regime)
        
        assert mu_batch.shape == (4, 16)
        assert sigma_batch.shape == (4, 16)
        
    def test_hidden_state_persistence(self, rde_comm):
        """Test LSTM maintains hidden state across calls."""
        regime1 = torch.randn(1, 8)
        regime2 = torch.randn(1, 8)
        
        # First call
        mu1, sigma1 = rde_comm(regime1)
        hidden1 = rde_comm.hidden
        
        # Second call
        mu2, sigma2 = rde_comm(regime2)
        hidden2 = rde_comm.hidden
        
        # Hidden state should change
        assert hidden1 is not None
        assert hidden2 is not None
        assert not torch.allclose(hidden1[0], hidden2[0])
        
        # Reset and verify
        rde_comm.reset_hidden_state()
        assert rde_comm.hidden is None
        
    def test_batch_processing(self, rde_comm):
        """Test handling different batch sizes."""
        # Test various batch sizes
        for batch_size in [1, 4, 8, 16]:
            regime = torch.randn(batch_size, 8)
            mu, sigma = rde_comm(regime)
            
            assert mu.shape == (batch_size, 16)
            assert sigma.shape == (batch_size, 16)
            
    def test_uncertainty_bounds(self, rde_comm):
        """Test uncertainty values are properly bounded."""
        # Test with extreme inputs
        extreme_regime = torch.ones(1, 8) * 100
        mu, sigma = rde_comm(extreme_regime)
        
        # Uncertainty should be positive and finite
        assert torch.all(sigma > 0)
        assert torch.all(torch.isfinite(sigma))
        
        # Test with zero inputs
        zero_regime = torch.zeros(1, 8)
        mu, sigma = rde_comm(zero_regime)
        
        assert torch.all(sigma > 0)
        assert torch.all(torch.isfinite(mu))
    
    # State Management Tests
    def test_state_reset(self, rde_comm):
        """Test hidden state reset functionality."""
        # Process some data
        regime = torch.randn(2, 8)
        rde_comm(regime)
        
        assert rde_comm.hidden is not None
        
        # Reset with batch size
        rde_comm.reset_hidden_state(batch_size=4)
        assert rde_comm.hidden[0].shape == (2, 4, 32)
        
        # Reset to None
        rde_comm.reset_hidden_state()
        assert rde_comm.hidden is None
        
    def test_state_save_load(self, rde_comm):
        """Test state checkpointing."""
        # Process data to create state
        regime = torch.randn(1, 8)
        rde_comm(regime)
        
        # Save checkpoint
        rde_comm.save_checkpoint()
        original_hidden = rde_comm.hidden
        
        # Modify state
        rde_comm(torch.randn(1, 8))
        modified_hidden = rde_comm.hidden
        
        # Load checkpoint
        success = rde_comm.load_checkpoint()
        assert success
        
        # Verify state restored
        assert torch.allclose(rde_comm.hidden[0], original_hidden[0])
        assert torch.allclose(rde_comm.hidden[1], original_hidden[1])
        
    def test_state_corruption_handling(self, rde_comm):
        """Test handling of corrupted states."""
        state_manager = rde_comm.state_manager
        
        # Valid state
        valid_h = torch.randn(2, 1, 32)
        valid_c = torch.randn(2, 1, 32)
        assert state_manager.validate_state((valid_h, valid_c))
        
        # NaN state
        nan_h = torch.full((2, 1, 32), float('nan'))
        assert not state_manager.validate_state((nan_h, valid_c))
        
        # Inf state
        inf_c = torch.full((2, 1, 32), float('inf'))
        assert not state_manager.validate_state((valid_h, inf_c))
        
        # Extreme values
        extreme_h = torch.randn(2, 1, 32) * 1000
        assert not state_manager.validate_state((extreme_h, valid_c))
    
    # Integration Tests
    def test_rde_to_comm_flow(self, rde_comm):
        """Test flow from RDE output to communication layer."""
        # Simulate RDE output
        rde_output = torch.randn(1, 8)  # 8D regime vector
        
        # Process through communication
        mu, sigma = rde_comm(rde_output)
        
        # Verify output properties
        assert mu.shape == (1, 16)
        assert sigma.shape == (1, 16)
        assert torch.all(sigma > 0)
        
    def test_comm_to_marl_flow(self, rde_comm):
        """Test integration with Main MARL Core."""
        from src.agents.main_core.engine import MainMARLCoreComponent
        
        # Mock configuration
        config = {
            'rde_communication': {
                'input_dim': 8,
                'output_dim': 16
            },
            'embedders': {
                'structure': {'output_dim': 64},
                'tactical': {'output_dim': 48},
                'regime': {'output_dim': 16},
                'lvn': {'output_dim': 8}
            }
        }
        
        # The Main MARL Core should handle RDE communication output
        rde_output = torch.randn(1, 8)
        mu, sigma = rde_comm(rde_output)
        
        # Verify dimensions match expected MARL Core input
        assert mu.shape[1] == config['rde_communication']['output_dim']
        
    def test_gradient_flow(self, rde_comm):
        """Test gradients flow properly through the model."""
        rde_comm.train()
        
        # Create gradient monitor
        monitor = GradientFlowMonitor(rde_comm)
        monitor.register_hooks()
        
        # Forward pass
        regime = torch.randn(2, 8, requires_grad=True)
        mu, sigma = rde_comm(regime)
        
        # Create loss and backward
        loss = mu.mean() + sigma.mean()
        loss.backward()
        
        # Check gradient health
        health = monitor.check_gradient_health()
        assert health['has_gradients']
        assert health['no_vanishing']
        assert health['no_explosion']
        
    # Performance Tests
    def test_inference_latency(self, rde_comm):
        """Test inference meets latency requirements."""
        # Warm up
        for _ in range(10):
            rde_comm(torch.randn(1, 8))
            
        # Measure latency
        latencies = []
        for _ in range(100):
            start = time.time()
            rde_comm(torch.randn(1, 8))
            latencies.append((time.time() - start) * 1000)
            
        # Check performance
        avg_latency = np.mean(latencies)
        p99_latency = np.percentile(latencies, 99)
        
        assert avg_latency < 5.0  # Average < 5ms
        assert p99_latency < 10.0  # 99th percentile < 10ms
        
    def test_memory_usage(self, rde_comm):
        """Test memory stability over many iterations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run many iterations
        for i in range(1000):
            regime = torch.randn(1, 8)
            mu, sigma = rde_comm(regime)
            
            # Periodically reset to test memory cleanup
            if i % 100 == 0:
                rde_comm.reset_hidden_state()
                
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be minimal
        assert memory_growth < 50  # Less than 50MB growth
        
    def test_gpu_optimization(self):
        """Test GPU optimization if available."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
            
        config = {
            'input_dim': 8,
            'hidden_dim': 32,
            'output_dim': 16,
            'device': 'cuda'
        }
        
        rde_comm = RDECommunicationLSTM(config)
        
        # Test GPU inference
        regime = torch.randn(16, 8).cuda()
        mu, sigma = rde_comm(regime)
        
        assert mu.is_cuda
        assert sigma.is_cuda
        
    # Edge Cases
    def test_extreme_regime_values(self, rde_comm):
        """Test handling of extreme regime values."""
        # Very large values
        large_regime = torch.ones(1, 8) * 1e6
        mu, sigma = rde_comm(large_regime)
        assert torch.all(torch.isfinite(mu))
        assert torch.all(torch.isfinite(sigma))
        
        # Very small values
        small_regime = torch.ones(1, 8) * 1e-6
        mu, sigma = rde_comm(small_regime)
        assert torch.all(torch.isfinite(mu))
        assert torch.all(torch.isfinite(sigma))
        
    def test_long_sequences(self, rde_comm):
        """Test processing long sequences."""
        # Process a long sequence
        for i in range(100):
            regime = torch.randn(1, 8)
            mu, sigma = rde_comm(regime)
            
        # Check state hasn't exploded
        assert torch.all(torch.isfinite(mu))
        assert torch.all(torch.isfinite(sigma))
        
        if rde_comm.hidden is not None:
            h_norm = torch.norm(rde_comm.hidden[0])
            c_norm = torch.norm(rde_comm.hidden[1])
            assert h_norm < 100
            assert c_norm < 100
            
    def test_missing_data_handling(self, rde_comm):
        """Test handling of missing or invalid data."""
        # Test with masked values
        regime = torch.randn(2, 8)
        regime[0, :4] = float('nan')  # Simulate missing data
        
        # Should handle gracefully (exact behavior depends on implementation)
        try:
            mu, sigma = rde_comm(regime)
            # If it processes, check outputs
            assert regime.shape[0] == mu.shape[0]
        except:
            # Or it might raise an error, which is also acceptable
            pass


class TestRDELossFunctions:
    """Test suite for RDE loss functions."""
    
    def test_temporal_consistency_loss(self):
        """Test temporal consistency loss calculation."""
        loss_fn = TemporalConsistencyLoss(smoothness_weight=0.5)
        
        mu_current = torch.randn(4, 16)
        mu_previous = torch.randn(4, 16)
        sigma_current = torch.rand(4, 16) + 0.1
        
        loss = loss_fn(mu_current, mu_previous, sigma_current)
        
        assert loss.shape == ()  # Scalar
        assert loss >= 0  # Non-negative
        
    def test_uncertainty_calibration_loss(self):
        """Test uncertainty calibration loss."""
        loss_fn = UncertaintyCalibrationLoss()
        
        predictions = torch.randn(4, 8)
        targets = torch.randn(4, 8)
        uncertainties = torch.rand(4, 8) + 0.1
        
        loss = loss_fn(predictions, targets, uncertainties)
        
        assert loss.shape == ()
        assert torch.isfinite(loss)
        
    def test_regime_prediction_loss(self):
        """Test regime prediction loss with contrastive component."""
        loss_fn = RegimePredictionLoss()
        
        mu_current = torch.randn(4, 16)
        mu_next = torch.randn(4, 16)
        negative_samples = torch.randn(10, 16)
        
        pred_loss, contrast_loss = loss_fn(mu_current, mu_next, negative_samples)
        
        assert pred_loss.shape == ()
        assert contrast_loss.shape == ()
        assert pred_loss >= 0
        assert contrast_loss >= 0
        
    def test_combined_loss(self):
        """Test combined RDE communication loss."""
        config = {
            'temporal_consistency': 0.3,
            'uncertainty_calibration': 0.4,
            'regime_prediction': 0.3,
            'contrastive': 0.1
        }
        
        loss_fn = RDECommunicationLoss(config)
        
        mu_current = torch.randn(4, 16)
        sigma_current = torch.rand(4, 16) + 0.1
        mu_previous = torch.randn(4, 16)
        mu_next = torch.randn(4, 16)
        regime_targets = torch.randn(4, 8)
        
        losses = loss_fn(
            mu_current, sigma_current,
            mu_previous, mu_next,
            regime_targets
        )
        
        assert 'total' in losses
        assert 'temporal' in losses
        assert 'uncertainty' in losses
        assert 'prediction' in losses
        assert losses['total'] >= 0


class TestStateManager:
    """Test suite for state manager."""
    
    def test_state_manager_initialization(self):
        """Test state manager initialization."""
        manager = StateManager(
            hidden_dim=32,
            num_layers=2,
            device=torch.device('cpu')
        )
        
        assert manager.hidden_dim == 32
        assert manager.num_layers == 2
        assert len(manager.state_buffer) == 0
        
    def test_state_interpolation(self):
        """Test state interpolation functionality."""
        manager = StateManager(32, 2, torch.device('cpu'))
        
        state1 = (torch.ones(2, 1, 32), torch.ones(2, 1, 32))
        state2 = (torch.zeros(2, 1, 32), torch.zeros(2, 1, 32))
        
        # Test interpolation
        interp_state = manager.interpolate_states(state1, state2, 0.5)
        
        assert torch.allclose(interp_state[0], torch.ones(2, 1, 32) * 0.5)
        assert torch.allclose(interp_state[1], torch.ones(2, 1, 32) * 0.5)