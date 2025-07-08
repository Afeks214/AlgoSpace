"""
Comprehensive tests for BiLSTM implementation in TacticalEmbedder.

This module provides extensive testing for the BiLSTM upgrade including
dimension handling, gradient flow, performance, and integration tests.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import time

from src.agents.main_core.models import TacticalEmbedder
from src.agents.main_core.tactical_bilstm_components import (
    BiLSTMGateController,
    TemporalPyramidPooling,
    BiLSTMPositionalEncoding,
    DirectionalFeatureFusion,
    BiLSTMTemporalMasking
)


class TestBiLSTMImplementation:
    """Test suite specifically for BiLSTM upgrade."""
    
    @pytest.fixture
    def embedder(self):
        """Create tactical embedder with BiLSTM."""
        return TacticalEmbedder(
            input_dim=7,
            hidden_dim=64,
            output_dim=48,
            dropout_rate=0.2
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample 60x7 matrix."""
        return torch.randn(4, 60, 7)  # Batch of 4
    
    def test_bilstm_configuration(self, embedder):
        """Test that BiLSTM is properly configured."""
        info = embedder.get_bilstm_info()
        
        assert info['is_bilstm'] == True
        assert info['bilstm_output_dim'] == info['hidden_dim'] * 2
        assert info['has_gate_controller'] == True
        assert info['has_pyramid_pooling'] == True
        assert info['has_positional_encoding'] == True
        assert info['has_directional_fusion'] == True
        assert info['has_temporal_masking'] == True
    
    def test_bilstm_output_dimensions(self, embedder, sample_data):
        """Test that BiLSTM outputs correct dimensions."""
        # Process through first BiLSTM layer
        h = embedder.input_projection(sample_data)
        h = embedder.position_encoder(h)
        
        # First LSTM layer
        bilstm = embedder.lstm_layers[0]
        output, (hn, cn) = bilstm(h)
        
        # Check dimensions
        batch_size, seq_len, _ = sample_data.shape
        expected_output_dim = embedder.hidden_dim * 2  # Bidirectional
        
        assert output.shape == (batch_size, seq_len, expected_output_dim)
        assert hn.shape == (2, batch_size, embedder.hidden_dim)  # 2 directions
        assert cn.shape == (2, batch_size, embedder.hidden_dim)
    
    def test_forward_backward_split(self, embedder, sample_data):
        """Test proper splitting of forward and backward features."""
        # Process through BiLSTM
        h = embedder.input_projection(sample_data)
        h = embedder.position_encoder(h)
        bilstm = embedder.lstm_layers[0]
        output, _ = bilstm(h)
        
        # Split forward and backward
        hidden_dim = embedder.hidden_dim
        forward = output[:, :, :hidden_dim]
        backward = output[:, :, hidden_dim:]
        
        # Check dimensions
        assert forward.shape == (4, 60, hidden_dim)
        assert backward.shape == (4, 60, hidden_dim)
        
        # Check that they're different (not just duplicated)
        assert not torch.allclose(forward, backward)
    
    def test_gate_controller(self):
        """Test BiLSTM gate controller."""
        controller = BiLSTMGateController(hidden_dim=64)
        
        # Create dummy forward and backward features
        forward = torch.randn(2, 10, 64)
        backward = torch.randn(2, 10, 64)
        
        # Apply gating
        gated = controller(forward, backward)
        
        # Check output
        assert gated.shape == (2, 10, 128)  # Concatenated dimension
        assert not torch.isnan(gated).any()
    
    def test_temporal_pyramid_pooling(self):
        """Test temporal pyramid pooling."""
        pooling = TemporalPyramidPooling(
            input_dim=128,
            pyramid_levels=[1, 2, 4, 8]
        )
        
        # Create dummy BiLSTM output
        bilstm_output = torch.randn(2, 60, 128)
        
        # Apply pooling
        pooled = pooling(bilstm_output)
        
        # Check output
        assert pooled.shape == (2, 128)
        assert not torch.isnan(pooled).any()
    
    def test_positional_encoding(self):
        """Test BiLSTM positional encoding."""
        encoding = BiLSTMPositionalEncoding(hidden_dim=64, max_len=100)
        
        # Create dummy features
        forward = torch.randn(2, 60, 64)
        backward = torch.randn(2, 60, 64)
        
        # Apply encoding
        forward_enc, backward_enc = encoding(forward, backward)
        
        # Check outputs
        assert forward_enc.shape == forward.shape
        assert backward_enc.shape == backward.shape
        
        # Check that encoding was applied (features changed)
        assert not torch.allclose(forward, forward_enc)
        assert not torch.allclose(backward, backward_enc)
    
    def test_directional_fusion(self):
        """Test directional feature fusion."""
        fusion = DirectionalFeatureFusion(hidden_dim=64)
        
        # Create dummy features
        forward = torch.randn(2, 60, 64)
        backward = torch.randn(2, 60, 64)
        
        # Apply fusion
        fused = fusion(forward, backward)
        
        # Check output
        assert fused.shape == (2, 60, 64)
        assert not torch.isnan(fused).any()
    
    def test_temporal_masking(self):
        """Test BiLSTM temporal masking."""
        masking = BiLSTMTemporalMasking(hidden_dim=64)
        
        # Create dummy BiLSTM output
        bilstm_out = torch.randn(2, 60, 128)
        
        # Apply masking
        masked = masking(bilstm_out)
        
        # Check output
        assert masked.shape == bilstm_out.shape
        assert not torch.isnan(masked).any()
    
    def test_full_forward_pass(self, embedder, sample_data):
        """Test complete forward pass with BiLSTM."""
        mu, sigma = embedder(sample_data)
        
        # Check final output
        assert mu.shape == (4, 48)  # Batch size 4, output dim 48
        assert sigma.shape == (4, 48)
        assert not torch.isnan(mu).any()
        assert not torch.isnan(sigma).any()
        assert torch.isfinite(mu).all()
        assert torch.isfinite(sigma).all()
        assert (sigma > 0).all()  # Ensure positive uncertainty
    
    def test_gradient_flow_bilstm(self, embedder, sample_data):
        """Test gradient flow through BiLSTM layers."""
        sample_data.requires_grad = True
        
        # Forward pass
        mu, sigma = embedder(sample_data)
        loss = mu.mean() + sigma.mean()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        assert sample_data.grad is not None
        assert not torch.isnan(sample_data.grad).any()
        
        # Check BiLSTM parameters have gradients
        for i, lstm in enumerate(embedder.lstm_layers):
            for name, param in lstm.named_parameters():
                if param.requires_grad:
                    assert param.grad is not None, f"No gradient for LSTM {i} {name}"
                    assert not torch.isnan(param.grad).any(), f"NaN gradient in LSTM {i} {name}"
        
        # Check enhancement components have gradients
        for component_name in ['gate_controller', 'pyramid_pooling', 'directional_fusion']:
            if hasattr(embedder, component_name):
                component = getattr(embedder, component_name)
                for name, param in component.named_parameters():
                    if param.requires_grad:
                        assert param.grad is not None, f"No gradient for {component_name}.{name}"
    
    def test_performance_with_bilstm(self, embedder, sample_data):
        """Test that BiLSTM meets performance requirements."""
        embedder.eval()
        
        # Warm up
        for _ in range(10):
            _ = embedder(sample_data)
        
        # Time inference
        times = []
        for _ in range(100):
            start = time.perf_counter()
            _ = embedder(sample_data)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append((time.perf_counter() - start) * 1000)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"Average BiLSTM inference time: {avg_time:.2f}ms ± {std_time:.2f}ms")
        
        # Should still meet reasonable performance despite BiLSTM
        assert avg_time < 10.0, f"BiLSTM inference too slow: {avg_time:.2f}ms"
    
    def test_mc_dropout_with_bilstm(self, embedder, sample_data):
        """Test MC Dropout functionality with BiLSTM."""
        # Enable MC dropout
        embedder.enable_mc_dropout()
        
        # Run multiple forward passes
        outputs = []
        for _ in range(10):
            mu, sigma = embedder(sample_data)
            outputs.append(mu)
        
        outputs = torch.stack(outputs)
        
        # Check variance across runs (should be non-zero due to dropout)
        variance = outputs.var(dim=0)
        assert (variance > 0).any(), "MC Dropout not producing variation"
    
    def test_attention_weights_extraction(self, embedder, sample_data):
        """Test extraction of attention weights."""
        mu, sigma, attention_weights = embedder(sample_data, return_attention_weights=True)
        
        # Check attention weights
        assert attention_weights is not None
        assert attention_weights.shape[0] == 4  # Batch size
        assert attention_weights.shape[1] == 60  # Sequence length
        assert torch.allclose(attention_weights.sum(dim=1), torch.ones(4))  # Sum to 1
    
    def test_enhancement_integration(self):
        """Test integration with tactical enhancements."""
        embedder = TacticalEmbedder(
            input_dim=7,
            hidden_dim=64,
            output_dim=48,
            use_enhancements=True
        )
        
        # Check that enhancement integrator is present
        assert hasattr(embedder, 'enhancement_integrator')
        assert hasattr(embedder, 'enhanced_projection')
        
        # Test forward pass with enhancements
        sample_data = torch.randn(2, 60, 7)
        mu, sigma = embedder(sample_data)
        
        assert mu.shape == (2, 48)
        assert sigma.shape == (2, 48)
    
    def test_bilstm_state_persistence(self, embedder, sample_data):
        """Test that BiLSTM states can be accessed for analysis."""
        # Enable return_all_states
        mu, sigma, all_states = embedder(sample_data, return_all_states=True)
        
        # Check that we get states from all layers
        assert 'lstm_states' in all_states
        assert len(all_states['lstm_states']) == len(embedder.lstm_layers)
        
        # Check state dimensions
        for i, state in enumerate(all_states['lstm_states']):
            assert state.shape == (4, 60, embedder.hidden_dim * 2)
    
    def test_optimized_forward_compatibility(self, embedder, sample_data):
        """Test optimized forward pass with BiLSTM."""
        embedder.eval()
        
        # Regular forward
        mu1, sigma1 = embedder(sample_data)
        
        # Optimized forward
        mu2, sigma2 = embedder.forward_optimized(sample_data)
        
        # Should produce similar results
        assert torch.allclose(mu1, mu2, atol=1e-4)
        assert torch.allclose(sigma1, sigma2, atol=1e-4)


class TestBiLSTMComponentsIntegration:
    """Test integration of individual BiLSTM components."""
    
    def test_component_chain(self):
        """Test chaining BiLSTM components together."""
        # Create components
        hidden_dim = 64
        positional = BiLSTMPositionalEncoding(hidden_dim)
        gate = BiLSTMGateController(hidden_dim)
        fusion = DirectionalFeatureFusion(hidden_dim)
        pyramid = TemporalPyramidPooling(hidden_dim * 2)
        
        # Create dummy BiLSTM output
        batch_size, seq_len = 2, 60
        forward = torch.randn(batch_size, seq_len, hidden_dim)
        backward = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Process through chain
        forward_enc, backward_enc = positional(forward, backward)
        gated = gate(forward_enc, backward_enc)
        fused = fusion(forward_enc, backward_enc)
        pooled = pyramid(gated)
        
        # Check final output
        assert pooled.shape == (batch_size, hidden_dim * 2)
        assert not torch.isnan(pooled).any()
    
    def test_component_gradients(self):
        """Test gradient flow through component chain."""
        # Create components with requires_grad
        hidden_dim = 32
        components = nn.ModuleList([
            BiLSTMPositionalEncoding(hidden_dim),
            BiLSTMGateController(hidden_dim),
            DirectionalFeatureFusion(hidden_dim),
            TemporalPyramidPooling(hidden_dim * 2)
        ])
        
        # Create input
        forward = torch.randn(2, 30, hidden_dim, requires_grad=True)
        backward = torch.randn(2, 30, hidden_dim, requires_grad=True)
        
        # Forward through components
        f_enc, b_enc = components[0](forward, backward)
        gated = components[1](f_enc, b_enc)
        fused = components[2](f_enc, b_enc)
        pooled = components[3](gated)
        
        # Compute loss and backward
        loss = pooled.mean()
        loss.backward()
        
        # Check gradients exist
        assert forward.grad is not None
        assert backward.grad is not None
        
        for component in components:
            for param in component.parameters():
                if param.requires_grad:
                    assert param.grad is not None