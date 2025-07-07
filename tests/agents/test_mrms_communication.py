"""
Test suite for MRMS Communication LSTM Layer.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

from src.agents.mrms.communication import MRMSCommunicationLSTM, RiskMemory
from src.agents.mrms.losses import MRMSCommunicationLoss


class TestMRMSCommunication:
    """Test suite for MRMS Communication Layer."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            'risk_vector_dim': 4,
            'outcome_dim': 3,
            'hidden_dim': 16,
            'output_dim': 8,
            'memory_size': 20
        }
    
    @pytest.fixture
    def comm_lstm(self, config):
        """Create MRMS Communication LSTM instance."""
        return MRMSCommunicationLSTM(config)
    
    def test_initialization(self, comm_lstm, config):
        """Test proper initialization of communication layer."""
        assert comm_lstm.input_dim == config['risk_vector_dim']
        assert comm_lstm.hidden_dim == config['hidden_dim']
        assert comm_lstm.output_dim == config['output_dim']
        assert comm_lstm.memory_size == config['memory_size']
        
        # Check risk memory initialization
        assert isinstance(comm_lstm.risk_memory, RiskMemory)
        assert comm_lstm.risk_memory.recent_stops.shape == (20,)
        assert comm_lstm.risk_memory.recent_targets.shape == (20,)
        assert comm_lstm.risk_memory.recent_sizes.shape == (20,)
        assert comm_lstm.risk_memory.performance_stats.shape == (4,)
        
    def test_forward_pass(self, comm_lstm):
        """Test forward pass produces correct shapes."""
        # Create dummy inputs
        risk_vector = torch.randn(1, 4)
        recent_outcome = torch.randn(1, 3)
        
        # Forward pass
        mu_risk, sigma_risk = comm_lstm(risk_vector, recent_outcome)
        
        # Check shapes
        assert mu_risk.shape == (1, 8)
        assert sigma_risk.shape == (1, 8)
        
        # Check uncertainty is positive
        assert torch.all(sigma_risk > 0)
        
    def test_batch_processing(self, comm_lstm):
        """Test processing multiple samples in batch."""
        batch_size = 4
        risk_vector = torch.randn(batch_size, 4)
        recent_outcome = torch.randn(batch_size, 3)
        
        mu_risk, sigma_risk = comm_lstm(risk_vector, recent_outcome)
        
        assert mu_risk.shape == (batch_size, 8)
        assert sigma_risk.shape == (batch_size, 8)
        
    def test_memory_update(self, comm_lstm):
        """Test risk memory updates correctly."""
        initial_stops = comm_lstm.risk_memory.recent_stops.clone()
        
        # Update memory
        risk_vector = torch.tensor([[2.0, 1.5, 2.0, 0.8]])
        outcome = torch.tensor([[1.0, 0.0, -0.02]])  # Hit stop
        
        comm_lstm._update_memory(risk_vector, outcome)
        
        # Check memory shifted
        assert torch.allclose(
            comm_lstm.risk_memory.recent_stops[:-1],
            initial_stops[1:]
        )
        assert comm_lstm.risk_memory.recent_stops[-1] == 1.0
        
    def test_performance_stats_update(self, comm_lstm):
        """Test performance statistics calculation."""
        # Simulate a series of trades
        for i in range(10):
            risk_vector = torch.tensor([[2.0, 1.5, 2.0, 0.8]])
            if i < 3:  # First 3 are wins
                outcome = torch.tensor([[0.0, 1.0, 0.04]])
            else:  # Rest are losses
                outcome = torch.tensor([[1.0, 0.0, -0.02]])
            
            comm_lstm._update_memory(risk_vector, outcome)
        
        # Check performance stats
        stats = comm_lstm.risk_memory.performance_stats
        assert stats[0] < 0.5  # Win rate should be low (3/10)
        assert stats[3] > 0  # Should have some drawdown
        
    def test_hidden_state_persistence(self, comm_lstm):
        """Test LSTM maintains hidden state across calls."""
        risk_vector = torch.randn(1, 4)
        outcome = torch.randn(1, 3)
        
        # First call
        mu1, _ = comm_lstm(risk_vector, outcome)
        hidden1 = comm_lstm.hidden
        
        # Second call  
        mu2, _ = comm_lstm(risk_vector, outcome)
        hidden2 = comm_lstm.hidden
        
        # Hidden state should change
        assert not torch.allclose(hidden1[0], hidden2[0])
        
        # Reset and verify
        comm_lstm.reset_hidden_state()
        assert comm_lstm.hidden is None
        
    def test_performance_adaptation(self, comm_lstm):
        """Test risk adaptation based on performance."""
        # Simulate losing streak
        for _ in range(5):
            risk_vector = torch.tensor([[3.0, 1.5, 2.0, 0.8]])
            outcome = torch.tensor([[1.0, 0.0, -0.02]])  # Stops hit
            mu, sigma = comm_lstm(risk_vector, outcome, update_memory=True)
        
        # Uncertainty should increase
        assert sigma.mean() > 0.1
        
        # Check performance stats updated
        stats = comm_lstm.risk_memory.performance_stats
        assert stats[0] < 0.5  # Win rate should be low
        
    def test_risk_adaptation(self, comm_lstm):
        """Test position size adaptation logic."""
        # Test with normal conditions
        adapted_size = comm_lstm._adapt_risk_parameters(3.0, 0.2)
        assert adapted_size == 3.0  # No reduction
        
        # Test with high uncertainty
        adapted_size = comm_lstm._adapt_risk_parameters(3.0, 0.5)
        assert adapted_size < 3.0  # Should reduce
        
        # Simulate losing streak
        comm_lstm.risk_memory.performance_stats[0] = 0.3  # Low win rate
        comm_lstm.risk_memory.performance_stats[3] = 0.3  # High drawdown
        
        adapted_size = comm_lstm._adapt_risk_parameters(3.0, 0.2)
        assert adapted_size < 3.0  # Should reduce due to poor performance
        
    def test_get_recent_outcome_vector(self, comm_lstm):
        """Test recent outcome vector generation."""
        # Initially should return zeros
        outcome_vec = comm_lstm.get_recent_outcome_vector()
        assert outcome_vec.shape == (1, 3)
        assert torch.allclose(outcome_vec, torch.zeros(1, 3))
        
        # Add some outcomes
        for i in range(3):
            risk_vector = torch.tensor([[2.0, 1.5, 2.0, 0.8]])
            outcome = torch.tensor([[float(i % 2), float((i+1) % 2), 0.02]])
            comm_lstm._update_memory(risk_vector, outcome)
        
        outcome_vec = comm_lstm.get_recent_outcome_vector()
        assert outcome_vec.shape == (1, 3)
        assert not torch.allclose(outcome_vec, torch.zeros(1, 3))
        
    def test_integration_with_mrms(self, config):
        """Test integration with MRMS engine."""
        from src.agents.mrms.engine import MRMSComponent
        
        # Create MRMS with communication
        mrms_config = {'communication': config}
        with patch('src.agents.mrms.engine.RiskManagementEnsemble'):
            mrms = MRMSComponent(mrms_config)
            
            # Verify communication layer exists
            assert hasattr(mrms, 'communication_lstm')
            assert isinstance(mrms.communication_lstm, MRMSCommunicationLSTM)
    
    def test_loss_calculation(self, comm_lstm, config):
        """Test loss function calculations."""
        loss_fn = MRMSCommunicationLoss(config)
        
        # Create dummy data
        mu_risk = torch.randn(8, 8)
        sigma_risk = torch.rand(8, 8) + 0.1
        target_risk = torch.randn(8, 4)
        target_outcome = torch.randint(0, 2, (8, 3)).float()
        
        # Calculate losses
        losses = loss_fn(mu_risk, sigma_risk, target_risk, target_outcome)
        
        # Verify all components present
        assert 'risk' in losses
        assert 'outcome' in losses
        assert 'uncertainty' in losses
        assert 'temporal' in losses
        assert 'total' in losses
        
        # Verify total loss is weighted sum
        expected_total = (
            0.3 * losses['risk'] +
            0.3 * losses['outcome'] +
            0.2 * losses['uncertainty'] +
            0.2 * losses['temporal']
        )
        assert torch.allclose(losses['total'], expected_total)
        
    def test_temporal_consistency_loss(self, config):
        """Test temporal consistency in loss calculation."""
        loss_fn = MRMSCommunicationLoss(config)
        
        # Create data with previous embedding
        mu_risk = torch.randn(4, 8)
        sigma_risk = torch.rand(4, 8) + 0.1
        target_risk = torch.randn(4, 4)
        target_outcome = torch.randint(0, 2, (4, 3)).float()
        previous_mu = torch.randn(4, 8)
        
        # Calculate with temporal consistency
        losses = loss_fn(mu_risk, sigma_risk, target_risk, target_outcome, previous_mu)
        
        # Temporal loss should be non-zero
        assert losses['temporal'] > 0
        
    def test_model_save_load(self, comm_lstm, tmp_path):
        """Test saving and loading model weights."""
        # Save model
        save_path = tmp_path / "comm_lstm.pth"
        torch.save(comm_lstm.state_dict(), save_path)
        
        # Create new instance and load
        new_lstm = MRMSCommunicationLSTM(comm_lstm.__dict__)
        new_lstm.load_state_dict(torch.load(save_path))
        
        # Verify weights match
        for (n1, p1), (n2, p2) in zip(comm_lstm.named_parameters(), 
                                       new_lstm.named_parameters()):
            assert n1 == n2
            assert torch.allclose(p1, p2)
            
    def test_gradient_flow(self, comm_lstm):
        """Test gradients flow through the model."""
        comm_lstm.train()
        
        risk_vector = torch.randn(2, 4, requires_grad=True)
        outcome = torch.randn(2, 3)
        
        mu, sigma = comm_lstm(risk_vector, outcome)
        loss = mu.mean() + sigma.mean()
        loss.backward()
        
        # Check gradients exist
        assert risk_vector.grad is not None
        for param in comm_lstm.parameters():
            assert param.grad is not None
            
    def test_edge_cases(self, comm_lstm):
        """Test edge cases and error handling."""
        # Test with zero inputs
        risk_vector = torch.zeros(1, 4)
        outcome = torch.zeros(1, 3)
        
        mu, sigma = comm_lstm(risk_vector, outcome)
        assert not torch.isnan(mu).any()
        assert not torch.isnan(sigma).any()
        
        # Test with extreme values
        risk_vector = torch.ones(1, 4) * 100
        outcome = torch.ones(1, 3)
        
        mu, sigma = comm_lstm(risk_vector, outcome)
        assert torch.all(torch.abs(mu) <= 1.0)  # Tanh bounded
        assert torch.all(sigma > 0)  # Positive uncertainty