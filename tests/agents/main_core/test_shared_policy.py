"""
File: tests/agents/main_core/test_shared_policy.py (NEW FILE)
Comprehensive test suite for shared policy
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import time

from src.agents.main_core.shared_policy import (
    SharedPolicy, PolicyOutput, CrossFeatureAttention,
    TemporalConsistencyModule, MultiHeadReasoner,
    ActionDistributionModule
)
from src.agents.main_core.mc_dropout_policy import MCDropoutConsensus
from src.agents.main_core.multi_objective_value import MultiObjectiveValueFunction

class TestSharedPolicy:
    """Test suite for shared policy network."""
    
    @pytest.fixture
    def config(self):
        return {
            'input_dim': 136,
            'hidden_dim': 256,
            'action_dim': 2,
            'dropout_rate': 0.2,
            'use_temporal_consistency': True
        }
        
    @pytest.fixture
    def policy(self, config):
        return SharedPolicy(config)
        
    def test_initialization(self, policy, config):
        """Test proper initialization."""
        assert policy.input_dim == config['input_dim']
        assert policy.hidden_dim == config['hidden_dim']
        assert policy.action_dim == config['action_dim']
        
    def test_forward_pass(self, policy):
        """Test basic forward pass."""
        batch_size = 4
        unified_state = torch.randn(batch_size, 136)
        
        output = policy(unified_state)
        
        assert isinstance(output, PolicyOutput)
        assert output.action_logits.shape == (batch_size, 2)
        assert output.action_probs.shape == (batch_size, 2)
        assert output.state_value.shape == (batch_size, 1)
        
        # Check probabilities sum to 1
        assert torch.allclose(output.action_probs.sum(dim=-1), 
                            torch.ones(batch_size), atol=1e-6)
                            
    def test_cross_feature_attention(self):
        """Test cross-feature attention module."""
        attention = CrossFeatureAttention([64, 48, 16, 8], n_heads=4)
        
        embeddings = [
            torch.randn(2, 64),  # structure
            torch.randn(2, 48),  # tactical
            torch.randn(2, 16),  # regime
            torch.randn(2, 8)    # lvn
        ]
        
        output, weights = attention(embeddings)
        
        assert output.shape == (2, 128)
        assert weights is not None
        
    def test_temporal_consistency(self):
        """Test temporal consistency module."""
        temporal = TemporalConsistencyModule(hidden_dim=64)
        
        # Add some history
        for _ in range(5):
            features = torch.randn(1, 64)
            probs = torch.softmax(torch.randn(1, 2), dim=-1)
            _, score = temporal(features, probs)
            
        # Check consistency score
        assert 0 <= score <= 1
        assert len(temporal.decision_memory) == 5
        
    def test_multi_head_reasoner(self):
        """Test multi-head reasoning."""
        reasoner = MultiHeadReasoner(input_dim=128, hidden_dim=256)
        features = torch.randn(2, 128)
        
        output, scores = reasoner(features)
        
        assert output.shape == (2, 256)
        assert 'structure' in scores
        assert 'timing' in scores
        assert 'risk' in scores
        assert 'regime' in scores
        
        # Check weights sum to 1
        total_weight = sum(v for k, v in scores.items() if k.endswith('_weight'))
        assert abs(total_weight - 1.0) < 1e-6
        
    def test_action_distribution(self):
        """Test action distribution module."""
        action_dist = ActionDistributionModule(input_dim=256, action_dim=2)
        features = torch.randn(4, 256)
        
        logits, probs, temp = action_dist(features, use_temperature=True)
        
        assert logits.shape == (4, 2)
        assert probs.shape == (4, 2)
        assert 0.5 <= temp <= 2.0
        
    def test_deterministic_action(self, policy):
        """Test deterministic action selection."""
        state = torch.randn(1, 136)
        
        action1, conf1 = policy.get_action(state, deterministic=True)
        action2, conf2 = policy.get_action(state, deterministic=True)
        
        # Should be consistent
        assert action1 == action2
        assert conf1 == conf2
        assert action1 in [0, 1]
        assert 0 <= conf1 <= 1
        
    def test_evaluate_actions(self, policy):
        """Test action evaluation for training."""
        states = torch.randn(8, 136)
        actions = torch.randint(0, 2, (8,))
        
        result = policy.evaluate_actions(states, actions)
        
        assert 'log_probs' in result
        assert 'entropy' in result
        assert 'values' in result
        assert 'action_probs' in result
        
        assert result['log_probs'].shape == (8,)
        assert result['entropy'].shape == (8,)
        assert result['values'].shape == (8,)
        
    def test_gradient_flow(self, policy):
        """Test gradient flow through network."""
        state = torch.randn(2, 136, requires_grad=True)
        
        output = policy(state)
        loss = output.action_logits.sum() + output.state_value.sum()
        loss.backward()
        
        assert state.grad is not None
        assert not torch.isnan(state.grad).any()
        
    def test_mc_dropout_integration(self, policy):
        """Test MC Dropout consensus integration."""
        mc_config = {
            'n_samples': 20,
            'confidence_threshold': 0.8,
            'use_adaptive_sampling': False
        }
        mc_dropout = MCDropoutConsensus(mc_config)
        
        state = torch.randn(1, 136)
        result = mc_dropout.evaluate(policy, state)
        
        assert isinstance(result.should_qualify, bool)
        assert 0 <= result.confidence <= 1
        assert 0 <= result.uncertainty <= 1
        assert result.mean_probs.shape == (1, 2)
        assert result.std_probs.shape == (1, 2)
        
    def test_multi_objective_value(self):
        """Test multi-objective value function."""
        value_fn = MultiObjectiveValueFunction(input_dim=256)
        features = torch.randn(4, 256)
        
        result = value_fn(features, return_components=True)
        
        assert 'value' in result
        assert 'components' in result
        assert 'weights' in result
        
        assert result['value'].shape == (4, 1)
        assert len(result['components']) == 4
        assert sum(result['weights'].values()) == pytest.approx(1.0, rel=1e-6)
        
    def test_performance_requirements(self, policy):
        """Test performance meets requirements."""
        state = torch.randn(1, 136)
        
        # Warm up
        _ = policy(state)
        
        # Time inference
        times = []
        for _ in range(100):
            start = time.time()
            _ = policy(state, return_value=False)
            times.append((time.time() - start) * 1000)
            
        avg_time = np.mean(times)
        assert avg_time < 10.0  # Must be under 10ms
        
    def test_batch_consistency(self, policy):
        """Test consistency across batch sizes."""
        single_state = torch.randn(1, 136)
        batch_state = single_state.repeat(4, 1)
        
        single_output = policy(single_state)
        batch_output = policy(batch_state)
        
        # First item in batch should match single
        assert torch.allclose(single_output.action_probs, 
                            batch_output.action_probs[0:1], atol=1e-6)