"""
Comprehensive test suite for LVN Embedder.

Tests both simple and advanced implementations, context building,
loss functions, and integration with the Main MARL Core.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List

from src.agents.main_core.models import LVNEmbedder
from src.agents.main_core.lvn_embedder import (
    LVNContext,
    SpatialRelationshipModule,
    TemporalTrackingModule,
    AttentionRelevanceModule
)
from src.agents.main_core.lvn_context_builder import LVNContextBuilder
from src.agents.main_core.lvn_losses import (
    LVNRelevanceLoss,
    LVNInteractionLoss,
    LVNSpatialConsistencyLoss,
    LVNTemporalConsistencyLoss,
    LVNCompositeLoss
)


class TestLVNEmbedder:
    """Test suite for LVN Embedder."""
    
    @pytest.fixture
    def simple_embedder(self):
        """Create simple LVN embedder."""
        return LVNEmbedder(
            input_dim=5,
            output_dim=8,
            hidden_dim=16,
            use_advanced=False
        )
        
    @pytest.fixture
    def advanced_embedder(self):
        """Create advanced LVN embedder."""
        return LVNEmbedder(
            input_dim=5,
            output_dim=32,
            hidden_dim=64,
            use_advanced=True,
            max_levels=10
        )
        
    @pytest.fixture
    def sample_lvn_data(self):
        """Generate sample LVN data."""
        return {
            'all_levels': [
                {'price': 100.0, 'strength': 80.0, 'distance': 0.5, 'volume': 1000},
                {'price': 98.5, 'strength': 70.0, 'distance': 2.0, 'volume': 800},
                {'price': 101.5, 'strength': 85.0, 'distance': 1.0, 'volume': 1200},
                {'price': 97.0, 'strength': 60.0, 'distance': 3.5, 'volume': 600},
                {'price': 103.0, 'strength': 75.0, 'distance': 2.5, 'volume': 900}
            ]
        }
        
    @pytest.fixture
    def market_state(self):
        """Generate sample market state."""
        return {
            'current_price': 100.5,
            'volatility': 0.015,
            'volume': 50000,
            'regime': 'normal'
        }
        
    def test_simple_embedder_forward(self, simple_embedder):
        """Test simple embedder forward pass."""
        # Create input
        batch_size = 4
        input_features = torch.randn(batch_size, 5)
        
        # Forward pass
        output = simple_embedder(input_features)
        
        # Check output shape
        assert output.shape == (batch_size, 8)
        
        # Check output is valid
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()
        
    def test_advanced_embedder_forward(self, advanced_embedder, sample_lvn_data, market_state):
        """Test advanced embedder forward pass."""
        # Create context
        context_builder = LVNContextBuilder(max_levels=10)
        context = context_builder.build_context(sample_lvn_data, market_state)
        
        # Forward pass
        results = advanced_embedder(context)
        
        # Check outputs
        assert isinstance(results, dict)
        assert 'embedding' in results
        assert 'uncertainty' in results
        assert 'relevance_scores' in results
        assert 'interaction_predictions' in results
        
        # Check shapes
        assert results['embedding'].shape == (1, 32)
        assert results['uncertainty'].shape == (1, 1)
        assert results['relevance_scores'].shape[1] == len(sample_lvn_data['all_levels'])
        
        # Check values are valid
        assert not torch.isnan(results['embedding']).any()
        assert results['uncertainty'].item() > 0
        assert (results['relevance_scores'] >= 0).all()
        assert (results['relevance_scores'] <= 1).all()
        
    def test_context_builder(self, sample_lvn_data, market_state):
        """Test LVN context builder."""
        builder = LVNContextBuilder(max_levels=10)
        
        # Build context
        context = builder.build_context(sample_lvn_data, market_state)
        
        # Check context
        assert isinstance(context, LVNContext)
        assert context.current_price == market_state['current_price']
        assert len(context.prices) == len(sample_lvn_data['all_levels'])
        assert len(context.strengths) == len(context.prices)
        assert len(context.distances) == len(context.prices)
        assert len(context.volumes) == len(context.prices)
        assert len(context.price_history) == 20  # Default history length
        
        # Check data validity
        assert (context.strengths >= 0).all()
        assert (context.strengths <= 100).all()
        assert (context.distances >= 0).all()
        
    def test_spatial_relationship_module(self):
        """Test spatial relationship module."""
        module = SpatialRelationshipModule(hidden_dim=64)
        
        batch_size = 2
        n_levels = 5
        hidden_dim = 64
        
        # Create inputs
        embeddings = torch.randn(batch_size, n_levels, hidden_dim)
        distances = torch.rand(batch_size, n_levels, 1) * 10
        strengths = torch.rand(batch_size, n_levels, 1) * 100
        
        # Forward pass
        output = module(embeddings, distances, strengths)
        
        # Check output
        assert output.shape == (batch_size, n_levels, hidden_dim)
        assert not torch.isnan(output).any()
        
    def test_temporal_tracking_module(self):
        """Test temporal tracking module."""
        module = TemporalTrackingModule(hidden_dim=64, sequence_length=20)
        
        batch_size = 2
        seq_len = 20
        n_levels = 5
        hidden_dim = 64
        
        # Create inputs
        price_trajectory = torch.randn(batch_size, seq_len, hidden_dim)
        lvn_embeddings = torch.randn(batch_size, n_levels, hidden_dim)
        
        # Forward pass
        temporal_features, interaction_preds = module(price_trajectory, lvn_embeddings)
        
        # Check outputs
        assert temporal_features.shape == (batch_size, hidden_dim)
        assert interaction_preds.shape == (batch_size, n_levels, 3)
        
        # Check interaction predictions sum to 1
        interaction_probs = torch.softmax(interaction_preds, dim=-1)
        assert torch.allclose(interaction_probs.sum(dim=-1), torch.ones(batch_size, n_levels))
        
    def test_attention_relevance_module(self):
        """Test attention relevance module."""
        module = AttentionRelevanceModule(hidden_dim=64)
        
        batch_size = 2
        n_levels = 5
        hidden_dim = 64
        
        # Create inputs
        market_state = torch.randn(batch_size, hidden_dim)
        lvn_features = torch.randn(batch_size, n_levels, hidden_dim)
        
        # Forward pass
        weighted_features, relevance_scores = module(market_state, lvn_features)
        
        # Check outputs
        assert weighted_features.shape == (batch_size, hidden_dim)
        assert relevance_scores.shape == (batch_size, n_levels)
        assert (relevance_scores >= 0).all()
        assert (relevance_scores <= 1).all()
        
    def test_mc_dropout_support(self, advanced_embedder):
        """Test MC Dropout support in embedder."""
        # Enable MC dropout
        advanced_embedder.enable_mc_dropout()
        
        # Get dropout layers
        dropout_layers = advanced_embedder.get_dropout_layers()
        assert len(dropout_layers) > 0
        
        # Set dropout rate
        advanced_embedder.set_dropout_rate(0.5)
        for layer in dropout_layers:
            assert layer.p == 0.5
            
    def test_top_levels_extraction(self, advanced_embedder, sample_lvn_data, market_state):
        """Test extraction of top relevant levels."""
        # Build context
        context_builder = LVNContextBuilder()
        context = context_builder.build_context(sample_lvn_data, market_state)
        
        # Get top levels
        top_levels = advanced_embedder.get_top_levels(context, k=3)
        
        # Check results
        assert 'indices' in top_levels
        assert 'prices' in top_levels
        assert 'strengths' in top_levels
        assert 'relevance_scores' in top_levels
        
        assert len(top_levels['indices']) <= 3
        assert len(top_levels['prices']) == len(top_levels['indices'])


class TestLVNLosses:
    """Test suite for LVN loss functions."""
    
    def test_relevance_loss(self):
        """Test relevance loss calculation."""
        loss_fn = LVNRelevanceLoss(margin=0.2)
        
        batch_size = 4
        n_levels = 5
        
        # Create inputs
        relevance_scores = torch.rand(batch_size, n_levels)
        actual_interactions = torch.randint(0, 2, (batch_size, n_levels))
        distances = torch.rand(batch_size, n_levels) * 10
        
        # Calculate loss
        loss = loss_fn(relevance_scores, actual_interactions, distances)
        
        # Check loss
        assert loss.shape == ()
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        
    def test_interaction_loss(self):
        """Test interaction prediction loss."""
        loss_fn = LVNInteractionLoss()
        
        batch_size = 4
        n_levels = 5
        
        # Create inputs
        predictions = torch.randn(batch_size, n_levels, 3)
        targets = torch.randint(0, 3, (batch_size, n_levels))
        strengths = torch.rand(batch_size, n_levels) * 100
        
        # Calculate loss
        loss = loss_fn(predictions, targets, strengths)
        
        # Check loss
        assert loss.shape == ()
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        
    def test_spatial_consistency_loss(self):
        """Test spatial consistency loss."""
        loss_fn = LVNSpatialConsistencyLoss(temperature=0.1)
        
        batch_size = 2
        n_levels = 5
        hidden_dim = 64
        
        # Create inputs
        embeddings = torch.randn(batch_size, n_levels, hidden_dim)
        prices = torch.tensor([[100.0, 100.5, 98.0, 102.0, 97.0],
                              [100.0, 99.5, 101.0, 98.5, 103.0]])
        
        # Calculate loss
        loss = loss_fn(embeddings, prices, price_threshold=1.0)
        
        # Check loss
        assert loss.shape == ()
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        
    def test_temporal_consistency_loss(self):
        """Test temporal consistency loss."""
        loss_fn = LVNTemporalConsistencyLoss(alpha=0.5)
        
        batch_size = 4
        n_levels = 5
        
        # Create inputs
        current_relevance = torch.rand(batch_size, n_levels)
        previous_relevance = torch.rand(batch_size, n_levels)
        price_change = torch.rand(batch_size, 1) * 0.02
        
        # Calculate loss
        loss = loss_fn(current_relevance, previous_relevance, price_change)
        
        # Check loss
        assert loss.shape == ()
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        
    def test_composite_loss(self):
        """Test composite loss function."""
        loss_fn = LVNCompositeLoss()
        
        batch_size = 2
        n_levels = 5
        hidden_dim = 64
        
        # Create predictions
        predictions = {
            'relevance_scores': torch.rand(batch_size, n_levels),
            'interaction_predictions': torch.randn(batch_size, n_levels, 3),
            'spatial_features': torch.randn(batch_size, n_levels, hidden_dim)
        }
        
        # Create targets
        targets = {
            'actual_interactions': torch.randint(0, 2, (batch_size, n_levels)),
            'interaction_targets': torch.randint(0, 3, (batch_size, n_levels))
        }
        
        # Create context
        context = {
            'distances': torch.rand(batch_size, n_levels),
            'strengths': torch.rand(batch_size, n_levels) * 100,
            'prices': torch.rand(batch_size, n_levels) * 10 + 95
        }
        
        # Calculate loss
        total_loss, loss_components = loss_fn(predictions, targets, context)
        
        # Check outputs
        assert total_loss.shape == ()
        assert total_loss.item() >= 0
        assert isinstance(loss_components, dict)
        assert all(k.startswith('lvn_') for k in loss_components)


class TestIntegration:
    """Integration tests for LVN embedder."""
    
    def test_embedder_with_models_integration(self):
        """Test integration with models.py."""
        from src.agents.main_core.models import UnifiedIntelligenceSystem
        
        # Create system with advanced LVN embedder
        system = UnifiedIntelligenceSystem()
        
        # Check LVN embedder exists
        assert hasattr(system.embedders, 'lvn')
        
        # Create sample input
        batch_size = 2
        embedder_inputs = {
            'structure': torch.randn(batch_size, 384),
            'tactical': torch.randn(batch_size, 420),
            'regime': torch.randn(batch_size, 10),
            'lvn': torch.randn(batch_size, 5)
        }
        
        # Forward pass
        output = system(embedder_inputs)
        
        # Check output
        assert 'action_logits' in output
        assert output['action_logits'].shape == (batch_size, 2)
        
    def test_context_builder_interaction_tracking(self):
        """Test interaction tracking in context builder."""
        builder = LVNContextBuilder(interaction_lookback=50)
        
        # Simulate multiple updates
        for i in range(10):
            lvn_data = {
                'all_levels': [
                    {'price': 100.0 + i * 0.1, 'strength': 80.0, 'distance': 0.5, 'volume': 1000}
                ]
            }
            market_state = {'current_price': 100.0 + i * 0.2}
            
            context = builder.build_context(lvn_data, market_state)
            
        # Check interaction history
        stats = builder.get_statistics()
        assert stats['price_history_length'] == 10
        assert stats['interaction_buffer_size'] >= 0
        
    def test_embedder_performance(self, advanced_embedder):
        """Test embedder performance and latency."""
        import time
        
        # Create context
        context = LVNContext(
            prices=torch.rand(10) * 10 + 95,
            strengths=torch.rand(10) * 100,
            distances=torch.rand(10) * 5,
            volumes=torch.rand(10) * 1000,
            current_price=100.0,
            price_history=torch.rand(20) * 2 + 99
        )
        
        # Warm up
        for _ in range(5):
            _ = advanced_embedder(context)
            
        # Measure latency
        n_iterations = 100
        start_time = time.time()
        
        for _ in range(n_iterations):
            _ = advanced_embedder(context)
            
        avg_latency_ms = (time.time() - start_time) / n_iterations * 1000
        
        # Check performance meets target
        assert avg_latency_ms < 2.0  # Target: <2ms latency
        
    def test_error_handling(self, advanced_embedder):
        """Test error handling in embedder."""
        # Test with invalid context
        with pytest.raises(Exception):
            _ = advanced_embedder({})
            
        # Test with mismatched dimensions
        context = LVNContext(
            prices=torch.rand(5),
            strengths=torch.rand(3),  # Mismatched size
            distances=torch.rand(5),
            volumes=torch.rand(5),
            current_price=100.0,
            price_history=torch.rand(20)
        )
        
        # Should handle gracefully or raise informative error
        try:
            _ = advanced_embedder(context)
        except Exception as e:
            assert str(e)  # Should have error message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])