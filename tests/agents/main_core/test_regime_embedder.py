"""
File: tests/agents/main_core/test_regime_embedder.py (NEW FILE)
Comprehensive test suite for regime embedder
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import time

from src.agents.main_core.regime_embedder import (
    RegimeEmbedder, TemporalRegimeBuffer, 
    RegimeAttentionAnalyzer, RegimeTransitionDetector
)
from src.agents.main_core.regime_uncertainty import RegimeUncertaintyCalibrator
from src.agents.main_core.regime_patterns import RegimePatternBank

class TestRegimeEmbedder:
    """Test suite for advanced regime embedder."""
    
    @pytest.fixture
    def config(self):
        return {
            'regime_dim': 8,
            'output_dim': 16,
            'hidden_dim': 32,
            'buffer_size': 20,
            'n_heads': 4,
            'dropout': 0.1
        }
        
    @pytest.fixture
    def embedder(self, config):
        return RegimeEmbedder(config)
        
    def test_initialization(self, embedder, config):
        """Test proper initialization."""
        assert embedder.regime_dim == config['regime_dim']
        assert embedder.output_dim == config['output_dim']
        assert embedder.hidden_dim == config['hidden_dim']
        assert isinstance(embedder.regime_buffer, TemporalRegimeBuffer)
        
    def test_forward_pass_shapes(self, embedder):
        """Test forward pass output shapes."""
        batch_size = 4
        regime_vector = torch.randn(batch_size, 8)
        
        result = embedder(regime_vector)
        
        assert hasattr(result, 'mean')
        assert hasattr(result, 'std')
        assert result.mean.shape == (batch_size, 16)
        assert result.std.shape == (batch_size, 16)
        
    def test_temporal_buffer(self):
        """Test temporal buffer functionality."""
        buffer = TemporalRegimeBuffer(buffer_size=5, regime_dim=8)
        
        # Add regimes
        for i in range(7):
            regime = torch.randn(8)
            buffer.add(regime)
            
        # Check buffer size limit
        assert len(buffer.buffer) == 5
        
        # Get sequence
        sequence = buffer.get_sequence(torch.device('cpu'))
        assert sequence.shape == (1, 5, 8)
        
    def test_attention_analyzer(self):
        """Test attention mechanism."""
        analyzer = RegimeAttentionAnalyzer(regime_dim=8, n_heads=4)
        regime = torch.randn(2, 8)
        
        features, attention_info = analyzer(regime)
        
        assert features.shape == (2, 16)
        assert 'attention_weights' in attention_info
        assert 'component_importance' in attention_info
        
    def test_transition_detection(self):
        """Test regime transition detection."""
        detector = RegimeTransitionDetector(regime_dim=8, hidden_dim=32)
        
        current = torch.randn(2, 8)
        history = torch.randn(2, 5, 8)
        
        features, metrics = detector(current, history)
        
        assert features.shape == (2, 16)
        assert 'stability' in metrics
        assert 'volatility' in metrics
        assert 'magnitude' in metrics
        
    def test_uncertainty_calibration(self):
        """Test uncertainty calibration."""
        config = {'calibration_window': 100}
        calibrator = RegimeUncertaintyCalibrator(config)
        
        # Add samples
        for _ in range(50):
            pred = torch.randn(16)
            unc = torch.rand(16) * 0.5
            outcome = np.random.random()
            calibrator.add_sample(pred, unc, outcome)
            
        # Test calibration
        raw_std = torch.rand(4, 16) * 0.5
        calibrated = calibrator.calibrate_uncertainty(raw_std)
        
        assert calibrated.shape == raw_std.shape
        assert (calibrated > 0).all()
        
    def test_pattern_bank(self):
        """Test pattern bank functionality."""
        config = {'regime_dim': 8, 'n_patterns': 16}
        pattern_bank = RegimePatternBank(config)
        
        regime = torch.randn(2, 8)
        features, info = pattern_bank(regime)
        
        assert features.shape == (2, 16)
        assert 'best_match_idx' in info
        assert 'best_match_similarity' in info
        
    def test_regime_stability(self, embedder):
        """Test stability with repeated regimes."""
        regime = torch.randn(1, 8)
        
        # Process same regime multiple times
        embeddings = []
        for _ in range(5):
            result = embedder(regime)
            embeddings.append(result.mean)
            
        # Check consistency
        embeddings = torch.stack(embeddings)
        std = embeddings.std(dim=0)
        assert (std < 0.1).all()  # Should be relatively stable
        
    def test_regime_transitions(self, embedder):
        """Test handling of regime transitions."""
        # Simulate regime transition
        regime1 = torch.randn(1, 8)
        regime2 = regime1 + torch.randn(1, 8) * 2  # Large change
        
        # Process sequence
        result1 = embedder(regime1)
        result2 = embedder(regime2)
        
        # Transition should be detected
        assert result2.transition_score > 0.5
        
    def test_performance_requirements(self, embedder):
        """Test performance meets requirements."""
        regime = torch.randn(1, 8)
        
        # Warm up
        _ = embedder(regime)
        
        # Time inference
        times = []
        for _ in range(100):
            start = time.time()
            _ = embedder(regime)
            times.append((time.time() - start) * 1000)
            
        avg_time = np.mean(times)
        assert avg_time < 2.0  # Must be under 2ms
        
    def test_gpu_compatibility(self, embedder):
        """Test GPU compatibility."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
            
        embedder = embedder.cuda()
        regime = torch.randn(2, 8).cuda()
        
        result = embedder(regime)
        assert result.mean.is_cuda
        assert result.std.is_cuda
        
    def test_gradient_flow(self, embedder):
        """Test gradient flow through embedder."""
        regime = torch.randn(2, 8, requires_grad=True)
        
        result = embedder(regime)
        loss = result.mean.sum()
        loss.backward()
        
        assert regime.grad is not None
        assert not torch.isnan(regime.grad).any()
        
    def test_interpretability(self, embedder):
        """Test interpretability features."""
        regime = torch.randn(1, 8)
        
        # Process regime
        _ = embedder(regime)
        
        # Get interpretation
        interpretation = embedder.embedder.get_regime_interpretation()
        
        assert 'dominant_dimensions' in interpretation
        assert 'dimension_importance' in interpretation
        assert 'regime_stability' in interpretation