"""
Comprehensive test suite for MC Dropout consensus mechanism.

Tests all components including core functionality, optimizations,
calibration, and monitoring.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import time
from typing import Dict, Any

from src.agents.main_core.mc_dropout import (
    MCDropoutConsensus,
    UncertaintyMetrics,
    ConsensusResult
)
from src.agents.main_core.mc_dropout_optimization import (
    MCDropoutBatcher,
    StreamingStatistics,
    AdaptiveMCDropout,
    GPUMemoryOptimizer,
    benchmark_mc_dropout_performance
)
from src.agents.main_core.mc_calibration import (
    MCDropoutCalibrator,
    CalibrationDiagnostics
)
from src.agents.main_core.mc_monitoring import (
    MCDropoutMonitor,
    AlertSystem,
    PerformanceProfiler
)


class TestMCDropoutConsensus:
    """Test suite for MC Dropout consensus mechanism."""
    
    @pytest.fixture
    def config(self):
        return {
            'n_samples': 50,
            'confidence_threshold': 0.65,
            'temperature': 1.0,
            'gpu_optimization': False,  # Disable for testing
            'calibration': 'temperature'
        }
        
    @pytest.fixture
    def mock_model(self):
        """Create mock model with dropout."""
        model = Mock()
        model.train = Mock()
        model.eval = Mock()
        
        # Mock forward pass
        def forward(x):
            # Add randomness to simulate dropout
            base_logits = torch.tensor([[0.6, 0.4]])
            noise = torch.randn_like(base_logits) * 0.1
            return {'action_logits': base_logits + noise}
            
        model.forward = Mock(side_effect=forward)
        model.to = Mock(return_value=model)
        return model
        
    def test_basic_consensus(self, config, mock_model):
        """Test basic consensus functionality."""
        consensus = MCDropoutConsensus(config)
        
        input_state = torch.randn(1, 512)
        result = consensus.evaluate(mock_model, input_state)
        
        # Check result structure
        assert isinstance(result, ConsensusResult)
        assert isinstance(result.should_proceed, bool)
        assert isinstance(result.uncertainty_metrics, UncertaintyMetrics)
        assert result.action_probabilities.shape == (1, 2)
        
        # Check uncertainty decomposition
        metrics = result.uncertainty_metrics
        assert metrics.total_uncertainty >= 0
        assert metrics.epistemic_uncertainty >= 0
        assert metrics.aleatoric_uncertainty >= 0
        assert abs(metrics.total_uncertainty - 
                  (metrics.aleatoric_uncertainty + metrics.epistemic_uncertainty)) < 0.01
                  
    def test_convergence_detection(self, config, mock_model):
        """Test convergence detection."""
        consensus = MCDropoutConsensus(config)
        
        # Create stable model (low variance)
        stable_model = Mock()
        stable_model.train = Mock()
        stable_model.eval = Mock()
        stable_model.to = Mock(return_value=stable_model)
        
        def stable_forward(x):
            return {'action_logits': torch.tensor([[0.8, 0.2]])}
            
        stable_model.forward = Mock(side_effect=stable_forward)
        
        input_state = torch.randn(1, 512)
        result = consensus.evaluate(stable_model, input_state)
        
        # Should converge quickly with stable outputs
        assert result.convergence_info['converged'] == True
        assert result.convergence_info['stability'] > 0.9
        
    def test_outlier_detection(self, config, mock_model):
        """Test outlier sample detection."""
        consensus = MCDropoutConsensus(config)
        
        # Create model with occasional outliers
        outlier_count = 0
        
        def outlier_forward(x):
            nonlocal outlier_count
            outlier_count += 1
            
            if outlier_count % 10 == 0:
                # Outlier
                return {'action_logits': torch.tensor([[0.1, 0.9]])}
            else:
                # Normal
                return {'action_logits': torch.tensor([[0.7, 0.3]])}
                
        mock_model.forward = Mock(side_effect=outlier_forward)
        
        input_state = torch.randn(1, 512)
        result = consensus.evaluate(mock_model, input_state)
        
        # Should detect outliers
        assert len(result.outlier_samples) > 0
        assert len(result.outlier_samples) < config['n_samples'] * 0.2
        
    def test_adaptive_threshold(self, config):
        """Test adaptive threshold calculation."""
        consensus = MCDropoutConsensus(config)
        
        # Test different contexts
        market_contexts = [
            {'regime': 'trending'},
            {'regime': 'volatile'},
            {'regime': 'ranging'},
            {'regime': 'transitioning'}
        ]
        
        risk_contexts = [
            {'risk_level': 'low'},
            {'risk_level': 'medium'},
            {'risk_level': 'high'},
            {'risk_level': 'extreme'}
        ]
        
        base_threshold = config['confidence_threshold']
        
        for market_ctx in market_contexts:
            for risk_ctx in risk_contexts:
                threshold = consensus._calculate_adaptive_threshold(
                    base_threshold,
                    market_ctx,
                    risk_ctx,
                    UncertaintyMetrics(
                        total_uncertainty=0.3,
                        aleatoric_uncertainty=0.2,
                        epistemic_uncertainty=0.1,
                        predictive_entropy=0.3,
                        mutual_information=0.1,
                        expected_entropy=0.2,
                        variance_of_expectations=0.05,
                        confidence_score=0.7,
                        calibrated_confidence=0.7,
                        decision_boundary_distance=0.1
                    )
                )
                
                # Check threshold is adjusted appropriately
                if market_ctx['regime'] == 'volatile' or risk_ctx['risk_level'] == 'extreme':
                    assert threshold > base_threshold
                elif market_ctx['regime'] == 'trending' and risk_ctx['risk_level'] == 'low':
                    assert threshold <= base_threshold
                    
                # Check bounds
                assert 0.5 <= threshold <= 0.95
                
    def test_confidence_intervals(self, config, mock_model):
        """Test confidence interval calculation."""
        consensus = MCDropoutConsensus(config)
        
        input_state = torch.randn(1, 512)
        result = consensus.evaluate(mock_model, input_state)
        
        # Check confidence intervals
        assert 'ci_68' in result.confidence_intervals
        assert 'ci_95' in result.confidence_intervals
        assert 'ci_99' in result.confidence_intervals
        
        # Check interval ordering
        ci_68 = result.confidence_intervals['ci_68']
        ci_95 = result.confidence_intervals['ci_95']
        ci_99 = result.confidence_intervals['ci_99']
        
        assert ci_68[0] >= ci_95[0] >= ci_99[0]
        assert ci_68[1] <= ci_95[1] <= ci_99[1]


class TestOptimization:
    """Test MC Dropout optimization components."""
    
    def test_parallel_sampling_optimization(self):
        """Test parallel MC sampling optimization."""
        device = torch.device('cpu')
        batcher = MCDropoutBatcher(device, max_batch_size=25)
        
        mock_model = Mock()
        
        # Track number of forward calls
        forward_calls = 0
        
        def batched_forward(x):
            nonlocal forward_calls
            forward_calls += 1
            batch_size = x.size(0)
            return torch.randn(batch_size, 2)
            
        mock_model.forward = Mock(side_effect=batched_forward)
        
        input_state = torch.randn(1, 512)
        samples = batcher.batch_forward(mock_model, input_state, n_samples=50)
        
        # Should use 2 batches for 50 samples with max_batch_size=25
        assert forward_calls == 2
        assert samples.shape == (50, 1, 2)
        
    def test_streaming_statistics(self):
        """Test streaming statistics calculation."""
        stats = StreamingStatistics()
        
        # Generate samples in batches
        true_mean = torch.tensor([0.7, 0.3])
        true_std = torch.tensor([0.1, 0.1])
        
        for _ in range(10):
            batch = torch.normal(
                true_mean.unsqueeze(0).repeat(10, 1),
                true_std.unsqueeze(0).repeat(10, 1)
            )
            stats.update(batch)
            
        results = stats.get_statistics()
        
        # Check statistics are close to true values
        assert torch.allclose(results['mean'], true_mean, atol=0.05)
        assert torch.allclose(results['std'], true_std, atol=0.05)
        assert results['n_samples'] == 100
        
    def test_adaptive_mc_dropout(self):
        """Test adaptive MC dropout sampling."""
        adaptive = AdaptiveMCDropout(
            min_samples=10,
            max_samples=50,
            uncertainty_threshold=0.1
        )
        
        # Create mock model
        mock_model = Mock()
        mock_model.train = Mock()
        mock_model.eval = Mock()
        
        # Test with low uncertainty model
        def low_uncertainty_forward(x):
            return {'action_logits': torch.tensor([[0.9, 0.1]])}
            
        mock_model.forward = Mock(side_effect=low_uncertainty_forward)
        
        input_state = torch.randn(1, 512)
        result = adaptive.adaptive_sampling(mock_model, input_state)
        
        # Should use minimum samples for low uncertainty
        assert result['n_samples'] == adaptive.min_samples
        assert result['confidence_level'] == 'high'
        assert result['computation_saved'] > 0.5
        
    def test_gpu_memory_optimizer(self):
        """Test GPU memory optimization."""
        device = torch.device('cpu')  # Use CPU for testing
        optimizer = GPUMemoryOptimizer(device)
        
        # Create simple model
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        )
        
        # Test optimal batch size calculation
        batch_size = optimizer.get_optimal_batch_size(
            model,
            input_shape=(1, 100),
            max_samples=50
        )
        
        # On CPU, should return max_samples
        assert batch_size == 50
        
    def test_benchmark_performance(self):
        """Test performance benchmarking."""
        # Create simple model
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(50, 2)
        )
        
        # Run benchmark
        metrics = benchmark_mc_dropout_performance(
            model,
            input_shape=(1, 100),
            n_samples=20,
            device=torch.device('cpu')
        )
        
        # Check metrics
        assert 'sequential_time' in metrics
        assert 'batch_time' in metrics
        assert 'speedup' in metrics
        assert metrics['speedup'] > 0


class TestCalibration:
    """Test calibration system."""
    
    @pytest.fixture
    def calibrator_config(self):
        return {
            'method': 'ensemble',
            'update_frequency': 10
        }
        
    def test_calibration_integration(self, calibrator_config):
        """Test calibration integration."""
        calibrator = MCDropoutCalibrator(calibrator_config)
        
        # Test calibration
        raw_probs = torch.tensor([[0.8, 0.2]])
        uncertainty_metrics = {
            'epistemic_uncertainty': 0.1,
            'aleatoric_uncertainty': 0.2
        }
        
        calibrated = calibrator.calibrate(raw_probs, uncertainty_metrics)
        
        # Check output shape
        assert calibrated.shape == raw_probs.shape
        
        # Check probabilities sum to 1
        assert torch.allclose(calibrated.sum(dim=-1), torch.ones(1))
        
    def test_calibration_update(self, calibrator_config):
        """Test calibration model updates."""
        calibrator = MCDropoutCalibrator(calibrator_config)
        
        # Generate fake data
        predictions = [
            {'probability': 0.8} for _ in range(100)
        ]
        outcomes = [True] * 80 + [False] * 20  # 80% accuracy
        
        # Update calibration
        calibrator.update(predictions, outcomes)
        
        # Check data was stored
        assert len(calibrator.calibration_data) == 100
        assert len(calibrator.outcome_data) == 100
        
    def test_calibration_diagnostics(self):
        """Test calibration diagnostic metrics."""
        # Generate test data
        predictions = torch.rand(100)
        targets = (predictions > 0.5).float()
        
        # Add some calibration error
        predictions = predictions * 0.8 + 0.1
        
        # Calculate ECE
        ece = CalibrationDiagnostics.expected_calibration_error(
            predictions, targets
        )
        
        # ECE should be positive
        assert ece > 0
        
        # Calculate reliability diagram
        diagram = CalibrationDiagnostics.reliability_diagram(
            predictions, targets
        )
        
        assert 'confidences' in diagram
        assert 'accuracies' in diagram
        assert 'counts' in diagram


class TestMonitoring:
    """Test monitoring system."""
    
    @pytest.fixture
    def monitor_config(self):
        return {
            'window_size': 100,
            'enable_dashboard': False
        }
        
    @pytest.fixture
    def mock_consensus_result(self):
        """Create mock consensus result."""
        return ConsensusResult(
            should_proceed=True,
            predicted_action=0,
            action_probabilities=torch.tensor([[0.7, 0.3]]),
            uncertainty_metrics=UncertaintyMetrics(
                total_uncertainty=0.3,
                aleatoric_uncertainty=0.2,
                epistemic_uncertainty=0.1,
                predictive_entropy=0.3,
                mutual_information=0.1,
                expected_entropy=0.2,
                variance_of_expectations=0.05,
                confidence_score=0.7,
                calibrated_confidence=0.75,
                decision_boundary_distance=0.25
            ),
            sample_statistics={'mean_probs': torch.tensor([[0.7, 0.3]])},
            confidence_intervals={'ci_95': (0.6, 0.8)},
            outlier_samples=[],
            convergence_info={'converged': True, 'r_hat': 1.05}
        )
        
    def test_monitoring_record(self, monitor_config, mock_consensus_result):
        """Test recording decisions."""
        monitor = MCDropoutMonitor(monitor_config)
        
        # Record decision
        monitor.record_decision(mock_consensus_result)
        
        # Check recording
        assert len(monitor.decision_history) == 1
        assert len(monitor.uncertainty_history) == 1
        
        # Get stats
        stats = monitor.get_current_stats()
        assert stats['total_decisions'] == 1
        assert stats['decision_rate'] == 1.0
        
    def test_alert_system(self, mock_consensus_result):
        """Test alert system."""
        alert_config = {
            'low_confidence_threshold': 0.8,
            'high_uncertainty_threshold': 0.2
        }
        
        alert_system = AlertSystem(alert_config)
        
        # Check for alerts
        alerts = alert_system.check_alerts(mock_consensus_result)
        
        # Should have alerts for low confidence and high uncertainty
        assert len(alerts) >= 2
        alert_types = [a['type'] for a in alerts]
        assert 'low_confidence' in alert_types
        assert 'high_uncertainty' in alert_types
        
    def test_performance_profiler(self):
        """Test performance profiling."""
        profiler = PerformanceProfiler()
        
        # Profile a simple function
        def test_func():
            time.sleep(0.01)
            return torch.randn(100, 100)
            
        result, metrics = profiler.profile_evaluation(test_func)
        
        # Check metrics
        assert 'execution_time_ms' in metrics
        assert 'memory_current_mb' in metrics
        assert 'memory_peak_mb' in metrics
        
        # Execution time should be at least 10ms
        assert metrics['execution_time_ms'] >= 10
        
        # Get performance stats
        stats = profiler.get_performance_stats()
        assert 'avg_time_ms' in stats


class TestIntegration:
    """Integration tests for complete MC Dropout system."""
    
    def test_full_pipeline(self):
        """Test complete MC Dropout pipeline."""
        # Configuration
        config = {
            'n_samples': 20,
            'confidence_threshold': 0.6,
            'gpu_optimization': False,
            'calibration': 'temperature'
        }
        
        # Create components
        consensus = MCDropoutConsensus(config)
        calibrator = MCDropoutCalibrator({'method': 'temperature'})
        monitor = MCDropoutMonitor({'window_size': 100, 'enable_dashboard': False})
        
        # Create model
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(50, 2)
        )
        
        # Run evaluation
        input_state = torch.randn(1, 100)
        result = consensus.evaluate(model, input_state)
        
        # Record result
        monitor.record_decision(result)
        
        # Check complete pipeline
        assert result.should_proceed in [True, False]
        assert len(monitor.decision_history) == 1
        
    def test_error_handling(self):
        """Test error handling in MC Dropout."""
        config = {'n_samples': 50}
        consensus = MCDropoutConsensus(config)
        
        # Test with invalid model
        invalid_model = Mock()
        invalid_model.train = Mock()
        invalid_model.eval = Mock()
        invalid_model.to = Mock(return_value=invalid_model)
        invalid_model.forward = Mock(side_effect=RuntimeError("Test error"))
        
        input_state = torch.randn(1, 100)
        
        # Should handle error gracefully
        with pytest.raises(RuntimeError):
            consensus.evaluate(invalid_model, input_state)
            
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        config = {'n_samples': 50}
        consensus = MCDropoutConsensus(config)
        
        # Test with deterministic model (no dropout)
        deterministic_model = nn.Linear(100, 2)
        input_state = torch.randn(1, 100)
        
        result = consensus.evaluate(deterministic_model, input_state)
        
        # Should still work but with low epistemic uncertainty
        assert result.uncertainty_metrics.epistemic_uncertainty < 0.01
        
        # Test with extreme probabilities
        def extreme_forward(x):
            return {'action_logits': torch.tensor([[100.0, -100.0]])}
            
        extreme_model = Mock()
        extreme_model.train = Mock()
        extreme_model.eval = Mock()
        extreme_model.to = Mock(return_value=extreme_model)
        extreme_model.forward = Mock(side_effect=extreme_forward)
        
        result = consensus.evaluate(extreme_model, input_state)
        
        # Should handle extreme values gracefully
        assert torch.allclose(result.action_probabilities[0, 0], torch.tensor(1.0), atol=1e-6)
        assert result.uncertainty_metrics.total_uncertainty < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])