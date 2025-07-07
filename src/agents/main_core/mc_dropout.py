"""
MC Dropout Consensus Mechanism for AlgoSpace.

This module implements a state-of-the-art Monte Carlo Dropout consensus system
that transforms single-point neural network predictions into probability distributions.
By running 50 forward passes with dropout enabled, it creates a "superposition" of 
possible decisions, ensuring only high-confidence trades proceed through the system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio
import time

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyMetrics:
    """Comprehensive uncertainty metrics."""
    total_uncertainty: float
    aleatoric_uncertainty: float
    epistemic_uncertainty: float
    predictive_entropy: float
    mutual_information: float
    expected_entropy: float
    variance_of_expectations: float
    confidence_score: float
    calibrated_confidence: float
    decision_boundary_distance: float


@dataclass
class ConsensusResult:
    """Complete consensus evaluation result."""
    should_proceed: bool
    predicted_action: int
    action_probabilities: torch.Tensor
    uncertainty_metrics: UncertaintyMetrics
    sample_statistics: Dict[str, torch.Tensor]
    confidence_intervals: Dict[str, Tuple[float, float]]
    outlier_samples: List[int]
    convergence_info: Dict[str, float]


class MCDropoutConsensus:
    """
    State-of-the-art Monte Carlo Dropout consensus mechanism.
    
    Implements advanced uncertainty quantification with GPU optimization,
    calibration, and adaptive thresholding for high-stakes trading decisions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.n_samples = config.get('n_samples', 50)
        self.base_threshold = config.get('confidence_threshold', 0.65)
        self.temperature = config.get('temperature', 1.0)
        self.use_gpu_optimization = config.get('gpu_optimization', True)
        self.calibration_method = config.get('calibration', 'temperature')
        
        # Adaptive threshold parameters
        self.adaptive_thresholds = {
            'regime_adjustments': {
                'trending': 0.0,      # No adjustment
                'volatile': 0.05,     # Higher threshold
                'ranging': 0.03,      # Slightly higher
                'transitioning': 0.08 # Much higher
            },
            'risk_adjustments': {
                'low': -0.02,         # Lower threshold okay
                'medium': 0.0,        # No adjustment
                'high': 0.05,         # Higher threshold
                'extreme': 0.10       # Much higher
            }
        }
        
        # Calibration models
        self.calibrators = {
            'temperature': TemperatureScaling(),
            'platt': PlattScaling(),
            'isotonic': IsotonicCalibration()
        }
        
        # Performance tracking
        self.decision_history = []
        self.calibration_history = []
        
        # GPU optimization
        if self.use_gpu_optimization and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self._init_cuda_kernels()
        else:
            self.device = torch.device('cpu')
            
    def evaluate(
        self,
        model: nn.Module,
        input_state: torch.Tensor,
        market_context: Optional[Dict[str, Any]] = None,
        risk_context: Optional[Dict[str, Any]] = None
    ) -> ConsensusResult:
        """
        Perform comprehensive MC Dropout evaluation.
        
        Args:
            model: Neural network with dropout layers
            input_state: Input state tensor [batch, features]
            market_context: Current market regime and conditions
            risk_context: Current risk levels and constraints
            
        Returns:
            ConsensusResult with all metrics and decision
        """
        # Move to device
        input_state = input_state.to(self.device)
        
        # Enable dropout
        model.train()
        
        # Parallel MC sampling
        samples = self._parallel_mc_sampling(model, input_state)
        
        # Calculate uncertainty metrics
        uncertainty_metrics = self._calculate_uncertainty_metrics(samples)
        
        # Analyze sample statistics
        sample_stats = self._analyze_sample_statistics(samples)
        
        # Detect outliers
        outliers = self._detect_outlier_samples(samples, sample_stats)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(samples)
        
        # Check convergence
        convergence_info = self._check_convergence(samples)
        
        # Apply calibration
        calibrated_probs = self._calibrate_probabilities(
            sample_stats['mean_probs'],
            uncertainty_metrics
        )
        
        # Determine threshold with context
        threshold = self._calculate_adaptive_threshold(
            base_threshold=self.base_threshold,
            market_context=market_context,
            risk_context=risk_context,
            uncertainty_metrics=uncertainty_metrics
        )
        
        # Make decision
        predicted_action = calibrated_probs.argmax(dim=-1)
        confidence = calibrated_probs.max(dim=-1)[0]
        should_proceed = (predicted_action == 0) & (confidence >= threshold)
        
        # Calculate decision boundary distance
        boundary_distance = self._calculate_decision_boundary_distance(
            calibrated_probs,
            threshold
        )
        
        # Update calibrated confidence in metrics
        uncertainty_metrics.calibrated_confidence = confidence.item()
        uncertainty_metrics.decision_boundary_distance = boundary_distance
        
        # Track for learning
        self._track_decision(
            input_state,
            should_proceed,
            uncertainty_metrics,
            market_context
        )
        
        # Restore model state
        model.eval()
        
        return ConsensusResult(
            should_proceed=should_proceed.item(),
            predicted_action=predicted_action.item(),
            action_probabilities=calibrated_probs,
            uncertainty_metrics=uncertainty_metrics,
            sample_statistics=sample_stats,
            confidence_intervals=confidence_intervals,
            outlier_samples=outliers,
            convergence_info=convergence_info
        )
        
    def _parallel_mc_sampling(
        self, 
        model: nn.Module, 
        input_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Optimized parallel MC sampling using batch processing.
        
        Instead of sequential passes, process multiple samples in parallel.
        """
        if self.use_gpu_optimization and input_state.size(0) == 1:
            # Replicate input for batch processing
            batch_input = input_state.repeat(self.n_samples, 1)
            
            # Single batched forward pass
            with torch.no_grad():
                outputs = model(batch_input)
                
            # Extract action probabilities
            if isinstance(outputs, dict):
                logits = outputs.get('action_logits', outputs.get('logits'))
            else:
                logits = outputs
                
            probs = F.softmax(logits / self.temperature, dim=-1)
            
            # Reshape to [n_samples, batch=1, n_actions]
            return probs.unsqueeze(1)
            
        else:
            # Standard sequential sampling
            samples = []
            
            with torch.no_grad():
                for _ in range(self.n_samples):
                    outputs = model(input_state)
                    
                    if isinstance(outputs, dict):
                        logits = outputs.get('action_logits', outputs.get('logits'))
                    else:
                        logits = outputs
                        
                    probs = F.softmax(logits / self.temperature, dim=-1)
                    samples.append(probs)
                    
            return torch.stack(samples)
            
    def _calculate_uncertainty_metrics(
        self, 
        samples: torch.Tensor
    ) -> UncertaintyMetrics:
        """
        Calculate comprehensive uncertainty metrics.
        
        Decomposes uncertainty into aleatoric and epistemic components.
        """
        # samples shape: [n_samples, batch_size, n_actions]
        
        # Mean prediction
        mean_probs = samples.mean(dim=0)
        
        # Total uncertainty (predictive entropy)
        predictive_entropy = -torch.sum(
            mean_probs * torch.log(mean_probs + 1e-8), 
            dim=-1
        ).mean().item()
        
        # Expected entropy (aleatoric uncertainty)
        sample_entropies = -torch.sum(
            samples * torch.log(samples + 1e-8), 
            dim=-1
        )
        expected_entropy = sample_entropies.mean().item()
        
        # Mutual information (epistemic uncertainty)
        mutual_information = predictive_entropy - expected_entropy
        
        # Variance of expectations
        variance_of_exp = samples.var(dim=0).mean().item()
        
        # Confidence score (inverse uncertainty)
        confidence_score = 1.0 / (1.0 + predictive_entropy)
        
        return UncertaintyMetrics(
            total_uncertainty=predictive_entropy,
            aleatoric_uncertainty=expected_entropy,
            epistemic_uncertainty=mutual_information,
            predictive_entropy=predictive_entropy,
            mutual_information=mutual_information,
            expected_entropy=expected_entropy,
            variance_of_expectations=variance_of_exp,
            confidence_score=confidence_score,
            calibrated_confidence=confidence_score,  # Will be updated
            decision_boundary_distance=0.0  # Will be calculated
        )
        
    def _analyze_sample_statistics(
        self, 
        samples: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Analyze statistical properties of MC samples."""
        return {
            'mean_probs': samples.mean(dim=0),
            'std_probs': samples.std(dim=0),
            'median_probs': samples.median(dim=0)[0],
            'min_probs': samples.min(dim=0)[0],
            'max_probs': samples.max(dim=0)[0],
            'quantile_25': samples.quantile(0.25, dim=0),
            'quantile_75': samples.quantile(0.75, dim=0),
            'mode_action': samples.argmax(dim=-1).mode(dim=0)[0]
        }
        
    def _detect_outlier_samples(
        self, 
        samples: torch.Tensor,
        stats: Dict[str, torch.Tensor]
    ) -> List[int]:
        """
        Detect outlier samples using robust statistics.
        
        Outliers might indicate numerical issues or edge cases.
        """
        outliers = []
        
        # Calculate robust statistics
        median = stats['median_probs']
        mad = torch.median(
            torch.abs(samples - median.unsqueeze(0)), 
            dim=0
        )[0]
        
        # Modified Z-score
        threshold = 3.5
        for i in range(samples.size(0)):
            z_score = torch.abs(samples[i] - median) / (mad + 1e-8)
            if (z_score > threshold).any():
                outliers.append(i)
                
        return outliers
        
    def _calculate_confidence_intervals(
        self, 
        samples: torch.Tensor,
        confidence_levels: List[float] = [0.68, 0.95, 0.99]
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for predictions."""
        intervals = {}
        
        for level in confidence_levels:
            alpha = (1 - level) / 2
            lower = samples.quantile(alpha, dim=0)
            upper = samples.quantile(1 - alpha, dim=0)
            
            # Store for action 0 (qualify/execute)
            intervals[f'ci_{int(level*100)}'] = (
                lower[0, 0].item(),
                upper[0, 0].item()
            )
            
        return intervals
        
    def _check_convergence(
        self, 
        samples: torch.Tensor
    ) -> Dict[str, float]:
        """
        Check if MC sampling has converged.
        
        Uses running statistics to detect convergence.
        """
        # Calculate running mean
        running_means = []
        for i in range(10, self.n_samples + 1, 5):
            subset_mean = samples[:i].mean(dim=0)
            running_means.append(subset_mean)
            
        if len(running_means) < 2:
            return {'converged': False, 'stability': 0.0}
            
        # Check stability of running mean
        running_means = torch.stack(running_means)
        stability = 1.0 - running_means.std(dim=0).mean().item()
        
        # Gelman-Rubin statistic approximation
        n_chains = 5
        chain_size = self.n_samples // n_chains
        
        chains = []
        for i in range(n_chains):
            start_idx = i * chain_size
            end_idx = start_idx + chain_size
            chain_mean = samples[start_idx:end_idx].mean(dim=0)
            chains.append(chain_mean)
            
        chains = torch.stack(chains)
        
        # Between-chain variance
        B = chain_size * chains.var(dim=0).mean()
        
        # Within-chain variance
        W = 0
        for i in range(n_chains):
            start_idx = i * chain_size
            end_idx = start_idx + chain_size
            W += samples[start_idx:end_idx].var(dim=0).mean()
        W /= n_chains
        
        # Gelman-Rubin statistic
        var_est = ((chain_size - 1) * W / chain_size) + (B / chain_size)
        R_hat = torch.sqrt(var_est / W).item() if W > 0 else float('inf')
        
        converged = R_hat < 1.1 and stability > 0.95
        
        return {
            'converged': converged,
            'stability': stability,
            'r_hat': R_hat,
            'effective_samples': self.n_samples / max(R_hat, 1.0)
        }
        
    def _calibrate_probabilities(
        self,
        raw_probs: torch.Tensor,
        uncertainty_metrics: UncertaintyMetrics
    ) -> torch.Tensor:
        """Apply probability calibration."""
        calibrator = self.calibrators.get(
            self.calibration_method,
            self.calibrators['temperature']
        )
        
        # Apply calibration
        calibrated = calibrator.calibrate(
            raw_probs,
            uncertainty_metrics.epistemic_uncertainty
        )
        
        # Ensure valid probabilities
        calibrated = torch.clamp(calibrated, min=1e-8, max=1-1e-8)
        calibrated = calibrated / calibrated.sum(dim=-1, keepdim=True)
        
        return calibrated
        
    def _calculate_adaptive_threshold(
        self,
        base_threshold: float,
        market_context: Optional[Dict[str, Any]],
        risk_context: Optional[Dict[str, Any]],
        uncertainty_metrics: UncertaintyMetrics
    ) -> float:
        """
        Calculate context-aware confidence threshold.
        
        Adjusts threshold based on market regime, risk level, and uncertainty.
        """
        threshold = base_threshold
        
        # Market regime adjustment
        if market_context:
            regime = market_context.get('regime', 'unknown')
            regime_adj = self.adaptive_thresholds['regime_adjustments'].get(
                regime, 0.0
            )
            threshold += regime_adj
            
        # Risk level adjustment
        if risk_context:
            risk_level = risk_context.get('risk_level', 'medium')
            risk_adj = self.adaptive_thresholds['risk_adjustments'].get(
                risk_level, 0.0
            )
            threshold += risk_adj
            
        # Uncertainty adjustment
        # Higher epistemic uncertainty → higher threshold
        if uncertainty_metrics.epistemic_uncertainty > 0.3:
            uncertainty_adj = min(
                0.1,
                (uncertainty_metrics.epistemic_uncertainty - 0.3) * 0.2
            )
            threshold += uncertainty_adj
            
        # Clamp to valid range
        threshold = np.clip(threshold, 0.5, 0.95)
        
        logger.debug(f"Adaptive threshold: {threshold:.3f} (base: {base_threshold:.3f})")
        
        return threshold
        
    def _calculate_decision_boundary_distance(
        self,
        probs: torch.Tensor,
        threshold: float
    ) -> float:
        """
        Calculate distance from decision boundary.
        
        Positive = confident proceed, Negative = confident reject
        """
        # Get probability of primary action (qualify/execute)
        primary_prob = probs[0, 0].item()
        
        # Distance from threshold
        distance = primary_prob - threshold
        
        # Normalize by uncertainty region
        uncertainty_band = 0.1  # ±10% is uncertain region
        normalized_distance = distance / uncertainty_band
        
        return normalized_distance
        
    def _track_decision(
        self,
        input_state: torch.Tensor,
        decision: bool,
        metrics: UncertaintyMetrics,
        context: Optional[Dict[str, Any]]
    ):
        """Track decisions for calibration learning."""
        self.decision_history.append({
            'timestamp': torch.tensor(time.time()),
            'decision': decision,
            'confidence': metrics.confidence_score,
            'uncertainty': metrics.total_uncertainty,
            'context': context
        })
        
        # Keep limited history
        if len(self.decision_history) > 10000:
            self.decision_history = self.decision_history[-5000:]
            
    def update_calibration(
        self,
        outcomes: List[Dict[str, Any]]
    ):
        """
        Update calibration models based on observed outcomes.
        
        Should be called periodically with trading results.
        """
        if len(outcomes) < 100:
            return
            
        # Prepare calibration data
        predictions = []
        actuals = []
        
        for outcome in outcomes:
            pred_prob = outcome['predicted_probability']
            actual = outcome['was_profitable']
            predictions.append(pred_prob)
            actuals.append(actual)
            
        predictions = torch.tensor(predictions)
        actuals = torch.tensor(actuals, dtype=torch.float32)
        
        # Update each calibrator
        for name, calibrator in self.calibrators.items():
            calibrator.fit(predictions, actuals)
            
        logger.info("Calibration models updated")
        
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about consensus mechanism."""
        if not self.decision_history:
            return {}
            
        recent = self.decision_history[-1000:]
        
        confidences = [d['confidence'] for d in recent]
        uncertainties = [d['uncertainty'] for d in recent]
        decisions = [d['decision'] for d in recent]
        
        return {
            'avg_confidence': np.mean(confidences),
            'avg_uncertainty': np.mean(uncertainties),
            'decision_rate': np.mean(decisions),
            'confidence_std': np.std(confidences),
            'total_decisions': len(self.decision_history),
            'calibration_quality': self._assess_calibration_quality()
        }
        
    def _assess_calibration_quality(self) -> float:
        """Assess how well calibrated the confidence scores are."""
        if len(self.calibration_history) < 100:
            return 0.5
            
        # Group by confidence buckets
        buckets = np.linspace(0.5, 1.0, 11)
        calibration_error = 0
        
        for i in range(len(buckets) - 1):
            bucket_mask = [
                buckets[i] <= h['confidence'] < buckets[i+1]
                for h in self.calibration_history
            ]
            
            if sum(bucket_mask) > 10:
                bucket_items = [
                    h for h, m in zip(self.calibration_history, bucket_mask) if m
                ]
                
                expected_acc = (buckets[i] + buckets[i+1]) / 2
                actual_acc = np.mean([h['was_correct'] for h in bucket_items])
                
                calibration_error += abs(expected_acc - actual_acc)
                
        # Lower error is better, convert to quality score
        quality = 1.0 - min(calibration_error / 5.0, 1.0)
        
        return quality
        
    def _init_cuda_kernels(self):
        """Initialize CUDA kernels if available."""
        # This would contain actual CUDA kernel initialization
        # For now, we'll use standard PyTorch operations
        logger.info("GPU optimization enabled for MC Dropout")


class TemperatureScaling(nn.Module):
    """Temperature scaling for probability calibration."""
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
        
    def calibrate(self, logits: torch.Tensor, uncertainty: float = 0.0) -> torch.Tensor:
        """Apply temperature scaling."""
        # Adjust temperature based on uncertainty
        temp = self.temperature * (1.0 + uncertainty * 0.5)
        return F.softmax(logits / temp, dim=-1)
        
    def fit(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Fit temperature parameter."""
        # This would implement proper fitting
        # For now, simple heuristic
        accuracy = (predictions.round() == targets).float().mean()
        if accuracy > 0.8:
            self.temperature.data *= 1.1
        else:
            self.temperature.data *= 0.9
            
        self.temperature.data = torch.clamp(self.temperature.data, 0.5, 2.0)


class PlattScaling(nn.Module):
    """Platt scaling for probability calibration."""
    
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))
        
    def calibrate(self, probs: torch.Tensor, uncertainty: float = 0.0) -> torch.Tensor:
        """Apply Platt scaling."""
        # Convert to logits
        logits = torch.log(probs / (1 - probs + 1e-8))
        
        # Apply linear transformation
        calibrated_logits = self.a * logits + self.b
        
        # Adjust for uncertainty
        if uncertainty > 0:
            calibrated_logits *= (1.0 - uncertainty * 0.3)
            
        return F.softmax(calibrated_logits, dim=-1)
        
    def fit(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Fit Platt scaling parameters using logistic regression."""
        # Simplified fitting
        # In practice, use proper optimization
        error = predictions - targets
        self.a.data *= (1.0 - 0.01 * error.mean())
        self.b.data -= 0.01 * error.mean()


class IsotonicCalibration:
    """Isotonic regression for probability calibration."""
    
    def __init__(self):
        self.calibration_map = None
        
    def calibrate(self, probs: torch.Tensor, uncertainty: float = 0.0) -> torch.Tensor:
        """Apply isotonic calibration."""
        if self.calibration_map is None:
            return probs
            
        # Apply learned monotonic mapping
        calibrated = self._apply_isotonic_map(probs)
        
        # Adjust for uncertainty
        if uncertainty > 0:
            # Push towards 0.5 based on uncertainty
            calibrated = calibrated * (1 - uncertainty * 0.2) + 0.5 * uncertainty * 0.2
            
        return calibrated
        
    def _apply_isotonic_map(self, probs: torch.Tensor) -> torch.Tensor:
        """Apply the learned isotonic mapping."""
        # This would implement the actual isotonic regression mapping
        # For now, return as-is
        return probs
        
    def fit(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Fit isotonic regression."""
        # This would implement proper isotonic regression fitting
        # Using sklearn.isotonic.IsotonicRegression in practice
        pass


# CUDA kernel implementations for GPU optimization
CUDA_AVAILABLE = False

if torch.cuda.is_available():
    try:
        # Check if we can use CUDA operations
        test_tensor = torch.randn(1, 1).cuda()
        CUDA_AVAILABLE = True
        logger.info("CUDA available for MC Dropout optimization")
    except:
        CUDA_AVAILABLE = False
        logger.warning("CUDA not fully available, using CPU fallback")