"""
File: src/agents/main_core/mc_dropout_policy.py (NEW FILE)
Enhanced MC Dropout consensus for shared policy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class MCDropoutResult:
    """Results from MC Dropout evaluation."""
    should_qualify: bool
    confidence: float
    uncertainty: float
    mean_probs: torch.Tensor
    std_probs: torch.Tensor
    sample_variance: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    decision_distribution: Dict[str, float]

class MCDropoutConsensus:
    """
    Enhanced MC Dropout consensus mechanism for shared policy.
    Provides sophisticated uncertainty quantification and decision confidence.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.n_samples = config.get('n_samples', 50)
        self.confidence_threshold = config.get('confidence_threshold', 0.80)
        self.uncertainty_threshold = config.get('uncertainty_threshold', 0.3)
        self.min_agreement = config.get('min_agreement', 0.75)
        
        # Adaptive sampling
        self.use_adaptive = config.get('use_adaptive_sampling', True)
        self.min_samples = config.get('min_adaptive_samples', 20)
        self.max_samples = config.get('max_adaptive_samples', 100)
        self.early_stop_threshold = config.get('early_stop_threshold', 0.95)
        
        # Calibration
        self.temperature = config.get('temperature', 1.0)
        self.calibration_alpha = 0.99  # EMA for calibration stats
        self.calibration_stats = {
            'mean_confidence': 0.5,
            'mean_accuracy': 0.5
        }
        
    def evaluate(self, policy: nn.Module, 
                unified_state: torch.Tensor,
                track_samples: bool = False) -> MCDropoutResult:
        """
        Perform MC Dropout evaluation with enhanced analysis.
        
        Args:
            policy: Shared policy network
            unified_state: Input state
            track_samples: Whether to track individual samples
            
        Returns:
            Comprehensive MCDropoutResult
        """
        device = unified_state.device
        
        # Enable dropout
        policy.train()
        
        # Collect samples
        samples = []
        all_logits = []
        all_features = []
        
        with torch.no_grad():
            n_samples = self._determine_sample_count() if self.use_adaptive else self.n_samples
            
            for i in range(n_samples):
                output = policy(unified_state, return_value=False, return_features=True)
                
                samples.append(output.action_probs)
                all_logits.append(output.action_logits)
                
                if track_samples:
                    all_features.append(output.policy_features)
                    
                # Early stopping check
                if self.use_adaptive and i >= self.min_samples:
                    if self._check_early_stop(samples):
                        logger.debug(f"Early stopping at {i+1} samples")
                        break
                        
        # Convert to tensors
        all_probs = torch.stack(samples)  # [n_samples, batch, action_dim]
        all_logits = torch.stack(all_logits)
        
        # Calculate statistics
        mean_probs = all_probs.mean(dim=0)
        std_probs = all_probs.std(dim=0)
        
        # Uncertainty decomposition
        total_uncertainty = self._calculate_total_uncertainty(mean_probs)
        aleatoric = self._calculate_aleatoric_uncertainty(all_probs)
        epistemic = self._calculate_epistemic_uncertainty(all_probs, mean_probs)
        
        # Decision analysis
        qualify_prob = mean_probs[0, 0].item()  # Probability of 'Initiate_Trade'
        qualify_std = std_probs[0, 0].item()
        
        # Agreement analysis
        decisions = (all_probs[:, :, 0] > 0.5).float()  # Binary decisions
        agreement_rate = decisions.mean(dim=0).item()
        
        # Sample variance (measure of consistency)
        sample_variance = all_probs.var(dim=0).mean().item()
        
        # Make decision
        should_qualify = self._make_decision(
            qualify_prob, qualify_std, agreement_rate, epistemic
        )
        
        # Calculate confidence (calibrated)
        raw_confidence = qualify_prob if should_qualify else (1 - qualify_prob)
        calibrated_confidence = self._calibrate_confidence(raw_confidence, epistemic)
        
        # Decision distribution
        decision_dist = {
            'initiate_trade': (decisions == 1).sum().item() / len(decisions),
            'do_nothing': (decisions == 0).sum().item() / len(decisions),
            'uncertain': ((all_probs[:, :, 0] > 0.4) & (all_probs[:, :, 0] < 0.6)).sum().item() / len(all_probs)
        }
        
        # Back to eval mode
        policy.eval()
        
        return MCDropoutResult(
            should_qualify=should_qualify,
            confidence=calibrated_confidence,
            uncertainty=epistemic,
            mean_probs=mean_probs,
            std_probs=std_probs,
            sample_variance=sample_variance,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            decision_distribution=decision_dist
        )
        
    def _determine_sample_count(self) -> int:
        """Adaptively determine number of samples needed."""
        # Use recent uncertainty to guide sampling
        if hasattr(self, 'recent_uncertainties'):
            avg_uncertainty = np.mean(self.recent_uncertainties[-10:])
            
            # More samples for high uncertainty
            if avg_uncertainty > 0.3:
                return self.max_samples
            elif avg_uncertainty > 0.2:
                return (self.min_samples + self.max_samples) // 2
            else:
                return self.min_samples
        else:
            return self.n_samples
            
    def _check_early_stop(self, samples: List[torch.Tensor]) -> bool:
        """Check if we can stop sampling early."""
        if len(samples) < self.min_samples:
            return False
            
        # Calculate running statistics
        probs = torch.stack(samples)
        mean_prob = probs.mean(dim=0)
        std_prob = probs.std(dim=0)
        
        # Check if decision is clear
        max_prob = mean_prob.max(dim=-1)[0].item()
        uncertainty = std_prob.max(dim=-1)[0].item()
        
        return max_prob > self.early_stop_threshold and uncertainty < 0.1
        
    def _calculate_total_uncertainty(self, mean_probs: torch.Tensor) -> float:
        """Calculate total predictive uncertainty (entropy)."""
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
        return entropy.mean().item()
        
    def _calculate_aleatoric_uncertainty(self, all_probs: torch.Tensor) -> float:
        """Calculate aleatoric (data) uncertainty."""
        # Expected entropy
        individual_entropies = -torch.sum(
            all_probs * torch.log(all_probs + 1e-8), dim=-1
        )
        return individual_entropies.mean().item()
        
    def _calculate_epistemic_uncertainty(self, all_probs: torch.Tensor, 
                                       mean_probs: torch.Tensor) -> float:
        """Calculate epistemic (model) uncertainty."""
        # Mutual information between predictions and model
        total = self._calculate_total_uncertainty(mean_probs)
        aleatoric = self._calculate_aleatoric_uncertainty(all_probs)
        return total - aleatoric
        
    def _make_decision(self, qualify_prob: float, qualify_std: float,
                      agreement_rate: float, epistemic: float) -> bool:
        """Make decision based on multiple criteria."""
        # Check confidence threshold
        confidence_met = qualify_prob > self.confidence_threshold
        
        # Check uncertainty threshold
        uncertainty_ok = epistemic < self.uncertainty_threshold
        
        # Check agreement threshold
        agreement_ok = agreement_rate > self.min_agreement
        
        # Require all conditions
        return confidence_met and uncertainty_ok and agreement_ok
        
    def _calibrate_confidence(self, raw_confidence: float, 
                            epistemic: float) -> float:
        """Calibrate confidence score based on uncertainty."""
        # Reduce confidence based on epistemic uncertainty
        uncertainty_penalty = epistemic * 0.5
        calibrated = raw_confidence * (1 - uncertainty_penalty)
        
        # Apply temperature scaling
        calibrated = calibrated ** (1 / self.temperature)
        
        # Update calibration statistics
        self.calibration_stats['mean_confidence'] = (
            self.calibration_alpha * self.calibration_stats['mean_confidence'] +
            (1 - self.calibration_alpha) * calibrated
        )
        
        return np.clip(calibrated, 0.0, 1.0)
        
    def update_calibration(self, decision: bool, outcome: bool):
        """Update calibration statistics based on outcome."""
        accuracy = float(decision == outcome)
        
        self.calibration_stats['mean_accuracy'] = (
            self.calibration_alpha * self.calibration_stats['mean_accuracy'] +
            (1 - self.calibration_alpha) * accuracy
        )
        
        # Adjust temperature based on calibration
        if self.calibration_stats['mean_confidence'] > self.calibration_stats['mean_accuracy'] + 0.1:
            self.temperature *= 1.01  # Increase temperature (reduce confidence)
        elif self.calibration_stats['mean_confidence'] < self.calibration_stats['mean_accuracy'] - 0.1:
            self.temperature *= 0.99  # Decrease temperature (increase confidence)
            
        self.temperature = np.clip(self.temperature, 0.5, 2.0)