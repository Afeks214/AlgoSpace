"""
Advanced calibration system for MC Dropout confidence scores.

This module implements multiple calibration methods (temperature scaling,
Platt scaling, isotonic regression, beta calibration, histogram binning)
and learns optimal calibration from historical trading data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.stats import beta
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Any
import pickle
import logging

logger = logging.getLogger(__name__)


class MCDropoutCalibrator:
    """
    Comprehensive calibration system for MC Dropout predictions.
    
    Implements multiple calibration methods and learns optimal
    calibration from historical data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.calibration_method = config.get('method', 'ensemble')
        
        # Calibration models
        self.calibrators = {
            'temperature': TemperatureCalibrator(),
            'platt': PlattCalibrator(),
            'isotonic': IsotonicCalibrator(),
            'beta': BetaCalibrator(),
            'histogram': HistogramCalibrator(n_bins=15)
        }
        
        # Ensemble weights
        self.ensemble_weights = {
            'temperature': 0.3,
            'platt': 0.2,
            'isotonic': 0.3,
            'beta': 0.1,
            'histogram': 0.1
        }
        
        # Historical data for learning
        self.calibration_data = []
        self.outcome_data = []
        
    def calibrate(
        self,
        raw_probabilities: torch.Tensor,
        uncertainty_metrics: Dict[str, float],
        market_context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Apply calibration to raw MC Dropout probabilities.
        
        Args:
            raw_probabilities: Uncalibrated probabilities
            uncertainty_metrics: Uncertainty decomposition
            market_context: Current market conditions
            
        Returns:
            Calibrated probabilities
        """
        if self.calibration_method == 'ensemble':
            # Ensemble of calibration methods
            calibrated_probs = []
            weights = []
            
            for name, calibrator in self.calibrators.items():
                if name in self.ensemble_weights:
                    # Adjust weight based on uncertainty
                    weight = self._adjust_weight(
                        name,
                        self.ensemble_weights[name],
                        uncertainty_metrics
                    )
                    
                    cal_prob = calibrator.calibrate(
                        raw_probabilities,
                        uncertainty_metrics
                    )
                    
                    calibrated_probs.append(cal_prob)
                    weights.append(weight)
                    
            # Weighted average
            weights = torch.tensor(weights)
            weights = weights / weights.sum()
            
            calibrated = torch.zeros_like(raw_probabilities)
            for i, (prob, weight) in enumerate(zip(calibrated_probs, weights)):
                calibrated += weight * prob
                
        else:
            # Single calibration method
            calibrator = self.calibrators[self.calibration_method]
            calibrated = calibrator.calibrate(
                raw_probabilities,
                uncertainty_metrics
            )
            
        # Market context adjustment
        if market_context:
            calibrated = self._apply_market_adjustment(
                calibrated,
                market_context
            )
            
        return calibrated
        
    def _adjust_weight(
        self,
        method: str,
        base_weight: float,
        uncertainty_metrics: Dict[str, float]
    ) -> float:
        """Adjust calibration method weight based on uncertainty."""
        epistemic = uncertainty_metrics.get('epistemic_uncertainty', 0)
        aleatoric = uncertainty_metrics.get('aleatoric_uncertainty', 0)
        
        # Temperature scaling works better with low uncertainty
        if method == 'temperature':
            if epistemic < 0.1:
                return base_weight * 1.5
            else:
                return base_weight * 0.7
                
        # Isotonic regression works better with high uncertainty
        elif method == 'isotonic':
            if epistemic > 0.3:
                return base_weight * 1.3
            else:
                return base_weight * 0.8
                
        # Beta calibration for extreme probabilities
        elif method == 'beta':
            if aleatoric > 0.4:
                return base_weight * 1.2
            else:
                return base_weight
                
        return base_weight
        
    def _apply_market_adjustment(
        self,
        probabilities: torch.Tensor,
        market_context: Dict[str, Any]
    ) -> torch.Tensor:
        """Adjust calibration based on market conditions."""
        regime = market_context.get('regime', 'normal')
        volatility = market_context.get('volatility', 1.0)
        
        # More conservative in volatile regimes
        if regime == 'volatile' or volatility > 1.5:
            # Push probabilities toward 0.5
            probabilities = probabilities * 0.8 + 0.1
            
        # More confident in trending regimes
        elif regime == 'trending':
            # Sharpen probabilities
            probabilities = torch.pow(probabilities, 0.8)
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
            
        return probabilities
        
    def update(
        self,
        predictions: List[Dict[str, Any]],
        outcomes: List[bool]
    ):
        """
        Update calibration models with new data.
        
        Args:
            predictions: List of prediction dictionaries
            outcomes: List of actual outcomes (profitable or not)
        """
        if len(predictions) != len(outcomes):
            raise ValueError("Predictions and outcomes must have same length")
            
        # Store data
        self.calibration_data.extend(predictions)
        self.outcome_data.extend(outcomes)
        
        # Limit history
        if len(self.calibration_data) > 10000:
            self.calibration_data = self.calibration_data[-5000:]
            self.outcome_data = self.outcome_data[-5000:]
            
        # Update each calibrator
        if len(self.calibration_data) >= 100:
            pred_probs = torch.tensor([
                p['probability'] for p in self.calibration_data
            ])
            true_outcomes = torch.tensor(self.outcome_data, dtype=torch.float32)
            
            for calibrator in self.calibrators.values():
                calibrator.fit(pred_probs, true_outcomes)
                
            # Update ensemble weights based on performance
            self._update_ensemble_weights()
            
        logger.info(f"Calibration updated with {len(predictions)} new samples")
        
    def _update_ensemble_weights(self):
        """Update ensemble weights based on calibration performance."""
        if len(self.calibration_data) < 500:
            return
            
        # Evaluate each calibrator
        performances = {}
        
        for name, calibrator in self.calibrators.items():
            # Get calibration error
            error = self._evaluate_calibrator(calibrator)
            performances[name] = 1.0 / (1.0 + error)  # Convert to score
            
        # Update weights proportionally
        total_performance = sum(performances.values())
        
        for name in self.ensemble_weights:
            if name in performances:
                self.ensemble_weights[name] = (
                    performances[name] / total_performance
                )
                
        logger.info(f"Updated ensemble weights: {self.ensemble_weights}")
        
    def _evaluate_calibrator(
        self,
        calibrator: Any,
        n_bins: int = 10
    ) -> float:
        """Evaluate calibrator using ECE (Expected Calibration Error)."""
        pred_probs = torch.tensor([
            p['probability'] for p in self.calibration_data[-500:]
        ])
        true_outcomes = torch.tensor(
            self.outcome_data[-500:], 
            dtype=torch.float32
        )
        
        # Apply calibration
        cal_probs = calibrator.calibrate(pred_probs.unsqueeze(-1))
        cal_probs = cal_probs.squeeze()
        
        # Calculate ECE
        ece = 0.0
        for i in range(n_bins):
            # Bin boundaries
            bin_lower = i / n_bins
            bin_upper = (i + 1) / n_bins
            
            # Select samples in bin
            in_bin = (cal_probs > bin_lower) & (cal_probs <= bin_upper)
            
            if in_bin.sum() > 0:
                # Accuracy in bin
                bin_accuracy = true_outcomes[in_bin].mean()
                
                # Average confidence in bin
                bin_confidence = cal_probs[in_bin].mean()
                
                # Weighted absolute difference
                bin_weight = in_bin.float().mean()
                ece += bin_weight * abs(bin_accuracy - bin_confidence)
                
        return ece.item()
        
    def save_calibration(self, path: str):
        """Save calibration models and data."""
        save_dict = {
            'calibrators': self.calibrators,
            'ensemble_weights': self.ensemble_weights,
            'calibration_data': self.calibration_data[-1000:],  # Recent data
            'outcome_data': self.outcome_data[-1000:],
            'config': self.config
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
            
        logger.info(f"Calibration saved to {path}")
        
    def load_calibration(self, path: str):
        """Load calibration models and data."""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
            
        self.calibrators = save_dict['calibrators']
        self.ensemble_weights = save_dict['ensemble_weights']
        self.calibration_data = save_dict['calibration_data']
        self.outcome_data = save_dict['outcome_data']
        self.config = save_dict['config']
        
        logger.info(f"Calibration loaded from {path}")


class TemperatureCalibrator:
    """Temperature scaling for probability calibration."""
    
    def __init__(self):
        self.temperature = 1.0
        
    def calibrate(
        self,
        probs: torch.Tensor,
        uncertainty_metrics: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """Apply temperature scaling."""
        # Adjust temperature based on uncertainty
        temp = self.temperature
        if uncertainty_metrics:
            epistemic = uncertainty_metrics.get('epistemic_uncertainty', 0)
            temp = self.temperature * (1.0 + epistemic * 0.5)
            
        # Apply temperature scaling to logits
        if probs.min() > 0 and probs.max() < 1:
            # Convert probabilities to logits
            logits = torch.log(probs / (1 - probs + 1e-8))
            # Scale and convert back
            scaled_logits = logits / temp
            return torch.sigmoid(scaled_logits)
        else:
            # Already logits
            return torch.sigmoid(probs / temp)
        
    def fit(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Fit temperature parameter using NLL loss."""
        from torch.optim import LBFGS
        
        # Convert to logits if needed
        if predictions.min() > 0 and predictions.max() < 1:
            logits = torch.log(predictions / (1 - predictions + 1e-8))
        else:
            logits = predictions
            
        # Temperature as learnable parameter
        temperature = nn.Parameter(torch.ones(1) * self.temperature)
        optimizer = LBFGS([temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = logits / temperature
            loss = F.binary_cross_entropy_with_logits(scaled_logits, targets)
            loss.backward()
            return loss
            
        optimizer.step(eval_loss)
        
        self.temperature = temperature.item()
        self.temperature = np.clip(self.temperature, 0.5, 2.0)


class PlattCalibrator:
    """Platt scaling for probability calibration."""
    
    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        
    def calibrate(
        self,
        probs: torch.Tensor,
        uncertainty_metrics: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """Apply Platt scaling."""
        # Convert to logits
        if probs.min() > 0 and probs.max() < 1:
            logits = torch.log(probs / (1 - probs + 1e-8))
        else:
            logits = probs
            
        # Apply linear transformation
        calibrated_logits = self.a * logits + self.b
        
        # Adjust for uncertainty
        if uncertainty_metrics:
            epistemic = uncertainty_metrics.get('epistemic_uncertainty', 0)
            if epistemic > 0:
                calibrated_logits *= (1.0 - epistemic * 0.3)
                
        return torch.sigmoid(calibrated_logits)
        
    def fit(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Fit Platt scaling parameters using logistic regression."""
        # Use sklearn for robust fitting
        lr = LogisticRegression()
        
        # Convert to numpy
        X = predictions.numpy().reshape(-1, 1)
        y = targets.numpy()
        
        # Fit logistic regression
        lr.fit(X, y)
        
        # Extract parameters
        self.a = float(lr.coef_[0])
        self.b = float(lr.intercept_[0])


class IsotonicCalibrator:
    """Isotonic regression for probability calibration."""
    
    def __init__(self):
        self.isotonic_regressor = None
        
    def calibrate(
        self,
        probs: torch.Tensor,
        uncertainty_metrics: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """Apply isotonic calibration."""
        if self.isotonic_regressor is None:
            return probs
            
        # Apply isotonic regression
        probs_np = probs.cpu().numpy().flatten()
        calibrated_np = self.isotonic_regressor.transform(probs_np)
        calibrated = torch.from_numpy(calibrated_np).to(probs.device)
        calibrated = calibrated.reshape(probs.shape)
        
        # Adjust for uncertainty
        if uncertainty_metrics:
            epistemic = uncertainty_metrics.get('epistemic_uncertainty', 0)
            if epistemic > 0:
                # Push towards 0.5 based on uncertainty
                calibrated = calibrated * (1 - epistemic * 0.2) + 0.5 * epistemic * 0.2
                
        return calibrated
        
    def fit(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Fit isotonic regression."""
        # Convert to numpy
        X = predictions.cpu().numpy()
        y = targets.cpu().numpy()
        
        # Fit isotonic regression
        self.isotonic_regressor = IsotonicRegression(out_of_bounds='clip')
        self.isotonic_regressor.fit(X, y)


class BetaCalibrator:
    """Beta calibration for probability adjustment."""
    
    def __init__(self):
        self.alpha = 1.0
        self.beta_param = 1.0
        
    def calibrate(
        self,
        probs: torch.Tensor,
        uncertainty_metrics: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """Apply beta calibration."""
        # Ensure probs are in valid range
        probs = torch.clamp(probs, 1e-7, 1-1e-7)
        
        # Apply beta transformation
        calibrated = []
        for p in probs.flatten():
            cal_p = beta.cdf(p.item(), self.alpha, self.beta_param)
            calibrated.append(cal_p)
            
        calibrated = torch.tensor(calibrated, device=probs.device)
        calibrated = calibrated.reshape(probs.shape)
        
        return calibrated
        
    def fit(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Fit beta parameters using MLE."""
        def neg_log_likelihood(params):
            a, b = params
            if a <= 0 or b <= 0:
                return np.inf
                
            # Beta log likelihood
            ll = 0
            for pred, target in zip(predictions, targets):
                p = pred.item()
                p = np.clip(p, 1e-7, 1-1e-7)
                
                if target == 1:
                    ll += np.log(beta.cdf(p, a, b) + 1e-10)
                else:
                    ll += np.log(1 - beta.cdf(p, a, b) + 1e-10)
                    
            return -ll
            
        # Optimize
        result = minimize(
            neg_log_likelihood,
            x0=[self.alpha, self.beta_param],
            bounds=[(0.1, 10), (0.1, 10)],
            method='L-BFGS-B'
        )
        
        if result.success:
            self.alpha, self.beta_param = result.x


class HistogramCalibrator:
    """Histogram binning calibration."""
    
    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins
        self.bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_calibrations = torch.ones(n_bins) * 0.5
        
    def calibrate(
        self,
        probs: torch.Tensor,
        uncertainty_metrics: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """Apply histogram calibration."""
        calibrated = torch.zeros_like(probs)
        
        for i in range(self.n_bins):
            # Find probabilities in this bin
            if i == 0:
                mask = probs <= self.bin_boundaries[i + 1]
            elif i == self.n_bins - 1:
                mask = probs > self.bin_boundaries[i]
            else:
                mask = (probs > self.bin_boundaries[i]) & (
                    probs <= self.bin_boundaries[i + 1]
                )
                
            # Apply bin calibration
            calibrated[mask] = self.bin_calibrations[i]
            
        return calibrated
        
    def fit(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Fit histogram bins."""
        for i in range(self.n_bins):
            # Find samples in this bin
            if i == 0:
                mask = predictions <= self.bin_boundaries[i + 1]
            elif i == self.n_bins - 1:
                mask = predictions > self.bin_boundaries[i]
            else:
                mask = (predictions > self.bin_boundaries[i]) & (
                    predictions <= self.bin_boundaries[i + 1]
                )
                
            if mask.sum() > 0:
                # Set bin calibration to average outcome
                self.bin_calibrations[i] = targets[mask].mean()


class CalibrationDiagnostics:
    """Diagnostic tools for calibration quality assessment."""
    
    @staticmethod
    def reliability_diagram(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """Generate reliability diagram data."""
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        
        bin_confidences = []
        bin_accuracies = []
        bin_counts = []
        
        for i in range(n_bins):
            if i == 0:
                mask = predictions <= bin_boundaries[i + 1]
            elif i == n_bins - 1:
                mask = predictions > bin_boundaries[i]
            else:
                mask = (predictions > bin_boundaries[i]) & (
                    predictions <= bin_boundaries[i + 1]
                )
                
            if mask.sum() > 0:
                bin_conf = predictions[mask].mean().item()
                bin_acc = targets[mask].mean().item()
                bin_count = mask.sum().item()
                
                bin_confidences.append(bin_conf)
                bin_accuracies.append(bin_acc)
                bin_counts.append(bin_count)
                
        return {
            'confidences': bin_confidences,
            'accuracies': bin_accuracies,
            'counts': bin_counts,
            'ece': CalibrationDiagnostics.expected_calibration_error(
                predictions, targets, n_bins
            )
        }
        
    @staticmethod
    def expected_calibration_error(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        n_bins: int = 10
    ) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        n_samples = len(predictions)
        ece = 0.0
        
        for i in range(n_bins):
            if i == 0:
                mask = predictions <= bin_boundaries[i + 1]
            elif i == n_bins - 1:
                mask = predictions > bin_boundaries[i]
            else:
                mask = (predictions > bin_boundaries[i]) & (
                    predictions <= bin_boundaries[i + 1]
                )
                
            if mask.sum() > 0:
                bin_confidence = predictions[mask].mean()
                bin_accuracy = targets[mask].mean()
                bin_weight = mask.sum().float() / n_samples
                
                ece += bin_weight * torch.abs(bin_accuracy - bin_confidence)
                
        return ece.item()
        
    @staticmethod
    def maximum_calibration_error(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        n_bins: int = 10
    ) -> float:
        """Calculate Maximum Calibration Error (MCE)."""
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        mce = 0.0
        
        for i in range(n_bins):
            if i == 0:
                mask = predictions <= bin_boundaries[i + 1]
            elif i == n_bins - 1:
                mask = predictions > bin_boundaries[i]
            else:
                mask = (predictions > bin_boundaries[i]) & (
                    predictions <= bin_boundaries[i + 1]
                )
                
            if mask.sum() > 0:
                bin_confidence = predictions[mask].mean()
                bin_accuracy = targets[mask].mean()
                error = torch.abs(bin_accuracy - bin_confidence).item()
                mce = max(mce, error)
                
        return mce