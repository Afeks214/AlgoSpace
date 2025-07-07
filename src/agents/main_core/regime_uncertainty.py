"""
File: src/agents/main_core/regime_uncertainty.py (NEW FILE)
Uncertainty calibration for regime embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.isotonic import IsotonicRegression
import logging

logger = logging.getLogger(__name__)

class RegimeUncertaintyCalibrator:
    """
    Calibrates uncertainty estimates from regime embedder.
    Uses historical performance to ensure reliable confidence scores.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.calibration_window = config.get('calibration_window', 1000)
        
        # Calibration data
        self.predictions = []
        self.uncertainties = []
        self.outcomes = []
        
        # Calibration models
        self.isotonic_calibrator = IsotonicRegression(out_of_bounds='clip')
        self.temperature_scale = 1.0
        
        # Metrics
        self.calibration_metrics = {
            'ece': 0.0,  # Expected Calibration Error
            'mce': 0.0,  # Maximum Calibration Error
            'temperature': 1.0
        }
        
    def add_sample(self, prediction: torch.Tensor, uncertainty: torch.Tensor, 
                   outcome: float):
        """Add a sample for calibration."""
        self.predictions.append(prediction.detach().cpu())
        self.uncertainties.append(uncertainty.detach().cpu())
        self.outcomes.append(outcome)
        
        # Maintain window
        if len(self.predictions) > self.calibration_window:
            self.predictions.pop(0)
            self.uncertainties.pop(0)
            self.outcomes.pop(0)
            
    def calibrate_uncertainty(self, raw_std: torch.Tensor) -> torch.Tensor:
        """Apply calibration to raw uncertainty estimates."""
        # Temperature scaling
        calibrated_std = raw_std / self.temperature_scale
        
        # Ensure reasonable bounds
        calibrated_std = torch.clamp(calibrated_std, min=1e-3, max=2.0)
        
        return calibrated_std
        
    def update_calibration(self):
        """Update calibration models based on recent data."""
        if len(self.predictions) < 100:
            return
            
        # Calculate calibration metrics
        predictions = torch.stack(self.predictions)
        uncertainties = torch.stack(self.uncertainties)
        outcomes = torch.tensor(self.outcomes)
        
        # Compute confidence scores
        confidence_scores = 1.0 / (1.0 + uncertainties.mean(dim=-1))
        
        # Fit isotonic regression
        try:
            self.isotonic_calibrator.fit(
                confidence_scores.numpy(),
                outcomes.numpy()
            )
        except Exception as e:
            logger.warning(f"Isotonic calibration failed: {e}")
            
        # Update temperature scaling
        self._update_temperature_scaling(predictions, uncertainties, outcomes)
        
        # Calculate metrics
        self._calculate_calibration_metrics(confidence_scores, outcomes)
        
    def _update_temperature_scaling(self, predictions: torch.Tensor, 
                                   uncertainties: torch.Tensor,
                                   outcomes: torch.Tensor):
        """Update temperature scaling parameter."""
        # Simple grid search for optimal temperature
        best_temp = 1.0
        best_nll = float('inf')
        
        for temp in np.linspace(0.5, 2.0, 16):
            scaled_std = uncertainties / temp
            
            # Calculate negative log likelihood
            nll = -torch.distributions.Normal(predictions, scaled_std).log_prob(
                outcomes.unsqueeze(-1).expand_as(predictions)
            ).mean()
            
            if nll < best_nll:
                best_nll = nll
                best_temp = temp
                
        self.temperature_scale = best_temp
        self.calibration_metrics['temperature'] = best_temp
        
    def _calculate_calibration_metrics(self, confidence: torch.Tensor, 
                                      outcomes: torch.Tensor):
        """Calculate ECE and MCE metrics."""
        n_bins = 10
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        
        ece = 0.0
        mce = 0.0
        
        for i in range(n_bins):
            mask = (confidence > bin_boundaries[i]) & (confidence <= bin_boundaries[i+1])
            if mask.sum() > 0:
                bin_confidence = confidence[mask].mean()
                bin_accuracy = outcomes[mask].mean()
                bin_error = abs(bin_confidence - bin_accuracy)
                
                ece += (mask.sum().float() / len(confidence)) * bin_error
                mce = max(mce, bin_error)
                
        self.calibration_metrics['ece'] = float(ece)
        self.calibration_metrics['mce'] = float(mce)