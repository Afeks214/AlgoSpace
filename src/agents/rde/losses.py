"""
Uncertainty-aware loss functions for RDE Communication LSTM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TemporalConsistencyLoss(nn.Module):
    """
    Loss function that penalizes rapid regime changes and encourages smooth transitions.
    """
    
    def __init__(self, smoothness_weight: float = 0.5):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        
    def forward(
        self,
        mu_current: torch.Tensor,
        mu_previous: torch.Tensor,
        sigma_current: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate temporal consistency loss.
        
        Args:
            mu_current: Current regime embedding [batch, 16]
            mu_previous: Previous regime embedding [batch, 16]
            sigma_current: Current uncertainty [batch, 16]
            
        Returns:
            Temporal consistency loss scalar
        """
        # L2 distance between consecutive embeddings
        temporal_distance = torch.norm(mu_current - mu_previous, p=2, dim=-1)
        
        # Weight by inverse uncertainty (more certain = stronger consistency requirement)
        uncertainty_weight = 1.0 / (1.0 + sigma_current.mean(dim=-1))
        
        # Penalize large jumps more heavily
        consistency_loss = uncertainty_weight * temporal_distance.pow(2)
        
        # Add smoothness regularization
        smoothness_penalty = self.smoothness_weight * temporal_distance
        
        return (consistency_loss + smoothness_penalty).mean()


class UncertaintyCalibrationLoss(nn.Module):
    """
    Aleatoric uncertainty loss for proper uncertainty calibration.
    """
    
    def __init__(self, min_uncertainty: float = 1e-6):
        super().__init__()
        self.min_uncertainty = min_uncertainty
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate uncertainty calibration loss using negative log-likelihood.
        
        L_unc = E[(ε/σ²) + log(σ)]
        
        Args:
            predictions: Predicted values [batch, dim]
            targets: Target values [batch, dim]
            uncertainties: Predicted uncertainties (log variance) [batch, dim]
            
        Returns:
            Uncertainty calibration loss
        """
        # Ensure positive uncertainty
        sigma_squared = uncertainties.pow(2) + self.min_uncertainty
        
        # Prediction error
        error = (predictions - targets).pow(2)
        
        # Negative log-likelihood
        nll = 0.5 * (error / sigma_squared + torch.log(sigma_squared))
        
        return nll.mean()


class RegimePredictionLoss(nn.Module):
    """
    Self-supervised loss for next-step regime prediction with contrastive learning.
    """
    
    def __init__(self, temperature: float = 0.07, margin: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(
        self,
        mu_current: torch.Tensor,
        mu_next: torch.Tensor,
        negative_samples: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate regime prediction loss with optional contrastive component.
        
        Args:
            mu_current: Current regime embeddings [batch, 16]
            mu_next: Next timestep regime embeddings [batch, 16]
            negative_samples: Negative regime samples [n_neg, 16]
            
        Returns:
            prediction_loss: Next-step prediction loss
            contrastive_loss: Contrastive discrimination loss
        """
        batch_size = mu_current.size(0)
        
        # Next-step prediction loss (L2)
        prediction_loss = F.mse_loss(mu_current, mu_next)
        
        # Contrastive loss
        contrastive_loss = torch.tensor(0.0, device=mu_current.device)
        
        if negative_samples is not None:
            # Normalize embeddings
            mu_current_norm = F.normalize(mu_current, dim=-1)
            mu_next_norm = F.normalize(mu_next, dim=-1)
            neg_norm = F.normalize(negative_samples, dim=-1)
            
            # Positive pairs similarity
            pos_sim = torch.sum(mu_current_norm * mu_next_norm, dim=-1) / self.temperature
            
            # Negative pairs similarity
            neg_sim = torch.matmul(mu_current_norm, neg_norm.T) / self.temperature
            
            # InfoNCE loss
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
            labels = torch.zeros(batch_size, dtype=torch.long, device=mu_current.device)
            
            contrastive_loss = F.cross_entropy(logits, labels)
        
        return prediction_loss, contrastive_loss


class RDECommunicationLoss(nn.Module):
    """
    Combined loss function for RDE Communication LSTM training.
    """
    
    def __init__(self, config: Dict[str, float]):
        super().__init__()
        
        # Loss weights
        self.w_temporal = config.get('temporal_consistency', 0.3)
        self.w_uncertainty = config.get('uncertainty_calibration', 0.4)
        self.w_prediction = config.get('regime_prediction', 0.3)
        self.w_contrastive = config.get('contrastive', 0.1)
        
        # Individual loss functions
        self.temporal_loss = TemporalConsistencyLoss()
        self.uncertainty_loss = UncertaintyCalibrationLoss()
        self.prediction_loss = RegimePredictionLoss()
        
        logger.info(f"RDE Communication Loss initialized with weights: "
                   f"temporal={self.w_temporal}, uncertainty={self.w_uncertainty}, "
                   f"prediction={self.w_prediction}, contrastive={self.w_contrastive}")
        
    def forward(
        self,
        mu_current: torch.Tensor,
        sigma_current: torch.Tensor,
        mu_previous: Optional[torch.Tensor] = None,
        mu_next: Optional[torch.Tensor] = None,
        regime_targets: Optional[torch.Tensor] = None,
        negative_samples: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate combined loss with all components.
        
        Args:
            mu_current: Current regime embedding [batch, 16]
            sigma_current: Current uncertainty [batch, 16]
            mu_previous: Previous regime embedding [batch, 16]
            mu_next: Next regime embedding for prediction [batch, 16]
            regime_targets: Target regime vectors [batch, 8]
            negative_samples: Negative samples for contrastive learning
            
        Returns:
            Dictionary of loss components and total loss
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=mu_current.device)
        
        # Temporal consistency loss
        if mu_previous is not None:
            temporal_loss = self.temporal_loss(mu_current, mu_previous, sigma_current)
            losses['temporal'] = temporal_loss
            total_loss += self.w_temporal * temporal_loss
        else:
            losses['temporal'] = torch.tensor(0.0)
        
        # Uncertainty calibration loss
        if regime_targets is not None:
            # Project embeddings back to regime space for uncertainty calibration
            projection = nn.Linear(mu_current.size(-1), regime_targets.size(-1)).to(mu_current.device)
            projected_mu = projection(mu_current)
            projected_sigma = projection(sigma_current)
            
            uncertainty_loss = self.uncertainty_loss(projected_mu, regime_targets, projected_sigma)
            losses['uncertainty'] = uncertainty_loss
            total_loss += self.w_uncertainty * uncertainty_loss
        else:
            losses['uncertainty'] = torch.tensor(0.0)
        
        # Regime prediction loss
        if mu_next is not None:
            pred_loss, contrast_loss = self.prediction_loss(mu_current, mu_next, negative_samples)
            losses['prediction'] = pred_loss
            losses['contrastive'] = contrast_loss
            total_loss += self.w_prediction * pred_loss + self.w_contrastive * contrast_loss
        else:
            losses['prediction'] = torch.tensor(0.0)
            losses['contrastive'] = torch.tensor(0.0)
        
        losses['total'] = total_loss
        
        return losses


class GradientFlowMonitor:
    """
    Monitor gradient flow through the model for debugging.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradient_stats = {}
        
    def register_hooks(self):
        """Register backward hooks to monitor gradients."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.register_hook(lambda grad, name=name: self._save_gradient(name, grad))
                
    def _save_gradient(self, name: str, grad: torch.Tensor):
        """Save gradient statistics."""
        self.gradient_stats[name] = {
            'mean': grad.mean().item(),
            'std': grad.std().item(),
            'norm': grad.norm().item(),
            'max': grad.max().item(),
            'min': grad.min().item()
        }
        
    def check_gradient_health(self) -> Dict[str, bool]:
        """
        Check for gradient issues.
        
        Returns:
            Dictionary of health checks
        """
        health = {
            'has_gradients': len(self.gradient_stats) > 0,
            'no_vanishing': True,
            'no_explosion': True,
            'balanced_flow': True
        }
        
        if not health['has_gradients']:
            return health
            
        norms = [stats['norm'] for stats in self.gradient_stats.values()]
        
        # Check for vanishing gradients
        if max(norms) < 1e-7:
            health['no_vanishing'] = False
            
        # Check for exploding gradients
        if max(norms) > 100:
            health['no_explosion'] = False
            
        # Check for balanced gradient flow
        if len(norms) > 1:
            norm_ratio = max(norms) / (min(norms) + 1e-8)
            if norm_ratio > 1000:
                health['balanced_flow'] = False
                
        return health
    
    def get_gradient_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of gradient statistics."""
        return self.gradient_stats.copy()