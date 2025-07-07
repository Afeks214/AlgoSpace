"""
Loss functions for MRMS Communication LSTM training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any


class MRMSCommunicationLoss(nn.Module):
    """
    Combined loss function for MRMS Communication LSTM training.
    
    Components:
    1. Risk prediction loss (predict next risk parameters)
    2. Outcome prediction loss (predict trade outcome)
    3. Uncertainty calibration loss
    4. Temporal consistency loss
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Loss weights
        self.w_risk = config.get('weight_risk', 0.3)
        self.w_outcome = config.get('weight_outcome', 0.3)
        self.w_uncertainty = config.get('weight_uncertainty', 0.2)
        self.w_temporal = config.get('weight_temporal', 0.2)
        
    def forward(
        self,
        mu_risk: torch.Tensor,
        sigma_risk: torch.Tensor,
        target_risk: torch.Tensor,
        target_outcome: torch.Tensor,
        previous_mu: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate combined loss.
        
        Args:
            mu_risk: Predicted risk embedding [batch, 8]
            sigma_risk: Predicted uncertainty [batch, 8]
            target_risk: Next timestep risk parameters [batch, 4]
            target_outcome: Trade outcome [batch, 3]
            previous_mu: Previous timestep embedding for consistency
            
        Returns:
            Dictionary of loss components and total loss
        """
        losses = {}
        
        # 1. Risk prediction loss
        risk_pred = mu_risk[:, :4]  # First 4 dims predict risk params
        losses['risk'] = F.mse_loss(risk_pred, target_risk)
        
        # 2. Outcome prediction loss  
        outcome_pred = mu_risk[:, 4:7]  # Next 3 dims predict outcome
        losses['outcome'] = F.binary_cross_entropy_with_logits(
            outcome_pred, target_outcome
        )
        
        # 3. Uncertainty calibration loss
        # Aleatoric uncertainty: L = (ε/σ²) + log(σ)
        risk_error = (risk_pred - target_risk).pow(2)
        sigma_risk_params = sigma_risk[:, :4]
        losses['uncertainty'] = torch.mean(
            risk_error / (sigma_risk_params.pow(2) + 1e-6) + 
            torch.log(sigma_risk_params + 1e-6)
        )
        
        # 4. Temporal consistency loss
        if previous_mu is not None:
            # Penalize rapid changes in risk embedding
            losses['temporal'] = F.mse_loss(
                mu_risk, 
                previous_mu.detach()
            ) * torch.exp(-sigma_risk.mean())  # Less penalty when uncertain
        else:
            losses['temporal'] = torch.tensor(0.0)
            
        # Combined loss
        losses['total'] = (
            self.w_risk * losses['risk'] +
            self.w_outcome * losses['outcome'] +
            self.w_uncertainty * losses['uncertainty'] +
            self.w_temporal * losses['temporal']
        )
        
        return losses