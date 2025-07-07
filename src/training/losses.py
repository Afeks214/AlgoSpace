"""
Training loss functions for AlgoSpace neural network components.

This module contains specialized loss functions for training the various
components of the AlgoSpace trading system, including the Structure Embedder
with uncertainty quantification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class StructureEmbedderLoss(nn.Module):
    """
    Specialized loss for training structure embedder.
    
    Combines:
    1. Prediction loss (next bar prediction)
    2. Uncertainty calibration loss
    3. Attention diversity loss
    4. Reconstruction loss
    
    This loss function trains the transformer to:
    - Predict future market structure features
    - Calibrate uncertainty estimates with actual prediction errors
    - Maintain diverse attention patterns across time steps
    - Learn meaningful feature representations
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Loss component weights
        self.w_prediction = config.get('prediction_weight', 0.4)
        self.w_uncertainty = config.get('uncertainty_weight', 0.3)
        self.w_attention = config.get('attention_weight', 0.2)
        self.w_reconstruction = config.get('reconstruction_weight', 0.1)
        
        # Prediction horizon for next-step prediction
        self.prediction_horizon = config.get('prediction_horizon', 1)
        
        # Uncertainty calibration parameters
        self.uncertainty_target_factor = config.get('uncertainty_target_factor', 1.0)
        self.min_uncertainty = config.get('min_uncertainty', 1e-6)
        
        # Attention diversity parameters
        self.min_attention_entropy = config.get('min_attention_entropy', 2.0)
        
        logger.info(f"Initialized StructureEmbedderLoss with weights: "
                   f"pred={self.w_prediction}, unc={self.w_uncertainty}, "
                   f"att={self.w_attention}, rec={self.w_reconstruction}")
        
    def forward(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        attention_weights: torch.Tensor,
        input_sequence: torch.Tensor,
        target_features: Optional[torch.Tensor] = None,
        actual_error: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate structure embedder loss.
        
        Args:
            mu: Mean predictions [batch, output_dim]
            sigma: Uncertainty estimates [batch, output_dim]
            attention_weights: Attention over time steps [batch, seq_len]
            input_sequence: Input sequence [batch, seq_len, input_dim]
            target_features: Target for next-bar prediction [batch, output_dim]
            actual_error: Actual prediction error for uncertainty calibration [batch, output_dim]
            
        Returns:
            Dictionary containing individual loss components and total loss
        """
        losses = {}
        batch_size = mu.size(0)
        
        # 1. Prediction loss (if we have targets)
        if target_features is not None:
            # Use predicted mean for prediction loss
            prediction_loss = F.mse_loss(mu, target_features)
            losses['prediction'] = prediction_loss
        else:
            # Use sequence continuation as prediction target
            if input_sequence.size(1) > self.prediction_horizon:
                # Create targets from input sequence (next step prediction)
                current_features = input_sequence[:, :-self.prediction_horizon, :].mean(dim=2)  # [batch, seq-h]
                target_features = input_sequence[:, self.prediction_horizon:, :].mean(dim=2)   # [batch, seq-h]
                
                # Use final attention-weighted features as prediction
                attended_features = torch.sum(
                    current_features.unsqueeze(-1) * attention_weights[:, :-self.prediction_horizon].unsqueeze(-1),
                    dim=1
                )  # [batch, 1]
                
                target_mean = target_features.mean(dim=1)  # [batch]
                prediction_loss = F.mse_loss(attended_features.squeeze(-1), target_mean)
                losses['prediction'] = prediction_loss
            else:
                losses['prediction'] = torch.tensor(0.0, device=mu.device)
        
        # 2. Uncertainty calibration loss
        if actual_error is not None:
            # Uncertainty should correlate with actual error
            target_uncertainty = torch.sqrt(torch.abs(actual_error) + self.min_uncertainty)
            target_uncertainty = target_uncertainty * self.uncertainty_target_factor
            
            # Use log-likelihood loss for uncertainty calibration
            log_likelihood = -0.5 * torch.log(2 * torch.pi * sigma**2) - 0.5 * (actual_error**2) / (sigma**2)
            uncertainty_loss = -log_likelihood.mean()
            
            losses['uncertainty'] = uncertainty_loss
        else:
            # Regularize uncertainty to prevent collapse and encourage calibration
            # Prevent too small uncertainties
            small_sigma_penalty = F.relu(self.min_uncertainty - sigma).mean()
            
            # Encourage reasonable uncertainty range
            uncertainty_reg = -torch.log(sigma + self.min_uncertainty).mean() * 0.01
            
            losses['uncertainty'] = small_sigma_penalty + uncertainty_reg
        
        # 3. Attention diversity loss
        # Encourage attention to not focus on single timestep
        attention_entropy = -torch.sum(
            attention_weights * torch.log(attention_weights + 1e-8), 
            dim=-1
        )  # [batch]
        
        # Penalize low entropy (high concentration)
        target_entropy = self.min_attention_entropy
        diversity_loss = F.relu(target_entropy - attention_entropy).mean()
        
        # Also penalize too uniform attention (entropy too high)
        max_entropy = torch.log(torch.tensor(attention_weights.size(-1), dtype=torch.float))
        uniform_penalty = F.relu(attention_entropy - max_entropy * 0.9).mean()
        
        losses['attention'] = diversity_loss + uniform_penalty * 0.1
        
        # 4. Reconstruction loss (feature consistency)
        if input_sequence.size(1) > 1:
            # Use attention weights to reconstruct input features
            weighted_input = torch.sum(
                input_sequence * attention_weights.unsqueeze(-1), 
                dim=1
            )  # [batch, input_dim]
            
            # Project mu back to input space for reconstruction
            reconstruction_target = weighted_input.mean(dim=-1, keepdim=True).expand_as(mu[:, :1])
            reconstruction_loss = F.mse_loss(mu[:, :1], reconstruction_target)
            
            losses['reconstruction'] = reconstruction_loss
        else:
            losses['reconstruction'] = torch.tensor(0.0, device=mu.device)
        
        # Combined loss
        total_loss = (
            self.w_prediction * losses['prediction'] +
            self.w_uncertainty * losses['uncertainty'] +
            self.w_attention * losses['attention'] +
            self.w_reconstruction * losses['reconstruction']
        )
        
        losses['total'] = total_loss
        
        # Add metrics for monitoring
        losses['metrics'] = {
            'attention_entropy_mean': attention_entropy.mean().item(),
            'attention_entropy_std': attention_entropy.std().item(),
            'attention_max': attention_weights.max().item(),
            'attention_min': attention_weights.min().item(),
            'sigma_mean': sigma.mean().item(),
            'sigma_std': sigma.std().item(),
            'mu_norm': torch.norm(mu).item()
        }
        
        return losses


class AdversarialStructureLoss(nn.Module):
    """
    Adversarial training loss for structure embedder robustness.
    
    This loss encourages the model to be robust to input perturbations
    and maintain consistent predictions under small market noise.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.epsilon = config.get('adversarial_epsilon', 0.01)
        self.alpha = config.get('adversarial_alpha', 0.1)
        self.num_steps = config.get('adversarial_steps', 3)
        
    def forward(
        self,
        model: nn.Module,
        input_sequence: torch.Tensor,
        clean_mu: torch.Tensor,
        clean_sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adversarial loss using FGSM-style perturbations.
        
        Args:
            model: Structure embedder model
            input_sequence: Clean input [batch, seq_len, input_dim]
            clean_mu: Clean predictions [batch, output_dim]
            clean_sigma: Clean uncertainties [batch, output_dim]
            
        Returns:
            Adversarial consistency loss
        """
        # Generate adversarial perturbation
        perturbation = torch.zeros_like(input_sequence, requires_grad=True)
        
        for _ in range(self.num_steps):
            perturbed_input = input_sequence + perturbation
            
            # Forward pass through model
            adv_mu, adv_sigma = model(perturbed_input)
            
            # Compute loss w.r.t. clean predictions
            consistency_loss = F.mse_loss(adv_mu, clean_mu.detach())
            
            # Compute gradients
            grad = torch.autograd.grad(
                consistency_loss, 
                perturbation, 
                retain_graph=True,
                create_graph=True
            )[0]
            
            # Update perturbation
            perturbation = perturbation + self.alpha * grad.sign()
            perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
            perturbation = perturbation.detach().requires_grad_(True)
        
        # Final adversarial pass
        final_perturbed = input_sequence + perturbation
        final_mu, final_sigma = model(final_perturbed)
        
        # Compute final adversarial loss
        adv_loss = F.mse_loss(final_mu, clean_mu.detach())
        
        return adv_loss


class ContrastivePredictionLoss(nn.Module):
    """
    Contrastive learning loss for structure embedder.
    
    Encourages similar market structures to have similar embeddings
    while pushing different structures apart.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.temperature = config.get('contrastive_temperature', 0.1)
        self.margin = config.get('contrastive_margin', 1.0)
        
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embeddings: Structure embeddings [batch, embed_dim]
            labels: Market regime labels [batch]
            
        Returns:
            Contrastive loss
        """
        batch_size = embeddings.size(0)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create positive and negative masks
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(batch_size, device=mask.device)
        
        # Compute positive and negative similarities
        positive_mask = mask
        negative_mask = 1 - mask - torch.eye(batch_size, device=mask.device)
        
        # Compute loss
        positive_similarities = similarity_matrix * positive_mask
        negative_similarities = similarity_matrix * negative_mask
        
        # InfoNCE-style loss
        positive_sum = torch.sum(torch.exp(positive_similarities), dim=1)
        negative_sum = torch.sum(torch.exp(negative_similarities), dim=1)
        
        loss = -torch.log(positive_sum / (positive_sum + negative_sum + 1e-8))
        
        return loss.mean()


class UncertaintyCalibrationLoss(nn.Module):
    """
    Specialized loss for calibrating uncertainty estimates.
    
    Ensures that predicted uncertainties are well-calibrated with
    actual prediction errors across different confidence levels.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.n_bins = config.get('calibration_bins', 10)
        self.calibration_weight = config.get('calibration_weight', 1.0)
        
    def forward(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute uncertainty calibration loss.
        
        Args:
            predictions: Model predictions [batch, dim]
            uncertainties: Predicted uncertainties [batch, dim]
            targets: Ground truth targets [batch, dim]
            
        Returns:
            Dictionary with calibration losses and metrics
        """
        # Compute prediction errors
        errors = torch.abs(predictions - targets)
        
        # Create confidence bins
        confidence_scores = 1.0 / (1.0 + uncertainties)  # Higher confidence = lower uncertainty
        
        losses = {}
        
        # Bin-wise calibration
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=predictions.device)
        
        total_calibration_error = 0.0
        valid_bins = 0
        
        for i in range(self.n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find samples in this confidence bin
            in_bin = (confidence_scores >= bin_lower) & (confidence_scores < bin_upper)
            
            if in_bin.sum() > 0:
                bin_confidence = confidence_scores[in_bin].mean()
                bin_accuracy = (errors[in_bin] < uncertainties[in_bin]).float().mean()
                
                # Calibration error for this bin
                bin_error = torch.abs(bin_confidence - bin_accuracy)
                total_calibration_error += bin_error
                valid_bins += 1
        
        if valid_bins > 0:
            expected_calibration_error = total_calibration_error / valid_bins
        else:
            expected_calibration_error = torch.tensor(0.0, device=predictions.device)
        
        losses['calibration'] = expected_calibration_error * self.calibration_weight
        
        # Additional uncertainty metrics
        losses['metrics'] = {
            'ece': expected_calibration_error.item(),
            'mean_confidence': confidence_scores.mean().item(),
            'mean_uncertainty': uncertainties.mean().item(),
            'mean_error': errors.mean().item(),
            'error_uncertainty_corr': torch.corrcoef(
                torch.stack([errors.flatten(), uncertainties.flatten()])
            )[0, 1].item() if errors.numel() > 1 else 0.0
        }
        
        return losses


class MultiTaskStructureLoss(nn.Module):
    """
    Multi-task loss combining all structure embedder objectives.
    
    This is the main loss function used for training the structure embedder,
    combining prediction, uncertainty, attention, and calibration objectives.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.structure_loss = StructureEmbedderLoss(config.get('structure', {}))
        self.adversarial_loss = AdversarialStructureLoss(config.get('adversarial', {}))
        self.contrastive_loss = ContrastivePredictionLoss(config.get('contrastive', {}))
        self.calibration_loss = UncertaintyCalibrationLoss(config.get('calibration', {}))
        
        # Multi-task weights
        self.w_structure = config.get('structure_weight', 1.0)
        self.w_adversarial = config.get('adversarial_weight', 0.1)
        self.w_contrastive = config.get('contrastive_weight', 0.2)
        self.w_calibration = config.get('calibration_weight', 0.3)
        
        # Training phase controls
        self.enable_adversarial = config.get('enable_adversarial', False)
        self.enable_contrastive = config.get('enable_contrastive', False)
        
        logger.info(f"Initialized MultiTaskStructureLoss with weights: "
                   f"structure={self.w_structure}, adversarial={self.w_adversarial}, "
                   f"contrastive={self.w_contrastive}, calibration={self.w_calibration}")
        
    def forward(
        self,
        model: nn.Module,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        attention_weights: torch.Tensor,
        input_sequence: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        regime_labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            model: Structure embedder model
            mu: Predicted means [batch, output_dim]
            sigma: Predicted uncertainties [batch, output_dim]
            attention_weights: Attention weights [batch, seq_len]
            input_sequence: Input sequence [batch, seq_len, input_dim]
            targets: Optional targets for prediction [batch, output_dim]
            regime_labels: Optional regime labels for contrastive learning [batch]
            
        Returns:
            Dictionary containing all loss components
        """
        losses = {}
        
        # 1. Main structure loss
        structure_losses = self.structure_loss(
            mu, sigma, attention_weights, input_sequence, targets
        )
        losses.update(structure_losses)
        
        total_loss = self.w_structure * structure_losses['total']
        
        # 2. Adversarial loss (if enabled)
        if self.enable_adversarial:
            adv_loss = self.adversarial_loss(model, input_sequence, mu, sigma)
            losses['adversarial'] = adv_loss
            total_loss += self.w_adversarial * adv_loss
        
        # 3. Contrastive loss (if enabled and labels provided)
        if self.enable_contrastive and regime_labels is not None:
            cont_loss = self.contrastive_loss(mu, regime_labels)
            losses['contrastive'] = cont_loss
            total_loss += self.w_contrastive * cont_loss
        
        # 4. Calibration loss (if targets provided)
        if targets is not None:
            cal_losses = self.calibration_loss(mu, sigma, targets)
            losses['calibration'] = cal_losses['calibration']
            losses['calibration_metrics'] = cal_losses['metrics']
            total_loss += self.w_calibration * cal_losses['calibration']
        
        # Update total loss
        losses['total'] = total_loss
        
        return losses


class TacticalEmbedderLoss(nn.Module):
    """
    Specialized loss for training tactical embedder.
    
    Combines:
    1. Momentum prediction loss
    2. Uncertainty calibration loss
    3. Attention consistency loss
    4. State smoothness regularization
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.w_momentum = config.get('momentum_weight', 0.4)
        self.w_uncertainty = config.get('uncertainty_weight', 0.3)
        self.w_attention = config.get('attention_weight', 0.2)
        self.w_smoothness = config.get('smoothness_weight', 0.1)
        
    def forward(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        attention_weights: torch.Tensor,
        lstm_states: List[torch.Tensor],
        target_momentum: torch.Tensor,
        actual_volatility: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate tactical embedder loss.
        
        Args:
            mu: Mean predictions
            sigma: Uncertainty estimates
            attention_weights: Attention over time steps
            lstm_states: LSTM hidden states
            target_momentum: Target momentum for next period
            actual_volatility: Actual price volatility for calibration
        """
        losses = {}
        
        # 1. Momentum prediction loss
        momentum_loss = F.mse_loss(mu, target_momentum)
        losses['momentum'] = momentum_loss
        
        # 2. Uncertainty calibration
        if actual_volatility is not None:
            # Uncertainty should correlate with market volatility
            target_uncertainty = torch.sqrt(actual_volatility + 1e-6)
            uncertainty_loss = F.mse_loss(sigma.mean(dim=-1), target_uncertainty)
            losses['uncertainty'] = uncertainty_loss
        else:
            # Prevent uncertainty collapse
            losses['uncertainty'] = -torch.log(sigma).mean() * 0.01
            
        # 3. Attention consistency loss
        # Encourage smooth attention transitions
        attention_diff = torch.diff(attention_weights, dim=1)
        attention_smoothness = torch.mean(attention_diff ** 2)
        losses['attention'] = attention_smoothness
        
        # 4. State smoothness regularization
        if lstm_states:
            state_diffs = []
            for states in lstm_states:
                state_diff = torch.diff(states, dim=1)
                state_diffs.append(torch.mean(state_diff ** 2))
            
            state_smoothness = torch.stack(state_diffs).mean()
            losses['smoothness'] = state_smoothness
        else:
            losses['smoothness'] = torch.tensor(0.0)
            
        # Combined loss
        losses['total'] = (
            self.w_momentum * losses['momentum'] +
            self.w_uncertainty * losses['uncertainty'] +
            self.w_attention * losses['attention'] +
            self.w_smoothness * losses['smoothness']
        )
        
        return losses