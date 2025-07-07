"""
LVN-specific loss functions for training the advanced LVN embedder.

This module provides specialized loss functions designed to optimize
the LVN embedder's ability to identify and prioritize support/resistance levels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
import numpy as np


class LVNRelevanceLoss(nn.Module):
    """
    Loss function for training LVN relevance scoring.
    
    Encourages the model to assign high relevance scores to levels
    that are subsequently tested by price action.
    """
    
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
        
    def forward(
        self,
        relevance_scores: torch.Tensor,
        actual_interactions: torch.Tensor,
        distances: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate relevance loss.
        
        Args:
            relevance_scores: Predicted relevance [batch, n_levels]
            actual_interactions: Binary interaction indicators [batch, n_levels]
            distances: Distance from current price [batch, n_levels]
            
        Returns:
            Scalar loss value
        """
        # Weight interactions by inverse distance (closer levels more important)
        distance_weights = 1.0 / (1.0 + distances)
        
        # Binary cross-entropy for interaction prediction
        interaction_loss = F.binary_cross_entropy(
            relevance_scores,
            actual_interactions.float(),
            reduction='none'
        )
        
        # Apply distance weighting
        weighted_loss = interaction_loss * distance_weights
        
        # Add margin-based ranking loss
        positive_indices = actual_interactions == 1
        negative_indices = actual_interactions == 0
        
        if positive_indices.any() and negative_indices.any():
            positive_scores = relevance_scores[positive_indices]
            negative_scores = relevance_scores[negative_indices]
            
            # Ensure positive examples have higher scores
            ranking_loss = F.relu(
                negative_scores.mean() - positive_scores.mean() + self.margin
            )
            
            return weighted_loss.mean() + ranking_loss
        else:
            return weighted_loss.mean()


class LVNInteractionLoss(nn.Module):
    """
    Loss function for predicting price-LVN interactions.
    
    Multi-class classification for bounce/break/no_interaction.
    """
    
    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.class_weights = class_weights or torch.tensor([1.0, 2.0, 0.5])
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        strengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate interaction prediction loss.
        
        Args:
            predictions: Predicted interactions [batch, n_levels, 3]
            targets: Actual interactions [batch, n_levels]
            strengths: LVN strength scores [batch, n_levels]
            
        Returns:
            Scalar loss value
        """
        batch_size, n_levels, n_classes = predictions.shape
        
        # Reshape for cross-entropy
        predictions_flat = predictions.view(-1, n_classes)
        targets_flat = targets.view(-1)
        strengths_flat = strengths.view(-1)
        
        # Calculate cross-entropy loss
        ce_loss = F.cross_entropy(
            predictions_flat,
            targets_flat,
            weight=self.class_weights.to(predictions.device),
            reduction='none'
        )
        
        # Weight by LVN strength (stronger levels more important)
        strength_weights = strengths_flat / 100.0  # Normalize to [0, 1]
        weighted_loss = ce_loss * (0.5 + 0.5 * strength_weights)  # Min weight 0.5
        
        return weighted_loss.mean()


class LVNSpatialConsistencyLoss(nn.Module):
    """
    Encourages spatial consistency in LVN embeddings.
    
    Similar LVN levels should have similar embeddings.
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self,
        embeddings: torch.Tensor,
        prices: torch.Tensor,
        price_threshold: float = 1.0
    ) -> torch.Tensor:
        """
        Calculate spatial consistency loss.
        
        Args:
            embeddings: LVN embeddings [batch, n_levels, hidden_dim]
            prices: LVN price levels [batch, n_levels]
            price_threshold: Price difference threshold for similarity
            
        Returns:
            Scalar loss value
        """
        batch_size, n_levels, hidden_dim = embeddings.shape
        
        # Calculate pairwise price differences
        price_diff = prices.unsqueeze(2) - prices.unsqueeze(1)  # [batch, n_levels, n_levels]
        
        # Create similarity mask (1 if similar, 0 otherwise)
        similarity_mask = (torch.abs(price_diff) < price_threshold).float()
        
        # Calculate pairwise embedding similarities
        embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
        embedding_sim = torch.bmm(embeddings_norm, embeddings_norm.transpose(1, 2))
        
        # Apply temperature scaling
        embedding_sim = embedding_sim / self.temperature
        
        # Calculate contrastive loss
        # Similar prices should have high embedding similarity
        positive_loss = -torch.log(torch.sigmoid(embedding_sim)) * similarity_mask
        negative_loss = -torch.log(1 - torch.sigmoid(embedding_sim)) * (1 - similarity_mask)
        
        # Mask out diagonal (self-similarity)
        mask = torch.eye(n_levels, device=embeddings.device).unsqueeze(0)
        positive_loss = positive_loss * (1 - mask)
        negative_loss = negative_loss * (1 - mask)
        
        total_loss = positive_loss + negative_loss
        
        # Average over valid pairs
        valid_pairs = (1 - mask).sum()
        return total_loss.sum() / (valid_pairs + 1e-8)


class LVNTemporalConsistencyLoss(nn.Module):
    """
    Encourages temporal consistency in LVN predictions.
    
    Predictions should evolve smoothly over time.
    """
    
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        
    def forward(
        self,
        current_relevance: torch.Tensor,
        previous_relevance: torch.Tensor,
        price_change: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate temporal consistency loss.
        
        Args:
            current_relevance: Current relevance scores [batch, n_levels]
            previous_relevance: Previous relevance scores [batch, n_levels]
            price_change: Normalized price change since last update
            
        Returns:
            Scalar loss value
        """
        # Expected change based on price movement
        expected_change_factor = 1.0 - torch.exp(-torch.abs(price_change))
        
        # Actual change in relevance
        relevance_change = torch.abs(current_relevance - previous_relevance)
        
        # Penalize large changes when price hasn't moved much
        consistency_loss = relevance_change * (1 - expected_change_factor)
        
        # Also penalize no change when price has moved significantly
        stagnation_loss = (1 - relevance_change) * expected_change_factor
        
        total_loss = self.alpha * consistency_loss + (1 - self.alpha) * stagnation_loss
        
        return total_loss.mean()


class LVNCompositeLoss(nn.Module):
    """
    Composite loss function combining all LVN-specific losses.
    """
    
    def __init__(
        self,
        relevance_weight: float = 1.0,
        interaction_weight: float = 1.0,
        spatial_weight: float = 0.5,
        temporal_weight: float = 0.3
    ):
        super().__init__()
        
        self.relevance_loss = LVNRelevanceLoss()
        self.interaction_loss = LVNInteractionLoss()
        self.spatial_loss = LVNSpatialConsistencyLoss()
        self.temporal_loss = LVNTemporalConsistencyLoss()
        
        self.weights = {
            'relevance': relevance_weight,
            'interaction': interaction_weight,
            'spatial': spatial_weight,
            'temporal': temporal_weight
        }
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        context: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate composite loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            context: Additional context (prices, distances, etc.)
            
        Returns:
            total_loss: Scalar total loss
            loss_components: Dictionary of individual loss values
        """
        losses = {}
        
        # Relevance loss
        if 'relevance_scores' in predictions and 'actual_interactions' in targets:
            losses['relevance'] = self.relevance_loss(
                predictions['relevance_scores'],
                targets['actual_interactions'],
                context['distances']
            )
            
        # Interaction loss
        if 'interaction_predictions' in predictions and 'interaction_targets' in targets:
            losses['interaction'] = self.interaction_loss(
                predictions['interaction_predictions'],
                targets['interaction_targets'],
                context['strengths']
            )
            
        # Spatial consistency loss
        if 'spatial_features' in predictions:
            losses['spatial'] = self.spatial_loss(
                predictions['spatial_features'],
                context['prices']
            )
            
        # Temporal consistency loss
        if 'relevance_scores' in predictions and 'previous_relevance' in context:
            losses['temporal'] = self.temporal_loss(
                predictions['relevance_scores'],
                context['previous_relevance'],
                context['price_change']
            )
            
        # Calculate weighted total
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        loss_components = {}
        
        for name, loss in losses.items():
            weighted_loss = self.weights[name] * loss
            total_loss = total_loss + weighted_loss
            loss_components[f'lvn_{name}_loss'] = loss.item()
            
        return total_loss, loss_components


class LVNUncertaintyLoss(nn.Module):
    """
    Loss function for uncertainty quantification in LVN predictions.
    
    Encourages calibrated uncertainty estimates.
    """
    
    def __init__(self, beta: float = 0.5):
        super().__init__()
        self.beta = beta
        
    def forward(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Calculate uncertainty-aware loss.
        
        Args:
            predictions: Mean predictions
            uncertainties: Predicted uncertainties (std or variance)
            targets: Ground truth values
            eps: Small constant for numerical stability
            
        Returns:
            Scalar loss value
        """
        # Negative log-likelihood with predicted uncertainty
        diff = predictions - targets
        
        # Ensure positive uncertainty
        uncertainties = F.softplus(uncertainties) + eps
        
        # NLL loss
        nll_loss = 0.5 * torch.log(2 * np.pi * uncertainties) + \
                   0.5 * (diff ** 2) / uncertainties
        
        # Regularization to prevent uncertainty collapse
        uncertainty_reg = -self.beta * torch.log(uncertainties)
        
        return (nll_loss + uncertainty_reg).mean()