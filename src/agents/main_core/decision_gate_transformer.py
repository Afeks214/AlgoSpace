"""
File: src/agents/main_core/decision_gate_transformer.py (NEW FILE)
Complete DecisionGate Transformer implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
import logging
import math
import time

logger = logging.getLogger(__name__)


@dataclass
class DecisionOutput:
    """Structured output from DecisionGate."""
    decision: str  # 'EXECUTE' or 'REJECT'
    confidence: float
    execute_probability: float
    risk_score: float
    validation_scores: Dict[str, float]
    attention_weights: Optional[torch.Tensor]
    threshold_used: float
    decision_factors: Dict[str, Any]
    safety_checks: Dict[str, bool]


class MultiFactorValidation(nn.Module):
    """Multi-factor validation module for comprehensive checks."""
    
    def __init__(self, hidden_dim: int = 384):
        super().__init__()
        
        # Risk validation heads
        self.risk_validator = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 4)  # portfolio_heat, correlation, concentration, leverage
        )
        
        # Market conditions validator
        self.market_validator = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 4)  # volatility, liquidity, regime_alignment, momentum
        )
        
        # Technical validity validator
        self.technical_validator = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 4)  # entry_quality, signal_strength, pattern_clarity, timing
        )
        
        # Aggregation network
        self.aggregator = nn.Sequential(
            nn.Linear(12, 24),  # 4 + 4 + 4 validation scores
            nn.ReLU(),
            nn.Linear(24, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform multi-factor validation.
        
        Returns:
            Tuple of (aggregated_score, individual_scores)
        """
        # Individual validations
        risk_scores = torch.sigmoid(self.risk_validator(features))
        market_scores = torch.sigmoid(self.market_validator(features))
        technical_scores = torch.sigmoid(self.technical_validator(features))
        
        # Concatenate all scores
        all_scores = torch.cat([risk_scores, market_scores, technical_scores], dim=-1)
        
        # Aggregate
        final_score = self.aggregator(all_scores)
        
        scores_dict = {
            'risk': risk_scores,
            'market': market_scores,
            'technical': technical_scores,
            'all': all_scores
        }
        
        return final_score, scores_dict


class RiskAwareAttention(nn.Module):
    """Cross-attention mechanism between state and risk features."""
    
    def __init__(self, state_dim: int = 512, risk_dim: int = 128, hidden_dim: int = 384):
        super().__init__()
        
        # Project to common dimension
        self.state_projection = nn.Linear(state_dim, hidden_dim)
        self.risk_projection = nn.Linear(risk_dim, hidden_dim)
        
        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Risk importance gates
        self.risk_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(risk_dim, hidden_dim),
                nn.Sigmoid()
            ) for _ in range(4)  # 4 different risk aspects
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self, 
        state_features: torch.Tensor, 
        risk_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply risk-aware cross-attention.
        
        Args:
            state_features: Unified state [batch, state_dim]
            risk_features: Risk proposal [batch, risk_dim]
            
        Returns:
            Tuple of (attended_features, attention_weights)
        """
        # Project features
        state_proj = self.state_projection(state_features).unsqueeze(1)  # [batch, 1, hidden]
        risk_proj = self.risk_projection(risk_features).unsqueeze(1)    # [batch, 1, hidden]
        
        # Apply risk gates for importance weighting
        gated_risks = []
        for gate in self.risk_gates:
            gate_weight = gate(risk_features).unsqueeze(1)
            gated_risk = risk_proj * gate_weight
            gated_risks.append(gated_risk)
            
        # Stack gated risks
        risk_sequence = torch.cat([risk_proj] + gated_risks, dim=1)  # [batch, 5, hidden]
        
        # Cross-attention: state attends to risk
        attended, attention_weights = self.cross_attention(
            query=state_proj,
            key=risk_sequence,
            value=risk_sequence
        )
        
        # Output projection
        output = self.output_projection(attended.squeeze(1))
        
        return output, attention_weights


class DynamicThresholdLayer(nn.Module):
    """Dynamic threshold adjustment based on context."""
    
    def __init__(self, hidden_dim: int = 384):
        super().__init__()
        
        # Base threshold predictor
        self.base_threshold = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Context adjusters
        self.regime_adjuster = nn.Sequential(
            nn.Linear(hidden_dim + 4, 32),  # +4 for regime one-hot
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # Can adjust ±0.1
        )
        
        self.risk_adjuster = nn.Sequential(
            nn.Linear(hidden_dim + 8, 32),  # +8 for risk metrics
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # Can adjust ±0.1
        )
        
        self.uncertainty_adjuster = nn.Sequential(
            nn.Linear(hidden_dim + 3, 32),  # +3 for uncertainty metrics
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # Can adjust ±0.1
        )
        
    def forward(
        self,
        features: torch.Tensor,
        regime_context: torch.Tensor,
        risk_metrics: torch.Tensor,
        uncertainty_metrics: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate dynamic threshold.
        
        Returns:
            Tuple of (threshold, adjustment_breakdown)
        """
        # Base threshold
        base = self.base_threshold(features) * 0.3 + 0.6  # Range: 0.6-0.9
        
        # Context adjustments
        regime_adj = self.regime_adjuster(
            torch.cat([features, regime_context], dim=-1)
        ) * 0.1
        
        risk_adj = self.risk_adjuster(
            torch.cat([features, risk_metrics], dim=-1)
        ) * 0.1
        
        uncertainty_adj = self.uncertainty_adjuster(
            torch.cat([features, uncertainty_metrics], dim=-1)
        ) * 0.1
        
        # Final threshold
        threshold = base + regime_adj + risk_adj + uncertainty_adj
        threshold = torch.clamp(threshold, 0.5, 0.95)
        
        adjustments = {
            'base': base.item(),
            'regime_adjustment': regime_adj.item(),
            'risk_adjustment': risk_adj.item(),
            'uncertainty_adjustment': uncertainty_adj.item(),
            'final': threshold.item()
        }
        
        return threshold, adjustments


class DecisionGateTransformer(nn.Module):
    """
    State-of-the-art DecisionGate Transformer for final trade validation.
    
    Combines transformer architecture with risk-aware attention, multi-factor
    validation, and dynamic thresholding for sophisticated decision making.
    
    Architecture:
        1. Input projection and encoding
        2. Risk-aware transformer layers
        3. Cross-attention with risk proposal
        4. Multi-factor validation
        5. Dynamic threshold calculation
        6. Final decision head
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Configuration
        self.state_dim = config.get('state_dim', 512)
        self.risk_dim = config.get('risk_dim', 128)
        self.hidden_dim = config.get('hidden_dim', 384)
        self.n_layers = config.get('n_layers', 4)
        self.n_heads = config.get('n_heads', 8)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        
        # Input processing
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        self.risk_encoder = nn.Sequential(
            nn.Linear(self.risk_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # Positional encoding for state components
        self.positional_encoding = self._create_positional_encoding()
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.n_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout_rate,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layers
        )
        
        # Risk-aware attention
        self.risk_attention = RiskAwareAttention(
            state_dim=self.hidden_dim,
            risk_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim
        )
        
        # Multi-factor validation
        self.validator = MultiFactorValidation(self.hidden_dim)
        
        # Dynamic threshold
        self.threshold_layer = DynamicThresholdLayer(self.hidden_dim)
        
        # Final decision head
        self.decision_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 1 + 12, 256),  # +1 for validation score, +12 for factors
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            
            nn.Linear(64, 2)  # [EXECUTE, REJECT]
        )
        
        # Safety mechanisms
        self.safety_checks = SafetyValidator(config.get('safety', {}))
        
        # Initialize weights
        self._init_weights()
        
    def _create_positional_encoding(self) -> nn.Parameter:
        """Create learnable positional encoding."""
        # Positions for different state components
        n_positions = 5  # structure, tactical, regime, lvn, risk
        encoding = torch.randn(1, n_positions, self.hidden_dim) * 0.02
        return nn.Parameter(encoding)
        
    def _init_weights(self):
        """Initialize weights with careful scaling."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
    def forward(
        self,
        unified_state: torch.Tensor,
        risk_proposal: Dict[str, Any],
        mc_consensus: Dict[str, torch.Tensor],
        market_context: Optional[Dict[str, Any]] = None
    ) -> DecisionOutput:
        """
        Make final execution decision.
        
        Args:
            unified_state: Combined state vector [batch, state_dim]
            risk_proposal: Risk parameters from M-RMS
            mc_consensus: MC Dropout consensus results
            market_context: Current market conditions
            
        Returns:
            DecisionOutput with comprehensive decision information
        """
        batch_size = unified_state.size(0)
        
        # 1. Encode inputs
        state_encoded = self.state_encoder(unified_state)
        risk_vector = self._encode_risk_proposal(risk_proposal)
        risk_encoded = self.risk_encoder(risk_vector)
        
        # 2. Create sequence with positional encoding
        # Split state into components (approximate)
        state_components = self._split_state_components(state_encoded)
        
        # Add positional encoding
        sequence = torch.stack(state_components, dim=1)  # [batch, 5, hidden]
        sequence = sequence + self.positional_encoding
        
        # 3. Apply transformer encoder
        transformer_out = self.transformer_encoder(sequence)
        
        # Pool transformer output
        pooled = transformer_out.mean(dim=1)  # [batch, hidden]
        
        # 4. Risk-aware cross-attention
        risk_attended, risk_attention_weights = self.risk_attention(
            pooled,
            risk_encoded
        )
        
        # Combine features
        combined_features = pooled + risk_attended
        
        # 5. Multi-factor validation
        validation_score, validation_breakdown = self.validator(combined_features)
        
        # 6. Extract context tensors
        regime_context = self._encode_regime(market_context)
        risk_metrics = self._extract_risk_metrics(risk_proposal)
        uncertainty_metrics = self._extract_uncertainty_metrics(mc_consensus)
        
        # 7. Calculate dynamic threshold
        threshold, threshold_breakdown = self.threshold_layer(
            combined_features,
            regime_context,
            risk_metrics,
            uncertainty_metrics
        )
        
        # 8. Prepare final decision input
        decision_input = torch.cat([
            combined_features,
            validation_score,
            validation_breakdown['all']
        ], dim=-1)
        
        # 9. Final decision
        decision_logits = self.decision_head(decision_input)
        decision_probs = F.softmax(decision_logits, dim=-1)
        
        execute_prob = decision_probs[:, 0]
        
        # 10. Safety checks
        safety_results = self.safety_checks.validate(
            risk_proposal,
            mc_consensus,
            market_context
        )
        
        # 11. Make final decision
        should_execute = (
            execute_prob > threshold and
            validation_score > 0.6 and
            all(safety_results.values()) and
            mc_consensus['should_qualify']
        )
        
        # Create comprehensive output
        return DecisionOutput(
            decision='EXECUTE' if should_execute else 'REJECT',
            confidence=execute_prob.item(),
            execute_probability=execute_prob.item(),
            risk_score=risk_metrics.mean().item(),
            validation_scores={
                'overall': validation_score.item(),
                'risk': validation_breakdown['risk'].mean().item(),
                'market': validation_breakdown['market'].mean().item(),
                'technical': validation_breakdown['technical'].mean().item()
            },
            attention_weights=risk_attention_weights,
            threshold_used=threshold.item(),
            decision_factors={
                'mc_consensus_passed': mc_consensus['should_qualify'].item(),
                'validation_passed': validation_score.item() > 0.6,
                'threshold_passed': execute_prob.item() > threshold.item(),
                'safety_passed': all(safety_results.values()),
                'threshold_breakdown': threshold_breakdown
            },
            safety_checks=safety_results
        )
        
    def _encode_risk_proposal(self, risk_proposal: Dict[str, Any]) -> torch.Tensor:
        """Encode risk proposal into tensor."""
        features = []
        
        # Position sizing
        features.extend([
            risk_proposal['position_size'] / 1000,
            risk_proposal['position_size_pct'],
            risk_proposal['leverage'],
            risk_proposal['dollar_risk'] / 10000,
            risk_proposal['portfolio_heat']
        ])
        
        # Stop loss
        features.extend([
            risk_proposal['stop_loss_distance'] / 100,
            risk_proposal['stop_loss_atr_multiple'],
            float(risk_proposal['use_trailing_stop'])
        ])
        
        # Take profit
        features.extend([
            risk_proposal['take_profit_distance'] / 100,
            risk_proposal['risk_reward_ratio'],
            risk_proposal['expected_return'] / 1000
        ])
        
        # Risk metrics
        risk_metrics = risk_proposal['risk_metrics']
        features.extend([
            risk_metrics['portfolio_risk_score'],
            risk_metrics['correlation_risk'],
            risk_metrics['concentration_risk'],
            risk_metrics['market_risk_multiplier']
        ])
        
        # Confidence scores
        conf_scores = risk_proposal['confidence_scores']
        features.extend([
            conf_scores['overall_confidence'],
            conf_scores['sl_confidence'],
            conf_scores['tp_confidence'],
            conf_scores['size_confidence']
        ])
        
        # Pad to risk_dim
        while len(features) < self.risk_dim:
            features.append(0.0)
            
        return torch.tensor(features[:self.risk_dim]).unsqueeze(0).to(next(self.parameters()).device)
        
    def _split_state_components(self, state: torch.Tensor) -> List[torch.Tensor]:
        """Split encoded state into components."""
        # This is approximate - in practice would map to actual components
        chunk_size = self.hidden_dim
        components = []
        
        # Create 5 components (structure, tactical, regime, lvn, synergy)
        for i in range(5):
            # Use different projections of the state
            weight = torch.randn(self.hidden_dim, self.hidden_dim).to(state.device) * 0.1
            component = torch.matmul(state, weight)
            components.append(component)
            
        return components
        
    def _encode_regime(self, market_context: Optional[Dict[str, Any]]) -> torch.Tensor:
        """Encode market regime as one-hot."""
        if not market_context:
            # Default: unknown regime
            return torch.zeros(1, 4).to(next(self.parameters()).device)
            
        regime = market_context.get('regime', 'unknown')
        regime_map = {
            'trending': 0,
            'volatile': 1,
            'ranging': 2,
            'transitioning': 3
        }
        
        regime_idx = regime_map.get(regime, 3)
        regime_tensor = torch.zeros(1, 4)
        regime_tensor[0, regime_idx] = 1.0
        
        return regime_tensor.to(next(self.parameters()).device)
        
    def _extract_risk_metrics(self, risk_proposal: Dict[str, Any]) -> torch.Tensor:
        """Extract key risk metrics."""
        metrics = risk_proposal['risk_metrics']
        
        risk_tensor = torch.tensor([
            metrics['portfolio_risk_score'],
            metrics['correlation_risk'],
            metrics['concentration_risk'],
            metrics['market_risk_multiplier'],
            risk_proposal['portfolio_heat'],
            risk_proposal['leverage'],
            risk_proposal['dollar_risk'] / 10000,
            risk_proposal['risk_reward_ratio']
        ]).unsqueeze(0)
        
        return risk_tensor.to(next(self.parameters()).device)
        
    def _extract_uncertainty_metrics(self, mc_consensus: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract uncertainty metrics from MC consensus."""
        uncertainty_tensor = torch.tensor([
            mc_consensus.get('entropy', 0.5),
            mc_consensus.get('epistemic_uncertainty', 0.3),
            mc_consensus.get('aleatoric_uncertainty', 0.2)
        ]).unsqueeze(0)
        
        return uncertainty_tensor.to(next(self.parameters()).device)


class SafetyValidator:
    """Production safety checks."""
    
    def __init__(self, config: Dict[str, Any]):
        self.max_position_size = config.get('max_position_size', 10000)
        self.max_portfolio_heat = config.get('max_portfolio_heat', 0.20)
        self.max_correlation = config.get('max_correlation', 0.70)
        self.min_risk_reward = config.get('min_risk_reward', 1.5)
        self.max_daily_trades = config.get('max_daily_trades', 10)
        
        # Track daily trades
        self.daily_trades = []
        
    def validate(
        self,
        risk_proposal: Dict[str, Any],
        mc_consensus: Dict[str, torch.Tensor],
        market_context: Optional[Dict[str, Any]]
    ) -> Dict[str, bool]:
        """Perform safety validation."""
        checks = {}
        
        # Position size check
        checks['position_size_valid'] = (
            risk_proposal['position_size'] <= self.max_position_size
        )
        
        # Portfolio heat check
        checks['portfolio_heat_valid'] = (
            risk_proposal['portfolio_heat'] <= self.max_portfolio_heat
        )
        
        # Correlation check
        checks['correlation_valid'] = (
            risk_proposal['risk_metrics']['correlation_risk'] <= self.max_correlation
        )
        
        # Risk-reward check
        checks['risk_reward_valid'] = (
            risk_proposal['risk_reward_ratio'] >= self.min_risk_reward
        )
        
        # Daily trade limit
        today_trades = [t for t in self.daily_trades if self._is_today(t)]
        checks['daily_limit_valid'] = len(today_trades) < self.max_daily_trades
        
        # Market conditions check
        if market_context:
            regime = market_context.get('regime', 'unknown')
            checks['regime_valid'] = regime != 'unknown'
            
            volatility = market_context.get('volatility', 1.0)
            checks['volatility_valid'] = volatility < 3.0  # Extreme volatility check
        else:
            checks['regime_valid'] = True
            checks['volatility_valid'] = True
            
        # Confidence check
        checks['confidence_valid'] = mc_consensus.get('qualify_prob', 0) > 0.5
        
        return checks
        
    def _is_today(self, trade_time) -> bool:
        """Check if trade was today."""
        from datetime import datetime, timedelta
        return datetime.now() - trade_time < timedelta(days=1)