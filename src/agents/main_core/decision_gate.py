"""
Decision Gate Implementation for Main MARL Core.

This module implements the two-gate decision system for the Main MARL Core.
The Decision Gate is the final step (Gate 2) that integrates risk proposals
from M-RMS and makes the ultimate EXECUTE/REJECT decision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class DecisionGate(nn.Module):
    """
    Final decision gate with risk integration.
    
    Takes the unified state plus risk proposal and makes the final
    execute/reject decision. This is Gate 2 in the two-gate system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Input: unified_state (512) + risk_vector (128) = 640
        self.input_dim = config.get('input_dim', 640)
        self.hidden_dim = config.get('hidden_dim', 256)
        
        # Risk encoder
        self.risk_encoder = RiskProposalEncoder(config)
        
        # Attention mechanism between state and risk
        self.state_risk_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Decision network
        self.decision_network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(self.hidden_dim // 2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            
            nn.Linear(64, 2)  # [execute, reject]
        )
        
        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Risk adjustment factors
        self.risk_threshold = config.get('risk_threshold', 0.3)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        
    def forward(
        self,
        unified_state: torch.Tensor,
        risk_proposal: Dict[str, Any],
        mc_consensus: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Make final execution decision.
        
        Args:
            unified_state: State vector from shared policy [batch, 512]
            risk_proposal: Risk management proposal from M-RMS
            mc_consensus: Consensus results from MC Dropout
            
        Returns:
            Dictionary with decision outputs
        """
        # Encode risk proposal
        risk_vector = self.risk_encoder(risk_proposal)  # [batch, 128]
        
        # Combine state and risk
        combined_state = torch.cat([unified_state, risk_vector], dim=-1)
        
        # Process through decision network
        decision_features = self.decision_network[:-1](combined_state)  # Get features before final layer
        decision_logits = self.decision_network[-1](decision_features)
        
        # Calculate confidence
        confidence = self.confidence_head(decision_features)
        
        # Apply risk-aware adjustments
        risk_score = risk_proposal['risk_metrics']['portfolio_risk_score']
        if risk_score > self.risk_threshold:
            # Increase rejection probability for high-risk situations
            risk_penalty = (risk_score - self.risk_threshold) * 2.0
            decision_logits[:, 1] += risk_penalty  # Increase reject logit
            
        # Get probabilities
        decision_probs = F.softmax(decision_logits, dim=-1)
        
        # Final decision logic
        execute_prob = decision_probs[:, 0]
        consensus_qualify = mc_consensus['should_qualify']
        confidence_met = confidence.squeeze() > self.confidence_threshold
        
        # Execute only if all conditions met
        should_execute = (
            consensus_qualify &  # Gate 1 passed
            (execute_prob > 0.6) &  # Gate 2 probability
            confidence_met &  # High confidence
            (risk_score < 0.8)  # Risk not too high
        )
        
        return {
            'decision_logits': decision_logits,
            'decision_probs': decision_probs,
            'execute_prob': execute_prob,
            'confidence': confidence,
            'should_execute': should_execute,
            'risk_adjusted': True,
            'decision_factors': {
                'gate1_passed': consensus_qualify,
                'gate2_prob': execute_prob,
                'confidence': confidence.squeeze(),
                'risk_score': risk_score
            }
        }


class RiskProposalEncoder(nn.Module):
    """Encode M-RMS risk proposal into vector representation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Risk feature dimensions
        self.position_encoder = nn.Linear(6, 32)  # position size, leverage, etc.
        self.stop_loss_encoder = nn.Linear(4, 24)  # stop distance, ATR multiple, etc.
        self.take_profit_encoder = nn.Linear(4, 24)  # TP levels, R:R ratio
        self.risk_metrics_encoder = nn.Linear(8, 48)  # various risk scores
        
        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(32 + 24 + 24 + 48, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
    def forward(self, risk_proposal: Dict[str, Any]) -> torch.Tensor:
        """Encode risk proposal into vector."""
        # Extract position sizing features
        position_features = torch.tensor([
            risk_proposal['position_size'],
            risk_proposal['position_size_pct'],
            risk_proposal['leverage'],
            risk_proposal['dollar_risk'],
            risk_proposal['portfolio_heat'],
            risk_proposal.get('correlation_adjustment', 1.0)
        ]).unsqueeze(0)
        position_encoded = self.position_encoder(position_features)
        
        # Stop loss features
        sl_features = torch.tensor([
            risk_proposal['stop_loss_distance'],
            risk_proposal['stop_loss_atr_multiple'],
            float(risk_proposal['use_trailing_stop']),
            risk_proposal.get('stop_confidence', 0.8)
        ]).unsqueeze(0)
        sl_encoded = self.stop_loss_encoder(sl_features)
        
        # Take profit features
        tp_features = torch.tensor([
            risk_proposal['take_profit_distance'],
            risk_proposal['risk_reward_ratio'],
            risk_proposal['expected_return'],
            risk_proposal.get('tp_confidence', 0.7)
        ]).unsqueeze(0)
        tp_encoded = self.take_profit_encoder(tp_features)
        
        # Risk metrics
        risk_features = torch.tensor([
            risk_proposal['risk_metrics']['portfolio_risk_score'],
            risk_proposal['risk_metrics']['correlation_risk'],
            risk_proposal['risk_metrics']['concentration_risk'],
            risk_proposal['risk_metrics']['market_risk_multiplier'],
            risk_proposal['confidence_scores']['overall_confidence'],
            risk_proposal['confidence_scores']['sl_confidence'],
            risk_proposal['confidence_scores']['tp_confidence'],
            risk_proposal['confidence_scores']['size_confidence']
        ]).unsqueeze(0)
        risk_encoded = self.risk_metrics_encoder(risk_features)
        
        # Combine all features
        combined = torch.cat([
            position_encoded,
            sl_encoded,
            tp_encoded,
            risk_encoded
        ], dim=-1)
        
        return self.projection(combined)


class RiskAwareDecisionModule(nn.Module):
    """
    Advanced risk-aware decision module.
    
    Incorporates multiple risk factors and market conditions
    to make sophisticated execution decisions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Multi-scale risk assessment
        self.portfolio_risk_head = nn.Linear(128, 1)
        self.market_risk_head = nn.Linear(128, 1)
        self.execution_risk_head = nn.Linear(128, 1)
        
        # Dynamic threshold computation
        self.threshold_network = nn.Sequential(
            nn.Linear(640, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Risk factor weights
        self.risk_weights = nn.Parameter(torch.tensor([0.4, 0.3, 0.3]))  # portfolio, market, execution
        
    def forward(
        self,
        combined_state: torch.Tensor,
        risk_vector: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute risk-aware decision metrics.
        
        Args:
            combined_state: Combined state vector
            risk_vector: Encoded risk proposal
            
        Returns:
            Risk assessment dictionary
        """
        # Multi-scale risk assessment
        portfolio_risk = torch.sigmoid(self.portfolio_risk_head(risk_vector))
        market_risk = torch.sigmoid(self.market_risk_head(risk_vector))
        execution_risk = torch.sigmoid(self.execution_risk_head(risk_vector))
        
        # Weighted risk score
        risk_scores = torch.stack([portfolio_risk, market_risk, execution_risk], dim=-1)
        overall_risk = torch.sum(risk_scores * self.risk_weights, dim=-1)
        
        # Dynamic threshold based on market conditions
        dynamic_threshold = self.threshold_network(combined_state)
        
        # Risk-adjusted recommendation
        risk_adjusted_accept = overall_risk < dynamic_threshold
        
        return {
            'portfolio_risk': portfolio_risk,
            'market_risk': market_risk,
            'execution_risk': execution_risk,
            'overall_risk': overall_risk,
            'dynamic_threshold': dynamic_threshold,
            'risk_adjusted_accept': risk_adjusted_accept,
            'risk_margin': dynamic_threshold - overall_risk
        }


class AdaptiveDecisionGate(DecisionGate):
    """
    Adaptive version of DecisionGate that learns from outcomes.
    
    Incorporates feedback mechanisms to improve decision quality over time.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Outcome tracking
        self.outcome_buffer = []
        self.adaptation_rate = config.get('adaptation_rate', 0.01)
        
        # Adaptive components
        self.risk_aware_module = RiskAwareDecisionModule(config)
        
        # Performance tracking
        self.decision_accuracy = 0.0
        self.total_decisions = 0
        
    def update_from_outcome(self, decision_data: Dict[str, Any], outcome: Dict[str, Any]):
        """
        Update decision parameters based on trade outcomes.
        
        Args:
            decision_data: Original decision context
            outcome: Trade outcome (profit/loss, execution quality, etc.)
        """
        # Track outcome
        self.outcome_buffer.append({
            'decision': decision_data,
            'outcome': outcome,
            'timestamp': torch.tensor(outcome.get('timestamp', 0))
        })
        
        # Keep buffer size manageable
        if len(self.outcome_buffer) > 1000:
            self.outcome_buffer.pop(0)
            
        # Update accuracy metrics
        was_correct = self._evaluate_decision_quality(decision_data, outcome)
        self.total_decisions += 1
        
        # Running average of accuracy
        self.decision_accuracy = (
            self.decision_accuracy * (self.total_decisions - 1) + was_correct
        ) / self.total_decisions
        
        # Adaptive threshold adjustment
        if len(self.outcome_buffer) >= 50:
            self._adapt_thresholds()
            
    def _evaluate_decision_quality(self, decision: Dict[str, Any], outcome: Dict[str, Any]) -> float:
        """
        Evaluate the quality of a decision based on outcome.
        
        Returns:
            Quality score between 0 and 1
        """
        if decision['should_execute']:
            # For executed trades, evaluate based on P&L and risk metrics
            pnl = outcome.get('pnl', 0)
            risk_taken = outcome.get('risk_taken', 1)
            
            # Risk-adjusted return
            if risk_taken > 0:
                risk_adjusted_return = pnl / risk_taken
                # Good decision if risk-adjusted return > 0
                return 1.0 if risk_adjusted_return > 0 else 0.0
            else:
                return 0.5  # Neutral for zero risk
        else:
            # For rejected trades, check if it was actually profitable
            opportunity_pnl = outcome.get('hypothetical_pnl', 0)
            # Good decision if we avoided a loss
            return 1.0 if opportunity_pnl <= 0 else 0.0
            
    def _adapt_thresholds(self):
        """Adapt decision thresholds based on recent performance."""
        if len(self.outcome_buffer) < 50:
            return
            
        # Analyze recent performance
        recent_outcomes = self.outcome_buffer[-50:]
        execution_rate = sum(1 for o in recent_outcomes if o['decision']['should_execute']) / len(recent_outcomes)
        
        # Adjust thresholds based on performance
        if self.decision_accuracy < 0.6:
            # Too many bad decisions, be more conservative
            self.confidence_threshold = min(0.8, self.confidence_threshold + self.adaptation_rate)
            self.risk_threshold = max(0.2, self.risk_threshold - self.adaptation_rate)
        elif self.decision_accuracy > 0.8 and execution_rate < 0.1:
            # High accuracy but low execution, be more aggressive
            self.confidence_threshold = max(0.6, self.confidence_threshold - self.adaptation_rate)
            self.risk_threshold = min(0.4, self.risk_threshold + self.adaptation_rate)
            
        logger.info(
            f"Adapted decision thresholds",
            accuracy=self.decision_accuracy,
            execution_rate=execution_rate,
            confidence_threshold=self.confidence_threshold,
            risk_threshold=self.risk_threshold
        )