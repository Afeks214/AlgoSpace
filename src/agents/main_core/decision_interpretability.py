"""
File: src/agents/main_core/decision_interpretability.py (NEW FILE)
Interpretability tools for DecisionGate
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import pandas as pd


class DecisionInterpreter:
    """
    Interprets DecisionGate decisions for explainability.
    
    Provides:
    1. Attention visualizations
    2. Factor importance analysis
    3. Decision path tracking
    4. Counterfactual analysis
    """
    
    def __init__(self, model: 'DecisionGateTransformer'):
        self.model = model
        self.decision_paths = []
        
    def interpret_decision(
        self,
        decision_output: 'DecisionOutput',
        unified_state: torch.Tensor,
        risk_proposal: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive interpretation of a decision.
        
        Returns:
            Dictionary with interpretability insights
        """
        interpretation = {
            'decision': decision_output.decision,
            'confidence': decision_output.confidence,
            'factors': self._analyze_decision_factors(decision_output),
            'attention': self._analyze_attention_patterns(decision_output),
            'risk_impact': self._analyze_risk_impact(risk_proposal, decision_output),
            'counterfactuals': self._generate_counterfactuals(
                unified_state, 
                risk_proposal,
                decision_output
            ),
            'feature_importance': self._calculate_feature_importance(
                unified_state,
                risk_proposal
            )
        }
        
        return interpretation
        
    def _analyze_decision_factors(
        self, 
        decision_output: 'DecisionOutput'
    ) -> Dict[str, Any]:
        """Analyze key factors in decision."""
        factors = decision_output.decision_factors
        
        # Identify critical factors
        critical_factors = []
        
        # Check each validation score
        for key, score in decision_output.validation_scores.items():
            if score < 0.6:  # Below threshold
                critical_factors.append({
                    'factor': key,
                    'score': score,
                    'status': 'critical',
                    'impact': 'negative'
                })
            elif score > 0.8:  # Strong positive
                critical_factors.append({
                    'factor': key,
                    'score': score,
                    'status': 'strong',
                    'impact': 'positive'
                })
                
        # Threshold analysis
        threshold_margin = decision_output.execute_probability - decision_output.threshold_used
        
        return {
            'critical_factors': critical_factors,
            'threshold_margin': threshold_margin,
            'decision_strength': abs(threshold_margin),
            'was_close_call': abs(threshold_margin) < 0.05
        }
        
    def _analyze_attention_patterns(
        self,
        decision_output: 'DecisionOutput'
    ) -> Dict[str, Any]:
        """Analyze attention weight patterns."""
        if decision_output.attention_weights is None:
            return {}
            
        weights = decision_output.attention_weights.squeeze().cpu().numpy()
        
        # Find peak attention
        peak_idx = np.argmax(weights)
        peak_value = weights[peak_idx]
        
        # Calculate attention entropy
        entropy = -np.sum(weights * np.log(weights + 1e-8))
        
        # Identify focused areas
        focused_indices = np.where(weights > weights.mean() + weights.std())[0]
        
        return {
            'peak_attention_index': peak_idx,
            'peak_attention_value': peak_value,
            'attention_entropy': entropy,
            'attention_focus': 'concentrated' if entropy < 1.0 else 'distributed',
            'focused_components': focused_indices.tolist()
        }
        
    def _analyze_risk_impact(
        self,
        risk_proposal: Dict[str, Any],
        decision_output: 'DecisionOutput'
    ) -> Dict[str, float]:
        """Analyze how risk factors impacted decision."""
        risk_scores = {
            'position_size_impact': self._calculate_impact(
                risk_proposal['position_size_pct'],
                decision_output.risk_score
            ),
            'risk_reward_impact': self._calculate_impact(
                risk_proposal['risk_reward_ratio'],
                decision_output.execute_probability
            ),
            'portfolio_heat_impact': self._calculate_impact(
                risk_proposal['portfolio_heat'],
                1.0 - decision_output.execute_probability
            ),
            'overall_risk_impact': decision_output.risk_score
        }
        
        return risk_scores
        
    def _calculate_impact(self, factor_value: float, outcome: float) -> float:
        """Calculate impact correlation."""
        # Simplified impact calculation
        return np.tanh(factor_value * outcome)
        
    def _generate_counterfactuals(
        self,
        unified_state: torch.Tensor,
        risk_proposal: Dict[str, Any],
        original_decision: 'DecisionOutput'
    ) -> List[Dict[str, Any]]:
        """
        Generate counterfactual scenarios.
        
        What would happen if we changed certain inputs?
        """
        counterfactuals = []
        
        # Counterfactual 1: Lower risk
        low_risk_proposal = risk_proposal.copy()
        low_risk_proposal['risk_metrics']['portfolio_risk_score'] *= 0.5
        low_risk_proposal['portfolio_heat'] *= 0.5
        
        low_risk_decision = self._evaluate_counterfactual(
            unified_state,
            low_risk_proposal
        )
        
        counterfactuals.append({
            'scenario': 'low_risk',
            'description': 'Risk reduced by 50%',
            'original_decision': original_decision.decision,
            'counterfactual_decision': low_risk_decision,
            'decision_changed': low_risk_decision != original_decision.decision
        })
        
        # Counterfactual 2: Better risk-reward
        better_rr_proposal = risk_proposal.copy()
        better_rr_proposal['risk_reward_ratio'] = 3.0
        
        better_rr_decision = self._evaluate_counterfactual(
            unified_state,
            better_rr_proposal
        )
        
        counterfactuals.append({
            'scenario': 'better_risk_reward',
            'description': 'Risk-reward ratio = 3.0',
            'original_decision': original_decision.decision,
            'counterfactual_decision': better_rr_decision,
            'decision_changed': better_rr_decision != original_decision.decision
        })
        
        # Counterfactual 3: Different market regime
        # This would require changing the state encoding
        
        return counterfactuals
        
    def _evaluate_counterfactual(
        self,
        unified_state: torch.Tensor,
        risk_proposal: Dict[str, Any]
    ) -> str:
        """Evaluate counterfactual scenario."""
        # Simplified - would use actual model
        # This is a placeholder
        risk_score = risk_proposal['risk_metrics']['portfolio_risk_score']
        return 'EXECUTE' if risk_score < 0.5 else 'REJECT'
        
    def _calculate_feature_importance(
        self,
        unified_state: torch.Tensor,
        risk_proposal: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate feature importance scores."""
        # Use gradient-based importance
        unified_state.requires_grad_(True)
        
        # Forward pass
        with torch.enable_grad():
            output = self.model(
                unified_state,
                risk_proposal,
                {'should_qualify': torch.tensor(True)},
                None
            )
            
            # Backward pass
            output.execute_probability.backward()
            
        # Get gradients
        gradients = unified_state.grad.abs().cpu().numpy()
        
        # Aggregate by feature groups
        # Assuming state is split into: structure(64), tactical(48), regime(16), lvn(8), rest
        importance = {
            'structure': gradients[:64].mean(),
            'tactical': gradients[64:112].mean(),
            'regime': gradients[112:128].mean(),
            'lvn': gradients[128:136].mean(),
            'synergy_context': gradients[136:].mean()
        }
        
        # Normalize
        total = sum(importance.values())
        importance = {k: v/total for k, v in importance.items()}
        
        return importance
        
    def create_decision_report(
        self,
        interpretation: Dict[str, Any]
    ) -> str:
        """Create human-readable decision report."""
        report = f"""
DecisionGate Interpretation Report
==================================

Decision: {interpretation['decision']}
Confidence: {interpretation['confidence']:.3f}

Key Decision Factors:
--------------------
"""
        
        for factor in interpretation['factors']['critical_factors']:
            report += f"- {factor['factor']}: {factor['score']:.3f} ({factor['status']})\n"
            
        report += f"\nThreshold Analysis:
------------------
Margin from threshold: {interpretation['factors']['threshold_margin']:.3f}
Close call: {'Yes' if interpretation['factors']['was_close_call'] else 'No'}

Attention Analysis:
------------------
Focus type: {interpretation['attention'].get('attention_focus', 'N/A')}
Peak attention: Component {interpretation['attention'].get('peak_attention_index', 'N/A')}

Risk Impact:
-----------
Overall risk impact: {interpretation['risk_impact']['overall_risk_impact']:.3f}
Position size impact: {interpretation['risk_impact']['position_size_impact']:.3f}
Risk-reward impact: {interpretation['risk_impact']['risk_reward_impact']:.3f}

Feature Importance:
------------------
"""
        
        for feature, importance in interpretation['feature_importance'].items():
            report += f"- {feature}: {importance:.2%}\n"
            
        report += "\nCounterfactual Analysis:\n"
        report += "----------------------\n"
        
        for cf in interpretation['counterfactuals']:
            if cf['decision_changed']:
                report += f"- {cf['description']}: Would change decision to {cf['counterfactual_decision']}\n"
            else:
                report += f"- {cf['description']}: No change in decision\n"
                
        return report


class AttentionVisualizer:
    """Visualize attention patterns from DecisionGate."""
    
    @staticmethod
    def plot_attention_heatmap(
        attention_weights: torch.Tensor,
        component_names: List[str] = None
    ) -> plt.Figure:
        """Create attention heatmap."""
        weights = attention_weights.squeeze().cpu().numpy()
        
        if component_names is None:
            component_names = ['Structure', 'Tactical', 'Regime', 'LVN', 'Risk']
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            weights.reshape(-1, 1),
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            yticklabels=component_names,
            xticklabels=['Attention'],
            cbar_kws={'label': 'Attention Weight'},
            ax=ax
        )
        
        ax.set_title('DecisionGate Attention Weights', fontsize=16)
        
        return fig
        
    @staticmethod
    def plot_decision_path(
        decision_factors: Dict[str, Any],
        validation_scores: Dict[str, float]
    ) -> plt.Figure:
        """Visualize decision path."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Validation scores
        scores = list(validation_scores.values())
        labels = list(validation_scores.keys())
        
        colors = ['green' if s > 0.7 else 'orange' if s > 0.5 else 'red' 
                 for s in scores]
        
        bars = ax1.barh(labels, scores, color=colors)
        ax1.set_xlim(0, 1)
        ax1.set_xlabel('Validation Score')
        ax1.set_title('Multi-Factor Validation Scores')
        ax1.axvline(x=0.6, color='black', linestyle='--', label='Threshold')
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax1.text(score + 0.02, i, f'{score:.3f}', va='center')
            
        # Decision factors
        factors_df = pd.DataFrame([
            {'Factor': k, 'Value': v} 
            for k, v in decision_factors.items() 
            if isinstance(v, (int, float, bool))
        ])
        
        if not factors_df.empty:
            factors_df['Value'] = factors_df['Value'].astype(float)
            
            ax2.barh(factors_df['Factor'], factors_df['Value'])
            ax2.set_xlabel('Value')
            ax2.set_title('Decision Factors')
            
        plt.tight_layout()
        return fig