"""
File: src/agents/main_core/decision_gate_integration.py (NEW FILE)
Integration utilities for DecisionGate
"""

import torch
import asyncio
from typing import Dict, Any, Optional
import logging
import os
from datetime import datetime, timedelta

from .decision_gate_transformer import DecisionGateTransformer, DecisionOutput
from .decision_interpretability import DecisionInterpreter, AttentionVisualizer
from .decision_threshold_learning import AdaptiveThresholdLearner

logger = logging.getLogger(__name__)


class DecisionGateSystem:
    """
    Complete DecisionGate system with all components integrated.
    
    Combines:
    1. Transformer-based decision making
    2. Interpretability tools
    3. Adaptive threshold learning
    4. Performance monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.transformer = DecisionGateTransformer(config['transformer'])
        self.interpreter = DecisionInterpreter(self.transformer)
        self.threshold_learner = AdaptiveThresholdLearner(config['threshold_learning'])
        
        # Monitoring
        self.decision_history = []
        self.performance_metrics = {}
        
        # Load pre-trained weights if available
        self._load_weights(config.get('model_path'))
        
    async def make_decision(
        self,
        unified_state: torch.Tensor,
        risk_proposal: Dict[str, Any],
        mc_consensus: Dict[str, torch.Tensor],
        market_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make a complete decision with all enhancements.
        
        Returns:
            Comprehensive decision dictionary
        """
        try:
            # Get adaptive thresholds
            current_thresholds = self._get_current_thresholds(market_context)
            
            # Update transformer thresholds
            self._update_transformer_thresholds(current_thresholds)
            
            # Make decision
            decision_output = self.transformer(
                unified_state,
                risk_proposal,
                mc_consensus,
                market_context
            )
            
            # Interpret decision
            interpretation = self.interpreter.interpret_decision(
                decision_output,
                unified_state,
                risk_proposal
            )
            
            # Create comprehensive result
            result = {
                'decision': decision_output.decision,
                'confidence': decision_output.confidence,
                'execute_probability': decision_output.execute_probability,
                'thresholds_used': current_thresholds,
                'validation_scores': decision_output.validation_scores,
                'safety_checks': decision_output.safety_checks,
                'interpretation': interpretation,
                'timestamp': datetime.now()
            }
            
            # Record decision
            self._record_decision(result)
            
            # Log decision
            logger.info(
                f"DecisionGate: {result['decision']} "
                f"(confidence: {result['confidence']:.3f}, "
                f"threshold: {current_thresholds['execution']:.3f})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"DecisionGate error: {e}")
            return self._create_error_response(str(e))
            
    def update_from_outcome(
        self,
        decision_id: str,
        outcome: Dict[str, Any]
    ):
        """Update system based on trading outcome."""
        # Find original decision
        decision = self._find_decision(decision_id)
        if not decision:
            logger.warning(f"Decision {decision_id} not found")
            return
            
        # Update threshold learner
        self.threshold_learner.update_from_outcome(
            decision['decision'],
            outcome,
            decision['thresholds_used'],
            decision.get('market_context', {})
        )
        
        # Update performance metrics
        self._update_performance_metrics(decision, outcome)
        
    def _get_current_thresholds(
        self,
        market_context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Get current adaptive thresholds."""
        regime = market_context.get('regime', 'unknown')
        recent_performance = self._calculate_recent_performance()
        
        thresholds = {}
        for threshold_type in ['execution', 'validation', 'risk']:
            thresholds[threshold_type] = self.threshold_learner.get_threshold(
                threshold_type,
                regime,
                recent_performance
            )
            
        return thresholds
        
    def _update_transformer_thresholds(self, thresholds: Dict[str, float]):
        """Update transformer with current thresholds."""
        # This would update the dynamic threshold layer
        # Implementation depends on specific architecture
        pass
        
    def _calculate_recent_performance(self) -> float:
        """Calculate recent win rate."""
        if not self.decision_history:
            return 0.5
            
        recent = self.decision_history[-50:]
        executed = [d for d in recent if d['decision'] == 'EXECUTE']
        
        if not executed:
            return 0.5
            
        wins = sum(1 for d in executed if d.get('outcome', {}).get('profitable', False))
        return wins / len(executed)
        
    def _record_decision(self, decision: Dict[str, Any]):
        """Record decision for history."""
        decision['id'] = f"DEC_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.decision_history.append(decision)
        
        # Maintain history size
        if len(self.decision_history) > 10000:
            self.decision_history = self.decision_history[-5000:]
            
    def _find_decision(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Find decision by ID."""
        for decision in reversed(self.decision_history):
            if decision.get('id') == decision_id:
                return decision
        return None
        
    def _update_performance_metrics(
        self,
        decision: Dict[str, Any],
        outcome: Dict[str, Any]
    ):
        """Update performance tracking."""
        if decision['decision'] == 'EXECUTE':
            self.performance_metrics['total_executions'] = \
                self.performance_metrics.get('total_executions', 0) + 1
                
            if outcome.get('profitable', False):
                self.performance_metrics['total_wins'] = \
                    self.performance_metrics.get('total_wins', 0) + 1
                    
        else:
            self.performance_metrics['total_rejections'] = \
                self.performance_metrics.get('total_rejections', 0) + 1
                
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        threshold_stats = self.threshold_learner.get_statistics()
        
        total_decisions = len(self.decision_history)
        total_executions = self.performance_metrics.get('total_executions', 0)
        total_wins = self.performance_metrics.get('total_wins', 0)
        
        return {
            'total_decisions': total_decisions,
            'execution_rate': total_executions / total_decisions if total_decisions > 0 else 0,
            'win_rate': total_wins / total_executions if total_executions > 0 else 0,
            'threshold_stats': threshold_stats,
            'recent_decisions': self.decision_history[-10:] if self.decision_history else []
        }
        
    def _create_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Create error response."""
        return {
            'decision': 'REJECT',
            'confidence': 0.0,
            'error': error_msg,
            'timestamp': datetime.now()
        }
        
    def _load_weights(self, model_path: Optional[str]):
        """Load pre-trained weights."""
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            self.transformer.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded DecisionGate weights from {model_path}")


# Testing utilities
class DecisionGateTestSuite:
    """Comprehensive test suite for DecisionGate."""
    
    @staticmethod
    def test_basic_functionality():
        """Test basic decision making."""
        config = {
            'transformer': {
                'state_dim': 512,
                'risk_dim': 128,
                'hidden_dim': 384,
                'n_layers': 4,
                'n_heads': 8
            },
            'threshold_learning': {
                'learning_rate': 0.01
            }
        }
        
        system = DecisionGateSystem(config)
        
        # Test inputs
        unified_state = torch.randn(1, 512)
        risk_proposal = create_test_risk_proposal()
        mc_consensus = {
            'should_qualify': torch.tensor([True]),
            'qualify_prob': torch.tensor([0.8]),
            'entropy': torch.tensor([0.3])
        }
        market_context = {'regime': 'trending'}
        
        # Make decision
        result = asyncio.run(system.make_decision(
            unified_state,
            risk_proposal,
            mc_consensus,
            market_context
        ))
        
        # Verify result structure
        assert 'decision' in result
        assert 'confidence' in result
        assert 'interpretation' in result
        assert result['decision'] in ['EXECUTE', 'REJECT']
        
        print("✅ Basic functionality test passed")
        
    @staticmethod
    def test_threshold_adaptation():
        """Test threshold learning."""
        learner = AdaptiveThresholdLearner({'learning_rate': 0.01})
        
        # Simulate outcomes
        for i in range(100):
            decision = 'EXECUTE' if i % 2 == 0 else 'REJECT'
            outcome = {
                'profitable': i % 3 == 0,
                'pnl_ratio': 1.5 if i % 3 == 0 else -0.5
            }
            thresholds = {'execution': 0.65, 'validation': 0.60}
            market_context = {'regime': 'trending'}
            
            learner.update_from_outcome(
                decision,
                outcome,
                thresholds,
                market_context
            )
            
        # Check adaptation
        stats = learner.get_statistics()
        assert 'current_adjustments' in stats
        assert 'effective_thresholds' in stats
        
        print("✅ Threshold adaptation test passed")
        
    @staticmethod
    def test_interpretability():
        """Test decision interpretation."""
        config = {'state_dim': 512, 'risk_dim': 128}
        transformer = DecisionGateTransformer(config)
        interpreter = DecisionInterpreter(transformer)
        
        # Create test decision
        decision_output = DecisionOutput(
            decision='EXECUTE',
            confidence=0.75,
            execute_probability=0.75,
            risk_score=0.4,
            validation_scores={
                'risk': 0.7,
                'market': 0.8,
                'technical': 0.65
            },
            attention_weights=torch.rand(1, 5),
            threshold_used=0.65,
            decision_factors={'mc_consensus_passed': True},
            safety_checks={'position_size_valid': True}
        )
        
        # Interpret
        interpretation = interpreter.interpret_decision(
            decision_output,
            torch.randn(1, 512),
            create_test_risk_proposal()
        )
        
        # Generate report
        report = interpreter.create_decision_report(interpretation)
        assert len(report) > 0
        
        print("✅ Interpretability test passed")


def create_test_risk_proposal() -> Dict[str, Any]:
    """Create test risk proposal."""
    return {
        'position_size': 100,
        'position_size_pct': 0.02,
        'leverage': 1.0,
        'dollar_risk': 200,
        'portfolio_heat': 0.06,
        'stop_loss_distance': 20,
        'stop_loss_atr_multiple': 1.5,
        'use_trailing_stop': True,
        'take_profit_distance': 60,
        'risk_reward_ratio': 3.0,
        'expected_return': 600,
        'risk_metrics': {
            'portfolio_risk_score': 0.4,
            'correlation_risk': 0.2,
            'concentration_risk': 0.1,
            'market_risk_multiplier': 1.2
        },
        'confidence_scores': {
            'overall_confidence': 0.75,
            'sl_confidence': 0.8,
            'tp_confidence': 0.7,
            'size_confidence': 0.8
        }
    }


if __name__ == "__main__":
    # Run tests
    print("Running DecisionGate tests...")
    
    DecisionGateTestSuite.test_basic_functionality()
    DecisionGateTestSuite.test_threshold_adaptation()
    DecisionGateTestSuite.test_interpretability()
    
    print("\n✅ All tests passed!")