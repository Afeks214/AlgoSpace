"""
Example integration of MC Dropout consensus mechanism with Main MARL Core.

This example demonstrates how to integrate the MC Dropout system
with the unified intelligence architecture for production trading.
"""

import torch
import asyncio
import logging
import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path

from src.agents.main_core.mc_dropout import MCDropoutConsensus
from src.agents.main_core.models import SharedPolicyNetwork
from src.agents.main_core.mc_monitoring import MCDropoutMonitor, AlertSystem
from src.agents.main_core.mc_calibration import MCDropoutCalibrator
from src.agents.main_core.mc_dropout_optimization import MCDropoutOptimizationConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MCDropoutIntegration:
    """Example integration of MC Dropout consensus system."""
    
    def __init__(self, config_path: str = "config/mc_dropout_config.yaml"):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.consensus = MCDropoutConsensus(self.config['mc_dropout_consensus'])
        self.monitor = MCDropoutMonitor(self.config['mc_dropout_consensus']['monitoring'])
        self.calibrator = MCDropoutCalibrator(self.config['mc_dropout_consensus']['calibration'])
        self.alert_system = AlertSystem(self.config['alerts'])
        
        # Initialize optimization suite
        opt_config = MCDropoutOptimizationConfig()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizers = opt_config.get_optimizer_suite(device)
        
        # Initialize shared policy network
        self.policy = SharedPolicyNetwork(
            input_dim=144,  # 64 + 48 + 16 + 8 + 8
            hidden_dims=[256, 128, 64],
            dropout_rate=0.2,
            action_dim=2
        )
        
        # Move to device
        self.policy = self.policy.to(device)
        self.device = device
        
        # Load pre-trained weights if available
        self._load_weights()
        
        logger.info(f"MC Dropout Integration initialized on {device}")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
        
    def _load_weights(self, checkpoint_path: Optional[str] = None):
        """Load pre-trained model weights."""
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            logger.info(f"Loaded model from {checkpoint_path}")
        else:
            logger.info("No checkpoint found, using random initialization")
            
    async def process_trading_decision(
        self,
        synergy_event: Dict[str, Any],
        unified_state: torch.Tensor,
        market_context: Dict[str, Any],
        risk_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Complete decision flow with MC Dropout consensus.
        
        Args:
            synergy_event: Trading opportunity from synergy detection
            unified_state: Combined state vector [1, 144]
            market_context: Current market conditions
            risk_context: Risk parameters
            
        Returns:
            Decision dictionary with all relevant information
        """
        # Ensure state is on correct device
        unified_state = unified_state.to(self.device)
        
        # 1. Run MC Dropout consensus evaluation
        logger.info(f"Processing synergy event: {synergy_event['type']}")
        
        try:
            consensus_result = self.consensus.evaluate(
                model=self.policy,
                input_state=unified_state,
                market_context=market_context,
                risk_context=risk_context
            )
        except Exception as e:
            logger.error(f"MC Dropout evaluation failed: {e}")
            return self._create_error_response(synergy_event, str(e))
            
        # 2. Check for alerts
        alerts = self.alert_system.check_alerts(consensus_result)
        if alerts:
            logger.warning(f"MC Dropout alerts: {[a['message'] for a in alerts]}")
            
        # 3. Record for monitoring
        self.monitor.record_decision(consensus_result)
        
        # 4. Log decision metrics
        logger.info(
            f"MC Dropout metrics - "
            f"Confidence: {consensus_result.uncertainty_metrics.calibrated_confidence:.3f}, "
            f"Epistemic: {consensus_result.uncertainty_metrics.epistemic_uncertainty:.3f}, "
            f"Aleatoric: {consensus_result.uncertainty_metrics.aleatoric_uncertainty:.3f}, "
            f"Converged: {consensus_result.convergence_info['converged']}"
        )
        
        # 5. Check if we should proceed
        if not consensus_result.should_proceed:
            return self._create_rejection_response(synergy_event, consensus_result)
            
        # 6. Prepare execution request
        execution_request = {
            'decision': 'EXECUTE',
            'synergy_event': synergy_event,
            'confidence': consensus_result.uncertainty_metrics.calibrated_confidence,
            'mc_dropout_metrics': {
                'total_uncertainty': consensus_result.uncertainty_metrics.total_uncertainty,
                'epistemic_uncertainty': consensus_result.uncertainty_metrics.epistemic_uncertainty,
                'aleatoric_uncertainty': consensus_result.uncertainty_metrics.aleatoric_uncertainty,
                'predictive_entropy': consensus_result.uncertainty_metrics.predictive_entropy,
                'decision_boundary_distance': consensus_result.uncertainty_metrics.decision_boundary_distance
            },
            'convergence': {
                'converged': consensus_result.convergence_info['converged'],
                'r_hat': consensus_result.convergence_info['r_hat'],
                'effective_samples': consensus_result.convergence_info['effective_samples']
            },
            'action_probabilities': consensus_result.action_probabilities.tolist(),
            'confidence_intervals': consensus_result.confidence_intervals,
            'outlier_count': len(consensus_result.outlier_samples),
            'alerts': [a['type'] for a in alerts]
        }
        
        logger.info(
            f"MC Dropout Decision: EXECUTE "
            f"(confidence: {execution_request['confidence']:.3f})"
        )
        
        return execution_request
        
    def update_calibration_from_results(
        self,
        recent_trades: List[Dict[str, Any]]
    ):
        """Update calibration based on trading results."""
        if not recent_trades:
            return
            
        # Extract predictions and outcomes
        predictions = []
        outcomes = []
        
        for trade in recent_trades:
            if 'mc_dropout_confidence' in trade:
                predictions.append({
                    'probability': trade['mc_dropout_confidence'],
                    'uncertainty': trade.get('mc_dropout_metrics', {}).get('total_uncertainty', 0)
                })
                
                # Determine if trade was successful
                profitable = trade['pnl'] > 0
                outcomes.append(profitable)
                
        # Update calibrator
        if predictions:
            self.calibrator.update(predictions, outcomes)
            
            # Update consensus mechanism with new calibration
            self.consensus.calibrators = self.calibrator.calibrators
            
            logger.info(f"Updated calibration with {len(predictions)} trades")
            
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics."""
        return {
            'consensus': self.consensus.get_diagnostics(),
            'monitor': self.monitor.get_current_stats(),
            'calibration': {
                'ensemble_weights': self.calibrator.ensemble_weights,
                'n_calibration_samples': len(self.calibrator.calibration_data)
            },
            'recent_alerts': list(self.alert_system.alert_history)[-10:]
        }
        
    def generate_report(self) -> str:
        """Generate comprehensive report."""
        report = self.monitor.generate_report()
        
        # Add diagnostics
        diagnostics = self.get_diagnostics()
        report += f"\n\nCalibration Weights: {diagnostics['calibration']['ensemble_weights']}"
        report += f"\nRecent Alerts: {len(diagnostics['recent_alerts'])}"
        
        return report
        
    def _create_rejection_response(
        self,
        synergy_event: Dict[str, Any],
        consensus_result: 'ConsensusResult'
    ) -> Dict[str, Any]:
        """Create detailed rejection response."""
        return {
            'decision': 'REJECT',
            'synergy_event': synergy_event,
            'reason': 'low_confidence',
            'confidence': consensus_result.uncertainty_metrics.calibrated_confidence,
            'details': {
                'predicted_action': consensus_result.predicted_action,
                'action_probabilities': consensus_result.action_probabilities.tolist(),
                'raw_confidence': consensus_result.uncertainty_metrics.confidence_score,
                'calibrated_confidence': consensus_result.uncertainty_metrics.calibrated_confidence,
                'uncertainty_breakdown': {
                    'total': consensus_result.uncertainty_metrics.total_uncertainty,
                    'epistemic': consensus_result.uncertainty_metrics.epistemic_uncertainty,
                    'aleatoric': consensus_result.uncertainty_metrics.aleatoric_uncertainty
                },
                'convergence': consensus_result.convergence_info['converged'],
                'outliers': len(consensus_result.outlier_samples),
                'boundary_distance': consensus_result.uncertainty_metrics.decision_boundary_distance
            }
        }
        
    def _create_error_response(
        self,
        synergy_event: Dict[str, Any],
        error_message: str
    ) -> Dict[str, Any]:
        """Create error response."""
        return {
            'decision': 'ERROR',
            'synergy_event': synergy_event,
            'error': error_message,
            'fallback_decision': self.config['mc_dropout_consensus']['safety']['fallback_decision']
        }


async def main():
    """Example usage of MC Dropout integration."""
    
    # Initialize integration
    mc_integration = MCDropoutIntegration()
    
    # Example 1: Process a trading decision
    print("\n=== Example 1: Trading Decision ===")
    
    # Simulate synergy event
    synergy_event = {
        'type': 'bullish_momentum',
        'strength': 0.8,
        'symbol': 'BTC/USDT',
        'timeframe': '5m',
        'confidence': 0.75
    }
    
    # Simulate unified state (would come from embedders)
    unified_state = torch.randn(1, 144)
    
    # Market and risk context
    market_context = {
        'regime': 'trending',
        'volatility': 1.2,
        'volume': 'high'
    }
    
    risk_context = {
        'risk_level': 'medium',
        'portfolio_heat': 0.06,
        'max_position_size': 0.02
    }
    
    # Make decision
    decision = await mc_integration.process_trading_decision(
        synergy_event,
        unified_state,
        market_context,
        risk_context
    )
    
    print(f"\nDecision: {decision['decision']}")
    if 'confidence' in decision:
        print(f"Confidence: {decision['confidence']:.3f}")
    if 'mc_dropout_metrics' in decision:
        print(f"Epistemic Uncertainty: {decision['mc_dropout_metrics']['epistemic_uncertainty']:.3f}")
        print(f"Aleatoric Uncertainty: {decision['mc_dropout_metrics']['aleatoric_uncertainty']:.3f}")
        
    # Example 2: Update calibration with results
    print("\n=== Example 2: Calibration Update ===")
    
    # Simulate trading results
    recent_trades = [
        {
            'mc_dropout_confidence': 0.75,
            'mc_dropout_metrics': {'total_uncertainty': 0.3},
            'pnl': 50.0
        },
        {
            'mc_dropout_confidence': 0.68,
            'mc_dropout_metrics': {'total_uncertainty': 0.4},
            'pnl': -20.0
        },
        {
            'mc_dropout_confidence': 0.82,
            'mc_dropout_metrics': {'total_uncertainty': 0.2},
            'pnl': 80.0
        }
    ]
    
    mc_integration.update_calibration_from_results(recent_trades)
    
    # Example 3: Get diagnostics
    print("\n=== Example 3: Diagnostics ===")
    
    diagnostics = mc_integration.get_diagnostics()
    print(f"Average Confidence: {diagnostics['monitor']['avg_confidence']:.3f}")
    print(f"Decision Rate: {diagnostics['monitor']['decision_rate']:.2%}")
    print(f"Calibration Samples: {diagnostics['calibration']['n_calibration_samples']}")
    
    # Example 4: Generate report
    print("\n=== Example 4: Report ===")
    report = mc_integration.generate_report()
    print(report)
    
    # Example 5: Stress test with multiple decisions
    print("\n=== Example 5: Stress Test ===")
    
    n_decisions = 10
    decisions = []
    
    for i in range(n_decisions):
        # Generate random states
        state = torch.randn(1, 144)
        
        # Vary market conditions
        market_ctx = {
            'regime': ['trending', 'volatile', 'ranging'][i % 3],
            'volatility': 0.8 + (i % 5) * 0.1
        }
        
        risk_ctx = {
            'risk_level': ['low', 'medium', 'high'][i % 3]
        }
        
        decision = await mc_integration.process_trading_decision(
            synergy_event,
            state,
            market_ctx,
            risk_ctx
        )
        
        decisions.append(decision['decision'])
        
    print(f"Decisions made: {decisions}")
    print(f"Execution rate: {decisions.count('EXECUTE') / len(decisions):.2%}")
    
    # Final diagnostics
    final_stats = mc_integration.get_diagnostics()
    print(f"\nFinal decision rate: {final_stats['monitor']['decision_rate']:.2%}")
    print(f"Total decisions: {final_stats['monitor']['total_decisions']}")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())