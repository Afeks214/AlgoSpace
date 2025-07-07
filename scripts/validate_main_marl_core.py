"""
Production Validation Script for Main MARL Core Shared Policy.

This script performs comprehensive validation of the Main MARL Core
implementation to ensure production readiness with performance benchmarking.
"""

import torch
import time
import numpy as np
import asyncio
from typing import Dict, Any, List
import json
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.main_core.models import (
    SharedPolicyNetwork,
    MCDropoutConsensus,
    StructureEmbedder,
    TacticalEmbedder,
    RegimeEmbedder,
    LVNEmbedder
)
from src.agents.main_core.decision_gate import DecisionGate, RiskProposalEncoder
from src.training.marl_training import MAPPOTrainer, ExperienceBuffer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionValidator:
    """Comprehensive production validation for Main MARL Core."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.device = torch.device(self.config.get('device', 'cpu'))
        
        # Initialize components
        self.shared_policy = None
        self.mc_consensus = None
        self.decision_gate = None
        self.embedders = {}
        
        # Validation results
        self.results = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for validation."""
        return {
            'device': 'cpu',
            'policy': {
                'state_dim': 512,
                'hidden_dim': 512,
                'n_heads': 8,
                'n_layers': 6,
                'dropout_rate': 0.2
            },
            'mc_dropout': {
                'n_samples': 50,
                'confidence_threshold': 0.65
            },
            'decision_gate': {
                'input_dim': 640,
                'hidden_dim': 256
            },
            'validation': {
                'latency_samples': 100,
                'stress_test_duration': 30,
                'memory_test_batches': [1, 4, 16, 64, 128]
            }
        }
        
    def initialize_components(self):
        """Initialize all components for validation."""
        logger.info("Initializing Main MARL Core components...")
        
        # Shared Policy Network
        self.shared_policy = SharedPolicyNetwork(
            **self.config['policy']
        ).to(self.device)
        
        # MC Dropout Consensus
        self.mc_consensus = MCDropoutConsensus(
            **self.config['mc_dropout']
        )
        
        # Decision Gate
        self.decision_gate = DecisionGate(
            self.config['decision_gate']
        ).to(self.device)
        
        # Embedders
        self.embedders = {
            'structure': StructureEmbedder(output_dim=64).to(self.device),
            'tactical': TacticalEmbedder(output_dim=48).to(self.device),
            'regime': RegimeEmbedder(output_dim=16).to(self.device),
            'lvn': LVNEmbedder(output_dim=8).to(self.device)
        }
        
        logger.info("✅ Components initialized successfully")
        
    async def run_validation(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        logger.info("🚀 Starting Main MARL Core Production Validation")
        
        # Initialize components
        self.initialize_components()
        
        # Run validation tests
        await self._test_latency()
        await self._test_mc_dropout_stability()
        await self._test_memory_usage()
        await self._test_stress_performance()
        await self._test_decision_quality()
        await self._test_numerical_stability()
        await self._test_gradient_flow()
        
        # Evaluate overall results
        production_ready = self._evaluate_production_readiness()
        self.results['production_ready'] = production_ready
        self.results['validation_timestamp'] = time.time()
        
        return self.results
        
    async def _test_latency(self):
        """Test end-to-end latency performance."""
        logger.info("⏱️  Testing latency performance...")
        
        latencies = []
        n_samples = self.config['validation']['latency_samples']
        
        # Warmup
        for _ in range(10):
            unified_state = torch.randn(1, 512).to(self.device)
            with torch.no_grad():
                _ = self.shared_policy(unified_state)
                
        # Measure latency
        for _ in range(n_samples):
            unified_state = torch.randn(1, 512).to(self.device)
            
            start_time = time.perf_counter()
            
            with torch.no_grad():
                # Policy forward pass
                policy_output = self.shared_policy(unified_state)
                
                # MC Dropout consensus
                consensus_results = self.mc_consensus.evaluate(
                    self.shared_policy, unified_state
                )
                
                # Risk proposal (mocked)
                risk_proposal = self._create_mock_risk_proposal()
                
                # Decision gate
                decision_result = self.decision_gate(
                    unified_state, risk_proposal, consensus_results
                )
                
            latency_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(latency_ms)
            
        self.results['latency'] = {
            'mean_ms': np.mean(latencies),
            'std_ms': np.std(latencies),
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'max_ms': np.max(latencies),
            'samples': n_samples
        }
        
        logger.info(f"  Mean latency: {self.results['latency']['mean_ms']:.2f}ms")
        logger.info(f"  P99 latency: {self.results['latency']['p99_ms']:.2f}ms")
        
    async def _test_mc_dropout_stability(self):
        """Test MC Dropout consensus stability."""
        logger.info("🎲 Testing MC Dropout stability...")
        
        test_state = torch.randn(1, 512).to(self.device)
        
        # Run multiple evaluations
        decisions = []
        confidences = []
        consensus_scores = []
        
        for _ in range(20):
            results = self.mc_consensus.evaluate(
                self.shared_policy, test_state
            )
            decisions.append(results['should_qualify'].item())
            confidences.append(results['qualify_prob'].item())
            consensus_scores.append(results['consensus_score'].item())
            
        # Calculate stability metrics
        decision_consistency = np.std(decisions)
        confidence_stability = 1.0 - np.std(confidences)
        consensus_stability = 1.0 - np.std(consensus_scores)
        
        self.results['mc_dropout_stability'] = {
            'decision_consistency': decision_consistency,
            'confidence_stability': confidence_stability,
            'consensus_stability': consensus_stability,
            'mean_confidence': np.mean(confidences),
            'runs': 20
        }
        
        logger.info(f"  Confidence stability: {confidence_stability:.3f}")
        logger.info(f"  Consensus stability: {consensus_stability:.3f}")
        
    async def _test_memory_usage(self):
        """Test memory usage patterns."""
        logger.info("💾 Testing memory usage...")
        
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            
        memory_results = {}
        batch_sizes = self.config['validation']['memory_test_batches']
        
        for batch_size in batch_sizes:
            # Test batch processing
            unified_state = torch.randn(batch_size, 512).to(self.device)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated()
                
            with torch.no_grad():
                _ = self.shared_policy(unified_state)
                
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
                memory_after = torch.cuda.memory_allocated()
                memory_used = memory_after - memory_before
                memory_results[f'batch_{batch_size}'] = memory_used / (1024**2)  # MB
            else:
                memory_results[f'batch_{batch_size}'] = 0  # CPU memory tracking not implemented
                
        if self.device.type == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
            memory_results['peak_memory_mb'] = peak_memory
            
        self.results['memory_usage'] = memory_results
        
        if self.device.type == 'cuda':
            logger.info(f"  Peak memory: {peak_memory:.2f} MB")
        else:
            logger.info("  Memory tracking: CPU mode")
            
    async def _test_stress_performance(self):
        """Test performance under stress conditions."""
        logger.info("🔥 Running stress test...")
        
        duration = self.config['validation']['stress_test_duration']
        start_time = time.time()
        decisions_processed = 0
        errors = 0
        
        while time.time() - start_time < duration:
            try:
                # Random batch size
                batch_size = np.random.randint(1, 17)
                unified_state = torch.randn(batch_size, 512).to(self.device)
                
                with torch.no_grad():
                    # Full decision pipeline
                    _ = self.shared_policy(unified_state)
                    
                    # MC consensus on subset
                    if batch_size <= 4:  # Limit MC Dropout to small batches
                        _ = self.mc_consensus.evaluate(
                            self.shared_policy, 
                            unified_state[:1]  # Single sample
                        )
                        
                decisions_processed += batch_size
                
            except Exception as e:
                errors += 1
                if errors > 10:  # Too many errors
                    break
                    
        elapsed_time = time.time() - start_time
        throughput = decisions_processed / elapsed_time
        error_rate = errors / max(decisions_processed, 1)
        
        self.results['stress_test'] = {
            'duration_s': elapsed_time,
            'decisions_processed': decisions_processed,
            'throughput_per_s': throughput,
            'errors': errors,
            'error_rate': error_rate
        }
        
        logger.info(f"  Throughput: {throughput:.1f} decisions/sec")
        logger.info(f"  Error rate: {error_rate:.2%}")
        
    async def _test_decision_quality(self):
        """Test decision quality and consistency."""
        logger.info("📊 Testing decision quality...")
        
        # Create diverse test scenarios
        scenarios = self._create_test_scenarios(100)
        
        qualifications = 0
        executions = 0
        high_confidence_decisions = 0
        
        for scenario in scenarios:
            unified_state = scenario['state'].to(self.device)
            risk_proposal = scenario['risk_proposal']
            
            with torch.no_grad():
                # MC Dropout consensus
                consensus_results = self.mc_consensus.evaluate(
                    self.shared_policy, unified_state
                )
                
                # Check qualification
                if consensus_results['should_qualify'].item():
                    qualifications += 1
                    
                    # Decision gate
                    decision_result = self.decision_gate(
                        unified_state, risk_proposal, consensus_results
                    )
                    
                    if decision_result['should_execute'].item():
                        executions += 1
                        
                    if decision_result['confidence'].item() > 0.8:
                        high_confidence_decisions += 1
                        
        total_scenarios = len(scenarios)
        
        self.results['decision_quality'] = {
            'qualification_rate': qualifications / total_scenarios,
            'execution_rate': executions / total_scenarios,
            'high_confidence_rate': high_confidence_decisions / max(executions, 1),
            'total_scenarios': total_scenarios
        }
        
        logger.info(f"  Qualification rate: {self.results['decision_quality']['qualification_rate']:.2%}")
        logger.info(f"  Execution rate: {self.results['decision_quality']['execution_rate']:.2%}")
        
    async def _test_numerical_stability(self):
        """Test numerical stability with edge cases."""
        logger.info("🔢 Testing numerical stability...")
        
        stability_results = {
            'nan_detected': False,
            'inf_detected': False,
            'extreme_values_handled': True
        }
        
        # Test with extreme values
        test_cases = [
            torch.full((1, 512), 1000.0),  # Very large values
            torch.full((1, 512), -1000.0),  # Very negative values
            torch.zeros(1, 512),           # All zeros
            torch.randn(1, 512) * 1e6      # Extreme random values
        ]
        
        for i, test_state in enumerate(test_cases):
            test_state = test_state.to(self.device)
            
            try:
                with torch.no_grad():
                    output = self.shared_policy(test_state)
                    
                # Check for NaN/Inf
                if torch.isnan(output.action_logits).any():
                    stability_results['nan_detected'] = True
                    
                if torch.isinf(output.action_logits).any():
                    stability_results['inf_detected'] = True
                    
                if torch.isnan(output.confidence).any():
                    stability_results['nan_detected'] = True
                    
            except Exception as e:
                stability_results['extreme_values_handled'] = False
                logger.warning(f"  Failed on test case {i}: {e}")
                
        self.results['numerical_stability'] = stability_results
        
        logger.info(f"  NaN detected: {stability_results['nan_detected']}")
        logger.info(f"  Inf detected: {stability_results['inf_detected']}")
        logger.info(f"  Extreme values handled: {stability_results['extreme_values_handled']}")
        
    async def _test_gradient_flow(self):
        """Test gradient flow for training."""
        logger.info("🌊 Testing gradient flow...")
        
        # Create training scenario
        unified_state = torch.randn(4, 512, requires_grad=True).to(self.device)
        targets = torch.tensor([0, 1, 0, 1]).to(self.device)
        
        # Forward pass
        output = self.shared_policy(unified_state)
        loss = torch.nn.functional.cross_entropy(output.action_logits, targets)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        gradient_flow = {
            'input_gradients': unified_state.grad is not None,
            'parameter_gradients': 0,
            'nan_gradients': 0,
            'zero_gradients': 0,
            'total_parameters': 0
        }
        
        for name, param in self.shared_policy.named_parameters():
            if param.requires_grad:
                gradient_flow['total_parameters'] += 1
                
                if param.grad is not None:
                    gradient_flow['parameter_gradients'] += 1
                    
                    if torch.isnan(param.grad).any():
                        gradient_flow['nan_gradients'] += 1
                        
                    if torch.allclose(param.grad, torch.zeros_like(param.grad)):
                        gradient_flow['zero_gradients'] += 1
                        
        gradient_flow['gradient_coverage'] = gradient_flow['parameter_gradients'] / gradient_flow['total_parameters']
        
        self.results['gradient_flow'] = gradient_flow
        
        logger.info(f"  Gradient coverage: {gradient_flow['gradient_coverage']:.2%}")
        logger.info(f"  NaN gradients: {gradient_flow['nan_gradients']}")
        
    def _create_mock_risk_proposal(self) -> Dict[str, Any]:
        """Create mock risk proposal for testing."""
        return {
            'position_size': 2,
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
                'portfolio_risk_score': np.random.uniform(0.1, 0.8),
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
        
    def _create_test_scenarios(self, n: int) -> List[Dict[str, Any]]:
        """Create diverse test scenarios."""
        scenarios = []
        
        for i in range(n):
            # Create diverse states
            if i % 3 == 0:
                # High confidence scenario
                state = torch.randn(1, 512) * 0.5  # Lower variance
            elif i % 3 == 1:
                # Medium confidence scenario
                state = torch.randn(1, 512)
            else:
                # Low confidence scenario
                state = torch.randn(1, 512) * 2.0  # Higher variance
                
            scenarios.append({
                'state': state,
                'risk_proposal': self._create_mock_risk_proposal(),
                'expected_difficulty': 'easy' if i % 3 == 0 else 'medium' if i % 3 == 1 else 'hard'
            })
            
        return scenarios
        
    def _evaluate_production_readiness(self) -> bool:
        """Evaluate overall production readiness."""
        logger.info("📋 Evaluating production readiness...")
        
        passed = True
        checks = []
        
        # Latency check
        if self.results['latency']['p99_ms'] > 20:
            checks.append("❌ Latency too high (>20ms)")
            passed = False
        else:
            checks.append("✅ Latency acceptable")
            
        # MC Dropout stability
        if self.results['mc_dropout_stability']['confidence_stability'] < 0.8:
            checks.append("❌ MC Dropout unstable")
            passed = False
        else:
            checks.append("✅ MC Dropout stable")
            
        # Stress test
        if self.results['stress_test']['error_rate'] > 0.01:
            checks.append("❌ High error rate under stress")
            passed = False
        else:
            checks.append("✅ Stress test passed")
            
        # Decision quality
        if self.results['decision_quality']['execution_rate'] < 0.05:
            checks.append("⚠️  Very low execution rate")
        elif self.results['decision_quality']['execution_rate'] > 0.5:
            checks.append("⚠️  Very high execution rate")
        else:
            checks.append("✅ Execution rate reasonable")
            
        # Numerical stability
        if (self.results['numerical_stability']['nan_detected'] or 
            self.results['numerical_stability']['inf_detected']):
            checks.append("❌ Numerical instability detected")
            passed = False
        else:
            checks.append("✅ Numerically stable")
            
        # Gradient flow
        if self.results['gradient_flow']['gradient_coverage'] < 0.9:
            checks.append("❌ Poor gradient coverage")
            passed = False
        else:
            checks.append("✅ Good gradient flow")
            
        # Memory usage (if CUDA)
        if 'peak_memory_mb' in self.results['memory_usage']:
            if self.results['memory_usage']['peak_memory_mb'] > 1000:
                checks.append("⚠️  High memory usage")
            else:
                checks.append("✅ Memory usage acceptable")
                
        # Log all checks
        for check in checks:
            logger.info(f"  {check}")
            
        return passed
        
    def save_results(self, output_path: str = "main_marl_validation.json"):
        """Save validation results to file."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"📄 Results saved to {output_path}")


async def main():
    """Main validation function."""
    validator = ProductionValidator()
    
    try:
        results = await validator.run_validation()
        validator.save_results()
        
        # Print summary
        if results['production_ready']:
            logger.info("🎉 PASSED - Main MARL Core is production ready!")
        else:
            logger.info("⚠️  FAILED - Issues detected in Main MARL Core")
            
        return results
        
    except Exception as e:
        logger.error(f"❌ Validation failed with error: {e}")
        return None


if __name__ == "__main__":
    results = asyncio.run(main())
    
    if results:
        print("\n" + "="*60)
        print("Main MARL Core Production Validation Summary")
        print("="*60)
        
        if results['production_ready']:
            print("🎉 STATUS: PRODUCTION READY")
        else:
            print("⚠️  STATUS: NEEDS ATTENTION")
            
        print(f"\nLatency P99: {results['latency']['p99_ms']:.2f}ms")
        print(f"MC Dropout Stability: {results['mc_dropout_stability']['confidence_stability']:.3f}")
        print(f"Stress Test Throughput: {results['stress_test']['throughput_per_s']:.1f} decisions/sec")
        print(f"Decision Quality: {results['decision_quality']['execution_rate']:.2%} execution rate")
        
        if 'peak_memory_mb' in results['memory_usage']:
            print(f"Peak Memory: {results['memory_usage']['peak_memory_mb']:.2f} MB")
            
        print("\n" + "="*60)
    else:
        print("❌ Validation failed to complete")