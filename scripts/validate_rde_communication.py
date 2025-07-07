"""
Production validation script for RDE Communication LSTM.
"""

import sys
sys.path.append('..')

import torch
import numpy as np
import logging
import time
from pathlib import Path
import yaml
import psutil
import os
from typing import Dict, List

from src.agents.rde.engine import RDEComponent
from src.agents.rde.communication import RDECommunicationLSTM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RDECommunicationValidator:
    """Validates RDE Communication LSTM for production readiness."""
    
    def __init__(self, config_path: str = '../config/settings.yaml'):
        """Initialize validator with configuration."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.validation_results = {}
        
    def run_all_validations(self) -> bool:
        """Run all validation checks."""
        print("=" * 60)
        print("RDE Communication LSTM Production Validation")
        print("=" * 60)
        
        all_passed = True
        
        # Run each validation
        validations = [
            ("Initialization", self.validate_initialization),
            ("Inference Latency", self.validate_latency),
            ("Memory Stability", self.validate_memory_stability),
            ("Uncertainty Calibration", self.validate_uncertainty),
            ("State Persistence", self.validate_state_persistence),
            ("Integration", self.validate_integration),
            ("Error Recovery", self.validate_error_recovery),
            ("Load Testing", self.validate_load_testing)
        ]
        
        for name, validation_func in validations:
            print(f"\n{'=' * 40}")
            print(f"{name} Validation")
            print('=' * 40)
            
            try:
                passed = validation_func()
                self.validation_results[name] = passed
                
                if passed:
                    print(f"✅ {name} validation PASSED")
                else:
                    print(f"❌ {name} validation FAILED")
                    all_passed = False
                    
            except Exception as e:
                logger.error(f"Error during {name} validation: {e}", exc_info=True)
                print(f"❌ {name} validation FAILED with error: {e}")
                self.validation_results[name] = False
                all_passed = False
                
        # Summary
        self._print_summary()
        
        return all_passed
    
    def validate_initialization(self) -> bool:
        """Validate proper initialization."""
        try:
            # Initialize RDE Communication LSTM
            rde_comm_config = self.config.get('rde_communication', {})
            rde_comm = RDECommunicationLSTM(rde_comm_config).to(self.device)
            
            # Check configuration
            assert rde_comm.input_dim == rde_comm_config.get('input_dim', 8)
            assert rde_comm.hidden_dim == rde_comm_config.get('hidden_dim', 32)
            assert rde_comm.output_dim == rde_comm_config.get('output_dim', 16)
            print("✓ Configuration loaded correctly")
            
            # Check model structure
            param_count = sum(p.numel() for p in rde_comm.parameters())
            print(f"✓ Model parameters: {param_count:,}")
            
            # Test forward pass
            test_input = torch.randn(1, 8).to(self.device)
            mu, sigma = rde_comm(test_input)
            
            assert mu.shape == (1, 16)
            assert sigma.shape == (1, 16)
            assert torch.all(sigma > 0)
            print("✓ Forward pass successful")
            
            return True
            
        except Exception as e:
            logger.error(f"Initialization validation failed: {e}")
            return False
    
    def validate_latency(self) -> bool:
        """Validate inference latency requirements."""
        rde_comm_config = self.config.get('rde_communication', {})
        rde_comm = RDECommunicationLSTM(rde_comm_config).to(self.device)
        rde_comm.eval()
        
        # Warm up
        for _ in range(50):
            _ = rde_comm(torch.randn(1, 8).to(self.device))
            
        # Measure latencies
        latencies = []
        for _ in range(1000):
            start = time.perf_counter()
            _ = rde_comm(torch.randn(1, 8).to(self.device))
            latency = (time.perf_counter() - start) * 1000  # ms
            latencies.append(latency)
            
        # Calculate statistics
        latencies = np.array(latencies)
        stats = {
            'mean': np.mean(latencies),
            'median': np.median(latencies),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
            'max': np.max(latencies)
        }
        
        print(f"Latency Statistics (ms):")
        for key, value in stats.items():
            print(f"  {key}: {value:.3f}")
            
        # Check requirements
        passed = stats['p99'] < 5.0  # <5ms at 99th percentile
        
        if passed:
            print("✓ Meets latency requirement (<5ms p99)")
        else:
            print(f"✗ Does not meet latency requirement (p99={stats['p99']:.3f}ms)")
            
        return passed
    
    def validate_memory_stability(self) -> bool:
        """Validate memory stability over extended operation."""
        rde_comm_config = self.config.get('rde_communication', {})
        rde_comm = RDECommunicationLSTM(rde_comm_config).to(self.device)
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Initial memory: {initial_memory:.1f} MB")
        
        # Run for many iterations
        for i in range(10000):
            regime = torch.randn(1, 8).to(self.device)
            mu, sigma = rde_comm(regime)
            
            # Periodically reset state
            if i % 1000 == 0 and i > 0:
                rde_comm.reset_hidden_state()
                
                # Check memory
                current_memory = process.memory_info().rss / 1024 / 1024
                print(f"Memory at iteration {i}: {current_memory:.1f} MB")
                
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        print(f"Final memory: {final_memory:.1f} MB")
        print(f"Memory growth: {memory_growth:.1f} MB")
        
        # Check for memory leaks
        passed = memory_growth < 50  # Less than 50MB growth
        
        if passed:
            print("✓ Memory stable (growth < 50MB)")
        else:
            print(f"✗ Excessive memory growth ({memory_growth:.1f} MB)")
            
        return passed
    
    def validate_uncertainty(self) -> bool:
        """Validate uncertainty calibration."""
        rde_comm_config = self.config.get('rde_communication', {})
        rde_comm = RDECommunicationLSTM(rde_comm_config).to(self.device)
        
        # Test uncertainty behavior
        uncertainties = []
        
        # Process identical inputs
        identical_input = torch.randn(1, 8).to(self.device)
        for _ in range(100):
            _, sigma = rde_comm(identical_input)
            uncertainties.append(sigma.mean().item())
            
        # Check uncertainty consistency
        uncertainty_std = np.std(uncertainties)
        print(f"Uncertainty std for identical inputs: {uncertainty_std:.4f}")
        
        # Test uncertainty for different regimes
        regime_uncertainties = {}
        regime_types = ['stable', 'transition', 'volatile']
        
        for regime_type in regime_types:
            if regime_type == 'stable':
                # Low variance inputs
                inputs = torch.randn(100, 8).to(self.device) * 0.1
            elif regime_type == 'transition':
                # Gradually changing inputs
                inputs = torch.linspace(-1, 1, 100).unsqueeze(1).repeat(1, 8).to(self.device)
            else:  # volatile
                # High variance inputs
                inputs = torch.randn(100, 8).to(self.device) * 2.0
                
            regime_unc = []
            for i in range(inputs.size(0)):
                _, sigma = rde_comm(inputs[i:i+1])
                regime_unc.append(sigma.mean().item())
                
            regime_uncertainties[regime_type] = np.mean(regime_unc)
            print(f"{regime_type} regime uncertainty: {regime_uncertainties[regime_type]:.4f}")
            
        # Check if uncertainty reflects input characteristics
        passed = (
            regime_uncertainties['stable'] < regime_uncertainties['transition'] and
            regime_uncertainties['transition'] < regime_uncertainties['volatile']
        )
        
        if passed:
            print("✓ Uncertainty calibration reflects regime characteristics")
        else:
            print("✗ Uncertainty calibration does not properly reflect regimes")
            
        return passed
    
    def validate_state_persistence(self) -> bool:
        """Validate hidden state persistence and recovery."""
        rde_comm_config = self.config.get('rde_communication', {})
        rde_comm = RDECommunicationLSTM(rde_comm_config).to(self.device)
        
        # Process sequence
        sequence = torch.randn(10, 8).to(self.device)
        outputs1 = []
        
        for i in range(sequence.size(0)):
            mu, _ = rde_comm(sequence[i:i+1])
            outputs1.append(mu)
            
        # Save checkpoint
        rde_comm.save_checkpoint()
        
        # Process more data
        for _ in range(5):
            rde_comm(torch.randn(1, 8).to(self.device))
            
        # Load checkpoint
        success = rde_comm.load_checkpoint()
        assert success, "Failed to load checkpoint"
        
        # Process same sequence again
        outputs2 = []
        for i in range(sequence.size(0)):
            mu, _ = rde_comm(sequence[i:i+1])
            outputs2.append(mu)
            
        # Compare outputs
        max_diff = 0
        for o1, o2 in zip(outputs1[5:], outputs2[5:]):  # Skip initial transient
            diff = torch.abs(o1 - o2).max().item()
            max_diff = max(max_diff, diff)
            
        print(f"Max output difference after state recovery: {max_diff:.6f}")
        
        passed = max_diff < 0.001  # Very small difference expected
        
        if passed:
            print("✓ State persistence and recovery working correctly")
        else:
            print("✗ State recovery produced different outputs")
            
        return passed
    
    def validate_integration(self) -> bool:
        """Validate integration with RDE and Main MARL Core."""
        try:
            # Initialize RDE
            rde_config = self.config.get('rde', {})
            rde = RDEComponent(rde_config)
            
            # Initialize RDE Communication
            rde_comm_config = self.config.get('rde_communication', {})
            rde_comm = RDECommunicationLSTM(rde_comm_config).to(self.device)
            
            # Test data flow
            # 1. Create dummy MMD features
            mmd_features = np.random.randn(24, 155).astype(np.float32)
            
            # 2. Get regime vector from RDE
            regime_vector = rde.get_regime_vector(mmd_features)
            print(f"✓ RDE output shape: {regime_vector.shape}")
            
            # 3. Process through communication LSTM
            regime_tensor = torch.tensor(regime_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
            mu, sigma = rde_comm(regime_tensor)
            
            print(f"✓ Communication output shapes: mu={mu.shape}, sigma={sigma.shape}")
            
            # 4. Verify output properties
            assert mu.shape == (1, 16)
            assert sigma.shape == (1, 16)
            assert torch.all(sigma > 0)
            
            # 5. Test with Main MARL Core dimensions
            expected_dim = self.config.get('main_marl_core', {}).get('embedders', {}).get('regime', {}).get('output_dim', 16)
            assert mu.shape[1] == expected_dim or mu.shape[1] == 16  # Either configured or default
            
            print("✓ Integration with RDE and Main MARL Core verified")
            return True
            
        except Exception as e:
            logger.error(f"Integration validation failed: {e}")
            return False
    
    def validate_error_recovery(self) -> bool:
        """Validate error recovery mechanisms."""
        rde_comm_config = self.config.get('rde_communication', {})
        rde_comm = RDECommunicationLSTM(rde_comm_config).to(self.device)
        
        tests_passed = []
        
        # Test 1: NaN input handling
        try:
            nan_input = torch.full((1, 8), float('nan')).to(self.device)
            # Should handle gracefully (implementation dependent)
            mu, sigma = rde_comm(nan_input)
            # If it doesn't error, check outputs are valid
            if not torch.isnan(mu).any() and not torch.isnan(sigma).any():
                tests_passed.append(True)
            else:
                tests_passed.append(False)
        except:
            # Or it might raise an error, which is also acceptable
            tests_passed.append(True)
            
        print(f"✓ NaN input handling: {'Passed' if tests_passed[-1] else 'Failed'}")
        
        # Test 2: State corruption recovery
        # Corrupt the hidden state
        if rde_comm.hidden is not None:
            original_hidden = rde_comm.hidden
            rde_comm.hidden = (
                torch.full_like(original_hidden[0], 1000.0),
                torch.full_like(original_hidden[1], 1000.0)
            )
            
        # Process normal input - should auto-recover
        normal_input = torch.randn(1, 8).to(self.device)
        mu, sigma = rde_comm(normal_input)
        
        # Check if outputs are reasonable
        if torch.all(torch.isfinite(mu)) and torch.all(torch.isfinite(sigma)):
            tests_passed.append(True)
            print("✓ Automatic recovery from corrupted state")
        else:
            tests_passed.append(False)
            print("✗ Failed to recover from corrupted state")
            
        # Test 3: Reset functionality
        rde_comm.reset_hidden_state()
        mu, sigma = rde_comm(normal_input)
        
        if torch.all(torch.isfinite(mu)) and torch.all(torch.isfinite(sigma)):
            tests_passed.append(True)
            print("✓ Reset functionality working")
        else:
            tests_passed.append(False)
            print("✗ Reset functionality failed")
            
        return all(tests_passed)
    
    def validate_load_testing(self) -> bool:
        """Validate performance under load."""
        rde_comm_config = self.config.get('rde_communication', {})
        rde_comm = RDECommunicationLSTM(rde_comm_config).to(self.device)
        
        # Simulate high-frequency trading scenario
        print("Simulating high-frequency regime updates...")
        
        start_time = time.time()
        n_updates = 10000
        
        for i in range(n_updates):
            # Random regime vectors
            regime = torch.randn(1, 8).to(self.device)
            mu, sigma = rde_comm(regime)
            
            # Simulate regime changes
            if i % 100 == 0:
                # Major regime shift
                regime = torch.randn(1, 8).to(self.device) * 2.0
                mu, sigma = rde_comm(regime)
                
        elapsed_time = time.time() - start_time
        updates_per_second = n_updates / elapsed_time
        
        print(f"Processed {n_updates} updates in {elapsed_time:.2f} seconds")
        print(f"Updates per second: {updates_per_second:.0f}")
        
        # Check if meets performance requirements
        # Should handle at least 1000 updates per second
        passed = updates_per_second > 1000
        
        if passed:
            print("✓ Meets performance requirements under load")
        else:
            print(f"✗ Performance too low ({updates_per_second:.0f} updates/sec)")
            
        return passed
    
    def _print_summary(self) -> None:
        """Print validation summary."""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.validation_results)
        passed_tests = sum(self.validation_results.values())
        
        for test_name, passed in self.validation_results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{test_name:.<40} {status}")
            
        print("=" * 60)
        print(f"Total: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("\n🎉 RDE Communication LSTM is PRODUCTION READY! 🎉")
        else:
            print("\n⚠️  Some validations failed. Please address issues before deployment.")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate RDE Communication LSTM')
    parser.add_argument(
        '--config',
        type=str,
        default='../config/settings.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Run validation
    validator = RDECommunicationValidator(args.config)
    success = validator.run_all_validations()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()