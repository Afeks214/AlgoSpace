#!/usr/bin/env python3
"""
Production validation script for Structure Embedder.

This script performs comprehensive validation of the Transformer-based Structure Embedder
to ensure it meets production requirements for latency, accuracy, and reliability.
"""

import torch
import time
import numpy as np
import logging
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.main_core.models import StructureEmbedder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StructureEmbedderValidator:
    """Comprehensive validator for Structure Embedder production readiness."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize validator with configuration.
        
        Args:
            config: Configuration dictionary for the Structure Embedder
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.results = {}
        
        logger.info(f"Initializing validator on device: {self.device}")
        
    def initialize_model(self) -> bool:
        """Initialize and load the Structure Embedder model."""
        try:
            # Initialize model with configuration
            self.model = StructureEmbedder(**self.config)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("✅ Model initialized successfully")
            logger.info(f"Model architecture: d_model={self.config.get('d_model', 128)}, "
                       f"n_heads={self.config.get('n_heads', 4)}, "
                       f"n_layers={self.config.get('n_layers', 3)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize model: {e}")
            return False
    
    def validate_latency(self) -> Dict[str, float]:
        """Test inference latency requirements."""
        logger.info("\n⏱️  Testing inference latency...")
        
        latency_results = {}
        batch_sizes = [1, 4, 8, 16, 32]
        
        for batch_size in batch_sizes:
            test_input = torch.randn(batch_size, 48, 8).to(self.device)
            
            # Warmup runs
            for _ in range(10):
                with torch.no_grad():
                    _ = self.model(test_input)
            
            # Synchronize GPU operations
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Time multiple runs
            times = []
            for _ in range(100):
                start_time = time.perf_counter()
                
                with torch.no_grad():
                    mu, sigma = self.model(test_input)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    
                elapsed = (time.perf_counter() - start_time) * 1000  # Convert to ms
                times.append(elapsed)
            
            # Calculate statistics
            avg_time = np.mean(times)
            p50_time = np.percentile(times, 50)
            p95_time = np.percentile(times, 95)
            p99_time = np.percentile(times, 99)
            
            latency_results[f'batch_{batch_size}'] = {
                'avg_ms': avg_time,
                'p50_ms': p50_time,
                'p95_ms': p95_time,
                'p99_ms': p99_time,
                'per_sample_ms': avg_time / batch_size
            }
            
            logger.info(f"  Batch size {batch_size:2d}: "
                       f"avg={avg_time:.2f}ms, p95={p95_time:.2f}ms, p99={p99_time:.2f}ms")
        
        # Check if latency requirements are met
        single_sample_p99 = latency_results['batch_1']['p99_ms']
        meets_requirement = single_sample_p99 <= 10.0
        
        latency_results['meets_requirement'] = meets_requirement
        latency_results['requirement_ms'] = 10.0
        
        if meets_requirement:
            logger.info("  ✅ Latency requirement met (p99 ≤ 10ms)")
        else:
            logger.warning(f"  ⚠️  Latency requirement NOT met (p99={single_sample_p99:.2f}ms > 10ms)")
        
        return latency_results
    
    def validate_numerical_stability(self) -> Dict[str, Any]:
        """Test numerical stability with edge cases."""
        logger.info("\n🔢 Testing numerical stability...")
        
        stability_results = {}
        
        test_cases = [
            ('zeros', torch.zeros(1, 48, 8)),
            ('ones', torch.ones(1, 48, 8)),
            ('large_values', torch.ones(1, 48, 8) * 1000),
            ('small_values', torch.ones(1, 48, 8) * 1e-6),
            ('negative_values', torch.ones(1, 48, 8) * -100),
            ('mixed_extreme', torch.randn(1, 48, 8) * 100),
            ('inf_values', torch.full((1, 48, 8), float('inf'))),
            ('random_normal', torch.randn(1, 48, 8))
        ]
        
        for case_name, test_input in test_cases:
            test_input = test_input.to(self.device)
            
            try:
                with torch.no_grad():
                    mu, sigma = self.model(test_input)
                
                # Check for NaN/Inf in outputs
                has_nan_mu = torch.isnan(mu).any().item()
                has_inf_mu = torch.isinf(mu).any().item()
                has_nan_sigma = torch.isnan(sigma).any().item()
                has_inf_sigma = torch.isinf(sigma).any().item()
                
                # Check if sigma is positive
                sigma_positive = torch.all(sigma > 0).item()
                
                case_result = {
                    'mu_has_nan': has_nan_mu,
                    'mu_has_inf': has_inf_mu,
                    'sigma_has_nan': has_nan_sigma,
                    'sigma_has_inf': has_inf_sigma,
                    'sigma_positive': sigma_positive,
                    'success': True
                }
                
                # Overall stability check
                is_stable = not (has_nan_mu or has_inf_mu or has_nan_sigma or has_inf_sigma) and sigma_positive
                case_result['is_stable'] = is_stable
                
                status = "✅" if is_stable else "❌"
                logger.info(f"  {status} {case_name}: stable={is_stable}")
                
            except Exception as e:
                case_result = {
                    'success': False,
                    'error': str(e),
                    'is_stable': False
                }
                logger.info(f"  ❌ {case_name}: ERROR - {e}")
            
            stability_results[case_name] = case_result
        
        # Overall stability assessment
        stable_cases = sum(1 for result in stability_results.values() 
                          if result.get('is_stable', False))
        total_cases = len(test_cases)
        stability_rate = stable_cases / total_cases
        
        stability_results['summary'] = {
            'stable_cases': stable_cases,
            'total_cases': total_cases,
            'stability_rate': stability_rate,
            'passes_requirement': stability_rate >= 0.75  # At least 75% stable
        }
        
        logger.info(f"  Overall stability: {stable_cases}/{total_cases} ({stability_rate:.1%})")
        
        return stability_results
    
    def validate_uncertainty_behavior(self) -> Dict[str, Any]:
        """Test uncertainty estimation behavior."""
        logger.info("\n📊 Testing uncertainty behavior...")
        
        uncertainty_results = {}
        
        # Test 1: Uncertainty should increase with input noise
        base_sequence = torch.randn(5, 48, 8).to(self.device)
        noise_levels = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
        uncertainties = []
        
        for noise_level in noise_levels:
            noisy_input = base_sequence + torch.randn_like(base_sequence) * noise_level
            
            with torch.no_grad():
                mu, sigma = self.model(noisy_input)
            
            avg_uncertainty = sigma.mean().item()
            uncertainties.append(avg_uncertainty)
            
            logger.info(f"  Noise level {noise_level:.1f}: uncertainty={avg_uncertainty:.4f}")
        
        # Check if uncertainty generally increases with noise
        uncertainty_correlation = np.corrcoef(noise_levels, uncertainties)[0, 1]
        uncertainty_results['noise_correlation'] = {
            'correlation': uncertainty_correlation,
            'noise_levels': noise_levels,
            'uncertainties': uncertainties,
            'is_increasing': uncertainty_correlation > 0.5
        }
        
        # Test 2: Attention pattern analysis
        with torch.no_grad():
            dummy_input = torch.randn(3, 48, 8).to(self.device)
            mu, sigma, attention_weights = self.model(
                dummy_input, 
                return_attention_weights=True
            )
        
        # Analyze attention patterns
        attention_entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1)
        attention_peak = attention_weights.max(dim=-1)[0]
        attention_uniformity = attention_weights.std(dim=-1)
        
        uncertainty_results['attention_analysis'] = {
            'avg_entropy': attention_entropy.mean().item(),
            'avg_peak': attention_peak.mean().item(),
            'avg_uniformity': attention_uniformity.mean().item(),
            'attention_sums_to_one': torch.allclose(attention_weights.sum(dim=-1), torch.ones(3)).item()
        }
        
        # Test 3: MC Dropout consistency
        self.model.train()  # Enable dropout for MC sampling
        mc_samples = []
        
        test_input = torch.randn(2, 48, 8).to(self.device)
        for _ in range(20):
            with torch.no_grad():
                mu, sigma = self.model(test_input)
                mc_samples.append(mu)
        
        mc_predictions = torch.stack(mc_samples)
        mc_mean = mc_predictions.mean(dim=0)
        mc_std = mc_predictions.std(dim=0)
        
        uncertainty_results['mc_dropout'] = {
            'mc_std_mean': mc_std.mean().item(),
            'mc_std_max': mc_std.max().item(),
            'prediction_variance': mc_std.var().item()
        }
        
        self.model.eval()  # Return to eval mode
        
        logger.info(f"  Noise-uncertainty correlation: {uncertainty_correlation:.3f}")
        logger.info(f"  Attention entropy: {attention_entropy.mean():.3f}")
        logger.info(f"  MC Dropout std: {mc_std.mean():.4f}")
        
        return uncertainty_results
    
    def validate_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage and efficiency."""
        logger.info("\n💾 Testing memory usage...")
        
        memory_results = {}
        
        if self.device.type == 'cuda':
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats(self.device)
            
            # Test different batch sizes
            batch_sizes = [1, 8, 16, 32, 64]
            memory_usage = []
            
            for batch_size in batch_sizes:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(self.device)
                
                test_input = torch.randn(batch_size, 48, 8).to(self.device)
                
                with torch.no_grad():
                    mu, sigma = self.model(test_input)
                
                peak_memory = torch.cuda.max_memory_allocated(self.device) / 1024**2  # MB
                memory_usage.append(peak_memory)
                
                logger.info(f"  Batch size {batch_size:2d}: {peak_memory:.2f} MB")
            
            memory_results['gpu_memory'] = {
                'batch_sizes': batch_sizes,
                'peak_memory_mb': memory_usage,
                'memory_per_sample': [mem/batch for mem, batch in zip(memory_usage, batch_sizes)]
            }
            
            # Check memory efficiency
            max_memory = max(memory_usage)
            memory_results['meets_memory_requirement'] = max_memory <= 512  # 512MB limit
            
        else:
            logger.info("  CPU mode - detailed memory tracking not available")
            memory_results['device'] = 'cpu'
        
        # Test for memory leaks
        initial_objects = len(torch.cuda.memory_summary().split('\n')) if self.device.type == 'cuda' else 0
        
        for _ in range(10):
            test_input = torch.randn(16, 48, 8).to(self.device)
            with torch.no_grad():
                mu, sigma = self.model(test_input)
            del test_input, mu, sigma
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            final_objects = len(torch.cuda.memory_summary().split('\n'))
            memory_results['memory_leak_check'] = {
                'initial_objects': initial_objects,
                'final_objects': final_objects,
                'objects_increased': final_objects > initial_objects * 1.1
            }
        
        return memory_results
    
    def validate_gradient_flow(self) -> Dict[str, Any]:
        """Test gradient flow and training capability."""
        logger.info("\n🎯 Testing gradient flow...")
        
        gradient_results = {}
        
        # Enable training mode
        self.model.train()
        
        # Create test data with targets
        test_input = torch.randn(4, 48, 8, requires_grad=True).to(self.device)
        target_mu = torch.randn(4, 64).to(self.device)
        target_sigma = torch.rand(4, 64).to(self.device) + 0.1  # Ensure positive
        
        # Forward pass
        mu, sigma = self.model(test_input)
        
        # Compute loss
        loss_mu = torch.nn.functional.mse_loss(mu, target_mu)
        loss_sigma = torch.nn.functional.mse_loss(sigma, target_sigma)
        total_loss = loss_mu + loss_sigma
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients
        grad_stats = {}
        params_with_grad = 0
        params_without_grad = 0
        grad_norms = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    params_with_grad += 1
                    grad_norm = param.grad.norm().item()
                    grad_norms.append(grad_norm)
                    
                    # Check for NaN gradients
                    if torch.isnan(param.grad).any():
                        gradient_results[f'nan_gradient_{name}'] = True
                        logger.warning(f"  ⚠️  NaN gradient in {name}")
                else:
                    params_without_grad += 1
                    logger.warning(f"  ⚠️  No gradient for {name}")
        
        gradient_results['gradient_stats'] = {
            'params_with_grad': params_with_grad,
            'params_without_grad': params_without_grad,
            'avg_grad_norm': np.mean(grad_norms) if grad_norms else 0.0,
            'max_grad_norm': np.max(grad_norms) if grad_norms else 0.0,
            'min_grad_norm': np.min(grad_norms) if grad_norms else 0.0
        }
        
        # Test gradient clipping
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        gradient_results['gradient_clipping'] = {
            'total_norm': total_norm.item(),
            'clipping_applied': total_norm > 1.0
        }
        
        logger.info(f"  Parameters with gradients: {params_with_grad}")
        logger.info(f"  Average gradient norm: {gradient_results['gradient_stats']['avg_grad_norm']:.6f}")
        logger.info(f"  Total gradient norm: {total_norm:.6f}")
        
        # Return to eval mode
        self.model.eval()
        
        return gradient_results
    
    def validate_consistency(self) -> Dict[str, Any]:
        """Test model consistency and reproducibility."""
        logger.info("\n🎲 Testing consistency and reproducibility...")
        
        consistency_results = {}
        
        # Test deterministic behavior
        torch.manual_seed(42)
        test_input = torch.randn(2, 48, 8).to(self.device)
        
        # Multiple runs with same seed
        outputs = []
        for _ in range(5):
            torch.manual_seed(42)  # Reset seed
            with torch.no_grad():
                mu, sigma = self.model(test_input)
                outputs.append((mu.clone(), sigma.clone()))
        
        # Check if outputs are identical
        all_identical = True
        for i in range(1, len(outputs)):
            mu_identical = torch.allclose(outputs[0][0], outputs[i][0])
            sigma_identical = torch.allclose(outputs[0][1], outputs[i][1])
            if not (mu_identical and sigma_identical):
                all_identical = False
                break
        
        consistency_results['deterministic'] = {
            'is_deterministic': all_identical,
            'runs_tested': len(outputs)
        }
        
        # Test output ranges
        large_batch = torch.randn(32, 48, 8).to(self.device)
        with torch.no_grad():
            mu_batch, sigma_batch = self.model(large_batch)
        
        consistency_results['output_ranges'] = {
            'mu_min': mu_batch.min().item(),
            'mu_max': mu_batch.max().item(),
            'mu_mean': mu_batch.mean().item(),
            'mu_std': mu_batch.std().item(),
            'sigma_min': sigma_batch.min().item(),
            'sigma_max': sigma_batch.max().item(),
            'sigma_mean': sigma_batch.mean().item(),
            'sigma_std': sigma_batch.std().item()
        }
        
        # Check if uncertainty is always positive
        sigma_positive = torch.all(sigma_batch > 0).item()
        consistency_results['sigma_positive'] = sigma_positive
        
        logger.info(f"  Deterministic behavior: {'✅' if all_identical else '❌'}")
        logger.info(f"  Mu range: [{mu_batch.min():.3f}, {mu_batch.max():.3f}]")
        logger.info(f"  Sigma range: [{sigma_batch.min():.6f}, {sigma_batch.max():.6f}]")
        logger.info(f"  Sigma always positive: {'✅' if sigma_positive else '❌'}")
        
        return consistency_results
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation tests."""
        logger.info("🔍 Starting comprehensive Structure Embedder validation...")
        
        if not self.initialize_model():
            return {'error': 'Failed to initialize model'}
        
        # Run all validation tests
        self.results['latency'] = self.validate_latency()
        self.results['numerical_stability'] = self.validate_numerical_stability()
        self.results['uncertainty_behavior'] = self.validate_uncertainty_behavior()
        self.results['memory_usage'] = self.validate_memory_usage()
        self.results['gradient_flow'] = self.validate_gradient_flow()
        self.results['consistency'] = self.validate_consistency()
        
        # Overall assessment
        self._assess_overall_readiness()
        
        return self.results
    
    def _assess_overall_readiness(self):
        """Assess overall production readiness."""
        logger.info("\n📋 Overall Production Readiness Assessment:")
        
        checks = []
        
        # Latency check
        latency_ok = self.results['latency'].get('meets_requirement', False)
        checks.append(('Latency ≤ 10ms', latency_ok))
        
        # Stability check
        stability_rate = self.results['numerical_stability']['summary']['stability_rate']
        stability_ok = stability_rate >= 0.75
        checks.append(('Numerical stability ≥ 75%', stability_ok))
        
        # Uncertainty behavior check
        uncertainty_corr = self.results['uncertainty_behavior']['noise_correlation']['correlation']
        uncertainty_ok = uncertainty_corr > 0.3
        checks.append(('Uncertainty correlation > 0.3', uncertainty_ok))
        
        # Memory check
        if 'gpu_memory' in self.results['memory_usage']:
            memory_ok = self.results['memory_usage'].get('meets_memory_requirement', True)
        else:
            memory_ok = True  # Assume OK for CPU
        checks.append(('Memory usage acceptable', memory_ok))
        
        # Gradient flow check
        grad_params = self.results['gradient_flow']['gradient_stats']['params_with_grad']
        gradient_ok = grad_params > 0
        checks.append(('Gradient flow working', gradient_ok))
        
        # Deterministic check
        deterministic_ok = self.results['consistency']['deterministic']['is_deterministic']
        checks.append(('Deterministic behavior', deterministic_ok))
        
        # Sigma positivity check
        sigma_ok = self.results['consistency']['sigma_positive']
        checks.append(('Uncertainty always positive', sigma_ok))
        
        # Print results
        passed_checks = 0
        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            logger.info(f"  {status} {check_name}")
            if passed:
                passed_checks += 1
        
        # Final verdict
        production_ready = passed_checks >= len(checks) * 0.8  # 80% pass rate
        
        self.results['production_assessment'] = {
            'total_checks': len(checks),
            'passed_checks': passed_checks,
            'pass_rate': passed_checks / len(checks),
            'production_ready': production_ready,
            'individual_checks': {name: passed for name, passed in checks}
        }
        
        verdict = "✅ PASSED" if production_ready else "❌ FAILED"
        logger.info(f"\n🎯 Production Readiness: {verdict} ({passed_checks}/{len(checks)} checks passed)")
        
        return production_ready


def main():
    """Main function to run validation."""
    parser = argparse.ArgumentParser(description="Validate Structure Embedder for Production")
    parser.add_argument('--config', type=str, help='Path to model configuration file')
    parser.add_argument('--output', type=str, default='validation_results.json', 
                       help='Output file for results')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Default configuration
    default_config = {
        'input_channels': 8,
        'output_dim': 64,
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 3,
        'd_ff': 512,
        'dropout_rate': 0.2,
        'max_seq_len': 48
    }
    
    # Load configuration if provided
    if args.config:
        try:
            import yaml
            with open(args.config, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Extract structure embedder config
            if 'structure_embedder' in config_data:
                model_config = config_data['structure_embedder'].get('architecture', {})
            elif 'main_core' in config_data:
                model_config = config_data['main_core']['embedders'].get('structure', {})
            else:
                model_config = config_data
                
            # Update default config
            default_config.update(model_config)
            logger.info(f"Loaded configuration from {args.config}")
            
        except Exception as e:
            logger.warning(f"Failed to load config file: {e}. Using default configuration.")
    
    # Run validation
    validator = StructureEmbedderValidator(default_config)
    results = validator.run_comprehensive_validation()
    
    # Save results
    try:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {args.output}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
    
    # Return appropriate exit code
    production_ready = results.get('production_assessment', {}).get('production_ready', False)
    return 0 if production_ready else 1


if __name__ == "__main__":
    exit(main())