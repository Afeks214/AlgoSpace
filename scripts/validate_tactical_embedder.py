#!/usr/bin/env python3
"""
Production validation script for the Tactical Embedder.

This script performs comprehensive validation to ensure the advanced BiLSTM
tactical embedder is ready for production deployment.
"""

import torch
import time
import numpy as np
from typing import Dict, Any
import json
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.main_core.models import TacticalEmbedder, MomentumAnalyzer


def validate_tactical_embedder_production() -> Dict[str, Any]:
    """Comprehensive validation for production deployment."""
    
    print("🔍 Validating Tactical Embedder for Production...")
    
    # Initialize model
    model = TacticalEmbedder(
        input_dim=7,
        hidden_dim=128,
        output_dim=48,
        n_layers=3,
        attention_scales=[5, 15, 30]
    )
    model.eval()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    results = {}
    
    # Test 1: Latency Testing
    print("\n⏱️  Testing inference latency...")
    
    # Enable optimized mode
    model.enable_inference_mode()
    
    latencies = []
    batch_sizes = [1, 8, 16, 32]
    
    for batch_size in batch_sizes:
        test_input = torch.randn(batch_size, 60, 7).to(device)
        
        # Warmup
        for _ in range(10):
            _ = model.forward_optimized(test_input)
            
        # Time inference
        torch.cuda.synchronize() if device.type == 'cuda' else None
        
        times = []
        for _ in range(100):
            start = time.perf_counter()
            
            with torch.no_grad():
                mu, sigma = model.forward_optimized(test_input)
                
            torch.cuda.synchronize() if device.type == 'cuda' else None
            times.append((time.perf_counter() - start) * 1000)  # ms
            
        avg_time = np.mean(times)
        p99_time = np.percentile(times, 99)
        
        print(f"  Batch size {batch_size}: avg={avg_time:.2f}ms, p99={p99_time:.2f}ms")
        
        results[f'latency_batch{batch_size}_avg'] = avg_time
        results[f'latency_batch{batch_size}_p99'] = p99_time
        
    # Test 2: Momentum Pattern Detection
    print("\n📊 Testing momentum pattern detection...")
    
    analyzer = MomentumAnalyzer()
    
    # Generate test sequences with known patterns
    test_patterns = {
        'acceleration': torch.cat([
            torch.linspace(0, 1, 30).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 7) * 0.5,
            torch.linspace(1, 2, 30).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 7)
        ], dim=1),
        'deceleration': torch.cat([
            torch.linspace(2, 1, 30).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 7),
            torch.linspace(1, 0.5, 30).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 7) * 0.5
        ], dim=1),
        'reversal': torch.cat([
            torch.linspace(0, 1, 30).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 7),
            torch.linspace(1, 0, 30).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 7)
        ], dim=1)
    }
    
    pattern_detection_accuracy = {}
    
    for pattern_name, pattern_data in test_patterns.items():
        pattern_data = pattern_data.to(device)
        
        with torch.no_grad():
            mu, sigma, attention_weights, lstm_states = model(
                pattern_data,
                return_attention_weights=True,
                return_all_states=True
            )
            
        metrics = analyzer.analyze(mu, attention_weights, lstm_states)
        detected_patterns = metrics.get('patterns', [])
        
        # Check if correct pattern was detected
        pattern_detected = pattern_name in detected_patterns
        pattern_detection_accuracy[pattern_name] = pattern_detected
        
        print(f"  {pattern_name}: {'✅ Detected' if pattern_detected else '❌ Not detected'}")
        
    results['pattern_detection'] = pattern_detection_accuracy
    
    # Test 3: Numerical Stability
    print("\n🔢 Testing numerical stability...")
    
    stability_tests = [
        ('zeros', torch.zeros(1, 60, 7)),
        ('ones', torch.ones(1, 60, 7)),
        ('large', torch.ones(1, 60, 7) * 100),
        ('small', torch.ones(1, 60, 7) * 1e-6),
        ('high_variance', torch.randn(1, 60, 7) * 10),
        ('momentum_spike', torch.cat([
            torch.ones(1, 50, 7) * 0.1,
            torch.ones(1, 10, 7) * 10
        ], dim=1))
    ]
    
    for test_name, test_input in stability_tests:
        test_input = test_input.to(device)
        
        with torch.no_grad():
            mu, sigma = model(test_input)
            
        has_nan = torch.isnan(mu).any() or torch.isnan(sigma).any()
        has_inf = torch.isinf(mu).any() or torch.isinf(sigma).any()
        
        print(f"  {test_name}: NaN={has_nan}, Inf={has_inf}")
        
        results[f'stability_{test_name}_nan'] = has_nan.item()
        results[f'stability_{test_name}_inf'] = has_inf.item()
        
    # Test 4: MC Dropout Consistency
    print("\n🎲 Testing MC Dropout consistency...")
    
    test_input = torch.randn(4, 60, 7).to(device)
    mc_predictions = []
    
    model.train()  # Enable dropout
    for _ in range(20):
        with torch.no_grad():
            mu, _ = model(test_input)
            mc_predictions.append(mu)
            
    mc_predictions = torch.stack(mc_predictions)
    mc_mean = mc_predictions.mean(dim=0)
    mc_std = mc_predictions.std(dim=0)
    
    # Check consistency
    cv = (mc_std / (mc_mean.abs() + 1e-8)).mean()
    print(f"  Coefficient of variation: {cv:.3f}")
    print(f"  Average uncertainty: {mc_std.mean():.3f}")
    
    results['mc_dropout_cv'] = cv.item()
    results['mc_dropout_avg_uncertainty'] = mc_std.mean().item()
    
    # Test 5: Memory Usage
    print("\n💾 Testing memory usage...")
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        
        # Run inference with large batch
        large_batch = torch.randn(32, 60, 7).to(device)
        
        with torch.no_grad():
            _ = model(large_batch)
            
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        print(f"  Peak GPU memory: {peak_memory:.2f} MB")
        
        results['peak_memory_mb'] = peak_memory
        
    # Test 6: Architecture Validation
    print("\n🏗️  Testing architecture components...")
    
    # Test all return modes
    test_input = torch.randn(2, 60, 7).to(device)
    
    # Basic forward
    mu, sigma = model(test_input)
    results['basic_forward_success'] = True
    
    # With attention weights
    mu, sigma, attention = model(test_input, return_attention_weights=True)
    results['attention_weights_success'] = attention.shape == (2, 60)
    
    # With all states
    mu, sigma, states = model(test_input, return_all_states=True)
    results['all_states_success'] = len(states) == 3
    
    # Both modes
    mu, sigma, attention, states = model(test_input, return_attention_weights=True, return_all_states=True)
    results['combined_modes_success'] = len(states) == 3 and attention.shape == (2, 60)
    
    print(f"  Basic forward: {'✅' if results['basic_forward_success'] else '❌'}")
    print(f"  Attention weights: {'✅' if results['attention_weights_success'] else '❌'}")
    print(f"  All states return: {'✅' if results['all_states_success'] else '❌'}")
    print(f"  Combined modes: {'✅' if results['combined_modes_success'] else '❌'}")
    
    # Test 7: Gradient Flow
    print("\n🔄 Testing gradient flow...")
    
    model.train()
    test_input = torch.randn(2, 60, 7, requires_grad=True).to(device)
    
    mu, sigma = model(test_input)
    loss = mu.mean() + sigma.mean()
    loss.backward()
    
    gradient_flow_good = True
    zero_grad_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                gradient_flow_good = False
                zero_grad_params.append(name)
            elif torch.isnan(param.grad).any():
                gradient_flow_good = False
                zero_grad_params.append(f"{name} (NaN)")
                
    results['gradient_flow_success'] = gradient_flow_good
    results['zero_grad_params'] = zero_grad_params
    
    print(f"  Gradient flow: {'✅' if gradient_flow_good else '❌'}")
    if zero_grad_params:
        print(f"  Issues with: {zero_grad_params}")
    
    # Final verdict
    print("\n✅ Validation Summary:")
    
    passed = True
    
    # Check latency requirement
    if results.get('latency_batch1_p99', float('inf')) > 5:
        print("  ❌ Latency too high (>5ms)")
        passed = False
    else:
        print("  ✅ Latency acceptable")
        
    # Check pattern detection
    pattern_accuracy = sum(results['pattern_detection'].values()) / len(results['pattern_detection'])
    if pattern_accuracy < 0.66:
        print("  ❌ Pattern detection accuracy too low")
        passed = False
    else:
        print("  ✅ Pattern detection working")
        
    # Check stability
    stability_issues = any(
        results.get(f'stability_{case}_nan', True) or 
        results.get(f'stability_{case}_inf', True)
        for case in ['zeros', 'ones', 'large', 'small', 'high_variance', 'momentum_spike']
    )
    
    if stability_issues:
        print("  ❌ Numerical stability issues detected")
        passed = False
    else:
        print("  ✅ Numerically stable")
        
    # Check MC Dropout
    if results.get('mc_dropout_cv', 1.0) > 0.5:
        print("  ⚠️  MC Dropout variance may be too high")
    else:
        print("  ✅ MC Dropout well-calibrated")
        
    # Check architecture components
    arch_tests = ['basic_forward_success', 'attention_weights_success', 'all_states_success', 'combined_modes_success']
    if not all(results.get(test, False) for test in arch_tests):
        print("  ❌ Architecture component issues")
        passed = False
    else:
        print("  ✅ All architecture components working")
        
    # Check gradient flow
    if not results.get('gradient_flow_success', False):
        print("  ❌ Gradient flow issues detected")
        passed = False
    else:
        print("  ✅ Gradient flow working correctly")
        
    results['production_ready'] = passed
    
    return results


def benchmark_inference_performance():
    """Benchmark detailed inference performance."""
    print("\n🚀 Running detailed performance benchmark...")
    
    model = TacticalEmbedder(
        input_dim=7,
        hidden_dim=128,
        output_dim=48,
        n_layers=3,
        attention_scales=[5, 15, 30]
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Benchmark different configurations
    configs = [
        {'batch_size': 1, 'name': 'Single sample'},
        {'batch_size': 8, 'name': 'Small batch'},
        {'batch_size': 32, 'name': 'Large batch'},
        {'batch_size': 64, 'name': 'Very large batch'}
    ]
    
    benchmark_results = {}
    
    for config in configs:
        batch_size = config['batch_size']
        name = config['name']
        
        test_input = torch.randn(batch_size, 60, 7).to(device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(test_input)
        
        # Benchmark regular forward
        times_regular = []
        for _ in range(100):
            start = time.perf_counter()
            with torch.no_grad():
                mu, sigma = model(test_input)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            times_regular.append((time.perf_counter() - start) * 1000)
            
        # Benchmark optimized forward
        model.enable_inference_mode()
        times_optimized = []
        for _ in range(100):
            start = time.perf_counter()
            with torch.no_grad():
                mu, sigma = model.forward_optimized(test_input)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            times_optimized.append((time.perf_counter() - start) * 1000)
        
        benchmark_results[name] = {
            'batch_size': batch_size,
            'regular_avg_ms': np.mean(times_regular),
            'regular_p99_ms': np.percentile(times_regular, 99),
            'optimized_avg_ms': np.mean(times_optimized),
            'optimized_p99_ms': np.percentile(times_optimized, 99),
            'speedup': np.mean(times_regular) / np.mean(times_optimized)
        }
        
        print(f"  {name} (batch={batch_size}):")
        print(f"    Regular: {benchmark_results[name]['regular_avg_ms']:.2f}ms avg, {benchmark_results[name]['regular_p99_ms']:.2f}ms p99")
        print(f"    Optimized: {benchmark_results[name]['optimized_avg_ms']:.2f}ms avg, {benchmark_results[name]['optimized_p99_ms']:.2f}ms p99")
        print(f"    Speedup: {benchmark_results[name]['speedup']:.2f}x")
    
    return benchmark_results


def validate_uncertainty_calibration():
    """Test uncertainty calibration quality."""
    print("\n🎯 Testing uncertainty calibration...")
    
    model = TacticalEmbedder(
        input_dim=7,
        hidden_dim=128,
        output_dim=48,
        n_layers=3,
        attention_scales=[5, 15, 30]
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Generate test data with known noise levels
    noise_levels = [0.001, 0.01, 0.1, 1.0]
    calibration_results = {}
    
    for noise_level in noise_levels:
        # Generate clean signal
        clean_signal = torch.randn(10, 60, 7).to(device)
        
        # Add noise
        noisy_signal = clean_signal + torch.randn_like(clean_signal) * noise_level
        
        with torch.no_grad():
            # Get predictions for both
            mu_clean, sigma_clean = model(clean_signal)
            mu_noisy, sigma_noisy = model(noisy_signal)
            
            # Calculate prediction error
            prediction_error = torch.abs(mu_clean - mu_noisy)
            predicted_uncertainty = sigma_noisy
            
            # Check if uncertainty correlates with error
            error_flat = prediction_error.flatten()
            uncertainty_flat = predicted_uncertainty.flatten()
            
            # Calculate correlation
            if len(error_flat) > 1:
                correlation = torch.corrcoef(torch.stack([error_flat, uncertainty_flat]))[0, 1]
            else:
                correlation = torch.tensor(0.0)
            
            calibration_results[f'noise_{noise_level}'] = {
                'avg_error': prediction_error.mean().item(),
                'avg_uncertainty': predicted_uncertainty.mean().item(),
                'correlation': correlation.item() if not torch.isnan(correlation) else 0.0
            }
            
            print(f"  Noise level {noise_level}: error={prediction_error.mean():.4f}, uncertainty={predicted_uncertainty.mean():.4f}, corr={correlation:.3f}")
    
    return calibration_results


def main():
    """Main validation function."""
    print("🎯 Starting Tactical Embedder Production Validation")
    print("=" * 60)
    
    # Run main validation
    validation_results = validate_tactical_embedder_production()
    
    # Run performance benchmark
    benchmark_results = benchmark_inference_performance()
    
    # Run uncertainty calibration test
    calibration_results = validate_uncertainty_calibration()
    
    # Combine all results
    all_results = {
        'validation': validation_results,
        'benchmark': benchmark_results,
        'calibration': calibration_results,
        'timestamp': time.time(),
        'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    }
    
    # Save results
    output_file = 'tactical_embedder_validation.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
        
    print(f"\n📊 Results saved to {output_file}")
    
    # Final status
    production_ready = validation_results.get('production_ready', False)
    status = "✅ PASSED" if production_ready else "❌ FAILED"
    
    print(f"\n{status} production validation")
    
    if not production_ready:
        print("\n⚠️  Issues detected:")
        for key, value in validation_results.items():
            if key.endswith('_success') and not value:
                print(f"   - {key}")
            elif 'stability' in key and value:
                print(f"   - {key}")
    
    return production_ready


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)