#!/usr/bin/env python3
"""
Comprehensive verification of BiLSTM implementation.

This script performs thorough verification of the BiLSTM upgrade including
architecture checks, dimension handling, performance testing, and integration
validation.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import time
from datetime import datetime
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.main_core.models import TacticalEmbedder
from src.agents.main_core.tactical_bilstm_components import (
    BiLSTMGateController,
    TemporalPyramidPooling,
    BiLSTMPositionalEncoding,
    DirectionalFeatureFusion,
    BiLSTMTemporalMasking
)


def verify_bilstm_architecture():
    """Verify BiLSTM architecture is correctly implemented."""
    print("🔍 Verifying BiLSTM Architecture...")
    
    # Create model
    model = TacticalEmbedder(
        input_dim=7,
        hidden_dim=64,
        output_dim=48
    )
    
    # Check BiLSTM configuration
    info = model.get_bilstm_info()
    
    checks = {
        'is_bilstm': info['is_bilstm'] == True,
        'correct_output_dim': info['bilstm_output_dim'] == info['hidden_dim'] * 2,
        'has_gate_controller': info['has_gate_controller'],
        'has_pyramid_pooling': info['has_pyramid_pooling'],
        'has_positional_encoding': info['has_positional_encoding'],
        'has_directional_fusion': info['has_directional_fusion'],
        'has_temporal_masking': info['has_temporal_masking']
    }
    
    print("\n📋 Architecture Checks:")
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}: {passed}")
    
    # Additional architecture verification
    print("\n📊 Architecture Details:")
    print(f"  - Hidden dimension: {info['hidden_dim']}")
    print(f"  - BiLSTM output dimension: {info['bilstm_output_dim']}")
    print(f"  - Number of LSTM layers: {len(model.lstm_layers)}")
    
    return all(checks.values())


def verify_dimension_handling():
    """Verify proper dimension handling in BiLSTM."""
    print("\n🔍 Verifying Dimension Handling...")
    
    model = TacticalEmbedder()
    test_input = torch.randn(4, 60, 7)  # Batch of 4
    
    # Track dimensions through the network
    dimension_checks = {}
    
    # Input projection
    h = model.input_projection(test_input)
    dimension_checks['input_projection'] = h.shape == (4, 60, 128)
    
    # First LSTM layer
    h_with_pos = model.position_encoder(h)
    bilstm = model.lstm_layers[0]
    bilstm_out, (hn, cn) = bilstm(h_with_pos)
    
    dimension_checks['bilstm_output'] = bilstm_out.shape == (4, 60, 256)  # 128*2
    dimension_checks['hidden_state'] = hn.shape == (2, 4, 128)  # 2 directions
    dimension_checks['cell_state'] = cn.shape == (2, 4, 128)
    
    # Final output
    mu, sigma = model(test_input)
    dimension_checks['final_mu'] = mu.shape == (4, 48)
    dimension_checks['final_sigma'] = sigma.shape == (4, 48)
    
    print("\n📋 Dimension Checks:")
    for check, passed in dimension_checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}: {passed}")
    
    return all(dimension_checks.values())


def verify_component_functionality():
    """Verify each BiLSTM component works correctly."""
    print("\n🔍 Verifying Component Functionality...")
    
    component_checks = {}
    
    # Test BiLSTMGateController
    try:
        gate = BiLSTMGateController(64)
        forward = torch.randn(2, 30, 64)
        backward = torch.randn(2, 30, 64)
        gated = gate(forward, backward)
        component_checks['gate_controller'] = gated.shape == (2, 30, 128)
    except Exception as e:
        component_checks['gate_controller'] = False
        print(f"  ❌ Gate controller error: {e}")
    
    # Test TemporalPyramidPooling
    try:
        pyramid = TemporalPyramidPooling(128)
        bilstm_out = torch.randn(2, 60, 128)
        pooled = pyramid(bilstm_out)
        component_checks['pyramid_pooling'] = pooled.shape == (2, 128)
    except Exception as e:
        component_checks['pyramid_pooling'] = False
        print(f"  ❌ Pyramid pooling error: {e}")
    
    # Test BiLSTMPositionalEncoding
    try:
        pos_enc = BiLSTMPositionalEncoding(64)
        forward = torch.randn(2, 60, 64)
        backward = torch.randn(2, 60, 64)
        f_enc, b_enc = pos_enc(forward, backward)
        component_checks['positional_encoding'] = (
            f_enc.shape == forward.shape and b_enc.shape == backward.shape
        )
    except Exception as e:
        component_checks['positional_encoding'] = False
        print(f"  ❌ Positional encoding error: {e}")
    
    # Test DirectionalFeatureFusion
    try:
        fusion = DirectionalFeatureFusion(64)
        forward = torch.randn(2, 60, 64)
        backward = torch.randn(2, 60, 64)
        fused = fusion(forward, backward)
        component_checks['directional_fusion'] = fused.shape == (2, 60, 64)
    except Exception as e:
        component_checks['directional_fusion'] = False
        print(f"  ❌ Directional fusion error: {e}")
    
    # Test BiLSTMTemporalMasking
    try:
        masking = BiLSTMTemporalMasking(64)
        bilstm_out = torch.randn(2, 60, 128)
        masked = masking(bilstm_out)
        component_checks['temporal_masking'] = masked.shape == bilstm_out.shape
    except Exception as e:
        component_checks['temporal_masking'] = False
        print(f"  ❌ Temporal masking error: {e}")
    
    print("\n📋 Component Checks:")
    for check, passed in component_checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}: {passed}")
    
    return all(component_checks.values())


def verify_performance():
    """Verify performance meets requirements."""
    print("\n⚡ Verifying Performance...")
    
    model = TacticalEmbedder()
    model.eval()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    test_input = torch.randn(32, 60, 7).to(device)  # Larger batch
    
    # Warm up
    for _ in range(10):
        _ = model(test_input)
    
    # Benchmark
    times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = model(test_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    max_time = np.max(times)
    min_time = np.min(times)
    
    print(f"\n📊 Performance Metrics:")
    print(f"  - Average: {avg_time:.2f}ms ± {std_time:.2f}ms")
    print(f"  - Min/Max: {min_time:.2f}ms / {max_time:.2f}ms")
    print(f"  - Device: {device}")
    print(f"  - Batch size: 32")
    
    # Performance criteria (relaxed for CPU)
    if device.type == 'cuda':
        passed = avg_time < 5.0  # 5ms for GPU
    else:
        passed = avg_time < 20.0  # 20ms for CPU
    
    status = "✅" if passed else "❌"
    print(f"\n  {status} Meets performance requirement: {passed}")
    
    return passed


def verify_gradient_flow():
    """Verify gradients flow properly through BiLSTM."""
    print("\n🔄 Verifying Gradient Flow...")
    
    model = TacticalEmbedder()
    model.train()
    
    test_input = torch.randn(2, 60, 7, requires_grad=True)
    
    # Forward pass
    mu, sigma = model(test_input)
    loss = mu.mean() + sigma.mean()
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    gradient_checks = {}
    
    # Check input gradients
    gradient_checks['input_gradients'] = (
        test_input.grad is not None and 
        not torch.isnan(test_input.grad).any() and
        test_input.grad.abs().mean() > 0
    )
    
    # Check BiLSTM parameter gradients
    bilstm_grad_ok = True
    for i, lstm in enumerate(model.lstm_layers):
        for name, param in lstm.named_parameters():
            if param.requires_grad and param.grad is not None:
                if torch.isnan(param.grad).any() or param.grad.abs().mean() == 0:
                    bilstm_grad_ok = False
                    break
    
    gradient_checks['bilstm_gradients'] = bilstm_grad_ok
    
    # Check enhancement component gradients
    enhancement_grad_ok = True
    for comp_name in ['gate_controller', 'pyramid_pooling', 'directional_fusion']:
        if hasattr(model, comp_name):
            component = getattr(model, comp_name)
            for param in component.parameters():
                if param.requires_grad and param.grad is not None:
                    if torch.isnan(param.grad).any():
                        enhancement_grad_ok = False
                        break
    
    gradient_checks['enhancement_gradients'] = enhancement_grad_ok
    
    print("\n📋 Gradient Checks:")
    for check, passed in gradient_checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}: {passed}")
    
    return all(gradient_checks.values())


def verify_checkpoint_compatibility():
    """Verify checkpoint save/load functionality."""
    print("\n💾 Verifying Checkpoint Compatibility...")
    
    # Create and save model
    model1 = TacticalEmbedder()
    test_input = torch.randn(1, 60, 7)
    mu1, sigma1 = model1(test_input)
    
    # Save checkpoint
    checkpoint = {
        'model_state_dict': model1.state_dict(),
        'architecture': 'BiLSTM-Enhanced',
        'timestamp': datetime.now().isoformat(),
        'bilstm_info': model1.get_bilstm_info()
    }
    
    temp_path = '/tmp/test_bilstm_checkpoint.pth'
    torch.save(checkpoint, temp_path)
    
    # Load into new model
    model2 = TacticalEmbedder()
    model2.load_state_dict(checkpoint['model_state_dict'])
    model2.eval()
    
    # Verify outputs match
    mu2, sigma2 = model2(test_input)
    outputs_match = (
        torch.allclose(mu1, mu2, atol=1e-6) and
        torch.allclose(sigma1, sigma2, atol=1e-6)
    )
    
    print(f"\n  ✅ Checkpoint saved successfully")
    print(f"  {'✅' if outputs_match else '❌'} Outputs match after reload: {outputs_match}")
    
    # Clean up
    Path(temp_path).unlink(missing_ok=True)
    
    return outputs_match


def verify_mc_dropout():
    """Verify MC Dropout functionality."""
    print("\n🎲 Verifying MC Dropout...")
    
    model = TacticalEmbedder()
    test_input = torch.randn(2, 60, 7)
    
    # Enable MC dropout
    model.enable_mc_dropout()
    
    # Run multiple forward passes
    outputs = []
    for _ in range(20):
        mu, _ = model(test_input)
        outputs.append(mu)
    
    outputs = torch.stack(outputs)
    
    # Check variance across runs
    variance = outputs.var(dim=0)
    mean_variance = variance.mean().item()
    
    mc_dropout_working = mean_variance > 1e-6
    
    print(f"\n  {'✅' if mc_dropout_working else '❌'} MC Dropout produces variation")
    print(f"  - Mean variance: {mean_variance:.6f}")
    print(f"  - Variance range: [{variance.min():.6f}, {variance.max():.6f}]")
    
    return mc_dropout_working


def generate_verification_report(results: dict):
    """Generate a comprehensive verification report."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'results': results,
        'overall_status': all(results.values())
    }
    
    report_path = '/tmp/bilstm_verification_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Verification report saved to: {report_path}")
    return report


def main():
    """Run all verification tests."""
    print("="*60)
    print("BiLSTM Upgrade Verification Suite")
    print("="*60)
    
    tests = [
        ("Architecture", verify_bilstm_architecture),
        ("Dimensions", verify_dimension_handling),
        ("Components", verify_component_functionality),
        ("Performance", verify_performance),
        ("Gradients", verify_gradient_flow),
        ("Checkpoints", verify_checkpoint_compatibility),
        ("MC Dropout", verify_mc_dropout)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} test failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    # Generate report
    report = generate_verification_report(results)
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED - BiLSTM upgrade is complete!")
        print("\nNext steps:")
        print("1. Run migration script on existing checkpoints")
        print("2. Update training configurations")
        print("3. Retrain or fine-tune with BiLSTM architecture")
        print("4. Monitor performance improvements")
    else:
        print("\n❌ SOME TESTS FAILED - Please fix issues before deployment!")
        sys.exit(1)


if __name__ == '__main__':
    main()