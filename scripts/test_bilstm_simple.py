#!/usr/bin/env python3
"""
Simple BiLSTM verification test without circular imports.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def test_bilstm_basic():
    """Test basic BiLSTM functionality."""
    print("🔍 Testing Basic BiLSTM Implementation...")
    
    # Create a simple BiLSTM
    input_dim = 7
    hidden_dim = 64
    batch_size = 4
    seq_len = 60
    
    bilstm = nn.LSTM(
        input_size=input_dim,
        hidden_size=hidden_dim,
        num_layers=2,
        batch_first=True,
        dropout=0.2,
        bidirectional=True
    )
    
    # Test input
    test_input = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    output, (hn, cn) = bilstm(test_input)
    
    # Check dimensions
    print("\n📋 Dimension Checks:")
    print(f"  - Input shape: {test_input.shape}")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Expected output: ({batch_size}, {seq_len}, {hidden_dim * 2})")
    print(f"  - Hidden state shape: {hn.shape}")
    print(f"  - Cell state shape: {cn.shape}")
    
    # Verify dimensions
    assert output.shape == (batch_size, seq_len, hidden_dim * 2), "Output dimension mismatch"
    assert hn.shape == (4, batch_size, hidden_dim), "Hidden state dimension mismatch"
    assert cn.shape == (4, batch_size, hidden_dim), "Cell state dimension mismatch"
    
    print("\n✅ Basic BiLSTM test passed!")
    
    # Test forward/backward split
    forward_features = output[:, :, :hidden_dim]
    backward_features = output[:, :, hidden_dim:]
    
    print("\n📋 Directional Features:")
    print(f"  - Forward features shape: {forward_features.shape}")
    print(f"  - Backward features shape: {backward_features.shape}")
    
    # Check they're different
    assert not torch.allclose(forward_features, backward_features), "Forward and backward should differ"
    print("  ✅ Forward and backward features are properly distinct")
    
    return True


def test_bilstm_components():
    """Test BiLSTM enhancement components."""
    print("\n🔍 Testing BiLSTM Components...")
    
    try:
        from src.agents.main_core.tactical_bilstm_components import (
            BiLSTMGateController,
            TemporalPyramidPooling,
            BiLSTMPositionalEncoding
        )
        
        # Test gate controller
        gate = BiLSTMGateController(64)
        forward = torch.randn(2, 30, 64)
        backward = torch.randn(2, 30, 64)
        gated = gate(forward, backward)
        assert gated.shape == (2, 30, 128), "Gate controller output mismatch"
        print("  ✅ BiLSTMGateController working")
        
        # Test pyramid pooling
        pyramid = TemporalPyramidPooling(128)
        pooled = pyramid(torch.randn(2, 60, 128))
        assert pooled.shape == (2, 128), "Pyramid pooling output mismatch"
        print("  ✅ TemporalPyramidPooling working")
        
        # Test positional encoding
        pos_enc = BiLSTMPositionalEncoding(64)
        f_enc, b_enc = pos_enc(forward, backward)
        assert f_enc.shape == forward.shape, "Positional encoding mismatch"
        print("  ✅ BiLSTMPositionalEncoding working")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Component test error: {e}")
        return False


def test_performance():
    """Test BiLSTM performance."""
    import time
    import numpy as np
    
    print("\n⚡ Testing BiLSTM Performance...")
    
    # Create BiLSTM
    bilstm = nn.LSTM(
        input_size=7,
        hidden_size=64,
        num_layers=2,
        batch_first=True,
        bidirectional=True
    )
    bilstm.eval()
    
    # Test data
    test_input = torch.randn(32, 60, 7)
    
    # Warm up
    for _ in range(10):
        _ = bilstm(test_input)
    
    # Benchmark
    times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = bilstm(test_input)
        times.append((time.perf_counter() - start) * 1000)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\n📊 Performance Metrics:")
    print(f"  - Average: {avg_time:.2f}ms ± {std_time:.2f}ms")
    print(f"  - Min/Max: {np.min(times):.2f}ms / {np.max(times):.2f}ms")
    
    # Reasonable performance check
    if avg_time < 50:  # 50ms is reasonable for CPU
        print("  ✅ Performance is acceptable")
        return True
    else:
        print("  ⚠️  Performance may need optimization")
        return False


def main():
    """Run simple BiLSTM tests."""
    print("="*60)
    print("Simple BiLSTM Verification")
    print("="*60)
    
    tests = [
        ("Basic BiLSTM", test_bilstm_basic),
        ("BiLSTM Components", test_bilstm_components),
        ("Performance", test_performance)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} failed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    if all(results.values()):
        print("\n✅ All tests passed! BiLSTM implementation is working.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")


if __name__ == '__main__':
    main()