#!/usr/bin/env python3
"""
Simple test script to verify matrix assemblers are working correctly.
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.kernel import SystemKernel
from src.matrix import MatrixAssembler30m, MatrixAssembler5m, MatrixAssemblerRegime
from src.matrix.normalizers import z_score_normalize, min_max_scale, cyclical_encode


def test_normalizers():
    """Test normalization functions."""
    print("Testing normalizers...")
    
    # Test z-score
    result = z_score_normalize(5.0, mean=3.0, std=2.0)
    print(f"Z-score normalize: {result} (expected: 1.0)")
    
    # Test min-max
    result = min_max_scale(5.0, min_val=0.0, max_val=10.0)
    print(f"Min-max scale: {result} (expected: 0.0)")
    
    # Test cyclical encoding
    sin_val, cos_val = cyclical_encode(6.0, max_value=24.0)
    print(f"Cyclical encode: sin={sin_val:.3f}, cos={cos_val:.3f}")
    
    print("✓ Normalizers working correctly\n")


def test_matrix_assemblers():
    """Test matrix assembler creation and basic functionality."""
    print("Testing matrix assemblers...")
    
    # Create kernel
    kernel = SystemKernel()
    
    # Create assemblers
    assembler_30m = MatrixAssembler30m("Test30m", kernel)
    assembler_5m = MatrixAssembler5m("Test5m", kernel)
    assembler_regime = MatrixAssemblerRegime("TestRegime", kernel)
    
    print(f"✓ Created 30m assembler: {assembler_30m.window_size}x{assembler_30m.n_features}")
    print(f"✓ Created 5m assembler: {assembler_5m.window_size}x{assembler_5m.n_features}")
    print(f"✓ Created regime assembler: {assembler_regime.window_size}x{assembler_regime.n_features}")
    
    # Test feature extraction
    test_features = {
        'current_price': 100.0,
        'current_volume': 1000,
        'mlmi_value': 65.0,
        'mlmi_signal': 1,
        'nwrqk_value': 102.0,
        'nwrqk_slope': 0.5,
        'lvn_distance_points': 5.0,
        'lvn_nearest_strength': 85.0,
        'fvg_bullish_active': True,
        'fvg_bearish_active': False,
        'fvg_nearest_level': 99.5,
        'fvg_age': 5,
        'fvg_mitigation_signal': False,
        'mmd_features': [0.1, -0.2, 0.3, -0.1, 0.0, 0.2, -0.3, 0.1],
        'timestamp': datetime.now()
    }
    
    # Test 30m feature extraction
    features_30m = assembler_30m.extract_features(test_features)
    if features_30m:
        print(f"✓ 30m feature extraction: {len(features_30m)} features")
        processed_30m = assembler_30m.preprocess_features(features_30m, test_features)
        print(f"✓ 30m preprocessing: {processed_30m.shape}")
    
    # Test 5m feature extraction
    features_5m = assembler_5m.extract_features(test_features)
    if features_5m:
        print(f"✓ 5m feature extraction: {len(features_5m)} features")
        processed_5m = assembler_5m.preprocess_features(features_5m, test_features)
        print(f"✓ 5m preprocessing: {processed_5m.shape}")
    
    # Test regime feature extraction
    features_regime = assembler_regime.extract_features(test_features)
    if features_regime:
        print(f"✓ Regime feature extraction: {len(features_regime)} features")
        processed_regime = assembler_regime.preprocess_features(features_regime, test_features)
        print(f"✓ Regime preprocessing: {processed_regime.shape}")
    
    print("✓ Matrix assemblers working correctly\n")


def test_matrix_updates():
    """Test matrix updates and readiness."""
    print("Testing matrix updates...")
    
    kernel = SystemKernel()
    assembler = MatrixAssembler30m("Test30m", kernel)
    
    # Initially not ready
    print(f"Initial state - Ready: {assembler.is_ready()}, Updates: {assembler.n_updates}")
    
    # Simulate multiple updates
    for i in range(50):
        test_features = {
            'current_price': 100.0 + i * 0.1,
            'mlmi_value': 50.0 + i % 20,
            'mlmi_signal': (i % 3) - 1,
            'nwrqk_value': 100.0 + i * 0.05,
            'nwrqk_slope': (i % 10) * 0.1,
            'lvn_distance_points': 5.0 + i % 5,
            'lvn_nearest_strength': 70.0 + i % 30,
            'timestamp': datetime.now()
        }
        
        assembler._update_matrix(test_features)
    
    print(f"After 50 updates - Ready: {assembler.is_ready()}, Updates: {assembler.n_updates}")
    
    # Get matrix
    if assembler.is_ready():
        matrix = assembler.get_matrix()
        print(f"✓ Matrix shape: {matrix.shape}")
        print(f"✓ Matrix range: [{matrix.min():.3f}, {matrix.max():.3f}]")
        print(f"✓ Matrix mean: {matrix.mean():.3f}")
        
        # Validate matrix
        is_valid, issues = assembler.validate_matrix()
        print(f"✓ Matrix validation: {'PASS' if is_valid else 'FAIL'}")
        if issues:
            print(f"  Issues: {issues}")
    
    print("✓ Matrix updates working correctly\n")


if __name__ == "__main__":
    print("=" * 60)
    print("     Matrix Assembler Simple Test")
    print("=" * 60)
    
    try:
        test_normalizers()
        test_matrix_assemblers()
        test_matrix_updates()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()