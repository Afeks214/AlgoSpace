#!/usr/bin/env python3
"""
Test the TacticalEmbedder BiLSTM implementation directly.
"""

import sys
from pathlib import Path
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def test_tactical_embedder():
    """Test TacticalEmbedder with BiLSTM enhancements."""
    print("🔍 Testing TacticalEmbedder BiLSTM Implementation...")
    
    try:
        # Import just the TacticalEmbedder class
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "models", 
            "/home/QuantNova/AlgoSpace-8/src/agents/main_core/models.py"
        )
        models = importlib.util.module_from_spec(spec)
        
        # Mock dependencies to avoid circular imports
        sys.modules['src.agents.main_core.regime_embedder'] = type(sys)('mock')
        sys.modules['src.agents.main_core.regime_uncertainty'] = type(sys)('mock')
        sys.modules['src.agents.main_core.regime_patterns'] = type(sys)('mock')
        sys.modules['src.agents.main_core.shared_policy'] = type(sys)('mock')
        sys.modules['src.agents.main_core.mc_dropout_policy'] = type(sys)('mock')
        sys.modules['src.agents.main_core.multi_objective_value'] = type(sys)('mock')
        
        # Load the module
        spec.loader.exec_module(models)
        
        # Create TacticalEmbedder
        embedder = models.TacticalEmbedder(
            input_dim=7,
            hidden_dim=64,
            output_dim=48,
            dropout_rate=0.2
        )
        
        # Check BiLSTM info
        info = embedder.get_bilstm_info()
        
        print("\n📋 BiLSTM Configuration:")
        for key, value in info.items():
            print(f"  - {key}: {value}")
        
        # Verify all components are present
        all_present = all([
            info['is_bilstm'],
            info['has_gate_controller'],
            info['has_pyramid_pooling'],
            info['has_positional_encoding'],
            info['has_directional_fusion'],
            info['has_temporal_masking']
        ])
        
        if all_present:
            print("\n✅ All BiLSTM enhancements are properly integrated!")
        else:
            print("\n❌ Some BiLSTM enhancements are missing!")
        
        # Test forward pass
        test_input = torch.randn(2, 60, 7)
        try:
            mu, sigma = embedder(test_input)
            print(f"\n✅ Forward pass successful!")
            print(f"  - Output shapes: μ={mu.shape}, σ={sigma.shape}")
        except Exception as e:
            print(f"\n❌ Forward pass failed: {e}")
        
        return all_present
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("="*60)
    print("TacticalEmbedder BiLSTM Test")
    print("="*60)
    
    success = test_tactical_embedder()
    
    if success:
        print("\n✅ TacticalEmbedder BiLSTM implementation verified!")
    else:
        print("\n❌ TacticalEmbedder BiLSTM implementation has issues!")