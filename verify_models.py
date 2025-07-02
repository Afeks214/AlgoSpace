import torch
import sys
sys.path.append('.')

def verify_model_dimensions():
    """Verify all models have correct input/output dimensions"""
    
    print("=== Main MARL Core Models ===")
    from src.agents.main_core.models import (
        StructureEmbedder, TacticalEmbedder, 
        RegimeEmbedder, LVNEmbedder,
        SharedPolicy, DecisionGate
    )
    
    # Test embedders
    structure_emb = StructureEmbedder()
    tactical_emb = TacticalEmbedder()
    regime_emb = RegimeEmbedder()
    lvn_emb = LVNEmbedder()
    
    # Test dimensions
    test_structure = torch.randn(1, 48, 8)  # 30-min matrix
    test_tactical = torch.randn(1, 60, 7)   # 5-min matrix
    test_regime = torch.randn(1, 8)         # Regime vector
    test_lvn = torch.randn(1, 5)            # LVN features
    
    out_structure = structure_emb(test_structure)
    out_tactical = tactical_emb(test_tactical)
    out_regime = regime_emb(test_regime)
    out_lvn = lvn_emb(test_lvn)
    
    print(f"Structure Embedder: {test_structure.shape} -> {out_structure.shape}")
    print(f"Tactical Embedder: {test_tactical.shape} -> {out_tactical.shape}")
    print(f"Regime Embedder: {test_regime.shape} -> {out_regime.shape}")
    print(f"LVN Embedder: {test_lvn.shape} -> {out_lvn.shape}")
    
    # Test unified state
    unified_state = torch.cat([out_structure, out_tactical, out_regime, out_lvn], dim=1)
    print(f"\nUnified State Vector: {unified_state.shape}")
    
    # Test SharedPolicy
    policy = SharedPolicy(input_dim=unified_state.shape[1])
    policy_out = policy(unified_state)
    print(f"SharedPolicy Output: {policy_out.shape} (should be [1, 2])")
    
    print("\n=== RDE Model ===")
    from src.agents.rde.model import RegimeDetectionEngine
    rde = RegimeDetectionEngine(input_dim=155)
    test_mmd = torch.randn(1, 100, 155)  # MMD sequence
    regime_vector = rde(test_mmd)
    print(f"RDE: {test_mmd.shape} -> {regime_vector.shape} (should be [1, 8])")
    
    print("\n=== M-RMS Models ===")
    from src.agents.mrms.models import RiskManagementEnsemble
    mrms = RiskManagementEnsemble(input_dim=40)
    test_state = torch.randn(1, 40)
    risk_output = mrms(test_state)
    print(f"M-RMS outputs: {risk_output.keys()}")
    
    return True

if __name__ == "__main__":
    verify_model_dimensions()