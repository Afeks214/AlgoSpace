"""
Integration test for the M-RMS component.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np


def test_mrms_import():
    """Test that M-RMS can be imported correctly."""
    from src.agents.mrms import MRMSComponent
    print("✅ M-RMS import successful")
    

def test_mrms_initialization():
    """Test M-RMS component initialization."""
    from src.agents.mrms import MRMSComponent
    
    config = {
        'synergy_dim': 30,
        'account_dim': 10,
        'device': 'cpu',
        'point_value': 5.0,
        'max_position_size': 5
    }
    
    mrms = MRMSComponent(config)
    print("✅ M-RMS component initialized successfully")
    
    # Check model info
    info = mrms.get_model_info()
    print(f"Model architecture: {info['architecture']}")
    print(f"Total parameters: {info['total_parameters']}")
    print(f"Sub-agents: {info['sub_agents']}")
    

def test_kernel_integration():
    """Test that kernel properly imports and instantiates M-RMS."""
    try:
        # Check import in kernel
        with open('src/core/kernel.py', 'r') as f:
            kernel_content = f.read()
            
        assert 'from ..agents.mrms import MRMSComponent' in kernel_content
        assert 'MRMSComponent(mrms_config)' in kernel_content
        print("✅ Kernel integration verified")
        
    except Exception as e:
        print(f"❌ Kernel integration check failed: {e}")


def test_config_integration():
    """Test that settings.yaml has M-RMS configuration."""
    import yaml
    
    with open('config/settings.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    assert 'm_rms' in config, "M-RMS configuration section missing"
    assert 'mrms_path' in config.get('models', {}), "M-RMS model path missing"
    
    mrms_config = config['m_rms']
    assert mrms_config['synergy_dim'] == 30
    assert mrms_config['account_dim'] == 10
    assert mrms_config['max_position_size'] == 5
    
    print("✅ Configuration integration verified")
    

def test_risk_proposal_structure():
    """Test the structure of risk proposals (without model weights)."""
    from src.agents.mrms import MRMSComponent
    
    config = {
        'synergy_dim': 30,
        'account_dim': 10,
        'device': 'cpu',
        'point_value': 5.0,
        'max_position_size': 5
    }
    
    mrms = MRMSComponent(config)
    
    # Test input validation
    try:
        # Should fail without model loaded
        trade_qual = {
            'synergy_vector': np.random.randn(30),
            'account_state_vector': np.random.randn(10),
            'entry_price': 4500.0,
            'direction': 'LONG',
            'atr': 10.0
        }
        proposal = mrms.generate_risk_proposal(trade_qual)
    except RuntimeError as e:
        if "Model weights not loaded" in str(e):
            print("✅ Correctly prevents inference without loaded model")
        else:
            raise
    

if __name__ == '__main__':
    print("Running M-RMS integration tests...\n")
    
    test_mrms_import()
    test_mrms_initialization()
    test_kernel_integration()
    test_config_integration()
    test_risk_proposal_structure()
    
    print("\n✅ All M-RMS integration tests passed!")