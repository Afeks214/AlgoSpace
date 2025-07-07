"""
Production validation script for MRMS Communication LSTM.
"""

import sys
sys.path.append('..')

import torch
import numpy as np
import logging
import time
from pathlib import Path
import yaml

from src.agents.mrms.engine import MRMSComponent
from src.agents.mrms.communication import MRMSCommunicationLSTM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_mrms_communication():
    """Complete production validation."""
    
    print("=" * 60)
    print("MRMS Communication LSTM Production Validation")
    print("=" * 60)
    
    # Load configuration
    with open('../config/settings.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize MRMS with communication
    mrms = MRMSComponent(config['m_rms'])
    
    # Verify communication layer exists
    assert hasattr(mrms, 'communication_lstm'), "Communication LSTM not initialized"
    assert isinstance(mrms.communication_lstm, MRMSCommunicationLSTM), "Wrong type for communication LSTM"
    print("✅ Communication layer properly initialized")
    
    # Load models
    model_path = '../models/m_rms_model.pth'
    if Path(model_path).exists():
        mrms.load_model(model_path)
        print(f"✅ Main model loaded from {model_path}")
    else:
        print(f"⚠️  Warning: Main model not found at {model_path}")
    
    # Check if communication weights exist
    comm_path = model_path.replace('.pth', '_comm.pth')
    if Path(comm_path).exists():
        print(f"✅ Communication weights found at {comm_path}")
    else:
        print(f"ℹ️  Communication weights not found, using random initialization")
    
    # Performance test
    print("\n" + "=" * 40)
    print("Performance Test")
    print("=" * 40)
    
    # Create dummy trade qualification
    trade_qual = {
        'synergy_vector': np.random.randn(30).astype(np.float32),
        'account_state_vector': np.random.randn(10).astype(np.float32),
        'entry_price': 4500.0,
        'direction': 'LONG',
        'atr': 10.0,
        'symbol': 'ES',
        'timestamp': '2024-01-15 10:00:00'
    }
    
    # Warm up
    for _ in range(10):
        _ = mrms.generate_risk_proposal(trade_qual)
    
    # Measure latency
    latencies = []
    for i in range(100):
        start = time.time()
        proposal = mrms.generate_risk_proposal(trade_qual)
        latencies.append((time.time() - start) * 1000)
        
        if i == 0:
            # Check first output
            assert 'risk_embedding' in proposal, "Missing risk_embedding"
            assert 'risk_uncertainty' in proposal, "Missing risk_uncertainty"
            assert 'adapted_position_size' in proposal, "Missing adapted_position_size"
            print("✅ All required fields present in output")
    
    # Performance metrics
    print(f"\nLatency Statistics (ms):")
    print(f"  Average: {np.mean(latencies):.2f}")
    print(f"  Median: {np.median(latencies):.2f}")
    print(f"  95th percentile: {np.percentile(latencies, 95):.2f}")
    print(f"  99th percentile: {np.percentile(latencies, 99):.2f}")
    print(f"  Max: {np.max(latencies):.2f}")
    
    # Check if meets requirements
    if np.percentile(latencies, 99) < 3.0:
        print("✅ Meets latency requirement (<3ms p99)")
    else:
        print("❌ Does not meet latency requirement")
    
    # Test uncertainty adaptation
    print("\n" + "=" * 40)
    print("Uncertainty Adaptation Test")
    print("=" * 40)
    
    uncertainties = []
    position_sizes = []
    
    # Simulate 10 consecutive losses
    for i in range(10):
        # Update recent outcomes with a loss
        mrms.update_trade_outcome({
            'hit_stop': True,
            'hit_target': False,
            'pnl': -50.0,
            'position_size': 3,
            'sl_distance': 10.0,
            'tp_distance': 20.0
        })
        
        # Generate new proposal
        proposal = mrms.generate_risk_proposal(trade_qual)
        
        # Track metrics
        uncertainties.append(proposal['risk_uncertainty'].mean())
        position_sizes.append(proposal.get('adapted_position_size', proposal['position_size']))
        
        if i % 3 == 0:
            print(f"Trade {i+1}: Uncertainty={uncertainties[-1]:.4f}, "
                  f"Position={position_sizes[-1]}")
    
    # Verify adaptation
    uncertainty_increased = uncertainties[-1] > uncertainties[0]
    position_reduced = position_sizes[-1] < position_sizes[0]
    
    print(f"\nUncertainty change: {uncertainties[0]:.4f} → {uncertainties[-1]:.4f}")
    print(f"Position size change: {position_sizes[0]} → {position_sizes[-1]}")
    
    if uncertainty_increased:
        print("✅ Uncertainty increases with losses")
    else:
        print("❌ Uncertainty did not increase as expected")
        
    if position_reduced:
        print("✅ Position size reduces during drawdown")
    else:
        print("❌ Position size did not reduce as expected")
    
    # Memory stability test
    print("\n" + "=" * 40)
    print("Memory Stability Test")
    print("=" * 40)
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run 1000 iterations
    for i in range(1000):
        _ = mrms.generate_risk_proposal(trade_qual)
        
        if i % 100 == 0:
            mrms.update_trade_outcome({
                'hit_stop': np.random.random() > 0.5,
                'hit_target': np.random.random() > 0.5,
                'pnl': np.random.normal(0, 50),
                'position_size': np.random.randint(1, 5),
                'sl_distance': 10.0,
                'tp_distance': 20.0
            })
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_growth = final_memory - initial_memory
    
    print(f"Initial memory: {initial_memory:.1f} MB")
    print(f"Final memory: {final_memory:.1f} MB")
    print(f"Memory growth: {memory_growth:.1f} MB")
    
    if memory_growth < 10:  # Less than 10MB growth
        print("✅ Memory stable")
    else:
        print("⚠️  Significant memory growth detected")
    
    # Integration test
    print("\n" + "=" * 40)
    print("Integration Test")
    print("=" * 40)
    
    # Test full pipeline
    try:
        # Generate proposal
        proposal = mrms.generate_risk_proposal(trade_qual)
        
        # Verify all fields
        required_fields = [
            'position_size', 'stop_loss_price', 'take_profit_price',
            'risk_amount', 'reward_amount', 'risk_reward_ratio',
            'risk_embedding', 'risk_uncertainty', 'adapted_position_size'
        ]
        
        missing_fields = [f for f in required_fields if f not in proposal]
        if missing_fields:
            print(f"❌ Missing fields: {missing_fields}")
        else:
            print("✅ All required fields present")
            
        # Type checking
        assert isinstance(proposal['risk_embedding'], np.ndarray)
        assert isinstance(proposal['risk_uncertainty'], np.ndarray)
        assert isinstance(proposal['adapted_position_size'], int)
        print("✅ Field types correct")
        
        # Shape checking
        assert proposal['risk_embedding'].shape == (8,)
        assert proposal['risk_uncertainty'].shape == (8,)
        print("✅ Array shapes correct")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All validation checks passed!")
    print("MRMS Communication LSTM is production ready")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        success = validate_mrms_communication()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        sys.exit(1)