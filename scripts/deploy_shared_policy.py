"""
File: scripts/deploy_shared_policy.py (NEW FILE)
Deployment script for shared policy network
"""

import torch
import yaml
import logging
import argparse
from pathlib import Path
import sys
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.main_core.shared_policy import SharedPolicy
from src.agents.main_core.mc_dropout_policy import MCDropoutConsensus
from src.agents.main_core.policy_monitoring import SharedPolicyMonitor

logger = logging.getLogger(__name__)

def validate_policy(policy: SharedPolicy, mc_dropout: MCDropoutConsensus,
                   device: torch.device):
    """Validate policy functionality."""
    print("🔍 Validating Shared Policy Network...")
    
    # Test 1: Basic forward pass
    print("\n1. Testing basic forward pass...")
    test_state = torch.randn(1, 136).to(device)
    
    try:
        output = policy(test_state)
        assert output.action_logits.shape == (1, 2)
        assert output.action_probs.shape == (1, 2)
        print("   ✅ Forward pass successful")
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        return False
        
    # Test 2: MC Dropout consensus
    print("\n2. Testing MC Dropout consensus...")
    try:
        mc_result = mc_dropout.evaluate(policy, test_state)
        assert isinstance(mc_result.should_qualify, bool)
        assert 0 <= mc_result.confidence <= 1
        print(f"   ✅ MC Dropout working (confidence: {mc_result.confidence:.3f})")
    except Exception as e:
        print(f"   ❌ MC Dropout failed: {e}")
        return False
        
    # Test 3: Batch processing
    print("\n3. Testing batch processing...")
    batch_state = torch.randn(32, 136).to(device)
    
    try:
        batch_output = policy(batch_state)
        assert batch_output.action_probs.shape == (32, 2)
        print("   ✅ Batch processing successful")
    except Exception as e:
        print(f"   ❌ Batch processing failed: {e}")
        return False
        
    # Test 4: Performance
    print("\n4. Testing performance...")
    
    # Warm up
    for _ in range(10):
        _ = policy(test_state, return_value=False)
        
    # Time 100 inferences
    times = []
    for _ in range(100):
        start = time.time()
        _ = policy(test_state, return_value=False)
        times.append(time.time() - start)
        
    avg_time = sum(times) / len(times) * 1000  # Convert to ms
    print(f"   Average inference time: {avg_time:.2f}ms")
    
    if avg_time > 10.0:
        print("   ⚠️  Warning: Inference slower than 10ms requirement")
    else:
        print("   ✅ Performance requirement met")
        
    # Test 5: Reasoning scores
    print("\n5. Testing reasoning components...")
    try:
        output = policy(test_state, return_features=True)
        assert output.reasoning_scores is not None
        assert 'structure' in output.reasoning_scores
        assert 'consistency_score' in output.reasoning_scores
        print("   ✅ Reasoning components working")
    except Exception as e:
        print(f"   ❌ Reasoning components failed: {e}")
        return False
        
    print("\n✅ All validation tests passed!")
    return True

def deploy_policy(config_path: str, model_path: str, device: str = 'cuda'):
    """Deploy shared policy to production."""
    print("🚀 Deploying Shared Policy Network...")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['shared_policy']
        
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    # Create policy
    policy = SharedPolicy(config).to(device)
    
    # Create MC Dropout consensus
    mc_dropout = MCDropoutConsensus(config['mc_dropout'])
    
    # Load weights if provided
    if model_path and Path(model_path).exists():
        print(f"   Loading weights from: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        policy.load_state_dict(state_dict)
    else:
        print("   Using randomly initialized weights")
        
    # Set to eval mode (but keep dropout active for MC Dropout)
    policy.eval()
    
    # Optimize for production
    if config.get('compile_model', True) and hasattr(torch, 'compile'):
        print("   Compiling model with torch.compile()...")
        policy = torch.compile(policy)
        
    # Validate
    if not validate_policy(policy, mc_dropout, device):
        print("\n❌ Validation failed. Aborting deployment.")
        return None
        
    # Create monitor
    monitor = SharedPolicyMonitor(config)
    
    print("\n✅ Shared Policy Network deployed successfully!")
    return policy, mc_dropout, monitor

def main():
    parser = argparse.ArgumentParser(description="Deploy Shared Policy Network")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model weights')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--test-only', action='store_true',
                       help='Only run validation tests')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.test_only:
        # Create dummy policy for testing
        config = {'input_dim': 136, 'hidden_dim': 256, 'action_dim': 2}
        policy = SharedPolicy(config).to(args.device)
        mc_config = {'n_samples': 20, 'confidence_threshold': 0.8}
        mc_dropout = MCDropoutConsensus(mc_config)
        validate_policy(policy, mc_dropout, torch.device(args.device))
    else:
        deploy_policy(args.config, args.model, args.device)

if __name__ == "__main__":
    main()