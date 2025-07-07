"""
File: scripts/deploy_regime_embedder.py (NEW FILE)
Deployment script for regime embedder
"""

import torch
import yaml
import logging
import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.main_core.models import RegimeEmbedder
from src.agents.main_core.regime_monitoring import RegimeEmbedderMonitor

logger = logging.getLogger(__name__)

def validate_embedder(embedder: RegimeEmbedder, device: torch.device):
    """Validate embedder functionality."""
    print("🔍 Validating Regime Embedder...")
    
    # Test 1: Basic forward pass
    print("\n1. Testing forward pass...")
    test_regime = torch.randn(1, 8).to(device)
    
    try:
        embedding = embedder(test_regime)
        assert embedding.shape == (1, 16)
        print("   ✅ Forward pass successful")
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        return False
        
    # Test 2: Batch processing
    print("\n2. Testing batch processing...")
    batch_regime = torch.randn(32, 8).to(device)
    
    try:
        batch_embedding = embedder(batch_regime)
        assert batch_embedding.shape == (32, 16)
        print("   ✅ Batch processing successful")
    except Exception as e:
        print(f"   ❌ Batch processing failed: {e}")
        return False
        
    # Test 3: Performance
    print("\n3. Testing performance...")
    import time
    
    # Warm up
    for _ in range(10):
        _ = embedder(test_regime)
        
    # Time 100 inferences
    times = []
    for _ in range(100):
        start = time.time()
        _ = embedder(test_regime)
        times.append(time.time() - start)
        
    avg_time = sum(times) / len(times) * 1000  # Convert to ms
    print(f"   Average inference time: {avg_time:.2f}ms")
    
    if avg_time > 2.0:
        print("   ⚠️  Warning: Inference slower than 2ms requirement")
    else:
        print("   ✅ Performance requirement met")
        
    # Test 4: Uncertainty outputs
    print("\n4. Testing uncertainty quantification...")
    try:
        mean, std = embedder.get_embedding_with_uncertainty(test_regime)
        assert mean.shape == (1, 16)
        assert std.shape == (1, 16)
        assert (std > 0).all()
        print("   ✅ Uncertainty quantification working")
    except Exception as e:
        print(f"   ❌ Uncertainty quantification failed: {e}")
        return False
        
    print("\n✅ All validation tests passed!")
    return True

def deploy_embedder(config_path: str, model_path: str, device: str = 'cuda'):
    """Deploy regime embedder to production."""
    print("🚀 Deploying Regime Embedder...")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['regime_embedder']
        
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    # Create embedder
    embedder = RegimeEmbedder(
        input_dim=config['regime_dim'],
        output_dim=config['output_dim'],
        hidden_dim=config['hidden_dim'],
        buffer_size=config['buffer_size'],
        n_heads=config['n_heads'],
        dropout=config['dropout'],
        n_patterns=config['n_patterns']
    ).to(device)
    
    # Load weights if provided
    if model_path and Path(model_path).exists():
        print(f"   Loading weights from: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        embedder.load_state_dict(state_dict)
    else:
        print("   Using randomly initialized weights")
        
    # Set to eval mode
    embedder.eval()
    
    # Optimize for production
    if config.get('compile_model', True) and hasattr(torch, 'compile'):
        print("   Compiling model with torch.compile()...")
        embedder = torch.compile(embedder)
        
    # Validate
    if not validate_embedder(embedder, device):
        print("\n❌ Validation failed. Aborting deployment.")
        return None
        
    # Create monitor
    monitor = RegimeEmbedderMonitor(config)
    
    print("\n✅ Regime Embedder deployed successfully!")
    return embedder, monitor

def main():
    parser = argparse.ArgumentParser(description="Deploy Regime Embedder")
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
        # Create dummy embedder for testing
        embedder = RegimeEmbedder().to(args.device)
        validate_embedder(embedder, torch.device(args.device))
    else:
        deploy_embedder(args.config, args.model, args.device)

if __name__ == "__main__":
    main()