#!/usr/bin/env python3
"""
Migration script to update existing checkpoints to BiLSTM architecture.

This script migrates existing TacticalEmbedder checkpoints to the enhanced
BiLSTM architecture while preserving as much of the learned weights as possible.
"""

import torch
import argparse
from pathlib import Path
import logging
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.main_core.models import TacticalEmbedder

logger = logging.getLogger(__name__)


def migrate_checkpoint(old_checkpoint_path: str, new_checkpoint_path: str, 
                      dry_run: bool = False) -> bool:
    """
    Migrate old LSTM checkpoint to BiLSTM architecture.
    
    Args:
        old_checkpoint_path: Path to old checkpoint
        new_checkpoint_path: Path to save migrated checkpoint
        dry_run: If True, only analyze without saving
        
    Returns:
        True if migration successful
    """
    print(f"🔄 Migrating checkpoint from {old_checkpoint_path}")
    
    # Load old checkpoint
    old_checkpoint = torch.load(old_checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(old_checkpoint, dict) and 'model_state_dict' in old_checkpoint:
        old_state = old_checkpoint['model_state_dict']
        metadata = {k: v for k, v in old_checkpoint.items() if k != 'model_state_dict'}
    else:
        old_state = old_checkpoint
        metadata = {}
    
    # Create new model with BiLSTM enhancements
    new_model = TacticalEmbedder(
        input_dim=7,
        hidden_dim=64,
        output_dim=48,
        dropout_rate=0.2
    )
    
    # Get new state dict
    new_state = new_model.state_dict()
    
    # Mapping old keys to new keys
    print("\n📋 Mapping old weights to new architecture...")
    
    mapped_count = 0
    skipped_count = 0
    new_components = []
    
    # Create mapping report
    mapping_report = {
        'timestamp': datetime.now().isoformat(),
        'old_checkpoint': old_checkpoint_path,
        'new_checkpoint': new_checkpoint_path,
        'mappings': [],
        'skipped': [],
        'new_components': []
    }
    
    # Map compatible weights
    for old_key, old_value in old_state.items():
        mapped = False
        
        # Direct mapping attempt
        if old_key in new_state:
            if old_value.shape == new_state[old_key].shape:
                new_state[old_key] = old_value
                mapped_count += 1
                mapping_report['mappings'].append({
                    'old': old_key,
                    'new': old_key,
                    'shape': list(old_value.shape)
                })
                print(f"  ✓ Direct map: {old_key}")
                mapped = True
        
        # Try mapping LSTM weights
        if not mapped and 'lstm' in old_key:
            # Handle LSTM layer mapping
            for i in range(3):  # Assuming 3 LSTM layers
                old_pattern = f'lstm_layers.{i}'
                if old_pattern in old_key:
                    new_key = old_key  # Keep same structure
                    if new_key in new_state and old_value.shape == new_state[new_key].shape:
                        new_state[new_key] = old_value
                        mapped_count += 1
                        mapping_report['mappings'].append({
                            'old': old_key,
                            'new': new_key,
                            'shape': list(old_value.shape)
                        })
                        print(f"  ✓ LSTM map: {old_key}")
                        mapped = True
                        break
        
        if not mapped:
            skipped_count += 1
            mapping_report['skipped'].append({
                'key': old_key,
                'shape': list(old_value.shape),
                'reason': 'No compatible mapping found'
            })
            print(f"  ⚠️  Skipped: {old_key} (shape: {old_value.shape})")
    
    # Identify new BiLSTM components
    for new_key in new_state.keys():
        if not any(new_key == m['new'] for m in mapping_report['mappings']):
            # Check if it's a BiLSTM enhancement component
            bilstm_components = [
                'gate_controller', 'bilstm_positional_encoding', 
                'pyramid_pooling', 'directional_fusion', 'temporal_masking'
            ]
            if any(comp in new_key for comp in bilstm_components):
                new_components.append(new_key)
                mapping_report['new_components'].append({
                    'key': new_key,
                    'shape': list(new_state[new_key].shape)
                })
    
    print(f"\n📊 Migration Summary:")
    print(f"  - Weights mapped: {mapped_count}")
    print(f"  - Weights skipped: {skipped_count}")
    print(f"  - New BiLSTM components: {len(new_components)}")
    
    if not dry_run:
        # Save migrated checkpoint
        migrated_checkpoint = {
            'model_state_dict': new_state,
            'architecture': 'BiLSTM-Enhanced',
            'migration_info': {
                'from': old_checkpoint_path,
                'mapped_weights': mapped_count,
                'skipped_weights': skipped_count,
                'new_components': len(new_components),
                'total_weights': len(new_state)
            },
            'original_metadata': metadata,
            'migration_timestamp': datetime.now().isoformat()
        }
        
        torch.save(migrated_checkpoint, new_checkpoint_path)
        print(f"\n✅ Migrated checkpoint saved to: {new_checkpoint_path}")
        
        # Save migration report
        report_path = new_checkpoint_path.replace('.pth', '_migration_report.json')
        with open(report_path, 'w') as f:
            json.dump(mapping_report, f, indent=2)
        print(f"📄 Migration report saved to: {report_path}")
    
    # Verify the migrated model loads correctly
    print("\n🔍 Verifying migrated checkpoint...")
    try:
        test_model = TacticalEmbedder()
        test_model.load_state_dict(new_state)
        print("  ✅ Checkpoint loads successfully!")
        
        # Test forward pass
        test_input = torch.randn(1, 60, 7)
        test_mu, test_sigma = test_model(test_input)
        print(f"  ✅ Forward pass successful! Output shapes: μ={test_mu.shape}, σ={test_sigma.shape}")
        
        # Verify BiLSTM components
        bilstm_info = test_model.get_bilstm_info()
        print("\n  📋 BiLSTM Components:")
        for key, value in bilstm_info.items():
            if key.startswith('has_'):
                print(f"    - {key}: {'✓' if value else '✗'}")
        
    except Exception as e:
        print(f"  ❌ Verification failed: {e}")
        return False
    
    return True


def analyze_checkpoint(checkpoint_path: str):
    """Analyze checkpoint structure without migration."""
    print(f"\n🔍 Analyzing checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("\n📋 Checkpoint metadata:")
        for key, value in checkpoint.items():
            if key != 'model_state_dict':
                print(f"  - {key}: {value}")
    else:
        state_dict = checkpoint
    
    print(f"\n📊 State dict analysis:")
    print(f"  - Total parameters: {len(state_dict)}")
    
    # Group by component
    components = {}
    for key in state_dict.keys():
        component = key.split('.')[0]
        if component not in components:
            components[component] = []
        components[component].append(key)
    
    print("\n📦 Components:")
    for comp, keys in components.items():
        print(f"  - {comp}: {len(keys)} parameters")
    
    # Check for BiLSTM indicators
    has_bilstm = any('bidirectional' in str(v.shape) for v in state_dict.values())
    print(f"\n🔄 BiLSTM detected: {'Yes' if has_bilstm else 'No'}")


def main():
    parser = argparse.ArgumentParser(description="Migrate Tactical Embedder to BiLSTM")
    parser.add_argument('--old-checkpoint', type=str, required=True,
                        help='Path to old checkpoint')
    parser.add_argument('--new-checkpoint', type=str,
                        help='Path to save migrated checkpoint')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze checkpoint without migration')
    parser.add_argument('--dry-run', action='store_true',
                        help='Perform migration analysis without saving')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify BiLSTM implementation')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.verify_only:
        # Just verify BiLSTM is working
        print("🔍 Verifying BiLSTM implementation...")
        model = TacticalEmbedder()
        info = model.get_bilstm_info()
        
        print("\n📋 BiLSTM Configuration:")
        for key, value in info.items():
            print(f"  - {key}: {value}")
        
        # Test forward pass
        test_input = torch.randn(2, 60, 7)
        mu, sigma = model(test_input)
        print(f"\n✅ Forward pass successful! Output shapes: μ={mu.shape}, σ={sigma.shape}")
        
        # Test MC dropout
        model.enable_mc_dropout()
        outputs = []
        for _ in range(5):
            mu, _ = model(test_input)
            outputs.append(mu)
        
        variance = torch.stack(outputs).var(dim=0).mean()
        print(f"✅ MC Dropout working! Average variance: {variance:.6f}")
        
    elif args.analyze_only:
        if not Path(args.old_checkpoint).exists():
            print(f"❌ Checkpoint not found: {args.old_checkpoint}")
            sys.exit(1)
        
        analyze_checkpoint(args.old_checkpoint)
        
    else:
        # Perform migration
        if not Path(args.old_checkpoint).exists():
            print(f"❌ Old checkpoint not found: {args.old_checkpoint}")
            sys.exit(1)
        
        if not args.new_checkpoint and not args.dry_run:
            # Auto-generate new checkpoint path
            old_path = Path(args.old_checkpoint)
            args.new_checkpoint = str(old_path.parent / f"{old_path.stem}_bilstm{old_path.suffix}")
            print(f"📝 Auto-generated new checkpoint path: {args.new_checkpoint}")
        
        success = migrate_checkpoint(
            args.old_checkpoint, 
            args.new_checkpoint,
            dry_run=args.dry_run
        )
        
        if not success:
            print("\n❌ Migration failed!")
            sys.exit(1)
        
        print("\n✅ Migration completed successfully!")


if __name__ == '__main__':
    main()