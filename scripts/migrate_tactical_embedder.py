#!/usr/bin/env python3
"""
Migration script to convert basic LSTM tactical embedder to advanced architecture.

This script helps migrate existing tactical embedder checkpoints from the basic
LSTM implementation to the new advanced BiLSTM architecture with attention and
uncertainty quantification.
"""

import torch
import logging
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.main_core.models import TacticalEmbedder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def migrate_lstm_to_advanced(
    old_checkpoint_path: str,
    output_path: str,
    config: dict
) -> None:
    """
    Migrate basic LSTM tactical embedder to advanced architecture.
    
    Maps compatible weights and initializes new components.
    
    Args:
        old_checkpoint_path: Path to the old LSTM checkpoint
        output_path: Path for the new advanced checkpoint
        config: Configuration for the new architecture
    """
    logger.info(f"Migrating LSTM checkpoint: {old_checkpoint_path}")
    
    # Load old checkpoint
    try:
        old_checkpoint = torch.load(old_checkpoint_path, map_location='cpu')
        logger.info("✅ Old checkpoint loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load old checkpoint: {e}")
        raise
    
    # Extract state dict
    if isinstance(old_checkpoint, dict):
        if 'model_state_dict' in old_checkpoint:
            old_state = old_checkpoint['model_state_dict']
        elif 'state_dict' in old_checkpoint:
            old_state = old_checkpoint['state_dict']
        else:
            old_state = old_checkpoint
    else:
        old_state = old_checkpoint
    
    logger.info(f"Found {len(old_state)} parameters in old checkpoint")
    
    # Initialize new model
    try:
        new_model = TacticalEmbedder(**config)
        new_state = new_model.state_dict()
        logger.info(f"✅ New model initialized with {len(new_state)} parameters")
    except Exception as e:
        logger.error(f"❌ Failed to initialize new model: {e}")
        raise
    
    # Track migration
    migrated_params = 0
    migration_map = {}
    initialization_log = []
    
    # Migration strategy
    logger.info("🔄 Starting parameter migration...")
    
    # 1. Try to map LSTM weights
    for old_name, old_param in old_state.items():
        if 'lstm' in old_name.lower():
            # Map to first LSTM layer if dimensions match
            if 'weight_ih_l0' in old_name:
                new_name = 'lstm_layers.0.weight_ih_l0'
            elif 'weight_hh_l0' in old_name:
                new_name = 'lstm_layers.0.weight_hh_l0'
            elif 'bias_ih_l0' in old_name:
                new_name = 'lstm_layers.0.bias_ih_l0'
            elif 'bias_hh_l0' in old_name:
                new_name = 'lstm_layers.0.bias_hh_l0'
            elif 'weight_ih_l1' in old_name:
                new_name = 'lstm_layers.1.weight_ih_l0'  # Map second layer to second BiLSTM
            elif 'weight_hh_l1' in old_name:
                new_name = 'lstm_layers.1.weight_hh_l0'
            elif 'bias_ih_l1' in old_name:
                new_name = 'lstm_layers.1.bias_ih_l0'
            elif 'bias_hh_l1' in old_name:
                new_name = 'lstm_layers.1.bias_hh_l0'
            else:
                continue
                
            if new_name in new_state:
                expected_shape = new_state[new_name].shape
                if old_param.shape == expected_shape:
                    new_state[new_name] = old_param.clone()
                    migration_map[old_name] = new_name
                    migrated_params += 1
                    logger.info(f"✅ Migrated LSTM weight: {old_name} → {new_name}")
                else:
                    logger.warning(f"⚠️  Shape mismatch for {old_name}: {old_param.shape} vs {expected_shape}")
    
    # 2. Try to map attention weights
    for old_name, old_param in old_state.items():
        if 'attention' in old_name.lower():
            # Map old attention to multi-scale attention (first scale)
            if 'in_proj_weight' in old_name:
                new_name = 'multi_scale_attention.scale_attentions.0.in_proj_weight'
            elif 'in_proj_bias' in old_name:
                new_name = 'multi_scale_attention.scale_attentions.0.in_proj_bias'
            elif 'out_proj.weight' in old_name:
                new_name = 'multi_scale_attention.scale_attentions.0.out_proj.weight'
            elif 'out_proj.bias' in old_name:
                new_name = 'multi_scale_attention.scale_attentions.0.out_proj.bias'
            else:
                continue
                
            if new_name in new_state and old_param.shape == new_state[new_name].shape:
                new_state[new_name] = old_param.clone()
                migration_map[old_name] = new_name
                migrated_params += 1
                logger.info(f"✅ Migrated attention weight: {old_name} → {new_name}")
    
    # 3. Try to map projection/output weights
    for old_name, old_param in old_state.items():
        if 'projection' in old_name.lower() or 'fc' in old_name.lower() or 'linear' in old_name.lower():
            # Map to mu_head (mean output)
            if 'weight' in old_name:
                new_name = 'mu_head.4.weight'  # Final linear layer in mu_head
            elif 'bias' in old_name:
                new_name = 'mu_head.4.bias'
            else:
                continue
                
            if new_name in new_state:
                expected_shape = new_state[new_name].shape
                if old_param.shape == expected_shape:
                    new_state[new_name] = old_param.clone()
                    migration_map[old_name] = new_name
                    migrated_params += 1
                    logger.info(f"✅ Migrated projection: {old_name} → {new_name}")
                else:
                    logger.warning(f"⚠️  Shape mismatch for projection {old_name}: {old_param.shape} vs {expected_shape}")
    
    # 4. Initialize new components with reasonable defaults
    logger.info("🔧 Initializing new components...")
    
    # Initialize sigma_head with small positive values
    for name, param in new_state.items():
        if 'sigma_head' in name and name not in migration_map.values():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param, gain=0.1)  # Small initialization
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0.1)  # Small positive bias
            initialization_log.append(f"Initialized {name} for uncertainty estimation")
    
    # Initialize temporal position encoding
    for name, param in new_state.items():
        if 'position_encoder' in name and name not in migration_map.values():
            if 'position_embeddings' in name:
                torch.nn.init.normal_(param, std=0.1)
            elif 'momentum_scale' in name:
                torch.nn.init.constant_(param, 1.0)
            initialization_log.append(f"Initialized {name} for position encoding")
    
    # Initialize momentum extractor
    for name, param in new_state.items():
        if 'momentum_extractor' in name and name not in migration_map.values():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            initialization_log.append(f"Initialized {name} for momentum extraction")
    
    # Initialize additional LSTM layers
    for name, param in new_state.items():
        if 'lstm_layers.2' in name and name not in migration_map.values():  # Third layer
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            initialization_log.append(f"Initialized {name} for third LSTM layer")
    
    # Initialize multi-scale attention components
    for name, param in new_state.items():
        if 'multi_scale_attention' in name and name not in migration_map.values():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            initialization_log.append(f"Initialized {name} for multi-scale attention")
    
    # Load migrated weights into new model
    try:
        new_model.load_state_dict(new_state)
        logger.info("✅ New model loaded with migrated weights")
    except Exception as e:
        logger.error(f"❌ Failed to load migrated weights: {e}")
        raise
    
    # Create comprehensive checkpoint
    new_checkpoint = {
        'model_state_dict': new_model.state_dict(),
        'config': config,
        'migration_info': {
            'migrated_from': old_checkpoint_path,
            'migrated_params': migrated_params,
            'total_old_params': len(old_state),
            'total_new_params': len(new_state),
            'migration_map': migration_map,
            'initialization_log': initialization_log,
            'migration_date': datetime.now().isoformat(),
            'migration_success': True
        }
    }
    
    # Add original checkpoint metadata if available
    if isinstance(old_checkpoint, dict):
        original_metadata = {k: v for k, v in old_checkpoint.items() 
                           if k not in ['model_state_dict', 'state_dict']}
        if original_metadata:
            new_checkpoint['original_metadata'] = original_metadata
    
    # Save new checkpoint
    try:
        torch.save(new_checkpoint, output_path)
        logger.info(f"✅ Migration complete. Saved new checkpoint to: {output_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save new checkpoint: {e}")
        raise
    
    # Migration summary
    migration_ratio = migrated_params / len(old_state) * 100 if old_state else 0
    
    logger.info("📊 Migration Summary:")
    logger.info(f"  • Migrated parameters: {migrated_params}/{len(old_state)} ({migration_ratio:.1f}%)")
    logger.info(f"  • New parameters initialized: {len(initialization_log)}")
    logger.info(f"  • Total new parameters: {len(new_state)}")
    
    # Verify the new model loads correctly
    try:
        verification_model = TacticalEmbedder(**config)
        verification_model.load_state_dict(new_checkpoint['model_state_dict'])
        logger.info("✅ Verification: New checkpoint loads successfully")
        
        # Test inference
        test_input = torch.randn(1, 60, 7)
        with torch.no_grad():
            mu, sigma = verification_model(test_input)
        logger.info(f"✅ Verification: Inference successful. Output shapes: μ={mu.shape}, σ={sigma.shape}")
        
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        raise


def load_config_from_file(config_file: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_file, 'r') as f:
            full_config = yaml.safe_load(f)
        
        # Extract tactical embedder config
        if 'main_marl_core' in full_config and 'embedders' in full_config['main_marl_core']:
            config = full_config['main_marl_core']['embedders']['tactical']
        elif 'main_core' in full_config and 'embedders' in full_config['main_core']:
            config = full_config['main_core']['embedders']['tactical']
        else:
            raise ValueError("Could not find tactical embedder config in file")
        
        logger.info(f"✅ Loaded config from {config_file}")
        return config
        
    except Exception as e:
        logger.error(f"❌ Failed to load config from {config_file}: {e}")
        raise


def create_default_config() -> dict:
    """Create default configuration for advanced tactical embedder."""
    return {
        'input_dim': 7,
        'hidden_dim': 128,
        'output_dim': 48,
        'n_layers': 3,
        'dropout_rate': 0.2,
        'attention_scales': [5, 15, 30]
    }


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(
        description="Migrate basic LSTM tactical embedder to advanced architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic migration with default config
  python migrate_tactical_embedder.py --old-checkpoint old_model.pth --output new_model.pth
  
  # Migration with custom config file
  python migrate_tactical_embedder.py --old-checkpoint old_model.pth --output new_model.pth --config config/settings.yaml
  
  # Migration with specific parameters
  python migrate_tactical_embedder.py --old-checkpoint old_model.pth --output new_model.pth --hidden-dim 256 --n-layers 4
        """
    )
    
    parser.add_argument('--old-checkpoint', required=True, help='Path to old LSTM checkpoint')
    parser.add_argument('--output', required=True, help='Path for new checkpoint')
    parser.add_argument('--config', help='Path to config file (YAML)')
    
    # Manual config options
    parser.add_argument('--input-dim', type=int, default=7, help='Input dimension')
    parser.add_argument('--hidden-dim', type=int, default=128, help='LSTM hidden dimension')
    parser.add_argument('--output-dim', type=int, default=48, help='Output dimension')
    parser.add_argument('--n-layers', type=int, default=3, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--attention-scales', nargs='+', type=int, default=[5, 15, 30], 
                       help='Attention scales')
    
    parser.add_argument('--verify', action='store_true', help='Run verification after migration')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if old checkpoint exists
    if not Path(args.old_checkpoint).exists():
        logger.error(f"❌ Old checkpoint file not found: {args.old_checkpoint}")
        return 1
    
    # Load configuration
    if args.config:
        try:
            config = load_config_from_file(args.config)
        except Exception as e:
            logger.error(f"❌ Failed to load config file: {e}")
            return 1
    else:
        # Use command line arguments or defaults
        config = {
            'input_dim': args.input_dim,
            'hidden_dim': args.hidden_dim,
            'output_dim': args.output_dim,
            'n_layers': args.n_layers,
            'dropout_rate': args.dropout,
            'attention_scales': args.attention_scales
        }
        logger.info("Using configuration from command line arguments")
    
    logger.info(f"Configuration: {config}")
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run migration
    try:
        migrate_lstm_to_advanced(
            args.old_checkpoint,
            str(output_path),
            config
        )
        
        if args.verify:
            logger.info("🔍 Running additional verification...")
            # Run basic validation
            verification_model = TacticalEmbedder(**config)
            checkpoint = torch.load(str(output_path), map_location='cpu')
            verification_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Test with various inputs
            test_cases = [
                torch.randn(1, 60, 7),
                torch.zeros(1, 60, 7),
                torch.ones(1, 60, 7)
            ]
            
            for i, test_input in enumerate(test_cases):
                with torch.no_grad():
                    mu, sigma = verification_model(test_input)
                    logger.info(f"✅ Test case {i+1}: μ_range=[{mu.min():.3f}, {mu.max():.3f}], σ_range=[{sigma.min():.3f}, {sigma.max():.3f}]")
        
        logger.info("🎉 Migration completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())