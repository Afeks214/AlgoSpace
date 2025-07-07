#!/usr/bin/env python3
"""
Migration script to convert CNN checkpoints to Transformer format.

This script migrates existing CNN-based StructureEmbedder weights to the new
Transformer architecture. Since architectures are completely different, we:
1. Keep the final projection layer if dimensions match
2. Initialize transformer layers with Xavier init
3. Log the migration process
"""

import torch
import logging
import argparse
import yaml
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.main_core.models import StructureEmbedder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def migrate_cnn_to_transformer(
    cnn_checkpoint_path: str,
    output_path: str,
    config: dict
) -> None:
    """
    Migrate CNN-based structure embedder weights to transformer format.
    
    Since architectures are completely different, we'll:
    1. Keep the final projection layer if dimensions match
    2. Initialize transformer layers with Xavier init
    3. Log the migration process
    
    Args:
        cnn_checkpoint_path: Path to existing CNN checkpoint
        output_path: Path for new transformer checkpoint
        config: Configuration dict for transformer architecture
    """
    logger.info(f"Migrating CNN checkpoint: {cnn_checkpoint_path}")
    
    # Load old checkpoint
    try:
        old_checkpoint = torch.load(cnn_checkpoint_path, map_location='cpu')
        logger.info("Successfully loaded CNN checkpoint")
    except Exception as e:
        logger.error(f"Failed to load CNN checkpoint: {e}")
        return
    
    # Initialize new transformer model
    try:
        new_model = StructureEmbedder(**config)
        logger.info("Successfully initialized new Transformer model")
        logger.info(f"Transformer architecture: d_model={config.get('d_model', 128)}, "
                   f"n_heads={config.get('n_heads', 4)}, n_layers={config.get('n_layers', 3)}")
    except Exception as e:
        logger.error(f"Failed to initialize Transformer model: {e}")
        return
    
    # Check if we can reuse any weights
    migrated_params = 0
    migration_details = []
    
    # Try to migrate projection layer if it exists
    old_state = old_checkpoint.get('model_state_dict', old_checkpoint)
    
    logger.info("Analyzing old checkpoint structure...")
    for old_name, old_param in old_state.items():
        logger.debug(f"Found parameter: {old_name}, shape: {old_param.shape}")
        
        # Try to find compatible layers
        if 'projection' in old_name and 'weight' in old_name:
            # Check if dimensions match with mu_head final layer
            mu_head_final = new_model.mu_head[-2]  # Second to last layer (before LayerNorm)
            if hasattr(mu_head_final, 'weight') and old_param.shape == mu_head_final.weight.shape:
                mu_head_final.weight.data = old_param.clone()
                migrated_params += 1
                migration_details.append(f"Migrated {old_name} to mu_head final layer")
                logger.info(f"Migrated projection weights: {old_name} -> mu_head")
                
        elif 'projection' in old_name and 'bias' in old_name:
            # Check if bias dimensions match
            mu_head_final = new_model.mu_head[-2]
            if hasattr(mu_head_final, 'bias') and mu_head_final.bias is not None:
                if old_param.shape == mu_head_final.bias.shape:
                    mu_head_final.bias.data = old_param.clone()
                    migrated_params += 1
                    migration_details.append(f"Migrated {old_name} to mu_head final bias")
                    logger.info(f"Migrated projection bias: {old_name} -> mu_head")
    
    # Log migration summary
    logger.info(f"Migration completed. Migrated {migrated_params} parameters.")
    if migration_details:
        logger.info("Migration details:")
        for detail in migration_details:
            logger.info(f"  - {detail}")
    else:
        logger.warning("No compatible parameters found for migration. "
                      "All transformer weights will be randomly initialized.")
    
    # Create new checkpoint with metadata
    new_checkpoint = {
        'model_state_dict': new_model.state_dict(),
        'config': config,
        'migration_metadata': {
            'migrated_from': cnn_checkpoint_path,
            'migrated_params': migrated_params,
            'migration_date': datetime.now().isoformat(),
            'migration_details': migration_details,
            'original_architecture': 'CNN',
            'new_architecture': 'Transformer'
        },
        'model_info': {
            'total_parameters': sum(p.numel() for p in new_model.parameters()),
            'trainable_parameters': sum(p.numel() for p in new_model.parameters() if p.requires_grad)
        }
    }
    
    # Save new checkpoint
    try:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(new_checkpoint, output_path)
        logger.info(f"New checkpoint saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save new checkpoint: {e}")
        return
    
    # Verify the new model loads correctly
    try:
        test_model = StructureEmbedder(**config)
        test_model.load_state_dict(new_checkpoint['model_state_dict'])
        logger.info("✅ Verification: New checkpoint loads successfully")
        
        # Test forward pass
        test_input = torch.randn(1, 48, 8)
        with torch.no_grad():
            mu, sigma = test_model(test_input)
            logger.info(f"✅ Verification: Forward pass successful, output shapes: mu={mu.shape}, sigma={sigma.shape}")
            
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        return
    
    logger.info("🎉 Migration completed successfully!")


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description="Migrate CNN to Transformer StructureEmbedder")
    parser.add_argument('--old-checkpoint', required=True, 
                       help='Path to CNN checkpoint')
    parser.add_argument('--output-path', required=True, 
                       help='Path for new checkpoint')
    parser.add_argument('--config-file', required=True, 
                       help='Path to configuration file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input files
    if not Path(args.old_checkpoint).exists():
        logger.error(f"CNN checkpoint not found: {args.old_checkpoint}")
        return 1
        
    if not Path(args.config_file).exists():
        logger.error(f"Config file not found: {args.config_file}")
        return 1
    
    # Load configuration
    try:
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract embedder config
        embedder_config = config.get('main_core', {}).get('embedders', {}).get('structure', {})
        
        # Set defaults if not specified
        transformer_config = {
            'input_channels': 8,
            'output_dim': embedder_config.get('output_dim', 64),
            'd_model': embedder_config.get('d_model', 128),
            'n_heads': embedder_config.get('n_heads', 4),
            'n_layers': embedder_config.get('n_layers', 3),
            'd_ff': embedder_config.get('d_ff', 512),
            'dropout_rate': embedder_config.get('dropout', 0.2),
            'max_seq_len': embedder_config.get('max_seq_len', 48)
        }
        
        logger.info(f"Loaded configuration: {transformer_config}")
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Perform migration
    migrate_cnn_to_transformer(
        args.old_checkpoint,
        args.output_path,
        transformer_config
    )
    
    return 0


if __name__ == "__main__":
    exit(main())