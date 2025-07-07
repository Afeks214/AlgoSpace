"""
Training utilities for RDE Communication LSTM.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.agents.rde.communication import RDECommunicationLSTM
from src.agents.rde.losses import RDECommunicationLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RDESequenceDataset(Dataset):
    """
    Dataset for RDE Communication LSTM training.
    Creates sequences of regime vectors for temporal learning.
    """
    
    def __init__(
        self,
        regime_data: np.ndarray,
        sequence_length: int = 32,
        stride: int = 1,
        add_temporal_gaps: bool = True
    ):
        """
        Args:
            regime_data: Array of regime vectors [n_samples, 8]
            sequence_length: Length of sequences to create
            stride: Stride between sequences
            add_temporal_gaps: Whether to add gaps for robustness
        """
        self.regime_data = regime_data
        self.sequence_length = sequence_length
        self.stride = stride
        self.add_temporal_gaps = add_temporal_gaps
        
        # Create sequence indices
        self.indices = self._create_sequence_indices()
        
        logger.info(f"Created RDE dataset with {len(self.indices)} sequences")
        
    def _create_sequence_indices(self) -> List[int]:
        """Create valid sequence start indices."""
        n_samples = len(self.regime_data)
        indices = []
        
        for i in range(0, n_samples - self.sequence_length - 1, self.stride):
            if i + self.sequence_length + 1 <= n_samples:
                indices.append(i)
                
        return indices
    
    def _add_temporal_gap(self, sequence: np.ndarray) -> np.ndarray:
        """Randomly mask parts of sequence to simulate gaps."""
        if np.random.random() < 0.2:  # 20% chance
            gap_start = np.random.randint(0, len(sequence) - 5)
            gap_length = np.random.randint(1, 5)
            sequence[gap_start:gap_start + gap_length] = 0
        return sequence
        
    def __len__(self) -> int:
        return len(self.indices)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sequence."""
        start_idx = self.indices[idx]
        end_idx = start_idx + self.sequence_length
        
        # Get sequence
        sequence = self.regime_data[start_idx:end_idx].copy()
        
        # Add temporal gaps if enabled
        if self.add_temporal_gaps:
            sequence = self._add_temporal_gap(sequence)
            
        # Target is next timestep
        target = self.regime_data[end_idx]
        
        return {
            'sequence': torch.FloatTensor(sequence),
            'target': torch.FloatTensor(target),
            'start_idx': start_idx
        }


class RDETrainer:
    """
    Trainer class for RDE Communication LSTM.
    """
    
    def __init__(
        self,
        model: RDECommunicationLSTM,
        config: Dict[str, Any],
        device: torch.device
    ):
        """
        Initialize trainer.
        
        Args:
            model: RDE Communication LSTM model
            config: Training configuration
            device: Torch device
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Loss function
        self.loss_fn = RDECommunicationLoss(config['loss_weights'])
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config.get('scheduler_patience', 10),
            min_lr=1e-6
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_temporal': [],
            'val_temporal': [],
            'train_uncertainty': [],
            'val_uncertainty': [],
            'train_prediction': [],
            'val_prediction': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of average losses
        """
        self.model.train()
        
        total_losses = {
            'total': 0.0,
            'temporal': 0.0,
            'uncertainty': 0.0,
            'prediction': 0.0,
            'contrastive': 0.0
        }
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch in pbar:
            sequences = batch['sequence'].to(self.device)
            targets = batch['target'].to(self.device)
            
            batch_size, seq_length, _ = sequences.shape
            
            # Reset hidden state for each batch
            self.model.reset_hidden_state(batch_size)
            
            # Process sequence
            mu_list = []
            sigma_list = []
            
            for t in range(seq_length):
                mu, sigma = self.model(sequences[:, t, :])
                mu_list.append(mu)
                sigma_list.append(sigma)
            
            # Stack outputs
            mu_stack = torch.stack(mu_list, dim=1)
            sigma_stack = torch.stack(sigma_list, dim=1)
            
            # Calculate losses
            losses = self._calculate_sequence_loss(
                mu_stack, sigma_stack, targets, sequences
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.get('max_grad_norm', 1.0)
            )
            
            self.optimizer.step()
            
            # Update totals
            for key, value in losses.items():
                total_losses[key] += value.item()
                
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'temporal': f"{losses['temporal'].item():.4f}"
            })
        
        # Average losses
        n_batches = len(train_loader)
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        
        return avg_losses
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary of average losses
        """
        self.model.eval()
        
        total_losses = {
            'total': 0.0,
            'temporal': 0.0,
            'uncertainty': 0.0,
            'prediction': 0.0,
            'contrastive': 0.0
        }
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                sequences = batch['sequence'].to(self.device)
                targets = batch['target'].to(self.device)
                
                batch_size, seq_length, _ = sequences.shape
                
                # Reset hidden state
                self.model.reset_hidden_state(batch_size)
                
                # Process sequence
                mu_list = []
                sigma_list = []
                
                for t in range(seq_length):
                    mu, sigma = self.model(sequences[:, t, :])
                    mu_list.append(mu)
                    sigma_list.append(sigma)
                
                mu_stack = torch.stack(mu_list, dim=1)
                sigma_stack = torch.stack(sigma_list, dim=1)
                
                # Calculate losses
                losses = self._calculate_sequence_loss(
                    mu_stack, sigma_stack, targets, sequences
                )
                
                # Update totals
                for key, value in losses.items():
                    total_losses[key] += value.item()
        
        # Average losses
        n_batches = len(val_loader)
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        
        return avg_losses
    
    def _calculate_sequence_loss(
        self,
        mu_stack: torch.Tensor,
        sigma_stack: torch.Tensor,
        targets: torch.Tensor,
        sequences: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate losses for a sequence.
        
        Args:
            mu_stack: Stacked mu outputs [batch, seq_len, 16]
            sigma_stack: Stacked sigma outputs [batch, seq_len, 16]
            targets: Target regime vectors [batch, 8]
            sequences: Input sequences [batch, seq_len, 8]
            
        Returns:
            Dictionary of losses
        """
        # Get last timestep outputs
        mu_current = mu_stack[:, -1, :]
        sigma_current = sigma_stack[:, -1, :]
        
        # Get previous timestep for temporal loss
        mu_previous = mu_stack[:, -2, :] if mu_stack.size(1) > 1 else None
        
        # Create negative samples for contrastive loss
        batch_size = targets.size(0)
        negative_indices = torch.randperm(batch_size)
        negative_samples = sequences[negative_indices, -1, :]
        
        # Calculate combined loss
        losses = self.loss_fn(
            mu_current=mu_current,
            sigma_current=sigma_current,
            mu_previous=mu_previous,
            mu_next=None,  # Not used in this setup
            regime_targets=targets,
            negative_samples=negative_samples
        )
        
        return losses
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int
    ) -> None:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Number of epochs to train
        """
        logger.info(f"Starting training for {n_epochs} epochs")
        
        for epoch in range(1, n_epochs + 1):
            # Train
            train_losses = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_losses = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_losses['total'])
            
            # Update history
            for key in ['total', 'temporal', 'uncertainty', 'prediction']:
                self.history[f'train_{key}'].append(train_losses.get(key, 0))
                self.history[f'val_{key}'].append(val_losses.get(key, 0))
            self.history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Log progress
            logger.info(
                f"Epoch {epoch}/{n_epochs} - "
                f"Train Loss: {train_losses['total']:.4f} - "
                f"Val Loss: {val_losses['total']:.4f} - "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )
            
            # Check for improvement
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                logger.info("✅ New best model saved")
            else:
                self.patience_counter += 1
                
            # Early stopping
            if self.patience_counter >= self.config['early_stopping']['patience']:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
                
            # Regular checkpoint
            if epoch % self.config.get('checkpoint_interval', 10) == 0:
                self.save_checkpoint(epoch, is_best=False)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config
        }
        
        # Save paths
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if is_best:
            path = checkpoint_dir / 'rde_communication_best.pth'
        else:
            path = checkpoint_dir / f'rde_communication_epoch_{epoch}.pth'
            
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Total loss
        axes[0, 0].plot(self.history['train_total'], label='Train')
        axes[0, 0].plot(self.history['val_total'], label='Val')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        
        # Temporal consistency loss
        axes[0, 1].plot(self.history['train_temporal'], label='Train')
        axes[0, 1].plot(self.history['val_temporal'], label='Val')
        axes[0, 1].set_title('Temporal Consistency Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        
        # Uncertainty calibration loss
        axes[1, 0].plot(self.history['train_uncertainty'], label='Train')
        axes[1, 0].plot(self.history['val_uncertainty'], label='Val')
        axes[1, 0].set_title('Uncertainty Calibration Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()
        
        # Learning rate
        axes[1, 1].plot(self.history['learning_rates'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Training history plot saved to {save_path}")
        else:
            plt.show()


def create_dataloaders(
    regime_data: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    config: Dict[str, Any] = {}
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        regime_data: Array of regime vectors
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    n_samples = len(regime_data)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    # Split data
    train_data = regime_data[:n_train]
    val_data = regime_data[n_train:n_train + n_val]
    test_data = regime_data[n_train + n_val:]
    
    # Create datasets
    train_dataset = RDESequenceDataset(
        train_data,
        sequence_length=config.get('sequence_length', 32),
        add_temporal_gaps=True
    )
    
    val_dataset = RDESequenceDataset(
        val_data,
        sequence_length=config.get('sequence_length', 32),
        add_temporal_gaps=False
    )
    
    test_dataset = RDESequenceDataset(
        test_data,
        sequence_length=config.get('sequence_length', 32),
        add_temporal_gaps=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 64),
        shuffle=True,
        num_workers=config.get('num_workers', 0)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 64),
        shuffle=False,
        num_workers=config.get('num_workers', 0)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 64),
        shuffle=False,
        num_workers=config.get('num_workers', 0)
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    import yaml
    
    # Load configuration
    with open('../../config/settings.yaml', 'r') as f:
        settings = yaml.safe_load(f)
    
    config = settings['rde_communication']
    
    # Create synthetic regime data for testing
    regime_data = np.random.randn(10000, 8).astype(np.float32)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        regime_data,
        config=config
    )
    
    # Initialize model
    model = RDECommunicationLSTM(config)
    
    # Initialize trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = RDETrainer(model, config, device)
    
    # Train
    trainer.train(train_loader, val_loader, n_epochs=config['n_epochs'])
    
    # Plot history
    trainer.plot_training_history('rde_training_history.png')