"""
Dataset for training MRMS Communication LSTM.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MRMSCommunicationDataset(Dataset):
    """
    Dataset for training MRMS Communication LSTM.
    
    Creates sequences of risk decisions and outcomes for
    temporal learning.
    """
    
    def __init__(
        self,
        trade_history: pd.DataFrame,
        sequence_length: int = 10,
        stride: int = 1
    ):
        """
        Args:
            trade_history: DataFrame with columns:
                - position_size, sl_distance, tp_distance, confidence
                - hit_stop, hit_target, pnl
                - timestamp
            sequence_length: Length of sequences to create
            stride: Stride between sequences
        """
        self.sequence_length = sequence_length
        self.stride = stride
        
        # Normalize and prepare data
        self.risk_data, self.outcome_data = self._prepare_data(trade_history)
        
        # Create sequence indices
        self.indices = self._create_sequence_indices()
        
        logger.info(
            f"Created MRMS dataset with {len(self.indices)} sequences "
            f"from {len(trade_history)} trades"
        )
        
    def _prepare_data(
        self, 
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and normalize data."""
        # Risk parameters
        risk_data = np.stack([
            df['position_size'].values / 5.0,  # Normalize by max
            df['sl_distance'].values / 50.0,   # Normalize by typical range
            df['tp_distance'].values / 100.0,  # Normalize by typical range
            df['confidence'].values
        ], axis=1)
        
        # Outcomes
        outcome_data = np.stack([
            df['hit_stop'].values.astype(float),
            df['hit_target'].values.astype(float),
            np.clip(df['pnl'].values / 100.0, -1, 1)  # Normalized PnL
        ], axis=1)
        
        return risk_data, outcome_data
        
    def _create_sequence_indices(self) -> List[int]:
        """Create valid sequence start indices."""
        n_samples = len(self.risk_data)
        indices = []
        
        for i in range(0, n_samples - self.sequence_length - 1, self.stride):
            # Ensure we have target for last timestep
            if i + self.sequence_length + 1 <= n_samples:
                indices.append(i)
                
        return indices
        
    def __len__(self) -> int:
        return len(self.indices)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sequence."""
        start_idx = self.indices[idx]
        end_idx = start_idx + self.sequence_length
        
        # Get sequences
        risk_seq = self.risk_data[start_idx:end_idx]
        outcome_seq = self.outcome_data[start_idx:end_idx]
        
        # Target is next timestep
        target_risk = self.risk_data[end_idx]
        target_outcome = self.outcome_data[end_idx]
        
        return {
            'risk_sequence': torch.FloatTensor(risk_seq),
            'outcome_sequence': torch.FloatTensor(outcome_seq),
            'target_risk': torch.FloatTensor(target_risk),
            'target_outcome': torch.FloatTensor(target_outcome)
        }


def create_synthetic_trade_history(
    n_trades: int = 1000,
    win_rate: float = 0.45,
    avg_rr: float = 2.0,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Create synthetic trade history for testing/development.
    
    Args:
        n_trades: Number of trades to generate
        win_rate: Win rate of trades
        avg_rr: Average risk-reward ratio
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with trade history
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate trade parameters
    position_sizes = np.random.randint(1, 6, n_trades)
    sl_distances = np.random.uniform(5, 25, n_trades)
    tp_distances = sl_distances * np.random.normal(avg_rr, 0.5, n_trades)
    confidences = np.random.beta(5, 2, n_trades)  # Skewed towards higher confidence
    
    # Generate outcomes
    outcomes = np.random.random(n_trades) < win_rate
    hit_stops = ~outcomes
    hit_targets = outcomes
    
    # Calculate PnL
    pnls = np.where(
        outcomes,
        position_sizes * tp_distances * 5,  # Win: position * distance * point_value
        -position_sizes * sl_distances * 5  # Loss: -position * distance * point_value
    )
    
    # Add some streaks
    for i in range(10):
        start = np.random.randint(0, n_trades - 10)
        if np.random.random() < 0.5:
            # Winning streak
            hit_stops[start:start+5] = False
            hit_targets[start:start+5] = True
            pnls[start:start+5] = position_sizes[start:start+5] * tp_distances[start:start+5] * 5
        else:
            # Losing streak
            hit_stops[start:start+5] = True
            hit_targets[start:start+5] = False
            pnls[start:start+5] = -position_sizes[start:start+5] * sl_distances[start:start+5] * 5
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_trades, freq='H'),
        'position_size': position_sizes,
        'sl_distance': sl_distances,
        'tp_distance': tp_distances,
        'confidence': confidences,
        'hit_stop': hit_stops,
        'hit_target': hit_targets,
        'pnl': pnls
    })
    
    return df


def collate_sequences(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for DataLoader.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched tensors
    """
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        tensors = [sample[key] for sample in batch]
        collated[key] = torch.stack(tensors, dim=0)
    
    return collated


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 32,
    sequence_length: int = 10,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        batch_size: Batch size for training
        sequence_length: Sequence length for LSTM
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = MRMSCommunicationDataset(train_df, sequence_length)
    val_dataset = MRMSCommunicationDataset(val_df, sequence_length)
    test_dataset = MRMSCommunicationDataset(test_df, sequence_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sequences,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset creation
    df = create_synthetic_trade_history(n_trades=100, seed=42)
    dataset = MRMSCommunicationDataset(df, sequence_length=10)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample keys: {dataset[0].keys()}")
    
    # Test dataloader
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_sequences)
    batch = next(iter(loader))
    
    print("\nBatch shapes:")
    for key, tensor in batch.items():
        print(f"  {key}: {tensor.shape}")