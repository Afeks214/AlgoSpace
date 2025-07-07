"""
Dataset preparation script for RDE Communication LSTM training.

This script processes historical RDE outputs to create training data.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import logging
import argparse
from typing import Dict, List, Tuple, Optional
import h5py
import json
from datetime import datetime, timedelta
import yaml

from src.agents.rde.engine import RDEComponent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RDEDatasetPreparator:
    """
    Prepares dataset from historical market data by running RDE.
    """
    
    def __init__(self, config: Dict[str, any]):
        """
        Initialize dataset preparator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.rde = self._initialize_rde()
        
    def _initialize_rde(self) -> RDEComponent:
        """Initialize RDE component."""
        rde_config = self.config.get('rde', {})
        rde = RDEComponent(rde_config)
        
        # Load pre-trained RDE model if available
        model_path = self.config.get('rde_model_path')
        if model_path and Path(model_path).exists():
            rde.load_model(model_path)
            logger.info(f"Loaded RDE model from {model_path}")
        else:
            logger.warning("No pre-trained RDE model found, using random initialization")
            
        return rde
    
    def process_historical_data(
        self,
        market_data_path: str,
        output_path: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> None:
        """
        Process historical market data through RDE.
        
        Args:
            market_data_path: Path to historical market data
            output_path: Path to save processed dataset
            start_date: Start date for processing (YYYY-MM-DD)
            end_date: End date for processing (YYYY-MM-DD)
        """
        logger.info(f"Processing historical data from {market_data_path}")
        
        # Load market data
        market_data = self._load_market_data(market_data_path, start_date, end_date)
        
        # Process through RDE in batches
        regime_vectors = []
        timestamps = []
        metadata = []
        
        batch_size = self.config.get('batch_size', 100)
        n_batches = len(market_data) // batch_size + 1
        
        logger.info(f"Processing {len(market_data)} samples in {n_batches} batches")
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(market_data))
            
            if start_idx >= len(market_data):
                break
                
            batch_data = market_data.iloc[start_idx:end_idx]
            
            # Extract MMD features for RDE
            mmd_features = self._extract_mmd_features(batch_data)
            
            # Process through RDE
            for j, features in enumerate(mmd_features):
                try:
                    regime_vector = self.rde.get_regime_vector(features)
                    regime_vectors.append(regime_vector)
                    timestamps.append(batch_data.iloc[j]['timestamp'])
                    
                    # Store metadata
                    meta = {
                        'volatility': batch_data.iloc[j].get('volatility', 0),
                        'volume': batch_data.iloc[j].get('volume', 0),
                        'price': batch_data.iloc[j].get('price', 0)
                    }
                    metadata.append(meta)
                    
                except Exception as e:
                    logger.warning(f"Failed to process sample {start_idx + j}: {e}")
                    
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {end_idx}/{len(market_data)} samples")
        
        # Convert to numpy arrays
        regime_vectors = np.array(regime_vectors, dtype=np.float32)
        timestamps = np.array(timestamps)
        
        # Add regime transition labels
        regime_transitions = self._identify_regime_transitions(regime_vectors)
        
        # Save dataset
        self._save_dataset(
            regime_vectors,
            timestamps,
            metadata,
            regime_transitions,
            output_path
        )
        
        logger.info(f"Dataset saved to {output_path}")
        logger.info(f"Total samples: {len(regime_vectors)}")
        logger.info(f"Regime transitions: {np.sum(regime_transitions)}")
    
    def _load_market_data(
        self,
        path: str,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """Load and filter market data."""
        # Support multiple file formats
        path = Path(path)
        
        if path.suffix == '.parquet':
            df = pd.read_parquet(path)
        elif path.suffix == '.csv':
            df = pd.read_csv(path)
        elif path.suffix == '.h5':
            df = pd.read_hdf(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
            
        # Convert timestamp column
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'datetime' in df.columns:
            df['timestamp'] = pd.to_datetime(df['datetime'])
        else:
            # Try to infer timestamp from index
            df['timestamp'] = pd.to_datetime(df.index)
            
        # Filter by date range
        if start_date:
            df = df[df['timestamp'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['timestamp'] <= pd.to_datetime(end_date)]
            
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        return df
    
    def _extract_mmd_features(self, data: pd.DataFrame) -> List[np.ndarray]:
        """
        Extract MMD features from market data.
        
        This should match the feature extraction used in production.
        """
        features_list = []
        
        for _, row in data.iterrows():
            # Create feature vector (155 dimensions as per RDE spec)
            features = []
            
            # Price-based features (example)
            features.extend([
                row.get('open', 0),
                row.get('high', 0),
                row.get('low', 0),
                row.get('close', 0),
                row.get('volume', 0),
                row.get('volatility', 0),
                row.get('rsi', 50),
                row.get('macd', 0),
                row.get('macd_signal', 0),
                row.get('bb_upper', 0),
                row.get('bb_lower', 0),
                row.get('atr', 0)
            ])
            
            # Pad to 155 dimensions with zeros (simplified)
            while len(features) < 155:
                features.append(0.0)
                
            features_array = np.array(features[:155], dtype=np.float32)
            
            # Reshape to expected format [sequence_length, n_features]
            # For single timestep, create sequence of length 24 (12 hours)
            sequence = np.tile(features_array, (24, 1))
            
            features_list.append(sequence)
            
        return features_list
    
    def _identify_regime_transitions(
        self,
        regime_vectors: np.ndarray,
        threshold: float = 0.3
    ) -> np.ndarray:
        """
        Identify regime transitions in the sequence.
        
        Args:
            regime_vectors: Array of regime vectors
            threshold: Distance threshold for transition
            
        Returns:
            Binary array indicating transitions
        """
        transitions = np.zeros(len(regime_vectors), dtype=bool)
        
        for i in range(1, len(regime_vectors)):
            # Calculate L2 distance between consecutive regimes
            distance = np.linalg.norm(regime_vectors[i] - regime_vectors[i-1])
            
            if distance > threshold:
                transitions[i] = True
                
        return transitions
    
    def _save_dataset(
        self,
        regime_vectors: np.ndarray,
        timestamps: np.ndarray,
        metadata: List[Dict],
        regime_transitions: np.ndarray,
        output_path: str
    ) -> None:
        """Save processed dataset."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as HDF5 for efficient storage
        with h5py.File(output_path, 'w') as f:
            # Main data
            f.create_dataset('regime_vectors', data=regime_vectors, compression='gzip')
            f.create_dataset('timestamps', data=timestamps.astype(str), compression='gzip')
            f.create_dataset('regime_transitions', data=regime_transitions, compression='gzip')
            
            # Metadata
            meta_group = f.create_group('metadata')
            for i, meta in enumerate(metadata):
                meta_group.create_dataset(f'sample_{i}', data=json.dumps(meta))
                
            # Dataset info
            f.attrs['n_samples'] = len(regime_vectors)
            f.attrs['regime_dim'] = regime_vectors.shape[1]
            f.attrs['creation_date'] = datetime.now().isoformat()
            f.attrs['config'] = json.dumps(self.config)
    
    def create_synthetic_dataset(
        self,
        n_samples: int = 100000,
        output_path: str = 'data/synthetic_regime_data.h5'
    ) -> None:
        """
        Create synthetic regime data for testing.
        
        Args:
            n_samples: Number of samples to generate
            output_path: Path to save dataset
        """
        logger.info(f"Creating synthetic dataset with {n_samples} samples")
        
        # Generate synthetic regime vectors
        regime_vectors = []
        timestamps = []
        
        # Define regime prototypes
        regime_prototypes = [
            np.array([1, 0, 0, 0, 0, 0, 0, 0]),  # Trending up
            np.array([0, 1, 0, 0, 0, 0, 0, 0]),  # Trending down
            np.array([0, 0, 1, 0, 0, 0, 0, 0]),  # Ranging
            np.array([0, 0, 0, 1, 0, 0, 0, 0]),  # Volatile
        ]
        
        current_regime = 0
        regime_duration = 0
        
        start_time = datetime(2020, 1, 1)
        
        for i in range(n_samples):
            # Decide if regime should change
            if regime_duration > np.random.poisson(100) or regime_duration == 0:
                # Change regime
                current_regime = np.random.randint(0, len(regime_prototypes))
                regime_duration = 0
                
            # Generate regime vector with noise
            base_regime = regime_prototypes[current_regime]
            noise = np.random.randn(8) * 0.1
            regime_vector = base_regime + noise
            regime_vector = regime_vector / (np.linalg.norm(regime_vector) + 1e-8)
            
            regime_vectors.append(regime_vector)
            timestamps.append(start_time + timedelta(minutes=30 * i))
            regime_duration += 1
            
        regime_vectors = np.array(regime_vectors, dtype=np.float32)
        timestamps = np.array(timestamps)
        
        # Create metadata
        metadata = [{'synthetic': True} for _ in range(n_samples)]
        
        # Identify transitions
        regime_transitions = self._identify_regime_transitions(regime_vectors)
        
        # Save dataset
        self._save_dataset(
            regime_vectors,
            timestamps,
            metadata,
            regime_transitions,
            output_path
        )
        
        logger.info(f"Synthetic dataset saved to {output_path}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Prepare RDE training dataset')
    parser.add_argument(
        '--config',
        type=str,
        default='config/settings.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Path to input market data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/rde_dataset.h5',
        help='Path to save processed dataset'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Create synthetic dataset instead'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=100000,
        help='Number of synthetic samples'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Create preparator
    preparator = RDEDatasetPreparator(config)
    
    if args.synthetic:
        # Create synthetic dataset
        preparator.create_synthetic_dataset(
            n_samples=args.n_samples,
            output_path=args.output
        )
    else:
        # Process real data
        if not args.input:
            raise ValueError("Input path required for real data processing")
            
        preparator.process_historical_data(
            market_data_path=args.input,
            output_path=args.output,
            start_date=args.start_date,
            end_date=args.end_date
        )


if __name__ == "__main__":
    main()