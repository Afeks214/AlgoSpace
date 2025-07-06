"""Data Pipeline for MARL Training.

This module handles the preparation of historical market data for training,
including matrix generation, data augmentation, and train/validation/test splits.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Generator
from pathlib import Path
import h5py
from datetime import datetime, timedelta
from collections import deque
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle

from src.generators.indicators import TechnicalIndicatorGenerator
from src.assemblers.matrix_assembler import MatrixAssembler5m, MatrixAssembler30m, RegimeMatrixAssembler


logger = logging.getLogger(__name__)


class MarketDataPipeline:
    """Pipeline for preparing market data for MARL training."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data pipeline.
        
        Args:
            config: Pipeline configuration including:
                - data_path: Path to raw market data
                - output_path: Path for processed data
                - symbols: List of trading symbols
                - start_date: Training data start date
                - end_date: Training data end date
                - val_split: Validation split ratio
                - test_split: Test split ratio
                - augmentation: Data augmentation settings
        """
        self.config = config
        self.data_path = Path(config['data_path'])
        self.output_path = Path(config['output_path'])
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Data parameters
        self.symbols = config.get('symbols', ['EUR_USD'])
        self.start_date = pd.to_datetime(config.get('start_date', '2020-01-01'))
        self.end_date = pd.to_datetime(config.get('end_date', '2023-12-31'))
        
        # Split ratios
        self.val_split = config.get('val_split', 0.15)
        self.test_split = config.get('test_split', 0.15)
        self.train_split = 1.0 - self.val_split - self.test_split
        
        # Initialize components
        self._initialize_components()
        
        logger.info(f"Initialized MarketDataPipeline for {len(self.symbols)} symbols")
    
    def _initialize_components(self):
        """Initialize indicator generators and matrix assemblers."""
        # Technical indicator generator
        indicator_config = {
            'ema_periods': [9, 21, 50, 200],
            'sma_periods': [20, 50],
            'rsi_period': 14,
            'macd_params': (12, 26, 9),
            'bb_params': (20, 2),
            'atr_period': 14,
            'volume_ema': 20
        }
        self.indicator_generator = TechnicalIndicatorGenerator(indicator_config)
        
        # Matrix assemblers
        assembler_config = {
            'normalize': True,
            'handle_missing': 'forward_fill',
            'feature_engineering': True
        }
        self.matrix_5m = MatrixAssembler5m(assembler_config)
        self.matrix_30m = MatrixAssembler30m(assembler_config)
        self.regime_matrix = RegimeMatrixAssembler(assembler_config)
        
        # Data buffers
        self.data_cache = {}
        self.matrix_cache = {}
    
    def prepare_training_data(self) -> Dict[str, Any]:
        """Prepare complete training dataset.
        
        Returns:
            Dictionary containing train/val/test splits and metadata
        """
        logger.info("Starting training data preparation...")
        
        # Load and process raw data
        raw_data = self._load_raw_data()
        
        # Generate features and matrices
        processed_data = self._process_data(raw_data)
        
        # Create train/val/test splits
        splits = self._create_splits(processed_data)
        
        # Apply data augmentation if enabled
        if self.config.get('augmentation', {}).get('enabled', False):
            splits['train'] = self._augment_data(splits['train'])
        
        # Save processed data
        self._save_processed_data(splits)
        
        # Generate metadata
        metadata = self._generate_metadata(splits)
        
        logger.info("Training data preparation completed")
        return {
            'splits': splits,
            'metadata': metadata,
            'config': self.config
        }
    
    def _load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Load raw market data from files.
        
        Returns:
            Dictionary of DataFrames by symbol
        """
        raw_data = {}
        
        for symbol in self.symbols:
            # Try multiple file formats
            file_paths = [
                self.data_path / f"{symbol}_5m.csv",
                self.data_path / f"{symbol}_5m.parquet",
                self.data_path / f"{symbol}.h5"
            ]
            
            for file_path in file_paths:
                if file_path.exists():
                    logger.info(f"Loading data for {symbol} from {file_path}")
                    
                    if file_path.suffix == '.csv':
                        df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
                    elif file_path.suffix == '.parquet':
                        df = pd.read_parquet(file_path)
                    elif file_path.suffix == '.h5':
                        df = pd.read_hdf(file_path, key='data')
                    
                    # Filter by date range
                    df = df.loc[self.start_date:self.end_date]
                    
                    # Ensure required columns
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    if all(col in df.columns for col in required_cols):
                        raw_data[symbol] = df
                        logger.info(f"Loaded {len(df)} bars for {symbol}")
                    else:
                        logger.warning(f"Missing required columns for {symbol}")
                    
                    break
            else:
                logger.warning(f"No data file found for {symbol}")
        
        return raw_data
    
    def _process_data(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, np.ndarray]]:
        """Process raw data into matrices.
        
        Args:
            raw_data: Raw market data by symbol
            
        Returns:
            Processed matrices by symbol and timeframe
        """
        processed_data = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            
            for symbol, df in raw_data.items():
                future = executor.submit(self._process_symbol_data, symbol, df)
                futures[future] = symbol
            
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    processed_data[symbol] = result
                    logger.info(f"Processed data for {symbol}")
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
        
        return processed_data
    
    def _process_symbol_data(self, symbol: str, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Process data for a single symbol.
        
        Args:
            symbol: Trading symbol
            df: Raw OHLCV data
            
        Returns:
            Dictionary of matrices by timeframe
        """
        # Generate technical indicators
        df_with_indicators = self.indicator_generator.generate_indicators(df)
        
        # Resample to 30m for higher timeframe
        df_30m = df_with_indicators.resample('30T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Generate 30m indicators
        df_30m_indicators = self.indicator_generator.generate_indicators(df_30m)
        
        # Generate matrices
        matrices = {}
        
        # 5-minute matrix (60×7)
        matrices['5m'] = self._generate_5m_matrices(df_with_indicators)
        
        # 30-minute matrix (48×8)
        matrices['30m'] = self._generate_30m_matrices(df_30m_indicators)
        
        # Regime matrix (96×N)
        matrices['regime'] = self._generate_regime_matrices(df_30m_indicators)
        
        # Store metadata
        matrices['timestamps'] = df_with_indicators.index.values
        matrices['prices'] = df_with_indicators[['open', 'high', 'low', 'close']].values
        
        return matrices
    
    def _generate_5m_matrices(self, df: pd.DataFrame) -> np.ndarray:
        """Generate 5-minute matrices for tactical agent.
        
        Args:
            df: DataFrame with 5-minute bars and indicators
            
        Returns:
            Array of shape (n_samples, 60, 7)
        """
        window_size = 60
        features = ['close', 'volume', 'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower']
        
        # Ensure all features exist
        for feature in features:
            if feature not in df.columns:
                if feature == 'macd':
                    df['macd'] = df.get('macd_line', 0)
                elif feature == 'macd_signal':
                    df['macd_signal'] = df.get('signal_line', 0)
                else:
                    df[feature] = 0
        
        # Create rolling windows
        matrices = []
        for i in range(window_size, len(df)):
            window = df.iloc[i-window_size:i][features].values
            matrices.append(window)
        
        return np.array(matrices, dtype=np.float32)
    
    def _generate_30m_matrices(self, df: pd.DataFrame) -> np.ndarray:
        """Generate 30-minute matrices for structure agent.
        
        Args:
            df: DataFrame with 30-minute bars and indicators
            
        Returns:
            Array of shape (n_samples, 48, 8)
        """
        window_size = 48
        features = ['open', 'high', 'low', 'close', 'volume', 'ema_21', 'ema_50', 'atr']
        
        # Ensure all features exist
        for feature in features:
            if feature not in df.columns:
                df[feature] = df.get('close', 0)
        
        # Create rolling windows
        matrices = []
        for i in range(window_size, len(df)):
            window = df.iloc[i-window_size:i][features].values
            matrices.append(window)
        
        return np.array(matrices, dtype=np.float32)
    
    def _generate_regime_matrices(self, df: pd.DataFrame) -> np.ndarray:
        """Generate regime matrices for regime agent.
        
        Args:
            df: DataFrame with 30-minute bars and indicators
            
        Returns:
            Array of shape (n_samples, 96, N) where N is number of features
        """
        window_size = 96
        
        # Extended feature set for regime detection
        features = [
            'close', 'volume', 'returns',
            'volatility', 'ema_9', 'ema_21', 'ema_50', 'ema_200',
            'rsi', 'macd', 'atr', 'volume_ratio'
        ]
        
        # Calculate additional features
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Ensure all features exist
        for feature in features:
            if feature not in df.columns:
                df[feature] = 0
        
        # Create rolling windows
        matrices = []
        for i in range(window_size, len(df)):
            window = df.iloc[i-window_size:i][features].values
            matrices.append(window)
        
        return np.array(matrices, dtype=np.float32)
    
    def _create_splits(self, processed_data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict]:
        """Create train/validation/test splits.
        
        Args:
            processed_data: Processed matrices by symbol
            
        Returns:
            Dictionary with train/val/test splits
        """
        splits = {'train': {}, 'val': {}, 'test': {}}
        
        for symbol, data in processed_data.items():
            # Get total number of samples (using 5m matrix as reference)
            n_samples = len(data['5m'])
            
            # Calculate split indices
            train_end = int(n_samples * self.train_split)
            val_end = train_end + int(n_samples * self.val_split)
            
            # Create splits for each matrix type
            for matrix_type in ['5m', '30m', 'regime']:
                if matrix_type in data:
                    splits['train'].setdefault(symbol, {)}[matrix_type] = data[matrix_type][:train_end]
                    splits['val'].setdefault(symbol, {})[matrix_type] = data[matrix_type][train_end:val_end]
                    splits['test'].setdefault(symbol, {})[matrix_type] = data[matrix_type][val_end:]
            
            # Also split metadata
            for meta_type in ['timestamps', 'prices']:
                if meta_type in data:
                    splits['train'].setdefault(symbol, {})[meta_type] = data[meta_type][:train_end]
                    splits['val'].setdefault(symbol, {})[meta_type] = data[meta_type][train_end:val_end]
                    splits['test'].setdefault(symbol, {})[meta_type] = data[meta_type][val_end:]
        
        # Log split sizes
        for split_name, split_data in splits.items():
            total_samples = sum(len(data.get('5m', [])) for data in split_data.values())
            logger.info(f"{split_name} split: {total_samples} samples")
        
        return splits
    
    def _augment_data(self, train_data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
        """Apply data augmentation techniques.
        
        Args:
            train_data: Training data to augment
            
        Returns:
            Augmented training data
        """
        augmented_data = {}
        aug_config = self.config.get('augmentation', {})
        
        for symbol, data in train_data.items():
            augmented_data[symbol] = data.copy()
            
            # Noise injection
            if aug_config.get('noise_injection', False):
                noise_level = aug_config.get('noise_level', 0.01)
                for matrix_type in ['5m', '30m', 'regime']:
                    if matrix_type in data:
                        noise = np.random.normal(0, noise_level, data[matrix_type].shape)
                        augmented_data[symbol][f"{matrix_type}_noisy"] = data[matrix_type] + noise
            
            # Time warping
            if aug_config.get('time_warping', False):
                warp_factor = aug_config.get('warp_factor', 0.1)
                for matrix_type in ['5m', '30m', 'regime']:
                    if matrix_type in data:
                        warped = self._apply_time_warping(data[matrix_type], warp_factor)
                        augmented_data[symbol][f"{matrix_type}_warped"] = warped
            
            # Synthetic minority oversampling for rare events
            if aug_config.get('smote', False):
                # This would implement SMOTE for rare market events
                pass
        
        return augmented_data
    
    def _apply_time_warping(self, data: np.ndarray, warp_factor: float) -> np.ndarray:
        """Apply time warping augmentation.
        
        Args:
            data: Input data array
            warp_factor: Warping intensity
            
        Returns:
            Time-warped data
        """
        # Simplified time warping - in practice would use DTW
        warped_data = np.zeros_like(data)
        for i in range(len(data)):
            # Random time shift within warp_factor
            shift = int(np.random.uniform(-warp_factor, warp_factor) * data.shape[1])
            if shift > 0:
                warped_data[i, shift:] = data[i, :-shift]
            elif shift < 0:
                warped_data[i, :shift] = data[i, -shift:]
            else:
                warped_data[i] = data[i]
        
        return warped_data
    
    def _save_processed_data(self, splits: Dict[str, Dict]):
        """Save processed data to disk.
        
        Args:
            splits: Train/val/test splits to save
        """
        for split_name, split_data in splits.items():
            split_path = self.output_path / split_name
            split_path.mkdir(exist_ok=True)
            
            # Save as HDF5 for efficient loading
            h5_path = split_path / f"{split_name}_data.h5"
            with h5py.File(h5_path, 'w') as f:
                for symbol, data in split_data.items():
                    symbol_group = f.create_group(symbol)
                    for matrix_type, matrix_data in data.items():
                        if isinstance(matrix_data, np.ndarray):
                            symbol_group.create_dataset(
                                matrix_type, 
                                data=matrix_data,
                                compression='gzip',
                                compression_opts=4
                            )
            
            logger.info(f"Saved {split_name} data to {h5_path}")
        
        # Save metadata
        metadata_path = self.output_path / 'metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(self._generate_metadata(splits), f)
    
    def _generate_metadata(self, splits: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate metadata for the processed dataset.
        
        Args:
            splits: Train/val/test splits
            
        Returns:
            Dataset metadata
        """
        metadata = {
            'creation_date': datetime.now().isoformat(),
            'config': self.config,
            'symbols': self.symbols,
            'date_range': {
                'start': self.start_date.isoformat(),
                'end': self.end_date.isoformat()
            },
            'split_sizes': {},
            'feature_info': {
                '5m': {
                    'shape': (60, 7),
                    'features': ['close', 'volume', 'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower']
                },
                '30m': {
                    'shape': (48, 8),
                    'features': ['open', 'high', 'low', 'close', 'volume', 'ema_21', 'ema_50', 'atr']
                },
                'regime': {
                    'shape': (96, 12),
                    'features': ['close', 'volume', 'returns', 'volatility', 'ema_9', 'ema_21', 
                               'ema_50', 'ema_200', 'rsi', 'macd', 'atr', 'volume_ratio']
                }
            }
        }
        
        # Calculate split sizes
        for split_name, split_data in splits.items():
            metadata['split_sizes'][split_name] = {}
            for symbol, data in split_data.items():
                if '5m' in data:
                    metadata['split_sizes'][split_name][symbol] = len(data['5m'])
        
        return metadata
    
    def create_data_loader(self, split: str, batch_size: int = 32, 
                          shuffle: bool = True) -> 'DataLoader':
        """Create a data loader for training.
        
        Args:
            split: Data split to load ('train', 'val', 'test')
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
            
        Returns:
            DataLoader instance
        """
        return DataLoader(
            data_path=self.output_path / split / f"{split}_data.h5",
            batch_size=batch_size,
            shuffle=shuffle,
            symbols=self.symbols
        )


class DataLoader:
    """Data loader for training MARL agents."""
    
    def __init__(self, data_path: Path, batch_size: int, 
                 shuffle: bool = True, symbols: List[str] = None):
        """Initialize the data loader.
        
        Args:
            data_path: Path to HDF5 data file
            batch_size: Batch size
            shuffle: Whether to shuffle data
            symbols: List of symbols to load
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.symbols = symbols
        
        # Load data into memory
        self._load_data()
        
        # Initialize indices
        self.n_samples = len(self.data['5m'])
        self.indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        self.current_idx = 0
    
    def _load_data(self):
        """Load data from HDF5 file."""
        self.data = {}
        
        with h5py.File(self.data_path, 'r') as f:
            # Load first symbol for now (extend for multi-symbol later)
            symbol = self.symbols[0] if self.symbols else list(f.keys())[0]
            
            if symbol in f:
                symbol_group = f[symbol]
                for key in symbol_group.keys():
                    self.data[key] = symbol_group[key][:]
                    
        logger.info(f"Loaded data with {len(self.data.get('5m', []))} samples")
    
    def __iter__(self):
        """Initialize iterator."""
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self
    
    def __next__(self) -> Dict[str, np.ndarray]:
        """Get next batch.
        
        Returns:
            Batch of data for all agents
        """
        if self.current_idx >= self.n_samples:
            raise StopIteration
        
        # Get batch indices
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        self.current_idx += self.batch_size
        
        # Create batch
        batch = {
            'regime': self.data['regime'][batch_indices],
            'structure': self.data['30m'][batch_indices],
            'tactical': self.data['5m'][batch_indices],
            'timestamps': self.data.get('timestamps', np.array([]))[batch_indices],
            'prices': self.data.get('prices', np.array([]))[batch_indices]
        }
        
        return batch
    
    def __len__(self) -> int:
        """Get number of batches."""
        return (self.n_samples + self.batch_size - 1) // self.batch_size