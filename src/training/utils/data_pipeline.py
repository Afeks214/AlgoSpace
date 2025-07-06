"""
Data pipeline for MARL training.

Handles data loading, preprocessing, and batch generation for
multi-agent reinforcement learning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Generator
from pathlib import Path
import h5py
from datetime import datetime, timedelta
import torch
from torch.utils.data import Dataset, DataLoader

import structlog

logger = structlog.get_logger()


class MarketDataLoader:
    """
    Loads and manages historical market data for training.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize market data loader.
        
        Args:
            config: Data configuration
        """
        self.config = config
        self.data_path = Path(config.get('data_path', 'data/market'))
        self.symbols = config.get('symbols', ['BTCUSDT'])
        self.start_date = pd.to_datetime(config.get('start_date', '2022-01-01'))
        self.end_date = pd.to_datetime(config.get('end_date', '2023-12-31'))
        
        # Cache for loaded data
        self.data_cache = {}
        
    def load_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Load market data for a symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe ('5m', '30m', etc.)
            
        Returns:
            Market data DataFrame
        """
        cache_key = f"{symbol}_{timeframe}"
        
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        # Load from file
        file_path = self.data_path / f"{symbol}_{timeframe}.parquet"
        
        if file_path.exists():
            df = pd.read_parquet(file_path)
            
            # Filter by date range
            df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
            
            # Ensure required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns in {file_path}")
            
            self.data_cache[cache_key] = df
            
            logger.info(f"Loaded market data symbol={symbol} timeframe={timeframe} rows={len(df}")
                start=df.index[0],
                end=df.index[-1]
            )
            
            return df
        else:
            # Generate synthetic data for testing
            logger.warning(f"Data file not found, generating synthetic data file_path={str(file_path}")
            )
            return self._generate_synthetic_data(symbol, timeframe)
    
    def _generate_synthetic_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Generate synthetic market data for testing."""
        # Time range
        if timeframe == '5m':
            freq = '5T'
            periods = 288 * 365  # 1 year of 5-minute bars
        elif timeframe == '30m':
            freq = '30T'
            periods = 48 * 365  # 1 year of 30-minute bars
        else:
            freq = '1H'
            periods = 24 * 365
        
        # Generate timestamps
        timestamps = pd.date_range(
            start=self.start_date,
            periods=periods,
            freq=freq
        )
        
        # Generate price data
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.01, periods)
        prices = 100 * np.exp(np.cumsum(returns))
        
        # OHLCV data
        data = {
            'open': prices * (1 + np.random.uniform(-0.001, 0.001, periods),
            'high': prices * (1 + np.random.uniform(0, 0.002, periods),
            'low': prices * (1 + np.random.uniform(-0.002, 0, periods),
            'close': prices,
            'volume': np.random.lognormal(10, 1, periods)
        }
        
        df = pd.DataFrame(data, index=timestamps)
        
        # Add technical indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['rsi'] = self._calculate_rsi(df['close'])
        
        return df.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_training_data(self) -> Dict[str, pd.DataFrame]:
        """
        Prepare all data needed for training.
        
        Returns:
            Dictionary of DataFrames by timeframe
        """
        training_data = {}
        
        for symbol in self.symbols:
            # Load different timeframes
            training_data[f"{symbol}_5m"] = self.load_data(symbol, '5m')
            training_data[f"{symbol}_30m"] = self.load_data(symbol, '30m')
        
        return training_data


class DataPipeline:
    """
    Complete data pipeline for MARL training.
    
    Handles data generation, augmentation, and batch creation for
    the multi-agent trading environment.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.data_loader = MarketDataLoader(config)
        
        # Pipeline parameters
        self.window_sizes = {
            'structure': 48,    # 48 * 30m = 24 hours
            'tactical': 60,     # 60 * 5m = 5 hours
            'arbitrageur': 100  # Combined view
        }
        
        self.feature_dims = {
            'structure': 8,
            'tactical': 7,
            'arbitrageur': 15
        }
        
        # Data augmentation
        self.augmentation_config = config.get('augmentation', {})
        self.noise_level = self.augmentation_config.get('noise_level', 0.001)
        self.time_shift_range = self.augmentation_config.get('time_shift_range', 5)
        
        # Prepare data
        self.market_data = self.data_loader.prepare_training_data()
        self._align_data()
        
    def _align_data(self):
        """Align multi-timeframe data."""
        # Ensure 5m and 30m data are properly aligned
        for symbol in self.data_loader.symbols:
            df_5m = self.market_data[f"{symbol}_5m"]
            df_30m = self.market_data[f"{symbol}_30m"]
            
            # Align 30m data to 5m timestamps
            # Each 30m bar corresponds to 6 5m bars
            aligned_timestamps = []
            for ts_30m in df_30m.index:
                # Find corresponding 5m timestamps
                mask = (df_5m.index >= ts_30m) & (df_5m.index < ts_30m + timedelta(minutes=30))
                aligned_timestamps.extend(df_5m.index[mask].tolist())
            
            # Store alignment mapping
            self.market_data[f"{symbol}_alignment"] = aligned_timestamps
    
    def create_matrix(self, agent_type: str, timestamp: pd.Timestamp, symbol: str) -> np.ndarray:
        """
        Create feature matrix for an agent at a specific timestamp.
        
        Args:
            agent_type: Type of agent ('structure', 'tactical', 'arbitrageur')
            timestamp: Current timestamp
            symbol: Trading symbol
            
        Returns:
            Feature matrix for the agent
        """
        window_size = self.window_sizes[agent_type]
        feature_dim = self.feature_dims[agent_type]
        
        if agent_type == 'structure':
            # 30m data for structure analyzer
            df = self.market_data[f"{symbol}_30m"]
            return self._create_structure_matrix(df, timestamp, window_size)
            
        elif agent_type == 'tactical':
            # 5m data for tactical agent
            df = self.market_data[f"{symbol}_5m"]
            return self._create_tactical_matrix(df, timestamp, window_size)
            
        elif agent_type == 'arbitrageur':
            # Combined data for arbitrageur
            df_5m = self.market_data[f"{symbol}_5m"]
            df_30m = self.market_data[f"{symbol}_30m"]
            return self._create_arbitrageur_matrix(df_5m, df_30m, timestamp, window_size)
        
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def _create_structure_matrix(self, df: pd.DataFrame, timestamp: pd.Timestamp, window: int) -> np.ndarray:
        """Create matrix for structure analyzer (30m data)."""
        # Get historical window
        end_idx = df.index.get_loc(timestamp, method='ffill')
        start_idx = max(0, end_idx - window + 1)
        
        window_data = df.iloc[start_idx:end_idx + 1]
        
        # Features: OHLC, volume, SMA20, SMA50, RSI
        features = []
        for _, row in window_data.iterrows():
            features.append([
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                row['volume'],
                row.get('sma_20', row['close']),
                row.get('sma_50', row['close']),
                row.get('rsi', 50.0)
            ])
        
        matrix = np.array(features)
        
        # Pad if necessary
        if len(matrix) < window:
            padding = np.repeat(matrix[0:1], window - len(matrix), axis=0)
            matrix = np.concatenate([padding, matrix], axis=0)
        
        return matrix.astype(np.float32)
    
    def _create_tactical_matrix(self, df: pd.DataFrame, timestamp: pd.Timestamp, window: int) -> np.ndarray:
        """Create matrix for tactical agent (5m data)."""
        # Get historical window
        end_idx = df.index.get_loc(timestamp, method='ffill')
        start_idx = max(0, end_idx - window + 1)
        
        window_data = df.iloc[start_idx:end_idx + 1]
        
        # Features: OHLC, volume, spread estimate, momentum
        features = []
        for i, (_, row) in enumerate(window_data.iterrows()):
            # Calculate spread estimate
            spread = (row['high'] - row['low']) / row['close']
            
            # Calculate momentum
            if i > 0:
                momentum = (row['close'] - window_data.iloc[i-1]['close']) / window_data.iloc[i-1]['close']
            else:
                momentum = 0.0
            
            features.append([
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                row['volume'],
                spread,
                momentum
            ])
        
        matrix = np.array(features)
        
        # Pad if necessary
        if len(matrix) < window:
            padding = np.repeat(matrix[0:1], window - len(matrix), axis=0)
            matrix = np.concatenate([padding, matrix], axis=0)
        
        return matrix.astype(np.float32)
    
    def _create_arbitrageur_matrix(self, df_5m: pd.DataFrame, df_30m: pd.DataFrame, 
                                   timestamp: pd.Timestamp, window: int) -> np.ndarray:
        """Create combined matrix for arbitrageur."""
        # Get windows for both timeframes
        # 30m window (first 40 rows)
        end_idx_30m = df_30m.index.get_loc(timestamp, method='ffill')
        start_idx_30m = max(0, end_idx_30m - 40 + 1)
        window_30m = df_30m.iloc[start_idx_30m:end_idx_30m + 1]
        
        # 5m window (last 60 rows)
        end_idx_5m = df_5m.index.get_loc(timestamp, method='ffill')
        start_idx_5m = max(0, end_idx_5m - 60 + 1)
        window_5m = df_5m.iloc[start_idx_5m:end_idx_5m + 1]
        
        # Create combined features
        features = []
        
        # Add 30m features (8 features)
        for _, row in window_30m.iterrows():
            features.append([
                row['open'], row['high'], row['low'], row['close'],
                row['volume'], row.get('sma_20', row['close']),
                row.get('sma_50', row['close']), row.get('rsi', 50.0)
            ])
        
        # Pad 30m data to 40 rows
        while len(features) < 40:
            features.insert(0, features[0] if features else [0] * 8)
        
        # Add 5m features (7 features)
        for i, (_, row) in enumerate(window_5m.iterrows()):
            spread = (row['high'] - row['low']) / row['close']
            momentum = 0.0
            if i > 0:
                momentum = (row['close'] - window_5m.iloc[i-1]['close']) / window_5m.iloc[i-1]['close']
            
            # Extend existing row or create new
            if len(features) < 40 + i + 1:
                features.append([0] * 8)  # Placeholder for 30m features
            
            features[40 + i].extend([
                row['open'], row['high'], row['low'], row['close'],
                row['volume'], spread, momentum
            ])
        
        # Ensure we have exactly 100 rows with 15 features
        matrix = []
        for i in range(100):
            if i < len(features) and len(features[i]) == 15:
                matrix.append(features[i])
            else:
                # Pad with zeros
                matrix.append([0] * 15)
        
        return np.array(matrix, dtype=np.float32)
    
    def augment_data(self, matrix: np.ndarray, augment_type: str = 'noise') -> np.ndarray:
        """
        Apply data augmentation to matrix.
        
        Args:
            matrix: Input feature matrix
            augment_type: Type of augmentation ('noise', 'shift', 'scale')
            
        Returns:
            Augmented matrix
        """
        if augment_type == 'noise':
            # Add Gaussian noise
            noise = np.random.normal(0, self.noise_level, matrix.shape)
            return matrix + noise * matrix  # Proportional noise
            
        elif augment_type == 'shift':
            # Time shift
            shift = np.random.randint(-self.time_shift_range, self.time_shift_range + 1)
            if shift > 0:
                # Shift forward (pad beginning)
                padding = np.repeat(matrix[0:1], shift, axis=0)
                return np.concatenate([padding, matrix[:-shift]], axis=0)
            elif shift < 0:
                # Shift backward (pad end)
                padding = np.repeat(matrix[-1:], -shift, axis=0)
                return np.concatenate([matrix[-shift:], padding], axis=0)
            else:
                return matrix
                
        elif augment_type == 'scale':
            # Random scaling
            scale = np.random.uniform(0.95, 1.05)
            return matrix * scale
            
        else:
            return matrix
    
    def create_episode_generator(
        self,
        episode_length: int = 1000,
        batch_size: int = 1,
        augment: bool = True
    ) -> Generator[Dict[str, Dict[str, Any]], None, None]:
        """
        Generate episodes for training.
        
        Args:
            episode_length: Length of each episode
            batch_size: Number of parallel episodes
            augment: Whether to apply data augmentation
            
        Yields:
            Batch of episode data
        """
        symbol = self.data_loader.symbols[0]  # Single symbol for now
        df_5m = self.market_data[f"{symbol}_5m"]
        
        # Valid starting points (ensure enough history)
        min_history = max(self.window_sizes.values())
        valid_starts = df_5m.index[min_history:-episode_length]
        
        while True:
            batch_data = []
            
            for _ in range(batch_size):
                # Random starting point
                start_idx = np.random.randint(0, len(valid_starts))
                start_time = valid_starts[start_idx]
                
                # Generate episode data
                episode_data = {
                    'structure': [],
                    'tactical': [],
                    'arbitrageur': [],
                    'timestamps': [],
                    'prices': [],
                    'regime_vectors': []
                }
                
                # Generate matrices for each timestep
                current_time = start_time
                for step in range(episode_length):
                    # Create matrices for each agent
                    structure_matrix = self.create_matrix('structure', current_time, symbol)
                    tactical_matrix = self.create_matrix('tactical', current_time, symbol)
                    arbitrageur_matrix = self.create_matrix('arbitrageur', current_time, symbol)
                    
                    # Apply augmentation if enabled
                    if augment:
                        aug_type = np.random.choice(['noise', 'shift', 'scale'])
                        structure_matrix = self.augment_data(structure_matrix, aug_type)
                        tactical_matrix = self.augment_data(tactical_matrix, aug_type)
                        arbitrageur_matrix = self.augment_data(arbitrageur_matrix, aug_type)
                    
                    # Generate regime vector (placeholder)
                    regime_vector = self._generate_regime_vector(current_time, symbol)
                    
                    # Store data
                    episode_data['structure'].append(structure_matrix)
                    episode_data['tactical'].append(tactical_matrix)
                    episode_data['arbitrageur'].append(arbitrageur_matrix)
                    episode_data['timestamps'].append(current_time)
                    episode_data['regime_vectors'].append(regime_vector)
                    
                    # Get current price
                    current_price = df_5m.loc[current_time, 'close']
                    episode_data['prices'].append(current_price)
                    
                    # Move to next timestamp
                    current_idx = df_5m.index.get_loc(current_time)
                    if current_idx + 1 < len(df_5m):
                        current_time = df_5m.index[current_idx + 1]
                    else:
                        break
                
                # Convert lists to arrays
                for key in ['structure', 'tactical', 'arbitrageur', 'regime_vectors', 'prices']:
                    episode_data[key] = np.array(episode_data[key])
                
                batch_data.append(episode_data)
            
            yield batch_data
    
    def _generate_regime_vector(self, timestamp: pd.Timestamp, symbol: str) -> np.ndarray:
        """
        Generate regime vector for a timestamp.
        
        Args:
            timestamp: Current timestamp
            symbol: Trading symbol
            
        Returns:
            8-dimensional regime vector
        """
        # Placeholder implementation
        # In production, this would use the actual regime classifier
        
        # Simple regime features based on market conditions
        df_30m = self.market_data[f"{symbol}_30m"]
        
        try:
            idx = df_30m.index.get_loc(timestamp, method='ffill')
            row = df_30m.iloc[idx]
            
            # Calculate simple regime features
            sma_trend = (row['close'] - row.get('sma_50', row['close'])) / row['close']
            volatility = (row['high'] - row['low']) / row['close']
            rsi_norm = (row.get('rsi', 50) - 50) / 50
            
            # Create regime vector
            regime = np.array([
                sma_trend,           # Trend strength
                volatility,          # Volatility
                rsi_norm,           # Momentum
                0.0,                # Volume profile (placeholder)
                0.0,                # Market structure (placeholder)
                0.0,                # Correlation (placeholder)
                0.0,                # Seasonality (placeholder)
                0.0                 # External factors (placeholder)
            ], dtype=np.float32)
            
            return np.clip(regime, -1, 1)
            
        except:
            return np.zeros(8, dtype=np.float32)
    
    def save_preprocessed_data(self, output_path: Path):
        """
        Save preprocessed data for faster loading.
        
        Args:
            output_path: Path to save preprocessed data
        """
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save market data
        for key, df in self.market_data.items():
            if isinstance(df, pd.DataFrame):
                df.to_parquet(output_path / f"{key}.parquet")
        
        logger.info(f"Saved preprocessed data path={str(output_path}"))