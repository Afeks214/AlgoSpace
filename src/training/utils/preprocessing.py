"""
Data preprocessing utilities for MARL training.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
import torch

import structlog

logger = structlog.get_logger()


class DataPreprocessor:
    """
    Preprocesses market data for neural network input.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.scaling_method = config.get('scaling_method', 'standard')
        
        # Initialize scalers for different feature types
        self.scalers = {
            'price': self._create_scaler(),
            'volume': self._create_scaler(),
            'technical': self._create_scaler()
        }
        
        # Normalization statistics
        self.stats = {}
        self.is_fitted = False
        
    def _create_scaler(self):
        """Create scaler based on configuration."""
        if self.scaling_method == 'standard':
            return StandardScaler()
        elif self.scaling_method == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
    
    def fit(self, data: Dict[str, pd.DataFrame]):
        """
        Fit preprocessing parameters on training data.
        
        Args:
            data: Dictionary of market data DataFrames
        """
        logger.info("Fitting preprocessor on training data")
        
        # Collect statistics for each feature type
        price_data = []
        volume_data = []
        technical_data = []
        
        for key, df in data.items():
            if isinstance(df, pd.DataFrame) and 'close' in df.columns:
                # Price features
                price_features = df[['open', 'high', 'low', 'close']].values
                price_data.append(price_features)
                
                # Volume features
                if 'volume' in df.columns:
                    volume_data.append(df[['volume']].values)
                
                # Technical indicators
                technical_cols = [col for col in df.columns 
                                if col.startswith(('sma', 'ema', 'rsi', 'macd'))]
                if technical_cols:
                    technical_data.append(df[technical_cols].values)
        
        # Fit scalers
        if price_data:
            price_array = np.concatenate(price_data, axis=0)
            self.scalers['price'].fit(price_array)
            self.stats['price_mean'] = np.mean(price_array, axis=0)
            self.stats['price_std'] = np.std(price_array, axis=0)
        
        if volume_data:
            volume_array = np.concatenate(volume_data, axis=0)
            self.scalers['volume'].fit(volume_array)
            self.stats['volume_mean'] = np.mean(volume_array)
            self.stats['volume_std'] = np.std(volume_array)
        
        if technical_data:
            technical_array = np.concatenate(technical_data, axis=0)
            self.scalers['technical'].fit(technical_array)
        
        self.is_fitted = True
        logger.info(f"Preprocessor fitting complete stats={self.stats}")
    
    def transform_matrix(self, matrix: np.ndarray, agent_type: str) -> np.ndarray:
        """
        Transform feature matrix for an agent.
        
        Args:
            matrix: Raw feature matrix
            agent_type: Type of agent
            
        Returns:
            Normalized matrix
        """
        if not self.is_fitted:
            logger.warning("Preprocessor not fitted, using default normalization")
            return self._default_normalize(matrix)
        
        transformed = matrix.copy()
        
        if agent_type == 'structure':
            # Structure analyzer matrix: OHLC, volume, SMA20, SMA50, RSI
            # Normalize price features (columns 0-3)
            if matrix.shape[1] >= 4:
                transformed[:, 0:4] = self.scalers['price'].transform(matrix[:, 0:4])
            
            # Normalize volume (column 4)
            if matrix.shape[1] >= 5:
                transformed[:, 4:5] = self.scalers['volume'].transform(matrix[:, 4:5])
            
            # Normalize technical indicators (columns 5-7)
            if matrix.shape[1] >= 8:
                # SMA features - normalize like prices
                transformed[:, 5:7] = self.scalers['price'].transform(matrix[:, 5:7])
                # RSI - already normalized to 0-100, just scale to 0-1
                transformed[:, 7] = matrix[:, 7] / 100.0
        
        elif agent_type == 'tactical':
            # Tactical matrix: OHLC, volume, spread, momentum
            # Normalize price features (columns 0-3)
            if matrix.shape[1] >= 4:
                transformed[:, 0:4] = self.scalers['price'].transform(matrix[:, 0:4])
            
            # Normalize volume (column 4)
            if matrix.shape[1] >= 5:
                transformed[:, 4:5] = self.scalers['volume'].transform(matrix[:, 4:5])
            
            # Spread and momentum are already relative values
            # Just clip to reasonable range
            if matrix.shape[1] >= 7:
                transformed[:, 5] = np.clip(matrix[:, 5], 0, 0.1)  # Spread
                transformed[:, 6] = np.clip(matrix[:, 6], -0.1, 0.1)  # Momentum
        
        elif agent_type == 'arbitrageur':
            # Combined matrix with both 30m and 5m features
            # First 8 features are 30m, last 7 are 5m
            if matrix.shape[1] >= 15:
                # 30m OHLC
                transformed[:, 0:4] = self.scalers['price'].transform(matrix[:, 0:4])
                # 30m volume
                transformed[:, 4:5] = self.scalers['volume'].transform(matrix[:, 4:5])
                # 30m technical indicators
                transformed[:, 5:7] = self.scalers['price'].transform(matrix[:, 5:7])
                transformed[:, 7] = matrix[:, 7] / 100.0  # RSI
                
                # 5m OHLC
                transformed[:, 8:12] = self.scalers['price'].transform(matrix[:, 8:12])
                # 5m volume
                transformed[:, 12:13] = self.scalers['volume'].transform(matrix[:, 12:13])
                # 5m spread and momentum
                transformed[:, 13] = np.clip(matrix[:, 13], 0, 0.1)
                transformed[:, 14] = np.clip(matrix[:, 14], -0.1, 0.1)
        
        # Final clipping to prevent extreme values
        transformed = np.clip(transformed, -10, 10)
        
        return transformed.astype(np.float32)
    
    def _default_normalize(self, matrix: np.ndarray) -> np.ndarray:
        """Default normalization when scaler not fitted."""
        # Simple z-score normalization
        mean = np.mean(matrix, axis=0, keepdims=True)
        std = np.std(matrix, axis=0, keepdims=True) + 1e-8
        normalized = (matrix - mean) / std
        return np.clip(normalized, -5, 5).astype(np.float32)
    
    def inverse_transform_action(self, action: np.ndarray, agent_type: str) -> np.ndarray:
        """
        Inverse transform action from normalized to original scale.
        
        Args:
            action: Normalized action
            agent_type: Type of agent
            
        Returns:
            Action in original scale
        """
        # Actions are already in the correct scale
        # [action_type: 0-2, size: 0-1, timing: 0-5]
        return action
    
    def save_stats(self, path: str):
        """Save preprocessing statistics."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'scalers': self.scalers,
                'stats': self.stats,
                'config': self.config
            }, f)
        logger.info(f"Saved preprocessing stats path={path}")
    
    def load_stats(self, path: str):
        """Load preprocessing statistics."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.scalers = data['scalers']
            self.stats = data['stats']
            self.is_fitted = True
        logger.info(f"Loaded preprocessing stats path={path}")


class FeatureEngineer:
    """
    Creates additional features for MARL agents.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature engineer.
        
        Args:
            config: Feature engineering configuration
        """
        self.config = config
        self.feature_list = config.get('features', [
            'returns', 'volatility', 'volume_profile',
            'price_position', 'trend_strength'
        ])
        
    def create_features(
        self,
        df: pd.DataFrame,
        lookback_periods: List[int] = [5, 10, 20, 50]
    ) -> pd.DataFrame:
        """
        Create engineered features from market data.
        
        Args:
            df: Market data DataFrame
            lookback_periods: Periods for rolling calculations
            
        Returns:
            DataFrame with additional features
        """
        features = df.copy()
        
        # Returns over different periods
        if 'returns' in self.feature_list:
            for period in lookback_periods:
                features[f'return_{period}'] = (
                    features['close'] / features['close'].shift(period) - 1
                )
        
        # Volatility measures
        if 'volatility' in self.feature_list:
            for period in lookback_periods:
                features[f'volatility_{period}'] = (
                    features['close'].pct_change().rolling(period).std()
                )
                
                # High-low volatility
                features[f'hl_volatility_{period}'] = (
                    (features['high'] - features['low']) / features['close']
                ).rolling(period).mean()
        
        # Volume profile
        if 'volume_profile' in self.feature_list:
            for period in lookback_periods:
                features[f'volume_ratio_{period}'] = (
                    features['volume'] / features['volume'].rolling(period).mean()
                )
                
                # Volume-weighted average price
                features[f'vwap_{period}'] = (
                    (features['close'] * features['volume']).rolling(period).sum() /
                    features['volume'].rolling(period).sum()
                )
        
        # Price position indicators
        if 'price_position' in self.feature_list:
            for period in lookback_periods:
                rolling_high = features['high'].rolling(period).max()
                rolling_low = features['low'].rolling(period).min()
                
                # Position in range
                features[f'price_position_{period}'] = (
                    (features['close'] - rolling_low) / 
                    (rolling_high - rolling_low + 1e-8)
                )
                
                # Distance from high/low
                features[f'dist_from_high_{period}'] = (
                    (rolling_high - features['close']) / features['close']
                )
                features[f'dist_from_low_{period}'] = (
                    (features['close'] - rolling_low) / features['close']
                )
        
        # Trend strength
        if 'trend_strength' in self.feature_list:
            for period in lookback_periods:
                # Linear regression slope
                features[f'trend_slope_{period}'] = self._calculate_trend_slope(
                    features['close'], period
                )
                
                # Moving average deviation
                sma = features['close'].rolling(period).mean()
                features[f'ma_deviation_{period}'] = (
                    (features['close'] - sma) / sma
                )
        
        # Microstructure features
        if 'microstructure' in self.feature_list:
            # Bid-ask spread proxy
            features['spread_proxy'] = (
                (features['high'] - features['low']) / features['close']
            )
            
            # Trade intensity
            features['trade_intensity'] = features['volume'] / features['volume'].rolling(20).mean()
            
            # Price efficiency
            features['price_efficiency'] = 1 - abs(
                features['close'] - (features['high'] + features['low']) / 2
            ) / ((features['high'] - features['low']) / 2 + 1e-8)
        
        return features
    
    def _calculate_trend_slope(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate linear regression slope over rolling window."""
        def slope(values):
            if len(values) < 2:
                return 0
            x = np.arange(len(values))
            try:
                coef = np.polyfit(x, values, 1)[0]
                return coef / (np.mean(values) + 1e-8)  # Normalize by mean
            except:
                return 0
        
        return series.rolling(period).apply(slope, raw=True)
    
    def create_synergy_features(
        self,
        synergy_context: Dict[str, Any],
        market_data: pd.DataFrame
    ) -> np.ndarray:
        """
        Create features from synergy detection context.
        
        Args:
            synergy_context: Synergy detection results
            market_data: Current market data
            
        Returns:
            Synergy feature vector
        """
        features = []
        
        # Synergy type encoding
        synergy_type = synergy_context.get('synergy_type', 'NONE')
        type_encoding = {
            'TYPE_1': [1, 0, 0, 0],
            'TYPE_2': [0, 1, 0, 0],
            'TYPE_3': [0, 0, 1, 0],
            'TYPE_4': [0, 0, 0, 1],
            'NONE': [0, 0, 0, 0]
        }
        features.extend(type_encoding.get(synergy_type, [0, 0, 0, 0]))
        
        # Signal strengths
        signal_strengths = synergy_context.get('signal_strengths', {})
        features.append(signal_strengths.get('mlmi', 0.0))
        features.append(signal_strengths.get('nwrqk', 0.0))
        features.append(signal_strengths.get('fvg', 0.0))
        
        # Completion metrics
        metadata = synergy_context.get('metadata', {})
        bars_to_complete = metadata.get('bars_to_complete', 10)
        features.append(1.0 / (1.0 + bars_to_complete))  # Speed score
        
        # Market context at synergy
        if 'timestamp' in synergy_context and not market_data.empty:
            try:
                idx = market_data.index.get_loc(synergy_context['timestamp'], method='nearest')
                row = market_data.iloc[idx]
                
                # Price momentum at synergy
                if idx >= 5:
                    momentum = (row['close'] - market_data.iloc[idx-5]['close']) / market_data.iloc[idx-5]['close']
                else:
                    momentum = 0.0
                
                # Volume spike
                if idx >= 20:
                    volume_ratio = row['volume'] / market_data['volume'].iloc[idx-20:idx].mean()
                else:
                    volume_ratio = 1.0
                
                features.extend([momentum, volume_ratio])
            except:
                features.extend([0.0, 1.0])
        else:
            features.extend([0.0, 1.0])
        
        return np.array(features, dtype=np.float32)