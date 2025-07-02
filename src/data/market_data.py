"""
Market Data Handler for AlgoSpace Training
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

class MarketDataHandler:
    """Handles market data operations for training"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def generate_synthetic_data(self, 
                              start_date: str = "2020-01-01",
                              end_date: str = "2023-12-31",
                              freq: str = "1min") -> pd.DataFrame:
        """Generate synthetic forex market data"""
        
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        n_points = len(date_range)
        
        # Generate realistic price movements
        base_price = 1.1000  # EUR/USD
        returns = np.random.normal(0, 0.0001, n_points)
        
        # Add trend and volatility regimes
        prices = [base_price]
        for i in range(1, n_points):
            price_change = returns[i] * prices[-1]
            new_price = prices[-1] + price_change
            prices.append(max(0.5, min(2.0, new_price)))  # Realistic bounds
        
        # Create OHLC data
        df = pd.DataFrame({
            'timestamp': date_range,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.0005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.0005))) for p in prices],
            'close': prices,
            'volume': np.random.randint(100, 1000, n_points)
        })
        
        return df
        
    def load_training_data(self) -> pd.DataFrame:
        """Load or generate training data"""
        # For now, generate synthetic data
        return self.generate_synthetic_data()