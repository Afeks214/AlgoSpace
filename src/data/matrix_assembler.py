"""
Matrix Assembler for AlgoSpace Training
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

class MatrixAssembler:
    """Assembles training matrices from market data"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def get_30min_matrix(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> np.ndarray:
        """Generate 48x8 structure matrix for 30-minute timeframe"""
        # Simulate matrix assembly
        return np.random.randn(48, 8).astype(np.float32)
        
    def get_5min_matrix(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> np.ndarray:
        """Generate 60x7 tactical matrix for 5-minute timeframe"""
        # Simulate matrix assembly
        return np.random.randn(60, 7).astype(np.float32)
        
    def get_mmd_sequence(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> np.ndarray:
        """Generate MMD sequence for RDE training"""
        # Simulate multi-modal data sequence
        return np.random.randn(100, 155).astype(np.float32)
        
    def get_portfolio_state(self, portfolio: Dict) -> np.ndarray:
        """Get current portfolio state vector"""
        return np.array([
            portfolio.get('cash', 100000) / 100000,
            portfolio.get('position', 0),
            portfolio.get('unrealized_pnl', 0) / 1000,
            portfolio.get('realized_pnl', 0) / 1000,
            portfolio.get('total_trades', 0) / 100
        ], dtype=np.float32)