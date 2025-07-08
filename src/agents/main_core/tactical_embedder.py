"""
State-of-the-art Tactical Embedder with enhanced BiLSTM implementation.

This module provides an advanced tactical embedder that processes 5-minute
price action data using bidirectional LSTM with enhanced temporal processing
capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import logging

from .tactical_enhancements import (
    PriceActionAnalyzer,
    TacticalPatternBank,
    MicrostructureAnalyzer,
    VolatilityAwareProcessor,
    ExecutionQualityPredictor
)
from .tactical_bilstm_components import (
    BiLSTMGateController,
    TemporalPyramidPooling,
    BiLSTMPositionalEncoding,
    DirectionalFeatureFusion,
    BiLSTMTemporalMasking
)

logger = logging.getLogger(__name__)


class TacticalEmbedder(nn.Module):
    """
    State-of-the-art Tactical Embedder with PROPER BiLSTM implementation.
    
    Processes 60×7 matrices from MatrixAssembler5m for short-term tactical
    decisions with enhanced bidirectional temporal processing.
    
    Key Features:
    - Proper BiLSTM with correct dimension handling
    - Adaptive gating for directional features
    - Temporal pyramid pooling
    - Advanced pattern recognition
    - Microstructure analysis
    - Execution quality prediction
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Configuration
        self.input_dim = config.get('input_dim', 7)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.output_dim = config.get('output_dim', 48)
        self.dropout_rate = config.get('dropout_rate', 0.2)
        self.num_layers = config.get('num_layers', 2)
        
        # CRITICAL FIX: Proper BiLSTM implementation
        self.bilstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0.0,
            bidirectional=True  # BiLSTM enabled
        )
        
        # CRITICAL: BiLSTM outputs 2*hidden_dim features
        self.bilstm_output_dim = self.hidden_dim * 2
        
        # BiLSTM enhancement components
        self.gate_controller = BiLSTMGateController(self.hidden_dim)
        self.positional_encoding = BiLSTMPositionalEncoding(self.hidden_dim)
        self.pyramid_pooling = TemporalPyramidPooling(
            self.bilstm_output_dim,
            pyramid_levels=[1, 3, 6, 12]
        )
        self.directional_fusion = DirectionalFeatureFusion(self.hidden_dim)
        self.temporal_masking = BiLSTMTemporalMasking(self.hidden_dim)
        
        # Temporal attention over BiLSTM outputs
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.bilstm_output_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # BiLSTM feature processor - handles 2*hidden_dim input
        self.bilstm_processor = nn.Sequential(
            nn.Linear(self.bilstm_output_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        
        # Forward-backward feature fusion
        self.direction_fusion = nn.Sequential(
            nn.Linear(self.bilstm_output_dim * 2, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # Tactical analysis components
        self.price_action = PriceActionAnalyzer(self.input_dim, self.hidden_dim)
        self.pattern_bank = TacticalPatternBank(n_patterns=32, pattern_dim=self.hidden_dim)
        self.microstructure = MicrostructureAnalyzer(self.hidden_dim)
        self.volatility_processor = VolatilityAwareProcessor(self.hidden_dim)
        self.execution_predictor = ExecutionQualityPredictor(input_dim=self.hidden_dim * 2)
        
        # Update feature aggregator to handle BiLSTM features properly
        self.feature_aggregator = nn.Sequential(
            nn.Linear(
                self.hidden_dim * 3 +  # BiLSTM processed + forward + backward
                8 +                    # Momentum
                4 +                    # Volume-price
                14 +                   # Microstructure
                8,                     # Execution
                self.hidden_dim * 2
            ),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        
        # Output heads for uncertainty quantification
        self.mean_head = nn.Linear(self.hidden_dim, self.output_dim)
        self.std_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Softplus()
        )
        
        # Cache for analysis
        self.last_attention_weights = None
        self.detected_patterns = None
        
        logger.info(f"Initialized TacticalEmbedder with BiLSTM (hidden_dim={self.hidden_dim}, output_dim={self.output_dim})")
    
    def forward(self, x: torch.Tensor, return_patterns: bool = False) -> torch.Tensor:
        """
        Process tactical data through BiLSTM embedder.
        
        Args:
            x: Input matrix [batch_size, 60, 7] from MatrixAssembler5m
            return_patterns: Whether to return detected patterns
            
        Returns:
            Embedded features [batch_size, output_dim] with uncertainty
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Process through BiLSTM
        bilstm_out, (hn, cn) = self.bilstm(x)
        # bilstm_out shape: [batch_size, seq_len, hidden_dim * 2]
        
        # 2. Split BiLSTM output into forward and backward
        forward_features = bilstm_out[:, :, :self.hidden_dim]
        backward_features = bilstm_out[:, :, self.hidden_dim:]
        
        # 3. Apply positional encoding
        forward_encoded, backward_encoded = self.positional_encoding(
            forward_features, backward_features
        )
        
        # 4. Apply gating
        gated_features = self.gate_controller(forward_encoded, backward_encoded)
        
        # 5. Apply temporal masking
        masked_features = self.temporal_masking(gated_features)
        
        # 6. Apply temporal attention
        attended_bilstm, attention_weights = self.temporal_attention(
            masked_features, masked_features, masked_features
        )
        self.last_attention_weights = attention_weights.detach()
        
        # 7. Process BiLSTM features
        processed_bilstm = self.bilstm_processor(attended_bilstm)
        # Shape: [batch_size, seq_len, hidden_dim]
        
        # 8. Extract directional features with fusion
        fused_directional = self.directional_fusion(forward_features, backward_features)
        
        # 9. Pyramid pooling for multi-scale features
        pyramid_features = self.pyramid_pooling(bilstm_out)
        
        # 10. Price action analysis
        price_analysis = self.price_action(x)
        pattern_features = price_analysis['pattern_features']
        
        # 11. Pattern matching
        pattern_matches = self.pattern_bank(pattern_features)
        
        # 12. Pool processed BiLSTM features
        bilstm_pooled = torch.mean(processed_bilstm, dim=1)  # [batch_size, hidden_dim]
        bilstm_last = processed_bilstm[:, -1, :]  # [batch_size, hidden_dim]
        
        # 13. Volatility-aware processing
        vol_processed, vol_info = self.volatility_processor(fused_directional)
        
        # 14. Microstructure analysis
        micro_analysis = self.microstructure(vol_processed)
        
        # 15. Execution quality prediction
        exec_features = torch.cat([bilstm_pooled, bilstm_last], dim=-1)
        exec_predictions = self.execution_predictor(exec_features)
        
        # 16. Aggregate all features including enhanced BiLSTM
        all_features = torch.cat([
            bilstm_pooled,          # Processed BiLSTM features
            pyramid_features,       # Multi-scale features
            fused_directional.mean(dim=1) if fused_directional.dim() > 2 else fused_directional,  # Directional fusion
            price_analysis['momentum'],
            price_analysis['volume_price'],
            micro_analysis['spread'],
            micro_analysis['liquidity'],
            micro_analysis['order_flow'],
            exec_predictions['timing_scores'][:, 0:1],
            exec_predictions['slippage_mean'].unsqueeze(1)
        ], dim=-1)
        
        aggregated = self.feature_aggregator(all_features)
        
        # 17. Generate output with uncertainty
        tactical_mean = self.mean_head(aggregated)
        tactical_std = self.std_head(aggregated) + 1e-6
        
        # Sample during training
        if self.training:
            eps = torch.randn_like(tactical_mean)
            output = tactical_mean + eps * tactical_std
        else:
            output = tactical_mean
        
        # Cache patterns if not training
        if not self.training:
            self._cache_patterns(pattern_matches, price_analysis, exec_predictions)
        
        if return_patterns:
            return output, self.detected_patterns
        else:
            return output
    
    def get_bilstm_analysis(self) -> Dict[str, Any]:
        """Get BiLSTM-specific analysis for debugging and monitoring."""
        return {
            'attention_weights': self.last_attention_weights.cpu().numpy() if self.last_attention_weights is not None else None,
            'hidden_dim': self.hidden_dim,
            'bilstm_output_dim': self.bilstm_output_dim,
            'num_layers': self.num_layers,
            'bidirectional': True,
            'has_gate_controller': hasattr(self, 'gate_controller'),
            'has_pyramid_pooling': hasattr(self, 'pyramid_pooling'),
            'has_positional_encoding': hasattr(self, 'positional_encoding')
        }
    
    def _cache_patterns(self, pattern_matches: Dict[str, torch.Tensor],
                       price_analysis: Dict[str, torch.Tensor],
                       exec_predictions: Dict[str, torch.Tensor]):
        """Cache detected patterns for analysis."""
        self.detected_patterns = {
            'pattern_matches': pattern_matches,
            'momentum_patterns': price_analysis.get('momentum_patterns', None),
            'execution_quality': exec_predictions
        }