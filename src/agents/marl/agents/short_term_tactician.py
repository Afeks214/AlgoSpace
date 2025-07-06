"""
Short-term Tactician Agent - Immediate execution specialist.

This agent focuses on precise entry/exit timing, immediate price action,
and execution quality. It provides the tactical view in the MARL
decision-making process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
import numpy as np

import structlog

from ..base import BaseTradeAgent, PolicyHead, SynergyEncoder, PositionalEncoding
from ..base.heads import TimingHead

logger = structlog.get_logger()


class ShortTermTactician(BaseTradeAgent):
    """
    Short-term Tactician Agent.
    
    Specializes in:
    - Immediate price action analysis
    - Entry/exit timing optimization
    - Execution quality assessment
    - Microstructure evaluation
    
    Input: 60×7 matrix from MatrixAssembler5m (5 hours of 5-min bars)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Short-term Tactician.
        
        Args:
            config: Agent configuration
        """
        # Set input features for 5m matrix
        config['input_features'] = 7  # From MatrixAssembler5m
        
        super().__init__(config)
        
        # Tactical-specific components
        self.synergy_encoder = TacticalSynergyEncoder()
        self.positional_encoding = PositionalEncoding(d_model=256)
        
        # Additional timing head for execution timing
        self.timing_head = TimingHead(
            input_dim=256 + 8 + 24,  # After concatenation
            max_delay=5  # Can delay up to 5 bars
        )
        
        # Agent weighting in ensemble
        self.agent_weight = 0.3  # 30% weight in consensus
        
        logger.info(f"Initialized Short-term Tactician window={config['window']} hidden_dim={config['hidden_dim']}")
    
    def _build_policy_head(self) -> nn.Module:
        """
        Build Short-term Tactician's policy head.
        
        Returns:
            Policy head with tactical-specific outputs
        """
        # Input: 256 (embedded) + 8 (regime) + 24 (synergy)
        input_dim = 256 + 8 + 24
        
        # Hidden layers for tactical analysis
        hidden_dims = [384, 192, 96]
        
        # Output heads
        output_heads = {
            'action': {'dim': 3, 'activation': None},      # [pass, long, short]
            'confidence': {'dim': 1, 'activation': None},  # Raw confidence
            'timing': {'dim': 5, 'activation': None},      # Timing delay (0-4 bars)
            'reasoning': {'dim': 48, 'activation': None}   # Interpretable features
        }
        
        return PolicyHead(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_heads=output_heads,
            dropout=self.dropout_rate
        )
    
    def forward(
        self,
        market_matrix: torch.Tensor,
        regime_vector: torch.Tensor,
        synergy_context: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass with positional encoding.
        
        Args:
            market_matrix: Market data matrix [batch, time, features]
            regime_vector: Regime context vector [batch, 8]
            synergy_context: Synergy detection context
            
        Returns:
            Dictionary with tactical decisions
        """
        # Get base forward pass
        decision = super().forward(market_matrix, regime_vector, synergy_context)
        
        # Add timing analysis
        # Use the concatenated context from parent forward
        context = self._get_last_context()
        if context is not None:
            timing_logits = self.timing_head(context)
            decision['timing'] = timing_logits
            
            # Convert to timing recommendation
            timing_probs = F.softmax(timing_logits, dim=-1)
            decision['timing_recommendation'] = torch.argmax(timing_probs, dim=-1)
        
        return decision
    
    def _get_last_context(self) -> Optional[torch.Tensor]:
        """Get the last context vector from forward pass."""
        # This would be stored during the parent's forward pass
        # For now, return None (to be implemented with state management)
        return None
    
    def _encode_synergy(self, synergy_context: Dict[str, Any]) -> torch.Tensor:
        """
        Extract tactical-relevant features from synergy context.
        
        Args:
            synergy_context: Synergy detection context
            
        Returns:
            Encoded features tensor [24]
        """
        return self.synergy_encoder.encode(synergy_context)
    
    def analyze_execution_quality(
        self,
        market_matrix: torch.Tensor,
        current_price: float
    ) -> Dict[str, float]:
        """
        Analyze execution quality factors.
        
        Args:
            market_matrix: Recent price data
            current_price: Current market price
            
        Returns:
            Execution quality metrics
        """
        quality_metrics = {}
        
        # Extract recent bars (last 10)
        recent_bars = market_matrix[:, -10:, :]
        
        # Price momentum (5-bar)
        if recent_bars.size(1) >= 5:
            close_prices = recent_bars[:, :, 3]  # Close prices
            momentum_5 = (close_prices[:, -1] - close_prices[:, -5]) / close_prices[:, -5]
            quality_metrics['momentum_5bar'] = momentum_5.item()
        
        # Volume analysis
        volumes = recent_bars[:, :, 4]  # Volume column
        avg_volume = volumes.mean().item()
        recent_volume = volumes[:, -1].item()
        quality_metrics['volume_ratio'] = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Volatility (high-low range)
        highs = recent_bars[:, :, 1]
        lows = recent_bars[:, :, 2]
        ranges = highs - lows
        avg_range = ranges.mean().item()
        quality_metrics['avg_range'] = avg_range
        quality_metrics['volatility_score'] = avg_range / current_price if current_price > 0 else 0.0
        
        # Microstructure score
        spread_estimate = ranges[:, -1].item()  # Last bar range as spread proxy
        quality_metrics['spread_estimate'] = spread_estimate
        quality_metrics['execution_cost_estimate'] = spread_estimate / current_price if current_price > 0 else 0.0
        
        return quality_metrics
    
    def get_timing_features(self, synergy_context: Dict[str, Any]) -> List[float]:
        """
        Extract timing-specific features.
        
        Args:
            synergy_context: Synergy context
            
        Returns:
            List of timing features
        """
        features = []
        
        # FVG timing features
        fvg_signal = next(
            (s for s in synergy_context['signal_sequence'] if s['type'] == 'fvg'),
            None
        )
        
        if fvg_signal:
            # Age of FVG (how many bars since creation)
            fvg_age = fvg_signal.get('metadata', {}).get('age', 0)
            features.append(min(fvg_age / 10, 1.0))  # Normalize to 10 bars
            
            # Size of gap
            gap_size = fvg_signal.get('metadata', {}).get('gap_size', 0)
            features.append(min(gap_size / 20, 1.0))  # Normalize to 20 points
        else:
            features.extend([0.0, 0.0])
        
        # Synergy completion speed
        bars_to_complete = synergy_context['metadata'].get('bars_to_complete', 10)
        features.append(1.0 / (1.0 + bars_to_complete))  # Faster = higher score
        
        # Current market momentum
        momentum = synergy_context['market_context'].get('price_momentum_5', 0)
        features.append(np.tanh(momentum * 100))  # Bounded momentum
        
        return features


class TacticalSynergyEncoder(SynergyEncoder):
    """
    Specialized synergy encoder for tactical analysis.
    
    Extracts features relevant to immediate execution decisions.
    """
    
    def __init__(self):
        """Initialize tactical synergy encoder."""
        super().__init__(output_dim=24)
    
    def encode(self, synergy_context: Dict[str, Any]) -> torch.Tensor:
        """
        Encode synergy context for tactical analysis.
        
        Args:
            synergy_context: Synergy detection context
            
        Returns:
            Encoded features tensor [24]
        """
        features = []
        
        # Synergy type encoding (4 features)
        synergy_type = synergy_context.get('synergy_type', 'TYPE_1')
        features.extend(self.encode_synergy_type(synergy_type).tolist())
        
        # Direction encoding (2 features)
        direction = synergy_context.get('direction', 1)
        features.extend(self.encode_direction(direction).tolist())
        
        # FVG characteristics (6 features)
        signal_sequence = synergy_context.get('signal_sequence', [])
        fvg_signal = next((s for s in signal_sequence if s['type'] == 'fvg'), None)
        
        if fvg_signal:
            # Extract FVG metadata
            metadata = fvg_signal.get('metadata', {})
            gap_type = metadata.get('gap_type', 'bullish')
            gap_size = metadata.get('gap_size', 0)
            gap_size_pct = metadata.get('gap_size_pct', 0)
            
            features.extend([
                1.0 if gap_type == 'bullish' else 0.0,
                1.0 if gap_type == 'bearish' else 0.0,
                min(gap_size / 20, 1.0),        # Normalized gap size
                min(gap_size_pct / 0.01, 1.0),  # Normalized percentage
                fvg_signal.get('strength', 0.5),
                1.0  # FVG present indicator
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Momentum quality (4 features)
        market_context = synergy_context.get('market_context', {})
        price_momentum = market_context.get('price_momentum_5', 0)
        volume_ratio = market_context.get('volume_profile', {}).get('volume_ratio', 1.0)
        
        features.extend([
            np.tanh(price_momentum * 10),      # Bounded momentum
            abs(price_momentum),                # Momentum magnitude
            min(volume_ratio / 2, 1.0),         # Volume surge indicator
            1.0 if volume_ratio > 1.2 else 0.0  # Volume breakout
        ])
        
        # Microstructure features (4 features)
        spread = market_context.get('spread', 0)
        current_price = market_context.get('current_price', 1)
        volatility = market_context.get('volatility', 10)
        
        spread_bps = (spread / current_price) * 10000 if current_price > 0 else 0
        features.extend([
            min(spread_bps / 10, 1.0),          # Normalized spread in bps
            min(volatility / 30, 1.0),          # Normalized volatility
            1.0 if spread_bps < 5 else 0.0,     # Tight spread indicator
            0.5  # Placeholder for future use
        ])
        
        # Timing features (4 features)
        bars_to_complete = synergy_context['metadata'].get('bars_to_complete', 10)
        signal_strengths = synergy_context.get('signal_strengths', {})
        
        features.extend([
            1.0 / (1.0 + bars_to_complete),     # Speed score
            min(bars_to_complete / 10, 1.0),    # Normalized completion time
            np.mean(list(signal_strengths.values())),  # Average signal strength
            np.std(list(signal_strengths.values()))    # Signal coherence
        ])
        
        # Ensure we have exactly 24 features
        assert len(features) == 24, f"Expected 24 features, got {len(features)}"
        
        return torch.tensor(features, dtype=torch.float32)