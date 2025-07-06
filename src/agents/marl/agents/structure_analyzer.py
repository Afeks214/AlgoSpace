"""
Structure Analyzer Agent - Long-term market structure specialist.

This agent focuses on identifying and analyzing major market trends,
support/resistance levels, and overall market structure. It provides
the strategic view in the MARL decision-making process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import numpy as np

import structlog

from ..base import BaseTradeAgent, PolicyHead, SynergyEncoder

logger = structlog.get_logger()


class StructureAnalyzer(BaseTradeAgent):
    """
    Long-term Structure Analyzer Agent.
    
    Specializes in:
    - Market structure analysis
    - Major trend identification
    - Support/resistance evaluation
    - Strategic positioning
    
    Input: 48×8 matrix from MatrixAssembler30m (24 hours of 30-min bars)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Structure Analyzer.
        
        Args:
            config: Agent configuration
        """
        # Set input features for 30m matrix
        config['input_features'] = 8  # From MatrixAssembler30m
        
        super().__init__(config)
        
        # Structure-specific components
        self.synergy_encoder = StructureSynergyEncoder()
        
        # Agent weighting in ensemble
        self.agent_weight = 0.4  # 40% weight in consensus
        
        logger.info(f"Initialized Structure Analyzer window={config['window']} hidden_dim={config['hidden_dim']}")
    
    def _build_policy_head(self) -> nn.Module:
        """
        Build Structure Analyzer's policy head.
        
        Returns:
            Policy head with structure-specific outputs
        """
        # Input: 256 (embedded) + 8 (regime) + 32 (synergy)
        input_dim = 256 + 8 + 32
        
        # Hidden layers for structure analysis
        hidden_dims = [512, 256, 128]
        
        # Output heads
        output_heads = {
            'action': {'dim': 3, 'activation': None},      # [pass, long, short]
            'confidence': {'dim': 1, 'activation': None},  # Raw confidence
            'reasoning': {'dim': 64, 'activation': None}   # Interpretable features
        }
        
        return PolicyHead(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_heads=output_heads,
            dropout=self.dropout_rate
        )
    
    def _encode_synergy(self, synergy_context: Dict[str, Any]) -> torch.Tensor:
        """
        Extract structure-relevant features from synergy context.
        
        Args:
            synergy_context: Synergy detection context
            
        Returns:
            Encoded features tensor [32]
        """
        return self.synergy_encoder.encode(synergy_context)
    
    def _calculate_structure_score(self, synergy_context: Dict[str, Any]) -> float:
        """
        Calculate market structure quality score.
        
        Args:
            synergy_context: Synergy detection context
            
        Returns:
            Structure score (0-1)
        """
        score = 0.0
        
        # MLMI alignment (trend strength)
        mlmi_strength = synergy_context['signal_strengths'].get('mlmi', 0)
        score += mlmi_strength * 0.3
        
        # NW-RQK slope consistency
        nwrqk_data = next(
            (s for s in synergy_context['signal_sequence'] if s['type'] == 'nwrqk'),
            None
        )
        if nwrqk_data:
            # Normalize slope to 0-1 range
            slope_strength = min(abs(nwrqk_data.get('value', 0)) / 2.0, 1.0)
            score += slope_strength * 0.3
        
        # LVN positioning quality
        lvn_context = synergy_context['market_context'].get('nearest_lvn', {})
        lvn_distance = lvn_context.get('distance', 100)
        lvn_strength = lvn_context.get('strength', 0)
        
        # Closer to strong LVN = better structure
        distance_score = max(0, 1 - (lvn_distance / 50))  # Within 50 points
        strength_score = lvn_strength / 100
        score += (distance_score * strength_score) * 0.2
        
        # Synergy pattern quality
        pattern_scores = {
            'TYPE_1': 0.9,  # Classic momentum-structure pattern
            'TYPE_2': 0.8,  # Early gap fill pattern
            'TYPE_3': 0.85, # Structure-first pattern
            'TYPE_4': 0.75  # Less common pattern
        }
        pattern_score = pattern_scores.get(synergy_context['synergy_type'], 0.5)
        score += pattern_score * 0.2
        
        return min(score, 1.0)
    
    def analyze_market_regime(self, regime_vector: torch.Tensor) -> Dict[str, float]:
        """
        Analyze market regime for structure context.
        
        Args:
            regime_vector: 8-dimensional regime vector
            
        Returns:
            Regime analysis dictionary
        """
        # Extract regime components (assumed structure)
        trend_strength = regime_vector[0].item()
        volatility = regime_vector[1].item()
        momentum = regime_vector[2].item()
        volume_profile = regime_vector[3].item()
        
        return {
            'trend_strength': trend_strength,
            'volatility': volatility,
            'momentum': momentum,
            'volume_profile': volume_profile,
            'regime_quality': (trend_strength + momentum) / 2
        }
    
    def get_structure_features(self, market_matrix: torch.Tensor) -> Dict[str, float]:
        """
        Extract interpretable structure features.
        
        Args:
            market_matrix: Input market data
            
        Returns:
            Dictionary of structure features
        """
        # Process through embedder
        x = market_matrix.transpose(1, 2)
        embedded = self.embedder(x)
        
        # Global statistics
        features = {}
        
        # Trend metrics
        close_prices = market_matrix[:, :, 3]  # Assuming OHLC format
        if close_prices.size(1) > 1:
            price_changes = close_prices[:, 1:] - close_prices[:, :-1]
            features['trend_consistency'] = (
                (price_changes > 0).float().mean().item()
            )
            features['price_momentum'] = price_changes.mean().item()
        
        # Volatility metrics
        if hasattr(self, 'attention_weights') and self.attention_weights is not None:
            # Use attention to identify important periods
            attention_mean = self.attention_weights.mean(dim=1)  # Average over heads
            features['attention_concentration'] = (
                attention_mean.max(dim=-1)[0].mean().item()
            )
        
        return features


class StructureSynergyEncoder(SynergyEncoder):
    """
    Specialized synergy encoder for structure analysis.
    
    Extracts features relevant to market structure evaluation.
    """
    
    def __init__(self):
        """Initialize structure synergy encoder."""
        super().__init__(output_dim=32)
    
    def encode(self, synergy_context: Dict[str, Any]) -> torch.Tensor:
        """
        Encode synergy context for structure analysis.
        
        Args:
            synergy_context: Synergy detection context
            
        Returns:
            Encoded features tensor [32]
        """
        features = []
        
        # Synergy type encoding (4 features)
        synergy_type = synergy_context.get('synergy_type', 'TYPE_1')
        features.extend(self.encode_synergy_type(synergy_type).tolist())
        
        # Direction encoding (2 features)
        direction = synergy_context.get('direction', 1)
        features.extend(self.encode_direction(direction).tolist())
        
        # Signal strengths (3 features)
        signal_strengths = synergy_context.get('signal_strengths', {})
        mlmi_strength = signal_strengths.get('mlmi', 0.5)
        nwrqk_strength = signal_strengths.get('nwrqk', 0.5)
        fvg_strength = signal_strengths.get('fvg', 0.5)
        features.extend([mlmi_strength, nwrqk_strength, fvg_strength])
        
        # Trend alignment features
        signal_sequence = synergy_context.get('signal_sequence', [])
        
        # MLMI features (4 features)
        mlmi_signal = next((s for s in signal_sequence if s['type'] == 'mlmi'), None)
        if mlmi_signal:
            mlmi_value = mlmi_signal.get('value', 50)
            mlmi_deviation = abs(mlmi_value - 50) / 50
            features.extend([
                mlmi_deviation,
                1.0 if mlmi_value > 50 else 0.0,  # Bullish bias
                mlmi_signal.get('signal', 0),     # Raw signal
                mlmi_strength                      # Strength again for emphasis
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # NW-RQK features (4 features)
        nwrqk_signal = next((s for s in signal_sequence if s['type'] == 'nwrqk'), None)
        if nwrqk_signal:
            slope = nwrqk_signal.get('value', 0)
            normalized_slope = np.tanh(slope)  # Bound to [-1, 1]
            features.extend([
                normalized_slope,
                abs(normalized_slope),          # Slope magnitude
                1.0 if slope > 0 else 0.0,      # Uptrend indicator
                nwrqk_strength
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # LVN features (5 features)
        lvn_context = synergy_context['market_context'].get('nearest_lvn', {})
        lvn_price = lvn_context.get('price', 0)
        lvn_strength = lvn_context.get('strength', 0) / 100  # Normalize
        lvn_distance = lvn_context.get('distance', 0)
        current_price = synergy_context['market_context'].get('current_price', 0)
        
        if current_price > 0 and lvn_price > 0:
            relative_position = (current_price - lvn_price) / current_price
            distance_normalized = min(abs(lvn_distance) / 50, 1.0)  # Within 50 points
        else:
            relative_position = 0.0
            distance_normalized = 1.0
        
        features.extend([
            lvn_strength,
            distance_normalized,
            relative_position,
            1.0 if current_price > lvn_price else 0.0,  # Above LVN
            lvn_strength * (1 - distance_normalized)     # Combined score
        ])
        
        # Market structure features (6 features)
        market_context = synergy_context.get('market_context', {})
        volatility = market_context.get('volatility', 10) / 50  # Normalize
        volume_ratio = market_context.get('volume_profile', {}).get('volume_ratio', 1.0)
        
        features.extend([
            volatility,
            min(volume_ratio / 2, 1.0),  # Normalize volume ratio
            synergy_context['metadata'].get('bars_to_complete', 10) / 10,  # Normalized
            1.0,  # Placeholder for future use
            0.5,  # Placeholder for future use
            0.5   # Placeholder for future use
        ])
        
        # Ensure we have exactly 32 features
        assert len(features) == 32, f"Expected 32 features, got {len(features)}"
        
        return torch.tensor(features, dtype=torch.float32)