"""
Mid-frequency Arbitrageur Agent - Cross-timeframe inefficiency specialist.

This agent bridges structure and tactics, identifying market inefficiencies
across multiple timeframes. It provides the arbitrage view in the MARL
decision-making process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import numpy as np

import structlog

from ..base import BaseTradeAgent, PolicyHead, SynergyEncoder

logger = structlog.get_logger()


class MidFrequencyArbitrageur(BaseTradeAgent):
    """
    Mid-frequency Arbitrageur Agent.
    
    Specializes in:
    - Cross-timeframe alignment analysis
    - Market inefficiency identification
    - Arbitrage opportunity detection
    - Risk-reward optimization
    
    Input: Combined view of both 30m and 5m data (custom 100-bar window)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Mid-frequency Arbitrageur.
        
        Args:
            config: Agent configuration
        """
        # Set input features for combined view
        # 8 features from 30m + 7 features from 5m = 15 features
        config['input_features'] = 15
        
        super().__init__(config)
        
        # Arbitrage-specific components
        self.synergy_encoder = ArbitrageSynergyEncoder()
        
        # Multi-scale feature extractors
        self.macro_extractor = nn.Conv1d(8, 64, kernel_size=5, padding=2)
        self.micro_extractor = nn.Conv1d(7, 64, kernel_size=3, padding=1)
        
        # Cross-timeframe attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            dropout=self.dropout_rate,
            batch_first=True
        )
        
        # Agent weighting in ensemble
        self.agent_weight = 0.3  # 30% weight in consensus
        
        logger.info(f"Initialized Mid-frequency Arbitrageur window={config['window']} hidden_dim={config['hidden_dim']}")
    
    def _build_policy_head(self) -> nn.Module:
        """
        Build Mid-frequency Arbitrageur's policy head.
        
        Returns:
            Policy head with arbitrage-specific outputs
        """
        # Input: 256 (embedded) + 8 (regime) + 28 (synergy)
        input_dim = 256 + 8 + 28
        
        # Hidden layers for arbitrage analysis
        hidden_dims = [448, 224, 112]
        
        # Output heads
        output_heads = {
            'action': {'dim': 3, 'activation': None},           # [pass, long, short]
            'confidence': {'dim': 1, 'activation': None},       # Raw confidence
            'inefficiency_score': {'dim': 1, 'activation': None},  # Opportunity quality
            'reasoning': {'dim': 56, 'activation': None}        # Interpretable features
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
        Enhanced forward pass with multi-scale processing.
        
        Args:
            market_matrix: Combined market data [batch, time, 15]
            regime_vector: Regime context vector [batch, 8]
            synergy_context: Synergy detection context
            
        Returns:
            Dictionary with arbitrage decisions
        """
        # Split combined matrix into macro (30m) and micro (5m) components
        macro_data = market_matrix[:, :, :8]   # First 8 features
        micro_data = market_matrix[:, :, 8:]   # Last 7 features
        
        # Process through separate extractors
        macro_features = self._process_macro_data(macro_data)
        micro_features = self._process_micro_data(micro_data)
        
        # Cross-timeframe attention
        combined_features = self._cross_timeframe_attention(
            macro_features, micro_features
        )
        
        # Continue with standard processing
        # Use combined features instead of standard embedder output
        attended, attention_weights = self.temporal_attention(combined_features)
        pooled = torch.mean(attended, dim=1)
        
        # Encode synergy
        synergy_features = self._encode_synergy(synergy_context)
        if synergy_features.dim() == 1:
            synergy_features = synergy_features.unsqueeze(0)
        
        # Ensure batch dimensions match
        if pooled.size(0) != synergy_features.size(0):
            synergy_features = synergy_features.expand(pooled.size(0), -1)
        if pooled.size(0) != regime_vector.size(0):
            regime_vector = regime_vector.expand(pooled.size(0), -1)
        
        # Concatenate context
        context = torch.cat([pooled, regime_vector, synergy_features], dim=-1)
        
        # Generate decision
        decision = self.policy_head(context)
        decision['attention_weights'] = attention_weights
        
        # Calculate inefficiency score
        if 'inefficiency_score' in decision:
            decision['inefficiency_normalized'] = torch.sigmoid(
                decision['inefficiency_score']
            )
        
        return decision
    
    def _process_macro_data(self, macro_data: torch.Tensor) -> torch.Tensor:
        """
        Process macro (30m) timeframe data.
        
        Args:
            macro_data: 30m data [batch, time, 8]
            
        Returns:
            Processed features [batch, time, 64]
        """
        # Transpose for Conv1D
        x = macro_data.transpose(1, 2)  # [batch, 8, time]
        features = self.macro_extractor(x)
        features = F.relu(features)
        
        # Transpose back
        return features.transpose(1, 2)  # [batch, time, 64]
    
    def _process_micro_data(self, micro_data: torch.Tensor) -> torch.Tensor:
        """
        Process micro (5m) timeframe data.
        
        Args:
            micro_data: 5m data [batch, time, 7]
            
        Returns:
            Processed features [batch, time, 64]
        """
        # Transpose for Conv1D
        x = micro_data.transpose(1, 2)  # [batch, 7, time]
        features = self.micro_extractor(x)
        features = F.relu(features)
        
        # Transpose back
        return features.transpose(1, 2)  # [batch, time, 64]
    
    def _cross_timeframe_attention(
        self,
        macro_features: torch.Tensor,
        micro_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply cross-timeframe attention mechanism.
        
        Args:
            macro_features: Macro timeframe features [batch, time, 64]
            micro_features: Micro timeframe features [batch, time, 64]
            
        Returns:
            Combined features [batch, time, 128]
        """
        # Concatenate features
        combined = torch.cat([macro_features, micro_features], dim=-1)
        
        # Apply cross-attention (query from macro, key/value from combined)
        attended, _ = self.cross_attention(
            macro_features,  # Query
            combined,        # Key
            combined         # Value
        )
        
        # Residual connection
        output = torch.cat([attended, micro_features], dim=-1)
        
        return output
    
    def _encode_synergy(self, synergy_context: Dict[str, Any]) -> torch.Tensor:
        """
        Extract arbitrage-relevant features from synergy context.
        
        Args:
            synergy_context: Synergy detection context
            
        Returns:
            Encoded features tensor [28]
        """
        return self.synergy_encoder.encode(synergy_context)
    
    def calculate_inefficiency_score(
        self,
        synergy_context: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> float:
        """
        Calculate market inefficiency score.
        
        Args:
            synergy_context: Synergy detection context
            market_data: Current market data
            
        Returns:
            Inefficiency score (0-1)
        """
        score = 0.0
        
        # Cross-timeframe alignment score
        synergy_type = synergy_context.get('synergy_type', 'TYPE_1')
        alignment_scores = {
            'TYPE_1': 0.9,  # Strong alignment pattern
            'TYPE_2': 0.7,  # Moderate alignment
            'TYPE_3': 0.8,  # Good alignment
            'TYPE_4': 0.6   # Weaker alignment
        }
        score += alignment_scores.get(synergy_type, 0.5) * 0.3
        
        # Signal coherence (low variance = high coherence)
        signal_strengths = list(synergy_context['signal_strengths'].values())
        if signal_strengths:
            coherence = 1.0 - np.std(signal_strengths)
            score += coherence * 0.3
        
        # Completion speed (faster = stronger inefficiency)
        bars_to_complete = synergy_context['metadata'].get('bars_to_complete', 10)
        speed_score = 1.0 / (1.0 + bars_to_complete / 10)
        score += speed_score * 0.2
        
        # Volume anomaly detection
        volume_ratio = market_data.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:  # 50% above average
            score += 0.2
        elif volume_ratio > 1.2:  # 20% above average
            score += 0.1
        
        return min(score, 1.0)


class ArbitrageSynergyEncoder(SynergyEncoder):
    """
    Specialized synergy encoder for arbitrage analysis.
    
    Extracts features relevant to cross-timeframe inefficiencies.
    """
    
    def __init__(self):
        """Initialize arbitrage synergy encoder."""
        super().__init__(output_dim=28)
    
    def encode(self, synergy_context: Dict[str, Any]) -> torch.Tensor:
        """
        Encode synergy context for arbitrage analysis.
        
        Args:
            synergy_context: Synergy detection context
            
        Returns:
            Encoded features tensor [28]
        """
        features = []
        
        # Synergy type encoding (4 features)
        synergy_type = synergy_context.get('synergy_type', 'TYPE_1')
        features.extend(self.encode_synergy_type(synergy_type).tolist())
        
        # Direction encoding (2 features)
        direction = synergy_context.get('direction', 1)
        features.extend(self.encode_direction(direction).tolist())
        
        # Cross-timeframe alignment (6 features)
        signal_sequence = synergy_context.get('signal_sequence', [])
        
        # Pattern-specific features based on synergy type
        if synergy_type == 'TYPE_1':  # MLMI → NW-RQK → FVG
            features.extend([1.0, 0.0, 0.0, 0.0, 0.9, 0.1])
        elif synergy_type == 'TYPE_2':  # MLMI → FVG → NW-RQK
            features.extend([0.0, 1.0, 0.0, 0.0, 0.7, 0.3])
        elif synergy_type == 'TYPE_3':  # NW-RQK → FVG → MLMI
            features.extend([0.0, 0.0, 1.0, 0.0, 0.8, 0.2])
        elif synergy_type == 'TYPE_4':  # NW-RQK → MLMI → FVG
            features.extend([0.0, 0.0, 0.0, 1.0, 0.6, 0.4])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.5, 0.5])
        
        # Completion characteristics (4 features)
        bars_to_complete = synergy_context['metadata'].get('bars_to_complete', 10)
        speed_score = 1.0 / (1.0 + bars_to_complete)
        
        features.extend([
            speed_score,                           # Completion speed
            min(bars_to_complete / 10, 1.0),      # Normalized time
            1.0 if bars_to_complete <= 5 else 0.0,  # Fast completion flag
            1.0 if bars_to_complete >= 8 else 0.0   # Slow completion flag
        ])
        
        # Signal coherence (4 features)
        signal_strengths = synergy_context.get('signal_strengths', {})
        strengths = list(signal_strengths.values())
        
        if strengths:
            mean_strength = np.mean(strengths)
            std_strength = np.std(strengths)
            min_strength = min(strengths)
            max_strength = max(strengths)
        else:
            mean_strength = std_strength = min_strength = max_strength = 0.5
        
        features.extend([
            mean_strength,
            1.0 - std_strength,  # Coherence (inverse of std)
            min_strength,
            max_strength
        ])
        
        # Market efficiency indicators (4 features)
        market_context = synergy_context.get('market_context', {})
        
        # Volume characteristics
        volume_ratio = market_context.get('volume_profile', {}).get('volume_ratio', 1.0)
        volume_anomaly = 1.0 if volume_ratio > 1.5 else 0.0
        
        # Price characteristics
        volatility = market_context.get('volatility', 10) / 30  # Normalize
        
        features.extend([
            min(volume_ratio / 2, 1.0),  # Normalized volume
            volume_anomaly,               # Volume spike indicator
            volatility,                   # Market volatility
            0.5                          # Placeholder
        ])
        
        # Risk-reward features (4 features)
        # These would normally come from backtesting or analysis
        features.extend([
            0.7,  # Expected win rate placeholder
            0.6,  # Risk-reward ratio placeholder
            0.5,  # Sharpe ratio placeholder
            0.8   # Opportunity quality placeholder
        ])
        
        # Ensure we have exactly 28 features
        assert len(features) == 28, f"Expected 28 features, got {len(features)}"
        
        return torch.tensor(features, dtype=torch.float32)