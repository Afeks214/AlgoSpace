"""
File: src/agents/main_core/regime_patterns.py (NEW FILE)
Regime pattern bank for enhanced regime understanding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RegimePattern:
    """Represents a learned regime pattern."""
    id: str
    prototype: torch.Tensor
    frequency: float
    typical_duration: float
    transition_probabilities: Dict[str, float]
    performance_stats: Dict[str, float]

class RegimePatternBank(nn.Module):
    """
    Maintains a bank of learned regime patterns for enhanced
    regime recognition and prediction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.regime_dim = config.get('regime_dim', 8)
        self.n_patterns = config.get('n_patterns', 16)
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        
        # Learnable pattern prototypes
        self.pattern_prototypes = nn.Parameter(
            torch.randn(self.n_patterns, self.regime_dim) * 0.1
        )
        
        # Pattern embedder
        self.pattern_embedder = nn.Sequential(
            nn.Linear(self.regime_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # Pattern statistics (not learnable)
        self.pattern_stats = {
            i: {
                'count': 0,
                'total_duration': 0,
                'transitions': {},
                'avg_return': 0.0,
                'volatility': 0.0
            }
            for i in range(self.n_patterns)
        }
        
    def forward(self, regime_vector: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Match regime to patterns and extract features.
        
        Args:
            regime_vector: [batch, regime_dim]
            
        Returns:
            Tuple of (pattern_features, pattern_info)
        """
        batch_size = regime_vector.size(0)
        
        # Calculate similarities to all patterns
        similarities = self._calculate_similarities(regime_vector)
        
        # Get top-k most similar patterns
        top_k = 3
        top_similarities, top_indices = torch.topk(similarities, top_k, dim=-1)
        
        # Soft assignment weights
        weights = F.softmax(top_similarities * 5.0, dim=-1)  # Temperature = 0.2
        
        # Weighted pattern features
        pattern_features = []
        for b in range(batch_size):
            weighted_features = torch.zeros(16).to(regime_vector.device)
            
            for k in range(top_k):
                pattern_idx = top_indices[b, k]
                pattern = self.pattern_prototypes[pattern_idx]
                
                # Concatenate regime and pattern
                combined = torch.cat([regime_vector[b], pattern])
                features = self.pattern_embedder(combined)
                
                weighted_features += weights[b, k] * features
                
            pattern_features.append(weighted_features)
            
        pattern_features = torch.stack(pattern_features)
        
        # Pattern info
        pattern_info = {
            'best_match_idx': int(top_indices[0, 0]),
            'best_match_similarity': float(top_similarities[0, 0]),
            'pattern_distribution': weights[0].cpu().numpy()
        }
        
        return pattern_features, pattern_info
        
    def _calculate_similarities(self, regime_vector: torch.Tensor) -> torch.Tensor:
        """Calculate cosine similarities to pattern prototypes."""
        # Normalize vectors
        regime_norm = F.normalize(regime_vector, dim=-1)
        pattern_norm = F.normalize(self.pattern_prototypes, dim=-1)
        
        # Cosine similarity
        similarities = torch.matmul(regime_norm, pattern_norm.t())
        
        return similarities
        
    def update_pattern_stats(self, pattern_idx: int, duration: float, 
                            next_pattern_idx: Optional[int], 
                            performance: Dict[str, float]):
        """Update statistics for a pattern."""
        stats = self.pattern_stats[pattern_idx]
        
        stats['count'] += 1
        stats['total_duration'] += duration
        
        if next_pattern_idx is not None:
            if next_pattern_idx not in stats['transitions']:
                stats['transitions'][next_pattern_idx] = 0
            stats['transitions'][next_pattern_idx] += 1
            
        # Update performance stats with EMA
        alpha = 0.1
        stats['avg_return'] = (1 - alpha) * stats['avg_return'] + alpha * performance['return']
        stats['volatility'] = (1 - alpha) * stats['volatility'] + alpha * performance['volatility']
        
    def get_pattern_interpretation(self, pattern_idx: int) -> Dict[str, Any]:
        """Get human-readable interpretation of a pattern."""
        stats = self.pattern_stats[pattern_idx]
        pattern = self.pattern_prototypes[pattern_idx]
        
        # Find dominant dimensions
        abs_pattern = torch.abs(pattern)
        top_dims = torch.topk(abs_pattern, 3).indices
        
        interpretation = {
            'pattern_id': f'Pattern_{pattern_idx}',
            'dominant_dimensions': top_dims.tolist(),
            'frequency': stats['count'],
            'avg_duration': stats['total_duration'] / max(stats['count'], 1),
            'avg_return': stats['avg_return'],
            'volatility': stats['volatility'],
            'common_transitions': sorted(
                stats['transitions'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
        }
        
        return interpretation